"""Offline replay of the reported conversation; no provider/model calls."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clarification_policy import clarification_questions, concise_question, must_return_results
from photo_query import build_photo_query, requests_direct_results
from test_ui import ui_functions


CANDIDATES = [
    dict(path="one.jpg", score=45, scene="海边", description="一只黑色猫戴草帽", matched=[], missing=[]),
    dict(path="two.jpg", score=44, scene="屋顶", description="两只白色猫没有帽子", matched=[], missing=[]),
]
QUESTIONS = ["是在海边还是屋顶？", "猫是黑色还是白色？", "照片里是一只还是两只猫？"]


class Dialogue:
    def __init__(self, history=None, responses=None, candidates=None):
        self.history = [] if history is None else history
        self.outputs = []
        self.rendered = []
        self.preview = Mock(return_value=CANDIDATES if candidates is None else candidates)
        response_iter = iter(responses or ["【行动】提问：是在海边还是屋顶？"])
        def complete(**_):
            text = next(response_iter)
            if isinstance(text, Exception):
                raise text
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])]
        self.create = Mock(side_effect=complete)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.search = Mock(return_value=dict(top1="one.jpg", top2="two.jpg", top3=None, report="", top_results=[]))
        self.st = SimpleNamespace(info=Mock(), error=Mock(), markdown=self.outputs.append,
                                  chat_message=lambda *_: nullcontext(), columns=lambda n: [nullcontext()] * n,
                                  expander=lambda *_, **__: nullcontext())
        def add(role, content, **kw):
            self.history.append(dict(role=role, content=content, **kw))
        self.scope = ui_functions("_run_assistant_with_client", "should_direct_search", st=self.st,
                                 preview_search_candidates=self.preview, get_current_messages=lambda: self.history,
                                 add_message=add, search_photos_tool=self.search,
                                 render_image_tile=lambda img, **_: self.rendered.append(img))

    def turn(self, text):
        self.scope["_run_assistant_with_client"](text, self.client)


@pytest.mark.parametrize("scores, expected", [([70, 69], True), ([55, 37], True), ([55, 38], False),
                                              ([54, 10], False), ([69, 68], False), ([70], True)])
def test_existing_score_thresholds_not_probabilities(scores, expected):
    direct = ui_functions("should_direct_search")["should_direct_search"]
    assert direct([{"score": score} for score in scores]) is expected


@pytest.mark.parametrize("hostile_response", ["【行动】提问：继续问？", "【行动】回答：还想问你几个问题？", "malformed"])
def test_high_confidence_zero_questions_even_hostile_provider(hostile_response):
    candidates = [dict(c, score=75) for c in CANDIDATES]
    dialogue = Dialogue(responses=[hostile_response], candidates=candidates)
    dialogue.turn("猫")
    dialogue.create.assert_not_called()
    dialogue.search.assert_called_once_with("猫", client=None)
    assert dialogue.history[-1]["kind"] == "results"
    assert dialogue.rendered == ["one.jpg", "two.jpg"]
    assert not clarification_questions(dialogue.history)


def test_three_actual_questions_then_results_never_fourth():
    dialogue = Dialogue(responses=["【行动】提问：" + question for question in QUESTIONS])
    for user in ["猫", "屋顶", "白色"]:
        dialogue.turn(user)
    assert clarification_questions(dialogue.history) == QUESTIONS
    assert dialogue.create.call_count == 3
    dialogue.search.assert_not_called()
    dialogue.turn("两只")
    assert dialogue.create.call_count == 3
    dialogue.search.assert_called_once_with("猫\n屋顶\n白色\n两只", client=None)
    assert dialogue.history[-1]["kind"] == "results"
    dialogue.turn("还有猫")
    assert len(clarification_questions(dialogue.history)) == 3  # results do not reset target budget


# Transcribed from inspected pixels of 2026-10-22 15.58.04 and 15.58.12 screenshots.
SCREENSHOT_QUESTIONS = [
    "您要找的是照片里包含金枪鱼拌饭这道菜，还是包含与“金枪鱼”相关的动物/食材画面？",
    "照片里能看出有金枪鱼或拌饭的食材吗？比如金枪鱼罐头、米饭、蔬菜等。",
    "照片里没有明确的金枪鱼拌饭，只有一碗山西餐桌环境的热食（含鸡蛋、蔬菜、豆腐皮等食材）。您记得那碗饭里有没有金枪鱼，还是更接近普通拌饭/汤饭？",
]
SCREENSHOT_FOURTH = "照片库中没有明确的金枪鱼拌饭候选，最接近的是山西餐桌环境中的一碗热食（深棕色汤底、煮鸡蛋、蔬菜、豆腐皮），并非金枪鱼拌饭。您要找的是这碗热食，还是必须为金枪鱼拌饭？"


def test_actual_screenshot_legacy_history_stops_before_fourth_question():
    history = [{"role": "assistant", "content": "你好，想找哪张照片呢？"}]
    for user, question in zip(["山西金枪鱼拌饭", "食材", "能看到"], SCREENSHOT_QUESTIONS):
        history.extend([dict(role="user", content=user), dict(role="assistant", content=question, thinking="旧检索依据")])
    dialogue = Dialogue(history, responses=["【行动】提问：" + SCREENSHOT_FOURTH])
    dialogue.turn("普通拌饭")
    dialogue.create.assert_not_called()
    dialogue.search.assert_called_once_with("山西金枪鱼拌饭\n食材\n能看到\n普通拌饭", client=None)
    assert len(clarification_questions(dialogue.history)) == 3
    assert dialogue.history[-1]["kind"] == "results"
    assert SCREENSHOT_FOURTH not in dialogue.outputs


def test_fresh_screenshot_first_long_question_falls_back_to_candidates():
    dialogue = Dialogue(responses=["【思考】冗长推理" + "【行动】提问：" + SCREENSHOT_QUESTIONS[0]])
    dialogue.turn("山西金枪鱼拌饭")
    assert dialogue.search.call_count == 1
    assert all("冗长" not in output for output in dialogue.outputs)
    assert not clarification_questions(dialogue.history)


@pytest.mark.parametrize("response", ["", "malformed", "【思考】只有思考", "【行动】回答：是什么颜色？",
    "【行动】提问：", "【行动】提问：在哪里？什么颜色？", "【行动】提问：戴帽子吗、在屋顶吗？",
    "【行动】提问：是黑色的猫吗，拍摄地点在哪儿？", "【行动】提问：" + "长" * 50 + "？",
    "【行动】提问：是金枪鱼拌饭吗？", "【行动】提问：是海边吗？【行动】提问：是屋顶吗？"])
def test_malformed_long_multifocus_and_unhelpful_questions_return_real_candidates(response):
    dialogue = Dialogue(responses=[response])
    dialogue.turn("猫")
    assert dialogue.search.call_count == 1
    assert dialogue.history[-1]["kind"] == "results"
    assert len(dialogue.outputs) == 1  # no unvalidated chunks or model rationale flashed on screen


def test_rephrased_repeated_question_returns_results_and_has_question_awareness():
    history = [dict(role="user", content="猫"), dict(role="assistant", content="是在海边还是屋顶？", kind="clarification")]
    dialogue = Dialogue(history, responses=["【行动】提问：照片背景是海边吗？"])
    dialogue.turn("能看到背景")
    assert dialogue.search.call_count == 1
    assert dialogue.create.call_args.kwargs["messages"][0]["content"].count("是在海边还是屋顶？") == 1
    assert "海边" not in dialogue.search.call_args.args[0]  # dialogue question != user fact


def test_current_user_clue_is_not_asked_again():
    dialogue = Dialogue(responses=["【行动】提问：是在海边还是屋顶？"])
    dialogue.turn("海边的猫")
    assert dialogue.search.call_count == 1


@pytest.mark.parametrize("command", ["直接给结果", "不用再问了", "不要追问", "请直接搜索", "先给照片",
                                      "不要再问", "别问", "直接给我看结果", "直接给我看照片", "直接出结果", "给我结果"])
def test_direct_request_uses_existing_user_clues_without_provider(command):
    dialogue = Dialogue([dict(role="user", content="猫")])
    dialogue.turn(command)
    dialogue.create.assert_not_called()
    dialogue.search.assert_called_once_with("猫", client=None)


@pytest.mark.parametrize("command", ["不要直接给结果", "别直接搜索", "不用直接展示结果"])
def test_negated_result_request_does_not_force_results(command):
    assert not requests_direct_results(command)


@pytest.mark.parametrize("unclear", ["不知道", "不知道了", "我也不知道", "不清楚", "记不清楚了", "忘了"])
def test_unclear_answer_exits_without_inventing_question_fact(unclear):
    dialogue = Dialogue([dict(role="user", content="猫"), dict(role="assistant", content="是在海边吗？")])
    dialogue.turn(unclear)
    dialogue.create.assert_not_called()
    dialogue.search.assert_called_once_with("猫", client=None)


def test_provider_failure_returns_local_candidates_without_leaking_error():
    dialogue = Dialogue(responses=[RuntimeError("private-token")])
    dialogue.turn("猫")
    dialogue.search.assert_called_once_with("猫", client=None)
    assert all("private-token" not in output for output in dialogue.outputs)
    assert "private-token" not in dialogue.st.info.call_args.args[0]


def test_greetings_results_errors_not_counted_but_legacy_imperatives_are():
    history = [dict(role="assistant", content="你好，想找哪张照片？"), dict(role="user", content="猫"),
               dict(role="assistant", content="找到了，是它吗？", images=["cat.jpg"]),
               dict(role="assistant", content="有你要的吗？", kind="results"),
               dict(role="assistant", content="请重试", kind="error")]
    assert clarification_questions(history) == []
    for question in ["告诉我拍摄地点", "描述一下颜色", "还有什么特征"]:
        history.append(dict(role="assistant", content=question, thinking="旧依据"))
    assert len(clarification_questions(history)) == 3
    assert must_return_results(history, "猫")


def test_reset_and_switching_do_not_share_budget_or_clues():
    dialogue = Dialogue(responses=["【行动】提问：" + question for question in QUESTIONS] * 2)
    for user in ["猫", "屋顶", "白色"]:
        dialogue.turn(user)
    old = dialogue.history
    dialogue.history = []
    dialogue.turn("猫")
    assert len(clarification_questions(dialogue.history)) == 1
    assert build_photo_query(dialogue.history) == "猫"
    dialogue.history = old
    dialogue.turn("两只")
    assert dialogue.history[-1]["kind"] == "results"
    assert len(clarification_questions(dialogue.history)) == 3
