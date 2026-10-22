"""User-only multi-turn regression cases; no real credentials/model/API calls."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import jieba.posseg as pseg
import pytest

from photo_query import build_photo_query, positive_photo_clues
from test_backend import load, module
from test_ui import ui_functions


@pytest.fixture
def core():
    # Exercise real POS/attribute parsing, but never initialize a vector model.
    return load("_entity_test_core", "AdaphotoRet_run.py", {
        "sentence_transformers": module("sentence_transformers", SentenceTransformer=Mock(
            side_effect=AssertionError("model downloads forbidden"))),
    })


def test_only_user_turns_are_combined_and_current_input_is_retained():
    history = [{"role": "user", "content": "猫"},
               {"role": "assistant", "content": "准确，请告诉我，是在屋顶吗？"},
               {"role": "system", "content": "北京"},
               {"role": "tool", "content": "游泳"},
               {"content": "无角色的内容"}]
    assert build_photo_query(history, "海边") == "猫\n海边"
    assert history[0]["content"] == "猫"  # no mutation or hidden state


def test_all_user_turns_survive_long_dialogue_and_repeats():
    history = []
    for clue in ["猫", "海边", "白色", "站立", "白天"]:
        history.extend([{"role": "user", "content": clue},
                        {"role": "assistant", "content": "准确，请告诉我"}])
    assert build_photo_query(history, "白天") == "猫\n海边\n白色\n站立\n白天"


def test_new_target_new_session_and_switching_are_isolated():
    a = [{"role": "user", "content": "猫"}]
    b = [{"role": "user", "content": "瀑布"}]
    assert build_photo_query(a, "海边") == "猫\n海边"
    assert build_photo_query(b, "山上") == "瀑布\n山上"
    assert build_photo_query([], "狗") == "狗"
    assert build_photo_query(a) == "猫"


@pytest.mark.parametrize("noise", ["", "   ", "谢谢！", "请告诉我", "准确", "记不清了", "不知道", "是的"])
def test_noise_adds_no_fact_and_does_not_confirm_assistant_guess(noise):
    assert build_photo_query([], noise) == ""
    assert build_photo_query([{"role": "user", "content": "猫"},
                              {"role": "assistant", "content": "是橘色的吗？"}], noise) == "猫"


def test_screenshot_exact_assistant_response_never_becomes_entities(core):
    # Screenshot: the old action payload included this complete machine answer.
    reply = ("已为您找到最匹配的照片：data/life/life_04.jpg，"
             "一只穿着传统节日装扮的黑色成年驴站在河岸边，背景是湍急的河水和远处绿植覆盖的山崖。"
             "这里有您想要的照片吗？如果不够准确，请告诉我哪里需要调整。")
    query = build_photo_query([{"role": "user", "content": "壶口瀑布 河流边景区"},
                               {"role": "assistant", "content": reply}], "驴 黑色 成年 站立")
    entities = core.parse_query(query)["decomposition"]["extracted_entities"]
    assert set(["壶口瀑布", "河流", "景区", "驴", "黑色", "成年", "站立"]) <= set(entities)
    assert not set(["准确", "请", "告诉", "找到", "匹配", "穿着", "传统节日", "山崖", "湍急"]) & set(entities)
    assert "life_04" not in query


def test_dialogue_token_filter_preserves_real_compounds_and_actions(core):
    entities = core.parse_query("请告诉我准确的照片 请柬 请客 邀请函 天津 天空 报告厅 跑步 跳 飞")["terms"]["extracted_entities"]
    assert not {"请", "告诉", "准确"} & set(entities)
    assert {"请柬", "请客", "邀请函", "天津", "天空", "报告厅", "跑步", "跳", "飞"} <= set(entities)


def test_noise_entities_are_empty_and_repeated_clue_entities_are_unique(core):
    assert core.parse_query("请告诉我准确的照片")["terms"]["extracted_entities"] == []
    assert core.parse_query("猫 海边 猫 海边")["terms"]["extracted_entities"] == ["猫", "海边"]


@pytest.mark.parametrize("correction", ["不是黑色，是白色", "不是黑色的，是白色"])
def test_explicit_correction_retracts_old_attribute_keeps_other_clues(core, correction):
    parsed = core.parse_query(build_photo_query([{"role": "user", "content": "黑色的猫在海边"}], correction))
    assert "黑色" not in parsed["expanded_query"]
    assert "白色" in parsed["terms"]["pet_coat_color"]
    assert "猫" in parsed["terms"]["pet_animal_type"]
    assert "海边" in parsed["expanded_query"]


def test_negative_clause_is_not_positive_evidence_and_not_substring_erasure():
    assert positive_photo_clues("猫\n不是猫，是狗", pseg.cut) == "是狗"
    assert "猫头鹰" in positive_photo_clues("猫头鹰\n不是猫", pseg.cut)
    assert "海边" not in positive_photo_clues("猫在海边\n不在海边，是在公园", pseg.cut)
    assert positive_photo_clues("不是猫", pseg.cut) == ""
    assert "黑色" in positive_photo_clues("黑色\n不是黑色，是白色\n不是白色，是黑色", pseg.cut)


@pytest.mark.parametrize("description", ["没有戴帽子的狗", "不在海边的猫", "没有人的海边风景"])
def test_attributive_negative_description_is_not_discarded(description):
    assert positive_photo_clues(description, pseg.cut) == description


@pytest.mark.parametrize("response,direct", [
    ("【思考】依据【行动】检索：准确 请 告诉我 北京 河流", False),
    ("【思考】依据【行动】提问：是海边吗？", True),
    ("【思考】没有行动", False),
    ("完全未格式化的机器回复", False),
    ("【思考】依据【行动】损坏的行动", False),
])
def test_every_ui_search_branch_uses_same_user_only_query(response, direct):
    history = [{"role": "user", "content": "屋顶上的猫"},
               {"role": "assistant", "content": "准确 请 告诉我 北京"}]
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=response))])
    create = Mock(return_value=[chunk])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    placeholder = SimpleNamespace(markdown=Mock(), error=Mock())
    st = SimpleNamespace(info=Mock(), error=Mock(), markdown=Mock(), chat_message=lambda *_: nullcontext(), empty=lambda: placeholder)
    preview = Mock(return_value=[])
    search = Mock(return_value={"top1": None, "top2": None, "top3": None, "report": ""})
    def add(role, content, **kwargs):
        history.append({"role": role, "content": content, **kwargs})
    scope = ui_functions("_run_assistant_with_client", st=st, preview_search_candidates=preview,
                         get_current_messages=lambda: history, add_message=add,
                         should_direct_search=lambda _: direct, search_photos_tool=search)
    scope["_run_assistant_with_client"]("橘色的", client)
    expected = "屋顶上的猫\n橘色的"
    preview.assert_called_once_with(expected, top_k=5)
    search.assert_called_once_with(expected, client=None if direct else client)
    assert [m["content"] for m in history if m["role"] == "user"] == ["屋顶上的猫", "橘色的"]
    assert create.call_count == (0 if direct else 1)  # confident search bypasses provider
    if not direct:
        assert create.call_args.kwargs["messages"][1:] == [
            {"role": "user", "content": "屋顶上的猫"}, {"role": "user", "content": "橘色的"}]


def test_empty_noise_does_not_preview_or_call_provider():
    client = SimpleNamespace(chat=Mock())
    preview = Mock()
    st = SimpleNamespace(info=Mock())
    scope = ui_functions("_run_assistant_with_client", st=st, get_current_messages=lambda: [],
                         preview_search_candidates=preview)
    scope["_run_assistant_with_client"]("谢谢！", client)
    preview.assert_not_called()
    assert not client.chat.mock_calls


def test_unclear_replies_reach_dialogue_control_but_not_retrieval():
    history = [{"role": "user", "content": "猫"},
               {"role": "assistant", "content": "是橘猫吗？"},
               {"role": "user", "content": "不知道"},
               {"role": "assistant", "content": "是在屋顶吗？"}]
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="【行动】检索：橘猫 屋顶"))])
    create = Mock(return_value=[chunk])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    placeholder = SimpleNamespace(markdown=Mock(), error=Mock())
    st = SimpleNamespace(info=Mock(), error=Mock(), markdown=Mock(), chat_message=lambda *_: nullcontext(), empty=lambda: placeholder)
    preview = Mock(return_value=[])
    search = Mock(return_value={"top1": None, "top2": None, "top3": None, "report": ""})
    scope = ui_functions("_run_assistant_with_client", st=st, preview_search_candidates=preview,
                         get_current_messages=lambda: history, add_message=Mock(),
                         should_direct_search=lambda _: False, search_photos_tool=search)
    scope["_run_assistant_with_client"]("记不清了", client)
    preview.assert_called_once_with("猫", top_k=5)
    search.assert_called_once_with("猫", client=None)
    create.assert_not_called()  # unclear replies now deterministically stop questioning
