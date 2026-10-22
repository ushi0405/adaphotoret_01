"""Report-only restoration: stored evidence, no model downloads or provider calls."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import llm_explainer as explainer
from test_ui import ui_functions
from test_clarification_policy import Dialogue, QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
QUERY = "山西晋祠，西游记打卡处照片\n山西晋祠大门古建筑"
# Transcribed from the inspected 2026-10-22 17.18.01 screenshot, not re-scored.
RESULTS = [
    {"img_path": "data/travel/travel_23.HEIC", "score": 82,
     "trace": [("base_semantic_similarity", .349, "向量语义相似 0.6979 × 0.50"),
               ("gps_address_partial_match", .15, "地址部分匹配: 晋祠"),
               ("entity_keyword_match", .3, "命中实体: 西游记, 打卡")]},
    {"img_path": "data/travel/travel_17.HEIC", "score": 76,
     "trace": [("base_semantic_similarity", .379, "向量语义相似 0.7582 × 0.50"),
               ("gps_address_partial_match", .15, "地址部分匹配: 晋祠"),
               ("entity_keyword_match", .15, "命中实体: 大门")]},
    {"img_path": "data/travel/travel_20.HEIC", "score": 74,
     "trace": [("base_semantic_similarity", .348, "向量语义相似 0.6954 × 0.50"),
               ("gps_address_partial_match", .15, "地址部分匹配: 晋祠"),
               ("entity_keyword_match", .15, "命中实体: 打卡")]},
]


def saved_report(query=QUERY, results=RESULTS):
    lines = [f"### 查询「{query}」的可解释检索结果", "", "**提取实体**：山西、晋祠、西游记、打卡、大门、古建筑",
             "", "#### 推理链（Top 3）"]
    for i, r in enumerate(results, 1):
        lines.extend([f"{i}. `{r['img_path']}` | 最终得分: {r['score']}",
                      " ".join(f"[{name} {delta:+.3f}]" for name, delta, _ in r["trace"]),
                      "📌 " + " | ".join(evidence for _, _, evidence in r["trace"]), ""])
    return "\n".join(lines)


@pytest.mark.parametrize("index", range(3))
def test_screenshot_photo_gets_individual_query_score_and_actual_evidence(index):
    r = RESULTS[index]
    text = explainer.score_explanation(QUERY, r)
    assert 80 <= len(text) <= 150
    assert f"最终得分{r['score']}（非概率）" in text
    assert "针对「山西晋祠" in text
    assert "语义加权贡献" in text
    assert r["trace"][1][2] in text and r["trace"][2][2] in text
    assert "未记录规则扣分" in text
    assert "扣分依据" not in text
    assert "%" not in text
    if index == 1:
        assert "命中实体: 西游记" not in text


def test_different_results_do_not_receive_same_generic_paragraph():
    bodies = [explainer.score_explanation(QUERY, r).split("规则支持：")[1] for r in RESULTS]
    assert len(set(bodies)) == 3


def test_semantic_only_negative_is_not_a_missing_rule_or_penalty():
    text = explainer.score_explanation("古建筑", {"score": 18,
        "trace": [("base_semantic_similarity", -.08, "向量语义相似 -0.16 × 0.50")]})
    assert "语义加权贡献-0.0800" in text
    assert "未记录规则加分" in text and "未记录规则扣分" in text
    assert "扣分依据" not in text
    assert 80 <= len(text) <= 150


def test_real_negative_evidence_distinguished_from_final_score():
    text = explainer.score_explanation("晋祠的猫", {"score": 12, "trace": [
        ("base_semantic_similarity", .23, "向量语义相似 0.46 × 0.50"),
        ("pet_animal_match", .16, "pet_animal_match"),
        ("geo_mismatch_penalty", -1.5, "未命中指定地点: 晋祠"),
    ]})
    assert "最终得分12" in text
    assert "动物种类符合查询" in text
    assert "扣分依据：未命中指定地点: 晋祠" in text
    assert "映射前原始分" in text
    assert "扣除1.5分" not in text
    assert 80 <= len(text) <= 150


def test_pet_bare_rule_names_do_not_invent_breed_colour_count_or_people():
    text = explainer.score_explanation("戴帽子的猫", {"score": 78, "trace": [
        ("pet_breed_match", .25), ("pet_color_mismatch", -.03, "pet_color_mismatch")]})
    assert "品种符合查询" in text and "毛色与查询不符" in text
    assert all(word not in text for word in ("金毛", "橘色", "三只", "人种", "一群人"))
    assert 80 <= len(text) <= 150


def test_prior_is_not_presented_as_confirmed_matching_clue():
    text = explainer.score_explanation("范文", {"score": 62, "trace": [
        ("type_phone_screenshot_prior", .04, "type_phone_screenshot_prior")]})
    assert "先验（并非线索命中）" in text


@pytest.mark.parametrize("query", ["猫", "山西晋祠，西游记打卡处照片", "线索" * 50])
def test_short_and_long_queries_keep_useful_summary_near_one_hundred_characters(query):
    for trace in [[("base_semantic_similarity", .1234, "向量语义"), ("category_match", .22, "类别匹配: 宠物")],
                  [("base_semantic_similarity", .1234, "向量语义"),
                   ("entity_keyword_match", .3, "命中实体: " + "长线索" * 20),
                   ("object_core_missing", -.2, "缺少核心对象: " + "未命中" * 20)]]:
        assert 80 <= len(explainer.score_explanation(query, dict(score=55, trace=trace))) <= 150


def test_missing_trace_is_honest_and_does_not_hallucinate():
    text = explainer.score_explanation("猫", {"score": 90.25, "trace": []})
    assert "最终得分90.25" in text and "无法确认具体命中线索或扣分原因" in text
    assert 80 <= len(text) <= 150


def test_report_appends_only_keeps_every_numeric_byte_order_and_identity():
    before = deepcopy(RESULTS)
    report = saved_report()
    restored = explainer.ensure_score_explanations(report, QUERY, RESULTS)
    assert restored.startswith(report + "\n\n")
    assert RESULTS == before
    assert restored.count(explainer.SCORE_EXPLANATION_HEADING) == 1
    summaries = restored.split(explainer.SCORE_EXPLANATION_HEADING)[1]
    assert [summaries.index(r["img_path"]) for r in RESULTS] == sorted(summaries.index(r["img_path"]) for r in RESULTS)
    assert explainer.ensure_score_explanations(restored, QUERY, RESULTS) == restored


def test_legacy_report_uses_only_its_saved_query_score_trace():
    report = saved_report()
    with patch.object(explainer, "get_deepseek_client", side_effect=AssertionError("no provider")), \
         patch.object(explainer, "generate_explanation", side_effect=AssertionError("no LLM")):
        result = explainer.ensure_score_explanations(report)
    assert result == explainer.ensure_score_explanations(report, QUERY, RESULTS)
    assert "最终得分82" in result


def test_legacy_multiline_query_cannot_add_a_report_shaped_fake_photo():
    query = "晋祠\n9. `not-a-photo.jpg` | 最终得分: 99\n[entity_keyword_match +0.3]\n📌 命中实体: 伪造"
    report = saved_report(query=query)
    restored = explainer.ensure_score_explanations(report)
    summaries = restored.split(explainer.SCORE_EXPLANATION_HEADING)[1]
    assert "**图片4" not in summaries
    assert "最终得分99" not in summaries
    assert "**图片1 · `data/travel/travel_23.HEIC`**" in summaries


def test_original_drive_small_tag_report_also_recovers_without_rescoring():
    import re
    report = re.sub(r"^(📌 .*)$", r"<small>\1</small>", saved_report(), flags=re.M)
    restored = explainer.ensure_score_explanations(report)
    assert restored.startswith(report)
    assert "最终得分82（非概率）" in restored
    assert "</small>" not in restored.split(explainer.SCORE_EXPLANATION_HEADING)[1]


@pytest.mark.parametrize("report", ["", "自定义历史报告", "解释生成失败：old message",
    saved_report().replace(" | 命中实体: 大门", ""),
    saved_report().replace("命中实体: 大门", "命中实体: 大门 | ambiguous extra evidence")])
def test_unrecognized_or_ambiguous_legacy_report_remains_untouched(report):
    assert explainer.ensure_score_explanations(report) == report


def test_legacy_existing_explanation_preserved_without_duplicates():
    report = saved_report() + "\n#### 🤖 智能推理说明\n已有说明"
    assert explainer.ensure_score_explanations(report) == report


@pytest.mark.parametrize("old_text", ["", "解释生成失败：API 暂时不可用", "智能说明暂不可用；本地检索结果不受影响。"])
def test_failed_or_empty_legacy_llm_explanation_recovers_locally(old_text):
    report = saved_report() + "\n#### 🤖 智能推理说明\n" + old_text
    result = explainer.ensure_score_explanations(report)
    assert result.startswith(report)
    assert "最终得分82（非概率）" in result


@pytest.mark.parametrize("results", [[{}], [None], [dict(img_path="one", score=12, trace=5)], {"bad": "shape"}])
def test_malformed_optional_data_never_discards_existing_report(results):
    report = saved_report()
    assert explainer.ensure_score_explanations(report, QUERY, results) == report


def test_trace_none_falls_back_and_heading_text_inside_query_does_not_hide_summary():
    result = explainer.ensure_score_explanations("查询逐图评分说明", "逐图评分说明", [dict(img_path="one", score=12, trace=None)])
    assert "无法确认具体命中线索或扣分原因" in result


@pytest.mark.parametrize("client", [None, object()])
def test_search_report_no_key_no_llm_and_no_retrieval_changes(client):
    report, table = saved_report(), [[r["img_path"], r["score"], "existing"] for r in RESULTS]
    search = Mock(return_value=(*(r["img_path"] for r in RESULTS), report, table, RESULTS))
    scope = ui_functions("search_photos_tool", search_photos=search)
    with patch.object(explainer, "generate_explanation", side_effect=AssertionError("malformed/failed LLM never needed")), \
         patch.object(explainer, "get_deepseek_client", side_effect=AssertionError("no key")):
        result = scope["search_photos_tool"](QUERY, client)
    search.assert_called_once_with(QUERY, client=client, allow_environment=False, include_explanation=False)
    assert [result[f"top{i}"] for i in range(1, 4)] == [r["img_path"] for r in RESULTS]
    assert result["top_results"] is RESULTS and result["table"] is table
    assert result["report"].startswith(report)
    assert "逐图评分说明" in result["report"]


@pytest.mark.parametrize("mode", ["direct", "third_cap", "normal", "malformed", "api_error"])
def test_all_result_branches_render_and_persist_per_photo_explanations(mode):
    history = []
    if mode == "third_cap":
        for q in QUESTIONS:
            history.extend([dict(role="user", content="猫"), dict(role="assistant", content=q, kind="clarification")])
    response = RuntimeError("private provider exception") if mode == "api_error" else (
        "malformed" if mode == "malformed" else "【行动】检索：")
    candidates = [dict(path="one.jpg", score=82, scene="", description="", matched=[], missing=[])] if mode == "direct" else None
    dialogue = Dialogue(history, responses=[response], candidates=candidates)
    backend_search = Mock(return_value=(*(r["img_path"] for r in RESULTS), saved_report(), [], RESULTS))
    scope = ui_functions("search_photos_tool", search_photos=backend_search)
    dialogue.scope["search_photos_tool"] = scope["search_photos_tool"]
    dialogue.turn(QUERY)
    assert dialogue.create.call_count == (0 if mode in {"direct", "third_cap"} else 1)
    assert len(dialogue.rendered) == 3
    assert dialogue.history[-1]["images"] == [r["img_path"] for r in RESULTS]
    assert "逐图评分说明" in dialogue.history[-1]["report"]
    assert any("最终得分82（非概率）" in s for s in dialogue.outputs)
    assert all("private provider exception" not in s for s in dialogue.outputs)


def test_approved_scoring_extraction_cache_and_data_are_frozen():
    # Baseline d5a7b45; changing retrieval requires separate user authorization.
    frozen = {
        "AdaphotoRet_run.py": "aa0a2d5dc173415b19962f8b24b8b207c48913bb9da54e88e6d8d1a63ecbbdc2",
        "attributes.py": "69b897e6aaea7c5d5e3f86679fc2bea32729e57b96a104b7649263554d6de25f",
        "photo_query.py": "01ce58f661d0228687d25833321bd82df3d5f95eef4ca8ef51011c13cd1ea59b",
        "clarification_policy.py": "932586fd4bdea4997f183d6c34bfe502b8a24fbf6f6b98650e431e1b429ca483",
        "backend_resources.py": "b75a4282401c64a6041f479b2b203055e8c495a13e2a31a1ab628edcc3b080ef",
        "metadata.demo.json": "e6e34282e7a98e63d59b3e4a6e37397ed98217baaf6da4431059593bb0b8970b",
        "DATA_MANIFEST.json": "975906aaf2ec53af839827709d3f720207b4e9a4261441af0ff6ee2d4f1978fe",
    }
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    functions = {
        "cached_global_search": "d3b1d1812636a76f8972914e42f75f1e8cf48a6937a2f0709afd7ec947df10ad",
        "preview_search_candidates": "d2dbfe83c78579c248470d49e6c56829c085656fab2a8eba5b94cc642d2f17a7",
        "should_direct_search": "fe1bc1dc9eb7d1d057041279e6b582ac530a7e3d95f7a2d3eff5d53b41e041cc",
        "run_assistant": "4843082bd6159ddd44b120001f3194dcfe989d39700e89194b11d1d687206d8b",
        "_run_assistant_with_client": "c8496321bfab918433a4576aee5a42bbe3923eaf36e117c5856a24b27e1ee765",
    }
    nodes = {n.name: n for n in ast.parse((ROOT / "streamlit_ui.py").read_text()).body if isinstance(n, ast.FunctionDef)}
    for name, digest in functions.items():
        assert hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest() == digest, name
