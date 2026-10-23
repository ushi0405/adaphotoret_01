"""Report-only restoration: stored evidence, no model downloads or provider calls."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import llm_explainer
import photo_report_narrative as explainer
from test_ui import ui_functions
from test_clarification_policy import Dialogue, QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads((ROOT / "metadata.demo.json").read_text())
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
def test_real_photo_description_precedes_matching_clues_then_unverified_details(index):
    r = RESULTS[index]
    text = explainer.score_explanation(QUERY, r, METADATA,
        query_clues=("山西", "晋祠", "西游记", "打卡", "大门", "古建筑"))
    assert 80 <= len(text) <= 150
    assert text.startswith("已有识别描述：")
    assert text.index("已有识别描述") < text.index("对应查询") < text.index("尚未获记录支持")
    assert ["石雕武士像", "大门牌坊", "中式古典建筑"][index] in text
    assert "地址记录对应「晋祠」" in text
    assert "最终得分" not in text and "语义加权贡献" not in text
    assert "不能直接判为不存在" in text
    if index == 0:
        assert "文字识别含「西游记、打卡」" in text
        assert "大门" in text.split("不过")[1]
        assert "单人" not in text and "真人" not in text
    elif index == 1:
        assert "描述或标签对应「大门」" in text
        assert "「西游记、打卡」尚未获记录支持" in text
    else:
        assert "文字识别含「打卡」" in text


def test_different_results_do_not_receive_same_generic_paragraph():
    bodies = [explainer.score_explanation(QUERY, r, METADATA) for r in RESULTS]
    assert len(set(bodies)) == 3
    assert len(set(text.split("对应查询")[0] for text in bodies)) == 3


def test_semantic_only_negative_is_not_a_missing_rule_or_penalty():
    text = explainer.score_explanation("古建筑", {"score": 18,
        "trace": [("base_semantic_similarity", -.08, "向量语义相似 -0.16 × 0.50")]})
    assert "不能据此确认具体线索已符合" in text
    assert "暂未记录明确冲突" in text
    assert "不太符合的是" not in text and "待确认的是" not in text
    assert 80 <= len(text) <= 150


def test_missing_geography_is_uncertain_not_a_confirmed_wrong_location():
    text = explainer.score_explanation("晋祠的猫", {"score": 12, "trace": [
        ("base_semantic_similarity", .23, "向量语义相似 0.46 × 0.50"),
        ("pet_animal_match", .16, "pet_animal_match"),
        ("geo_mismatch_penalty", -1.5, "未命中指定地点: 晋祠"),
    ]})
    assert "动物种类符合查询" in text
    assert "地点「晋祠」尚未获位置记录支持" in text
    assert "这不等于画面中一定没有" in text
    assert "不太符合的是" not in text and "扣除1.5分" not in text
    assert 80 <= len(text) <= 150


def test_pet_bare_rule_names_do_not_invent_breed_colour_count_or_people():
    text = explainer.score_explanation("戴帽子的猫", {"score": 78, "trace": [
        ("pet_breed_match", .25), ("pet_color_mismatch", -.03, "pet_color_mismatch")]})
    assert "品种符合查询" in text and "毛色记录不足" in text
    assert all(word not in text for word in ("金毛", "橘色", "三只", "人种", "一群人"))
    assert 80 <= len(text) <= 150


def test_prior_is_not_presented_as_confirmed_matching_clue():
    text = explainer.score_explanation("范文", {"score": 62, "trace": [
        ("type_phone_screenshot_prior", .04, "type_phone_screenshot_prior")]})
    assert "先验并非具体线索命中" in text
    assert "对应查询" not in text


@pytest.mark.parametrize("query", ["猫", "山西晋祠，西游记打卡处照片", "线索" * 50])
def test_short_and_long_queries_keep_useful_summary_near_one_hundred_characters(query):
    for trace in [[("base_semantic_similarity", .1234, "向量语义"), ("category_match", .22, "类别匹配: 宠物")],
                  [("base_semantic_similarity", .1234, "向量语义"),
                   ("entity_keyword_match", .3, "命中实体: " + "长线索" * 20),
                   ("object_core_missing", -.2, "缺少核心对象: " + "未命中" * 20)]]:
        assert 80 <= len(explainer.score_explanation(query, dict(score=55, trace=trace))) <= 150


def test_missing_trace_and_metadata_are_honest_not_query_as_photo_description():
    text = explainer.score_explanation("穿紫色衣服的猫在月球", {"score": 90.25, "trace": []})
    assert "暂无可用的已有识别描述" in text and "无法确认它与当前线索的具体对应关系" in text
    assert "紫色" not in text and "月球" not in text
    assert 80 <= len(text) <= 150


def test_report_appends_only_keeps_every_numeric_byte_order_and_identity():
    before = deepcopy(RESULTS)
    report = saved_report()
    restored = explainer.ensure_score_explanations(report, QUERY, RESULTS, METADATA)
    assert restored.startswith(report + "\n\n")
    assert RESULTS == before
    assert restored.count(explainer.SCORE_EXPLANATION_HEADING) == 1
    summaries = restored.split(explainer.SCORE_EXPLANATION_HEADING)[1]
    assert [summaries.index(r["img_path"]) for r in RESULTS] == sorted(summaries.index(r["img_path"]) for r in RESULTS)
    assert explainer.ensure_score_explanations(restored, QUERY, RESULTS, METADATA) == restored


def test_legacy_report_uses_only_its_saved_query_score_trace():
    report = saved_report()
    with patch.object(llm_explainer, "get_deepseek_client", side_effect=AssertionError("no provider")), \
         patch.object(llm_explainer, "generate_explanation", side_effect=AssertionError("no LLM")):
        result = explainer.ensure_score_explanations(report, metadata=METADATA)
    assert result == explainer.ensure_score_explanations(report, QUERY, RESULTS, METADATA)
    assert "最终得分: 82" in result and "石雕武士像" in result


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
    assert "已有识别" in restored and "最终得分: 82" in restored
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
    assert "已有识别" in result and "最终得分: 82" in result


@pytest.mark.parametrize("results", [[{}], [None], [dict(img_path="one", score=12, trace=5)], {"bad": "shape"}])
def test_malformed_optional_data_never_discards_existing_report(results):
    report = saved_report()
    assert explainer.ensure_score_explanations(report, QUERY, results) == report


def test_trace_none_falls_back_and_heading_text_inside_query_does_not_hide_summary():
    result = explainer.ensure_score_explanations("查询逐图评分说明", "逐图评分说明", [dict(img_path="one", score=12, trace=None)])
    assert "无法确认它与当前线索的具体对应关系" in result


@pytest.mark.parametrize("client", [None, object()])
def test_search_report_no_key_no_llm_and_no_retrieval_changes(client):
    report, table = saved_report(), [[r["img_path"], r["score"], "existing"] for r in RESULTS]
    search = Mock(return_value=(*(r["img_path"] for r in RESULTS), report, table, RESULTS))
    scope = ui_functions("search_photos_tool", search_photos=search, metadata=METADATA)
    with patch.object(llm_explainer, "generate_explanation", side_effect=AssertionError("malformed/failed LLM never needed")), \
         patch.object(llm_explainer, "get_deepseek_client", side_effect=AssertionError("no key")):
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
    scope = ui_functions("search_photos_tool", search_photos=backend_search, metadata=METADATA)
    dialogue.scope["search_photos_tool"] = scope["search_photos_tool"]
    dialogue.turn(QUERY)
    assert dialogue.create.call_count == (0 if mode in {"direct", "third_cap"} else 1)
    assert len(dialogue.rendered) == 3
    assert dialogue.history[-1]["images"] == [r["img_path"] for r in RESULTS]
    assert "逐图评分说明" in dialogue.history[-1]["report"]
    assert any("已有识别" in s for s in dialogue.outputs)
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


def test_metadata_and_results_are_immutable_and_scores_do_not_control_prose():
    before = deepcopy(METADATA), deepcopy(RESULTS)
    first = explainer.score_explanation(QUERY, RESULTS[0], METADATA)
    changed_score = dict(RESULTS[0], score=1)
    assert explainer.score_explanation(QUERY, changed_score, METADATA) == first
    assert (METADATA, RESULTS) == before


@pytest.mark.parametrize("path", ["data/travel/travel_33.HEIC", "data/travel/travel_36.HEIC", "data/travel/travel_40.HEIC"])
def test_nested_general_fields_description_not_lost_to_empty_top_level_fields(path):
    text = explainer.score_explanation("query", dict(img_path=path, score=50, trace=[]), METADATA)
    assert text.startswith("已有识别描述：")
    assert "暂无可用" not in text
    assert ["大学的校门", "传统中式风格", "夜间的古代寺庙"][
        ["data/travel/travel_33.HEIC", "data/travel/travel_36.HEIC", "data/travel/travel_40.HEIC"].index(path)] in text


def test_pet_description_does_not_use_generic_person_fields_or_append_slot_dump():
    r = dict(img_path="data/pet/pet_01.jpg", score=70, trace=[("pet_animal_match", .16)])
    text = explainer.score_explanation("屋顶上的猫", r, METADATA)
    assert "橘黄色猫咪站在黑色屋檐瓦片上" in text
    assert "【宠物】" not in text and "单人" not in text and "人种" not in text


@pytest.mark.parametrize("rule,evidence,expected", [
    ("object_core_missing", "缺少核心对象: 机器人", "对象「机器人」尚未获识别记录支持"),
    ("entity_keyword_missing", "未命中实体: 红帽子", "线索「红帽子」尚未获识别记录支持"),
])
def test_missing_word_is_not_asserted_to_be_absent_from_image(rule, evidence, expected):
    text = explainer.score_explanation("query", dict(score=30, trace=[(rule, -.2, evidence)]))
    assert expected in text
    assert "待确认的是" in text and "不太符合的是" not in text


def test_actual_recorded_conflict_precedes_other_uncertain_penalties():
    r = dict(img_path="cat.jpg", score=30, trace=[
        ("geo_mismatch_penalty", -1.5, "未满足地理约束: 晋祠"),
        ("pet_color_mismatch", -.03),
    ])
    text = explainer.score_explanation("黑猫", r, {"cat.jpg": {
        "description": "一只白猫趴在窗边。", "pet_details": {"coat_color": ["白色"]}}})
    assert "不太符合的是：毛色记录为「白色」，与要求不符" in text
    assert "具体原因" not in text and "黑猫" not in text.split("对应查询")[0]


@pytest.mark.parametrize("info,expected", [
    ({"scene": "室外庭院"}, "只标注「室外庭院」场景"),
    ({"ocr_text": ["西游打卡处"]}, "已有文字识别为「西游打卡处」"),
    ({"keywords": ["猫", "屋顶"]}, "已有识别标签为「猫、屋顶」"),
    ({"description": "未知", "scene": None, "keywords": {}}, "暂无可用的已有识别描述"),
    (None, "暂无可用的已有识别描述"),
])
def test_sparse_and_malformed_metadata_fallbacks_are_explicit(info, expected):
    text = explainer.score_explanation("海滩上的金毛", dict(img_path="one", score=50, trace=[]), {"one": info})
    assert expected in text
    assert "金毛" not in text and "海滩" not in text


def test_report_uses_saved_entities_for_specific_uncertainty_without_extraction():
    report = explainer.ensure_score_explanations(saved_report(), QUERY, RESULTS, METADATA)
    paragraphs = report.split(explainer.SCORE_EXPLANATION_HEADING)[1]
    assert "「西游记、打卡」尚未获记录支持" in paragraphs
    assert "不太符合的是" not in paragraphs


def test_saved_entities_not_taken_from_user_query_or_wrong_report_query():
    deceptive_query = "晋祠\n**提取实体**：火星"
    report = saved_report(query=deceptive_query)
    text = explainer.ensure_score_explanations(report, deceptive_query, RESULTS, METADATA)
    assert "火星" not in text.split(explainer.SCORE_EXPLANATION_HEADING)[1]
    text = explainer.ensure_score_explanations(saved_report(), "另一段查询", RESULTS, METADATA)
    assert "「西游记、打卡」尚未获记录支持" not in text.split(explainer.SCORE_EXPLANATION_HEADING)[1]


def test_old_owned_score_only_sections_upgrade_without_touching_numeric_report():
    numeric = saved_report().rstrip("\n")
    old = numeric + "\n\n" + explainer.SCORE_EXPLANATION_HEADING + "\n\n"
    old += "\n\n".join(f"**图片{i} · `{r['img_path']}`**\n\n针对「晋祠」，此图最终得分{r['score']}（非概率）。语义加权贡献+0.3490；规则支持：旧文字。"
                           for i, r in enumerate(RESULTS, 1))
    upgraded = explainer.ensure_score_explanations(old, metadata=METADATA)
    assert upgraded.startswith(numeric + "\n\n")
    assert upgraded.count(explainer.SCORE_EXPLANATION_HEADING) == 1
    assert "已有识别描述：一张石雕武士像" in upgraded
    assert "旧文字" not in upgraded and "语义加权贡献" not in upgraded
    assert explainer.ensure_score_explanations(upgraded, metadata=METADATA) == upgraded


def test_custom_existing_same_heading_prose_is_not_overwritten():
    report = saved_report() + "\n\n" + explainer.SCORE_EXPLANATION_HEADING + "\n\n用户自己写的说明。"
    assert explainer.ensure_score_explanations(report, metadata=METADATA) == report


def test_all_catalogue_descriptions_use_available_evidence_and_fit_short_paragraph():
    for path in METADATA:
        r = dict(img_path=path, score=50, trace=[("category_match", .22, "类别匹配: " + METADATA[path]["category"])])
        text = explainer.score_explanation("query", r, METADATA)
        assert text.startswith("已有识别描述："), path
        assert 65 <= len(text) <= 150, (path, len(text))


def test_statue_generic_person_count_does_not_become_confirmed_human_conflict():
    r = dict(img_path="data/travel/travel_23.HEIC", score=50,
             trace=[("person_count_mismatch", -.1, "人物数量不符")])
    text = explainer.score_explanation("两个人", r, METADATA)
    assert "石雕武士像" in text
    assert "实际人数仍需对照原图核实" in text
    assert "不太符合的是" not in text and "单人" not in text


def test_new_ui_import_ignores_stale_cached_score_report_and_llm_modules():
    import sys
    from types import ModuleType
    old_report = ModuleType("score_report")
    old_report.ensure_score_explanations = Mock(side_effect=AssertionError("stale report must not be called"))
    old_llm = ModuleType("llm_explainer")
    tree = ast.parse((ROOT / "streamlit_ui.py").read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and
               any(alias.name == "ensure_score_explanations" for alias in node.names)]
    assert len(imports) == 1 and imports[0].module == "photo_report_narrative"
    namespace = {}
    with patch.dict(sys.modules, {"score_report": old_report, "llm_explainer": old_llm}):
        exec(compile(ast.Module(body=imports, type_ignores=[]), "ui_report_import", "exec"), namespace)
        text = namespace["ensure_score_explanations"](saved_report(), QUERY, RESULTS, metadata=METADATA)
    assert "已有识别描述：一张石雕武士像" in text
    old_report.ensure_score_explanations.assert_not_called()


def test_legacy_upgrade_preserves_original_numeric_trailing_newlines_exactly():
    numeric = saved_report() + "\n\n\n"
    old = numeric + "\n\n" + explainer.SCORE_EXPLANATION_HEADING + "\n\n"
    old += "**图片1 · `data/travel/travel_23.HEIC`**\n\n针对「晋祠」，此图最终得分82（非概率）。语义加权贡献+0.3490；旧说明。"
    upgraded = explainer.ensure_score_explanations(old, metadata=METADATA)
    assert upgraded.split(explainer.SCORE_EXPLANATION_HEADING)[0] == numeric + "\n\n"
