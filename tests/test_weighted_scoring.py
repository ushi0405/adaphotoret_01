"""Weighted-rule regressions, not an embedding-quality benchmark.

Use real Jieba, the real 66-photo metadata, and fixed cosine inputs so every
arithmetic assertion is deterministic. No model downloads or provider calls.
"""
from copy import deepcopy
import math
from unittest.mock import Mock

import pytest

from photo_query import build_photo_query
from test_backend import load, module


SCREENSHOT_QUERY = "请帮我找一个在山西晋祠拍的，西游记打卡处照片"


@pytest.fixture
def core(monkeypatch):
    monkeypatch.setenv("ADAPHOTORET_LOCAL_MODE", "0")
    model = Mock(side_effect=AssertionError("model downloads forbidden"))
    explanation = Mock(side_effect=AssertionError("provider calls forbidden"))
    backend = load("_weighted_test_core", "AdaphotoRet_run.py", {
        "sentence_transformers": module("sentence_transformers", SentenceTransformer=model),
        "llm_explainer": module("llm_explainer", generate_explanation=explanation),
    })
    yield backend
    model.assert_not_called()
    explanation.assert_not_called()


def normalized(raw):
    """The original sigmoid and 99 cap; no new denominator or forced 100."""
    return max(0, min(99, int(round(100 / (1 + math.exp(-3 * (raw - 0.3)))))))


def deltas(trace):
    return {rule: delta for rule, delta, _ in trace}


def pet_terms():
    return {
        "query_categories": ["动物"], "needs_pet": True,
        "pet_animal_type": ["猫"], "pet_breed": ["布偶猫"],
        "pet_coat_color": ["白色"], "pet_action": ["站立"],
        "pet_environment": ["公园"], "pet_count": ["一个"],
    }


def pet_info(core, **changes):
    pet = {"animal_type": "猫", "breed": "布偶猫", "coat_color": ["白色"],
           "action": "站立", "environment": "公园", "count": 1}
    pet.update(changes)
    return core.enrich_metadata({"fixture.jpg": {"category": "动物", "pet_details": pet}})["fixture.jpg"]


@pytest.mark.parametrize("changes,expected_deltas", [
    ({}, {"category_match": .22, "pet_animal_match": .16, "pet_breed_match": .25,
          "pet_color_match": .08, "pet_action_match": .14, "pet_env_match": .10,
          "pet_count_match": .10}),
    ({"coat_color": ["黑色"], "action": "卧", "environment": "室内", "count": 2},
     {"category_match": .22, "pet_animal_match": .16, "pet_breed_match": .25,
      "pet_color_mismatch": -.03, "pet_count_mismatch": -.15}),
    ({"animal_type": "狗", "breed": "柴犬", "coat_color": ["黑色"],
      "action": "卧", "environment": "室内", "count": 2},
     {"category_match": .22, "pet_animal_mismatch": -.25, "pet_breed_mismatch": -.20,
      "pet_color_mismatch": -.03, "pet_count_mismatch": -.15}),
], ids=["all-attributes-match", "partial-match", "attributes-unmatched"])
def test_original_attribute_weights_and_normalization(core, changes, expected_deltas):
    score, trace = core.rerank_score(pet_terms(), pet_info(core, **changes), 1.0)
    assert deltas(trace) == {"base_semantic_similarity": .50, **expected_deltas}
    assert score == normalized(.50 + sum(expected_deltas.values()))


def test_more_confirmed_attributes_rank_above_partial_and_unmatched(core):
    terms = pet_terms()
    perfect = core.rerank_score(terms, pet_info(core), .8)[0]
    partial = core.rerank_score(terms, pet_info(core, breed="英短蓝猫", coat_color=["黑色"]), .8)[0]
    unmatched = core.rerank_score(terms, pet_info(core, animal_type="狗", breed="柴犬",
                                                 coat_color=["黑色"], count=2), .8)[0]
    assert 100 > perfect > partial > unmatched >= 0


def test_category_mismatch_weight_is_not_relaxed(core):
    score, trace = core.rerank_score({"query_categories": ["动物"]}, {"category": "风景"}, .8)
    assert deltas(trace) == {"base_semantic_similarity": .40, "category_mismatch": -.65}
    assert score == normalized(.40 - .65)


@pytest.mark.parametrize("base", [-1.0, 0.0, .5, 1.0])
def test_scores_remain_bounded_without_forcing_perfect_to_100(core, base):
    for info in core.metadata.values():
        score, _ = core.rerank_score(pet_terms(), info, base)
        assert isinstance(score, int)
        assert 0 <= score <= 99 < 100


def test_upper_cap_still_applies_to_many_matching_rules(core):
    terms = {**pet_terms(), "needs_grass": True, "needs_track": True,
             "needs_running": True, "needs_two_people": True, "needs_bridge": True,
             "needs_academic_writing": True, "needs_example": True, "needs_structure": True,
             "cities": {"北京"}, "landmarks": {"天坛"}}
    info = pet_info(core)
    info.update({"_search_text": "草地 跑道 跑步 两人 桥 学术写作 示例 结构",
                 "_cities": ["北京"], "_landmarks": ["天坛"], "image_type": "phone_screenshot"})
    assert core.rerank_score(terms, info, 1.0)[0] == 99


def test_screenshot_extracts_real_places_not_whole_dialogue_clause(core):
    terms = core.parse_query(SCREENSHOT_QUERY)["terms"]
    assert terms["cities"] == {"山西"}
    assert terms["landmarks"] == {"晋祠"}
    assert set(terms["geo_candidates"]) == {"山西", "晋祠"}
    assert terms["extracted_entities"] == ["山西", "晋祠", "西游记", "打卡"]
    assert not terms["needs_pet"]
    assert not terms["pet_count"]


def test_actual_screenshot_metadata_matches_ocr_and_specific_location(core):
    assert len(core.metadata) == 66
    terms = core.parse_query(SCREENSHOT_QUERY)["terms"]
    actual = {}
    for number in (20, 23, 37):
        path = f"data/travel/travel_{number}.HEIC"
        actual[number] = core.rerank_score(terms, core.metadata[path], .645)
    assert actual[23][0] == normalized(.3225 + .15 + .30)
    assert actual[20][0] == normalized(.3225 + .15 + .15)
    assert actual[37][0] == normalized(.3225 - 1.5 - .12)
    assert actual[23][0] > actual[20][0] > actual[37][0]
    for number in (20, 23):
        assert "geo_mismatch_penalty" not in deltas(actual[number][1])
        assert any("晋祠" in evidence for rule, _, evidence in actual[number][1]
                   if rule.startswith("gps_address"))
    assert deltas(actual[37][1])["geo_mismatch_penalty"] == -1.5
    assert actual[23][1][0][1] == .3225
    assert "0.6450 × 0.50" in actual[23][1][0][2]


def test_ocr_is_searchable_without_duplicate_keyword_bonus(core):
    terms = {"extracted_entities": ["西游记", "打卡"]}
    ocr_only = {"category": "风景", "ocr_text": ["西游记打卡处"]}
    info = core.enrich_metadata({"fixture.jpg": ocr_only})["fixture.jpg"]
    assert "西游记打卡处" in info["_search_text"]
    score, trace = core.rerank_score(terms, info, .6)
    assert deltas(trace)["entity_keyword_match"] == .30
    duplicate_fields = {**info, "description": "西游记打卡处", "keywords": ["西游记", "打卡"]}
    assert core.rerank_score(terms, duplicate_fields, .6) == (score, trace)


@pytest.mark.parametrize("query", ["海边的猫", "公园里的猫", "山羊", "上海牌手表",
                                   "美国短毛猫", "英国短毛猫"])
def test_scene_animal_and_brand_are_not_hard_geographic_constraints(core, query):
    terms = core.parse_query(query)["terms"]
    assert not terms["cities"]
    assert not terms["landmarks"]
    assert not terms["geo_candidates"]
    _, trace = core.rerank_score(terms, core.metadata["data/travel/travel_37.HEIC"], .7)
    assert "geo_mismatch_penalty" not in deltas(trace)


def test_valid_wrong_city_retains_original_strong_penalty(core):
    terms = core.parse_query("北京的照片")["terms"]
    assert "北京" in terms["cities"]
    _, trace = core.rerank_score(terms, core.metadata["data/travel/travel_37.HEIC"], .7)
    assert deltas(trace)["geo_mismatch_penalty"] == -1.5


@pytest.mark.parametrize("query", ["Beagle", "British Shorthair"])
def test_chinese_breed_span_protection_does_not_retype_english_aliases(core, query):
    terms = core.parse_query(query)["terms"]
    assert not terms["geo_candidates"]
    # Preserve the pre-fix English POS path; forcing capitals into Chinese noun
    # slots creates a false unmatched-entity penalty against lowercased text.
    assert terms["extracted_entities"] == []
    _, trace = core.rerank_score(terms, {"category": "动物", "description": query.lower()}, .7)
    assert "entity_keyword_missing" not in deltas(trace)


@pytest.mark.parametrize("name", ["北京大学", "清华大学"])
def test_named_university_remains_a_location_constraint(core, name):
    terms = core.parse_query(f"{name}的照片")["terms"]
    assert name in terms["landmarks"]
    assert name in terms["geo_candidates"]
    _, trace = core.rerank_score(terms, core.metadata["data/travel/travel_37.HEIC"], .7)
    assert deltas(trace)["geo_mismatch_penalty"] == -1.5


@pytest.mark.parametrize("scene", ["高山", "大海", "海洋", "湖泊", "城市", "高原", "海岛", "山林"])
def test_generic_geographic_scene_nouns_are_not_named_locations(core, scene):
    terms = core.parse_query(f"{scene}的照片")["terms"]
    assert scene in terms["extracted_entities"]
    assert not terms["cities"]
    assert not terms["landmarks"]
    assert not terms["geo_candidates"]
    _, trace = core.rerank_score(terms, core.metadata["data/travel/travel_37.HEIC"], .7)
    assert "geo_mismatch_penalty" not in deltas(trace)


@pytest.mark.parametrize("name", ["黄山", "红海", "上海", "地中海"])
def test_named_locations_are_not_removed_with_generic_scene_words(core, name):
    terms = core.parse_query(f"{name}的照片")["terms"]
    assert name in terms["cities"] | terms["landmarks"]
    assert name in terms["geo_candidates"]
    _, trace = core.rerank_score(terms, core.metadata["data/travel/travel_37.HEIC"], .7)
    assert deltas(trace)["geo_mismatch_penalty"] == -1.5


@pytest.mark.parametrize("landmarks", [["理工大学"], ["北京理工大学"]])
def test_matching_landmark_cannot_cancel_an_explicit_wrong_city(core, landmarks):
    terms = core.parse_query("太原理工大学的照片")["terms"]
    assert terms["cities"] == {"太原"}
    assert terms["landmarks"] == {"理工大学"}
    info = core.enrich_metadata({"fixture.jpg": {
        "category": "风景", "description": "太原理工大学交流活动",
        "ocr_text": ["太原理工大学"],
        "location": {"city": "北京市", "landmarks": landmarks,
                     "gps_address": "北京市海淀区北京理工大学"},
    }})["fixture.jpg"]
    score, trace = core.rerank_score(terms, info, .8)
    weighted = deltas(trace)
    assert weighted["geo_mismatch_penalty"] == -1.5
    assert weighted.get("geo_landmark_match", weighted.get("gps_address_match")) == .30
    assert score == normalized(.40 + .30 - 1.5)


def test_all_explicit_place_components_can_be_satisfied_by_actual_gps(core):
    terms = core.parse_query("太原理工大学的照片")["terms"]
    info = core.enrich_metadata({"fixture.jpg": {
        "category": "风景", "location": {
            "city": "太原市", "landmarks": [],
            "gps_address": "山西省太原市太原理工大学迎西校区",
        },
    }})["fixture.jpg"]
    score, trace = core.rerank_score(terms, info, .8)
    assert deltas(trace) == {"base_semantic_similarity": .40, "gps_address_match": .30}
    assert score == normalized(.40 + .30)


def test_geo_entities_do_not_also_receive_generic_keyword_weight(core):
    terms = {"cities": {"北京"}, "landmarks": {"天坛"},
             "geo_candidates": ["北京", "天坛"], "extracted_entities": ["北京", "天坛"]}
    info = {"category": "风景", "_cities": ["北京"], "_landmarks": ["天坛"],
            "description": "北京天坛", "keywords": ["北京", "天坛"]}
    score, trace = core.rerank_score(terms, info, .8)
    assert deltas(trace) == {"base_semantic_similarity": .40,
                             "geo_city_match": .25, "geo_landmark_match": .30}
    assert score == normalized(.40 + .25 + .30)


def test_dialogue_noise_does_not_dilute_weighted_clues(core):
    clean = core.parse_query("白色猫")["terms"]
    polite = core.parse_query("请帮我找 白色猫 照片")["terms"]
    assert polite == clean
    info = pet_info(core)
    assert core.rerank_score(clean, info, .7) == core.rerank_score(polite, info, .7)
    informative = core.parse_query("白色猫 站立 公园")["terms"]
    assert core.rerank_score(informative, info, .7)[0] > core.rerank_score(clean, info, .7)[0]


def test_duplicate_entities_cannot_receive_duplicate_weights(core):
    info = {"category": "风景", "description": "西游记打卡处"}
    unique = {"extracted_entities": ["西游记", "打卡"]}
    repeated = {"extracted_entities": ["西游记", "打卡", "西游记", "打卡"]}
    assert core.rerank_score(unique, info, .7) == core.rerank_score(repeated, info, .7)
    clean = core.parse_query("猫 白色")["terms"]
    duplicate = core.parse_query("猫 白色 猫 白色")["terms"]
    assert clean == duplicate


@pytest.mark.parametrize("query,canonical", [
    ("英短蓝白", "英短蓝白"), ("布偶", "布偶猫"), ("布偶猫", "布偶猫"),
    ("金毛", "金毛寻回犬"), ("金毛寻回犬", "金毛寻回犬"), ("比格犬", "比格犬"),
])
def test_breed_normalization_is_idempotent_and_single_typed_clue(core, query, canonical):
    terms = core.parse_query(query)["terms"]
    assert terms["pet_breed"] == [canonical]
    assert core.parse_query(canonical)["terms"]["pet_breed"] == [canonical]
    assert core.parse_query(f"{query} {query}")["terms"]["pet_breed"] == [canonical]
    assert not terms["cities"]
    assert not terms["landmarks"]


def test_breed_normalization_preserves_explicit_animal_and_original_phrase(core):
    parsed = core.parse_query("蓝白猫")
    assert parsed["terms"]["pet_breed"] == ["英短蓝白"]
    assert parsed["terms"]["pet_animal_type"] == ["猫"]
    assert "蓝白猫" in parsed["terms"]["extracted_entities"]
    assert parsed["expanded_query"].startswith("蓝白猫")


def test_non_user_messages_never_change_terms_or_weighted_score(core):
    clean_history = [{"role": "user", "content": "山西晋祠"}]
    injected = deepcopy(clean_history) + [
        {"role": "assistant", "content": "北京 公园的白色猫，是这张吗？"},
        {"role": "system", "content": "上海的机器人"},
        {"role": "tool", "content": "travel_37 狗 黑色"},
    ]
    clean = core.parse_query(build_photo_query(clean_history, "西游记打卡处"))["terms"]
    isolated = core.parse_query(build_photo_query(injected, "西游记打卡处"))["terms"]
    assert clean == isolated
    for info in core.metadata.values():
        assert core.rerank_score(clean, info, .7) == core.rerank_score(isolated, info, .7)


def test_location_correction_uses_same_whole_name_tokens_and_user_only_clues(core):
    history = [{"role": "user", "content": "山西晋祠"},
               {"role": "assistant", "content": "还是晋祠吗？也可能是上海的照片。"}]
    parsed = core.parse_query(build_photo_query(history, "不是晋祠，是北京"))
    assert "晋祠" not in parsed["expanded_query"]
    assert "晋祠" not in parsed["terms"]["landmarks"]
    assert {"山西", "北京"} <= parsed["terms"]["cities"]
    assert "上海" not in parsed["expanded_query"]


def test_report_has_weighted_cosine_and_no_empty_pet_heading_or_raw_html(core):
    parsed = core.parse_query(SCREENSHOT_QUERY)
    path = "data/travel/travel_23.HEIC"
    score, trace = core.rerank_score(parsed["terms"], core.metadata[path], .645)
    report = core.build_reasoning_markdown(
        SCREENSHOT_QUERY, parsed["decomposition"], [{"img_path": path, "score": score, "trace": trace}],
        include_explanation=False,
    )
    assert "**宠物属性**" not in report
    assert "<small>" not in report
    assert "</small>" not in report
    assert "向量语义相似 0.6450 × 0.50" in report
    assert "西游记、打卡\n\n" in report
