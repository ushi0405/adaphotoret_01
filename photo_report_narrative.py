"""Local per-photo narratives, independent of cached LLM/scoring modules.

Descriptions come from saved catalogue metadata, not a new image inspection.
Matches and conflicts come from the existing trace; missing text is uncertainty,
not proof that something is absent from the picture.
"""
import math
import re


SCORE_EXPLANATION_HEADING = "#### 逐图评分说明"
_NARRATIVE_MARKER = "<!-- photo-narrative:v2 -->"

_RULE_LABELS = {
    "pet_animal_match": "动物种类符合查询",
    "pet_breed_match": "品种符合查询",
    "pet_color_match": "毛色符合查询",
    "pet_action_match": "动物动作符合查询",
    "pet_env_match": "动物环境符合查询",
    "pet_count_match": "动物数量符合查询",
    "scene_grass_match": "草地场景符合查询",
    "scene_track_match": "跑道场景符合查询",
    "action_running_match": "奔跑动作符合查询",
    "entity_two_people_match": "两人线索符合查询",
    "scene_bridge_match": "桥梁场景符合查询",
    "domain_academic_writing_match": "学术写作主题符合查询",
    "intent_example_match": "示例范文线索符合查询",
    "intent_structure_match": "结构讲解线索符合查询",
}
_PET_FIELDS = {
    "pet_animal_mismatch": ("animal_type", "动物种类"),
    "pet_breed_mismatch": ("breed", "品种"),
    "pet_color_mismatch": ("coat_color", "毛色"),
    "pet_count_mismatch": ("count", "动物数量"),
}
_UNKNOWN = {"", "无", "未知", "不详", "不确定", "无关", "none", "unknown", "null"}


def _report_text(value, limit):
    text = " ".join(str(value).split()).replace("`", "")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _value_text(value):
    """Reject malformed fields and unknown placeholders rather than describe them."""
    if isinstance(value, (list, tuple)):
        return "、".join(filter(None, (_value_text(item) for item in value)))
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    value = _report_text(value, 2000)
    return "" if value.lower() in _UNKNOWN else value


def _field(info, name):
    value = _value_text(info.get(name))
    general = info.get("general_fields")
    return value or (_value_text(general.get(name)) if isinstance(general, dict) else "")


def _photo_info(result, metadata):
    info = metadata.get(result.get("img_path"), {}) if isinstance(metadata, dict) else {}
    return info if isinstance(info, dict) else {}


def _excerpt(text, limit):
    """Prefer a clause boundary, so the short description remains readable."""
    text = text.strip(" ，,。；;：:")
    if len(text) <= limit:
        return text
    boundaries = [m.start() for m in re.finditer("[，,。；;]", text[:limit + 1])]
    if boundaries and boundaries[-1] >= limit // 2:
        return text[:boundaries[-1]]
    return _report_text(text, limit)


def _description(info):
    description = _field(info, "description").split("【宠物】", 1)[0].strip()
    description = re.sub(r"^(?:这张照片展示了|这张照片展示|这是一张展示|照片中可以看到)", "", description)
    if description:
        return "已有识别描述：" + _excerpt(description, 48) + "。"
    scene = _field(info, "scene")
    if scene:
        return "已有识别只标注「" + _excerpt(scene, 30) + "」场景，尚无详细画面描述。"
    ocr = _field(info, "ocr_text")
    if ocr:
        return "已有文字识别为「" + _excerpt(ocr, 28) + "」，尚无完整画面描述。"
    keywords = _field(info, "keywords")
    if keywords:
        return "已有识别标签为「" + _excerpt(keywords, 28) + "」，尚无完整画面描述。"
    return "这张图暂无可用的已有识别描述，不能仅凭查询词或得分推断画面内容。"


def _report_trace(result):
    """Read saved contributions only; never run extraction/scoring or mutate data."""
    trace = []
    for item in result.get("trace") or []:
        if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
            continue
        rule, delta = item[:2]
        try:
            delta = float(delta)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(delta):
            continue
        rule = str(rule)
        evidence = str(item[2]) if len(item) == 3 and item[2] else rule
        if evidence == rule:
            evidence = _RULE_LABELS.get(rule, "")
        trace.append((rule, delta, evidence))
    return trace


def _terms(evidence):
    parts = re.split(r"[:：]", evidence, maxsplit=1)
    return [term.strip() for term in re.split(r"[,，、]", parts[1]) if term.strip()] if len(parts) == 2 else []


def _quoted(terms, limit=20):
    return "「" + _report_text("、".join(terms), limit) + "」"


def _support(trace, info):
    positive = [item for item in trace if item[1] > 0 and item[0] not in {
        "base_semantic_similarity", "type_phone_screenshot_prior"}]
    priorities = {"entity_keyword_match": 0, "object_core_match": 0,
                  "gps_address_match": 1, "gps_address_partial_match": 1,
                  "geo_city_match": 1, "geo_landmark_match": 1, "category_match": 3}
    positive.sort(key=lambda item: priorities.get(item[0], 2))
    statements = []
    for rule, _, evidence in positive:
        terms = _terms(evidence)
        if rule in {"entity_keyword_match", "object_core_match"} and terms:
            if all(term in _field(info, "ocr_text") for term in terms):
                text = "文字识别含" + _quoted(terms)
            elif all(term in (_field(info, "description") + _field(info, "keywords") + _field(info, "scene")) for term in terms):
                text = "描述或标签对应" + _quoted(terms)
            else:
                text = "检索记录支持" + _quoted(terms)
        elif rule in {"gps_address_match", "gps_address_partial_match"} and terms:
            text = "地址记录对应" + _quoted(terms)
        elif rule in {"geo_city_match", "geo_landmark_match"} and terms:
            text = "位置记录对应" + _quoted(terms)
        elif rule == "category_match" and terms:
            text = "类别" + _quoted(terms) + "相符"
        elif rule in _RULE_LABELS:
            text = _RULE_LABELS[rule]
        elif rule == "person_count_match":
            text = "人物数量线索相符"
        elif evidence:
            text = "检索记录：" + _report_text(evidence, 22)
        else:
            continue
        if text not in statements:
            statements.append(text)
        if len(statements) >= 2:
            break
    if statements:
        return "对应查询，" + _excerpt("，".join(statements), 46) + "。"
    if any(rule == "type_phone_screenshot_prior" for rule, _, _ in trace):
        return "截图类型先验并非具体线索命中，当前记录不足以确认对应关系。"
    if trace:
        return "目前仅有整体相似或不足项记录，不能据此确认具体线索已符合。"
    return "也未保存匹配依据，无法确认它与当前线索的具体对应关系。"


def _negative_statement(rule, evidence, info):
    terms = _terms(evidence)
    if rule == "geo_mismatch_penalty":
        return ("地点" + _quoted(terms, 18) if terms else "所需地点") + "尚未获位置记录支持", False
    if rule in {"object_core_missing", "entity_keyword_missing"}:
        return ("对象" if rule == "object_core_missing" else "线索") + (_quoted(terms, 18) if terms else "") + "尚未获识别记录支持", False
    if rule in _PET_FIELDS:
        field, label = _PET_FIELDS[rule]
        pet = info.get("animal_details") or info.get("pet_details") or {}
        value = _value_text(pet.get(field)) if isinstance(pet, dict) else ""
        if value:
            return label + "记录为「" + _report_text(value, 16) + "」，与要求不符", True
        return label + "记录不足，评分中的不符提示仍需核实", False
    if rule == "person_count_mismatch":
        # Raw/legacy generic counts also occur on pets and sculptures. A
        # scoring mismatch is not independent evidence of real human presence.
        return "人物数量规则提示不符，实际人数仍需对照原图核实", False
    if rule == "scene_grass_track_conflict":
        return "已记录跑道场景，与草地要求冲突", True
    if rule == "intent_example_vs_structure_penalty":
        return "记录为结构讲解页，而非所需示例范文", True
    if rule == "category_mismatch" and evidence:
        return "类别记录不符：" + _report_text(evidence, 28), True
    return "另有不足项记录，具体原因仍需核实", False


def _unverified_clues(query_clues, trace, info):
    # These are the EXISTING report's extracted entities. This is a literal
    # evidence-coverage check, not new extraction or a new ranking constraint.
    fields = [_field(info, name) for name in ("description", "scene", "keywords", "ocr_text")]
    location = info.get("location")
    if isinstance(location, dict):
        fields.extend(_value_text(location.get(name)) for name in ("city", "landmarks", "gps_address"))
    fields.extend(evidence for rule, delta, evidence in trace if delta > 0 and rule != "base_semantic_similarity")
    haystack = " ".join(fields).lower()
    return [term for term in query_clues if isinstance(term, str) and len(term) > 1 and term.lower() not in haystack]


def _caveat(trace, info, query_clues):
    negative = [item for item in trace if item[0] != "base_semantic_similarity" and item[1] < 0]
    if negative:
        statements = [_negative_statement(rule, evidence, info) for rule, _, evidence in negative]
        # Explain an actual conflict before missing evidence, regardless of trace order.
        statements.sort(key=lambda item: not item[1])
        text, conflict = statements[0]
        if conflict:
            return "不太符合的是：" + text + "；仍需对照原图确认。"
        return "待确认的是：" + text + "，这不等于画面中一定没有。"
    missing = _unverified_clues(query_clues, trace, info)
    if missing:
        return "不过" + _quoted(missing[:2], 18) + "尚未获记录支持，需看原图核实，不能直接判为不存在。"
    return "暂未记录明确冲突，但未覆盖的细节仍待核实，不能视为所有线索都已满足。"


def score_explanation(user_query, result, metadata=None, *, query_clues=()):
    """Roughly 100 Chinese characters: observed record → support → uncertainty.

    user_query is retained for caller compatibility. It is never repurposed as
    image evidence. query_clues must be previously extracted, saved report terms.
    """
    info = _photo_info(result, metadata)
    trace = _report_trace(result)
    return _description(info) + _support(trace, info) + _caveat(trace, info, query_clues)


def generate_score_explanations(user_query, top_results, metadata=None, *, query_clues=()):
    """Local deterministic prose; no keys, provider calls, disk I/O or shared cache."""
    return "\n\n".join(
        f"**图片{index} · `{result['img_path']}`**\n\n" +
        score_explanation(user_query, result, metadata, query_clues=query_clues)
        for index, result in enumerate(top_results, 1)
    )


def _stored_report_results(report):
    """Upgrade the known historic report format without consulting today's backend."""
    query_match = re.search(r"^### 查询「(.*?)」的可解释检索结果", report, re.M | re.S)
    if not query_match:
        return None, []
    remainder = report[query_match.end():]
    section = re.search(r"^#### 推理链（Top 3）\s*$", remainder, re.M)
    if not section:
        return None, []
    # A multiline query may quote report-like text; only read actual trace blocks.
    trace_report = re.split(r"^#### ", remainder[section.end():], maxsplit=1, flags=re.M)[0]
    entries = list(re.finditer(r"^\d+\. `([^`\n]+)` \| 最终得分: (\d+(?:\.\d+)?)\s*$", trace_report, re.M))
    results = []
    for index, entry in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(trace_report)
        block = trace_report[entry.end():end]
        tags = re.findall(r"\[([a-z_]+) ([+-]?\d+(?:\.\d+)?)\]", block)
        evidence_line = re.search(r"^(?:<small>)?📌 (.*?)(?:</small>)?$", block, re.M)
        evidence = evidence_line.group(1).split(" | ") if evidence_line else []
        # Only upgrade complete, unambiguous saved traces, never invent alignment.
        if not tags or len(tags) != len(evidence):
            return None, []
        results.append({"img_path": entry.group(1), "score": entry.group(2),
                        "trace": [(rule, float(delta), text) for (rule, delta), text in zip(tags, evidence)]})
    return query_match.group(1), results


def _saved_query_clues(report, user_query):
    query_match = re.search(r"^### 查询「(.*?)」的可解释检索结果", report, re.M | re.S)
    if not query_match or query_match.group(1) != user_query:
        return ()
    preamble = re.split(r"^#### ", report[query_match.end():], maxsplit=1, flags=re.M)[0]
    entities = re.search(r"^\*\*提取实体\*\*[：:]([^\n]*)$", preamble, re.M)
    return tuple(term.strip() for term in entities.group(1).split("、") if term.strip()) if entities else ()


def _owned_old_section(section):
    """Only replace our known score-only template, never arbitrary user/LLM prose."""
    bodies = re.split(r"^\*\*图片\d+ · `[^`\n]+`\*\*\s*$", section, flags=re.M)
    return (len(bodies) > 1 and not bodies[0].strip() and all(
        body.strip().startswith("针对「") and "（非概率）。" in body and
        any(marker in body for marker in ("语义加权贡献", "未保存语义贡献数值", "该历史结果未保存"))
        for body in bodies[1:]))


def ensure_score_explanations(report, user_query=None, top_results=None, metadata=None):
    """Append/upgrade our prose without changing any numeric report or result.

    History is reconstructed only from saved query/trace. The supplied catalogue
    may add an explicitly attributed description, never new scoring evidence.
    Unknown report formats and unrelated valid LLM explanations stay untouched.
    """
    if not isinstance(report, str):
        return report
    original = report
    section = re.search(r"^#### 逐图评分说明\s*$", report, re.M)
    if section:
        old_text = report[section.end():]
        if _NARRATIVE_MARKER in old_text or not _owned_old_section(old_text):
            return report
        prefix = report[:section.start()]
        # Our old writer inserted exactly two separator newlines. Remove only
        # those, preserving every original numeric-report byte on replacement.
        report = prefix[:-2] if prefix.endswith("\n\n") else prefix
    try:
        if top_results is None:
            old_section = re.search(r"^#### 🤖 智能推理说明\s*$", report, re.M)
            numeric_report = report
            if old_section:
                old_text = report[old_section.end():].strip()
                if old_text and not old_text.startswith(("解释生成失败", "智能说明暂不可用")):
                    return original
                numeric_report = report[:old_section.start()]
            user_query, top_results = _stored_report_results(numeric_report)
        if not top_results or user_query is None:
            return original
        clues = _saved_query_clues(report, user_query)
        prose = generate_score_explanations(user_query, top_results, metadata, query_clues=clues)
        return report + "\n\n" + SCORE_EXPLANATION_HEADING + "\n\n" + _NARRATIVE_MARKER + "\n\n" + prose
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        return original
