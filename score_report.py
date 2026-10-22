"""Pure per-photo report rendering, independent of the cached LLM client module."""
import math
import re


SCORE_EXPLANATION_HEADING = "#### 逐图评分说明"

# Some original scoring rules store only their name, not a descriptive value.
# Translate that name, but never infer an unrecorded breed, colour or count.
_RULE_LABELS = {
    "pet_animal_match": "动物种类符合查询",
    "pet_animal_mismatch": "动物种类与查询不符",
    "pet_breed_match": "品种符合查询",
    "pet_breed_mismatch": "品种与查询不符",
    "pet_color_match": "毛色符合查询",
    "pet_color_mismatch": "毛色与查询不符",
    "pet_action_match": "动物动作符合查询",
    "pet_env_match": "动物环境符合查询",
    "pet_count_match": "动物数量符合查询",
    "pet_count_mismatch": "动物数量与查询不符",
    "type_phone_screenshot_prior": "截图类型先验（并非线索命中）",
}


def _report_text(value, limit):
    text = " ".join(str(value).split()).replace("`", "")
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
        if rule == "type_phone_screenshot_prior":
            evidence = _RULE_LABELS[rule]
        elif evidence == rule:
            evidence = _RULE_LABELS.get(rule, "未保存文字依据的规则项")
        trace.append((rule, delta, evidence))
    return trace


def score_explanation(user_query, result):
    """About 100 Chinese characters, grounded exclusively in this result's trace."""
    trace = _report_trace(result)
    semantic = [item for item in trace if item[0] == "base_semantic_similarity"]
    positive = [item for item in trace if item[0] != "base_semantic_similarity" and item[1] > 0]
    negative = [item for item in trace if item[0] != "base_semantic_similarity" and item[1] < 0]
    # Specific query/entity evidence is more informative than a broad category.
    positive.sort(key=lambda item: item[0] not in {"entity_keyword_match", "object_core_match"})
    prefix = f"针对「{_report_text(user_query, 18)}」，此图最终得分{result['score']}（非概率）。"
    if not trace:
        return (prefix + "该历史结果未保存逐项评分依据，无法确认具体命中线索或扣分原因。"
                "这里只保留已有得分，不根据照片描述补造证据，也不重新检索或改变原排序。")
    sem = (f"语义加权贡献{semantic[0][1]:+.4f}；" if semantic else "未保存语义贡献数值；")
    if positive:
        # Keep multiple clues where possible; the original full trace remains above.
        evidence = "；".join(item[2] for item in positive)
        support = "规则支持：" + _report_text(evidence, 38 if not negative else 29) + "。"
    else:
        support = "未记录规则加分，不能把语义相似视为具体线索已命中。"
    if negative:
        evidence = "；".join(item[2] for item in negative)
        limit = 29 if positive else 38
        caveat = "扣分依据：" + _report_text(evidence, limit) + "。"
        ending = "各项加减贡献作用于映射前原始分，完整依据见明细。"
    else:
        caveat = "未记录规则扣分，不等于所有线索均满足。"
        ending = "完整的逐项加减贡献及对应证据见上方明细。"
    return prefix + sem + support + caveat + ending


def generate_score_explanations(user_query, top_results):
    """Local, deterministic report text: no keys, provider, metadata or shared cache."""
    blocks = []
    for index, result in enumerate(top_results, 1):
        blocks.append(f"**图片{index} · `{result['img_path']}`**\n\n" + score_explanation(user_query, result))
    return "\n\n".join(blocks)


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


def ensure_score_explanations(report, user_query=None, top_results=None):
    """Add per-photo prose, preserving the existing numeric report byte for byte.

    Legacy messages contain only a rendered report, so parse its saved evidence.
    Unknown formats stay untouched. Existing complete reports are idempotent.
    """
    if not isinstance(report, str) or re.search(r"^#### 逐图评分说明\s*$", report, re.M):
        return report
    try:
        if top_results is None:
            # Keep valid old prose, but recover empty/failed old LLM sections.
            old_section = re.search(r"^#### 🤖 智能推理说明\s*$", report, re.M)
            numeric_report = report
            if old_section:
                old_text = report[old_section.end():].strip()
                if old_text and not old_text.startswith(("解释生成失败", "智能说明暂不可用")):
                    return report
                numeric_report = report[:old_section.start()]
            user_query, top_results = _stored_report_results(numeric_report)
        if not top_results or user_query is None:
            return report
        return report + "\n\n" + SCORE_EXPLANATION_HEADING + "\n\n" + generate_score_explanations(user_query, top_results)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        # A malformed optional report must never discard successfully found photos.
        return report
