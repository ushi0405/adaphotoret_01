import os
import re
import math
from typing import Dict, List


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

def get_deepseek_client(api_key=None, *, allow_environment=True):
    """Create a request-scoped client; never cache a visitor's credentials."""
    if api_key is None and allow_environment:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("请先配置自己的 DeepSeek API 密钥")
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        timeout=30.0,
        max_retries=1,
    )


def safe_api_error(error):
    """Do not expose provider exceptions, request headers, or secret values."""
    status = getattr(error, "status_code", None)
    if status in (401, 403):
        return "API 密钥无效或没有权限，请在侧边栏检查自己的密钥。"
    if status == 402:
        return "API 账户余额不足，请检查自己的服务账户。"
    if status == 429:
        return "API 请求受限或额度不足，请稍后重试。"
    return "API 暂时不可用，请检查网络、密钥和账户额度后重试。"


def generate_explanation(
    user_query: str,
    top_results: List[Dict],
    metadata: Dict,
    *,
    client=None,
    api_key=None,
    allow_environment=True,
) -> str:
    if not top_results:
        return "无结果，无法生成解释。"

    info_text = ""
    for idx, r in enumerate(top_results, start=1):
        img_path = r["img_path"]
        img_info = metadata.get(img_path, {})
        desc = img_info.get("description", "无描述")
        scene = img_info.get("scene", "未知场景")
        keywords = img_info.get("keywords", [])
        people = img_info.get("main_subjects", {})
        pet = img_info.get("pet_details", {})
        category = img_info.get("category", "")
        is_pet = (category == "宠物") or bool(pet)

        trace = r["trace"]
        matched_rules = [f"{rule}: {evidence}" for rule, delta, evidence in trace if delta > 0]
        mismatched_rules = [f"{rule}: {evidence}" for rule, delta, evidence in trace if delta < 0]

        info_text += f"""
【图片 {idx}】路径：{img_path}
场景：{scene}
描述：{desc}
关键词：{', '.join(keywords) if keywords else '无'}
"""
        if is_pet:
            pet_breed = pet.get("breed", "未知")
            pet_color = pet.get("coat_color", [])
            pet_life = pet.get("life_stage", "未知")
            info_text += f"""宠物信息：品种={pet_breed}，毛色={pet_color}，年龄={pet_life}
"""
        else:
            if people:
                cnt = people.get("count", 0)
                cnt_cat = people.get("count_category", "未知")
                eth = people.get("primary_ethnicity", "未知")
                info_text += f"""人物信息：数量={cnt_cat}（实际{cnt}人），人种={eth}
"""

        info_text += f"""匹配的规则：{matched_rules if matched_rules else '无'}
不匹配的规则：{mismatched_rules if mismatched_rules else '无'}
得分：{r['score']}
"""

    prompt = f"""你是一个图像检索系统的可解释性助手。用户查询是："{user_query}"。

系统返回了 Top3 候选图片，请根据每张图片的元数据和匹配规则，必须用中文生成一段约150字的解释。
要求：
1. 首先概括整体检索情况。
2. 对每张图片，详细说明选择图片的原因，以及可能存在的不足（尤其对于排名靠后的图片）。
3. 语气客观、专业，避免重复。
4. 每张图片概述的最少字数不得少于100。
5. **特别注意**：
   - 如果图片包含“宠物信息”，则**严禁**在解释中提及任何人物数量、人种或“一群人”的相关描述，仅围绕宠物特征展开。
   - 本系统中，“一群人”的定义是 **4人及以上**，该规则仅适用于人物照片，不得用于宠物照片。
6. **格式要求（非常重要）**：
   - 禁止使用任何 Markdown 标题符号（如 `#`、`##`、`###`）或粗体（`**`）来标记图片编号。
   - 每张图片的描述必须用普通文本，格式为：`图片X（得分Y）：` 后直接接内容，冒号为中文冒号。
   - 不要添加额外的空行或分割线。
   - 图片描述之间用正常换行分隔即可。

以下是图片信息：
{info_text}

请输出解释文本（不要包含其他无关内容，严格按照格式要求）："""

    owns_client = client is None
    try:
        if owns_client:
            client = get_deepseek_client(api_key, allow_environment=allow_environment)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "解释生成失败：" + safe_api_error(e)
    finally:
        if owns_client and client is not None:
            try:
                client.close()
            except Exception:
                pass
