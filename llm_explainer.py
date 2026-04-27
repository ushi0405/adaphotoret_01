
import os
from typing import Dict, List
from openai import OpenAI

def get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",  
    )


def generate_explanation(
    user_query: str,
    top_results: List[Dict],
    metadata: Dict,
) -> str:
    """
    使用 DeepSeek 为 Top3 检索结果生成可解释的推理文本。

    Args:
        user_query: 用户原始查询
        top_results: 包含图片路径、得分、推理链的列表
        metadata: 图片元数据字典

    Returns:
        一段自然语言解释，说明每张图片的匹配/不匹配原因
    """
    if not top_results:
        return "无结果，无法生成解释。"

    # 构建提示词所需的信息
    info_text = ""
    for idx, r in enumerate(top_results, start=1):
        img_path = r["img_path"]
        img_info = metadata.get(img_path, {})
        desc = img_info.get("description", "无描述")
        scene = img_info.get("scene", "未知场景")
        keywords = img_info.get("keywords", [])
        people = img_info.get("main_subjects", {})
        trace = r["trace"]

        # 简要归纳推理链中的关键匹配项
        matched_rules = [f"{rule}: {evidence}" for rule, delta, evidence in trace if delta > 0]
        mismatched_rules = [f"{rule}: {evidence}" for rule, delta, evidence in trace if delta < 0]

        info_text += f"""
【图片 {idx}】路径：{img_path}
场景：{scene}
描述：{desc}
关键词：{', '.join(keywords) if keywords else '无'}
人物信息：数量={people.get('count_category', '未知')}，人种={people.get('primary_ethnicity', '未知')}
匹配的规则：{matched_rules if matched_rules else '无'}
不匹配的规则：{mismatched_rules if mismatched_rules else '无'}
得分：{r['score']}
"""

    prompt = f"""你是一个图像检索系统的可解释性助手。用户查询是："{user_query}"。

系统返回了 Top3 候选图片，请根据每张图片的元数据和匹配规则，必须用中文生成一段约200字的解释。
要求：
1. 首先概括整体检索情况。
2. 对每张图片，详细说明选择图片的原因，以及可能存在的不足（尤其对于排名靠后的图片）。
3. 语气客观、专业，避免重复。
4.每张图片概述的最少字数不得少于150.

以下是图片信息：
{info_text}

请输出解释文本（不要包含其他无关内容）："""

    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或 deepseek-reasoner 若需要推理模式
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"解释生成失败：{e}"