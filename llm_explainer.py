import os
from openai import OpenAI

def get_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def build_prompt(user_query, top_results, metadata):
    prompt = f"""你是一个图像检索系统的可解释性助手。用户查询是："{user_query}"。

系统返回了 Top3 候选图片，请按以下流程输出：

## 第一部分：思考过程
请用一段文字描述你是如何解析用户查询的，包括：
- 提取了哪些关键实体（如地点、菜品、宠物品种等）
- 识别了哪些意图（如地理约束、人物数量、宠物特征等）
- 整体检索策略（例如优先匹配地理信息，其次语义相似等）

## 第二部分：图片解释
根据每张图片的元数据和匹配规则，必须用中文生成解释。先概括整体检索情况，然后逐张图片说明。

### 格式要求：
- 禁止使用任何 Markdown 标题符号（如 #、##、###）或粗体（**）来标记图片编号。
- 每张图片的描述格式必须为：`图片X（得分Y）：` 后直接接内容，冒号为中文冒号。
- 不要添加额外的空行或分割线。
- 每张图片概述的最少字数不得少于100字。
- 解释必须客观、专业，避免重复。

### 内容要求：
1. 首先概括整体检索情况（例如共检索到多少相关图片，主要匹配依据）。
2. 对每张图片，详细说明：
   - 选择该图片的原因（命中哪些规则：语义相似、地理匹配、实体关键词、宠物属性等）
   - 可能存在的不足（尤其是排名靠后的图片，如未命中地理信息、语义偏差等）
3. **特别注意**：
   - 如果图片包含“宠物信息”，则**严禁**在解释中提及任何人物数量、人种或“一群人”的相关描述，仅围绕宠物特征展开。
   - 本系统中，“一群人”的定义是 4人及以上，该规则仅适用于人物照片。

### 检索结果数据：
"""
    for idx, r in enumerate(top_results, 1):
        img_path = r['img_path']
        score = r['score']
        info = metadata.get(img_path, {})
        desc = info.get('description', '无描述')
        category = info.get('category', '未知')
        trace_str = " | ".join([f"{rule}:{delta}" for rule, delta, _ in r['trace'] if abs(delta) > 0.01])
        prompt += f"{idx}. 图片：{img_path}\n   得分：{score}\n   类别：{category}\n   描述：{desc}\n   匹配线索：{trace_str}\n"
    prompt += "\n请严格按照上述格式要求输出，先输出【思考过程】，然后输出图片解释。"
    return prompt

def generate_explanation(user_query, top_results, metadata):
    client = get_deepseek_client()
    prompt = build_prompt(user_query, top_results, metadata)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            stream=False,
        )
        explanation = response.choices[0].message.content
        return explanation.encode('utf-8').decode('utf-8')
    except Exception as e:
        return f"解释生成失败：{str(e)}"