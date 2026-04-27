import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dashscope
from dashscope import MultiModalConversation

# API 配置区
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
VISION_MODEL = os.getenv("BAILIAN_VL_MODEL", "qwen2.5-vl-32b-instruct")

image_folder = "data"
output_json = "metadata_cache.json"

# 基础地标词典（仅用于真实户外场景）
LANDMARK_TO_CITY_BASE = {
    "成温立交": "成都",
    "chengwen flyover": "成都",
    "雍和宫": "北京",
    "yonghegong": "北京",
    "yonghegong lama temple": "北京",
    "马甸桥": "北京",
    "北太平桥": "北京",
    "十七孔桥": "北京",
    "17孔桥": "北京",
    "颐和园十七孔桥": "北京",
    "summer palace seventeen arch bridge": "北京",
    "九眼桥": "成都",
    "jiuyan bridge": "成都",
}

CITY_ALIASES = {
    "北京": "北京", "beijing": "北京", "北京市": "北京",
    "成都": "成都", "chengdu": "成都", "成都市": "成都",
}

GENERIC_TERMS = {"桥", "立交", "公园", "广场", "博物馆", "大学", "寺", "宫", "塔"}


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def normalize_city(city: str) -> str:
    c = str(city or "").strip()
    if not c:
        return ""
    c_lower = c.lower()
    if c_lower in CITY_ALIASES:
        return CITY_ALIASES[c_lower]
    return c.replace("市", "").replace("特别行政区", "").strip()


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def safe_parse_json(content: str) -> Optional[Dict]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def normalize_list_field(x) -> List[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if x:
        return [str(x).strip()]
    return []


def safe_flatten_list(lst) -> List[str]:
    """递归展平可能嵌套的列表，确保所有元素都是字符串"""
    result = []
    if isinstance(lst, (list, tuple)):
        for item in lst:
            if isinstance(item, (list, tuple)):
                result.extend(safe_flatten_list(item))
            elif isinstance(item, str):
                result.append(item)
            else:
                result.append(str(item))
    elif isinstance(lst, str):
        result.append(lst)
    return result


def is_geography_relevant(metadata: Dict) -> bool:
    img_type = normalize_text(metadata.get("image_type", ""))
    scene = normalize_text(metadata.get("scene", ""))
    blacklist_types = {"screenshot", "ppt", "phone", "computer", "document", "presentation", "slide", "projector"}
    blacklist_scenes = {"class", "lecture", "indoors", "classroom", "presentation", "document", "note", "text"}
    for kw in blacklist_types:
        if kw in img_type:
            return False
    for kw in blacklist_scenes:
        if kw in scene:
            return False
    outdoor_keywords = {
        "outdoor", "sunset", "beach", "mountain", "bridge", "street", "park", "landscape",
        "户外", "日落", "夕阳", "海滩", "山", "桥", "立交", "街道", "公园", "风景", "自然",
        "草地", "草坪", "沙滩", "森林"
    }
    for kw in outdoor_keywords:
        if kw in scene or kw in img_type:
            return True
    if "photo" in img_type and not any(kw in img_type for kw in blacklist_types):
        return True
    return False


def validate_pet_slots(slots: Dict) -> Dict:
    """校验并补全宠物硬槽位，包含年龄字段，且保证宠物count独立"""
    default_pet_slots = {
        "animal_type": "未知",
        "breed": "未知",
        "coat_color": [],
        "coat_pattern": "未知",
        "size": "未知",
        "pose": "未知",
        "expression": "无",
        "action": "无",
        "accessories": [],
        "environment": "未知",
        "interaction": "独处",
        "count": 1,          # 仅统计该宠物物种数量，不含人物
        "occlusion": "无",
        "life_stage": "未知"  # 新增：幼年/成年/老年/未知
    }
    validated = default_pet_slots.copy()
    for key, default_val in default_pet_slots.items():
        if key in slots and slots[key] is not None:
            val = slots[key]
            if isinstance(default_val, list):
                validated[key] = safe_flatten_list(val)
            elif isinstance(default_val, str):
                if isinstance(val, str):
                    validated[key] = val if val else default_val
                else:
                    validated[key] = str(val) if val else default_val
            elif isinstance(default_val, (int, float)):
                if isinstance(val, (int, float)):
                    validated[key] = val
                else:
                    try:
                        validated[key] = int(val)
                    except:
                        validated[key] = default_val
    # 动物类型标准化
    if validated["animal_type"] not in {"猫", "狗", "兔子", "鸟", "鱼", "其他"}:
        validated["animal_type"] = "其他"
    # 年龄标准化
    if validated["life_stage"] not in {"幼年", "成年", "老年", "未知"}:
        validated["life_stage"] = "未知"
    return validated


def call_bailian_vl(image_path: str) -> Optional[Dict]:
    print(f"  正在分析: {image_path}")
    prompt = """
你是一个图像元数据标注器。根据图片内容，先判断**主要主体类别**（category），然后输出对应的结构化 slots，最后仍保留通用字段。

**category 可选值**：
- "人物"：以人类为主体的照片
- "宠物"：以宠物（猫、狗、鸟等）为主体的照片
- "风景"：自然或城市景观，无明显主体
- "其他"：截图、文档、物品等

**输出格式**（严格 JSON，不要添加任何注释）：
{
  "category": "人物｜宠物｜风景｜其他",
  "slots": {
    // 若 category 为 "宠物"，必须完整填写以下字段
    "animal_type": "猫/狗/兔子/鸟/鱼/其他",
    "breed": "品种名或未知",
    "coat_color": ["颜色1", "颜色2"],
    "coat_pattern": "纯色/虎斑/斑点/其他",
    "size": "小型/中型/大型",
    "pose": "卧/坐/站立/奔跑/跳跃",
    "expression": "警觉/放松/欢快/无",
    "action": "叼球/挠痒/追逐/进食/无",
    "accessories": ["项圈", "蝴蝶结"] 或 [],
    "environment": "室内地毯/草地/海滩/街道",
    "interaction": "独处/被抚摸/与人玩耍/多宠互动",
    "count": 1,                     // 只统计画面中该宠物物种的个体数量，不统计人类
    "occlusion": "无/部分遮挡/严重遮挡",
    "life_stage": "幼年/成年/老年/未知"  // 根据体型、面部特征判断
  },
  // 通用字段（所有类别都需要）
  "scene": "场景类型",
  "description": "详细描述（中文）",
  "keywords": ["关键词1", "关键词2"],
  "has_text": false,
  "image_type": "photo/ppt_screenshot/phone_screenshot 等",
  "location": {
    "city": "若画面为真实户外场景且能确定城市则填写，否则留空",
    "landmarks": ["若画面为真实地标建筑则填写，否则留空"]
  },
  "landmark_candidates": [],
  "ocr_text": [],
  "weather": "晴天/多云/阴天/雨天/雪天/日落/夜晚/室内灯光/不确定",
  "main_subjects": {
    "count": 0,
    "count_category": "单人/两人/三人/一群人/无",
    "primary_ethnicity": "亚洲人/黄种人/西方人/白种人/非洲人/黑种人/混合/无",
    "facial_expression": "开心/平静/严肃/悲伤/兴奋/专注/无"
  },
  "background_people": "很多/零星几个/无"
}

**重要要求**：
- 宠物图片的 count 严格只统计该宠物物种的数量，不包含人物或其他物体。例：一人一猫，count=1。
- life_stage 根据体型、面部特征、毛发状态判断：幼年（体型小、圆脸）、成年（体型标准）、老年（毛发发白、神态衰老），不确定填"未知"。
- 若 category 为 "宠物"，slots 中所有字段必须填写，未知项填 "未知" 或 []。
- 仅输出 JSON，不要额外解释。
"""
    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{encode_image_to_base64(image_path)}"},
            {"text": prompt}
        ]
    }]
    try:
        response = MultiModalConversation.call(model=VISION_MODEL, messages=messages)
        if response.status_code != 200:
            print(f"  API 调用失败: {response.code} - {response.message}")
            return None
        content = response.output.choices[0].message.content[0]["text"]
        return safe_parse_json(content)
    except Exception as e:
        print(f"  请求异常: {e}")
        return None


def extract_ocr_landmarks(text: str) -> List[str]:
    if not text:
        return []
    text_lower = text.lower()
    landmarks = []
    for lm in LANDMARK_TO_CITY_BASE:
        if lm.lower() in text_lower:
            landmarks.append(lm)
    if not landmarks and len(text) < 40:
        match = re.search(r"[\u4e00-\u9fa5A-Za-z0-9·]{2,20}(桥|立交|宫|寺|塔|公园|广场|博物馆|大学)", text)
        if match:
            landmarks.append(match.group())
    return landmarks


def resolve_location(metadata: Dict) -> Dict:
    loc = metadata.get("location") if isinstance(metadata.get("location"), dict) else {}
    model_city = normalize_city(loc.get("city", ""))
    model_landmarks = normalize_list_field(loc.get("landmarks", []))
    candidates = metadata.get("landmark_candidates", [])
    ocr_texts = normalize_list_field(metadata.get("ocr_text", []))
    desc = str(metadata.get("description", ""))
    kw = normalize_list_field(metadata.get("keywords", []))

    pool = []
    for lm in model_landmarks:
        pool.append((lm, "model", 1.0))
    for obj in candidates:
        if isinstance(obj, dict):
            name = str(obj.get("name", "")).strip()
            conf = float(obj.get("confidence", 0.5))
            if name:
                pool.append((name, "candidate", conf))
    for text in ocr_texts:
        for phrase in extract_ocr_landmarks(text):
            pool.append((phrase, "ocr", 0.95))

    filtered = {}
    for name, src, conf in pool:
        if not name or name in GENERIC_TERMS:
            continue
        key = normalize_text(name)
        if key not in filtered or conf > filtered[key][2]:
            filtered[key] = (name, src, conf)

    resolved_city = model_city
    hits = []
    if not resolved_city:
        city_votes = {}
        for key, (name, src, conf) in filtered.items():
            city = LANDMARK_TO_CITY_BASE.get(name) or LANDMARK_TO_CITY_BASE.get(key)
            if not city:
                continue
            if src == "ocr" or (src == "candidate" and conf > 0.7) or src == "model":
                city_votes[city] = city_votes.get(city, 0) + conf
                hits.append({"landmark": name, "resolver": "dict", "city": city, "conf": conf})
        if len(city_votes) == 1:
            city, score = list(city_votes.items())[0]
            if score > 0.6:
                resolved_city = city
        elif city_votes:
            hits.append({"warning": "conflicting_cities", "votes": city_votes})

    if not resolved_city:
        full_text = normalize_text(" ".join([desc] + kw + ocr_texts))
        for alias, city in CITY_ALIASES.items():
            if normalize_text(alias) in full_text:
                resolved_city = city
                hits.append({"alias": alias, "resolver": "text_alias", "city": city})
                break

    final_landmarks = sorted({name for name, _, _ in filtered.values()})
    metadata["location"] = {
        "city": normalize_city(resolved_city),
        "landmarks": final_landmarks
    }
    metadata["geo_reasoning"] = {
        "model_city": model_city,
        "model_landmarks": model_landmarks,
        "final_city": metadata["location"]["city"],
        "final_landmarks": final_landmarks,
        "resolver_hits": hits,
    }
    return metadata


def normalize_metadata_schema(metadata: Dict) -> Dict:
    metadata["scene"] = str(metadata.get("scene", "")).strip()
    metadata["description"] = str(metadata.get("description", "")).strip()
    metadata["keywords"] = normalize_list_field(metadata.get("keywords", []))
    metadata["has_text"] = bool(metadata.get("has_text", False))
    metadata["image_type"] = str(metadata.get("image_type", "photo")).strip() or "photo"

    if "weather" not in metadata or not isinstance(metadata["weather"], str):
        metadata["weather"] = "不确定"
    else:
        metadata["weather"] = str(metadata["weather"]).strip()

    if "main_subjects" not in metadata or not isinstance(metadata["main_subjects"], dict):
        metadata["main_subjects"] = {"count": 0, "count_category": "无", "primary_ethnicity": "无", "facial_expression": "无"}
    else:
        p = metadata["main_subjects"]
        p["count"] = p.get("count", 0)
        p["count_category"] = str(p.get("count_category", "无"))
        p["primary_ethnicity"] = str(p.get("primary_ethnicity", "无"))
        p["facial_expression"] = str(p.get("facial_expression", "无"))

    if "background_people" not in metadata:
        metadata["background_people"] = "无"
    else:
        metadata["background_people"] = str(metadata["background_people"]).strip()

    category = metadata.get("category", "其他")
    metadata["category"] = category
    slots = metadata.get("slots", {})
    if category == "宠物":
        validated_slots = validate_pet_slots(slots)
        metadata["pet_details"] = validated_slots

        # 安全提取属性
        animal_type = validated_slots.get("animal_type", "")
        breed = validated_slots.get("breed", "")
        action = validated_slots.get("action", "")
        environment = validated_slots.get("environment", "")
        pose = validated_slots.get("pose", "")
        life_stage = validated_slots.get("life_stage", "未知")
        coat_color_list = validated_slots.get("coat_color", [])

        # 构建额外关键词
        extra_kw = []
        for attr_str in [animal_type, breed, action, environment, pose, life_stage]:
            if isinstance(attr_str, str) and attr_str and attr_str != "未知":
                extra_kw.append(attr_str)
        for color in coat_color_list:
            if isinstance(color, str) and color and color != "未知":
                extra_kw.append(color)

        # 安全合并关键词
        combined = metadata["keywords"] + extra_kw
        seen = set()
        new_keywords = []
        for k in combined:
            if isinstance(k, str) and k not in seen:
                seen.add(k)
                new_keywords.append(k)
        metadata["keywords"] = new_keywords

        # 补充描述
        coat_str = ", ".join(coat_color_list) if coat_color_list else ""
        pet_desc = f"【宠物】品种：{breed}，毛色：{coat_str}，姿态：{pose}，动作：{action}，年龄：{life_stage}"
        metadata["description"] = metadata["description"] + " " + pet_desc

    metadata["location"] = metadata.get("location", {"city": "", "landmarks": []})
    return metadata


def analyze_image(image_path: str) -> Optional[Dict]:
    parsed = call_bailian_vl(image_path)
    if not parsed:
        return None
    parsed = normalize_metadata_schema(parsed)

    if parsed.get("category") == "宠物":
        print(f"    [宠物图片] 保留环境描述，跳过地标解析")
        parsed["location"] = {"city": "", "landmarks": []}
        parsed["geo_reasoning"] = {"note": "pet_image_skip_geo"}
    elif is_geography_relevant(parsed):
        print(f"    [地理图片] 进行地标解析")
        parsed = resolve_location(parsed)
    else:
        print(f"    [非地理图片] 跳过地标解析")
        parsed["location"] = {"city": "", "landmarks": []}
        parsed["geo_reasoning"] = {"note": "non_geographic_image"}
    return parsed


def list_image_files(folder: str) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
    files = []
    for ext in exts:
        files.extend(Path(folder).rglob(ext))
    return files


def should_reanalyze(rel_path: str, existing: Dict, force_all: bool, force_paths: Set[str]) -> bool:
    if force_all or rel_path in force_paths:
        return True
    old = existing.get(rel_path)
    if not old:
        return True
    if "weather" not in old or "main_subjects" not in old:
        return True
    if old.get("geo_reasoning", {}).get("used_amap") is True:
        return True
    loc = old.get("location")
    if not isinstance(loc, dict):
        return True
    img_type = old.get("image_type", "")
    if ("screenshot" in img_type or "ppt" in img_type or "phone" in img_type) and (loc.get("city") or loc.get("landmarks")):
        return True
    if "data/pet" in rel_path and ("pet_details" not in old or "life_stage" not in old.get("pet_details", {})):
        return True
    return False


def main():
    if not dashscope.api_key:
        raise RuntimeError("请设置 DASHSCOPE_API_KEY 环境变量")

    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            all_meta = json.load(f)
    else:
        all_meta = {}

    force_all = os.getenv("FORCE_REANALYZE", "0") == "1"
    force_paths = {x.strip().replace("\\", "/") for x in os.getenv("FORCE_PATHS", "").split(",") if x.strip()}

    images = list_image_files(image_folder)
    print(f"共 {len(images)} 张图片，模型: {VISION_MODEL}")

    for img in images:
        rel = str(img).replace("\\", "/")
        if not should_reanalyze(rel, all_meta, force_all, force_paths):
            print(f"  跳过: {rel}")
            continue

        meta = analyze_image(str(img))
        if not meta:
            print(f"  失败: {rel}")
            continue

        all_meta[rel] = meta
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, ensure_ascii=False, indent=2)

        city = meta.get("location", {}).get("city", "")
        people_count = meta.get("main_subjects", {}).get("count", 0)
        weather = meta.get("weather", "不确定")
        pet_info = ""
        if meta.get("category") == "宠物":
            pd = meta["pet_details"]
            pet_info = f"动物: {pd.get('animal_type','')}，品种: {pd.get('breed','')}，年龄: {pd.get('life_stage','')}"
        print(f"  已保存: {rel} | city={city or '(空)'} | 中心人数={people_count} | 天气={weather} {pet_info}")

    print(f"\n完成，保存至 {output_json}")


if __name__ == "__main__":
    main()