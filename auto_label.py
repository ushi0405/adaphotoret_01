import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dashscope
from dashscope import MultiModalConversation
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
from geopy.geocoders import Nominatim
import requests
import time

register_heif_opener()  # 让 Pillow 支持 HEIC 格式

# API 配置区
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
VISION_MODEL = os.getenv("BAILIAN_VL_MODEL", "qwen2.5-vl-32b-instruct")
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "")  # 高德地图 Web API Key，不设置则不启用

image_folder = "data"
output_json = "metadata_cache.json"


# ────────── 辅助函数 ──────────
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


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


# ────────── 宠物硬槽位验证 ──────────
def validate_pet_slots(slots: Dict) -> Dict:
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
        "count": 1,
        "occlusion": "无",
        "life_stage": "未知"
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
    if validated["life_stage"] not in {"幼年", "成年", "老年", "未知"}:
        validated["life_stage"] = "未知"
    return validated


# ────────── 视觉模型调用 ──────────
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
    "animal_type": "请根据图片实际内容填写，可以是任意动物名称（如猫、狗、兔子、鸟、鱼、梅花鹿、羊等），不确定时填“未知”",
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
- animal_type 请根据实际动物自由填写，不要拘泥于猫狗；若无法辨认，填"未知"。
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


# ────────── 元数据规范化 ──────────
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
        original_count = p.get("count", 0)
        if not isinstance(original_count, int):
            try:
                original_count = int(original_count)
            except:
                original_count = 0
        p["count"] = original_count

        # 标准化 count_category
        if original_count == 1:
            p["count_category"] = "单人"
        elif original_count == 2:
            p["count_category"] = "两人"
        elif original_count == 3:
            p["count_category"] = "三人"
        elif original_count >= 4:
            p["count_category"] = "一群人"
        else:
            p["count_category"] = "无"

        p["primary_ethnicity"] = str(p.get("primary_ethnicity", "无"))
        p["facial_expression"] = str(p.get("facial_expression", "无"))

    if "background_people" not in metadata:
        metadata["background_people"] = "无"
    else:
        metadata["background_people"] = str(metadata["background_people"]).strip()

    # 确保 category 存在
    category = metadata.get("category", "")
    if not category or category.lower() in ["", "其他", "未知"]:
        if "pet_details" in metadata and metadata["pet_details"].get("animal_type", "未知") != "未知":
            category = "宠物"
        else:
            main = metadata.get("main_subjects", {})
            if main.get("count", 0) > 0 and main.get("count_category") not in ["无", "单个物体", "单个对象", "单个动物"]:
                category = "人物"
            else:
                category = "风景"
    metadata["category"] = category

    # 处理宠物硬槽位
    slots = metadata.get("slots", {})
    if category == "宠物":
        validated_slots = validate_pet_slots(slots)
        metadata["pet_details"] = validated_slots

        animal_type = validated_slots.get("animal_type", "")
        breed = validated_slots.get("breed", "")
        action = validated_slots.get("action", "")
        environment = validated_slots.get("environment", "")
        pose = validated_slots.get("pose", "")
        life_stage = validated_slots.get("life_stage", "未知")
        coat_color_list = validated_slots.get("coat_color", [])

        extra_kw = []
        for attr_str in [animal_type, breed, action, environment, pose, life_stage]:
            if isinstance(attr_str, str) and attr_str and attr_str != "未知":
                extra_kw.append(attr_str)
        for color in coat_color_list:
            if isinstance(color, str) and color and color != "未知":
                extra_kw.append(color)

        combined = metadata["keywords"] + extra_kw
        seen = set()
        new_keywords = []
        for k in combined:
            if isinstance(k, str) and k not in seen:
                seen.add(k)
                new_keywords.append(k)
        metadata["keywords"] = new_keywords

        coat_str = ", ".join(coat_color_list) if coat_color_list else ""
        pet_desc = f"【宠物】品种：{breed}，毛色：{coat_str}，姿态：{pose}，动作：{action}，年龄：{life_stage}"
        metadata["description"] = metadata["description"] + " " + pet_desc

    metadata["location"] = metadata.get("location", {"city": "", "landmarks": []})
    return metadata


# ────────── EXIF GPS 提取（兼容 HEIC）──────────
def get_gps_from_image(image_path: str):
    """从图片（包括 HEIC）中提取 GPS 坐标，返回 (纬度, 经度) 或 None"""
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            return None
        gps_ifd = exif.get_ifd(34853)
        if not gps_ifd:
            return None

        gps_info = {}
        for tag, value in gps_ifd.items():
            tag_name = GPSTAGS.get(tag, str(tag))
            gps_info[tag_name] = value

        if not gps_info:
            return None

        def parse_gps_value(value, ref):
            if value is None or ref is None:
                return None
            degrees, minutes, seconds = value
            decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
            if ref in ('S', 'W'):
                decimal = -decimal
            return decimal

        latitude = parse_gps_value(
            gps_info.get("GPSLatitude"),
            gps_info.get("GPSLatitudeRef")
        )
        longitude = parse_gps_value(
            gps_info.get("GPSLongitude"),
            gps_info.get("GPSLongitudeRef")
        )
        if latitude is not None and longitude is not None:
            return (latitude, longitude)
    except Exception as e:
        print(f"  读取 GPS 信息失败: {e}")
    return None


# ────────── 逆地理编码（高德优先，回退 Nominatim）──────────
def get_address_from_amap(lat, lon):
    """使用高德地图 Web API 将经纬度转换为地址"""
    if not AMAP_API_KEY:
        return None
    url = f"https://restapi.amap.com/v3/geocode/regeo?location={lon},{lat}&key={AMAP_API_KEY}"
    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data["status"] == "1":
            return data["regeocode"]["formatted_address"]
    except Exception as e:
        print(f"  高德逆地理编码失败: {e}")
    return None


def get_address_from_nominatim(lat, lon):
    """免费的 Nominatim 逆地理编码"""
    try:
        geolocator = Nominatim(user_agent="adaphotoret")
        location = geolocator.reverse((lat, lon), language="zh-CN")
        if location:
            return location.address
    except Exception as e:
        print(f"  Nominatim 逆地理编码失败: {e}")
    return None


def extract_city_from_address(address: str) -> str:
    """从地址中提取城市名，如'四川省成都市郫都区' -> '成都'"""
    if "市" in address:
        idx = address.index("市")
        start = address.rfind("省", 0, idx)
        if start != -1:
            return address[start+1:idx+1]
        else:
            parts = address[:idx+1].split(" ")
            return parts[-1] if parts else address[:idx+1]
    return ""


# ────────── 图片分析 ──────────
def analyze_image(image_path: str) -> Optional[Dict]:
    parsed = call_bailian_vl(image_path)
    if not parsed:
        return None
    parsed = normalize_metadata_schema(parsed)

    # 自动读取 GPS 信息并逆地理编码
    gps_coords = get_gps_from_image(image_path)
    if gps_coords:
        lat, lon = gps_coords
        parsed.setdefault("location", {})
        parsed["location"]["gps_latitude"] = lat
        parsed["location"]["gps_longitude"] = lon

        # 优先使用高德，没有 Key 则回退到 Nominatim
        address = get_address_from_amap(lat, lon) if AMAP_API_KEY else get_address_from_nominatim(lat, lon)
        if address:
            parsed["location"]["gps_address"] = address
            # 如果视觉模型没识别出城市，从地址中提取
            if not parsed.get("location", {}).get("city"):
                city = extract_city_from_address(address)
                if city:
                    parsed["location"]["city"] = city

    parsed["geo_reasoning"] = {
        "source": "exif" if gps_coords else "none"
    }
    return parsed


# ────────── 文件扫描 ──────────
def list_image_files(folder: str) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.heic"]
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
    if "gps_address" not in old.get("location", {}) and rel_path.lower().endswith(('.jpg', '.jpeg', '.heic')):
        return True
    return False


# ────────── 主流程 ──────────
def main():
    if not dashscope.api_key:
        raise RuntimeError("请设置 DASHSCOPE_API_KEY 环境变量")
    if AMAP_API_KEY:
        print("检测到高德地图 API Key，将使用高德逆地理编码。")
    else:
        print("未设置 AMAP_API_KEY，将使用免费 Nominatim 逆地理编码。")

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
        gps_addr = meta.get("location", {}).get("gps_address", "")
        people_count = meta.get("main_subjects", {}).get("count", 0)
        pet_info = ""
        if meta.get("category") == "宠物":
            pd = meta.get("pet_details", {})
            pet_info = f"动物: {pd.get('animal_type','')}，品种: {pd.get('breed','')}，年龄: {pd.get('life_stage','')}"
        print(f"  已保存: {rel} | city={city or '(空)'} | GPS地址={gps_addr[:30] if gps_addr else '无'} | 中心人数={people_count} | {pet_info}")

    print(f"\n完成，保存至 {output_json}")


if __name__ == "__main__":
    main()