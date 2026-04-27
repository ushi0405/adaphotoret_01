
import json
import math
import os
from typing import Dict, List, Set, Tuple

import faiss
import gradio as gr
import jieba
import jieba.posseg as pseg

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

from attributes import (
    extract_person_attributes, expand_person_attributes,
    extract_pet_attributes, expand_pet_attributes,
    PET_COLORS, PET_ANIMAL_TYPES,
    BREED_ALIASES, BREED_NORMALIZATION
)
from llm_explainer import generate_explanation

print("正在加载语义模型...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("语义模型加载完成。")

# --- 基础地标知识 ---
LANDMARK_TO_CITY_BASE = {
    "成温立交": "成都", "chengwen flyover": "成都",
    "雍和宫": "北京", "yonghegong": "北京", "yonghegong lama temple": "北京",
    "马甸桥": "北京", "北太平桥": "北京", "十七孔桥": "北京", "17孔桥": "北京",
    "颐和园十七孔桥": "北京",
    "九眼桥": "成都", "jiuyan bridge": "成都",
}

CITY_TO_LANDMARKS_BASE = {
    "成都": {"成温立交", "chengwen flyover", "九眼桥", "jiuyan bridge"},
    "北京": {"雍和宫", "yonghegong", "yonghegong lama temple",
             "马甸桥", "北太平桥", "十七孔桥", "17孔桥", "颐和园十七孔桥"},
}

LANDMARK_TO_CITY_RUNTIME: Dict[str, str] = {}
CITY_TO_LANDMARKS_RUNTIME: Dict[str, Set[str]] = {}

GRASS_TERMS = {"草地", "草坪", "草场", "grass", "lawn", "field"}
TRACK_TERMS = {"跑道", "赛道", "track", "runway"}
RUN_TERMS = {"跑", "奔跑", "跑步", "run", "running"}
TWO_PEOPLE_TERMS = {"两个人", "两人", "2个人", "2人", "two people", "two persons"}
BRIDGE_TERMS = {"桥", "拱桥", "bridge", "arch bridge", "flyover"}
ACADEMIC_WRITING_TERMS = {"学术英语", "academic english", "academic writing", "学术写作"}
EXAMPLE_TERMS = {"示例", "例文", "范文", "sample", "example", "example essay", "示例论文"}
STRUCTURE_TERMS = {"结构", "框架", "essay title", "thesis statement",
                   "topic sentence", "reference list", "main body"}


def normalize_text(text: str) -> str:
    if not text: return ""
    return str(text).lower().strip()

def normalize_city(city: str) -> str:
    c = str(city or "").strip()
    if not c: return ""
    return c.replace("市", "").strip()

def merge_geo_maps() -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    l2c = dict(LANDMARK_TO_CITY_BASE)
    for k, v in LANDMARK_TO_CITY_RUNTIME.items():
        l2c[k] = v
    c2l: Dict[str, Set[str]] = {k: set(v) for k, v in CITY_TO_LANDMARKS_BASE.items()}
    for c, landmarks in CITY_TO_LANDMARKS_RUNTIME.items():
        c2l.setdefault(c, set()).update(landmarks)
    return l2c, c2l

def extract_geo_from_text(raw_text, l2c, c2l):
    text = normalize_text(raw_text)
    cities, landmarks = set(), set()
    for lm in sorted(l2c.keys(), key=lambda x: len(str(x)), reverse=True):
        if normalize_text(lm) in text:
            landmarks.add(lm)
            cities.add(l2c[lm])
    if "北京" in text or "beijing" in text: cities.add("北京")
    if "成都" in text or "chengdu" in text: cities.add("成都")
    for city in list(cities):
        for lm in c2l.get(city, set()):
            if normalize_text(lm) in text: landmarks.add(lm)
    return {"cities": cities, "landmarks": landmarks}

def match_terms(text: str, terms: Set[str]) -> bool:
    return any(term in normalize_text(text) for term in terms)

def parse_location_from_info(info: Dict):
    cities, landmarks = set(), set()
    loc = info.get("location")
    if isinstance(loc, dict):
        city = normalize_city(loc.get("city", ""))
        if city: cities.add(city)
        lm = loc.get("landmarks", [])
        if isinstance(lm, list):
            landmarks.update(str(x).strip() for x in lm if str(x).strip())
        elif lm: landmarks.add(str(lm).strip())
    return cities, landmarks

def flatten_info_text(info: Dict) -> str:
    scene = str(info.get("scene", ""))
    desc = str(info.get("description", ""))
    image_type = str(info.get("image_type", ""))
    keywords = info.get("keywords", [])
    kw_text = " ".join(str(k) for k in keywords) if isinstance(keywords, list) else str(keywords)
    loc = info.get("location", {}) if isinstance(info.get("location"), dict) else {}
    loc_text = f"{loc.get('city', '')} {' '.join(loc.get('landmarks', []) if isinstance(loc.get('landmarks', []), list) else [])}"
    return f"{scene} {image_type} {desc} {kw_text} {loc_text}".strip()

def enrich_metadata(metadata_dict, l2c, c2l):
    enriched = {}
    for path, info in metadata_dict.items():
        cloned = dict(info)
        flat = flatten_info_text(cloned)
        loc_cities, loc_landmarks = parse_location_from_info(cloned)
        geo = extract_geo_from_text(flat, l2c, c2l)
        cities = set(loc_cities).union(geo["cities"])
        landmarks = set(loc_landmarks).union(geo["landmarks"])
        cloned["_search_text"] = flat
        cloned["_cities"] = sorted(cities)
        cloned["_landmarks"] = sorted(landmarks)
        enriched[path] = cloned
    return enriched

def build_runtime_geo_maps(metadata_dict):
    LANDMARK_TO_CITY_RUNTIME.clear()
    CITY_TO_LANDMARKS_RUNTIME.clear()
    for info in metadata_dict.values():
        loc = info.get("location") if isinstance(info.get("location"), dict) else {}
        city = normalize_city(loc.get("city", ""))
        lms = loc.get("landmarks", []) if isinstance(loc.get("landmarks", []), list) else []
        cleaned = [str(x).strip() for x in lms if str(x).strip()]
        if city and cleaned:
            CITY_TO_LANDMARKS_RUNTIME.setdefault(city, set()).update(cleaned)
            for lm in cleaned:
                LANDMARK_TO_CITY_RUNTIME[lm] = city

def extract_entities_by_pos(text: str) -> List[str]:
    stop_words = {'的','了','是','在','和','与','或','一个','两个','那种','这个','那个','中','里','有','被','把','照片','图片','图像'}
    words = pseg.cut(text)
    entities = []
    for word, flag in words:
        if word in stop_words: continue
        if flag[0] in ('n','v','ns','nt','t','a','f','s','i','j','l'):
            if len(word)==1 and flag[0] not in ('n','ns','nt','v'): continue
            entities.append(word)
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e); unique.append(e)
    return unique

def parse_query(user_query: str, l2c, c2l) -> Dict:
    # 品种短语归一化
    processed = user_query
    for key, normalized in BREED_NORMALIZATION.items():
        processed = processed.replace(key, normalized)

    q = normalize_text(processed)
    geo = extract_geo_from_text(q, l2c, c2l)
    entities = extract_entities_by_pos(processed)
    person_attr = extract_person_attributes(processed)
    person_expand = expand_person_attributes(person_attr)
    pet_attr = extract_pet_attributes(processed)
    pet_expand = expand_pet_attributes(pet_attr)

    expanded_geo_terms = []
    for city in geo["cities"]:
        expanded_geo_terms.append(city)
        expanded_geo_terms.extend(c2l.get(city, set()))
    for lm in geo["landmarks"]:
        expanded_geo_terms.append(lm)
        city = l2c.get(lm)
        if city: expanded_geo_terms.append(city)

    terms = {
        "needs_grass": any(t in q for t in GRASS_TERMS),
        "needs_track": any(t in q for t in TRACK_TERMS),
        "needs_running": any(t in q for t in RUN_TERMS),
        "needs_two_people": any(t in q for t in TWO_PEOPLE_TERMS),
        "needs_bridge": any(t in q for t in BRIDGE_TERMS),
        "needs_academic_writing": any(t in q for t in ACADEMIC_WRITING_TERMS),
        "needs_example": any(t in q for t in EXAMPLE_TERMS),
        "needs_structure": any(t in q for t in STRUCTURE_TERMS),
        "cities": set(geo["cities"]),
        "landmarks": set(geo["landmarks"]),
        "person_count": person_attr["count"],
        "person_gender": person_attr["gender"],
        "person_ethnicity": person_attr["ethnicity"],
        "extracted_entities": entities,
        # ---------- 关键修复 ----------
        # 原来仅依赖 PET_ANIMAL_TYPES，现在如果提取到了品种或动物类型，也当作宠物查询
        "needs_pet": any(t in user_query for t in PET_ANIMAL_TYPES) or len(pet_attr["animal_type"]) > 0 or len(pet_attr["breed"]) > 0,
        "pet_animal_type": pet_attr["animal_type"],
        "pet_breed": pet_attr["breed"],
        "pet_coat_color": pet_attr["coat_color"],
        "pet_action": pet_attr["action"],
        "pet_environment": pet_attr["environment"],
        "pet_count": pet_attr["count"],
        "pet_life_stage": pet_attr["life_stage"],
    }

    entity_str = " ".join(entities)
    geo_str = " ".join(sorted(set(expanded_geo_terms)))
    person_str = " ".join(person_expand)
    pet_str = " ".join(pet_expand)
    expanded_query = f"{processed} {entity_str} {geo_str} {person_str} {pet_str}".strip()

    decomposition = {
        "query": user_query,
        "extracted_entities": entities,
        "person_attributes": person_attr,
        "pet_attributes": pet_attr,
        "expanded_query": expanded_query,
        "semantic_slots": {
            "scene_constraints": [
                x for x, ok in [("grass", terms["needs_grass"]),
                                ("track", terms["needs_track"]),
                                ("bridge", terms["needs_bridge"])] if ok
            ],
            "action_constraints": [x for x, ok in [("running", terms["needs_running"])] if ok],
            "entity_constraints": [x for x, ok in [("two_people", terms["needs_two_people"])] if ok],
            "domain_constraints": [
                x for x, ok in [("academic_writing", terms["needs_academic_writing"]),
                                ("example", terms["needs_example"]),
                                ("structure", terms["needs_structure"])] if ok
            ],
            "geo_constraints": {
                "cities": sorted(list(terms["cities"])),
                "landmarks": sorted(list(terms["landmarks"])),
            },
            "pet_constraints": {
                "animal_type": terms["pet_animal_type"],
                "breed": terms["pet_breed"],
                "coat_color": terms["pet_coat_color"],
                "action": terms["pet_action"],
                "environment": terms["pet_environment"],
                "count": terms["pet_count"],
                "life_stage": terms["pet_life_stage"],
            },
        },
    }
    return {"expanded_query": expanded_query, "terms": terms, "decomposition": decomposition}


# ---------- 加载元数据 ----------
with open("metadata_cache.json", "r", encoding="utf-8") as f:
    metadata_raw = json.load(f)

build_runtime_geo_maps(metadata_raw)
L2C_ALL, C2L_ALL = merge_geo_maps()
metadata = enrich_metadata(metadata_raw, L2C_ALL, C2L_ALL)


def build_vector_index(metadata_dict):
    paths, texts = [], []
    for path, info in metadata_dict.items():
        paths.append(path)
        geo_suffix = " ".join(info.get("_cities", []) + info.get("_landmarks", []))
        texts.append(f"{info.get('_search_text', '')} {geo_suffix}".strip())
    print("正在为照片库生成向量...")
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"向量库构建完成，共 {index.ntotal} 张照片。")
    return paths, index

image_paths, vector_index = build_vector_index(metadata)


def rerank_score(query_terms: Dict, img_info: Dict, base_sim: float):
    score = base_sim * 0.65      # 语义权重降为 0.65
    trace = [("base_semantic_similarity", round(base_sim, 4), "向量语义相似")]

    doc_text = normalize_text(img_info.get("_search_text", ""))
    doc_cities = set(img_info.get("_cities", []))
    doc_landmarks = set(img_info.get("_landmarks", []))
    image_type = normalize_text(img_info.get("image_type", ""))

    # 场景约束
    if query_terms.get("needs_grass"):
        if match_terms(doc_text, GRASS_TERMS):
            score += 0.12; trace.append(("scene_grass_match", 0.12, "命中草地/草坪"))
        elif match_terms(doc_text, TRACK_TERMS):
            score -= 0.20; trace.append(("scene_grass_track_conflict", -0.20, "草地约束与跑道冲突"))
    if query_terms.get("needs_track") and match_terms(doc_text, TRACK_TERMS):
        score += 0.10; trace.append(("scene_track_match", 0.10, "命中跑道/赛道"))
    if query_terms.get("needs_running") and match_terms(doc_text, RUN_TERMS):
        score += 0.06; trace.append(("action_running_match", 0.06, "命中奔跑动作"))
    if query_terms.get("needs_two_people") and ("两名" in doc_text or "两人" in doc_text or "two" in doc_text):
        score += 0.06; trace.append(("entity_two_people_match", 0.06, "命中两人约束"))
    if query_terms.get("needs_bridge") and match_terms(doc_text, BRIDGE_TERMS):
        score += 0.08; trace.append(("scene_bridge_match", 0.08, "命中桥梁场景"))

    # 地理匹配
    if query_terms.get("cities"):
        overlap_city = query_terms["cities"].intersection(doc_cities)
        if overlap_city:
            score += 0.25; trace.append(("geo_city_match", 0.25, f"命中城市:{','.join(sorted(overlap_city))}"))
        else:
            related_lms = set()
            for c in query_terms["cities"]: related_lms.update(C2L_ALL.get(c, set()))
            if related_lms.intersection(doc_landmarks):
                score += 0.15; trace.append(("geo_city_landmark_bridge", 0.15, "命中城市关联地标"))
    if query_terms.get("landmarks"):
        overlap_lm = query_terms["landmarks"].intersection(doc_landmarks)
        if overlap_lm:
            score += 0.30; trace.append(("geo_landmark_match", 0.30, f"命中地标:{','.join(sorted(overlap_lm))}"))
        else:
            related_city = {L2C_ALL.get(lm) for lm in query_terms["landmarks"]}
            related_city.discard(None)
            if related_city.intersection(doc_cities):
                score += 0.12; trace.append(("geo_landmark_city_backoff", 0.12, "命中地标对应城市"))
            elif query_terms.get("needs_bridge") and match_terms(doc_text, BRIDGE_TERMS):
                score += 0.06; trace.append(("geo_landmark_visual_backoff", 0.06, "地标缺失，桥梁视觉线索兜底"))

    if query_terms.get("needs_academic_writing") and match_terms(doc_text, ACADEMIC_WRITING_TERMS):
        score += 0.08; trace.append(("domain_academic_writing_match", 0.08, "命中学术写作域"))
    if query_terms.get("needs_example"):
        if match_terms(doc_text, EXAMPLE_TERMS):
            score += 0.15; trace.append(("intent_example_match", 0.15, "命中示例/范文意图"))
        if image_type == "phone_screenshot":
            score += 0.04; trace.append(("type_phone_screenshot_prior", 0.04, "示例更常见于截图"))
        if match_terms(doc_text, STRUCTURE_TERMS) and image_type == "ppt_screenshot":
            score -= 0.10; trace.append(("intent_example_vs_structure_penalty", -0.10, "结构讲解页，非示例范文"))
    if query_terms.get("needs_structure") and match_terms(doc_text, STRUCTURE_TERMS):
        score += 0.08; trace.append(("intent_structure_match", 0.08, "命中结构讲解意图"))

    # ---------- 人物属性 (仅当不是宠物查询时) ----------
    if not query_terms.get("needs_pet"):
        people = img_info.get("main_subjects", {})
        if people:
            cnt_str = people.get("count_category", "")
            ethnicity = people.get("primary_ethnicity", "")
            cnt = 0
            if cnt_str in ["单人", "单个物体"]: cnt = 1
            elif cnt_str == "两人": cnt = 2
            elif cnt_str in ["三人", "四人"]: cnt = 3 if cnt_str=="三人" else 4
            elif cnt_str in ["五人", "五人及以上", "一群人"]: cnt = 10

            person_count_q = query_terms.get("person_count", [])
            if person_count_q:
                qc = person_count_q[0]
                matched = False
                if qc == "一个" and cnt == 1: score += 0.12; matched = True
                elif qc == "两个" and cnt == 2: score += 0.12; matched = True
                elif qc == "三个" and cnt == 3: score += 0.12; matched = True
                elif qc == "一群" and cnt >= 5: score += 0.12; matched = True
                if not matched:
                    penalty = -0.10
                    if qc == "一个" and cnt >= 5: penalty = -0.20
                    elif qc == "两个" and cnt >= 5: penalty = -0.15
                    elif qc == "一群" and cnt <= 2: penalty = -0.15
                    score += penalty; trace.append(("person_count_mismatch", penalty, "人物数量不符"))
                else: trace.append(("person_count_match", 0.12, f"命中{qc}约束"))

            ethnicity_q = query_terms.get("person_ethnicity", [])
            if ethnicity_q:
                qe = ethnicity_q[0]
                matched = False
                if ("西方/白种人" in qe and "白种人" in ethnicity) or \
                   ("亚洲/黄种人" in qe and "黄种人" in ethnicity) or \
                   ("非洲/黑种人" in qe and "黑种人" in ethnicity):
                    score += 0.06; trace.append(("person_ethnicity_match", 0.06, f"命中{qe}约束")); matched = True
                elif "外国" in qe and ("白种人" in ethnicity or "黑种人" in ethnicity):
                    score += 0.04; trace.append(("person_ethnicity_match", 0.04, "命中外国约束")); matched = True
                if not matched:
                    score -= 0.06; trace.append(("person_ethnicity_mismatch", -0.06, "人种不符"))

    # ---------- 宠物匹配 ----------
    if query_terms.get("needs_pet") and "pet_details" in img_info:
        pet = img_info["pet_details"]

        q_animal = query_terms.get("pet_animal_type", [])
        if q_animal:
            img_animal = pet.get("animal_type", "")
            if any(a in img_animal for a in q_animal) or any(a in pet.get("breed", "") for a in q_animal):
                score += 0.16
                trace.append(("pet_animal_match", 0.16, f"动物类型匹配:{','.join(q_animal)}"))
            else:
                score -= 0.25
                trace.append(("pet_animal_mismatch", -0.25, f"动物类型不符(查询:{q_animal[0]}, 图片:{img_animal})"))

        q_breed = query_terms.get("pet_breed", [])
        if q_breed:
            img_breed = pet.get("breed", "")
            matched = False
            for b in q_breed:
                aliases = BREED_ALIASES.get(b, [b])
                if any(alias in img_breed for alias in aliases) or b in img_breed:
                    matched = True; break
            if matched:
                score += 0.25
                trace.append(("pet_breed_match", 0.25, f"品种匹配:{','.join(q_breed)}"))
            else:
                score -= 0.20
                trace.append(("pet_breed_mismatch", -0.20, f"品种不符(查询:{q_breed[0]})"))

        q_colors = query_terms.get("pet_coat_color", [])
        if q_colors:
            img_colors = pet.get("coat_color", [])
            matched = [c for c in q_colors if c in img_colors]
            if matched:
                bonus = 0.08 * len(matched); score += bonus
                trace.append(("pet_color_match", bonus, f"毛色匹配:{','.join(matched)}"))
            else:
                score -= 0.03
                trace.append(("pet_color_mismatch", -0.03, "毛色不符"))

        q_act = query_terms.get("pet_action", [])
        if q_act:
            img_act = pet.get("action", "")
            if any(a in img_act for a in q_act):
                score += 0.14
                trace.append(("pet_action_match", 0.14, f"动作匹配:{','.join(q_act)}"))

        q_env = query_terms.get("pet_environment", [])
        if q_env:
            img_env = pet.get("environment", "")
            if any(e in img_env for e in q_env):
                score += 0.10
                trace.append(("pet_env_match", 0.10, f"环境匹配:{','.join(q_env)}"))

        q_petc = query_terms.get("pet_count", [])
        if q_petc:
            img_cnt = pet.get("count", 1)
            qc = q_petc[0]
            if (qc == "一个" and img_cnt == 1) or (qc == "两个" and img_cnt == 2) or \
               (qc == "三个" and img_cnt == 3) or (qc == "一群" and img_cnt >= 4):
                score += 0.10
                trace.append(("pet_count_match", 0.10, "数量匹配"))
            else:
                score -= 0.15
                trace.append(("pet_count_mismatch", -0.15, f"数量不符(查询:{qc})"))

        q_life = query_terms.get("pet_life_stage", [])
        if q_life:
            img_stage = pet.get("life_stage", "")
            if any(s in img_stage for s in q_life):
                score += 0.12
                trace.append(("pet_life_stage_match", 0.12, f"年龄匹配:{','.join(q_life)}"))
            else:
                score -= 0.12
                trace.append(("pet_life_stage_mismatch", -0.12, f"年龄不符(查询:{','.join(q_life)})"))

    # ---------- 实体关键词匹配 ----------
    query_entities = query_terms.get("extracted_entities", [])
    if query_entities:
        img_keywords = [normalize_text(k) for k in img_info.get("keywords", [])]
        img_desc = normalize_text(img_info.get("description", ""))
        SYNONYM_MAP = {
            "沙滩排球": ["沙滩排球", "beach volleyball", "排球比赛", "海滩排球"],
        }
        PET_NICKNAMES = {"小狗", "小猫", "小狗狗", "小猫咪", "狗子", "喵星人", "汪星人"}

        matched = []
        for e in query_entities:
            if e in PET_NICKNAMES: continue
            if e in PET_COLORS or e in PET_ANIMAL_TYPES or e in {"幼年", "成年", "老年", "证件照"}:
                continue
            if e in img_desc or any(e in kw for kw in img_keywords):
                matched.append(e)
            elif len(e) > 1 and e in SYNONYM_MAP:
                synonyms = SYNONYM_MAP[e]
                if any(syn in img_desc or any(syn in kw for kw in img_keywords) for syn in synonyms):
                    matched.append(e)

        if matched:
            bonus = 0.0
            weak_entities = {"人", "打", "跑", "亚洲人", "西方人", "黑人", "白人"}
            for e in matched:
                if len(e) > 1 and e not in weak_entities:
                    bonus += 0.15
                else:
                    bonus += 0.05
            bonus = min(bonus, 0.35)
            score += bonus
            trace.append(("entity_keyword_match", round(bonus, 3), f"命中实体: {', '.join(matched)}"))

    compressed = 1.0 / (1.0 + math.exp(-3.0 * (score - 0.3)))
    final = compressed * 100
    if final > 99.0: final = 99.0
    final = int(round(final))
    final = max(0, min(99, final))
    return final, trace


def build_reasoning_markdown(user_query, decomposition, top_results):
    lines = [f"### 查询「{user_query}」的可解释检索结果", ""]
    entities = decomposition.get("extracted_entities", [])
    if entities: lines.append(f"**提取实体**：{'、'.join(entities)}")
    person_attr = decomposition.get("person_attributes", {})
    if any(person_attr.values()):
        parts = []
        if person_attr.get("count"): parts.append(f"数量：{','.join(person_attr['count'])}")
        if person_attr.get("gender"): parts.append(f"性别：{','.join(person_attr['gender'])}")
        if person_attr.get("ethnicity"): parts.append(f"人种/国籍：{','.join(person_attr['ethnicity'])}")
        lines.append(f"**人物属性**：{'；'.join(parts)}")
    pet_attr = decomposition.get("pet_attributes", {})
    if any(pet_attr.values()):
        parts = []
        if pet_attr.get("animal_type"): parts.append(f"动物：{','.join(pet_attr['animal_type'])}")
        if pet_attr.get("breed"): parts.append(f"品种：{','.join(pet_attr['breed'])}")
        if pet_attr.get("coat_color"): parts.append(f"毛色：{','.join(pet_attr['coat_color'])}")
        if pet_attr.get("action"): parts.append(f"动作：{','.join(pet_attr['action'])}")
        if pet_attr.get("life_stage"): parts.append(f"年龄：{','.join(pet_attr['life_stage'])}")
        lines.append(f"**宠物属性**：{'；'.join(parts)}")

    lines.append(""); lines.append("#### 推理链（Top 3）")
    for idx, r in enumerate(top_results, start=1):
        lines.append(f"{idx}. `{r['img_path']}` | 最终得分: {r['score']}")
        for rule, delta, evidence in r["trace"]:
            sign = "+" if delta >= 0 else ""
            lines.append(f"   - {rule}: {sign}{delta:.3f} ({evidence})")
    try:
        explanation = generate_explanation(user_query, top_results, metadata)
        lines.append(""); lines.append("#### 🤖 智能推理说明"); lines.append(explanation)
    except Exception as e:
        print(f"生成解释失败: {e}")
    return "\n".join(lines)

def search_photos(user_query: str):
    if not user_query or not user_query.strip():
        return [None, None, None, "请输入检索描述。", []]
    parsed = parse_query(user_query, L2C_ALL, C2L_ALL)
    qv = model.encode([parsed["expanded_query"]], normalize_embeddings=True).astype("float32")
    k = min(20, len(image_paths))
    sims, indices = vector_index.search(qv, k)
    results = []
    for i in range(k):
        idx = int(indices[0][i])
        if idx < 0: continue
        p = image_paths[idx]
        info = metadata[p]
        base_sim = float(sims[0][i])
        sc, trace = rerank_score(parsed["terms"], info, base_sim)
        results.append({"img_path": p, "score": sc, "raw_score": sc, "trace": trace})
    results.sort(key=lambda x: x["raw_score"], reverse=True)
    if not results:
        return [None, None, None, "未找到匹配照片。", []]
    top3 = results[:3]
    paths = [r["img_path"] for r in top3]
    while len(paths) < 3: paths.append(None)
    report = build_reasoning_markdown(user_query, parsed["decomposition"], top3)
    rows = []
    for r in top3:
        contrib = " | ".join([f"{name}:{delta:+.2f}" for name, delta, _ in r["trace"]])
        rows.append([r["img_path"], r["score"], contrib])
    return paths[0], paths[1], paths[2], report, rows


# ==================== 界面 (Gradio) ====================
CUSTOM_CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #D4E8FC 0%, #B2D4F5 100%);
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 90vh;
    text-align: center;
    padding: 2rem 1rem;
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(25px);
    border-radius: 48px;
    margin: 1.5rem;
    box-shadow: 0 25px 50px -8px rgba(30, 80, 130, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.5);
}
.book {
    width: 240px;
    height: 320px;
    position: relative;
    perspective: 1500px;
    margin-bottom: 2rem;
}
.book .page {
    display: block;
    width: 120px;
    height: 320px;
    background-color: rgba(255, 255, 255, 0.7);
    position: absolute;
    top: 0;
    box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    background-image: linear-gradient(to right, rgba(160, 190, 220, 0.1) 0%, transparent 10%);
    border: 1px solid rgba(255, 255, 255, 0.8);
    transform-origin: left center;
    animation: flipPage 3s ease-in-out forwards;
}
.book .page:first-child {
    left: 0;
    border-radius: 4px 0 0 4px;
    border-right: none;
}
.book .page:last-child {
    left: 120px;
    border-radius: 0 4px 4px 0;
    border-left: none;
    transform-origin: right center;
    animation: flipPageRight 3s ease-in-out forwards;
}
.book::after {
    content: '';
    position: absolute;
    top: 2%;
    left: 50%;
    width: 4px;
    height: 96%;
    background: linear-gradient(to right, rgba(0,0,0,0.1), transparent);
    transform: translateX(-50%);
    z-index: 2;
}
@keyframes flipPage {
    0% { transform: rotateY(0deg); }
    30% { transform: rotateY(-25deg); box-shadow: -5px 10px 15px rgba(0,0,0,0.1); }
    70% { transform: rotateY(-5deg); }
    100% { transform: rotateY(0deg); }
}
@keyframes flipPageRight {
    0% { transform: rotateY(0deg); }
    30% { transform: rotateY(25deg); box-shadow: 5px 10px 15px rgba(0,0,0,0.1); }
    70% { transform: rotateY(5deg); }
    100% { transform: rotateY(0deg); }
}
.main-title {
    font-size: 5.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF7675, #FDCB6E, #00CEC9, #A29BFE, #6C5CE7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 4px;
    margin-bottom: 0.8rem;
    animation: fadeInUp 1.5s ease-out;
}
.sub-title {
    font-family: 'Georgia', 'Times New Roman', serif;
    font-style: italic;
    font-size: 1.5rem;
    color: #2C3E50;
    margin-top: 0;
    margin-bottom: 2rem;
    animation: fadeInUp 1.8s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.start-btn {
    background: transparent;
    border: 2px solid rgba(255,255,255,0.6);
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.3rem !important;
    font-weight: bold;
    padding: 0.7rem 3rem !important;
    border-radius: 50px !important;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.25);
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    transition: all 0.3s;
    cursor: pointer;
}
.start-btn:hover {
    background: rgba(255, 255, 255, 0.4);
    border-color: rgba(255,255,255,0.9);
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
}
.search-panel {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    padding: 2rem 1.5rem;
    box-shadow: 0 20px 40px -10px rgba(30, 80, 130, 0.08);
    margin: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.6);
}
.image-card {
    border-radius: 20px;
    overflow: hidden;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gr-text-input > label,
.gr-text-input > .wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gr-text-input input {
    background: rgba(255, 255, 255, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    border-radius: 40px !important;
    padding: 0.7rem 1.2rem !important;
    font-size: 1rem !important;
    backdrop-filter: blur(15px);
    color: #1e293b !important;
}
.gr-text-input input:focus {
    border-color: #74B9FF !important;
    box-shadow: 0 0 0 3px rgba(116, 185, 255, 0.3) !important;
}
.gr-image {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
"""

WELCOME_HTML = """
<div class="welcome-container">
    <div class="book">
        <span class="page"></span>
        <span class="page"></span>
    </div>
    <h1 class="main-title">AdaphotoRet</h1>
    <p class="sub-title">Finding the moments you remember,<br>with a heart that truly understands</p>
    <div style="margin: 2rem 0; font-size: 1.1rem; color: #2d3748; line-height: 2; max-width: 600px;">
        <p style="font-size: 1.2rem; font-weight: 600;">“照片是瞬间的收藏，记忆是永恒的底片。”</p>
        <p style="font-style: italic;">The best thing about a picture is that it never changes, even when the people in it do.</p>
        <p style="font-size: 2rem; margin: 0.5rem 0;">📷 ✨ 🌊</p>
    </div>
</div>
"""

def switch_to_search():
    return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(css=CUSTOM_CSS, title="AdaphotoRet - 记忆相册") as demo:
    with gr.Column(visible=True, elem_classes="welcome-container") as welcome_block:
        gr.HTML(WELCOME_HTML)
        start_btn = gr.Button("✨ 开启记忆 ✨", elem_classes="start-btn", size="lg")

    with gr.Column(visible=False, elem_classes="search-panel") as search_block:
        gr.Markdown("## 📷 用自然语言寻找你的照片")
        with gr.Row():
            query_input = gr.Textbox(
                label="输入描述",
                placeholder="例如：一只成年蓝白猫；菜花田里的小狗；盯着镜头的梅花鹿",
                scale=6
            )
            search_btn = gr.Button("检索", variant="primary", scale=1, size="lg")

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=200):
                top1_img = gr.Image(label="🥇 最佳匹配", type="filepath", height=380, elem_classes="image-card")
            with gr.Column(scale=1, min_width=200):
                top2_img = gr.Image(label="🥈 第二候选", type="filepath", height=240, elem_classes="image-card")
            with gr.Column(scale=1, min_width=200):
                top3_img = gr.Image(label="🥉 第三候选", type="filepath", height=240, elem_classes="image-card")

        report_output = gr.Markdown(label="📋 可解释推理报告")
        table_output = gr.Dataframe(
            headers=["候选图片", "最终匹配度", "规则贡献分解"],
            datatype=["str", "number", "str"],
            label="📊 Top3 打分明细",
            interactive=False,
            wrap=True,
        )

    start_btn.click(fn=switch_to_search, outputs=[welcome_block, search_block])
    search_btn.click(
        fn=search_photos,
        inputs=query_input,
        outputs=[top1_img, top2_img, top3_img, report_output, table_output]
    )
    query_input.submit(
        fn=search_photos,
        inputs=query_input,
        outputs=[top1_img, top2_img, top3_img, report_output, table_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)