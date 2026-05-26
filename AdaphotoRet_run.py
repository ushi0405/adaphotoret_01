import json
import math
import os
import re
from typing import Dict, List, Set, Tuple

import faiss
import gradio as gr
import jieba
import jieba.posseg as pseg
from pillow_heif import register_heif_opener
register_heif_opener()

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

from attributes import (
    extract_person_attributes, expand_person_attributes,
    extract_pet_attributes, expand_pet_attributes,
    PET_COLORS, PET_ANIMAL_TYPES,
    BREED_ALIASES, BREED_NORMALIZATION
)
from llm_explainer import generate_explanation

print("正在加载语义模型...")
model = SentenceTransformer("./local_model")
print("语义模型加载完成。")

# 基础地标知识完全移除，只保留必要的场景/活动词典
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


def match_terms(text: str, terms: Set[str]) -> bool:
    return any(term in normalize_text(text) for term in terms)


def parse_location_from_info(info: Dict):
    cities, landmarks = set(), set()
    loc = info.get("location")
    if isinstance(loc, dict):
        city = loc.get("city", "").strip()
        if city: cities.add(city)
        lm = loc.get("landmarks", [])
        if isinstance(lm, list):
            landmarks.update(str(x).strip() for x in lm if str(x).strip())
        elif lm:
            landmarks.add(str(lm).strip())
    return cities, landmarks


def flatten_info_text(info: Dict) -> str:
    scene = str(info.get("scene", ""))
    desc = str(info.get("description", ""))
    image_type = str(info.get("image_type", ""))
    keywords = info.get("keywords", [])
    kw_text = " ".join(str(k) for k in keywords) if isinstance(keywords, list) else str(keywords)
    loc = info.get("location", {}) if isinstance(info.get("location"), dict) else {}
    city = loc.get("city", "")
    landmarks = loc.get("landmarks", [])
    if isinstance(landmarks, list):
        landmarks_text = " ".join(str(x) for x in landmarks)
    else:
        landmarks_text = str(landmarks) if landmarks else ""
    loc_text = f"{city} {landmarks_text}".strip()
    return f"{scene} {image_type} {desc} {kw_text} {loc_text}".strip()


def enrich_metadata(metadata_dict):
    enriched = {}
    for path, info in metadata_dict.items():
        cloned = dict(info)
        flat_text = flatten_info_text(cloned)
        doc_cities, doc_landmarks = parse_location_from_info(cloned)
        cloned["_search_text"] = flat_text
        cloned["_cities"] = sorted(doc_cities)
        cloned["_landmarks"] = sorted(doc_landmarks)
        enriched[path] = cloned
    return enriched


def extract_entities_by_pos(text: str) -> List[str]:
    stop_words = {'的', '了', '是', '在', '和', '与', '或', '一个', '两个', '那种', '这个', '那个', '中', '里', '有', '被', '把', '照片', '图片', '图像'}
    words = pseg.cut(text)
    entities = []
    for word, flag in words:
        if word in stop_words: continue
        if flag[0] in ('n', 'v', 'ns', 'nt', 't', 'a', 'f', 's', 'i', 'j', 'l'):
            if len(word) == 1 and flag[0] not in ('n', 'ns', 'nt', 'v'): continue
            entities.append(word)
    seen = set()
    unique = []
    for e in entities:
        if e not in seen:
            seen.add(e); unique.append(e)
    return unique


def extract_geo_candidates(text: str) -> List[str]:
    """从查询中提取可能的地理候选词（地名、专名、长名词等）"""
    candid = []
    words = pseg.cut(text)
    for w, flag in words:
        if len(w) >= 2 and flag in ('ns', 'nz', 'n') and w not in {"照片", "图片", "一个", "两个", "一群", "多个", "宠物", "动物", "比赛", "娱乐"}:
            candid.append(w)
    chinese_chunks = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    for chunk in chinese_chunks:
        if chunk not in candid:
            candid.append(chunk)
    return list(set(candid))


def parse_query(user_query: str) -> Dict:
    processed = user_query
    for key, normalized in BREED_NORMALIZATION.items():
        processed = processed.replace(key, normalized)

    q = normalize_text(processed)
    entities = extract_entities_by_pos(processed)
    person_attr = extract_person_attributes(processed)
    person_expand = expand_person_attributes(person_attr)
    pet_attr = extract_pet_attributes(processed)
    pet_expand = expand_pet_attributes(pet_attr)
    geo_candidates = extract_geo_candidates(processed)

    cities = set()
    landmarks = set()
    for word, flag in pseg.cut(processed):
        if flag == 'ns':
            cities.add(word)
        elif flag == 'n' and len(word) >= 3:
            landmarks.add(word)

    terms = {
        "needs_grass": any(t in q for t in GRASS_TERMS),
        "needs_track": any(t in q for t in TRACK_TERMS),
        "needs_running": any(t in q for t in RUN_TERMS),
        "needs_two_people": any(t in q for t in TWO_PEOPLE_TERMS),
        "needs_bridge": any(t in q for t in BRIDGE_TERMS),
        "needs_academic_writing": any(t in q for t in ACADEMIC_WRITING_TERMS),
        "needs_example": any(t in q for t in EXAMPLE_TERMS),
        "needs_structure": any(t in q for t in STRUCTURE_TERMS),
        "cities": cities,
        "landmarks": landmarks,
        "person_count": person_attr["count"],
        "person_gender": person_attr["gender"],
        "person_ethnicity": person_attr["ethnicity"],
        "extracted_entities": entities,
        "needs_pet": len(pet_attr["animal_type"]) > 0 or len(pet_attr["breed"]) > 0,
        "pet_animal_type": pet_attr["animal_type"],
        "pet_breed": pet_attr["breed"],
        "pet_coat_color": pet_attr["coat_color"],
        "pet_action": pet_attr["action"],
        "pet_environment": pet_attr["environment"],
        "pet_count": pet_attr["count"],
        "pet_life_stage": pet_attr["life_stage"],
        "geo_candidates": geo_candidates,
    }

    entity_str = " ".join(entities)
    person_str = " ".join(person_expand)
    pet_str = " ".join(pet_expand)
    geo_str = " ".join(list(cities) + list(landmarks) + geo_candidates)
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

metadata = enrich_metadata(metadata_raw)


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
    score = base_sim * 0.50
    trace = [("base_semantic_similarity", round(base_sim, 4), "向量语义相似")]

    # ---------- 一级硬槽位 ----------
    q_needs_pet = query_terms.get("needs_pet", False)
    q_needs_person = (
        bool(query_terms.get("person_count")) or
        bool(query_terms.get("person_ethnicity")) or
        bool(query_terms.get("person_gender"))
    )

    if q_needs_pet:
        query_category = "宠物"
    elif q_needs_person:
        query_category = "人物"
    else:
        query_category = None

    img_category = img_info.get("category", "")
    if not img_category:
        if "pet_details" in img_info:
            img_category = "宠物"
        else:
            main = img_info.get("main_subjects", {})
            if main.get("count", 0) > 0 and main.get("count_category") not in ["无", "单个物体"]:
                img_category = "人物"
            else:
                img_category = "风景或其他"

    if query_category and query_category != img_category:
        score = -2.0
        trace.append(("hard_category_mismatch", -2.0, f"查询需要{query_category}，但图片是{img_category}"))
        compressed = 1.0 / (1.0 + math.exp(-3.0 * (score - 0.3)))
        final = compressed * 100
        if final > 99.0: final = 99.0
        final = int(round(final))
        final = max(0, min(99, final))
        return final, trace

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

    # ---------- 地理匹配（精确 + GPS 模糊）----------
    q_cities = query_terms.get("cities", set())
    q_landmarks = query_terms.get("landmarks", set())
    geo_candidates = query_terms.get("geo_candidates", [])
    geo_hit = False

    if q_cities:
        overlap_city = q_cities.intersection(doc_cities)
        if overlap_city:
            score += 0.25
            trace.append(("geo_city_match", 0.25, f"命中城市:{','.join(sorted(overlap_city))}"))
            geo_hit = True

    if q_landmarks:
        overlap_lm = q_landmarks.intersection(doc_landmarks)
        if overlap_lm:
            score += 0.30
            trace.append(("geo_landmark_match", 0.30, f"命中地标:{','.join(sorted(overlap_lm))}"))
            geo_hit = True

    if geo_candidates and not geo_hit:
        gps_address = img_info.get("location", {}).get("gps_address", "")
        if gps_address:
            for gc in geo_candidates:
                if gc in gps_address:
                    if len(gc) >= 3:
                        score += 0.30
                        trace.append(("gps_address_match", 0.30, f"地址匹配: {gc}"))
                    else:
                        score += 0.15
                        trace.append(("gps_address_partial_match", 0.15, f"地址部分匹配: {gc}"))
                    geo_hit = True
                    break

    if (q_cities or q_landmarks or geo_candidates) and not geo_hit:
        penalty = -1.5
        score += penalty
        trace.append(("geo_mismatch_penalty", penalty, "未命中任何地理信息"))

    # 学术/示例/结构等
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

    # ---------- 人物属性 ----------
    if not query_terms.get("needs_pet"):
        people = img_info.get("main_subjects", {})
        if people:
            cnt = people.get("count", 0)
            ethnicity = people.get("primary_ethnicity", "")
            person_count_q = query_terms.get("person_count", [])
            if person_count_q:
                qc = person_count_q[0]
                matched = False
                if qc == "一个" and cnt == 1: score += 0.12; matched = True
                elif qc == "两个" and cnt == 2: score += 0.12; matched = True
                elif qc == "三个" and cnt == 3: score += 0.12; matched = True
                elif qc == "一群" and cnt >= 4: score += 0.12; matched = True
                if not matched:
                    penalty = -0.10
                    if qc == "一个" and cnt >= 5: penalty = -0.20
                    elif qc == "两个" and cnt >= 5: penalty = -0.15
                    elif qc == "一群" and cnt <= 2: penalty = -0.15
                    score += penalty
                    trace.append(("person_count_mismatch", penalty, "人物数量不符"))
                else:
                    trace.append(("person_count_match", 0.12, f"命中{qc}约束"))
            ethnicity_q = query_terms.get("person_ethnicity", [])
            if ethnicity_q:
                qe = ethnicity_q[0]
                matched = False
                if ("西方/白种人" in qe and "白种人" in ethnicity) or \
                   ("亚洲/黄种人" in qe and "黄种人" in ethnicity) or \
                   ("非洲/黑种人" in qe and "黑种人" in ethnicity):
                    score += 0.06; trace.append(("person_ethnicity_match", 0.06)); matched = True
                elif "外国" in qe and ("白种人" in ethnicity or "黑种人" in ethnicity):
                    score += 0.04; trace.append(("person_ethnicity_match", 0.04)); matched = True
                if not matched:
                    score -= 0.06; trace.append(("person_ethnicity_mismatch", -0.06))

    # ---------- 宠物匹配 ----------
    if query_terms.get("needs_pet") and "pet_details" in img_info:
        pet = img_info["pet_details"]
        q_animal = query_terms.get("pet_animal_type", [])
        if q_animal:
            img_animal = pet.get("animal_type", "")
            if any(a in img_animal for a in q_animal) or any(a in pet.get("breed", "") for a in q_animal):
                score += 0.16; trace.append(("pet_animal_match", 0.16))
            else:
                score -= 0.25; trace.append(("pet_animal_mismatch", -0.25))
        q_breed = query_terms.get("pet_breed", [])
        if q_breed:
            img_breed = pet.get("breed", "")
            matched = False
            for b in q_breed:
                aliases = BREED_ALIASES.get(b, [b])
                if any(alias in img_breed for alias in aliases) or b in img_breed:
                    matched = True; break
            if matched:
                score += 0.25; trace.append(("pet_breed_match", 0.25))
            else:
                score -= 0.20; trace.append(("pet_breed_mismatch", -0.20))
        q_colors = query_terms.get("pet_coat_color", [])
        if q_colors:
            img_colors = pet.get("coat_color", [])
            matched_colors = [c for c in q_colors if c in img_colors]
            if matched_colors:
                bonus = 0.08 * len(matched_colors); score += bonus
                trace.append(("pet_color_match", bonus))
            else:
                score -= 0.03; trace.append(("pet_color_mismatch", -0.03))
        q_act = query_terms.get("pet_action", [])
        if q_act:
            img_act = pet.get("action", "")
            if any(a in img_act for a in q_act):
                score += 0.14; trace.append(("pet_action_match", 0.14))
        q_env = query_terms.get("pet_environment", [])
        if q_env:
            img_env = pet.get("environment", "")
            if any(e in img_env for e in q_env):
                score += 0.10; trace.append(("pet_env_match", 0.10))
        q_petc = query_terms.get("pet_count", [])
        if q_petc:
            img_cnt = pet.get("count", 1)
            qc = q_petc[0]
            if (qc == "一个" and img_cnt == 1) or (qc == "两个" and img_cnt == 2) or \
               (qc == "三个" and img_cnt == 3) or (qc == "一群" and img_cnt >= 4):
                score += 0.10; trace.append(("pet_count_match", 0.10))
            else:
                score -= 0.15; trace.append(("pet_count_mismatch", -0.15))

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
            CORE_ACTIVITY_TERMS = {"沙滩排球", "beach volleyball", "排球比赛"}
            bonus = 0.0
            weak_entities = {"人", "打", "跑", "亚洲人", "西方人", "黑人", "白人"}
            for e in matched:
                if len(e) > 1 and e not in weak_entities:
                    if e in CORE_ACTIVITY_TERMS:
                        bonus += 0.25
                    else:
                        bonus += 0.15
                else:
                    bonus += 0.05
            bonus = min(bonus, 0.50)
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
        tags = []
        for rule, delta, evidence in r["trace"]:
            sign = "+" if delta >= 0 else ""
            tags.append(f"[{rule} {sign}{delta:.3f}]")
        lines.append(" ".join(tags))
        evi = " | ".join([evidence for _, _, evidence in r["trace"]])
        lines.append(f"<small>📌 {evi}</small>")
    try:
        explanation = generate_explanation(user_query, top_results, metadata)
        lines.append(""); lines.append("#### 🤖 智能推理说明"); lines.append(explanation)
    except Exception as e:
        print(f"生成解释失败: {e}")
    return "\n".join(lines)


def search_photos(user_query: str):
    if not user_query or not user_query.strip():
        return [None, None, None, "请输入检索描述。", [], []]
    parsed = parse_query(user_query)
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
        results.append({"img_path": p, "score": sc, "trace": trace})
    results.sort(key=lambda x: x["score"], reverse=True)
    if not results:
        return [None, None, None, "未找到匹配照片。", [], []]
    top3 = results[:3]
    paths = [r["img_path"] for r in top3]
    while len(paths) < 3:
        paths.append(None)
    report = build_reasoning_markdown(user_query, parsed["decomposition"], top3)
    rows = []
    for r in top3:
        contrib = " | ".join([f"{name}:{delta:+.2f}" for name, delta, _ in r["trace"]])
        rows.append([r["img_path"], r["score"], contrib])
    return paths[0], paths[1], paths[2], report, rows, results[:3]

def rank_all_photos(user_query: str) -> List[str]:
    """
    对全部照片按相关性得分降序排序，返回图片路径列表（得分仅内部使用，不返回）
    与 search_photos 使用完全相同的打分逻辑（parse_query + model.encode + rerank_score）
    """
    if not user_query or not user_query.strip():
        # 无查询时返回所有图片（保持原顺序）
        return image_paths[:]
    
    parsed = parse_query(user_query)
    qv = model.encode([parsed["expanded_query"]], normalize_embeddings=True).astype("float32")
    
    # 遍历所有图片，计算得分
    all_results = []
    for idx, p in enumerate(image_paths):
        info = metadata[p]
        # 获取该图片的向量（直接从索引重建，避免重复编码）
        vec = vector_index.reconstruct(idx)
        base_sim = float(qv @ vec)
        sc, _ = rerank_score(parsed["terms"], info, base_sim)
        all_results.append((p, sc))
    
    # 按得分降序排序
    all_results.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in all_results]


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
/* ... 其余样式省略，不影响功能 */
"""

WELCOME_HTML = """
<div class="welcome-container">
    <div class="book">
        <span class="page"></span>
        <span class="page"></span>
    </div>
    <h1 class="main-title">AdaphotoRet</h1>
    <p class="sub-title">Finding the moments you remember,<br>with a heart that truly understands</p>
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