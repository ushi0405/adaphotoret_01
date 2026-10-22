import json
import os
from pathlib import Path
from io import BytesIO
from html import escape
from datetime import datetime
import streamlit as st
from backend_resources import load_core_resources
from PIL import Image, ImageOps, UnidentifiedImageError
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass
from llm_explainer import get_deepseek_client, safe_api_error
from photo_query import build_photo_query
from clarification_policy import clarification_questions, must_return_results, concise_question

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODE = os.getenv("ADAPHOTORET_LOCAL_MODE", "0") == "1"
STATE_DIR = Path(os.getenv("ADAPHOTORET_STATE_DIR", str(PROJECT_ROOT / ".local_data")))

# ---------- 页面配置 ----------
st.set_page_config(page_title="AdaphotoRet", page_icon="🐱", layout="wide")

APP_CSS = """
<style>
:root {
    --ada-bg: #0f0f10;
    --ada-panel: #1e1e20;
    --ada-panel-2: #262433;
    --ada-border: rgba(255,255,255,.08);
    --ada-text: #f5f6fb;
    --ada-muted: #7e8598;
    --ada-accent: #8b78ff;
    --ada-accent-2: #b26df3;
}

.stApp {
    background: var(--ada-bg);
    color: var(--ada-text);
}

[data-testid="stSidebar"] {
    background: #202021;
    border-right: 1px solid var(--ada-border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--ada-muted);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--ada-text);
}

.ada-brand {
    padding: 1.35rem .25rem 1.25rem;
    border-bottom: 1px solid var(--ada-border);
    margin: -1rem -1rem 1.2rem;
}

.ada-logo {
    color: var(--ada-accent);
    font-size: 1.75rem;
    line-height: 1;
    font-weight: 800;
    letter-spacing: 0;
    padding: 0 1rem;
}

.ada-subtitle {
    color: var(--ada-muted);
    font-size: .9rem;
    padding: .45rem 1rem 0;
}

.ada-sidebar-label {
    color: var(--ada-muted);
    font-size: .84rem;
    margin: .65rem 0 .45rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] {
    background: #111112;
    border-radius: 8px;
    padding: 4px;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 7px;
    min-height: 36px;
    justify-content: center;
    margin: 0;
    padding: 0 .35rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #7567e8;
    box-shadow: none;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
    color: #fff;
    font-weight: 700;
}

.ada-hint-list {
    color: var(--ada-muted);
    border-bottom: 1px solid var(--ada-border);
    margin: 1rem -1rem 1.1rem;
    padding: .4rem 1.25rem 1.15rem;
}

.ada-hint-list li {
    margin: .75rem 0;
    color: #8d94aa;
}

.ada-page-title {
    margin: .1rem 0 1rem;
}

.ada-page-title h1 {
    font-size: 1.25rem;
    margin: 0;
}

.ada-page-title p {
    color: var(--ada-muted);
    margin: .2rem 0 0;
}

.ada-date-label {
    color: var(--ada-muted);
    font-size: .78rem;
    margin: 1rem 0 .35rem;
}

.ada-footer {
    color: #5c6476;
    font-size: .78rem;
    position: fixed;
    bottom: 1rem;
}

.main .block-container {
    padding-top: 1.55rem;
    max-width: 1540px;
}

[data-testid="stTextInput"] input {
    background: #1b1b1d;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 8px;
    color: var(--ada-text);
    min-height: 42px;
    font-size: 1rem;
}

[data-testid="stTextInput"] label {
    color: var(--ada-muted);
}

.stButton > button {
    background: #242427;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 8px;
    color: #dfe3f8;
    min-height: 36px;
}

.stButton > button:hover {
    border-color: rgba(139,120,255,.8);
    color: #fff;
}

[data-testid="stSidebar"] .stButton > button {
    justify-content: flex-start;
    background: #2b293a;
    border-color: rgba(139,120,255,.25);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #7567e8;
    color: #fff;
}

[data-testid="stChatMessage"] {
    background: transparent;
}

[data-testid="stChatInput"] textarea {
    background: #1b1b1d;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 10px;
    color: var(--ada-text);
    min-height: 46px;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

[data-testid="stChatInput"] button {
    background: #303036;
    border: 0;
    border-radius: 8px;
    box-shadow: none;
}

[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    border-color: rgba(139,120,255,.35) !important;
}

[data-testid="stImage"] img {
    border-radius: 8px;
    aspect-ratio: 4 / 3;
    object-fit: cover;
    width: 100%;
}

hr {
    border-color: var(--ada-border);
}

.ada-library-strip {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: .55rem;
    margin: .8rem 0 1rem;
}

.ada-library-chip {
    border: 1px solid rgba(255,255,255,.08);
    background: #18181a;
    border-radius: 8px;
    padding: .7rem .75rem;
}

.ada-library-chip strong {
    display: block;
    color: var(--ada-text);
    font-size: .95rem;
}

.ada-library-chip span {
    color: var(--ada-muted);
    font-size: .78rem;
}

.ada-section-title {
    color: #dfe3f8;
    font-size: 1rem;
    font-weight: 700;
    margin: 1.15rem 0 .55rem;
}

.ada-photo-pick {
    border: 1px solid rgba(255,255,255,.10);
    background: #171719;
    border-radius: 8px;
    padding: .35rem;
}

.ada-photo-pick.is-selected {
    border-color: rgba(139,120,255,.8);
    background: rgba(139,120,255,.12);
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ---------- Shared immutable catalogue; semantic model loads on first query ----------
try:
    backend_path = PROJECT_ROOT / "AdaphotoRet_run.py"
    core = load_core_resources(backend_path.read_bytes(), str(backend_path))
except Exception:
    st.error("照片库暂时无法加载。请检查项目依赖和 metadata.demo.json；本地模式可使用自己的元数据。")
    st.stop()
backend = core["backend"]
image_paths = core["image_paths"]
metadata = core["metadata"]
search_photos = backend.search_photos
parse_query = backend.parse_query
rerank_score = backend.rerank_score
resolve_photo_path = backend.resolve_photo_path


def cached_global_search(query, top_k=120):
    # Do not globally cache visitor queries or credentials.
    query = (query or "").strip()
    if not query:
        return tuple(image_paths[:top_k])
    if not image_paths:
        return ()
    resources = backend.get_search_resources()
    model, vector_index = resources["model"], resources["vector_index"]
    paths, info_by_path = resources["image_paths"], resources["metadata"]
    if model is None or vector_index is None or not paths:
        return ()
    parsed = parse_query(query)
    qv = model.encode([parsed["expanded_query"]], normalize_embeddings=True).astype("float32")
    k = min(top_k, len(paths))
    sims, indices = vector_index.search(qv, k)
    ranked = []
    for index, similarity in zip(indices[0], sims[0]):
        if int(index) < 0:
            continue
        path = paths[int(index)]
        score, _ = rerank_score(parsed["terms"], info_by_path[path], float(similarity))
        ranked.append((path, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return tuple(path for path, _ in ranked)


def session_api_key():
    key = (st.session_state.get("visitor_deepseek_key") or "").strip()
    if not key and LOCAL_MODE:
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    return key


def fine_search_ready():
    return bool(session_api_key() and st.session_state.get("deepseek_consent", False))


def clear_visitor_credentials():
    st.session_state.pop("visitor_deepseek_key", None)
    st.session_state["deepseek_consent"] = False


def render_api_settings():
    with st.sidebar.expander("🔑 配置自己的 API", expanded=not bool(session_api_key())):
        st.text_input("DeepSeek API 密钥", type="password", key="visitor_deepseek_key", max_chars=512,
                      help="只保留在当前浏览器会话对应的服务端内存中；不会写入文件或共享给其他访客。")
        if LOCAL_MODE and os.getenv("DEEPSEEK_API_KEY"):
            st.caption("本地模式已检测到 DEEPSEEK_API_KEY；输入框留空时使用该环境变量。")
        st.checkbox("我同意将当前照片对话中的用户线索和候选照片描述发送到 DeepSeek，费用由我的 API 账户承担。",
                    key="deepseek_consent")
        st.caption("全局搜索和照片库无需 API 密钥。精细检索发送文字元数据，不发送原始照片；请勿输入敏感信息。")
        st.button("清除密钥并撤回同意", on_click=clear_visitor_credentials,
                  width="stretch")
    st.sidebar.caption("本地模式：数据保存到本机。" if LOCAL_MODE else
                       "公共演示：对话、标签和相册仅在当前会话保存，刷新或断开后可能丢失。")
    st.sidebar.caption(f"当前照片库已加载 {len(image_paths)} 张照片。")


@st.cache_data(show_spinner=False)
def cached_image_tile(path, mtime, size=(520, 390), crop=True):
    try:
        with Image.open(resolve_photo_path(path)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            if crop:
                img = ImageOps.fit(img, size, method=resample, centering=(0.5, 0.5))
            else:
                img.thumbnail(size, resample=resample)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
            return buf.getvalue()
    except (OSError, UnidentifiedImageError):
        return None

def render_image_tile(path, crop=True):
    try:
        mtime = os.path.getmtime(resolve_photo_path(path))
    except OSError:
        return
    tile = cached_image_tile(path, mtime, crop=crop)
    if tile:
        st.image(tile, width="stretch")

# ---------- 对话历史管理（与原来相同）----------
CHAT_HISTORY_FILE = STATE_DIR / "chat_history.json"
PHOTO_LIBRARY_FILE = STATE_DIR / "photo_libraries.json"

def load_all_conversations():
    if LOCAL_MODE and CHAT_HISTORY_FILE.is_file():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            st.warning("本地对话记录无法读取，已启动空会话；原文件未更改。")
    return {}

def save_all_conversations(convs):
    if LOCAL_MODE:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(convs, f, ensure_ascii=False, indent=2)

if "conversations" not in st.session_state:
    st.session_state.conversations = load_all_conversations()
if "current_conv_id" not in st.session_state:
    convs = st.session_state.conversations
    if convs:
        latest_id = max(convs.keys(), key=lambda x: convs[x]["created"])
        st.session_state.current_conv_id = latest_id
    else:
        now = datetime.now().isoformat()
        new_id = now
        st.session_state.conversations[new_id] = {
            "name": "新对话",
            "created": now,
            "messages": []
        }
        st.session_state.current_conv_id = new_id
if "mode" not in st.session_state:
    st.session_state.mode = "全局搜索"
elif st.session_state.mode == "🔍 全局搜索":
    st.session_state.mode = "全局搜索"
elif st.session_state.mode == "💬 精细搜索":
    st.session_state.mode = "精细搜索"
if "mode_selector" not in st.session_state:
    st.session_state.mode_selector = st.session_state.mode
if "editing_conv_id" not in st.session_state:
    st.session_state.editing_conv_id = None
if "global_visible_count" not in st.session_state:
    st.session_state.global_visible_count = 12
if "last_global_query" not in st.session_state:
    st.session_state.last_global_query = ""
if "selected_tag_paths" not in st.session_state:
    st.session_state.selected_tag_paths = []

def load_photo_library_data():
    if LOCAL_MODE and PHOTO_LIBRARY_FILE.is_file():
        try:
            with open(PHOTO_LIBRARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            st.warning("本地照片库配置无法读取，已使用空配置；原文件未更改。")
            data = {}
    else:
        data = {}
    data.setdefault("photo_tags", {})
    data.setdefault("libraries", {})
    for config in data["libraries"].values():
        config.setdefault("photos", [])
        config.setdefault("tags", [])
        for tag in config.pop("aliases", []):
            if tag not in config["tags"]:
                config["tags"].append(tag)
        for aliases in config.pop("sections", {}).values():
            for tag in aliases:
                if tag not in config["tags"]:
                    config["tags"].append(tag)
    return data

def save_photo_library_data(data):
    if LOCAL_MODE:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(PHOTO_LIBRARY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

if "photo_library_data" not in st.session_state:
    st.session_state.photo_library_data = load_photo_library_data()

def split_tags(text):
    raw = (text or "").replace("，", ",").replace("；", ";").replace("、", ",")
    parts = []
    for chunk in raw.replace(";", ",").split(","):
        tag = chunk.strip()
        if tag and tag not in parts:
            parts.append(tag)
    return parts

def parse_section_rules(text):
    sections = {}
    for row in (text or "").splitlines():
        row = row.strip()
        if not row or ":" not in row:
            continue
        title, aliases = row.split(":", 1)
        section_name = title.strip()
        if section_name:
            sections[section_name] = split_tags(aliases)
    return sections

def get_photo_tags(path):
    return st.session_state.photo_library_data.get("photo_tags", {}).get(path, [])

def add_tags_to_photos(paths, tags):
    data = st.session_state.photo_library_data
    for path in paths:
        existing = data["photo_tags"].setdefault(path, [])
        for tag in tags:
            if tag not in existing:
                existing.append(tag)
    save_photo_library_data(data)

def add_photos_to_library(paths, library_name, tags=None):
    data = st.session_state.photo_library_data
    library = data["libraries"].setdefault(library_name, {"photos": [], "tags": []})
    library.setdefault("photos", [])
    library.setdefault("tags", [])
    for path in paths:
        if path not in library["photos"]:
            library["photos"].append(path)
    for tag in tags or []:
        if tag not in library["tags"]:
            library["tags"].append(tag)
    add_tags_to_photos(paths, tags or [])
    save_photo_library_data(data)

def remove_tag_from_photo(path, tag):
    data = st.session_state.photo_library_data
    tags = data.get("photo_tags", {}).get(path, [])
    if tag in tags:
        tags.remove(tag)
    if tags:
        data["photo_tags"][path] = tags
    else:
        data.get("photo_tags", {}).pop(path, None)
    save_photo_library_data(data)

def has_location(info):
    loc = info.get("location", {}) if isinstance(info.get("location"), dict) else {}
    return bool(loc.get("city") or loc.get("gps_address") or loc.get("gps_latitude"))

def location_library_name(info):
    loc = info.get("location", {}) if isinstance(info.get("location"), dict) else {}
    city = str(loc.get("city", "")).strip()
    if city:
        return city
    address = str(loc.get("gps_address", "")).strip()
    if "市" in address:
        idx = address.index("市")
        start = address.rfind("省", 0, idx)
        return address[start + 1:idx + 1] if start != -1 else address[:idx + 1]
    if "省" in address:
        return address[:address.index("省") + 1]
    return ""

def build_auto_libraries():
    libraries = {
        "全部照片": list(image_paths),
        "人物": [],
        "动物": [],
        "美食": [],
        "风景": [],
        "物品": [],
        "有位置信息": [],
    }
    for path in image_paths:
        info = metadata.get(path, {})
        category = info.get("category", "其他")
        if category == "宠物":
            category = "动物"
        if category in libraries:
            libraries[category].append(path)
        if has_location(info):
            libraries["有位置信息"].append(path)
            loc_name = location_library_name(info)
            if loc_name:
                libraries.setdefault(f"地点 · {loc_name}", []).append(path)
    return {name: paths for name, paths in libraries.items() if paths}

def custom_library_matches(config):
    explicit = set(config.get("photos", []))
    tags = set(config.get("tags", []))
    matches = set(explicit)
    for path, photo_tags in st.session_state.photo_library_data.get("photo_tags", {}).items():
        if any(tag in tags for tag in photo_tags):
            matches.add(path)
    return [p for p in image_paths if p in matches]

def toggle_selected_path(path):
    selected = st.session_state.selected_tag_paths
    if path in selected:
        selected.remove(path)
    else:
        selected.append(path)

def render_selectable_photo_grid(paths, cols_per_row=6):
    if not paths:
        st.caption("暂无照片")
        return
    total = len(paths)
    limit = min(st.session_state.get("tag_visible_count", 12), total)
    paths = paths[:limit]
    for i in range(0, len(paths), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(paths):
                path = paths[idx]
                selected = path in st.session_state.selected_tag_paths
                with col:
                    st.markdown(
                        f'<div class="ada-photo-pick {"is-selected" if selected else ""}">',
                        unsafe_allow_html=True,
                    )
                    render_image_tile(path)
                    label = "已选中" if selected else "选择"
                    if st.button(label, key=f"pick_{path}", width="stretch"):
                        toggle_selected_path(path)
                        st.rerun()
                    st.caption(os.path.basename(path))
                    st.markdown("</div>", unsafe_allow_html=True)
    st.caption(f"已显示 {limit} / 共 {total} 张可选照片")
    if limit < total and st.button("展开更多可选照片", key="more_tag_photos", width="stretch"):
        st.session_state.tag_visible_count = min(limit + 12, total)
        st.rerun()


def render_photo_grid(paths, cols_per_row=6, page_key="library_grid"):
    if not paths:
        st.caption("暂无照片")
        return
    limits = st.session_state.setdefault("photo_grid_limits", {})
    total = len(paths)
    limit = min(limits.get(page_key, 12), total)
    paths = paths[:limit]
    for i in range(0, len(paths), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(paths):
                with col:
                    render_image_tile(paths[idx])
                    tags = get_photo_tags(paths[idx])
                    if tags:
                        st.caption(" · ".join(tags[:3]))
    st.caption(f"已显示 {limit} / 共 {total} 张照片")
    if limit < total and st.button("展开更多照片", key=f"more_grid_{page_key}", width="stretch"):
        limits[page_key] = min(limit + 12, total)
        st.rerun()


def make_conversation_title(text, max_len=14):
    title = (text or "").strip()
    for token in ["帮我找", "帮我搜", "我想找", "想找", "查找", "搜索", "照片", "图片", "一下", "请"]:
        title = title.replace(token, "")
    title = " ".join(title.replace("\n", " ").split())
    title = title.strip(" ，。！？,.!?;；:：、")
    if not title:
        return "新对话"
    return title[:max_len]

def create_conversation():
    now = datetime.now().isoformat()
    st.session_state.conversations[now] = {
        "name": "新对话",
        "created": now,
        "messages": []
    }
    st.session_state.current_conv_id = now
    st.session_state.editing_conv_id = None
    save_all_conversations(st.session_state.conversations)

def delete_conversation(cid):
    if cid in st.session_state.conversations:
        del st.session_state.conversations[cid]
    if not st.session_state.conversations:
        create_conversation()
        return
    if st.session_state.current_conv_id == cid:
        convs = st.session_state.conversations
        st.session_state.current_conv_id = max(convs.keys(), key=lambda x: convs[x]["created"])
    if st.session_state.editing_conv_id == cid:
        st.session_state.editing_conv_id = None
    save_all_conversations(st.session_state.conversations)

def maybe_auto_name_current_conversation(user_message):
    conv = get_current_conversation()
    if not conv:
        return
    old_name = conv.get("name", "").strip()
    has_user_message = any(m.get("role") == "user" for m in conv.get("messages", []))
    if old_name in ["", "新对话", "未命名"] and not has_user_message:
        conv["name"] = make_conversation_title(user_message)
        save_all_conversations(st.session_state.conversations)

def conversation_day_label(created):
    try:
        dt = datetime.fromisoformat(created)
    except ValueError:
        return "更早"
    today = datetime.now().date()
    if dt.date() == today:
        return "今天"
    if (today - dt.date()).days == 1:
        return "昨天"
    return dt.strftime("%Y-%m-%d")

def render_brand_sidebar():
    st.sidebar.markdown(
        """
        <div class="ada-brand">
          <div class="ada-logo">AdaphotoRet</div>
          <div class="ada-subtitle">AI 照片检索助手</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_mode_selector():
    st.sidebar.markdown('<div class="ada-sidebar-label">搜索模式</div>', unsafe_allow_html=True)
    options = ["全局搜索", "精细搜索", "照片库"]
    if st.session_state.mode not in options:
        st.session_state.mode = options[0]
    if st.session_state.mode_selector not in options:
        st.session_state.mode_selector = st.session_state.mode

    def sync_mode():
        st.session_state.mode = st.session_state.mode_selector

    st.sidebar.radio(
        "搜索模式",
        options,
        index=options.index(st.session_state.mode),
        key="mode_selector",
        on_change=sync_mode,
        label_visibility="collapsed",
    )

def ensure_greeting():
    conv = get_current_conversation()
    if conv and not conv["messages"]:
        greeting = {
            "role": "assistant",
            "content": "你好～我是AdaphotoRet，你的私人记忆相册。想找哪张照片呢？可以直接告诉我哦！",
            "thinking": "",
            "images": [],
            "report": "",
            "kind": "greeting"
        }
        conv["messages"].append(greeting)
        save_all_conversations(st.session_state.conversations)

def get_current_conversation():
    return st.session_state.conversations.get(st.session_state.current_conv_id)

def get_current_messages():
    conv = get_current_conversation()
    return conv["messages"] if conv else []

def add_message(role, content, thinking="", images=None, report="", kind=None):
    conv = get_current_conversation()
    if conv:
        conv["messages"].append({
            "role": role,
            "content": content,
            "thinking": thinking,
            "images": images or [],
            "report": report,
            "kind": kind
        })
        save_all_conversations(st.session_state.conversations)

def search_photos_tool(query, client=None):
    try:
        result = search_photos(query, client=client, allow_environment=False, include_explanation=client is not None)
        top1, top2, top3, report, table_data, top_results = result
    except Exception:
        top1 = top2 = top3 = None
        report = "检索暂时失败，请检查模型和照片库后重试。未生成任何替代或模拟结果。"
        table_data, top_results = [], []
    return {
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "report": report,
        "table": table_data,
        "top_results": top_results
    }

def preview_search_candidates(query, top_k=5):
    if not query or not query.strip() or not image_paths:
        return []
    resources = backend.get_search_resources()
    model, vector_index = resources["model"], resources["vector_index"]
    if model is None or vector_index is None:
        return []
    parsed = parse_query(query)
    qv = model.encode([parsed["expanded_query"]], normalize_embeddings=True).astype("float32")
    paths, info_by_path = resources["image_paths"], resources["metadata"]
    k = min(80, len(paths))
    if not k:
        return []
    sims, indices = vector_index.search(qv, k)
    previews = []
    for i in range(k):
        idx = int(indices[0][i])
        if idx < 0:
            continue
        path = paths[idx]
        info = info_by_path[path]
        score, trace = rerank_score(parsed["terms"], info, float(sims[0][i]))
        previews.append({
            "path": path,
            "score": score,
            "scene": info.get("scene") or info.get("general_fields", {}).get("scene", ""),
            "description": (info.get("description") or info.get("general_fields", {}).get("description", ""))[:80],
            "matched": [evidence for _, delta, evidence in trace if delta > 0],
            "missing": [evidence for _, delta, evidence in trace if delta < 0],
        })
    previews.sort(key=lambda x: x["score"], reverse=True)
    return previews[:top_k]

def should_direct_search(previews):
    if not previews:
        return False
    top_score = previews[0]["score"]
    second_score = previews[1]["score"] if len(previews) > 1 else 0
    if top_score >= 70:
        return True
    return top_score >= 55 and top_score - second_score >= 18

# ---------- 流式助手（与之前相同）----------
def run_assistant(user_message):
    if not fine_search_ready():
        st.info("请先在侧边栏配置自己的 API 密钥并同意发送文字数据。全局搜索无需密钥。")
        return
    if not image_paths:
        st.info("照片库为空，请先配置照片及元数据。")
        return
    client = None
    try:
        client = get_deepseek_client(session_api_key(), allow_environment=False)
        _run_assistant_with_client(user_message, client)
    except Exception:
        st.error("精细检索暂时不可用，请检查 API 配置、网络和照片库后重试。")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                # Cleanup errors must not disclose provider/request details.
                pass


def show_search_results(query, client=None):
    """Display real ranked candidates; no model-authored query or extra question."""
    results = search_photos_tool(query, client=client)
    imgs = [results["top1"], results["top2"], results["top3"]]
    imgs = [img for img in imgs if img]
    content = ("先展示最接近的候选照片，供你确认。" if imgs
               else "未返回可展示的照片，请查看报告或调整查询。")
    with st.chat_message("assistant"):
        st.markdown(content)
        if imgs:
            cols = st.columns(len(imgs))
            for col, img in zip(cols, imgs):
                with col:
                    render_image_tile(img, crop=False)
        if results.get("report"):
            with st.expander("📋 详细报告", expanded=False):
                st.markdown(results["report"])
    add_message("assistant", content, images=imgs, report=results.get("report", ""), kind="results")


def _run_assistant_with_client(user_message, client):
    # Retrieval evidence and dialogue-control history must remain separate.
    history = list(get_current_messages())
    query = build_photo_query(history, user_message)
    if not query:
        st.info("请描述照片中的人物、物品、地点或动作；寻找另一张照片请新建对话。")
        return
    try:
        preview_candidates = preview_search_candidates(query, top_k=5)
    except Exception:
        st.error("检索资源暂时不可用，请检查模型下载和依赖后重试。")
        return
    add_message("user", user_message)
    # The existing reranker score is a heuristic, NOT a calibrated probability.
    # Enforce exits before ANY provider call/stream, irrespective of model output.
    if must_return_results(history, user_message, should_direct_search(preview_candidates)):
        show_search_results(query)
        return

    preview_lines = []
    for idx, item in enumerate(preview_candidates[:3], start=1):
        preview_lines.append(
            f"{idx}. 场景:{item['scene']}；描述:{item['description']}；"
            f"命中:{'、'.join(item['matched']) if item['matched'] else '无'}；"
            f"不足:{'、'.join(item['missing']) if item['missing'] else '无'}"
        )
    preview_context = "\n".join(preview_lines) if preview_lines else "无候选"
    asked = clarification_questions(history)
    # Past questions help avoid repetitions, but are never inserted into query.
    asked_context = "\n".join(asked) if asked else "无"
    system_prompt = f"""你是照片检索助手 AdaphotoRet。优先展示候选照片，不要反复盘问。
只把用户亲自描述的内容视为检索事实；助手问题、候选描述都不是用户确认的线索。
仅当一个问题能区分现有候选时才提问：一次只问一个问题，不超过40字。
不解释推理，不罗列食材，不重复问已经问过或用户已经回答过的特征。
如果候选都缺少目标对象，直接检索展示最接近的候选，不要反复确认用户要不要它。
同一照片对话最多提问3次；本轮之前已提问{len(asked)}次。无法有效缩小范围就检索。
只输出以下一种格式，不要输出思考、解释或其他答复：
【行动】提问：一个简短问题？
【行动】检索：

已问过的问题（仅用于避免重复，不是检索事实）：
{asked_context}

照片库预检索候选（不是用户线索，不能补入查询）：
{preview_context}"""
    context = [{"role": "system", "content": system_prompt}]
    context.extend({"role": "user", "content": m["content"]} for m in history
                   if m.get("role") == "user" and isinstance(m.get("content"), str))
    context.append({"role": "user", "content": user_message})
    full_response = ""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", messages=context, temperature=0.3,
            max_tokens=120, stream=True,
        )
        # Buffer first: unvalidated streams can contain a fourth/long question.
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
    except Exception as exc:
        # Local retrieval remains available even if the optional dialogue fails.
        st.info(safe_api_error(exc))
        show_search_results(query)
        return

    question = concise_question(full_response, [*history, {"role": "user", "content": user_message}], preview_candidates)
    if question:
        with st.chat_message("assistant"):
            st.markdown(question)
        add_message("assistant", question, kind="clarification")
    else:
        # Malformed actions, freeform answers, repeated/long questions cannot loop.
        show_search_results(query, client=client)

# ---------- 全局照片检索页面 ----------
def show_global_search():
    st.markdown(
        """
        <div class="ada-page-title">
          <h1>全局照片检索</h1>
          <p>在全部照片中快速语义搜索</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # 检索输入框
    query = st.text_input(
        "输入关键词检索照片（留空则显示所有照片）",
        placeholder="例如：山西、沙滩、海边、2024 年...",
        label_visibility="collapsed",
    )

    normalized_query = (query or "").strip()
    if normalized_query != st.session_state.last_global_query:
        st.session_state.last_global_query = normalized_query
        st.session_state.global_visible_count = 12

    # 每行显示的照片数量（紧凑布局）
    cols_per_row = 6
    # 获取排序后的照片列表
    try:
        with st.spinner("检索中（首次语义检索需要下载并加载模型）..."):
            ranked_paths = list(cached_global_search(normalized_query, top_k=200))
    except Exception:
        st.error("语义模型暂时不可用。请检查网络和依赖后重试；清空关键词仍可浏览真实照片。")
        return

    if not ranked_paths:
        st.info("没有找到匹配的照片")
        return

    # 计算需要多少行
    total = len(ranked_paths)
    visible_count = min(st.session_state.global_visible_count, total)
    visible_paths = ranked_paths[:visible_count]
    for i in range(0, visible_count, cols_per_row):
        cols = st.columns(cols_per_row, gap="small")  # 小间距
        for j in range(cols_per_row):
            idx = i + j
            if idx < visible_count:
                with cols[j]:
                    render_image_tile(visible_paths[idx])
                    # 可选：显示文件名或得分（简洁）
                    # st.caption(os.path.basename(ranked_paths[idx]))
    # 显示总数量
    st.caption(f"已显示 {visible_count} / 共 {total} 张照片")
    if visible_count < total:
        if st.button("加载更多", width="stretch"):
            st.session_state.global_visible_count = min(visible_count + 12, total)
            st.rerun()

def show_photo_libraries():
    st.markdown(
        """
        <div class="ada-page-title">
          <h1>照片库</h1>
          <p>自动分类、手动标签和成长记录相册</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = st.session_state.photo_library_data
    auto_libraries = build_auto_libraries()
    custom_configs = data.get("libraries", {})
    custom_libraries = {name: custom_library_matches(config) for name, config in custom_configs.items()}

    summary_items = []
    for name, paths in {**auto_libraries, **custom_libraries}.items():
        summary_items.append(f'<div class="ada-library-chip"><strong>{escape(name)}</strong><span>{len(paths)} 张照片</span></div>')
    st.markdown(f'<div class="ada-library-strip">{"".join(summary_items)}</div>', unsafe_allow_html=True)

    tab_view, tab_manage, tab_tag = st.tabs(["浏览照片库", "自定义照片库", "添加标签"])

    with tab_view:
        library_names = [f"自动 · {name}" for name in auto_libraries] + [f"自定义 · {name}" for name in custom_libraries]
        if not library_names:
            st.caption("暂无照片库")
            return
        selected = st.selectbox("选择照片库", library_names, label_visibility="collapsed")
        is_custom = selected.startswith("自定义 · ")
        lib_name = selected.replace("自动 · ", "").replace("自定义 · ", "")

        if is_custom:
            config = custom_configs.get(lib_name, {})
            library_paths = custom_libraries.get(lib_name, [])
            tags = config.get("tags", [])
            if tags:
                for tag in tags:
                    section_paths = [path for path in library_paths if tag in get_photo_tags(path)]
                    if section_paths:
                        st.markdown(f'<div class="ada-section-title">{escape(tag)}</div>', unsafe_allow_html=True)
                        render_photo_grid(section_paths, page_key=f"custom:{lib_name}:tag:{tag}")
                untagged = [path for path in library_paths if not any(tag in get_photo_tags(path) for tag in tags)]
                if untagged:
                    st.markdown('<div class="ada-section-title">未标注</div>', unsafe_allow_html=True)
                    render_photo_grid(untagged, page_key=f"custom:{lib_name}:untagged")
            else:
                render_photo_grid(library_paths, page_key=f"custom:{lib_name}")
        else:
            render_photo_grid(auto_libraries.get(lib_name, []), page_key=f"auto:{lib_name}")

    with tab_manage:
        st.markdown("##### 自建照片库")
        existing_names = ["新建"] + sorted(custom_configs.keys())
        target = st.selectbox("编辑照片库", existing_names)
        current = custom_configs.get(target, {}) if target != "新建" else {}
        library_name = st.text_input("照片库名称", value="" if target == "新建" else target, placeholder="例如：儿子成长记录")
        col_save, col_delete = st.columns([1, 1])
        with col_save:
            if st.button("保存照片库", width="stretch"):
                name = library_name.strip()
                if name:
                    if target != "新建" and target != name:
                        custom_configs.pop(target, None)
                    custom_configs.setdefault(name, current or {"photos": [], "tags": []})
                    custom_configs[name].setdefault("photos", [])
                    custom_configs[name].setdefault("tags", [])
                    data["libraries"] = custom_configs
                    save_photo_library_data(data)
                    st.success("已保存")
                    st.rerun()
        with col_delete:
            if target != "新建" and st.button("删除照片库", width="stretch"):
                custom_configs.pop(target, None)
                data["libraries"] = custom_configs
                save_photo_library_data(data)
                st.rerun()
        if target != "新建" and current:
            st.caption(f"{len(current.get('photos', []))} 张照片 · 标签：{', '.join(current.get('tags', [])) or '暂无'}")

    with tab_tag:
        st.markdown("##### 看图选择并归类")
        available_libraries = sorted(set(custom_configs.keys()) | set(auto_libraries.keys()))
        if not available_libraries:
            st.info("请先在“自定义照片库”里新建一个照片库。")
        target_library = st.selectbox("添加到照片库", available_libraries, disabled=not available_libraries)
        tags_text = st.text_input("本次添加的标签", placeholder="例如：儿子1岁、儿子满一岁、两岁生日")
        search_text = st.text_input("筛选照片范围", placeholder="例如：儿子、生日、海边、山西；留空显示全部")
        if st.session_state.get("tag_last_query", "") != search_text.strip():
            st.session_state.tag_last_query = search_text.strip()
            st.session_state.tag_visible_count = 12
        try:
            candidates = list(cached_global_search(search_text, top_k=80)) if search_text.strip() else image_paths[:80]
        except Exception:
            st.warning("语义筛选暂时不可用，当前显示未筛选照片。")
            candidates = image_paths[:80]
        if st.session_state.selected_tag_paths:
            kept, clear = st.columns([1, 1])
            with kept:
                st.caption(f"已选择 {len(st.session_state.selected_tag_paths)} 张")
            with clear:
                if st.button("清空选择", width="stretch"):
                    st.session_state.selected_tag_paths = []
                    st.rerun()

        action_col, select_col = st.columns([1, 1])
        with action_col:
            if st.button("加入照片库并打标签", width="stretch", disabled=not available_libraries):
                tags = split_tags(tags_text)
                selected_paths = [p for p in st.session_state.selected_tag_paths if p in image_paths]
                if selected_paths and target_library:
                    add_photos_to_library(selected_paths, target_library, tags)
                    st.session_state.selected_tag_paths = []
                    st.success("已归类")
                    st.rerun()
        with select_col:
            if st.button("选择当前筛选范围（含未展开照片）", width="stretch"):
                st.session_state.selected_tag_paths = list(dict.fromkeys(st.session_state.selected_tag_paths + candidates))
                st.rerun()

        render_selectable_photo_grid(candidates, cols_per_row=6)

        st.markdown("##### 已有标签")
        tagged_rows = []
        for path, tags in data.get("photo_tags", {}).items():
            tagged_rows.append({"照片": path, "标签": "，".join(tags)})
        if tagged_rows:
            st.dataframe(tagged_rows, width="stretch", hide_index=True)
        else:
            st.caption("还没有手动标签")

# ---------- 可解释精细检索页面（对话助手）----------
def show_fine_retrieval():
    ensure_greeting()
    current_conv = get_current_conversation()
    title = current_conv.get("name", "新对话") if current_conv else "新对话"
    title = escape(title)
    st.markdown(
        f"""
        <div class="ada-page-title">
          <h1>{title}</h1>
          <p>精细检索 · 可解释 AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 侧边栏对话管理
    with st.sidebar:
        if st.button("＋ 新建对话", width="stretch", type="primary"):
            create_conversation()
            st.rerun()

        convs = st.session_state.conversations
        if convs:
            conv_ids = list(convs.keys())
            conv_ids.sort(key=lambda x: convs[x]["created"], reverse=True)
            last_day = None
            for cid in conv_ids:
                cname = convs[cid].get("name", "未命名")
                day_label = conversation_day_label(convs[cid].get("created", cid))
                if day_label != last_day:
                    st.markdown(f'<div class="ada-date-label">{day_label}</div>', unsafe_allow_html=True)
                    last_day = day_label

                col1, col2, col3 = st.columns([7, 1.4, 1.4])
                with col1:
                    label = f"▣ {cname}" if cid == st.session_state.current_conv_id else f"□ {cname}"
                    if st.button(label, key=f"load_{cid}", width="stretch"):
                        st.session_state.current_conv_id = cid
                        st.session_state.editing_conv_id = None
                        st.rerun()
                with col2:
                    if st.button("✎", key=f"edit_{cid}", width="stretch"):
                        st.session_state.editing_conv_id = cid
                        st.rerun()
                with col3:
                    if st.button("×", key=f"del_{cid}", width="stretch"):
                        delete_conversation(cid)
                        st.rerun()

                if st.session_state.editing_conv_id == cid:
                    new_name = st.text_input(
                        "重命名",
                        value=cname,
                        key=f"rename_{cid}",
                        label_visibility="collapsed",
                    )
                    save_col, cancel_col = st.columns(2)
                    with save_col:
                        if st.button("保存", key=f"save_name_{cid}", width="stretch"):
                            convs[cid]["name"] = new_name.strip() or "未命名"
                            st.session_state.editing_conv_id = None
                            save_all_conversations(st.session_state.conversations)
                            st.rerun()
                    with cancel_col:
                        if st.button("取消", key=f"cancel_name_{cid}", width="stretch"):
                            st.session_state.editing_conv_id = None
                            st.rerun()
        else:
            st.caption("暂无对话")

    # 显示历史消息
    for msg in get_current_messages():
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                if msg.get("thinking"):
                    with st.expander("检索依据", expanded=False):
                        st.caption(msg["thinking"])
                if msg.get("content"):
                    st.write(msg["content"])
                if msg.get("images"):
                    cols = st.columns(len(msg["images"]))
                    for i, img_path in enumerate(msg["images"]):
                        with cols[i]:
                            render_image_tile(img_path, crop=False)
                if msg.get("report"):
                    with st.expander("📋 详细报告", expanded=False):
                        st.markdown(msg["report"])

    st.caption("同一照片最多澄清3次；线索足够或输入「直接给结果」时立即展示候选。只合并你的线索，不使用助手回复；找另一张照片请点击「＋ 新建对话」。")
    # 输入框
    if not fine_search_ready():
        st.info("精细检索需要你的 DeepSeek API 密钥及发送数据同意；请在侧边栏配置。全局搜索与照片库可直接使用。")
    user_input = st.chat_input("描述你想找的照片...", max_chars=1500, disabled=not fine_search_ready() or not image_paths)
    if user_input:
        maybe_auto_name_current_conversation(user_input)
        with st.chat_message("user"):
            st.write(user_input)
        run_assistant(user_input)
        st.rerun()

# ---------- 主界面：左侧选择模式 ----------
render_brand_sidebar()
render_mode_selector()
render_api_settings()

if st.session_state.mode == "全局搜索":
    st.sidebar.markdown(
        """
        <div class="ada-hint-list">
          <ul>
            <li>输入关键词或自然语言描述</li>
            <li>实时语义匹配全部照片库</li>
            <li>按相关性得分自动排序</li>
            <li>支持中英文混合查询</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown('<div class="ada-footer">AdaphotoRet v0.1 · 演示模式</div>', unsafe_allow_html=True)

if st.session_state.mode == "全局搜索":
    show_global_search()
elif st.session_state.mode == "照片库":
    show_photo_libraries()
else:
    show_fine_retrieval()
