"""Photo-library metadata only: no images, credentials, chat, or shared user files.

Public deployments persist in the visitor's browser; this module validates the
small JSON document and coordinates restore/write acknowledgments per session.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

FORMAT = "adaphotoret.photo-library"
VERSION = 1
MAX_BYTES = 2 * 1024 * 1024


class LibraryDataError(ValueError):
    pass


def empty_library():
    return {"photo_tags": {}, "libraries": {}}


def _text(value, limit=200):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise LibraryDataError("名称、标签或照片引用无效。")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise LibraryDataError("名称、标签或照片引用包含无效字符。") from error
    return value


def _strings(value, *, limit=200, count=5000):
    if not isinstance(value, list) or len(value) > count:
        raise LibraryDataError("照片或标签列表无效或过大。")
    return list(dict.fromkeys(_text(item, limit) for item in value))


def normalize_library(value):
    """Validate without truncating/dropping data; migrate old aliases/sections."""
    if not isinstance(value, dict) or set(value) - {"photo_tags", "libraries"}:
        raise LibraryDataError("不是照片库配置；只接受照片引用、相册和标签。")
    photo_tags, libraries = value.get("photo_tags", {}), value.get("libraries", {})
    if not isinstance(photo_tags, dict) or not isinstance(libraries, dict):
        raise LibraryDataError("照片库结构无效。")
    if len(photo_tags) > 10000 or len(libraries) > 1000:
        raise LibraryDataError("照片库配置过大。")
    result = empty_library()
    for path, tags in photo_tags.items():
        result["photo_tags"][_text(path, 2048)] = _strings(tags)
    for name, config in libraries.items():
        _text(name)
        if not isinstance(config, dict) or set(config) - {"photos", "tags", "aliases", "sections"}:
            raise LibraryDataError("相册结构无效。")
        tags = _strings(config.get("tags", [])) + _strings(config.get("aliases", []))
        sections = config.get("sections", {})
        if not isinstance(sections, dict) or len(sections) > 1000:
            raise LibraryDataError("旧版分组结构无效。")
        for aliases in sections.values():
            tags.extend(_strings(aliases))
        result["libraries"][name] = {
            "photos": _strings(config.get("photos", []), limit=2048, count=10000),
            "tags": _strings(list(dict.fromkeys(tags))),
        }
    # Reserve the small version/format envelope used by backups and the bridge.
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_BYTES - 512:
        raise LibraryDataError("照片库配置超过 2 MB；原有保存内容未更改。")
    return result


def export_library(data):
    return json.dumps({"format": FORMAT, "version": VERSION, "data": normalize_library(data)},
                      ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def export_legacy_library(value):
    """Recovery-only export for pre-upgrade, over-limit labels; never persist it.

    Keep the same metadata allowlist, but preserve old sizes/Unicode verbatim so
    a user can repair the backup instead of losing an already-open session.
    """
    def string_list(items):
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise LibraryDataError("旧会话照片库结构无法安全导出。")
    if not isinstance(value, dict) or set(value) - {"photo_tags", "libraries"}:
        raise LibraryDataError("旧会话照片库结构无法安全导出。")
    tags, libraries = value.get("photo_tags", {}), value.get("libraries", {})
    if not isinstance(tags, dict) or not isinstance(libraries, dict):
        raise LibraryDataError("旧会话照片库结构无法安全导出。")
    for path, items in tags.items():
        if not isinstance(path, str):
            raise LibraryDataError("旧会话照片引用无效。")
        string_list(items)
    for name, config in libraries.items():
        if not isinstance(name, str) or not isinstance(config, dict) or set(config) - {"photos", "tags", "aliases", "sections"}:
            raise LibraryDataError("旧会话相册结构无法安全导出。")
        for field in ("photos", "tags", "aliases"):
            string_list(config.get(field, []))
        sections = config.get("sections", {})
        if not isinstance(sections, dict) or not all(isinstance(key, str) for key in sections):
            raise LibraryDataError("旧会话分组结构无效。")
        for items in sections.values():
            string_list(items)
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def import_library(content):
    if not isinstance(content, (bytes, str)) or len(content) > MAX_BYTES:
        raise LibraryDataError("备份文件无效或超过 2 MB。")
    try:
        value = json.loads(content)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise LibraryDataError("备份不是有效的 JSON 文件。") from error
    if isinstance(value, dict) and "format" in value:
        if (value.get("format") != FORMAT or type(value.get("version")) is not int
                or value["version"] != VERSION or set(value) != {"format", "version", "data"}):
            raise LibraryDataError("备份版本或格式不受支持；原有内容未更改。")
        value = value["data"]
    return normalize_library(value)


def merge_libraries(current, incoming):
    """Import is additive: a same-name album never replaces existing photos."""
    result, incoming = normalize_library(current), normalize_library(incoming)
    for path, tags in incoming["photo_tags"].items():
        result["photo_tags"][path] = list(dict.fromkeys(result["photo_tags"].get(path, []) + tags))
    for name, config in incoming["libraries"].items():
        dest = result["libraries"].setdefault(name, {"photos": [], "tags": []})
        for field in ("photos", "tags"):
            dest[field] = list(dict.fromkeys(dest[field] + config[field]))
    return normalize_library(result)


def load_local_library(path):
    path = Path(path)
    if not path.exists():
        return empty_library()
    if path.stat().st_size > MAX_BYTES:
        raise LibraryDataError("本地照片库配置超过 2 MB。")
    return import_library(path.read_bytes())


def save_local_library(path, data):
    """Only called in explicit local mode; replace atomically, not partial JSON."""
    data = normalize_library(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as target:
            name = target.name
            json.dump(data, target, ensure_ascii=False, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def new_browser_state():
    return {"phase": "loading", "revision": None, "request": {"op": "load", "id": uuid4().hex},
            "pending": None, "last_event": None, "error": None, "saved": False}


def request_browser_reload(state):
    state.update(phase="loading", request={"op": "load", "id": uuid4().hex}, error=None)


def queue_browser_save(state, data):
    if state["phase"] != "ready":
        raise LibraryDataError("尚未完成读取或保存，请等待后再修改照片库。")
    validated = normalize_library(data)
    state.update(phase="saving", error=None, saved=False, pending=deepcopy(validated),
                 request={"op": "save", "id": uuid4().hex, "base_revision": state["revision"], "data": validated})


def consume_browser_event(state, event):
    """Return a new display document only after restore or an acknowledged write."""
    if not isinstance(event, dict) or not event.get("event_id") or state["last_event"] == event["event_id"]:
        return None
    state["last_event"] = event["event_id"]
    kind = event.get("kind")
    external = kind == "external"
    if external and (state["phase"] != "ready" or event.get("request_id") != state["request"].get("id")):
        return None  # pending writes will independently detect their revision conflict
    if not external and event.get("request_id") != state["request"].get("id"):
        return None
    if kind in {"loaded", "saved", "external"}:
        try:
            value = normalize_library(state["pending"] if kind == "saved" else event.get("data"))
            revision = event.get("revision")
            if revision is not None and (not isinstance(revision, str) or len(revision) != 64):
                raise LibraryDataError("浏览器保存版本无效。")
        except (LibraryDataError, TypeError):
            state.update(phase="error", error="invalid", request={"op": "watch"})
            return None
        request = state["request"] if external else {"op": "watch", "id": uuid4().hex}
        state.update(phase="ready", revision=revision, pending=None, error=None,
                     saved=(state["saved"] if external and revision == state["revision"] else kind == "saved"), request=request)
        return value
    if kind in {"error", "conflict"}:
        state.update(phase="conflict" if kind == "conflict" else "error",
                     error="conflict" if kind == "conflict" else event.get("code", "unavailable"),
                     request={"op": "watch"})
    return None
