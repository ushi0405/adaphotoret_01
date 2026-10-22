"""Optional legacy HTTP interface. Streamlit is the supported hosted demo.

No server API key is used unless local mode is explicitly enabled. Public callers
may pass X-DeepSeek-API-Key and X-Allow-External-Processing: true per request.
Only catalogued images are served; the data directory is not exposed wholesale.
"""
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from AdaphotoRet_run import metadata, rank_all_photos, resolve_photo_path, search_photos
from llm_explainer import get_deepseek_client
from photo_query import build_photo_query

app = FastAPI(title="AdaphotoRet API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-DeepSeek-API-Key", "X-Allow-External-Processing"],
)

class SearchRequest(BaseModel):
    query: str = Field(max_length=2000)

class SearchMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(max_length=2000)


class FineSearchRequest(SearchRequest):
    # Stateless: callers send only the current target's history, or [] to reset.
    messages: list[SearchMessage] = Field(default_factory=list, max_length=100)


def to_url(path):
    return "/photos/" + quote(path.removeprefix("data/"), safe="/") if path else ""

@lru_cache(maxsize=32)
def heic_preview(path: str, mtime: float):
    """Browser-compatible JPEG response; keep the original HEIC unchanged."""
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener
    register_heif_opener()
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((1600, 1600))
        output = BytesIO()
        image.save(output, "JPEG", quality=85)
        return output.getvalue()

@app.get("/photos/{relative_path:path}")
def photo(relative_path: str):
    key = "data/" + relative_path
    if key not in metadata:
        raise HTTPException(404, "Photo not in the selected catalog")
    file = Path(resolve_photo_path(key))
    if not file.is_file():
        raise HTTPException(404, "Photo unavailable")
    if file.suffix.lower() in {".heic", ".heif"}:
        try:
            return Response(heic_preview(str(file), file.stat().st_mtime), media_type="image/jpeg")
        except (OSError, ValueError):
            raise HTTPException(422, "Photo could not be decoded") from None
    return FileResponse(file)

@app.post("/api/search")
def fine_search(
    req: FineSearchRequest,
    x_deepseek_api_key: str | None = Header(default=None),
    x_allow_external_processing: str | None = Header(default=None),
):
    client = None
    if x_deepseek_api_key:
        if x_allow_external_processing != "true":
            raise HTTPException(400, "External-processing consent is required")
        client = get_deepseek_client(x_deepseek_api_key, allow_environment=False)
    elif os.getenv("ADAPHOTORET_LOCAL_MODE") == "1" and os.getenv("DEEPSEEK_API_KEY"):
        client = get_deepseek_client()
    try:
        p1, p2, p3, report, table, _ = search_photos(
            build_photo_query([m.model_dump() for m in req.messages], req.query),
            client=client, include_explanation=client is not None
        )
    except Exception:
        raise HTTPException(503, "Search resources unavailable; check model setup") from None
    finally:
        if client is not None:
            client.close()
    return {"images": [to_url(p) for p in (p1, p2, p3) if p], "report": report, "table": table}

@app.post("/api/global-search")
def global_search(req: SearchRequest):
    try:
        paths = rank_all_photos(req.query) if req.query.strip() else sorted(metadata)
    except Exception:
        raise HTTPException(503, "Search resources unavailable; check model setup") from None
    return {"images": [to_url(p) for p in paths]}

@app.get("/api/all-photos")
def all_photos():
    return {"photos": [to_url(p) for p in sorted(metadata)]}

@app.get("/")
def root():
    return {"message": "AdaphotoRet API is running", "catalog_size": len(metadata)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
