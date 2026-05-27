from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import glob
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

os.environ["DEEPSEEK_API_KEY"] = "sk-fc83463934dd4c1c8e8f0b47f266938c"

from AdaphotoRet_run import (
    search_photos,
    rank_all_photos,
    metadata,
)

app = FastAPI()

# ========== 手动 CORS 中间件（强制添加头） ==========
class ForceCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

app.add_middleware(ForceCORSMiddleware)

# 处理 OPTIONS 预检请求（关键！）
@app.options("/{path:path}")
async def preflight_handler(path: str):
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )
# ============================================

PHOTOS_BASE = r"C:\Users\Lenovo\Desktop\AdaphotoRet_523\data"
app.mount("/photos", StaticFiles(directory=PHOTOS_BASE), name="photos")

def to_url(abs_path: str) -> str:
    if not abs_path:
        return ""
    rel = os.path.relpath(abs_path, PHOTOS_BASE).replace(os.sep, '/')
    if rel.lower().endswith('.heic'):
        rel = rel[:-5] + '.jpg'
    return f"/photos/{rel}"

class SearchRequest(BaseModel):
    query: str

@app.post("/api/search")
async def fine_search(req: SearchRequest):
    top1, top2, top3, report, table, top_results = search_photos(req.query)
    images = [to_url(p) for p in (top1, top2, top3) if p is not None]
    images = [url.replace('.heic', '.jpg').replace('.HEIC', '.jpg') for url in images]
    report = report.encode('utf-8').decode('utf-8')
    return {"images": images, "report": report, "table": table}

@app.post("/api/global-search")
async def global_search(req: SearchRequest):
    query = req.query
    print(f"[全局检索] 收到查询: {query}")
    if not query.strip():
        all_paths = list(metadata.keys())
        all_paths.sort()
        urls = [to_url(p) for p in all_paths]
        return {"images": urls}
    ranked_paths = rank_all_photos(query)
    urls = [to_url(p) for p in ranked_paths]
    urls = [url.replace('.heic', '.jpg').replace('.HEIC', '.jpg') for url in urls]
    return {"images": urls}

@app.get("/api/all-photos")
async def get_all_photos():
    exts = ('*.jpg', '*.jpeg', 'png', '*.JPG', '*.JPEG', '*.PNG')
    photo_paths = []
    for ext in exts:
        photo_paths.extend(glob.glob(os.path.join(PHOTOS_BASE, '**', ext), recursive=True))
    photo_paths = sorted(set(photo_paths))
    urls = [f"/photos/{os.path.relpath(p, PHOTOS_BASE).replace(os.sep, '/')}" for p in photo_paths]
    return {"photos": urls}

@app.get("/")
def root():
    return {"message": "AdaphotoRet API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)