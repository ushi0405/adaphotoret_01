
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from AdaphotoRet_run import search_photos

app = FastAPI(title="AdaphotoRet API")

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    top1, top2, top3, report, table_data = search_photos(request.query)
    return {
        "top1": top1,
        "top2": top2,
        "top3": top3,
        "report": report,
        "table": table_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)