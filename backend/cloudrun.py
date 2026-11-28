from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import CommentLensAnalyzer

analyzer = CommentLensAnalyzer()
app = FastAPI(title="CommentLens Cloud Run API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    video_id: str
    max_comments: int = 150


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "provider": "cloudrun"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return analyzer.analyze(req.video_id, req.max_comments)
