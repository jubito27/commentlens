import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from video_core import (
    extract_video_id,
    fetch_youtube_transcript,
    summarize_video,
    translate_text,
)

load_dotenv()

app = FastAPI(title="CommentLens Video API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscriptRequest(BaseModel):
    url: str


class SummarizeRequest(BaseModel):
    url: Optional[str] = None
    transcript: Optional[str] = None


class TranslateRequest(BaseModel):
    text: str
    target_language: str = "Hinglish"


@app.get("/")
@app.get("/api/health")
def health():
    configured = [
        name
        for name, env_vars in [
            ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
            ("openrouter", ("OPENROUTER_API_KEY",)),
            ("grok", ("XAI_API_KEY", "GROK_API_KEY")),
            ("groq", ("GROQ_API_KEY",)),
            ("cohere", ("COHERE_API_KEY",)),
        ]
        if any(os.getenv(k) for k in env_vars)
    ]
    return {"status": "ok", "service": "commentlens-video", "configured_llms": configured}


@app.get("/api/transcript")
def transcript_get(url: str = Query(...)):
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or Video ID.")
    res = fetch_youtube_transcript(video_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error"))
    return res


@app.post("/api/transcript")
def transcript_post(req: TranscriptRequest):
    return transcript_get(req.url)


@app.post("/api/summarize")
def summarize_post(req: SummarizeRequest):
    res = summarize_video(url_or_id=req.url, transcript=req.transcript)
    if not res.get("success"):
        status = 400 if "Invalid" in res.get("error", "") or "Provide" in res.get("error", "") else 502
        raise HTTPException(status_code=status, detail=res.get("error"))
    return res


@app.get("/api/summarize")
def summarize_get(url: str = Query(...)):
    return summarize_post(SummarizeRequest(url=url))


@app.post("/api/translate")
def translate_post(req: TranslateRequest):
    res = translate_text(req.text, req.target_language)
    if not res.get("success"):
        raise HTTPException(status_code=502, detail=res.get("error"))
    return res
