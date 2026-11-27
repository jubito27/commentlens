import os
import html
import re
from typing import List, Optional
import numpy as np
import requests
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
CONFIDENCE_THRESHOLD = 0.45

torch.set_num_threads(2)

# Startup par model ek hi baar load hoga
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# CardiffNLP [0:Neg, 1:Neu, 2:Pos] -> tumhara [0:Neu, 1:Pos, 2:Neg]
with torch.no_grad():
    reorder = [1, 2, 0]
    model.classifier.out_proj.weight.copy_(model.classifier.out_proj.weight[reorder])
    model.classifier.out_proj.bias.copy_(model.classifier.out_proj.bias[reorder])

model.config.id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
model.eval()

app = FastAPI(title="CommentLens API")

# Chrome/Firefox extension se requests allow karne ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache taaki same video dobara analyze karne par instant result mile
VIDEO_CACHE = {}


class AnalyzeRequest(BaseModel):
    video_id: Optional[str] = None
    comments: Optional[List[str]] = None
    max_comments: int = 150


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_youtube_comments(video_id: str, max_comments: int = 150) -> List[str]:
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY secret missing on server.")

    comments = []
    next_page_token = None
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "order": "relevance",
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.json().get("error", {}).get("message", "YouTube API error"))

        data = resp.json()
        for item in data.get("items", []):
            raw = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            cleaned = clean_text(raw)
            if cleaned:
                comments.append(cleaned)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments


@torch.inference_mode()
def score_comments(comments: List[str], batch_size: int = 32):
    order = np.argsort([len(c) for c in comments])
    results = [None] * len(comments)

    for start in range(0, len(comments), batch_size):
        batch_idx = order[start : start + batch_size]
        batch_texts = [comments[i] for i in batch_idx]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).numpy()

        for idx, text, p in zip(batch_idx, batch_texts, probs):
            pred_id = int(np.argmax(p))
            conf = float(p[pred_id])
            # Low-confidence Hinglish/borderline comments ko Neutral guardrail par bhejo
            sentiment = "Neutral" if conf < CONFIDENCE_THRESHOLD else model.config.id2label[pred_id]
            results[idx] = {
                "comment": text,
                "sentiment": sentiment,
                "confidence": round(conf, 4),
            }

    return results


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if req.video_id and req.video_id in VIDEO_CACHE:
        return VIDEO_CACHE[req.video_id]

    if req.comments:
        comments = [clean_text(c) for c in req.comments if clean_text(c)]
    elif req.video_id:
        comments = fetch_youtube_comments(req.video_id, min(req.max_comments, 200))
    else:
        raise HTTPException(status_code=400, detail="Send either video_id or comments list.")

    if not comments:
        raise HTTPException(status_code=404, detail="No comments found for this video.")

    scored = score_comments(comments)
    total = len(scored)
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for item in scored:
        counts[item["sentiment"]] += 1

    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    overall_mood = max(counts, key=counts.get)

    top_positive = sorted(
        [x for x in scored if x["sentiment"] == "Positive"],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:3]
    top_negative = sorted(
        [x for x in scored if x["sentiment"] == "Negative"],
        key=lambda x: x["confidence"],
        reverse=True,
    )[:3]

    response = {
        "video_id": req.video_id,
        "total_comments": total,
        "overall_mood": overall_mood,
        "counts": counts,
        "percentages": percentages,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }

    if req.video_id:
        VIDEO_CACHE[req.video_id] = response

    return response