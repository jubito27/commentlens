import html
import os
import re
from typing import List

import gradio as gr
import numpy as np
import requests
import spaces
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
CONFIDENCE_THRESHOLD = 0.45

# Load model and reorder CardiffNLP head [0:Neg, 1:Neu, 2:Pos] -> [0:Neu, 1:Pos, 2:Neg]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

with torch.no_grad():
    reorder = [1, 2, 0]
    model.classifier.out_proj.weight.copy_(model.classifier.out_proj.weight[reorder])
    model.classifier.out_proj.bias.copy_(model.classifier.out_proj.bias[reorder])

model.config.id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
model = model.to("cuda")
model.eval()

VIDEO_CACHE = {}


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_video_id(value: str) -> str:
    value = (value or "").strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", value):
        return value
    patterns = [
        r"[?&]v=([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/|youtube\.com/(?:shorts|embed|live)/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return ""

def fetch_youtube_comments(video_id: str, max_comments: int = 150) -> List[str]:
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY secret missing in Space Settings.")

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
            err_msg = resp.json().get("error", {}).get("message", "YouTube API error")
            raise ValueError(f"YouTube API error ({resp.status_code}): {err_msg}")

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


# Keep duration=15 (instead of default 60s) so each call uses minimal daily ZeroGPU quota
@spaces.GPU(duration=15)
@torch.inference_mode()
def analyze_video(video_id: str, max_comments: int = 150):
    video_id = extract_video_id(video_id)
    if not video_id:
        return {"error": "Please provide a valid YouTube video URL or 11-character video ID."}

    max_comments = max(10, min(int(max_comments), 200))
    cache_key = f"{video_id}:{max_comments}"
    if cache_key in VIDEO_CACHE:
        return VIDEO_CACHE[cache_key]

    try:
        comments = fetch_youtube_comments(video_id, max_comments)
    except Exception as exc:
        return {"error": str(exc)}

    if not comments:
        return {"error": "No comments found for this video."}

    order = np.argsort([len(c) for c in comments])
    scored = [None] * len(comments)
    batch_size = 64

    for start in range(0, len(comments), batch_size):
        batch_idx = order[start : start + batch_size]
        batch_texts = [comments[i] for i in batch_idx]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to("cuda")

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(**inputs).logits

        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        for idx, text, p in zip(batch_idx, batch_texts, probs):
            pred_id = int(np.argmax(p))
            conf = float(p[pred_id])
            sentiment = "Neutral" if conf < CONFIDENCE_THRESHOLD else model.config.id2label[pred_id]
            scored[idx] = {
                "comment": text,
                "sentiment": sentiment,
                "confidence": round(conf, 4),
            }

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
    top_negative = sorted(
            [x for x in scored if x["sentiment"] == "Neutral"],
            key=lambda x: x["confidence"],
            reverse=True,
        )[:3]
    response = {
        "video_id": video_id,
        "total_comments": total,
        "overall_mood": overall_mood,
        "counts": counts,
        "percentages": percentages,
        "top_positive": top_positive,
        "top_negative": top_negative,
    }
    VIDEO_CACHE[cache_key] = response
    return response


with gr.Blocks(title="CommentLens API") as demo:
    gr.Markdown("## CommentLens Sentiment API\nEnter a YouTube Video ID to test live sentiment analysis.")
    with gr.Row():
        vid_input = gr.Textbox(label="YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
        max_input = gr.Slider(minimum=20, maximum=200, value=150, step=10, label="Max Comments")
    run_btn = gr.Button("Analyze Comments", variant="primary")
    output_json = gr.JSON(label="Sentiment Analysis Result")

    run_btn.click(
        fn=analyze_video,
        inputs=[vid_input, max_input],
        outputs=output_json,
        api_name="analyze",
    )

if __name__ == "__main__":
    demo.launch()