from collections import Counter
from datetime import datetime
import html
import os
import re
from typing import Dict, List
import time
import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
CONFIDENCE_THRESHOLD = 0.45

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
    "have", "has", "had", "not", "but", "all", "can", "from", "they", "them", "their",
    "there", "here", "what", "when", "where", "which", "who", "why", "how", "will",
    "would", "could", "should", "just", "like", "very", "really", "actually", "about",
    "more", "most", "some", "any", "only", "also", "even", "still", "much", "many",
    "been", "being", "into", "over", "than", "then", "now", "out", "get", "got", "one",
    "two", "too", "its", "it's", "dont", "don't", "did", "does", "didn't", "cant",
    "can't", "won't", "you're", "i'm", "ive", "i've", "that's", "there's", "video",
    "videos", "youtube", "channel", "watch", "watching", "watched", "make", "made",
    "know", "think", "want", "need", "people", "time", "good", "great", "best", "nice",
    "well", "thanks", "thank", "please", "bro", "dude", "man", "guys", "sir", "https",
    "http", "www", "com",
    "hai", "hain", "tha", "thi", "the", "aur", "bhi", "nahi", "nhi", "kya", "bhai",
    "yaar", "kar", "karo", "kiya", "karna", "karke", "raha", "rahi", "rahe", "mein",
    "main", "mai", "liye", "bahut", "bohot", "ekdum", "wala", "wali", "wale", "hota",
    "hoti", "hote", "kuch", "sab", "abhi", "yeh", "woh", "tum", "aap", "hum", "mere",
    "mera", "meri", "tera", "teri", "apna", "apne", "uska", "uski", "unke", "iske",
    "baat", "log", "toh", "par", "koi", "sirf", "agar", "jab", "tab", "aise", "waise",
    "jaise", "kaun", "kitna", "itna", "hoga", "hogi", "honge", "gaya", "gayi", "gaye",
    "dekh", "dekha", "dekho", "bola", "bolo", "diya", "liya", "hua", "hui", "hue",
}


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


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_youtube_comments(video_id: str, max_comments: int = 150) -> List[Dict]:
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY secret missing on server.")

    items_out = []
    next_page_token = None
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    with requests.Session() as session:
        while len(items_out) < max_comments:
            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(items_out)),
                "order": "relevance",
                "textFormat": "plainText",
                "key": api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            resp = session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                err_msg = resp.json().get("error", {}).get("message", "YouTube API error")
                raise ValueError(f"YouTube API error ({resp.status_code}): {err_msg}")

            data = resp.json()
            for item in data.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                cleaned = clean_text(snippet.get("textDisplay", ""))
                if cleaned:
                    items_out.append(
                        {
                            "text": cleaned,
                            "published_at": snippet.get("publishedAt", ""),
                            "likes": int(snippet.get("likeCount", 0) or 0),
                        }
                    )

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

    return items_out



def extract_top_keywords(scored_items: List[Dict], limit: int = 8) -> List[Dict]:
    word_counts = Counter()
    word_sentiments = {}

    for item in scored_items:
        text_no_urls = re.sub(r"https?://\S+|www\.\S+", " ", item["comment"].lower())
        tokens = re.findall(r"[a-zA-Z]{3,}", text_no_urls)
        unique_words = {w for w in tokens if w not in STOPWORDS}
        sentiment = item["sentiment"]

        for w in unique_words:
            word_counts[w] += 1
            if w not in word_sentiments:
                word_sentiments[w] = {"Positive": 0, "Neutral": 0, "Negative": 0}
            word_sentiments[w][sentiment] += 1

    keywords = []
    for word, count in word_counts.most_common(limit):
        if count < 2 and len(scored_items) >= 15:
            continue
        breakdown = word_sentiments[word]
        dominant = max(breakdown, key=breakdown.get)
        keywords.append(
            {
                "word": word,
                "count": count,
                "sentiment": dominant,
                "breakdown": breakdown,
            }
        )
    return keywords


def build_timeline(scored_items: List[Dict]) -> List[Dict]:
    dated = []
    for item in scored_items:
        raw_ts = item.get("published_at", "")
        if not raw_ts:
            continue
        try:
            dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            dated.append((dt, item["sentiment"]))
        except ValueError:
            continue

    if not dated:
        return []

    dated.sort(key=lambda x: x[0])
    min_dt, max_dt = dated[0][0], dated[-1][0]
    span_seconds = (max_dt - min_dt).total_seconds()
    unique_days = sorted({dt.date() for dt, _ in dated})
    unique_months = sorted({(dt.year, dt.month) for dt, _ in dated})

    buckets = {}

    if span_seconds < 86400:
        for dt, sent in dated:
            key = dt.strftime("%Y-%m-%d %H")
            label = dt.strftime("%H:00")
            buckets.setdefault(key, {"label": label, "Positive": 0, "Neutral": 0, "Negative": 0})
            buckets[key][sent] += 1
        ordered_keys = sorted(buckets.keys())[-8:]
    elif len(unique_days) <= 7:
        for dt, sent in dated:
            key = dt.strftime("%Y-%m-%d")
            label = dt.strftime("%d %b")
            buckets.setdefault(key, {"label": label, "Positive": 0, "Neutral": 0, "Negative": 0})
            buckets[key][sent] += 1
        ordered_keys = sorted(buckets.keys())
    elif 2 <= len(unique_months) <= 8:
        for dt, sent in dated:
            key = dt.strftime("%Y-%m")
            label = dt.strftime("%b '%y")
            buckets.setdefault(key, {"label": label, "Positive": 0, "Neutral": 0, "Negative": 0})
            buckets[key][sent] += 1
        ordered_keys = sorted(buckets.keys())
    else:
        num_bins = 6
        bin_width = max(1.0, span_seconds / num_bins)
        same_year = min_dt.year == max_dt.year
        for dt, sent in dated:
            idx = min(num_bins - 1, int((dt - min_dt).total_seconds() / bin_width))
            bin_start = datetime.fromtimestamp(min_dt.timestamp() + idx * bin_width, tz=min_dt.tzinfo)
            label = bin_start.strftime("%d %b") if same_year else bin_start.strftime("%b '%y")
            buckets.setdefault(idx, {"label": label, "Positive": 0, "Neutral": 0, "Negative": 0})
            buckets[idx][sent] += 1
        ordered_keys = sorted(buckets.keys())

    timeline = []
    for key in ordered_keys:
        b = buckets[key]
        total = b["Positive"] + b["Neutral"] + b["Negative"]
        if total == 0:
            continue
        timeline.append(
            {
                "label": b["label"],
                "total": total,
                "Positive": b["Positive"],
                "Neutral": b["Neutral"],
                "Negative": b["Negative"],
                "pos_pct": round((b["Positive"] / total) * 100, 1),
                "neu_pct": round((b["Neutral"] / total) * 100, 1),
                "neg_pct": round((b["Negative"] / total) * 100, 1),
            }
        )
    return timeline


class CommentLensAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            num_threads = int(os.getenv("TORCH_THREADS", "4"))
            torch.set_num_threads(num_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # Reorder CardiffNLP [0:Neg, 1:Neu, 2:Pos] -> [0:Neu, 1:Pos, 2:Neg]
        with torch.no_grad():
            reorder = [1, 2, 0]
            self.model.classifier.out_proj.weight.copy_(self.model.classifier.out_proj.weight[reorder])
            self.model.classifier.out_proj.bias.copy_(self.model.classifier.out_proj.bias[reorder])

        self.model.config.id2label = {0: "Neutral", 1: "Positive", 2: "Negative"}
        self.model.eval()

        if self.device.type == "cpu":
            try:
                self.model = torch.ao.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass
        else:
            self.model = self.model.to(self.device)

        self.cache = {}

    @torch.inference_mode()
    def analyze(self, video_id: str, max_comments: int = 150) -> Dict:
        vid = extract_video_id(video_id)
        if not vid:
            return {"error": "Invalid YouTube URL or video ID.", "retryable": False}

        max_comments = max(10, min(int(max_comments), 200))
        cache_key = f"v2:{vid}:{max_comments}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        t0 = time.perf_counter()
        try:
            raw_items = fetch_youtube_comments(vid, max_comments)
        except ValueError as exc:
            return {"error": str(exc), "retryable": False}
        except Exception as exc:
            return {"error": str(exc), "retryable": True}
        yt_ms = round((time.perf_counter() - t0) * 1000)

        if not raw_items:
            return {"error": "No comments found for this video.", "retryable": False}

        t1 = time.perf_counter()
        # Identical comments ko sirf ek baar model se pass karo
        unique_texts = list(dict.fromkeys(item["text"] for item in raw_items))
        order = np.argsort([len(c) for c in unique_texts])
        batch_size = 64 if self.device.type == "cuda" else 32
        score_map = {}

        for start in range(0, len(unique_texts), batch_size):
            batch_idx = order[start : start + batch_size]
            batch_texts = [unique_texts[i] for i in batch_idx]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            ).to(self.device)

            if self.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**inputs).logits
            else:
                logits = self.model(**inputs).logits

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

            for text, p in zip(batch_texts, probs):
                pred_id = int(np.argmax(p))
                conf = float(p[pred_id])
                sentiment = "Neutral" if conf < CONFIDENCE_THRESHOLD else self.model.config.id2label[pred_id]
                score_map[text] = (sentiment, round(conf, 4))

        scored = []
        for item in raw_items:
            sentiment, conf = score_map[item["text"]]
            scored.append(
                {
                    "comment": item["text"],
                    "sentiment": sentiment,
                    "confidence": conf,
                    "published_at": item["published_at"],
                    "likes": item["likes"],
                }
            )
        inf_ms = round((time.perf_counter() - t1) * 1000)

        total = len(scored)
        counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for item in scored:
            counts[item["sentiment"]] += 1

        percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
        overall_mood = max(counts, key=counts.get)

        top_positive = [
            {"comment": x["comment"], "sentiment": x["sentiment"], "confidence": x["confidence"]}
            for x in sorted(
                [x for x in scored if x["sentiment"] == "Positive"],
                key=lambda x: x["confidence"],
                reverse=True,
            )[:3]
        ]
        top_negative = [
            {"comment": x["comment"], "sentiment": x["sentiment"], "confidence": x["confidence"]}
            for x in sorted(
                [x for x in scored if x["sentiment"] == "Negative"],
                key=lambda x: x["confidence"],
                reverse=True,
            )[:3]
        ]

        response = {
            "video_id": vid,
            "total_comments": total,
            "overall_mood": overall_mood,
            "counts": counts,
            "percentages": percentages,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "top_keywords": extract_top_keywords(scored, limit=8),
            "timeline": build_timeline(scored),
            "timing_ms": {"youtube": yt_ms, "inference": inf_ms},
        }
        self.cache[cache_key] = response
        return response