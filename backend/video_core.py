import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import requests
from cachetools import TTLCache
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# In-memory TTL caches (Render 512 MB RAM friendly)
TRANSCRIPT_CACHE: TTLCache = TTLCache(maxsize=200, ttl=86400)  # 24 hours
SUMMARY_CACHE: TTLCache = TTLCache(maxsize=200, ttl=86400)     # 24 hours

# Optional Redis (sirf tab use hoga jab REDIS_URL set ho)
_redis_client = None
if os.getenv("REDIS_URL"):
    try:
        import redis
        _redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    except Exception:
        _redis_client = None

_offline_model = None

SUMMARY_SYSTEM_PROMPT = (
    "You are an expert AI note-taker and summarizer. "
    "Provide clear, structured, and insightful summaries in clean Markdown."
)

SUMMARY_USER_TEMPLATE = """Analyze the following YouTube video transcript and generate a structured summary:

### Requirements:
1. **Core Summary**: 2-3 concise paragraphs summarizing the main theme.
2. **Key Takeaways**: Bullet points highlighting key learnings, tools, or conclusions.
3. **Actionable Insights**: Any direct recommendations or next steps mentioned.

Transcript:
{transcript}
"""

TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional multilingual translator. Translate the provided text accurately "
    "while preserving markdown formatting, bullet points, headings, tone, and technical precision. "
    "Do not add any conversational remarks; return only the translated text."
)

TRANSLATE_USER_TEMPLATE = """Translate the following text into **{target_language}**.

Text to translate:
{text}
"""


def extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    val = url_or_id.strip()
    if len(val) == 11 and re.fullmatch(r"[0-9A-Za-z_-]{11}", val):
        return val
    patterns = [
        r"[?&]v=([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/|youtube\.com\/(?:shorts|embed|live|v)\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, val)
        if match:
            return match.group(1)
    return None


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0 else f"{mins:02d}:{secs:02d}"


def _cache_get(cache_obj: TTLCache, key: str) -> Optional[Dict[str, Any]]:
    if key in cache_obj:
        return cache_obj[key]
    if _redis_client is not None:
        try:
            raw = _redis_client.get(key)
            if raw:
                data = json.loads(raw)
                cache_obj[key] = data
                return data
        except Exception:
            pass
    return None


def _cache_set(cache_obj: TTLCache, key: str, value: Dict[str, Any], ttl: int = 86400) -> None:
    cache_obj[key] = value
    if _redis_client is not None:
        try:
            _redis_client.set(key, json.dumps(value), ex=ttl)
        except Exception:
            pass


def fetch_youtube_transcript(video_id: str) -> Dict[str, Any]:
    cache_key = f"transcript:{video_id}"
    cached = _cache_get(TRANSCRIPT_CACHE, cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        # youtube-transcript-api v1.x (instance) + v0.x (classmethod) dono support
        api = YouTubeTranscriptApi()
        if hasattr(api, "list"):
            transcript_list = api.list(video_id)
        else:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript_obj = transcript_list.find_transcript(["en", "en-US", "en-GB", "hi"])
        except NoTranscriptFound:
            transcript_obj = next(iter(transcript_list))

        raw_transcript = transcript_obj.fetch()
        segments = []
        full_text_parts = []

        for item in raw_transcript:
            if hasattr(item, "text"):
                text_content = item.text
                start_time = float(getattr(item, "start", 0.0))
            elif isinstance(item, dict):
                text_content = item.get("text", "")
                start_time = float(item.get("start", 0.0))
            else:
                text_content = str(item)
                start_time = 0.0

            clean_text = re.sub(r"\s+", " ", text_content).strip()
            if clean_text:
                full_text_parts.append(clean_text)
                segments.append({
                    "start": round(start_time, 2),
                    "timestamp": format_timestamp(start_time),
                    "text": clean_text,
                })

        payload = {
            "success": True,
            "video_id": video_id,
            "language": transcript_obj.language_code,
            "full_transcript": " ".join(full_text_parts),
            "segments": segments,
        }
        _cache_set(TRANSCRIPT_CACHE, cache_key, payload)
        return {**payload, "cached": False}

    except TranscriptsDisabled:
        return {"success": False, "error": "Transcripts are disabled for this video."}
    except Exception as exc:
        return {"success": False, "error": f"Transcript fetch failed: {str(exc)}"}


# ---------------------------------------------------------
# 5-Tier LLM Fallback Engine
# 1. Gemini -> 2. OpenRouter -> 3. Grok / Groq -> 4. Cohere -> 5. Offline
# ---------------------------------------------------------

def _call_gemini(system_prompt: str, user_prompt: str, timeout: int = 35) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
    }
    resp = requests.post(url, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_openai_compat(
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = 35,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    body = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_openrouter(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not configured")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    return _call_openai_compat(
        url="https://openrouter.ai/api/v1/chat/completions",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_headers={
            "HTTP-Referer": "https://commentlens.app",
            "X-Title": "CommentLens",
        },
    )


def _call_xai_grok(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY/GROK_API_KEY not configured")
    model = os.getenv("GROK_MODEL", "grok-3-mini")
    return _call_openai_compat(
        url="https://api.x.ai/v1/chat/completions",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return _call_openai_compat(
        url="https://api.groq.com/openai/v1/chat/completions",
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _call_cohere(system_prompt: str, user_prompt: str, timeout: int = 35) -> str:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not configured")
    model = os.getenv("COHERE_MODEL", "command-r-08-2024")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post("https://api.cohere.com/v2/chat", headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:250]}")
    data = resp.json()
    return data["message"]["content"][0]["text"].strip()


def _call_offline(system_prompt: str, user_prompt: str) -> str:
    global _offline_model
    model_path = os.getenv("OFFLINE_MODEL_PATH", "/mnt/WindowsD/offline_model")
    if not os.path.isdir(model_path):
        raise RuntimeError("Offline model path not available on this host")

    import offline  # Lazy import so cloud servers never need local ML deps

    if _offline_model is None:
        _offline_model = offline.load_offline_chat_model(
            model_source=model_path,
            max_new_tokens=2048,
            temperature=0.3,
        )
    response = _offline_model.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
    return response.content if hasattr(response, "content") else str(response)


def generate_with_fallback(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    providers = [
        ("gemini", _call_gemini),
        ("openrouter", _call_openrouter),
        ("grok", _call_xai_grok),
        ("groq", _call_groq),
        ("cohere", _call_cohere),
        ("offline", _call_offline),
    ]
    errors = []

    for name, fn in providers:
        try:
            output = fn(system_prompt, user_prompt)
            if output:
                return output, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError("All LLM providers failed -> " + " | ".join(errors))


def summarize_video(url_or_id: Optional[str] = None, transcript: Optional[str] = None) -> Dict[str, Any]:
    video_id = extract_video_id(url_or_id) if url_or_id else None

    if video_id:
        cached = _cache_get(SUMMARY_CACHE, f"summary:{video_id}")
        if cached:
            return {**cached, "cached": True}

    if not transcript:
        if not video_id:
            return {"success": False, "error": "Provide a valid YouTube 'url' or raw 'transcript'."}
        t_res = fetch_youtube_transcript(video_id)
        if not t_res.get("success"):
            return t_res
        transcript = t_res["full_transcript"]

    truncated = transcript[:40000]
    try:
        summary_text, provider = generate_with_fallback(
            SUMMARY_SYSTEM_PROMPT,
            SUMMARY_USER_TEMPLATE.format(transcript=truncated),
        )
        payload = {
            "success": True,
            "video_id": video_id,
            "summary": summary_text,
            "llm_provider": provider,
        }
        if video_id:
            _cache_set(SUMMARY_CACHE, f"summary:{video_id}", payload)
        return {**payload, "cached": False}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def translate_text(text: str, target_language: str = "Hinglish") -> Dict[str, Any]:
    if not text or not text.strip():
        return {"success": False, "error": "Missing 'text' field to translate."}

    lang = (target_language or "Hinglish").strip()
    try:
        translated, provider = generate_with_fallback(
            TRANSLATE_SYSTEM_PROMPT,
            TRANSLATE_USER_TEMPLATE.format(target_language=lang, text=text[:30000]),
        )
        return {
            "success": True,
            "target_language": lang,
            "translated_text": translated,
            "llm_provider": provider,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
