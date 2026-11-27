const SPACE_URL = "https://abhishek2775-commentlens-api.hf.space";
const extApi = typeof browser !== "undefined" ? browser : chrome;

const videoBadge = document.getElementById("video-badge");
const maxCommentsSelect = document.getElementById("max-comments");
const analyzeBtn = document.getElementById("analyze-btn");
const refreshBtn = document.getElementById("refresh-btn");
const statusBox = document.getElementById("status-box");
const spinner = document.getElementById("spinner");
const statusText = document.getElementById("status-text");
const resultsEl = document.getElementById("results");

let currentVideoId = null;

function extractVideoId(url) {
  if (!url) return null;
  const patterns = [
    /[?&]v=([a-zA-Z0-9_-]{11})/,
    /(?:youtu\.be\/|youtube\.com\/(?:shorts|embed|live)\/)([a-zA-Z0-9_-]{11})/
  ];
  for (const regex of patterns) {
    const match = url.match(regex);
    if (match) return match[1];
  }
  return null;
}

function setStatus(message, { loading = false, isError = false } = {}) {
  statusBox.classList.remove("hidden", "error");
  if (isError) statusBox.classList.add("error");
  spinner.classList.toggle("hidden", !loading);
  statusText.textContent = message;
}

function hideStatus() {
  statusBox.classList.add("hidden");
  spinner.classList.add("hidden");
}

function renderCommentList(listEl, items, emptyText) {
  listEl.replaceChildren();
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty-msg";
    li.textContent = emptyText;
    listEl.appendChild(li);
    return;
  }

  for (const item of items) {
    const li = document.createElement("li");
    li.className = "comment-item";

    const textNode = document.createElement("span");
    textNode.textContent = item.comment;

    const confNode = document.createElement("span");
    confNode.className = "comment-conf";
    confNode.textContent = `Confidence: ${Math.round(item.confidence * 100)}%`;

    li.appendChild(textNode);
    li.appendChild(confNode);
    listEl.appendChild(li);
  }
}

function renderResults(data, fromCache = false) {
  hideStatus();
  resultsEl.classList.remove("hidden");
  refreshBtn.hidden = false;

  const moodEl = document.getElementById("overall-mood");
  moodEl.textContent = data.overall_mood || "Neutral";
  moodEl.className = "mood-value";
  if (data.overall_mood === "Positive") moodEl.classList.add("mood-positive");
  else if (data.overall_mood === "Negative") moodEl.classList.add("mood-negative");
  else moodEl.classList.add("mood-neutral");

  document.getElementById("total-comments").textContent = `${data.total_comments} comments`;
  document.getElementById("cache-tag").classList.toggle("hidden", !fromCache);

  const pct = data.percentages || { Positive: 0, Neutral: 0, Negative: 0 };
  const cnt = data.counts || { Positive: 0, Neutral: 0, Negative: 0 };

  document.getElementById("pos-stat").textContent = `${pct.Positive}% (${cnt.Positive})`;
  document.getElementById("neu-stat").textContent = `${pct.Neutral}% (${cnt.Neutral})`;
  document.getElementById("neg-stat").textContent = `${pct.Negative}% (${cnt.Negative})`;

  document.getElementById("pos-bar").style.width = `${pct.Positive}%`;
  document.getElementById("neu-bar").style.width = `${pct.Neutral}%`;
  document.getElementById("neg-bar").style.width = `${pct.Negative}%`;

  renderCommentList(
    document.getElementById("top-positive-list"),
    data.top_positive,
    "No strongly positive comments detected."
  );
  renderCommentList(
    document.getElementById("top-negative-list"),
    data.top_negative,
    "No strongly negative comments detected."
  );
}

async function callGradioAnalyze(videoId, maxComments) {
  const payload = JSON.stringify({ data: [videoId, Number(maxComments)] });

  // Try Gradio 5 endpoint first; fall back to Gradio 4 path if needed
  let endpoint = `${SPACE_URL}/gradio_api/call/analyze`;
  let postResp = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload
  });

  if (postResp.status === 404) {
    endpoint = `${SPACE_URL}/call/analyze`;
    postResp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    });
  }

  if (!postResp.ok) {
    throw new Error(`API request failed (HTTP ${postResp.status}). Ensure the Hugging Face Space is awake.`);
  }

  const { event_id: eventId } = await postResp.json();
  if (!eventId) {
    throw new Error("Backend did not return an event_id.");
  }

  const streamResp = await fetch(`${endpoint}/${eventId}`);
  if (!streamResp.ok) {
    throw new Error(`Failed to fetch analysis result (HTTP ${streamResp.status}).`);
  }

  const rawText = await streamResp.text();
  const blocks = rawText.split(/\r?\n\r?\n/);

  for (const block of blocks) {
    const lines = block.split(/\r?\n/);
    let eventType = "";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }

    if (eventType === "error") {
      const detail = dataLines.join(" ");
      throw new Error(detail || "ZeroGPU error or daily quota limit reached.");
    }

    if (eventType === "complete" && dataLines.length > 0) {
      const parsed = JSON.parse(dataLines.join("\n"));
      const result = Array.isArray(parsed) ? parsed[0] : parsed;
      if (result && result.error) {
        throw new Error(result.error);
      }
      return result;
    }
  }

  throw new Error("Incomplete response received from Hugging Face Space.");
}

async function runAnalysis(forceRefresh = false) {
  if (!currentVideoId) return;

  const maxComments = maxCommentsSelect.value;
  const cacheKey = `cl_${currentVideoId}_${maxComments}`;

  if (!forceRefresh) {
    const stored = await extApi.storage.local.get(cacheKey);
    if (stored && stored[cacheKey]) {
      renderResults(stored[cacheKey], true);
      return;
    }
  }

  analyzeBtn.disabled = true;
  refreshBtn.hidden = true;
  resultsEl.classList.add("hidden");
  setStatus("Fetching YouTube comments and running sentiment analysis…", { loading: true });

  try {
    const data = await callGradioAnalyze(currentVideoId, maxComments);
    await extApi.storage.local.set({ [cacheKey]: data });
    renderResults(data, false);
  } catch (err) {
    setStatus(err.message || "Something went wrong while contacting the API.", { isError: true });
  } finally {
    analyzeBtn.disabled = false;
  }
}

async function init() {
  try {
    const tabs = await extApi.tabs.query({ active: true, currentWindow: true });
    const activeTab = tabs[0];
    currentVideoId = extractVideoId(activeTab?.url);

    if (!currentVideoId) {
      videoBadge.textContent = "No video";
      setStatus("Open any YouTube video or Short, then click the CommentLens icon.");
      return;
    }

    videoBadge.textContent = currentVideoId;
    videoBadge.className = "badge badge-ready";
    analyzeBtn.disabled = false;

    // Auto-load cached result if this video was already analysed
    const cacheKey = `cl_${currentVideoId}_${maxCommentsSelect.value}`;
    const stored = await extApi.storage.local.get(cacheKey);
    if (stored && stored[cacheKey]) {
      renderResults(stored[cacheKey], true);
    }
  } catch (err) {
    setStatus("Could not read the active tab URL.", { isError: true });
  }
}

analyzeBtn.addEventListener("click", () => runAnalysis(false));
refreshBtn.addEventListener("click", () => runAnalysis(true));
maxCommentsSelect.addEventListener("change", async () => {
  if (!currentVideoId) return;
  const cacheKey = `cl_${currentVideoId}_${maxCommentsSelect.value}`;
  const stored = await extApi.storage.local.get(cacheKey);
  if (stored && stored[cacheKey]) {
    renderResults(stored[cacheKey], true);
  }
});

init();