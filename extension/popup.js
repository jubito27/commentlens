const SPACE_URL = "https://abhishek2775-commentlens-api.hf.space";
const SVG_NS = "http://www.w3.org/2000/svg";
const extApi = typeof browser !== "undefined" ? browser : chrome;

const videoBadge = document.getElementById("video-badge");
const maxCommentsSelect = document.getElementById("max-comments");
const analyzeBtn = document.getElementById("analyze-btn");
const refreshBtn = document.getElementById("refresh-btn");
const statusBox = document.getElementById("status-box");
const spinner = document.getElementById("spinner");
const statusText = document.getElementById("status-text");
const resultsEl = document.getElementById("results");
const chartModeBtn = document.getElementById("chart-mode-btn");

let currentVideoId = null;
let latestData = null;
let isPieMode = false;

const SENTIMENT_META = [
  { key: "Positive", color: "#22c55e" },
  { key: "Neutral",  color: "#f59e0b" },
  { key: "Negative", color: "#ef4444" }
];

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

function renderGauge(data) {
  const pct = data.percentages || { Positive: 0, Neutral: 0, Negative: 0 };
  const rawScore = Math.round(50 + (pct.Positive - pct.Negative) / 2);
  const score = Math.max(0, Math.min(100, rawScore));

  const arcLen = Math.PI * 80; // ~251.33
  const fillEl = document.getElementById("gauge-fill");
  const dotEl = document.getElementById("gauge-dot");
  const scoreEl = document.getElementById("vibe-score");
  const moodEl = document.getElementById("overall-mood");

  fillEl.style.strokeDashoffset = (arcLen * (1 - score / 100)).toFixed(2);

  const theta = Math.PI * (1 - score / 100);
  const dotX = 100 + 80 * Math.cos(theta);
  const dotY = 100 - 80 * Math.sin(theta);
  dotEl.setAttribute("cx", dotX.toFixed(1));
  dotEl.setAttribute("cy", dotY.toFixed(1));

  scoreEl.textContent = `${score}/100`;

  const mood = data.overall_mood || "Neutral";
  moodEl.textContent = mood;
  moodEl.className = "mood-tag";
  if (mood === "Positive") moodEl.classList.add("mood-positive");
  else if (mood === "Negative") moodEl.classList.add("mood-negative");
  else moodEl.classList.add("mood-neutral");
}

function renderPieOrDonut(data) {
  const svg = document.getElementById("pie-svg");
  const legendEl = document.getElementById("pie-legend");
  const centerBox = document.getElementById("donut-center");
  const mainVal = document.getElementById("donut-main-val");
  const mainLbl = document.getElementById("donut-main-lbl");

  svg.replaceChildren();
  legendEl.replaceChildren();

  const pct = data.percentages || { Positive: 0, Neutral: 0, Negative: 0 };
  const cnt = data.counts || { Positive: 0, Neutral: 0, Negative: 0 };
  const total = Math.max(1, data.total_comments || 1);

  const radius = isPieMode ? 30 : 50;
  const strokeWidth = isPieMode ? 60 : 20;
  const circumference = 2 * Math.PI * radius;

  centerBox.classList.toggle("hidden", isPieMode);
  chartModeBtn.textContent = isPieMode ? "Switch to Donut" : "Switch to Pie";

  const dominantKey = data.overall_mood || "Positive";
  const setDefaultCenter = () => {
    mainVal.textContent = `${Math.round(pct[dominantKey] || 0)}%`;
    mainLbl.textContent = dominantKey;
  };
  setDefaultCenter();

  // Background track for Donut mode
  if (!isPieMode) {
    const bgCircle = document.createElementNS(SVG_NS, "circle");
    bgCircle.setAttribute("cx", "70");
    bgCircle.setAttribute("cy", "70");
    bgCircle.setAttribute("r", String(radius));
    bgCircle.setAttribute("fill", "none");
    bgCircle.setAttribute("stroke", "#101522");
    bgCircle.setAttribute("stroke-width", String(strokeWidth));
    svg.appendChild(bgCircle);
  }

  let offset = 0;
  for (const { key, color } of SENTIMENT_META) {
    const count = cnt[key] || 0;
    const share = count / total;
    const dashLen = share * circumference;

    if (count > 0) {
      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", "70");
      circle.setAttribute("cy", "70");
      circle.setAttribute("r", String(radius));
      circle.setAttribute("stroke", color);
      circle.setAttribute("stroke-width", String(strokeWidth));
      circle.setAttribute("stroke-dasharray", `${dashLen.toFixed(2)} ${circumference.toFixed(2)}`);
      circle.setAttribute("stroke-dashoffset", (-offset).toFixed(2));
      circle.setAttribute("class", "pie-slice");

      const title = document.createElementNS(SVG_NS, "title");
      title.textContent = `${key}: ${pct[key]}% (${count})`;
      circle.appendChild(title);

      circle.addEventListener("mouseenter", () => {
        if (!isPieMode) {
          mainVal.textContent = `${Math.round(pct[key])}%`;
          mainLbl.textContent = key;
        }
      });
      circle.addEventListener("mouseleave", setDefaultCenter);

      svg.appendChild(circle);
      offset += dashLen;
    }

    // Legend row
    const li = document.createElement("li");
    li.className = "legend-item";

    const left = document.createElement("div");
    left.className = "legend-left";
    const dot = document.createElement("span");
    dot.className = "legend-dot";
    dot.style.backgroundColor = color;
    const name = document.createElement("span");
    name.textContent = key;
    left.appendChild(dot);
    left.appendChild(name);

    const right = document.createElement("div");
    right.className = "legend-right";
    right.textContent = `${pct[key]}%`;
    const sub = document.createElement("span");
    sub.className = "legend-sub";
    sub.textContent = `${count} comments`;
    right.appendChild(sub);

    li.appendChild(left);
    li.appendChild(right);

    li.addEventListener("mouseenter", () => {
      if (!isPieMode) {
        mainVal.textContent = `${Math.round(pct[key])}%`;
        mainLbl.textContent = key;
      }
    });
    li.addEventListener("mouseleave", setDefaultCenter);

    legendEl.appendChild(li);
  }
}

function renderBarChart(data) {
  const pct = data.percentages || { Positive: 0, Neutral: 0, Negative: 0 };
  const cnt = data.counts || { Positive: 0, Neutral: 0, Negative: 0 };

  const mapping = [
    { key: "Positive", bar: "bar-pos", val: "bar-val-pos", cntEl: "bar-cnt-pos" },
    { key: "Neutral",  bar: "bar-neu", val: "bar-val-neu", cntEl: "bar-cnt-neu" },
    { key: "Negative", bar: "bar-neg", val: "bar-val-neg", cntEl: "bar-cnt-neg" }
  ];

  for (const item of mapping) {
    const p = pct[item.key] || 0;
    const c = cnt[item.key] || 0;
    document.getElementById(item.val).textContent = `${p}%`;
    document.getElementById(item.cntEl).textContent = `${c} comments`;
    // Smooth animation on next paint
    const barEl = document.getElementById(item.bar);
    barEl.style.height = "0%";
    requestAnimationFrame(() => {
      barEl.style.height = `${Math.max(p, c > 0 ? 4 : 0)}%`;
    });
  }
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

    const textNode = document.createElement("div");
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
  latestData = data;
  hideStatus();
  resultsEl.classList.remove("hidden");
  refreshBtn.hidden = false;

  document.getElementById("total-comments").textContent = `${data.total_comments} comments`;
  document.getElementById("cache-tag").classList.toggle("hidden", !fromCache);

  renderGauge(data);
  renderPieOrDonut(data);
  renderBarChart(data);

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
    throw new Error(`API request failed (HTTP ${postResp.status}). Ensure your Hugging Face Space is awake.`);
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
  setStatus("Fetching YouTube comments and building charts…", { loading: true });

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

// Tab switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));
  });
});

// Donut <-> Pie toggle
chartModeBtn.addEventListener("click", () => {
  isPieMode = !isPieMode;
  if (latestData) renderPieOrDonut(latestData);
});

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