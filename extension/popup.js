const SVG_NS = "http://www.w3.org/2000/svg";
const CACHE_PREFIX = "cl_v3_";
const COOLDOWN_MS = 60 * 60 * 1000; // Quota fail hone par 60 min skip
const REQUEST_TIMEOUT_MS = 35000;   // Cold-start safe 35s timeout
const extApi = typeof browser !== "undefined" ? browser : chrome;

// ======================================================
// BACKEND PRIORITY ORDER (Upar se neeche failover hoga)
// Jo deploy abhi nahi kiya uska url "" chhod do -> auto skip hoga
// ======================================================
const PROVIDERS = [
  {
    id: "modal",
    name: "Modal",
    type: "rest",
    url: " https://chirag-sharma1008cs--commentlens-api-commentlensservice-analyze.us-east.modal.run" // e.g. https://abhishek--commentlens-api-commentlensservice-analyze.modal.run
  },
  {
    id: "huggingface",
    name: "HF Space",
    type: "gradio",
    url: "https://abhishek2775-commentlens-api.hf.space",
    token: "" // Personal use ke liye optional: "hf_xxxx" Read token
  }
];

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

class NonRetryableError extends Error {
  constructor(message) {
    super(message);
    this.name = "NonRetryableError";
  }
}

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

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function callRestProvider(provider, videoId, maxComments) {
  const resp = await fetchWithTimeout(provider.url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_id: videoId, max_comments: Number(maxComments) })
  });

  if (!resp.ok) {
    throw new Error(`${provider.name} HTTP ${resp.status}`);
  }

  const data = await resp.json();
  if (data && data.error) {
    if (data.retryable === false) {
      throw new NonRetryableError(data.error);
    }
    throw new Error(data.error);
  }
  return data;
}

async function callGradioProvider(provider, videoId, maxComments) {
  const payload = JSON.stringify({ data: [videoId, Number(maxComments)] });
  const authHeaders = provider.token ? { Authorization: `Bearer ${provider.token}` } : {};

  let endpoint = `${provider.url}/gradio_api/call/analyze`;
  let postResp = await fetchWithTimeout(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders },
    body: payload
  });

  if (postResp.status === 404) {
    endpoint = `${provider.url}/call/analyze`;
    postResp = await fetchWithTimeout(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders },
      body: payload
    });
  }

  if (!postResp.ok) {
    throw new Error(`HF Space HTTP ${postResp.status}`);
  }

  const { event_id: eventId } = await postResp.json();
  if (!eventId) throw new Error("HF Space did not return an event_id.");

  const streamResp = await fetchWithTimeout(`${endpoint}/${eventId}`, {
    headers: authHeaders
  });
  if (!streamResp.ok) throw new Error(`HF Space stream HTTP ${streamResp.status}`);

  const rawText = await streamResp.text();
  const blocks = rawText.split(/\r?\n\r?\n/);

  for (const block of blocks) {
    const lines = block.split(/\r?\n/);
    let eventType = "";
    const dataLines = [];

    for (const line of lines) {
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }

    if (eventType === "error") {
      throw new Error(dataLines.join(" ") || "ZeroGPU quota exceeded");
    }

    if (eventType === "complete" && dataLines.length > 0) {
      const parsed = JSON.parse(dataLines.join("\n"));
      const result = Array.isArray(parsed) ? parsed[0] : parsed;
      if (result && result.error) {
        if (result.error.includes("YouTube API error") || result.error.includes("No comments")) {
          throw new NonRetryableError(result.error);
        }
        throw new Error(result.error);
      }
      return result;
    }
  }

  throw new Error("Incomplete response from HF Space.");
}

async function getCooldowns() {
  const stored = await extApi.storage.local.get("cl_provider_cooldowns");
  return stored.cl_provider_cooldowns || {};
}

async function markProviderCooldown(providerId) {
  const cooldowns = await getCooldowns();
  cooldowns[providerId] = Date.now() + COOLDOWN_MS;
  await extApi.storage.local.set({ cl_provider_cooldowns: cooldowns });
}

async function clearProviderCooldown(providerId) {
  const cooldowns = await getCooldowns();
  if (cooldowns[providerId]) {
    delete cooldowns[providerId];
    await extApi.storage.local.set({ cl_provider_cooldowns: cooldowns });
  }
}

function isConfigured(provider) {
  return Boolean(provider.url && !provider.url.startsWith("PASTE_"));
}

async function analyzeWithFailover(videoId, maxComments) {
  const activeProviders = PROVIDERS.filter(isConfigured);
  if (activeProviders.length === 0) {
    throw new Error("No backend URL configured in popup.js.");
  }

  const cooldowns = await getCooldowns();
  const now = Date.now();

  // Healthy providers pehle, cooldown wale sirf emergency fallback ke roop mein end mein
  const healthy = activeProviders.filter((p) => !cooldowns[p.id] || cooldowns[p.id] <= now);
  const cooling = activeProviders.filter((p) => cooldowns[p.id] && cooldowns[p.id] > now);
  const queue = [...healthy, ...cooling];

  const failures = [];

  for (let i = 0; i < queue.length; i++) {
    const provider = queue[i];
    setStatus(
      i === 0
        ? `Analysing via ${provider.name}…`
        : `${queue[i - 1].name} unavailable — switching to ${provider.name}…`,
      { loading: true }
    );

    try {
      const data =
        provider.type === "gradio"
          ? await callGradioProvider(provider, videoId, maxComments)
          : await callRestProvider(provider, videoId, maxComments);

      await clearProviderCooldown(provider.id);
      return { ...data, _provider: provider.name };
    } catch (err) {
      if (err instanceof NonRetryableError) {
        throw err;
      }
      failures.push(`${provider.name}: ${err.name === "AbortError" ? "Timeout" : err.message}`);
      await markProviderCooldown(provider.id);
    }
  }

  throw new Error(`All backends failed. (${failures.join(" | ")})`);
}

// ======================================================
// CHART RENDERERS (Same visual dashboard)
// ======================================================
function renderGauge(data) {
  const pct = data.percentages || { Positive: 0, Neutral: 0, Negative: 0 };
  const score = Math.max(0, Math.min(100, Math.round(50 + (pct.Positive - pct.Negative) / 2)));

  const arcLen = Math.PI * 80;
  document.getElementById("gauge-fill").style.strokeDashoffset = (arcLen * (1 - score / 100)).toFixed(2);

  const theta = Math.PI * (1 - score / 100);
  const dotEl = document.getElementById("gauge-dot");
  dotEl.setAttribute("cx", (100 + 80 * Math.cos(theta)).toFixed(1));
  dotEl.setAttribute("cy", (100 - 80 * Math.sin(theta)).toFixed(1));

  document.getElementById("vibe-score").textContent = `${score}/100`;

  const mood = data.overall_mood || "Neutral";
  const moodEl = document.getElementById("overall-mood");
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
    const dashLen = (count / total) * circumference;

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
    const barEl = document.getElementById(item.bar);
    barEl.style.height = "0%";
    requestAnimationFrame(() => {
      barEl.style.height = `${Math.max(p, c > 0 ? 4 : 0)}%`;
    });
  }
}

function renderTimelineChart(timeline) {
  const svg = document.getElementById("timeline-svg");
  const readout = document.getElementById("timeline-readout");
  const emptyEl = document.getElementById("timeline-empty");

  svg.replaceChildren();

  if (!timeline || timeline.length === 0) {
    svg.classList.add("hidden");
    readout.classList.add("hidden");
    emptyEl.classList.remove("hidden");
    return;
  }

  svg.classList.remove("hidden");
  readout.classList.remove("hidden");
  emptyEl.classList.add("hidden");

  const latest = timeline[timeline.length - 1];
  const formatReadout = (b) =>
    `${b.label} (${b.total} comments) — Pos ${b.pos_pct}% · Neu ${b.neu_pct}% · Neg ${b.neg_pct}%`;
  readout.textContent = formatReadout(latest);

  const padL = 32;
  const padR = 14;
  const padT = 14;
  const padB = 26;
  const width = 360 - padL - padR;
  const height = 170 - padT - padB;

  const xAt = (i) =>
    timeline.length === 1 ? padL + width / 2 : padL + (i / (timeline.length - 1)) * width;
  const yAt = (pctVal) => padT + height - (pctVal / 100) * height;

  const defs = document.createElementNS(SVG_NS, "defs");
  const grad = document.createElementNS(SVG_NS, "linearGradient");
  grad.setAttribute("id", "posAreaGrad");
  grad.setAttribute("x1", "0%");
  grad.setAttribute("y1", "0%");
  grad.setAttribute("x2", "0%");
  grad.setAttribute("y2", "100%");
  const stop1 = document.createElementNS(SVG_NS, "stop");
  stop1.setAttribute("offset", "0%");
  stop1.setAttribute("stop-color", "#22c55e");
  stop1.setAttribute("stop-opacity", "0.22");
  const stop2 = document.createElementNS(SVG_NS, "stop");
  stop2.setAttribute("offset", "100%");
  stop2.setAttribute("stop-color", "#22c55e");
  stop2.setAttribute("stop-opacity", "0");
  grad.appendChild(stop1);
  grad.appendChild(stop2);
  defs.appendChild(grad);
  svg.appendChild(defs);

  for (const level of [0, 50, 100]) {
    const y = yAt(level);
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", String(padL));
    line.setAttribute("x2", String(padL + width));
    line.setAttribute("y1", y.toFixed(1));
    line.setAttribute("y2", y.toFixed(1));
    line.setAttribute("class", "tl-grid");
    svg.appendChild(line);

    const lbl = document.createElementNS(SVG_NS, "text");
    lbl.setAttribute("x", String(padL - 5));
    lbl.setAttribute("y", (y + 3).toFixed(1));
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("class", "tl-axis-text");
    lbl.textContent = `${level}%`;
    svg.appendChild(lbl);
  }

  if (timeline.length > 1) {
    const pts = timeline.map((b, i) => `${xAt(i).toFixed(1)},${yAt(b.pos_pct).toFixed(1)}`).join(" ");
    const area = document.createElementNS(SVG_NS, "polygon");
    area.setAttribute(
      "points",
      `${xAt(0).toFixed(1)},${yAt(0).toFixed(1)} ${pts} ${xAt(timeline.length - 1).toFixed(1)},${yAt(0).toFixed(1)}`
    );
    area.setAttribute("fill", "url(#posAreaGrad)");
    svg.appendChild(area);
  }

  const series = [
    { field: "pos_pct", color: "#22c55e" },
    { field: "neu_pct", color: "#f59e0b" },
    { field: "neg_pct", color: "#ef4444" }
  ];

  if (timeline.length > 1) {
    for (const s of series) {
      const polyline = document.createElementNS(SVG_NS, "polyline");
      const pts = timeline.map((b, i) => `${xAt(i).toFixed(1)},${yAt(b[s.field]).toFixed(1)}`).join(" ");
      polyline.setAttribute("points", pts);
      polyline.setAttribute("stroke", s.color);
      polyline.setAttribute("class", "tl-line");
      svg.appendChild(polyline);
    }
  }

  timeline.forEach((b, i) => {
    const x = xAt(i);
    for (const s of series) {
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", x.toFixed(1));
      dot.setAttribute("cy", yAt(b[s.field]).toFixed(1));
      dot.setAttribute("r", "3.8");
      dot.setAttribute("fill", s.color);
      dot.setAttribute("class", "tl-dot");
      dot.addEventListener("mouseenter", () => {
        readout.textContent = formatReadout(b);
      });
      svg.appendChild(dot);
    }

    const xLabel = document.createElementNS(SVG_NS, "text");
    xLabel.setAttribute("x", x.toFixed(1));
    xLabel.setAttribute("y", String(170 - 6));
    xLabel.setAttribute("text-anchor", "middle");
    xLabel.setAttribute("class", "tl-axis-text");
    xLabel.textContent = b.label;
    svg.appendChild(xLabel);
  });
}

function renderKeywords(keywords) {
  const listEl = document.getElementById("keywords-list");
  const emptyEl = document.getElementById("keywords-empty");
  listEl.replaceChildren();

  if (!keywords || keywords.length === 0) {
    emptyEl.classList.remove("hidden");
    return;
  }

  emptyEl.classList.add("hidden");
  const maxCount = Math.max(...keywords.map((k) => k.count), 1);

  for (const item of keywords) {
    const li = document.createElement("li");
    li.className = "kw-row";
    const bd = item.breakdown || { Positive: 0, Neutral: 0, Negative: 0 };
    li.title = `Pos: ${bd.Positive}, Neu: ${bd.Neutral}, Neg: ${bd.Negative}`;

    const top = document.createElement("div");
    top.className = "kw-top";

    const left = document.createElement("div");
    left.className = "kw-left";

    const wordSpan = document.createElement("span");
    wordSpan.className = "kw-word";
    wordSpan.textContent = `#${item.word}`;

    const pill = document.createElement("span");
    const suffix =
      item.sentiment === "Positive" ? "pos" : item.sentiment === "Negative" ? "neg" : "neu";
    pill.className = `kw-pill kw-pill-${suffix}`;
    pill.textContent = item.sentiment.slice(0, 3);

    left.appendChild(wordSpan);
    left.appendChild(pill);

    const countSpan = document.createElement("span");
    countSpan.className = "kw-count";
    countSpan.textContent = `${item.count} mentions`;

    top.appendChild(left);
    top.appendChild(countSpan);

    const track = document.createElement("div");
    track.className = "kw-track";

    const fill = document.createElement("div");
    fill.className = `kw-fill kw-fill-${suffix}`;
    fill.style.width = `${Math.max(10, Math.round((item.count / maxCount) * 100))}%`;

    track.appendChild(fill);
    li.appendChild(top);
    li.appendChild(track);
    listEl.appendChild(li);
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

  // Dikhao ki kis backend ne response diya (e.g. "Via Modal" ya "Cached · Cloud Run")
  const cacheTag = document.getElementById("cache-tag");
  const providerLabel = data._provider || "API";
  cacheTag.textContent = fromCache ? `Cached · ${providerLabel}` : `Via ${providerLabel}`;
  cacheTag.classList.remove("hidden");

  renderGauge(data);
  renderPieOrDonut(data);
  renderBarChart(data);
  renderTimelineChart(data.timeline || []);
  renderKeywords(data.top_keywords || []);

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

async function runAnalysis(forceRefresh = false) {
  if (!currentVideoId) return;

  const maxComments = maxCommentsSelect.value;
  const cacheKey = `${CACHE_PREFIX}${currentVideoId}_${maxComments}`;

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

  try {
    const data = await analyzeWithFailover(currentVideoId, maxComments);
    await extApi.storage.local.set({ [cacheKey]: data });
    renderResults(data, false);
  } catch (err) {
    setStatus(err.message || "All backends failed.", { isError: true });
  } finally {
    analyzeBtn.disabled = false;
  }
}

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${target}`));
  });
});

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

    const cacheKey = `${CACHE_PREFIX}${currentVideoId}_${maxCommentsSelect.value}`;
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
  const cacheKey = `${CACHE_PREFIX}${currentVideoId}_${maxCommentsSelect.value}`;
  const stored = await extApi.storage.local.get(cacheKey);
  if (stored && stored[cacheKey]) {
    renderResults(stored[cacheKey], true);
  }
});

init();