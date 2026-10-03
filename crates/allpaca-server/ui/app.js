"use strict";

const $ = (id) => document.getElementById(id);
const LEVELS = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
const POLL_MS = 1000;
const SPARK_POINTS = 60;
const LOG_CAP = 4000;
const CHAT_KEY = "allpaca.chat";
const SETTINGS_KEY = "allpaca.settings";

class Unauthorized extends Error {}

const state = {
  page: "dashboard",
  authed: false,
  pollTimer: null,
  lastTokens: null,
  lastTokensAt: 0,
  tps: [],
  messages: loadJson(CHAT_KEY, []),
  pending: [],
  streaming: null,
  logLines: [],
  logSource: null,
  logRetry: 1000,
  logRendered: 0,
  logFrame: 0,
};

/* ---------- storage ---------- */

function loadJson(key, fallback) {
  try {
    const v = JSON.parse(localStorage.getItem(key) || "null");
    return v ?? fallback;
  } catch {
    return fallback;
  }
}

function saveJson(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

function saveMessages() {
  const keep = state.messages.slice(-80);
  // Images can blow the storage quota: keep the text if they do.
  if (!saveJson(CHAT_KEY, keep)) {
    saveJson(CHAT_KEY, keep.map(({ images, ...m }) => m));
  }
}

/* ---------- formatting ---------- */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function bytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(1)} MiB`;
  return `${(v / 1024 ** 3).toFixed(2)} GiB`;
}

function num(n) {
  return (Number(n) || 0).toLocaleString();
}

function meter(frac) {
  const f = Math.max(0, Math.min(1, frac || 0));
  const cls = f > 0.93 ? "hot" : f > 0.8 ? "warn" : "";
  return `<div class="meter ${cls}"><span style="width:${(f * 100).toFixed(1)}%"></span></div>`;
}

function card(title, value, sub = "", extra = "") {
  const subHtml = sub ? `<p class="metric-sub">${sub}</p>` : "";
  return `<article class="card"><h3>${title}</h3><div class="metric">${value}</div>${subHtml}${extra}</article>`;
}

/* Minimal, escape-first markdown: fenced code, inline code, bold, headings,
 * lists and paragraphs. Enough for model replies without a dependency. */
function renderMarkdown(src) {
  const out = [];
  const fence = /```([^\n`]*)\n?([\s\S]*?)(```|$)/g;
  let last = 0;
  let m;
  while ((m = fence.exec(src)) !== null) {
    if (m.index > last) out.push(renderBlocks(src.slice(last, m.index)));
    const lang = escapeHtml(m[1].trim() || "code");
    out.push(
      `<div class="codeblock"><div class="code-head"><span>${lang}</span>` +
        `<button type="button" class="ghost copy">Copy</button></div>` +
        `<pre><code>${escapeHtml(m[2].replace(/\n$/, ""))}</code></pre></div>`
    );
    last = fence.lastIndex;
    if (m[0].length === 0) break;
  }
  if (last < src.length) out.push(renderBlocks(src.slice(last)));
  return out.join("");
}

function renderInline(s) {
  // Odd parts are `code` spans: left as written.
  return s
    .split(/(`[^`\n]+`)/)
    .map((part, i) =>
      i % 2
        ? `<code>${escapeHtml(part.slice(1, -1))}</code>`
        : escapeHtml(part)
            .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
            .replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?![*\w])/g, "$1<em>$2</em>")
    )
    .join("");
}

function renderBlocks(text) {
  const html = [];
  let list = null;
  let para = [];
  const flushPara = () => {
    if (para.length) html.push(`<p>${para.map(renderInline).join("<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) html.push(`<${list.tag}>${list.items.map((i) => `<li>${renderInline(i)}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  for (const line of text.split("\n")) {
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    const ul = /^\s*[-*]\s+(.*)$/.exec(line);
    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (!line.trim()) {
      flushPara();
      flushList();
    } else if (h) {
      flushPara();
      flushList();
      const lvl = Math.min(4, h[1].length + 1);
      html.push(`<h${lvl}>${renderInline(h[2])}</h${lvl}>`);
    } else if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) {
        flushList();
        list = { tag, items: [] };
      }
      list.items.push((ul || ol)[1]);
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  return html.join("");
}

/* ---------- auth + api ---------- */

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "include", ...opts });
  if (res.status === 401) {
    onUnauthorized();
    throw new Unauthorized();
  }
  return res;
}

function onUnauthorized() {
  state.authed = false;
  stopPolling();
  stopLogs();
  $("app").classList.add("hidden");
  $("login").classList.remove("hidden");
  $("password").focus();
}

function enterApp() {
  state.authed = true;
  $("login").classList.add("hidden");
  $("app").classList.remove("hidden");
  setPage(location.hash.slice(1) || state.page);
  startPolling();
  loadModel();
}

async function boot() {
  try {
    const res = await api("/v1/status");
    enterApp();
    if (res.ok) renderDashboard(await res.json());
  } catch (e) {
    if (!(e instanceof Unauthorized)) {
      // Server unreachable: show the app and let polling report it.
      enterApp();
      setConn("down");
    }
  }
}

/* ---------- dashboard ---------- */

function setConn(kind) {
  const el = $("conn");
  el.classList.toggle("ok", kind === "ok");
  el.classList.toggle("down", kind === "down");
  $("conn-text").textContent = kind === "ok" ? "live" : kind === "down" ? "offline" : "connecting";
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!state.authed) return;
    if (!document.hidden) await pollStatus();
    if (state.authed) state.pollTimer = setTimeout(tick, POLL_MS);
  };
  tick();
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

async function pollStatus() {
  try {
    const res = await api("/v1/status");
    if (!res.ok) {
      setConn("down");
      return;
    }
    renderDashboard(await res.json());
    setConn("ok");
  } catch (e) {
    if (!(e instanceof Unauthorized)) setConn("down");
  }
}

async function loadModel() {
  try {
    const body = await api("/v1/models").then((r) => r.json());
    const id = body && body.data && body.data[0] && body.data[0].id;
    $("chat-model").textContent = id || "no model";
    $("chat-model").classList.toggle("muted", !id);
  } catch {
    /* status polling reports connectivity */
  }
}

function sampleThroughput(s) {
  const now = performance.now();
  const toks = Number(s.tokens_generated) || 0;
  if (state.lastTokens !== null && now > state.lastTokensAt) {
    const dt = (now - state.lastTokensAt) / 1000;
    const rate = Math.max(0, toks - state.lastTokens) / dt;
    state.tps.push(rate);
    if (state.tps.length > SPARK_POINTS) state.tps.shift();
  }
  state.lastTokens = toks;
  state.lastTokensAt = now;
}

function renderSpark() {
  const pts = state.tps;
  const latest = pts.length ? pts[pts.length - 1] : 0;
  $("hero-tps").textContent = latest.toFixed(1);
  if (pts.length < 2) {
    $("spark-line").setAttribute("d", "");
    $("spark-fill").setAttribute("d", "");
    return;
  }
  const max = Math.max(1, ...pts);
  const step = 120 / (SPARK_POINTS - 1);
  const x0 = 120 - step * (pts.length - 1);
  const coords = pts.map((v, i) => [x0 + i * step, 31 - (v / max) * 29]);
  const line = coords.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  $("spark-line").setAttribute("d", line);
  $("spark-fill").setAttribute("d", `${line} L120,32 L${x0.toFixed(2)},32 Z`);
}

function renderDashboard(data) {
  const s = data.stats || {};
  if (data.stats) state.lastStats = data.stats;
  if (data.devices && data.devices.length) state.lastDevices = data.devices;
  renderBanner(data.engine);
  if (state.page === "model" && model.info) renderCtx();
  const h = data.host || {};
  const devices = data.devices || [];
  sampleThroughput(s);
  renderSpark();

  const gpuName = (id, fallback) => {
    const d = devices.find((x) => x.id === id && /gpu|hip|cuda|vulkan/i.test(x.kind || "gpu"));
    return d && d.name ? d.name.replace(/^AMD Radeon /, "") : fallback;
  };
  $("hero-model").textContent = s.model_id || "—";
  const active = Number(s.active_requests) || 0;
  $("hero-sub").textContent = [
    s.backend ? s.backend.toUpperCase() : "",
    `${active} active request${active === 1 ? "" : "s"}`,
    `${num(s.requests_total)} served`,
    `${num(s.tokens_generated)} tokens`,
  ]
    .filter(Boolean)
    .join(" · ");

  // No VRAM card for a GPU the engine does not report (CPU-only, one GPU).
  const vram = (used, total, label) =>
    total ? card(`${escapeHtml(label)} VRAM`, bytes(used), `of ${bytes(total)}`, meter(used / total)) : "";
  $("mem-cards").innerHTML = [
    card("Host RAM", bytes(h.ram_used_bytes), `of ${bytes(h.ram_total_bytes)}`, meter(h.ram_total_bytes ? h.ram_used_bytes / h.ram_total_bytes : 0)),
    vram(s.gpu_vram_used_bytes, s.gpu_vram_total_bytes, gpuName(s.target_device, "GPU 0")),
    vram(s.draft_gpu_vram_used_bytes, s.draft_gpu_vram_total_bytes, gpuName(s.draft_device, "GPU 1")),
    card("Weights on device", bytes(s.weight_bytes_on_device), `KV on device ${bytes(s.kv_bytes_on_device)}`),
  ].join("");

  // GPUs on the RPC workers, as each worker reports them.
  const remote = data.remote_gpus || [];
  $("remote-title").classList.toggle("hidden", !remote.length);
  $("remote-cards").classList.toggle("hidden", !remote.length);
  $("remote-cards").innerHTML = remote
    .map((g) => {
      const host = String(g.endpoint || "").replace(/:\d+$/, "");
      // The driver's product name can carry the wrong capacity (32 GB MI50s
      // report "MI50 16GB"); the measured total is shown below instead.
      const name = String(g.name || "GPU")
        .replace(/^(AMD )?Radeon /, "")
        .replace(/\s+\d+\s*GB$/i, "");
      const frac = g.mem_total_bytes ? g.mem_used_bytes / g.mem_total_bytes : 0;
      const live = [
        g.util_pct != null ? `${g.util_pct}% busy` : "",
        g.temp_c != null ? `${g.temp_c} °C` : "",
        g.power_w != null ? `${g.power_w} W` : "",
      ].filter(Boolean);
      return card(
        `${escapeHtml(name)} · ${escapeHtml(host)} #${g.index}`,
        bytes(g.mem_used_bytes),
        `of ${bytes(g.mem_total_bytes)}${live.length ? ` · ${live.join(" · ")}` : ""}`,
        meter(frac) + (g.util_pct != null ? `<div class="meter util" title="GPU busy"><span style="width:${g.util_pct}%"></span></div>` : "")
      );
    })
    .join("");

  const proposed = Number(s.draft_proposed) || 0;
  const steps = Number(s.batched_decode_steps) || 0;
  $("engine-cards").innerHTML = [
    card("Latency", `${(s.itl_ms_ema || 0).toFixed(1)} ms`, `ITL · TTFT ${num(s.ttft_ms_last)} ms`),
    card("Host CPU", `${(h.cpu_usage || 0).toFixed(0)}%`, "", meter((h.cpu_usage || 0) / 100)),
    card("GPU util", `${((s.gpu_util || 0) * 100).toFixed(0)}%`, "", meter(s.gpu_util || 0)),
    card("KV pages", `${num(s.vram_pages)} VRAM`, `${num(s.dram_pages)} DRAM`),
    card(
      "Speculation",
      proposed ? `${((s.acceptance_rate || 0) * 100).toFixed(1)}%` : "idle",
      proposed ? `${num(s.draft_accepted)} of ${num(proposed)} drafts accepted` : "no drafts proposed"
    ),
    card(
      "Batched decode",
      steps ? `${((Number(s.batched_decode_rows) || 0) / steps).toFixed(2)} rows` : "—",
      steps ? `avg per step · ${num(steps)} steps` : "single stream so far"
    ),
    card("RPC", `${num(s.rpc_roundtrips)}`, `round trips · ${bytes(s.rpc_activation_bytes)} activations`),
  ].join("");

  $("device-rows").innerHTML =
    devices
      .map(
        (d) =>
          `<tr><td>${escapeHtml(d.id)}</td><td>${escapeHtml(d.kind)}</td><td>${escapeHtml(d.name)}</td><td class="num">${bytes(d.memory_bytes)}</td></tr>`
      )
      .join("") || `<tr><td colspan="4" class="muted">No devices</td></tr>`;
}

/* ---------- pages ---------- */

function setPage(name) {
  if (!["dashboard", "chat", "model", "logs"].includes(name)) name = "dashboard";
  state.page = name;
  document.querySelectorAll("nav .tab").forEach((b) => {
    const on = b.dataset.page === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-current", on ? "page" : "false");
  });
  ["dashboard", "chat", "model", "logs"].forEach((p) => $(`page-${p}`).classList.toggle("hidden", p !== name));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  if (name === "logs") {
    startLogs();
    scheduleLogRender(true);
  }
  if (name === "chat") {
    renderMessages();
    $("prompt").focus();
  }
  if (name === "model") {
    loadModelSettings();
    loadAvailable();
  }
}

/* ---------- chat ---------- */

function loadSettings() {
  const s = loadJson(SETTINGS_KEY, {});
  // 512 was the old default, saved alongside other settings: not a choice.
  if (s.maxTokens && String(s.maxTokens) !== "512") $("max-tokens").value = s.maxTokens;
  if (s.temperature !== undefined) $("temperature").value = s.temperature;
  if (s.system) $("system-prompt").value = s.system;
}

function saveSettings() {
  saveJson(SETTINGS_KEY, {
    maxTokens: $("max-tokens").value,
    temperature: $("temperature").value,
    system: $("system-prompt").value,
  });
}

function messageHtml(m, i) {
  const live = state.streaming && state.streaming.index === i;
  const images = (m.images || []).map((src) => `<img src="${escapeHtml(src)}" alt="attached image">`).join("");
  const body =
    m.role === "assistant" ? renderMarkdown(m.content) : escapeHtml(m.content);
  const bubbleCls = `bubble${m.error ? " error-bubble" : ""}${live ? " cursor" : ""}`;
  const last = i === state.messages.length - 1;
  const more = m.truncated && last && !state.streaming
    ? ` <button type="button" class="ghost small continue-btn">Continue</button>`
    : "";
  const meta = m.meta ? `<div class="msg-meta">${escapeHtml(m.meta)}${more}</div>` : "";
  return (
    `<div class="msg ${m.role}" data-i="${i}">` +
    (images ? `<div class="msg-images">${images}</div>` : "") +
    (m.content || live || !images ? `<div class="${bubbleCls}">${body}</div>` : "") +
    meta +
    `</div>`
  );
}

function nearBottom(el) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

function renderMessages() {
  const box = $("messages");
  const empty = $("empty-chat");
  box.querySelectorAll(".msg").forEach((n) => n.remove());
  empty.classList.toggle("hidden", state.messages.length > 0);
  box.insertAdjacentHTML("beforeend", state.messages.map(messageHtml).join(""));
  box.scrollTop = box.scrollHeight;
}

let streamFrame = 0;
function renderStreaming() {
  if (streamFrame) return;
  streamFrame = requestAnimationFrame(() => {
    streamFrame = 0;
    const st = state.streaming;
    const i = st ? st.index : state.messages.length - 1;
    const box = $("messages");
    const node = box.querySelector(`.msg[data-i="${i}"]`);
    if (!node || !state.messages[i]) return renderMessages();
    const stick = nearBottom(box);
    node.outerHTML = messageHtml(state.messages[i], i);
    if (stick) box.scrollTop = box.scrollHeight;
  });
}

function setStreaming(on) {
  $("send").classList.toggle("hidden", on);
  $("stop").classList.toggle("hidden", !on);
}

function chatPayload() {
  const sys = $("system-prompt").value.trim();
  const msgs = state.messages
    .filter((m) => !m.error && (m.content || (m.images && m.images.length)))
    .map((m) => {
      if (!m.images || !m.images.length) return { role: m.role, content: m.content };
      return {
        role: m.role,
        content: [
          ...(m.content ? [{ type: "text", text: m.content }] : []),
          ...m.images.map((url) => ({ type: "image_url", image_url: { url } })),
        ],
      };
    });
  if (sys) msgs.unshift({ role: "system", content: sys });
  const temp = Number($("temperature").value);
  return {
    messages: msgs,
    stream: true,
    max_tokens: Math.max(1, Number($("max-tokens").value) || 4096),
    temperature: Number.isFinite(temp) ? temp : 0.7,
  };
}

function autosize() {
  const t = $("prompt");
  t.style.height = "auto";
  t.style.height = `${Math.min(t.scrollHeight, window.innerHeight * 0.4)}px`;
}

async function sendChat(ev) {
  ev.preventDefault();
  if (state.streaming) return;
  const prompt = $("prompt").value.trim();
  if (!prompt && !state.pending.length) return;
  const user = { role: "user", content: prompt };
  if (state.pending.length) user.images = state.pending.slice();
  state.messages.push(user);
  state.pending = [];
  renderAttachments();
  $("prompt").value = "";
  autosize();
  const payload = chatPayload();
  const assistant = { role: "assistant", content: "" };
  state.messages.push(assistant);
  const ctrl = new AbortController();
  state.streaming = { index: state.messages.length - 1, ctrl };
  setStreaming(true);
  renderMessages();

  const started = performance.now();
  let firstAt = 0;
  let chunks = 0;
  let finish = "";
  try {
    const res = await api("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    if (!res.ok || !res.body) {
      let msg = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j && j.error && j.error.message) msg = j.error.message;
      } catch {
        /* not JSON */
      }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let json;
        try {
          json = JSON.parse(data);
        } catch {
          continue;
        }
        if (json.error) throw new Error(json.error.message || "stream error");
        const choice = json.choices && json.choices[0];
        if (!choice) continue;
        if (choice.delta && typeof choice.delta.content === "string") {
          // One event per token; a character split across tokens arrives
          // with the first part as empty content.
          if (!firstAt) firstAt = performance.now();
          chunks += 1;
          if (choice.delta.content) {
            assistant.content += choice.delta.content;
            renderStreaming();
          }
        }
        if (choice.finish_reason) finish = choice.finish_reason;
      }
    }
  } catch (e) {
    if (e instanceof Unauthorized) {
      assistant.error = true;
      assistant.content = assistant.content || "Signed out.";
    } else if (e.name === "AbortError") {
      finish = "stopped";
    } else {
      assistant.error = !assistant.content;
      assistant.meta = `error: ${e.message}`;
      if (!assistant.content) assistant.content = e.message;
    }
  } finally {
    const end = performance.now();
    if (chunks && !assistant.error) {
      const gen = firstAt ? (end - firstAt) / 1000 : 0;
      const rate = gen > 0 && chunks > 1 ? ` · ${((chunks - 1) / gen).toFixed(1)} tok/s` : "";
      const fin = finish === "length" ? " · cut off at the max-tokens limit" : finish && finish !== "stop" ? ` · ${finish}` : "";
      assistant.meta = `${chunks} tokens · TTFT ${Math.round(firstAt - started)} ms${rate}${fin}`;
      assistant.truncated = finish === "length";
    } else if (finish === "stopped" && !assistant.meta) {
      assistant.meta = "stopped";
    }
    state.streaming = null;
    setStreaming(false);
    saveMessages();
    renderMessages();
  }
}

function renderAttachments() {
  const box = $("attachments");
  box.classList.toggle("hidden", !state.pending.length);
  box.innerHTML = state.pending
    .map(
      (src, i) =>
        `<div class="thumb"><img src="${escapeHtml(src)}" alt="attachment ${i + 1}">` +
        `<button type="button" data-rm="${i}" aria-label="Remove attachment">×</button></div>`
    )
    .join("");
}

function addFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    const reader = new FileReader();
    reader.onload = () => {
      state.pending.push(reader.result);
      renderAttachments();
    };
    reader.readAsDataURL(f);
  }
}

/* ---------- logs ---------- */

function startLogs() {
  if (state.logSource || !state.authed) return;
  // The snapshot replaces history; streamed lines newer than it are kept.
  api("/v1/logs")
    .then((r) => r.json())
    .then((body) => {
      const snap = body.lines || [];
      const top = snap.length ? snap[snap.length - 1].seq : -1;
      state.logLines = snap.concat(state.logLines.filter((l) => l.seq > top));
      scheduleLogRender(true);
    })
    .catch(() => {});
  const src = new EventSource("/v1/logs/stream", { withCredentials: true });
  src.onopen = () => {
    state.logRetry = 1000;
  };
  src.onmessage = (ev) => {
    try {
      state.logLines.push(JSON.parse(ev.data));
    } catch {
      return;
    }
    if (state.logLines.length > LOG_CAP) {
      state.logLines.splice(0, state.logLines.length - LOG_CAP);
      state.logRendered = -1; // force a full rebuild
    }
    if (state.page === "logs") scheduleLogRender(false);
  };
  src.onerror = () => {
    src.close();
    state.logSource = null;
    if (!state.authed) return;
    const wait = state.logRetry;
    state.logRetry = Math.min(state.logRetry * 2, 15000);
    setTimeout(() => {
      if (state.authed && state.page === "logs") startLogs();
    }, wait);
  };
  state.logSource = src;
}

function stopLogs() {
  if (state.logSource) state.logSource.close();
  state.logSource = null;
}

function logLineEl(l) {
  const lvl = (l.level || "INFO").toUpperCase();
  const div = document.createElement("div");
  div.className = `log-line lvl-${lvl}`;
  const ts = new Date(l.ts_ms || 0);
  const time = Number.isNaN(ts.getTime()) ? "" : ts.toISOString().replace("T", " ").replace("Z", "");
  div.innerHTML =
    `<span class="ts">${time}</span> <span class="lvl">${lvl}</span> ` +
    `<span class="tgt">${escapeHtml(l.target || "")}</span> ${escapeHtml(l.message || "")}`;
  return div;
}

function logPasses(l, min, needle) {
  if (LEVELS.indexOf((l.level || "INFO").toUpperCase()) < min) return false;
  if (!needle) return true;
  return `${l.target || ""} ${l.message || ""}`.toLowerCase().includes(needle);
}

function scheduleLogRender(full) {
  if (full) state.logRendered = -1;
  if (state.logFrame) return;
  state.logFrame = requestAnimationFrame(() => {
    state.logFrame = 0;
    renderLogs();
  });
}

function renderLogs() {
  const view = $("log-view");
  const min = LEVELS.indexOf($("log-level").value);
  const needle = $("log-filter").value.trim().toLowerCase();
  const rebuild = state.logRendered < 0 || state.logRendered > state.logLines.length;
  const from = rebuild ? 0 : state.logRendered;
  const frag = document.createDocumentFragment();
  for (let i = from; i < state.logLines.length; i++) {
    const l = state.logLines[i];
    if (logPasses(l, min, needle)) frag.appendChild(logLineEl(l));
  }
  if (rebuild) view.replaceChildren(frag);
  else view.appendChild(frag);
  state.logRendered = state.logLines.length;
  const shown = view.childElementCount;
  $("log-count").textContent = `${shown.toLocaleString()} of ${state.logLines.length.toLocaleString()} lines`;
  if ($("log-follow").checked) view.scrollTop = view.scrollHeight;
}

/* ---------- model page ---------- */

const CTX_STEPS = [
  1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
  786432, 1048576,
];

const model = {
  info: null,
  usage: null,
  layer: 0,
  sel: null,
  ctx: null,
  usageTimer: null,
  lastEngineState: "ready",
};

function toast(msg, err = false) {
  const el = document.createElement("div");
  el.className = `toast${err ? " err" : ""}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), err ? 4000 : 1800);
}

async function apiJson(path, opts = {}) {
  const res = await api(path, {
    ...opts,
    headers: opts.body ? { "content-type": "application/json" } : undefined,
  });
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* empty body */
  }
  if (!res.ok) {
    throw new Error((body && body.error && body.error.message) || `HTTP ${res.status}`);
  }
  return body;
}

async function loadModelSettings() {
  try {
    model.info = await apiJson("/v1/model/settings");
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
    return;
  }
  const s = model.info.settings || {};
  if (model.ctx === null) {
    model.ctx = s.context_tokens || (model.info.kv && model.info.kv.context_tokens) || 32768;
  }
  renderModel();
  if (model.info.experts && model.info.experts.recording) startUsagePolling();
  if (model.info.experts) refreshUsage();
}

function kvBytes(tokens) {
  const kv = model.info && model.info.kv;
  if (!kv) return 0;
  const swa = kv.swa_window ? Math.min(tokens, kv.swa_window) : tokens;
  return tokens * kv.full_bytes_per_token + swa * kv.swa_bytes_per_token;
}

function nearestStep(tokens) {
  let best = 0;
  CTX_STEPS.forEach((v, i) => {
    if (Math.abs(Math.log(v) - Math.log(tokens)) < Math.abs(Math.log(CTX_STEPS[best]) - Math.log(tokens))) best = i;
  });
  return best;
}

function renderModel() {
  const info = model.info;
  if (!info) return;
  $("m-name").textContent = info.model || "—";
  $("m-path").textContent = info.persisted
    ? `Settings file: ${info.path}`
    : "Settings are not saved to disk for this model (no model file).";
  $("m-pending").classList.toggle("hidden", !info.reload_pending);
  const loading = info.engine && info.engine.state === "loading";
  $("m-reload").disabled = loading;
  $("m-reload").className = info.reload_pending ? "primary" : "ghost";
  $("m-reload").textContent = loading ? "Reloading…" : "Reload model";
  renderCtx();
  renderExperts();
  renderPlacement();
}

const PLACE_COLORS = ["#2f81f7", "#8957e5", "#3fb950", "#d29922", "#db61a2", "#56d4dd", "#f85149", "#a371f7"];

function placeLabel(p) {
  const d = p.device || "";
  if (d.startsWith("hip:")) return `GPU ${d.slice(4)}`;
  if (d.startsWith("cuda:")) return `CUDA ${d.slice(5)}`;
  if (d.startsWith("rpc:")) {
    const [ep, gpu] = d.slice(4).split("/");
    return gpu === undefined ? `RPC ${ep}` : `RPC ${ep} · GPU ${gpu}`;
  }
  return "CPU";
}

/* Which device holds which layers: a strip with one cell per layer, then one
 * line per device (layer ranges, uploaded weights, other parts it holds). */
function renderPlacement() {
  const rows = (model.info && model.info.placement) || [];
  const map = $("m-layer-map");
  const box = $("m-placement");
  const color = (i) => PLACE_COLORS[i % PLACE_COLORS.length];
  const owner = [];
  rows.forEach((p, i) => p.layers.forEach(([a, b]) => {
    for (let l = a; l <= b; l++) owner[l] = i;
  }));
  map.classList.toggle("hidden", owner.length === 0);
  map.innerHTML = Array.from(owner, (i, l) => i === undefined
    ? `<span title="layer ${l}"></span>`
    : `<span style="background:${color(i)}" title="layer ${l}: ${escapeHtml(placeLabel(rows[i]))}"></span>`).join("");
  if (!rows.length) {
    box.innerHTML = `<span class="muted">No model loaded.</span>`;
    return;
  }
  box.innerHTML = rows.map((p, i) => {
    const parts = [];
    if (p.n_layers) {
      const ranges = p.layers.map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(", ");
      parts.push(`${p.n_layers} layer${p.n_layers === 1 ? "" : "s"} (${ranges})`);
    }
    if (p.expert_layers) parts.push(`routed experts of ${p.expert_layers} layers`);
    parts.push(...(p.roles || []));
    const name = p.name ? ` <span class="muted">${escapeHtml(p.name)}</span>` : "";
    const size = p.weight_bytes ? bytes(p.weight_bytes) : "";
    return `<div class="place-row"><span class="sw" style="background:${color(i)}"></span>` +
      `<div class="dev">${escapeHtml(placeLabel(p))}${name}</div><div class="size">${size}</div>` +
      `<div class="detail">${escapeHtml(parts.join(" · "))}</div></div>`;
  }).join("");
}

function renderCtx() {
  const info = model.info;
  const kv = info && info.kv;
  const ctx = model.ctx || 0;
  $("ctx-slider").value = String(nearestStep(ctx));
  if (document.activeElement !== $("ctx-input")) $("ctx-input").value = String(ctx);
  if (!kv) {
    $("ctx-kv").textContent = "—";
    $("ctx-per").textContent = "—";
    $("ctx-now").textContent = "—";
    $("ctx-save").disabled = true;
    return;
  }
  const bytesNew = kvBytes(ctx);
  $("ctx-kv").textContent = bytes(bytesNew);
  const per = kv.full_bytes_per_token + kv.swa_bytes_per_token;
  $("ctx-per").textContent = `${bytes(per)} · ${kv.attention_layers} attention layer${kv.attention_layers === 1 ? "" : "s"}`;
  $("ctx-now").textContent = `${num(kv.context_tokens)} tokens (${bytes(kv.budget_bytes)})`;
  const saved = info.settings && info.settings.context_tokens;
  $("ctx-save").disabled = ctx === saved || ctx < 256;
  renderVram(bytesNew);
}

function renderVram(kvNew) {
  const s = state.lastStats || {};
  const sameGpu = s.draft_device === s.target_device;
  const total = (s.gpu_vram_total_bytes || 0) + (sameGpu ? 0 : s.draft_gpu_vram_total_bytes || 0);
  const used = (s.gpu_vram_used_bytes || 0) + (sameGpu ? 0 : s.draft_gpu_vram_used_bytes || 0);
  const weights = s.weight_bytes_on_device || 0;
  const kvLive = s.kv_bytes_on_device || 0;
  const other = Math.max(0, used - weights - kvLive);
  const need = weights + other + kvNew;
  const scale = Math.max(total, need, 1);
  const over = total > 0 && need > total;
  const pct = (v) => `${((v / scale) * 100).toFixed(2)}%`;
  $("vram-bar").innerHTML =
    `<span class="seg-weights" style="width:${pct(weights)}" title="Weights ${bytes(weights)}"></span>` +
    `<span class="seg-other" style="width:${pct(other)}" title="Other ${bytes(other)}"></span>` +
    `<span class="seg-kv${over ? " over" : ""}" style="width:${pct(kvNew)}" title="KV ${bytes(kvNew)}"></span>`;
  $("vram-legend").innerHTML =
    `<span><i class="seg-weights"></i>Weights ${bytes(weights)}</span>` +
    `<span><i class="seg-other"></i>Other in use ${bytes(other)}</span>` +
    `<span><i class="seg-kv${over ? " over" : ""}"></i>KV at this context ${bytes(kvNew)}</span>` +
    `<span>Local VRAM ${total ? bytes(total) : "n/a"}</span>`;
  const note = $("vram-note");
  if (!total) {
    note.textContent = "No local GPU reports VRAM; the KV budget sizes host memory instead.";
  } else if (over) {
    note.textContent = `Estimated ${bytes(need)} exceeds local VRAM by ${bytes(need - total)}. Pick a smaller context, or the KV cache will not fit.`;
  } else {
    note.textContent = `Estimated ${bytes(need)} of ${bytes(total)} (${((need / total) * 100).toFixed(0)}%) with the context full. Weights on RPC workers are not counted. "Other" is what the GPUs hold beyond weights and live KV (workspaces, desktop).`;
  }
}

async function saveContext() {
  try {
    model.info = await apiJson("/v1/model/settings", {
      method: "PUT",
      body: JSON.stringify({ context_tokens: model.ctx }),
    });
    renderModel();
    toast(model.info.reload_pending ? "Saved. Reload the model to apply it." : "Saved.");
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
}

async function reloadModel() {
  const ok = window.confirm(
    "Reload the model with the saved settings?\n\nRunning requests finish first, then the model is unloaded and loaded again. The server cannot answer chats until the load is done (minutes for large models)."
  );
  if (!ok) return;
  try {
    await apiJson("/v1/model/reload", { method: "POST" });
    model.lastEngineState = "loading";
    pollStatus();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
}

function renderBanner(engine) {
  if (!engine) return;
  const prev = model.lastEngineState;
  model.lastEngineState = engine.state;
  if (prev === "loading" && engine.state !== "loading") {
    toast(engine.state === "ready" ? engine.message || "Model reloaded" : engine.message || "Reload failed", engine.state !== "ready");
    model.ctx = null;
    model.layer = 0;
    model.sel = null;
    model.usage = null;
    loadModelSettings();
    loadModel();
    loadAvailable();
  }
  const loading = engine.state === "loading";
  $("m-reload").disabled = loading;
  if (loading) $("m-reload").textContent = "Reloading…";
  const b = $("reload-banner");
  const show = engine.state !== "ready";
  b.classList.toggle("hidden", !show);
  if (!show) return;
  b.classList.toggle("failed", engine.state === "failed");
  if (engine.state === "loading") {
    const secs = Math.max(0, Math.round((Date.now() - engine.since_ms) / 1000));
    const t = secs >= 60 ? `${Math.floor(secs / 60)}m ${secs % 60}s` : `${secs}s`;
    b.textContent = `Model reloading: ${engine.message || "working"} (${t}). Chat is unavailable until it finishes.`;
  } else {
    b.textContent = engine.message || "The model failed to load.";
  }
}

/* Experts */

function disabledMap() {
  return { ...((model.info && model.info.settings && model.info.settings.disabled_experts) || {}) };
}

function namesMap() {
  return { ...((model.info && model.info.settings && model.info.settings.expert_names) || {}) };
}

function layerDisabled(l) {
  return new Set(disabledMap()[String(l)] || []);
}

function renderExperts() {
  const ex = model.info && model.info.experts;
  $("experts-wrap").classList.toggle("hidden", !ex);
  if (!ex) return;
  const dis = disabledMap();
  const sel = $("layer-select");
  if (sel.options.length !== ex.n_layers) {
    sel.innerHTML = Array.from({ length: ex.n_layers }, (_, l) => `<option value="${l}">${l}</option>`).join("");
  }
  for (let l = 0; l < ex.n_layers; l++) {
    const n = (dis[String(l)] || []).length;
    const label = n ? `${l} · ${n} off` : `${l}`;
    if (sel.options[l].textContent !== label) sel.options[l].textContent = label;
  }
  sel.value = String(model.layer);
  $("rec-toggle").checked = !!ex.recording;
  renderGrid();
  renderPanel();
  renderNamed();
}

function layerCounts(l) {
  const u = model.usage;
  const ex = model.info && model.info.experts;
  if (!u || !ex || u.n_experts !== ex.n_experts) return null;
  return u.counts.slice(l * u.n_experts, (l + 1) * u.n_experts);
}

function renderGrid() {
  const ex = model.info.experts;
  const n = ex.n_experts;
  const grid = $("exp-grid");
  const cols = n >= 256 ? 32 : n >= 64 ? 16 : Math.max(4, Math.min(n, 8));
  grid.style.setProperty("--cols", cols);
  const counts = layerCounts(model.layer);
  const max = counts ? Math.max(0, ...counts) : 0;
  const total = counts ? counts.reduce((a, b) => a + b, 0) : 0;
  const off = layerDisabled(model.layer);
  const names = namesMap();
  const cells = [];
  for (let e = 0; e < n; e++) {
    const hits = counts ? counts[e] : 0;
    const name = names[`${model.layer}:${e}`];
    let style = "";
    if (max > 0 && hits > 0) {
      const a = 0.18 + 0.82 * (Math.log1p(hits) / Math.log1p(max));
      style = ` style="background:rgba(63,185,80,${a.toFixed(3)})"`;
    }
    const cls = `cell${off.has(e) ? " off" : ""}${name ? " named" : ""}${model.sel === e ? " sel" : ""}`;
    const tip = `Expert ${e}${name ? ` · ${name}` : ""}${counts ? ` · ${num(hits)} hits` : ""}${off.has(e) ? " · disabled" : ""}`;
    cells.push(`<button type="button" class="${cls}" data-e="${e}" title="${escapeHtml(tip)}" aria-label="${escapeHtml(tip)}"${style}></button>`);
  }
  grid.innerHTML = cells.join("");
  const recorded = counts ? `${num(total)} hits recorded` : "no usage recorded";
  $("exp-summary").textContent = `${n} experts · top-${ex.n_active} · ${off.size} off here · ${ex.disabled_total} off in all · ${recorded}`;
}

function renderPanel() {
  const panel = $("exp-panel");
  const e = model.sel;
  if (e === null || !model.info || !model.info.experts) {
    panel.innerHTML =
      `<p class="muted">Select an expert to name or disable it. Turn on <em>Record usage</em>, send a prompt, and the grid shows which experts it routed to.</p>`;
    return;
  }
  const l = model.layer;
  const counts = layerCounts(l);
  const hits = counts ? counts[e] : null;
  const total = counts ? counts.reduce((a, b) => a + b, 0) : 0;
  const isOff = layerDisabled(l).has(e);
  const name = namesMap()[`${l}:${e}`] || "";
  panel.innerHTML =
    `<h4>Layer ${l} · Expert ${e}</h4>` +
    `<div class="stat muted">${hits === null ? "No usage recorded" : `${num(hits)} hits${total ? ` (${((hits / total) * 100).toFixed(1)}% of this layer)` : ""}`}</div>` +
    `<label>Name <input id="exp-name" type="text" maxlength="80" placeholder="e.g. rust syntax" value="${escapeHtml(name)}"></label>` +
    `<button id="exp-toggle" type="button" class="${isOff ? "primary" : "danger"}">${isOff ? "Enable expert" : "Disable expert"}</button>` +
    (isOff ? `<p class="small-text tag-off">Disabled: the router never picks it.</p>` : "");
  $("exp-name").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") ev.target.blur();
  });
  $("exp-name").addEventListener("change", (ev) => saveName(l, e, ev.target.value));
  $("exp-toggle").addEventListener("click", () => toggleExpert(l, e));
}

function renderNamed() {
  const names = namesMap();
  const dis = disabledMap();
  const keys = new Set(Object.keys(names));
  for (const [l, ids] of Object.entries(dis)) ids.forEach((e) => keys.add(`${l}:${e}`));
  const rows = [...keys]
    .map((k) => k.split(":").map(Number))
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .map(([l, e]) => {
      const off = (dis[String(l)] || []).includes(e);
      return (
        `<tr><td>${l}</td><td>${e}</td><td>${escapeHtml(names[`${l}:${e}`] || "")}</td>` +
        `<td>${off ? '<span class="tag-off">disabled</span>' : "enabled"} ` +
        `<button type="button" class="ghost" data-jump="${l}:${e}">Show</button></td></tr>`
      );
    });
  $("named-rows").innerHTML =
    rows.join("") || `<tr><td colspan="4" class="muted">No experts named or disabled yet.</td></tr>`;
}

async function putSettings(body, okMsg) {
  try {
    model.info = await apiJson("/v1/model/settings", { method: "PUT", body: JSON.stringify(body) });
    renderModel();
    if (okMsg) toast(okMsg);
    return true;
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
    renderModel();
    return false;
  }
}

function toggleExpert(l, e) {
  const dis = disabledMap();
  const ids = new Set(dis[String(l)] || []);
  const ex = model.info.experts;
  if (ids.has(e)) {
    ids.delete(e);
  } else {
    if (ex.n_experts - ids.size - 1 < ex.n_active) {
      toast(`Layer ${l} must keep at least ${ex.n_active} experts enabled`, true);
      return;
    }
    ids.add(e);
  }
  dis[String(l)] = [...ids].sort((a, b) => a - b);
  if (!dis[String(l)].length) delete dis[String(l)];
  putSettings({ disabled_experts: dis }, ids.has(e) ? `Expert ${l}:${e} disabled` : `Expert ${l}:${e} enabled`);
}

function saveName(l, e, value) {
  const names = namesMap();
  const v = value.trim();
  if (v) names[`${l}:${e}`] = v;
  else delete names[`${l}:${e}`];
  putSettings({ expert_names: names }, v ? "Name saved" : "Name cleared");
}

function enableLayer() {
  const dis = disabledMap();
  if (!dis[String(model.layer)]) return;
  delete dis[String(model.layer)];
  putSettings({ disabled_experts: dis }, `Layer ${model.layer}: all experts enabled`);
}

function setLayer(l) {
  const ex = model.info && model.info.experts;
  if (!ex) return;
  model.layer = Math.max(0, Math.min(ex.n_layers - 1, l));
  model.sel = null;
  renderExperts();
}

async function refreshUsage() {
  try {
    model.usage = await apiJson("/v1/model/experts/usage");
    if (state.page === "model" && model.info && model.info.experts) {
      renderGrid();
      renderPanel();
    }
  } catch {
    /* status polling reports connectivity */
  }
}

function startUsagePolling() {
  stopUsagePolling();
  model.usageTimer = setInterval(() => {
    if (state.page === "model" && !document.hidden) refreshUsage();
  }, 1500);
}

function stopUsagePolling() {
  clearInterval(model.usageTimer);
  model.usageTimer = null;
}

async function setRecording(on) {
  try {
    await apiJson("/v1/model/experts/recording", { method: "POST", body: JSON.stringify({ on }) });
    if (model.info && model.info.experts) model.info.experts.recording = on;
    if (on) startUsagePolling();
    else stopUsagePolling();
    refreshUsage();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
}

/* Model picker */

const picker = { data: null, sel: null };

function fmtCtx(n) {
  if (!n) return "—";
  return n >= 1024 ? `${Math.round(n / 1024)}k` : String(n);
}

function baseName(p) {
  return String(p || "").split(/[\\/]/).pop();
}

async function loadAvailable(rescan = false) {
  if (rescan) $("models-rows").innerHTML = `<tr><td colspan="7" class="muted">Scanning…</td></tr>`;
  try {
    picker.data = await apiJson("/v1/model/available");
  } catch (e) {
    if (!(e instanceof Unauthorized)) {
      $("models-rows").innerHTML = `<tr><td colspan="7" class="error">${escapeHtml(e.message)}</td></tr>`;
    }
    return;
  }
  if (picker.sel === null && picker.data.current) picker.sel = picker.data.current;
  renderAvailable();
}

function sameFile(a, b) {
  return String(a || "").replace(/\\/g, "/").toLowerCase() === String(b || "").replace(/\\/g, "/").toLowerCase();
}

function renderAvailable() {
  const d = picker.data;
  if (!d) return;
  $("models-dir").textContent = d.models_dir ? `Scanning ${d.models_dir}` : "No models folder (set model.models_dir)";
  const q = $("models-filter").value.trim().toLowerCase();
  const models = d.entries.filter((e) => e.kind === "model");
  const shown = models.filter((e) =>
    !q || [e.name, e.rel_path, e.architecture, e.quant, e.size_label].some((v) => String(v || "").toLowerCase().includes(q))
  );
  $("models-rows").innerHTML =
    shown
      .map((e) => {
        const cur = sameFile(e.path, d.current);
        const badges = [
          cur ? `<span class="badge on">loaded</span>` : "",
          e.companions.vision.length ? `<span class="badge">vision</span>` : "",
          e.companions.mtp.length || e.nextn_layers ? `<span class="badge">MTP</span>` : "",
          e.settings ? `<span class="badge">saved settings</span>` : "",
          e.shards > 1 ? `<span class="badge">${e.shards} shards</span>` : "",
          e.error ? `<span class="badge warn">unreadable</span>` : "",
        ].join("");
        const experts = e.expert_count ? `${e.expert_count} · top-${e.expert_used_count || "?"}` : "dense";
        return (
          `<tr data-path="${escapeHtml(e.path)}" class="${sameFile(e.path, picker.sel) ? "sel" : ""}">` +
          `<td><div class="m-name">${escapeHtml(e.name || baseName(e.path))}</div>` +
          `<div class="m-file">${escapeHtml(e.rel_path)}</div><div class="badges">${badges}</div></td>` +
          `<td>${escapeHtml(e.architecture || "—")}</td>` +
          `<td>${escapeHtml([e.size_label, e.quant].filter(Boolean).join(" · ") || "—")}</td>` +
          `<td class="num">${fmtCtx(e.context_length)}</td>` +
          `<td>${escapeHtml(experts)}</td>` +
          `<td class="num">${bytes(e.size_bytes)}</td>` +
          `<td><button type="button" class="ghost small" data-detail="${escapeHtml(e.path)}">${cur ? "Details" : "Select"}</button></td>` +
          `</tr>`
        );
      })
      .join("") || `<tr><td colspan="7" class="muted">${models.length ? "No model matches the filter." : "No models found."}</td></tr>`;
  const count = (k) => d.entries.filter((e) => e.kind === k).length;
  const extra = [
    count("mmproj") && `${count("mmproj")} vision projector${count("mmproj") > 1 ? "s" : ""}`,
    count("mtp") && `${count("mtp")} MTP draft${count("mtp") > 1 ? "s" : ""}`,
    count("imatrix") && `${count("imatrix")} imatrix file${count("imatrix") > 1 ? "s" : ""}`,
  ].filter(Boolean);
  const bad = d.entries.filter((e) => e.kind === "other");
  $("models-extra").textContent =
    (extra.length ? `Also found ${extra.join(", ")} (used alongside a model, not on their own). ` : "") +
    (bad.length ? `Could not read: ${bad.map((e) => e.rel_path).join(", ")}.` : "");
  renderModelDetail();
}

function fitText(e) {
  const s = state.lastStats || {};
  const sameGpu = s.draft_device === s.target_device;
  const local = (s.gpu_vram_total_bytes || 0) + (sameGpu ? 0 : s.draft_gpu_vram_total_bytes || 0);
  const devices = state.lastDevices || [];
  const rpc = devices.filter((d) => /^rpc\[/.test(d.name || "")).reduce((a, d) => a + (d.memory_bytes || 0), 0);
  const size = e.size_bytes;
  if (!local) return { cls: "ram", text: "No local GPU memory reported: weights would sit in system RAM." };
  if (size < local * 0.9) return { cls: "ok", text: `Fits in local VRAM: ${bytes(size)} of ${bytes(local)} (KV cache and workspaces come on top).` };
  if (rpc && size < (local + rpc) * 0.9)
    return { cls: "rpc", text: `Larger than local VRAM (${bytes(local)}): the rest goes to the RPC workers (${bytes(rpc)}).` };
  return { cls: "ram", text: `Larger than GPU memory${rpc ? " including RPC" : ""} (${bytes(local + rpc)}): part of it runs from system RAM, which is much slower.` };
}

function renderModelDetail() {
  const box = $("model-detail");
  const d = picker.data;
  const e = d && d.entries.find((x) => x.kind === "model" && sameFile(x.path, picker.sel));
  if (!e) {
    box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  const cur = sameFile(e.path, d.current);
  const row = (k, v) => (v === undefined || v === null || v === "" ? "" : `<div><dt>${k}</dt><dd>${escapeHtml(v)}</dd></div>`);
  const fit = fitText(e);
  const mtpOk = e.companions.mtp.length > 0 || (e.nextn_layers || 0) > 0;
  const specNow = cur ? d.speculation : "ngram";
  const opt = (v, label, on, dis = false) =>
    `<option value="${v}"${on ? " selected" : ""}${dis ? " disabled" : ""}>${label}</option>`;
  box.innerHTML =
    `<h3>${escapeHtml(e.name || baseName(e.path))}${cur ? ' <span class="badge on">loaded</span>' : ""}</h3>` +
    (e.description ? `<p class="model-desc">${escapeHtml(e.description)}</p>` : "") +
    `<dl class="kv-grid">` +
    row("File", e.rel_path) +
    row("Architecture", e.architecture) +
    row("Parameters", e.size_label) +
    row("Quantization", e.quant) +
    row("Context (trained)", e.context_length ? `${num(e.context_length)} tokens` : "") +
    row("Layers", e.block_count) +
    row("Width", e.embedding_length) +
    row("Experts", e.expert_count ? `${e.expert_count} routed, ${e.expert_used_count} per token` : "dense") +
    row("MTP / NextN layers", e.nextn_layers) +
    row("On disk", `${bytes(e.size_bytes)}${e.shards > 1 ? ` in ${e.shards} shards` : ""}`) +
    row("License", e.license) +
    row("Quantized by", e.quantized_by) +
    row(
      "Saved settings",
      e.settings
        ? [
            e.settings.context_tokens ? `context ${num(e.settings.context_tokens)}` : "",
            e.settings.disabled_experts ? `${e.settings.disabled_experts} experts off` : "",
            e.settings.named_experts ? `${e.settings.named_experts} named` : "",
          ].filter(Boolean).join(", ") || "yes"
        : ""
    ) +
    row("Unreadable", e.error) +
    `</dl>` +
    `<div class="fit ${fit.cls}">${escapeHtml(fit.text)}</div>` +
    `<div class="load-opts">` +
    `<label>Vision projector<select id="ld-vision">${opt("", "None", !e.companions.vision.length)}${e.companions.vision
      .map((v, i) => opt(escapeHtml(v), escapeHtml(baseName(v)), i === 0))
      .join("")}</select></label>` +
    `<label>Speculation<select id="ld-spec">${opt("off", "Off", specNow === "off")}${opt("ngram", "n-gram", specNow === "ngram" || specNow === "draft")}${opt(
      "mtp",
      mtpOk ? "MTP draft" : "MTP (not available)",
      specNow === "mtp" && mtpOk,
      !mtpOk
    )}</select></label>` +
    (e.companions.mtp.length
      ? `<label>MTP draft<select id="ld-mtp">${e.companions.mtp
          .map((v, i) => opt(escapeHtml(v), escapeHtml(baseName(v)), i === 0))
          .join("")}${e.nextn_layers ? opt("", "Model's own NextN layers", false) : ""}</select></label>`
      : "") +
    `<button id="ld-go" type="button" class="primary"${e.error ? " disabled" : ""}>${cur ? "Reload with these options" : "Load this model"}</button>` +
    `</div>`;
  $("ld-go").addEventListener("click", () => loadSelectedModel(e));
}

async function loadSelectedModel(e) {
  const vision = $("ld-vision").value || null;
  const speculation = $("ld-spec").value;
  const mtpSel = $("ld-mtp");
  const mtp_draft = speculation === "mtp" && mtpSel ? mtpSel.value || null : null;
  const ok = window.confirm(
    `Load ${e.name || baseName(e.path)}?\n\nRunning requests finish first, then the current model is unloaded and this one is loaded (minutes for large models). Chat is unavailable meanwhile.`
  );
  if (!ok) return;
  try {
    await apiJson("/v1/model/load", {
      method: "POST",
      body: JSON.stringify({ path: e.path, vision, speculation, mtp_draft }),
    });
    model.lastEngineState = "loading";
    toast("Loading model…");
    pollStatus();
  } catch (err) {
    if (!(err instanceof Unauthorized)) toast(err.message, true);
  }
}

/* ---------- wiring ---------- */

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const err = $("login-error");
  const btn = ev.submitter || ev.target.querySelector("button");
  err.classList.add("hidden");
  btn.disabled = true;
  try {
    const res = await fetch("/v1/auth/login", {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ password: $("password").value }),
    });
    if (!res.ok) {
      err.textContent = res.status === 401 ? "Invalid password" : `Sign-in failed (HTTP ${res.status})`;
      err.classList.remove("hidden");
      $("password").select();
      return;
    }
    $("password").value = "";
    enterApp();
  } catch {
    err.textContent = "Server unreachable";
    err.classList.remove("hidden");
  } finally {
    btn.disabled = false;
  }
});

$("logout").addEventListener("click", async () => {
  try {
    await fetch("/v1/auth/logout", { method: "POST", credentials: "include" });
  } finally {
    onUnauthorized();
  }
});

document.querySelectorAll("nav .tab").forEach((b) => b.addEventListener("click", () => setPage(b.dataset.page)));
window.addEventListener("hashchange", () => {
  const p = location.hash.slice(1);
  if (state.authed && p && p !== state.page) setPage(p);
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.authed) startPolling();
});

$("chat-form").addEventListener("submit", sendChat);
$("stop").addEventListener("click", () => state.streaming && state.streaming.ctrl.abort());
$("prompt").addEventListener("input", autosize);
$("prompt").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    $("chat-form").requestSubmit();
  }
});
$("prompt").addEventListener("paste", (ev) => {
  const files = [...(ev.clipboardData ? ev.clipboardData.files : [])];
  if (files.length) {
    ev.preventDefault();
    addFiles(files);
  }
});
$("file-input").addEventListener("change", (ev) => {
  addFiles(ev.target.files);
  ev.target.value = "";
});
$("attachments").addEventListener("click", (ev) => {
  const i = ev.target.dataset && ev.target.dataset.rm;
  if (i === undefined) return;
  state.pending.splice(Number(i), 1);
  renderAttachments();
});
const chatLayout = document.querySelector(".chat-layout");
chatLayout.addEventListener("dragover", (ev) => {
  ev.preventDefault();
  chatLayout.classList.add("drop-hint");
});
chatLayout.addEventListener("dragleave", (ev) => {
  if (!chatLayout.contains(ev.relatedTarget)) chatLayout.classList.remove("drop-hint");
});
chatLayout.addEventListener("drop", (ev) => {
  ev.preventDefault();
  chatLayout.classList.remove("drop-hint");
  addFiles(ev.dataTransfer.files);
});
$("messages").addEventListener("click", (ev) => {
  if (!ev.target.closest(".continue-btn") || state.streaming) return;
  $("prompt").value = "Continue exactly where you stopped.";
  $("chat-form").requestSubmit();
});
$("messages").addEventListener("click", async (ev) => {
  const btn = ev.target.closest(".copy");
  if (!btn) return;
  const code = btn.closest(".codeblock").querySelector("code").textContent;
  try {
    await navigator.clipboard.writeText(code);
    btn.textContent = "Copied";
  } catch {
    btn.textContent = "Copy failed";
  }
  setTimeout(() => (btn.textContent = "Copy"), 1200);
});
$("new-chat").addEventListener("click", () => {
  if (state.streaming) state.streaming.ctrl.abort();
  state.messages = [];
  state.pending = [];
  saveMessages();
  renderAttachments();
  renderMessages();
  $("prompt").focus();
});
$("settings-toggle").addEventListener("click", () => {
  const open = $("settings").classList.toggle("hidden") === false;
  $("settings-toggle").setAttribute("aria-expanded", String(open));
});
["max-tokens", "temperature", "system-prompt"].forEach((id) => $(id).addEventListener("change", saveSettings));

$("log-level").addEventListener("change", () => scheduleLogRender(true));
$("log-filter").addEventListener("input", () => scheduleLogRender(true));
$("log-follow").addEventListener("change", () => scheduleLogRender(false));
$("log-clear").addEventListener("click", () => {
  state.logLines = [];
  scheduleLogRender(true);
});
$("log-view").addEventListener("scroll", () => {
  const view = $("log-view");
  const atEnd = nearBottom(view);
  // Scrolling up pauses follow; returning to the end resumes it.
  if ($("log-follow").checked !== atEnd) $("log-follow").checked = atEnd;
});

$("ctx-slider").addEventListener("input", (ev) => {
  model.ctx = CTX_STEPS[Number(ev.target.value)] || model.ctx;
  $("ctx-input").value = String(model.ctx);
  renderCtx();
});
$("ctx-input").addEventListener("input", (ev) => {
  const v = Math.round(Number(ev.target.value));
  if (Number.isFinite(v) && v > 0) {
    model.ctx = v;
    renderCtx();
  }
});
$("ctx-save").addEventListener("click", saveContext);
$("m-reload").addEventListener("click", reloadModel);
$("layer-select").addEventListener("change", (ev) => setLayer(Number(ev.target.value)));
$("layer-prev").addEventListener("click", () => setLayer(model.layer - 1));
$("layer-next").addEventListener("click", () => setLayer(model.layer + 1));
$("rec-toggle").addEventListener("change", (ev) => setRecording(ev.target.checked));
$("usage-reset").addEventListener("click", async () => {
  try {
    await apiJson("/v1/model/experts/usage/reset", { method: "POST" });
    refreshUsage();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
});
$("layer-enable-all").addEventListener("click", enableLayer);
$("exp-grid").addEventListener("click", (ev) => {
  const cell = ev.target.closest(".cell");
  if (!cell) return;
  const e = Number(cell.dataset.e);
  if (ev.shiftKey) {
    toggleExpert(model.layer, e);
    return;
  }
  model.sel = model.sel === e ? null : e;
  renderGrid();
  renderPanel();
});
$("named-rows").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-jump]");
  if (!btn) return;
  const [l, e] = btn.dataset.jump.split(":").map(Number);
  setLayer(l);
  model.sel = e;
  renderGrid();
  renderPanel();
  $("exp-grid").scrollIntoView({ behavior: "smooth", block: "center" });
});

$("models-rescan").addEventListener("click", () => loadAvailable(true));
$("models-filter").addEventListener("input", renderAvailable);
$("models-rows").addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr[data-path]");
  if (!tr) return;
  picker.sel = tr.dataset.path;
  renderAvailable();
  $("model-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
});

loadSettings();
boot();
