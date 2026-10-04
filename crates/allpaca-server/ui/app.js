"use strict";

const $ = (id) => document.getElementById(id);
const LEVELS = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
const POLL_MS = 1000;
const SPARK_POINTS = 60;
const LOG_CAP = 4000;
const LOG_TRIM = 500;
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
  logPaused: false,
  logSub: "",
  topoFilter: "all",
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

function icon(name) {
  return `<svg class="ic"><use href="#i-${name}"/></svg>`;
}

/* "8.32 GiB" -> number in the display face, unit in the mono face. */
function metricHtml(value) {
  const m = /^(-?[\d.,]+)\s*([A-Za-z].*)$/.exec(String(value));
  return m ? `${m[1]}<small>${m[2]}</small>` : value;
}

/* opt: tag (chip or icon html, right of the title), foot ([left, right]),
 * tone ("primary" for local GPUs, "secondary" for RPC workers). */
function card(title, value, sub = "", extra = "", opt = {}) {
  const subHtml = sub ? `<p class="metric-sub">${sub}</p>` : "";
  const foot = opt.foot ? `<div class="card-foot"><span>${opt.foot[0] || ""}</span><span>${opt.foot[1] || ""}</span></div>` : "";
  return (
    `<article class="card${opt.tone ? ` t-${opt.tone}` : ""}"><div class="card-head"><h3>${title}</h3>${opt.tag || ""}</div>` +
    `<div class="metric">${metricHtml(value)}</div>${subHtml}${extra}${foot}</article>`
  );
}

function pct(frac) {
  return `${((frac || 0) * 100).toFixed(1)}%`;
}

function setFill(el) {
  const min = Number(el.min) || 0;
  const max = Number(el.max) || 100;
  el.style.setProperty("--fill", `${(((Number(el.value) - min) / (max - min)) * 100).toFixed(1)}%`);
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
  $("brand-sub").textContent = `Node: ${location.hostname || "local"}`;
  setPage(location.hash.slice(1) || state.page);
  startPolling();
  startLogs();
  loadModel();
  // The other pages read the loaded model's quantization, context and placement.
  if (state.page !== "model") {
    loadModelSettings();
    loadAvailable();
  }
}

async function boot() {
  try {
    // An operator endpoint: with API lockdown off, /v1/status answers
    // without a session and would skip the login.
    await api("/v1/config");
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
  $("conn-text").textContent = kind === "ok" ? "Live" : kind === "down" ? "Offline" : "Connecting";
  if (kind !== "ok") $("conn-detail").textContent = "";
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
    $("chat-model").classList.toggle("soft", !id);
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
  $("hero-tps-sub").textContent = `Rolling 60s · Peak: ${Math.max(0, ...pts).toFixed(1)} tok/s`;
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

/* VRAM across the local GPUs the engine uses, split the way the bars show it. */
function localVram(s) {
  const sameGpu = s.draft_device === s.target_device;
  const total = (s.gpu_vram_total_bytes || 0) + (sameGpu ? 0 : s.draft_gpu_vram_total_bytes || 0);
  const used = (s.gpu_vram_used_bytes || 0) + (sameGpu ? 0 : s.draft_gpu_vram_used_bytes || 0);
  const weights = s.weight_bytes_on_device || 0;
  const kv = s.kv_bytes_on_device || 0;
  return { total, used, weights, kv, other: Math.max(0, used - weights - kv) };
}

/* The scanned entry of the model the engine runs (quantization, parameters). */
function currentEntry() {
  const d = picker.data;
  return (d && d.entries.find((e) => e.kind === "model" && sameFile(e.path, d.current))) || null;
}

const SPEC_LABELS = { off: "", ngram: "Speculative n-gram", mtp: "Speculative MTP", draft: "Speculative draft model" };

function specLabel() {
  return (picker.data && SPEC_LABELS[picker.data.speculation]) || "";
}

function deviceClass(d) {
  if (/^rpc\[/.test(d.name || "")) return "rpc";
  return /cpu/i.test(d.kind) ? "cpu" : "gpu";
}

function remoteGpuFor(d, remote) {
  return remote.find((g) => String(d.name || "").startsWith(`rpc[${g.endpoint}#${g.index}]`));
}

function placementFor(d, cls) {
  const rows = (model.info && model.info.placement) || [];
  if (cls === "cpu") return rows.find((p) => p.device === "cpu");
  if (cls === "rpc") {
    const m = /^rpc\[(.+)#(\d+)\]/.exec(d.name || "");
    return m ? rows.find((p) => p.device === `rpc:${m[1]}/${m[2]}` || p.device === `rpc:${m[1]}`) : undefined;
  }
  return rows.find((p) => p.device === `${/cuda/i.test(d.kind) ? "cuda" : "hip"}:${d.id}`);
}

function roleText(p) {
  if (!p) return "";
  const parts = [];
  if (p.n_layers) parts.push(`Layers ${p.layers.map(([a, b]) => (a === b ? `${a}` : `${a}-${b}`)).join(", ")}`);
  if (p.expert_layers) parts.push(`Experts of ${p.expert_layers} layers`);
  if (!parts.length) parts.push(...(p.roles || []));
  return parts.join(" · ");
}

function gpuShortName(name) {
  // The driver's product name can carry the wrong capacity (32 GB MI50s
  // report "MI50 16GB"); the measured total is shown instead.
  return String(name || "GPU")
    .replace(/^(AMD )?Radeon /, "")
    .replace(/\s+\d+\s*GB$/i, "");
}

function liveLine(g) {
  return [g.temp_c != null ? `${g.temp_c}°C` : "", g.power_w != null ? `${g.power_w} W` : ""].filter(Boolean).join(" · ");
}

function renderHero(s) {
  const e = currentEntry();
  $("hero-model").textContent = s.model_id || "—";
  const spec = specLabel();
  $("hero-tags").innerHTML = [
    s.backend ? `<span class="chip chip-primary">${icon("bolt")}${escapeHtml(s.backend.toUpperCase())} backend engine</span>` : "",
    e && (e.architecture || e.quant)
      ? `<span class="chip">${escapeHtml([e.architecture, e.quant].filter(Boolean).join(" · "))}</span>`
      : "",
    spec ? `<span class="chip chip-secondary">${spec}</span>` : "",
  ].join("");
  $("hero-params").textContent = e
    ? [e.size_label ? `${e.size_label} parameters` : "", e.quantized_by ? `quantized by ${e.quantized_by}` : ""].filter(Boolean).join(" · ")
    : "";
  const active = Number(s.active_requests) || 0;
  const sep = `<span class="sep">•</span>`;
  $("hero-sub").innerHTML = [
    `<span><span class="pip${active ? " pulse" : ""}"></span> <b>${active}</b> active request${active === 1 ? "" : "s"}</span>`,
    `<span><b>${num(s.requests_total)}</b> requests served</span>`,
    `<span><b>${num(s.tokens_generated)}</b> tokens generated</span>`,
    `<span class="url">${icon("lan")}${escapeHtml(location.origin)}</span>`,
  ].join(sep);
}

function renderMemDist(s) {
  const v = localVram(s);
  $("mem-dist").classList.toggle("hidden", !v.total);
  if (!v.total) return;
  const scale = Math.max(v.total, v.weights + v.kv + v.other, 1);
  const w = (x) => `${((x / scale) * 100).toFixed(2)}%`;
  const free = Math.max(0, v.total - v.used);
  $("mem-dist-total").textContent = `${bytes(v.used)} / ${bytes(v.total)}`;
  $("mem-dist-pct").textContent = `${pct(v.used / v.total)} allocated`;
  $("mem-dist-legend").innerHTML =
    `<span><i class="seg-weights"></i>Weights: ${bytes(v.weights)}</span>` +
    `<span><i class="seg-kv"></i>Active KV cache: ${bytes(v.kv)}</span>` +
    `<span><i class="seg-sys"></i>Other in use: ${bytes(v.other)}</span>` +
    `<span><i class="seg-free"></i>Headroom: ${bytes(free)} free</span>`;
  $("mem-dist-bar").innerHTML =
    `<span class="seg-weights" style="width:${w(v.weights)}"></span>` +
    `<span class="seg-kv" style="width:${w(v.kv)}"></span>` +
    `<span class="seg-sys" style="width:${w(v.other)}"></span>`;
}

function renderMetricCards(s, h, devices, remote) {
  const gpuName = (id, fallback) => {
    const d = devices.find((x) => x.id === id && deviceClass(x) === "gpu");
    return d && d.name ? gpuShortName(d.name) : fallback;
  };
  // No VRAM card for a GPU the engine does not report (CPU-only, one GPU).
  const vram = (used, total, id, fallback) =>
    total
      ? card(`${escapeHtml(gpuName(id, fallback))} VRAM`, bytes(used), `of ${bytes(total)} (${pct(used / total)})`, meter(used / total), {
          tone: "primary",
          tag: `<span class="chip chip-primary">Dev #${id}</span>`,
          foot: [escapeHtml(roleText(placementFor({ id, kind: s.backend || "hip" }, "gpu"))), ""],
        })
      : "";
  const ramFrac = h.ram_total_bytes ? h.ram_used_bytes / h.ram_total_bytes : 0;
  const proposed = Number(s.draft_proposed) || 0;
  const steps = Number(s.batched_decode_steps) || 0;
  const spec = specLabel();
  $("metric-cards").innerHTML = [
    card("Host RAM", bytes(h.ram_used_bytes), `of ${bytes(h.ram_total_bytes)} (${pct(ramFrac)})`, meter(ramFrac), {
      foot: ["", `${bytes(Math.max(0, h.ram_total_bytes - h.ram_used_bytes))} free`],
    }),
    vram(s.gpu_vram_used_bytes, s.gpu_vram_total_bytes, s.target_device, "GPU 0"),
    s.draft_device === s.target_device ? "" : vram(s.draft_gpu_vram_used_bytes, s.draft_gpu_vram_total_bytes, s.draft_device, "GPU 1"),
    // GPUs on the RPC workers, as each worker reports them.
    ...remote.map((g) => {
      const frac = g.mem_total_bytes ? g.mem_used_bytes / g.mem_total_bytes : 0;
      const host = String(g.endpoint || "").replace(/:\d+$/, "");
      return card(
        `${escapeHtml(gpuShortName(g.name))} VRAM`,
        bytes(g.mem_used_bytes),
        `of ${bytes(g.mem_total_bytes)} (${pct(frac)})`,
        meter(frac) + (g.util_pct != null ? `<div class="meter util" title="GPU busy"><span style="width:${g.util_pct}%"></span></div>` : ""),
        {
          tone: "secondary",
          tag: `<span class="chip chip-secondary">${escapeHtml(host)} #${g.index}</span>`,
          foot: [liveLine(g), g.util_pct != null ? `${g.util_pct}% busy` : ""],
        }
      );
    }),
    card("Weights on device", bytes(s.weight_bytes_on_device), `KV cache: ${bytes(s.kv_bytes_on_device)}`, "", {
      tag: icon("layers"),
      foot: [s.vision_bytes_on_device ? `Vision: ${bytes(s.vision_bytes_on_device)}` : "", ""],
    }),
    card("Engine latency", `${(s.itl_ms_ema || 0).toFixed(1)} ms`, "Inter-token latency (ITL)", "", {
      tag: icon("timer"),
      foot: [`<span class="secondary-text">TTFT: ${num(s.ttft_ms_last)} ms</span>`, ""],
    }),
    card("Host CPU", `${(h.cpu_usage || 0).toFixed(0)}%`, "", meter((h.cpu_usage || 0) / 100)),
    card("GPU util (aggregate)", `${((s.gpu_util || 0) * 100).toFixed(0)}%`, "Local GPU compute average", meter(s.gpu_util || 0), {
      tone: "primary",
      tag: icon("gauge"),
    }),
    card("KV pages", `${num(s.vram_pages)} VRAM`, `${num(s.dram_pages)} DRAM spillover`, "", {
      tag: icon("chip"),
      foot: [`${num(s.page_migrations)} page migrations`, ""],
    }),
    card(
      "Speculation",
      proposed ? pct(s.acceptance_rate) : "idle",
      proposed ? `${num(s.draft_accepted)} / ${num(proposed)} drafts accepted` : "no drafts proposed",
      proposed ? meter(s.acceptance_rate).replace(/ (warn|hot)"/, '"') : "",
      { tone: "secondary", tag: spec ? `<span class="chip chip-secondary">${spec.replace("Speculative ", "")} active</span>` : "" }
    ),
    card(
      "Batched decode",
      steps ? `${((Number(s.batched_decode_rows) || 0) / steps).toFixed(2)} rows` : "—",
      steps ? `avg per step · ${num(steps)} steps` : "single stream so far"
    ),
    card("RPC & network", `${num(s.rpc_roundtrips)} rtt`, `${bytes(s.rpc_activation_bytes)} activations sent`, "", {
      tag: icon("hub"),
      foot: [remote.length ? `<span class="secondary-text">${remote.length} remote GPU${remote.length === 1 ? "" : "s"}</span>` : "Local only", ""],
    }),
  ].join("");
}

function renderTopology(s, h, devices, remote) {
  const memCell = (used, total) => {
    if (!total) return `<span class="soft">—</span>`;
    if (used == null) return `<div class="mem"><div class="mem-line"><span>${bytes(total)}</span></div></div>`;
    return (
      `<div class="mem"><div class="mem-line"><b>${bytes(used)}</b><span>${bytes(total)} (${((used / total) * 100).toFixed(0)}%)</span></div>` +
      `${meter(used / total)}</div>`
    );
  };
  const count = { all: devices.length, local: 0, rpc: 0 };
  const rows = devices.map((d) => {
    const cls = deviceClass(d);
    count[cls === "rpc" ? "rpc" : "local"] += 1;
    if (state.topoFilter !== "all" && (state.topoFilter === "rpc") !== (cls === "rpc")) return "";
    const g = cls === "rpc" ? remoteGpuFor(d, remote) : null;
    let used = null;
    let total = d.memory_bytes;
    let util = "—";
    let live = "—";
    let sub = "";
    if (cls === "cpu") {
      used = h.ram_used_bytes;
      total = h.ram_total_bytes || d.memory_bytes;
      util = `${(h.cpu_usage || 0).toFixed(0)}%`;
    } else if (g) {
      used = g.mem_used_bytes;
      total = g.mem_total_bytes || d.memory_bytes;
      if (g.util_pct != null) util = `${g.util_pct}%`;
      live = liveLine(g) || "—";
      sub = gpuShortName(g.name);
    } else if (cls === "gpu" && d.id === s.target_device && s.gpu_vram_total_bytes) {
      used = s.gpu_vram_used_bytes;
      total = s.gpu_vram_total_bytes;
    } else if (cls === "gpu" && d.id === s.draft_device && s.draft_gpu_vram_total_bytes) {
      used = s.draft_gpu_vram_used_bytes;
      total = s.draft_gpu_vram_total_bytes;
    }
    const role = roleText(placementFor(d, cls)) || (cls === "cpu" ? "Host orchestrator" : "");
    const kindCls = cls === "rpc" ? "chip-secondary" : cls === "gpu" ? "chip-primary" : "";
    return (
      `<tr class="${cls}"><td class="id">${escapeHtml(d.id)}</td>` +
      `<td><span class="chip ${kindCls}">${icon(cls === "rpc" ? "hub" : "chip")}${escapeHtml(d.kind)}${cls === "rpc" ? " (RPC)" : ""}</span></td>` +
      `<td><div class="dev-name">${escapeHtml(d.name)}</div>${sub ? `<div class="dev-sub">${escapeHtml(sub)}</div>` : ""}</td>` +
      `<td>${live}</td><td>${memCell(used, total)}</td><td class="util">${util}</td>` +
      `<td>${role ? `<span class="role">${escapeHtml(role)}</span>` : `<span class="role idle">Standby</span>`}</td></tr>`
    );
  });
  $("device-rows").innerHTML = rows.join("") || `<tr><td colspan="7" class="soft">No devices</td></tr>`;
  $("topo-count").textContent = `${count.all} device${count.all === 1 ? "" : "s"} registered`;
  document.querySelectorAll("#topo-filter button").forEach((b) => {
    const k = b.dataset.topo;
    b.classList.toggle("on", k === state.topoFilter);
    b.textContent = `${k === "all" ? "All" : k === "local" ? "Local" : "RPC Workers"} (${count[k]})`;
  });
}

function renderChrome(s, devices, remote) {
  const v = localVram(s);
  const tps = state.tps.length ? state.tps[state.tps.length - 1] : 0;
  const n = devices.length;
  $("conn-detail").innerHTML = n ? `· ${n} device${n === 1 ? "" : "s"} · <b>${tps.toFixed(1)} t/s</b>` : "";
  $("foot-left").innerHTML =
    `Allpaca Distributed Engine · Node <b>${escapeHtml(location.hostname || "local")}</b>` +
    (s.backend ? ` · ${escapeHtml(s.backend.toUpperCase())}` : "");
  const rpcMem = remote.reduce((a, g) => a + (g.mem_used_bytes || 0), 0);
  const rpcTotal = remote.reduce((a, g) => a + (g.mem_total_bytes || 0), 0);
  $("foot-right").innerHTML =
    (v.total + rpcTotal ? `VRAM: <span class="c">${bytes(v.used + rpcMem)} / ${bytes(v.total + rpcTotal)}</span> · ` : "") +
    `${n} device${n === 1 ? "" : "s"}`;
}

function renderDashboard(data) {
  state.status = data;
  if (data.stats) state.lastStats = data.stats;
  if (data.devices && data.devices.length) state.lastDevices = data.devices;
  state.lastRemote = data.remote_gpus || [];
  renderBanner(data.engine);
  if (state.page === "model" && model.info) renderCtx();
  sampleThroughput(data.stats || {});
  renderSpark();
  renderStatus();
}

/* Draws the last status for the page on screen (also on a page switch). */
function renderStatus() {
  const data = state.status;
  if (!data) return;
  const s = data.stats || {};
  const h = data.host || {};
  const devices = data.devices || [];
  const remote = state.lastRemote || [];
  renderChrome(s, devices, remote);
  if (state.page === "dashboard") {
    renderHero(s);
    renderMemDist(s);
    renderMetricCards(s, h, devices, remote);
    renderTopology(s, h, devices, remote);
  } else if (state.page === "chat") {
    renderChatTelemetry();
  } else if (state.page === "logs") {
    renderLogFoot();
  }
}

/* ---------- pages ---------- */

const PAGES = ["dashboard", "chat", "model", "logs", "api", "settings"];

function setPage(name) {
  if (!PAGES.includes(name)) name = "dashboard";
  state.page = name;
  document.querySelectorAll("nav .tab").forEach((b) => {
    const on = b.dataset.page === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-current", on ? "page" : "false");
  });
  PAGES.forEach((p) => $(`page-${p}`).classList.toggle("hidden", p !== name));
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  renderStatus();
  startLogs();
  scheduleLogRender(true);
  if (name === "logs") renderLogFoot();
  if (name === "chat") {
    renderMessages();
    renderChatTelemetry();
    $("prompt").focus();
  }
  if (name === "model") {
    loadModelSettings();
    loadAvailable();
  }
  if (name === "settings") loadConfig();
  if (name === "api") loadApiPage();
}

/* ---------- chat ---------- */

function loadSettings() {
  const s = loadJson(SETTINGS_KEY, {});
  // 512 was the old default, saved alongside other settings: not a choice.
  if (s.maxTokens && String(s.maxTokens) !== "512") $("max-tokens").value = s.maxTokens;
  if (s.temperature !== undefined) $("temperature").value = s.temperature;
  if (s.topP !== undefined) $("top-p").value = s.topP;
  if (s.system) $("system-prompt").value = s.system;
  setDrawer(!!s.drawer);
  syncSampling();
}

function saveSettings() {
  saveJson(SETTINGS_KEY, {
    maxTokens: $("max-tokens").value,
    temperature: $("temperature").value,
    topP: $("top-p").value,
    system: $("system-prompt").value,
    drawer: !$("drawer").classList.contains("hidden"),
  });
}

/* Slider fill and the number next to each sampling slider. */
function syncSampling() {
  ["temperature", "top-p"].forEach((id) => {
    setFill($(id));
    $(`${id}-out`).textContent = Number($(id).value).toFixed(2);
  });
}

function setDrawer(open) {
  $("drawer").classList.toggle("hidden", !open);
  $("drawer-toggle").setAttribute("aria-expanded", String(open));
  if (open) renderChatTelemetry();
}

/* Splits a reply into its <think> reasoning and the answer. Some chat
 * templates open the block in the prompt, so only the closing tag arrives. */
function splitThink(src) {
  const open = /^\s*<think>/.exec(src);
  const close = src.indexOf("</think>");
  if (!open && close < 0) return { think: "", body: src, thinking: false };
  const start = open ? open[0].length : 0;
  if (close < 0) return { think: src.slice(start).trim(), body: "", thinking: true };
  return { think: src.slice(start, close).trim(), body: src.slice(close + 8).replace(/^\s+/, ""), thinking: false };
}

function metaHtml(meta) {
  const [first, ...rest] = String(meta || "").split(" · ");
  const head = / tokens$/.test(first) ? `<span class="tok">${escapeHtml(first)}</span>` : escapeHtml(first);
  return [head, ...rest.map(escapeHtml)].join(" · ");
}

function messageHtml(m, i) {
  const live = state.streaming && state.streaming.index === i;
  const images = (m.images || []).map((src) => `<img src="${escapeHtml(src)}" alt="attached image">`).join("");
  if (m.role !== "assistant") {
    const at = m.at ? `<div class="msg-meta">${escapeHtml(new Date(m.at).toLocaleTimeString())}</div>` : "";
    return (
      `<div class="msg user" data-i="${i}">` +
      (images ? `<div class="msg-images">${images}</div>` : "") +
      (m.content || !images ? `<div class="bubble">${escapeHtml(m.content)}</div>` : "") +
      at +
      `</div>`
    );
  }
  const last = i === state.messages.length - 1;
  const t = m.error ? { think: "", body: m.content, thinking: false } : splitThink(m.content);
  const thinking = live && t.thinking;
  const words = t.think ? t.think.split(/\s+/).length : 0;
  const think =
    t.think || thinking
      ? `<details class="think"${thinking || m.thinkOpen ? " open" : ""}><summary>${icon("brain")}Reasoning process ` +
        `<span class="n">(${thinking ? "thinking…" : `${num(words)} words`})</span>` +
        `<span class="exp">${icon("chevron")}</span></summary>` +
        `<div class="think-body${thinking ? " cursor" : ""}"><span>${escapeHtml(t.think)}</span></div></details>`
      : "";
  const typing = live && !thinking;
  const bubble =
    t.body || typing ? `<div class="bubble${typing ? " cursor" : ""}">${m.error ? escapeHtml(t.body) : renderMarkdown(t.body)}</div>` : "";
  const idle = !state.streaming;
  const more = m.truncated && last && idle ? `<button type="button" class="secondary small continue-btn">Continue</button>` : "";
  const actions =
    !live && m.content && !m.error
      ? `<span class="spacer"></span>` +
        `<button type="button" class="icon-btn copy-msg" title="Copy reply" aria-label="Copy reply">${icon("copy")}</button>` +
        (last && idle
          ? `<button type="button" class="icon-btn regen" title="Regenerate" aria-label="Regenerate">${icon("refresh")}</button>`
          : "")
      : "";
  const meta = m.meta || actions ? `<div class="msg-meta"><span>${metaHtml(m.meta)}</span>${more}${actions}</div>` : "";
  return (
    `<div class="msg assistant" data-i="${i}"><div class="a-card${m.error ? " error-card" : ""}">` +
    think + bubble + meta +
    `</div></div>`
  );
}

/* Ribbon above the conversation and, when open, the drawer's live gauges. */
function renderChatTelemetry() {
  const s = state.lastStats || {};
  const e = currentEntry();
  const id = s.model_id;
  $("chat-model").textContent = id || "no model";
  $("chat-model").classList.toggle("soft", !id);
  $("prompt").placeholder = id ? `Send a message or code prompt to ${id}…` : "Send a message…";
  const quant = [s.backend ? s.backend.toUpperCase() : "", e && e.quant].filter(Boolean).join(" · ");
  $("chat-quant").textContent = quant;
  $("chat-quant").classList.toggle("hidden", !quant);
  const kv = model.info && model.info.kv;
  $("chat-ctx").classList.toggle("hidden", !kv);
  if (kv) $("chat-ctx").innerHTML = `${icon("chip")}Ctx limit: <b>${num(kv.context_tokens)}</b> tokens`;
  const r = state.lastReply;
  $("chat-last").classList.toggle("hidden", !r);
  if (r) {
    $("chat-last").innerHTML =
      `${icon("gauge")}Last TTFT: <span class="p">${num(r.ttft)} ms</span>` + (r.tps ? `<span class="c">${r.tps.toFixed(1)} tok/s</span>` : "");
  }
  const spec = specLabel().replace("Speculative ", "");
  const proposed = Number(s.draft_proposed) || 0;
  $("chat-spec").classList.toggle("hidden", !spec);
  if (spec) $("chat-spec").innerHTML = `${icon("hub")}${spec}${proposed ? ` · ${pct(s.acceptance_rate)} accepted` : ""}`;
  $("composer-tags").innerHTML =
    (s.vision_weight_bytes_on_device ? `<span class="chip chip-primary">${icon("image")}Vision projector active</span>` : "") +
    (spec ? `<span class="chip">Draft: ${spec}</span>` : "");
  if ($("drawer").classList.contains("hidden")) return;

  const node = (cls, label, used, total) =>
    `<div class="node ${cls}"><div class="node-line"><span><span class="pip"></span>${escapeHtml(label)}</span>` +
    `<span>${bytes(used)} / ${bytes(total)}</span></div>${meter(total ? used / total : 0)}</div>`;
  const devices = state.lastDevices || [];
  const local = (id, used, total) => {
    const d = devices.find((x) => x.id === id && deviceClass(x) === "gpu");
    return total ? node("", `Local: ${gpuShortName(d && d.name)}`, used, total) : "";
  };
  $("drawer-nodes").innerHTML =
    [
      local(s.target_device, s.gpu_vram_used_bytes, s.gpu_vram_total_bytes),
      s.draft_device === s.target_device ? "" : local(s.draft_device, s.draft_gpu_vram_used_bytes, s.draft_gpu_vram_total_bytes),
      ...(state.lastRemote || []).map((g) => node("rpc", `RPC[${g.endpoint}#${g.index}]`, g.mem_used_bytes, g.mem_total_bytes)),
    ].join("") || `<span class="soft">No GPU reports memory.</span>`;
  $("drawer-spec").innerHTML = spec
    ? `<div class="kv-line"><span>Mode</span><b>${spec}</b></div>` +
      `<div class="kv-line"><span>Drafts accepted</span><b>${num(s.draft_accepted)} / ${num(proposed)}</b></div>` +
      `<div class="kv-line"><span>Acceptance rate</span><b class="p">${proposed ? pct(s.acceptance_rate) : "—"}</b></div>`
    : `<span class="soft">Speculation is off for this model. Change it on Models &amp; VRAM.</span>`;
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
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
  const topP = Number($("top-p").value);
  return {
    messages: msgs,
    stream: true,
    max_tokens: Math.max(1, Number($("max-tokens").value) || 4096),
    temperature: Number.isFinite(temp) ? temp : 0.7,
    top_p: topP > 0 && topP <= 1 ? topP : 0.95,
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
  const user = { role: "user", content: prompt, at: Date.now() };
  if (state.pending.length) user.images = state.pending.slice();
  state.messages.push(user);
  state.pending = [];
  renderAttachments();
  $("prompt").value = "";
  autosize();
  await runCompletion();
}

/* Streams a reply to the conversation as it stands (it ends with a user turn). */
async function runCompletion() {
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
      const tps = gen > 0 && chunks > 1 ? (chunks - 1) / gen : 0;
      const rate = tps ? ` · ${tps.toFixed(1)} tok/s` : "";
      state.lastReply = { ttft: Math.round(firstAt - started), tps };
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
    renderChatTelemetry();
  }
}

function regenerate() {
  if (state.streaming) return;
  while (state.messages.length && state.messages[state.messages.length - 1].role === "assistant") state.messages.pop();
  if (!state.messages.length) return;
  runCompletion();
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
  setLogPipe("connecting");
  src.onopen = () => {
    state.logRetry = 1000;
    setLogPipe("ok");
  };
  src.onmessage = (ev) => {
    let line;
    try {
      line = JSON.parse(ev.data);
    } catch {
      return;
    }
    // The stream replays the ring on connect: skip what the snapshot already
    // gave us. A lower seq with a newer time is a restarted server.
    const top = state.logLines[state.logLines.length - 1];
    if (top && line.seq <= top.seq && line.ts_ms <= top.ts_ms) return;
    state.logLines.push(line);
    // Trim in batches: every trim rebuilds the view.
    if (state.logLines.length > LOG_CAP + LOG_TRIM) {
      state.logLines.splice(0, state.logLines.length - LOG_CAP);
      state.logRendered = -1; // force a full rebuild
    }
    scheduleLogRender(false);
  };
  src.onerror = () => {
    src.close();
    state.logSource = null;
    setLogPipe("connecting");
    if (!state.authed) return;
    const wait = state.logRetry;
    state.logRetry = Math.min(state.logRetry * 2, 15000);
    setTimeout(() => {
      if (state.authed) startLogs();
    }, wait);
  };
  state.logSource = src;
}

function stopLogs() {
  if (state.logSource) state.logSource.close();
  state.logSource = null;
}

function logRoot(l) {
  return String(l.target || "").split("::")[0] || "other";
}

function logTime(l, dateToo) {
  const ts = new Date(l.ts_ms || 0);
  if (Number.isNaN(ts.getTime())) return "";
  const iso = ts.toISOString().replace("Z", "");
  return dateToo ? iso.replace("T", " ") : iso.slice(11);
}

function logLineEl(l) {
  const lvl = (l.level || "INFO").toUpperCase();
  const div = document.createElement("div");
  div.className = `log-line lvl-${lvl}`;
  div.innerHTML =
    `<span class="ln">${l.seq ?? ""}</span><span class="ts">${logTime(l, true)}</span><span class="lvl">${lvl}</span>` +
    `<span class="tgt" title="${escapeHtml(l.target || "")}">${escapeHtml(l.target || "")}</span>` +
    `<span class="msg">${escapeHtml(l.message || "")}</span>`;
  return div;
}

/* The filter box takes a regex; text that is not a valid one is matched as written. */
function logMatcher() {
  const text = $("log-filter").value.trim();
  if (!text) return null;
  try {
    const re = new RegExp(text, "i");
    return (s) => re.test(s);
  } catch {
    const needle = text.toLowerCase();
    return (s) => s.toLowerCase().includes(needle);
  }
}

function logPasses(l, min, match) {
  if (LEVELS.indexOf((l.level || "INFO").toUpperCase()) < min) return false;
  if (state.logSub && logRoot(l) !== state.logSub) return false;
  return !match || match(`${l.target || ""} ${l.message || ""}`);
}

/* `full` rebuilds the view; `user` marks a change the operator made (filter,
 * level, clear), which redraws even while the stream is paused. */
function scheduleLogRender(full, user = false) {
  if (full) state.logRendered = -1;
  if (user) state.logForce = true;
  if (state.logFrame) return;
  state.logFrame = requestAnimationFrame(() => {
    state.logFrame = 0;
    if (state.page === "dashboard") renderEvents();
    else if (state.page === "logs") renderLogs();
  });
}

function setLogPipe(kind) {
  state.logPipe = kind;
  const paused = state.logPaused && kind === "ok";
  const el = $("log-pipe");
  el.classList.toggle("ok", kind === "ok" && !paused);
  el.classList.toggle("paused", paused);
  $("log-pipe-text").textContent = `Log stream: ${paused ? "paused" : kind === "ok" ? "active" : "connecting"}`;
  $("event-state").textContent = kind === "ok" ? "Streaming" : "Reconnecting…";
}

/* One chip per crate that has logged, taken from the lines' targets. */
function renderSubsystems() {
  const roots = [...new Set(state.logLines.map(logRoot))].sort();
  const key = `${roots.join(",")}|${state.logSub || ""}`;
  if (key === state.logSubKey) return;
  state.logSubKey = key;
  $("log-subsystems").innerHTML =
    `<button type="button" data-sub="" class="${state.logSub ? "" : "on"}">All subsystems</button>` +
    roots
      .map((r) => `<button type="button" data-sub="${escapeHtml(r)}" class="${state.logSub === r ? "on" : ""}">${escapeHtml(r)}</button>`)
      .join("");
}

function renderLogs() {
  const view = $("log-view");
  renderSubsystems();
  if (!state.logPaused || state.logForce) {
    state.logForce = false;
    const min = LEVELS.indexOf($("log-level").value);
    const match = logMatcher();
    const rebuild = state.logRendered < 0 || state.logRendered > state.logLines.length;
    const from = rebuild ? 0 : state.logRendered;
    const frag = document.createDocumentFragment();
    for (let i = from; i < state.logLines.length; i++) {
      const l = state.logLines[i];
      if (logPasses(l, min, match)) frag.appendChild(logLineEl(l));
    }
    if (rebuild) view.replaceChildren(frag);
    else view.appendChild(frag);
    state.logRendered = state.logLines.length;
    if ($("log-follow").checked) view.scrollTop = view.scrollHeight;
  }
  $("log-count").textContent = `${view.childElementCount.toLocaleString()} of ${state.logLines.length.toLocaleString()} lines`;
}

/* Dashboard: the newest few lines at INFO and above. */
function renderEvents() {
  const min = LEVELS.indexOf("INFO");
  const rows = [];
  for (let i = state.logLines.length - 1; i >= 0 && rows.length < 6; i--) {
    const l = state.logLines[i];
    const lvl = (l.level || "INFO").toUpperCase();
    if (LEVELS.indexOf(lvl) < min) continue;
    rows.push(
      `<div class="lvl-${lvl}"><span class="ts">[${logTime(l)}]</span> ` +
        `<span class="tgt">${escapeHtml(String(l.target || "").split("::").pop())}</span> ${escapeHtml(l.message || "")}</div>`
    );
  }
  $("event-stream").innerHTML = rows.reverse().join("") || `<span class="soft">No log lines yet.</span>`;
}

/* Summary cards under the terminal, from the last status and the buffer. */
function renderLogFoot() {
  const s = state.lastStats || {};
  const devices = state.lastDevices || [];
  const remote = state.lastRemote || [];
  const nRpc = devices.filter((d) => deviceClass(d) === "rpc").length;
  const v = localVram(s);
  const used = v.used + remote.reduce((a, g) => a + (g.mem_used_bytes || 0), 0);
  const total = v.total + remote.reduce((a, g) => a + (g.mem_total_bytes || 0), 0);
  const tps = state.tps.length ? state.tps[state.tps.length - 1] : 0;
  const proposed = Number(s.draft_proposed) || 0;
  let errors = 0;
  let warns = 0;
  let recent = 0;
  const since = Date.now() - 5000;
  for (const l of state.logLines) {
    const lvl = (l.level || "").toUpperCase();
    if (lvl === "ERROR") errors += 1;
    else if (lvl === "WARN") warns += 1;
    if (l.ts_ms > since) recent += 1;
  }
  const tile = (title, value, sub, badge, cls = "") =>
    `<article class="card"><div><div class="card-head"><h3>${title}</h3></div><div class="metric">${metricHtml(value)}</div>` +
    `<p class="metric-sub">${sub}</p></div><span class="badge-ic ${cls}">${icon(badge)}</span></article>`;
  $("log-foot").innerHTML =
    tile("Cluster devices", `${devices.length} live`, `Local: ${devices.length - nRpc} · Remote RPC: ${nRpc}`, "hub") +
    tile("VRAM allocation", total ? bytes(used) : "n/a", total ? `of ${bytes(total)} (${pct(used / total)})` : "no GPU reports memory", "chip", "c") +
    tile("Throughput rate", `${tps.toFixed(1)} tok/s`, proposed ? `Speculation ${pct(s.acceptance_rate)} accepted` : "No drafts proposed", "trend") +
    tile(
      "Log health",
      errors ? `${num(errors)} errors` : "No errors",
      `${num(warns)} warning${warns === 1 ? "" : "s"} in the buffer`,
      errors ? "close" : "check",
      errors ? "e" : ""
    );
  $("log-rate").textContent = `Stream rate: ${(recent / 5).toFixed(1)} lines/sec`;
  const chip = $("log-backend");
  chip.classList.toggle("hidden", !s.backend);
  if (s.backend) chip.textContent = `${s.backend.toUpperCase()} · ${devices.length} device${devices.length === 1 ? "" : "s"}`;
}

/* ---------- model page ---------- */

const CTX_STEPS = [
  1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288,
  786432, 1048576,
];

// Slider positions that carry a label.
const CTX_TICKS = [0, 3, 5, 7, 9, 12, 15];

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
  const layers = (info.placement || []).reduce((a, p) => a + (p.n_layers || 0), 0);
  $("m-active").classList.toggle("hidden", !layers);
  $("m-active").innerHTML = `<span class="pip"></span>Active · ${layers} layer${layers === 1 ? "" : "s"} resident`;
  const loading = info.engine && info.engine.state === "loading";
  $("m-reload").disabled = loading;
  $("m-reload").className = info.reload_pending ? "primary" : "secondary";
  $("m-reload-text").textContent = loading ? "Reloading…" : "Reload Model";
  renderStatus();
  renderCtx();
  renderExperts();
  renderPlacement();
}

const PLACE_COLORS = ["#4edea3", "#4cd7f6", "#b090ff", "#f59e0b", "#f472b6", "#10b981", "#ef4444", "#86948a"];

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
  const step = nearestStep(ctx);
  $("ctx-slider").value = String(step);
  setFill($("ctx-slider"));
  $("ctx-ticks").innerHTML = CTX_TICKS.map(
    (i) => `<span class="${i === step ? "on" : ""}" style="left:${((i / (CTX_STEPS.length - 1)) * 100).toFixed(2)}%">${num(CTX_STEPS[i])}</span>`
  ).join("");
  if (document.activeElement !== $("ctx-input")) $("ctx-input").value = String(ctx);
  const trained = currentEntry() && currentEntry().context_length;
  $("ctx-facts").textContent = [
    trained ? `Trained context: ${num(trained)} tokens` : "",
    kv ? `KV page: ${num(kv.page_tokens)} tokens` : "",
  ].filter(Boolean).join(" · ");
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
  const { total, weights, other } = localVram(state.lastStats || {});
  const need = weights + other + kvNew;
  const scale = Math.max(total, need, 1);
  const over = total > 0 && need > total;
  const w = (v) => `${((v / scale) * 100).toFixed(2)}%`;
  $("vram-title").textContent = `VRAM allocation breakdown${total ? ` (local ${bytes(total)} total)` : ""}`;
  $("vram-pct").textContent = total ? `${pct(need / total)} utilized` : "";
  $("vram-pct").className = `mono ${over ? "error" : "primary-text"}`;
  $("vram-bar").innerHTML =
    `<span class="seg-weights" style="width:${w(weights)}" title="Weights ${bytes(weights)}"></span>` +
    `<span class="seg-other" style="width:${w(other)}" title="Other ${bytes(other)}"></span>` +
    `<span class="seg-kv${over ? " over" : ""}" style="width:${w(kvNew)}" title="KV ${bytes(kvNew)}"></span>`;
  $("vram-legend").innerHTML =
    `<span><i class="seg-weights"></i>Weights ${bytes(weights)}</span>` +
    `<span><i class="seg-other"></i>Other in use ${bytes(other)}</span>` +
    `<span><i class="seg-kv${over ? " over" : ""}"></i>KV at this context ${bytes(kvNew)}</span>` +
    (total ? `<span><i class="seg-free"></i>Free headroom ${bytes(Math.max(0, total - need))}</span>` : "");
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
  if (loading) $("m-reload-text").textContent = "Reloading…";
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
  renderExpertPlacement();
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
      style = ` style="background:rgba(78,222,163,${a.toFixed(3)})"`;
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

/* Expert placement: which device holds each layer's routed experts, and
 * pins that move a layer's experts on the next reload. */

function expertPins() {
  return { ...((model.info && model.info.settings && model.info.settings.expert_homes) || {}) };
}

/* A pin against where the experts are now. */
function pinApplied(pin, device) {
  return pin === "local" ? !String(device).startsWith("rpc:") : pin === device;
}

function renderExpertPlacement() {
  const ep = model.info && model.info.expert_placement;
  $("exp-place").classList.toggle("hidden", !ep);
  if (!ep) return;
  const pins = expertPins();
  // Colours follow the Layer placement card where a device appears in both.
  const rows = (model.info.placement || []).map((p) => p.device);
  const devices = [...new Set([...rows, ...ep.layers.map((l) => l.device), ...ep.workers])];
  const color = (dev) => PLACE_COLORS[devices.indexOf(dev) % PLACE_COLORS.length];
  const label = (dev) => placeLabel({ device: dev });
  const nPins = Object.keys(pins).length;
  const pending = ep.layers.filter((l, i) => pins[i] && !pinApplied(pins[i], l.device)).length;
  $("exp-place-count").textContent =
    `${nPins} pinned` + (pending ? ` · ${pending} waiting for a reload` : "");
  $("exp-place-count").className = `chip ${pending ? "chip-warn" : nPins ? "chip-primary" : ""}`;
  $("exp-place-clear").disabled = !nPins;
  $("exp-place-map").innerHTML = ep.layers
    .map((l, i) => {
      const pin = pins[i];
      const cls = [i === model.layer ? "sel" : "", pin ? "pinned" : "", pin && !pinApplied(pin, l.device) ? "pending" : ""]
        .filter(Boolean)
        .join(" ");
      const tip = `Layer ${i}: experts on ${label(l.device)} (${bytes(l.bytes)})` + (pin ? ` · pinned to ${pin === "local" ? "local" : label(pin)}` : "");
      return `<span class="${cls}" data-l="${i}" role="option" aria-selected="${i === model.layer}" style="background:${color(l.device)}" title="${escapeHtml(tip)}"></span>`;
    })
    .join("");
  const totals = new Map();
  ep.layers.forEach((l) => {
    const t = totals.get(l.device) || { n: 0, bytes: 0 };
    t.n += 1;
    t.bytes += l.bytes;
    totals.set(l.device, t);
  });
  $("exp-place-legend").innerHTML = devices
    .filter((dev) => totals.has(dev) || ep.workers.includes(dev))
    .map((dev) => {
      const t = totals.get(dev) || { n: 0, bytes: 0 };
      return (
        `<div class="place-row"><span class="sw" style="background:${color(dev)}"></span>` +
        `<div class="dev">${escapeHtml(label(dev))}</div><div class="size">${t.n ? bytes(t.bytes) : ""}</div>` +
        `<div class="detail">${t.n ? `experts of ${t.n} layer${t.n === 1 ? "" : "s"}` : "no experts yet"}</div></div>`
      );
    })
    .join("");
  const cur = ep.layers[model.layer];
  if (!cur) return;
  const pin = pins[model.layer] || "";
  $("exp-place-layer").textContent = `Layer ${model.layer}`;
  $("exp-place-now").textContent =
    `${bytes(cur.bytes)} of experts, now on ${label(cur.device)}` +
    (pin && !pinApplied(pin, cur.device) ? ` · moves to ${pin === "local" ? "the local GPU" : label(pin)} on reload` : "");
  const body = (model.info.placement || []).find((p) => (p.layers || []).some(([a, b]) => model.layer >= a && model.layer <= b));
  const opt = (v, text) => `<option value="${escapeHtml(v)}"${v === pin ? " selected" : ""}>${escapeHtml(text)}</option>`;
  $("exp-place-select").innerHTML =
    opt("", "Automatic (local while it fits, else a worker)") +
    opt("local", `Local, beside the layer${body ? ` on ${placeLabel(body)}` : ""}`) +
    ep.workers.map((w) => opt(w, label(w))).join("");
}

async function setExpertPin(layer, home) {
  const pins = expertPins();
  if (home) pins[layer] = home;
  else delete pins[layer];
  await putSettings({ expert_homes: pins }, home ? `Layer ${layer} pinned. Reload the model to move its experts.` : `Layer ${layer} back to automatic.`);
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
  $("models-dir").textContent = d.models_dir ? `Scanning: ${d.models_dir}` : "No models folder (set model.models_dir)";
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
          cur ? `<span class="chip chip-primary outline">loaded</span>` : "",
          e.companions.vision.length ? `<span class="chip chip-secondary">vision</span>` : "",
          e.companions.mtp.length || e.nextn_layers ? `<span class="chip chip-tertiary">MTP</span>` : "",
          e.expert_count ? `<span class="chip chip-secondary">MoE ${e.expert_count}E</span>` : "",
          e.settings ? `<span class="chip">saved settings</span>` : "",
          e.shards > 1 ? `<span class="chip">${e.shards} shards</span>` : "",
          e.error ? `<span class="chip chip-warn">unreadable</span>` : "",
        ].join("");
        const moe = e.expert_count ? " hl" : "";
        const topology = e.expert_count ? `${e.expert_count} · top-${e.expert_used_count || "?"}` : "dense";
        const cls = [sameFile(e.path, picker.sel) ? "sel" : "", cur ? "cur" : ""].filter(Boolean).join(" ");
        return (
          `<tr data-path="${escapeHtml(e.path)}" class="${cls}">` +
          `<td><div class="m-name">${escapeHtml(e.name || baseName(e.path))}${badges}</div>` +
          `<div class="m-file">${escapeHtml(e.rel_path)}</div></td>` +
          `<td>${escapeHtml(e.architecture || "—")}</td>` +
          `<td class="${moe}">${escapeHtml([e.size_label, e.quant].filter(Boolean).join(" · ") || "—")}</td>` +
          `<td class="num">${fmtCtx(e.context_length)}</td>` +
          `<td class="${moe}">${escapeHtml(topology)}</td>` +
          `<td class="num${cur ? " hl" : ""}">${bytes(e.size_bytes)}</td>` +
          `<td class="num"><button type="button" class="${cur ? "primary" : "secondary"} small" data-detail="${escapeHtml(e.path)}">${cur ? "Details" : "Select"}</button></td>` +
          `</tr>`
        );
      })
      .join("") || `<tr><td colspan="7" class="soft">${models.length ? "No model matches the filter." : "No models found."}</td></tr>`;
  const count = (k) => d.entries.filter((e) => e.kind === k).length;
  const extra = [
    count("mmproj") && `${count("mmproj")} vision projector${count("mmproj") > 1 ? "s" : ""}`,
    count("mtp") && `${count("mtp")} MTP draft${count("mtp") > 1 ? "s" : ""}`,
    count("imatrix") && `${count("imatrix")} imatrix file${count("imatrix") > 1 ? "s" : ""}`,
  ].filter(Boolean);
  const bad = d.entries.filter((e) => e.kind === "other");
  $("models-extra").textContent =
    (extra.length ? `Also detected in the models tree: ${extra.join(", ")} (used alongside a model, not on their own). ` : "") +
    (bad.length ? `Could not read: ${bad.map((e) => e.rel_path).join(", ")}. ` : "") +
    (models.length ? `${models.length} model${models.length === 1 ? "" : "s"} indexed.` : "");
  renderModelDetail();
}

function fitText(e) {
  const s = state.lastStats || {};
  const local = localVram(s).total;
  const devices = state.lastDevices || [];
  const rpc = devices.filter((d) => /^rpc\[/.test(d.name || "")).reduce((a, d) => a + (d.memory_bytes || 0), 0);
  const size = e.size_bytes;
  if (!local) return { cls: "ram", head: "No local GPU memory reported:", text: "weights would sit in system RAM." };
  if (size < local * 0.9)
    return { cls: "ok", head: "Fits fully in local VRAM:", text: `${bytes(size)} of weights within ${bytes(local)} (KV cache and workspaces come on top).` };
  if (rpc && size < (local + rpc) * 0.9)
    return { cls: "rpc", head: "Needs RPC offloading:", text: `larger than local VRAM (${bytes(local)}); the rest goes to the RPC workers (${bytes(rpc)}).` };
  return {
    cls: "ram",
    head: "Spills to system RAM:",
    text: `larger than GPU memory${rpc ? " including RPC" : ""} (${bytes(local + rpc)}); part of it runs from system RAM, which is much slower.`,
  };
}

/* Inspection card of the selected model (the loaded one by default): its
 * metadata, whether it fits, and the options to load it with. */
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
  const row = (k, v, hl = false) =>
    v === undefined || v === null || v === "" ? "" : `<div><dt>${k}</dt><dd${hl ? ' class="hl"' : ""}>${escapeHtml(v)}</dd></div>`;
  const fit = fitText(e);
  const mtpOk = e.companions.mtp.length > 0 || (e.nextn_layers || 0) > 0;
  const specNow = cur ? d.speculation : "ngram";
  const opt = (v, label, on, dis = false) =>
    `<option value="${v}"${on ? " selected" : ""}${dis ? " disabled" : ""}>${label}</option>`;
  const chips = [
    cur ? `<span class="chip chip-primary outline">loaded</span>` : "",
    e.companions.vision.length ? `<span class="chip chip-secondary outline">vision</span>` : "",
    mtpOk ? `<span class="chip chip-tertiary outline">MTP</span>` : "",
    e.expert_count ? `<span class="chip chip-secondary outline">MoE</span>` : "",
  ].join("");
  box.innerHTML =
    `<div class="md-top"><div><div class="md-title"><h3>${escapeHtml(e.name || baseName(e.path))}</h3>${chips}</div>` +
    (e.description ? `<p class="model-desc">${escapeHtml(e.description)}</p>` : "") +
    `</div><div class="md-size"><span class="label">On disk</span><div class="v">${metricHtml(bytes(e.size_bytes))}</div></div></div>` +
    `<dl class="kv-grid">` +
    row("File", e.rel_path) +
    row("Architecture", e.architecture) +
    row("Parameters", e.size_label, true) +
    row("Quantization", e.quant) +
    row("Context (trained)", e.context_length ? `${num(e.context_length)} tokens` : "") +
    row("Layers", e.block_count) +
    row("Embed width", e.embedding_length) +
    row("Experts", e.expert_count ? `${e.expert_count} routed, ${e.expert_used_count} per token` : "dense (non-MoE)") +
    row("MTP / NextN layers", e.nextn_layers, true) +
    row("On disk size", `${bytes(e.size_bytes)}${e.shards > 1 ? ` in ${e.shards} shards` : ""}`) +
    row("License", e.license) +
    row("Quantized by", e.quantized_by, true) +
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
    `<div class="fit ${fit.cls}">${icon("check")}<div><b>${fit.head}</b> ${escapeHtml(fit.text)}</div></div>` +
    `<div class="load-opts">` +
    `<label>Vision projector<select id="ld-vision">${opt("", "None", !e.companions.vision.length)}${e.companions.vision
      .map((v, i) => opt(escapeHtml(v), escapeHtml(baseName(v)), i === 0))
      .join("")}</select></label>` +
    `<label>Speculation mode<select id="ld-spec">${opt("off", "Off", specNow === "off")}${opt("ngram", "n-gram", specNow === "ngram" || specNow === "draft")}${opt(
      "mtp",
      mtpOk ? "MTP draft" : "MTP (not available)",
      specNow === "mtp" && mtpOk,
      !mtpOk
    )}</select></label>` +
    (e.companions.mtp.length
      ? `<label>MTP draft weights<select id="ld-mtp">${e.companions.mtp
          .map((v, i) => opt(escapeHtml(v), escapeHtml(baseName(v)), i === 0))
          .join("")}${e.nextn_layers ? opt("", "Model's own NextN layers", false) : ""}</select></label>`
      : "") +
    `<button id="ld-go" type="button" class="primary"${e.error ? " disabled" : ""}>${icon("refresh")}${cur ? "Reload with these options" : "Load this model"}</button>` +
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

/* ---------- settings page ---------- */

const settings = { data: null, ep: 0, shell: "bash", bodies: {} };

async function loadConfig() {
  try {
    settings.data = await apiJson("/v1/config");
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
    return;
  }
  renderSettings();
}

/* Nested tables become dotted keys; arrays stay values. */
function flattenConfig(obj, prefix = "") {
  const out = [];
  for (const [k, v] of Object.entries(obj || {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) out.push(...flattenConfig(v, `${prefix}${k}.`));
    else out.push([`${prefix}${k}`, v]);
  }
  return out;
}

function configLookup(obj, dotted) {
  return dotted.split(".").reduce((o, k) => (o && typeof o === "object" ? o[k] : undefined), obj);
}

function configText(key, v) {
  if (v === null || v === undefined || v === "") return "unset";
  if (Array.isArray(v)) return v.length ? v.map((x) => (typeof x === "object" ? JSON.stringify(x) : String(x))).join(", ") : "none";
  if (typeof v === "number" && /_bytes$/.test(key)) return `${num(v)} (${bytes(v)})`;
  // Fractions are f32 on the server: 0.9 arrives as 0.8999999761581421.
  if (typeof v === "number") return Number.isInteger(v) ? num(v) : String(Number(v.toPrecision(6)));
  return String(v);
}

function configValueHtml(key, v) {
  const text = escapeHtml(configText(key, v));
  if (text === "unset" || text === "none") return `<span class="unset">${text}</span>`;
  if (v === true) return `<span class="on">true</span>`;
  if (typeof v === "string" && v.startsWith("(set, hidden")) return `<span class="secret">${text}</span>`;
  return text;
}

function renderSettings() {
  const d = settings.data;
  if (!d) return;
  const cfg = d.config || {};
  $("set-version").textContent = `v${d.version}`;
  $("set-version").classList.remove("hidden");
  const auth = $("set-auth");
  auth.classList.remove("hidden");
  auth.className = `chip ${d.auth_required ? "chip-primary" : "chip-warn"}`;
  auth.textContent = d.auth_required ? "Password required" : "No password";
  $("set-source").textContent = d.config_path ? `Config file: ${d.config_path}` : "No config file: built-in defaults and command-line flags.";
  const spec = cfg.speculative || {};
  const rpc = cfg.rpc || {};
  const model = cfg.model || {};
  $("set-summary").innerHTML = [
    card("Model", escapeHtml(model.synthetic ? "synthetic" : baseName(model.path) || "none"), escapeHtml(`id ${model.id || "—"}`), "", { tag: icon("chip") }),
    card(
      "Speculation",
      spec.enabled ? "on" : "off",
      escapeHtml(spec.enabled ? `${(spec.providers || []).join(", ")} · depth ${spec.min_depth}–${spec.max_depth}` : "drafting disabled"),
      "",
      { tone: spec.enabled ? "secondary" : "", tag: icon("hub") }
    ),
    card("RPC workers", `${(rpc.endpoints || []).length}`, escapeHtml((rpc.endpoints || []).join(", ") || "local only"), "", { tag: icon("hub") }),
    card("Concurrency", `${num((cfg.engine || {}).max_active_requests)} requests`, escapeHtml(`backend ${(cfg.engine || {}).gpu_backend || "auto"} · batch decode ${(cfg.engine || {}).batch_decode ? "on" : "off"}`), "", { tag: icon("layers") }),
  ].join("");

  const q = $("set-filter").value.trim().toLowerCase();
  const sections = Object.entries(cfg)
    // [http] is on the API page.
    .filter(([name, v]) => name !== "http" && v && typeof v === "object" && !Array.isArray(v))
    .map(([name, section]) => {
      const rows = flattenConfig(section)
        .map(([key, value]) => {
          const def = configLookup((d.defaults || {})[name], key);
          const changed = JSON.stringify(value ?? null) !== JSON.stringify(def ?? null);
          return { key, value, def, changed };
        })
        .filter((r) => !q || `${name}.${r.key} ${configText(r.key, r.value)}`.toLowerCase().includes(q));
      if (!rows.length) return "";
      const n = rows.filter((r) => r.changed).length;
      return (
        `<article class="panel set-section"><header><h3>${escapeHtml(name)}</h3><span class="spacer"></span>` +
        (n ? `<span class="chip chip-primary">${n} changed</span>` : `<span class="chip">defaults</span>`) +
        `</header><table><tbody>` +
        rows
          .map(
            (r) =>
              `<tr class="${r.changed ? "changed" : ""}"><td class="k">${escapeHtml(r.key)}</td><td class="v">${configValueHtml(r.key, r.value)}` +
              (r.changed && r.def !== undefined ? `<span class="def">default: ${escapeHtml(configText(r.key, r.def))}</span>` : "") +
              `</td></tr>`
          )
          .join("") +
        `</tbody></table></article>`
      );
    })
    .join("");
  $("set-sections").innerHTML = sections || `<p class="soft set-empty">No setting matches the filter.</p>`;
}

/* A starting body for each endpoint that takes one, from the live state. */
function apiExampleBody(ep) {
  const key = `${ep.method} ${ep.path}`;
  const modelId = (state.lastStats || {}).model_id || "allpaca";
  const kv = model.info && model.info.kv;
  const examples = {
    "POST /v1/chat/completions": {
      model: modelId,
      messages: [{ role: "user", content: "Hello!" }],
      max_tokens: 256,
      temperature: 0.7,
      stream: false,
    },
    "PUT /v1/model/settings": { context_tokens: (kv && kv.context_tokens) || 32768 },
    "POST /v1/model/load": {
      path: (picker.data && picker.data.current) || "<path from GET /v1/model/available>",
      vision: null,
      speculation: "off",
      mtp_draft: null,
    },
    "POST /v1/model/experts/recording": { on: true },
    "POST /v1/auth/login": { password: "<password>" },
    "PUT /v1/api": { enabled: true, require_keys: true },
    "POST /v1/api/keys": { description: "laptop IDE" },
  };
  return examples[key] ? JSON.stringify(examples[key], null, 2) : "";
}

function shQuote(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}

function psQuote(s) {
  return `'${String(s).replace(/'/g, "''")}'`;
}

/* What a caller must send to `ep`: nothing, an API key, or the operator
 * password. */
function apiCredential(ep) {
  const d = settings.data;
  if (!ep.auth || !d.auth_required) return null;
  if (!ep.client) return "password";
  return apiPage.data && !apiPage.data.require_keys ? null : "key";
}

const API_CRED = {
  key: { chip: "API key", bash: "$ALLPACA_API_KEY", powershell: "$env:ALLPACA_API_KEY" },
  password: { chip: "Operator password", bash: "$ALLPACA_PASSWORD", powershell: "$env:ALLPACA_PASSWORD" },
};

function buildCurl(ep, base, token, body, cred, shell) {
  const url = `${base.replace(/\/+$/, "")}${ep.path}`;
  const stream = ep.path.endsWith("/stream") || /"stream"\s*:\s*true/.test(body);
  const flags = `-sS${stream ? " -N" : ""}${ep.method === "GET" ? "" : ` -X ${ep.method}`}`;
  if (shell === "powershell") {
    // The body goes in on stdin: quoting JSON for a native command differs
    // between PowerShell versions, a here-string does not.
    const parts = [`curl.exe ${flags} ${psQuote(url)}`];
    if (cred) parts.push(token ? `-H ${psQuote(`Authorization: Bearer ${token}`)}` : `-H "Authorization: Bearer ${API_CRED[cred].powershell}"`);
    if (body) parts.push(`-H 'Content-Type: application/json'`, `--data-binary '@-'`);
    return (body ? `@'\n${body}\n'@ | ` : "") + parts.join(" `\n  ");
  }
  const parts = [`curl ${flags} ${shQuote(url)}`];
  if (cred) parts.push(token ? `-H ${shQuote(`Authorization: Bearer ${token}`)}` : `-H "Authorization: Bearer ${API_CRED[cred].bash}"`);
  if (body) parts.push(`-H 'Content-Type: application/json'`, `-d ${shQuote(body)}`);
  return parts.join(" \\\n  ");
}

function renderApi() {
  const d = settings.data;
  if (!d) return;
  const eps = d.endpoints || [];
  settings.ep = Math.min(settings.ep, Math.max(0, eps.length - 1));
  $("api-count").textContent = `${eps.length} endpoints`;
  $("api-list").innerHTML = eps
    .map(
      (e, i) =>
        `<button type="button" class="api-item${i === settings.ep ? " on" : ""}" data-ep="${i}" role="option" aria-selected="${i === settings.ep}">` +
        `<span class="chip m-${e.method}">${e.method}</span><span class="mono">${escapeHtml(e.path)}</span>` +
        `<span class="sum">${escapeHtml(e.summary)}${e.client ? " · API key" : e.auth ? " · operator" : ""}</span></button>`
    )
    .join("");
  const ep = eps[settings.ep];
  if (!ep) return;
  const key = `${ep.method} ${ep.path}`;
  if (!$("api-base").value) $("api-base").value = location.origin;
  $("api-method").className = `chip m-${ep.method}`;
  $("api-method").textContent = ep.method;
  $("api-path").textContent = ep.path;
  $("api-summary").textContent = ep.summary;
  const cred = apiCredential(ep);
  $("api-authchip").textContent = cred ? API_CRED[cred].chip : "No token";
  $("api-authchip").className = `chip ${cred === "password" ? "chip-warn" : cred ? "chip-primary" : ""}`;
  const template = apiExampleBody(ep);
  $("api-body-wrap").classList.toggle("hidden", !template);
  const box = $("api-body");
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.value = settings.bodies[key] ?? template;
  }
  renderCurl();
}

function renderCurl() {
  const d = settings.data;
  const ep = d && (d.endpoints || [])[settings.ep];
  if (!ep) return;
  const hasBody = !$("api-body-wrap").classList.contains("hidden");
  const body = hasBody ? $("api-body").value.trim() : "";
  let valid = true;
  if (body) {
    try {
      JSON.parse(body);
    } catch {
      valid = false;
    }
  }
  $("api-body-error").classList.toggle("hidden", valid);
  const token = $("api-token").value.trim();
  const base = $("api-base").value.trim() || location.origin;
  const cred = apiCredential(ep);
  $("api-curl").textContent = buildCurl(ep, base, token, body, cred, settings.shell);
  const off = apiPage.data && !apiPage.data.enabled ? "The API is off, so the server will refuse this (403). " : "";
  $("api-hint").textContent =
    off +
    (!cred
      ? "This endpoint needs no token."
      : token
        ? "The token is written into the command: mind where you paste it."
        : cred === "key"
          ? `Reads the key from ${API_CRED.key[settings.shell]}: set it to an API key generated above.`
          : `Operator endpoint: reads ${API_CRED.password[settings.shell]}, the server's password (http.password, or the one printed at start-up). API keys cannot call it.`);
}

/* ---------- API page ---------- */

const apiPage = { data: null, newKey: null, busy: false };

async function loadApiPage() {
  try {
    const [a, c] = await Promise.all([apiJson("/v1/api"), apiJson("/v1/config")]);
    apiPage.data = a;
    settings.data = c;
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
    return;
  }
  renderApiPage();
  renderApi();
}

function isLoopback(host) {
  return host === "127.0.0.1" || host === "::1" || host === "localhost";
}

function listenChoice() {
  const sel = $("api-host").value;
  return { host: sel === "custom" ? $("api-host-custom").value.trim() : sel, port: Number($("api-port").value) };
}

function fmtWhen(ms) {
  return ms ? new Date(ms).toLocaleString() : "never";
}

function renderApiPage() {
  const d = apiPage.data;
  if (!d) return;
  const chip = (id, cls, text) => {
    const el = $(id);
    el.className = `chip ${cls}`;
    el.textContent = text;
  };
  chip("api-state", d.enabled ? "chip-primary" : "chip-err", d.enabled ? "Enabled" : "Disabled");
  chip("api-lock", d.require_keys ? "chip-primary" : "chip-warn", d.require_keys ? "Keys required" : "Open: no key needed");
  chip("api-listen", isLoopback(d.host) ? "" : "chip-secondary", `${d.host}:${d.port}`);
  $("api-source").textContent = d.settings_path
    ? `Saved in ${d.settings_path}`
    : "These settings are not saved to disk on this server.";
  $("api-enabled").checked = d.enabled;
  $("api-require").checked = d.require_keys;
  $("api-require").disabled = !d.enabled;

  const warn = [];
  if (d.enabled && !d.require_keys && !isLoopback(d.host))
    warn.push(`Anyone who can reach ${d.host}:${d.port} can run chat completions on this machine without a key.`);
  else if (d.enabled && !d.require_keys) warn.push("Any program on this machine, including a web page open in a browser here, can call the client endpoints without a key.");
  if (d.enabled && d.require_keys && !d.keys.length) warn.push("No API keys yet: until you generate one, only the operator password works as a bearer token.");
  $("api-warn").classList.toggle("hidden", !warn.length);
  $("api-warn").textContent = warn.join(" ");

  // Listen address: presets, this machine's addresses, or any IP.
  const presets = [
    ["127.0.0.1", "127.0.0.1 · this machine only"],
    ["0.0.0.0", "0.0.0.0 · every network interface"],
    ...d.interfaces.map((i) => [i.addr, `${i.addr} · ${i.name}`]),
  ];
  const known = presets.some(([v]) => v === d.host);
  const sel = $("api-host");
  if (document.activeElement !== sel && document.activeElement !== $("api-host-custom")) {
    sel.innerHTML =
      presets.map(([v, label]) => `<option value="${escapeHtml(v)}">${escapeHtml(label)}</option>`).join("") +
      `<option value="custom">Another IP address…</option>`;
    sel.value = known ? d.host : "custom";
    $("api-host-custom").value = known ? "" : d.host;
    $("api-host-custom-wrap").classList.toggle("hidden", known);
  }
  if (document.activeElement !== $("api-port")) $("api-port").value = d.port;
  $("api-listen-note").textContent =
    (d.live_rebind
      ? "Applies at once, without reloading the model; this page follows the new address. "
      : "Takes effect the next time the server starts. ") +
    `The config file says ${d.config_host}:${d.config_port}` +
    (d.host !== d.config_host || d.port !== d.config_port ? "; the saved override is in use." : ".") +
    " A --host or --port flag overrides both.";
  syncListenApply();

  const cfg = (settings.data && settings.data.config) || {};
  const row = (k, note, v) => `<tr><td class="k">${k}<small>${note}</small></td><td>${escapeHtml(v)}</td></tr>`;
  const e = cfg.engine || {};
  const sm = cfg.sampling || {};
  const so = cfg.structured_output || {};
  $("api-related").innerHTML = [
    row("http.max_tokens", "largest max_tokens a request may ask for", num((cfg.http || {}).max_tokens)),
    row("engine.max_active_requests", "requests served at the same time", num(e.max_active_requests)),
    row("engine.batch_decode", "decode concurrent requests in one batch", e.batch_decode ? "on" : "off"),
    row("speculative.max_concurrent_requests", "speculation pauses above this many", num((cfg.speculative || {}).max_concurrent_requests)),
    row("sampling", "used when a request does not set them", `temperature ${configText("", sm.temperature)} · top_k ${sm.top_k} · top_p ${configText("", sm.top_p)}`),
    row("structured_output", "response_format support", `${so.enabled ? "on" : "off"}${so.json_mode ? " · JSON mode forced" : ""}`),
    row("http.password", "operator password (WebUI login and bearer)", (cfg.http || {}).password ? "set" : "generated at start-up"),
  ].join("");

  $("api-keys-count").textContent = `${d.keys.length} key${d.keys.length === 1 ? "" : "s"}`;
  $("api-key-rows").innerHTML =
    d.keys
      .map(
        (k) =>
          `<tr><td class="desc">${escapeHtml(k.description)}</td><td>${escapeHtml(k.prefix)}…</td>` +
          `<td>${escapeHtml(fmtWhen(k.created_ms))}</td><td class="${k.last_used_ms ? "" : "soft"}">${escapeHtml(fmtWhen(k.last_used_ms))}</td>` +
          `<td class="num"><button type="button" class="danger small" data-revoke="${escapeHtml(k.id)}">Revoke</button></td></tr>`
      )
      .join("") || `<tr><td colspan="5" class="soft">No API keys. Generate one for each client so it can be revoked on its own.</td></tr>`;

  const box = $("api-key-new");
  box.classList.toggle("hidden", !apiPage.newKey);
  if (apiPage.newKey) {
    box.innerHTML =
      `<p><strong>New key for “${escapeHtml(apiPage.newKey.description)}”.</strong> Copy it now: it is not stored and will not be shown again.</p>` +
      `<div class="row"><code id="api-key-secret">${escapeHtml(apiPage.newKey.key)}</code>` +
      `<button type="button" class="secondary small" data-key-act="copy">${icon("copy")}Copy</button>` +
      `<button type="button" class="secondary small" data-key-act="use">Use in curl below</button>` +
      `<button type="button" class="secondary small" data-key-act="dismiss">Done</button></div>`;
  }
}

function syncListenApply() {
  const d = apiPage.data;
  if (!d) return;
  const { host, port } = listenChoice();
  $("api-listen-apply").disabled = apiPage.busy || !host || !port || (host === d.host && port === d.port);
}

async function putApi(body, okMsg) {
  apiPage.busy = true;
  try {
    apiPage.data = await apiJson("/v1/api", { method: "PUT", body: JSON.stringify(body) });
    if (okMsg) toast(okMsg);
    return true;
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
    return false;
  } finally {
    apiPage.busy = false;
    renderApiPage();
    renderApi();
  }
}

async function applyListen() {
  const d = apiPage.data;
  const { host, port } = listenChoice();
  const exposes = !isLoopback(host) && isLoopback(d.host);
  const ok = window.confirm(
    `Listen on ${host}:${port} instead of ${d.host}:${d.port}?\n\n` +
      (exposes ? "This makes the server reachable from other machines on that network.\n\n" : "") +
      "The WebUI uses the same address, so this page will reopen there. Requests in progress finish on the old address."
  );
  if (!ok) return;
  if (!(await putApi({ host, port }))) return;
  if (!apiPage.data.live_rebind) {
    toast("Saved. It applies when the server restarts.");
    return;
  }
  // Follow the server if this page's own address no longer reaches it.
  const here = location.hostname;
  const sameHost = host === "0.0.0.0" || host === here || (isLoopback(host) && isLoopback(here));
  if (sameHost && String(port) === (location.port || "80")) {
    toast(`Listening on ${host}:${port}`);
    return;
  }
  toast(`Listening on ${host}:${port}. Reopening there…`);
  const target = sameHost ? here : host;
  setTimeout(() => (location.href = `${location.protocol}//${target}:${port}/#api`), 900);
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

$("log-level").addEventListener("change", () => scheduleLogRender(true, true));
$("log-filter").addEventListener("input", () => scheduleLogRender(true, true));
$("log-filter").addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  ev.target.value = "";
  scheduleLogRender(true, true);
});
$("log-follow").addEventListener("change", () => scheduleLogRender(false));
$("log-wrap").addEventListener("change", (ev) => $("log-view").classList.toggle("nowrap", !ev.target.checked));
$("log-clear").addEventListener("click", () => {
  state.logLines = [];
  scheduleLogRender(true, true);
});
$("log-pause").addEventListener("click", () => {
  state.logPaused = !state.logPaused;
  const btn = $("log-pause");
  btn.setAttribute("aria-pressed", String(state.logPaused));
  btn.innerHTML = state.logPaused ? `${icon("play")}<span>Resume</span>` : `${icon("pause")}<span>Pause</span>`;
  setLogPipe(state.logPipe || "connecting");
  scheduleLogRender(false);
});
$("log-dump").addEventListener("click", () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  download(`allpaca-logs-${stamp}.ndjson`, state.logLines.map((l) => JSON.stringify(l)).join("\n") + "\n", "application/x-ndjson");
});
$("log-subsystems").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-sub]");
  if (!btn) return;
  state.logSub = btn.dataset.sub;
  scheduleLogRender(true, true);
});

$("hero-switch").addEventListener("click", () => setPage("model"));
$("hero-chat").addEventListener("click", () => setPage("chat"));
$("event-open").addEventListener("click", () => setPage("logs"));
$("topo-filter").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-topo]");
  if (!btn) return;
  state.topoFilter = btn.dataset.topo;
  renderStatus();
});

$("drawer-toggle").addEventListener("click", () => {
  setDrawer($("drawer").classList.contains("hidden"));
  saveSettings();
});
$("drawer-close").addEventListener("click", () => {
  setDrawer(false);
  saveSettings();
});
["temperature", "top-p"].forEach((id) => $(id).addEventListener("input", syncSampling));
$("top-p").addEventListener("change", saveSettings);
$("sampling-reset").addEventListener("click", () => {
  $("temperature").value = "0.7";
  $("top-p").value = "0.95";
  $("max-tokens").value = "4096";
  syncSampling();
  saveSettings();
});
$("export-chat").addEventListener("click", () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const body = { model: (state.lastStats || {}).model_id || null, ...chatPayload(), stream: undefined, exported_at: new Date().toISOString() };
  download(`allpaca-chat-${stamp}.json`, JSON.stringify(body, null, 2), "application/json");
});
$("messages").addEventListener("click", async (ev) => {
  if (ev.target.closest(".regen")) return regenerate();
  const btn = ev.target.closest(".copy-msg");
  if (!btn) return;
  const m = state.messages[Number(btn.closest(".msg").dataset.i)];
  if (!m) return;
  try {
    await navigator.clipboard.writeText(splitThink(m.content).body || m.content);
    toast("Reply copied");
  } catch {
    toast("Copy failed", true);
  }
});
// Keeps a reasoning block open or closed across re-renders.
$("messages").addEventListener(
  "toggle",
  (ev) => {
    const d = ev.target;
    if (!d.classList || !d.classList.contains("think")) return;
    const i = Number(d.closest(".msg").dataset.i);
    if (state.messages[i] && !(state.streaming && state.streaming.index === i)) state.messages[i].thinkOpen = d.open;
  },
  true
);
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
$("exp-place-map").addEventListener("click", (ev) => {
  const cell = ev.target.closest("[data-l]");
  if (cell) setLayer(Number(cell.dataset.l));
});
$("exp-place-select").addEventListener("change", (ev) => setExpertPin(model.layer, ev.target.value));
$("exp-place-clear").addEventListener("click", () => {
  if (window.confirm("Clear every expert pin?\n\nPlacement goes back to automatic on the next reload.")) {
    putSettings({ expert_homes: {} }, "Pins cleared. Reload the model to apply.");
  }
});
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

$("set-refresh").addEventListener("click", loadConfig);

$("api-refresh").addEventListener("click", loadApiPage);
$("api-enabled").addEventListener("change", (ev) => {
  const on = ev.target.checked;
  if (!on && !window.confirm("Turn the API off?\n\nEvery API key and bearer token is refused until you turn it back on. This WebUI keeps working.")) {
    ev.target.checked = true;
    return;
  }
  putApi({ enabled: on }, on ? "API enabled" : "API disabled");
});
$("api-require").addEventListener("change", (ev) => {
  const on = ev.target.checked;
  const d = apiPage.data;
  if (!on) {
    const where = isLoopback(d.host) ? "any program on this machine" : `anyone who can reach ${d.host}:${d.port}`;
    if (!window.confirm(`Turn lockdown off?\n\nWithout a key, ${where} can run chat completions and read status and metrics.`)) {
      ev.target.checked = true;
      return;
    }
  }
  putApi({ require_keys: on }, on ? "API keys required" : "Lockdown off");
});
$("api-host").addEventListener("change", (ev) => {
  $("api-host-custom-wrap").classList.toggle("hidden", ev.target.value !== "custom");
  if (ev.target.value === "custom") $("api-host-custom").focus();
  syncListenApply();
});
["api-host-custom", "api-port"].forEach((id) => $(id).addEventListener("input", syncListenApply));
$("api-listen-apply").addEventListener("click", applyListen);
$("api-key-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const description = $("api-key-desc").value.trim();
  if (!description) return;
  try {
    apiPage.newKey = await apiJson("/v1/api/keys", { method: "POST", body: JSON.stringify({ description }) });
    $("api-key-desc").value = "";
    await loadApiPage();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
});
$("api-key-new").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-key-act]");
  if (!btn || !apiPage.newKey) return;
  const act = btn.dataset.keyAct;
  if (act === "copy") {
    try {
      await navigator.clipboard.writeText(apiPage.newKey.key);
      toast("Key copied");
    } catch {
      toast("Copy failed: select the key and copy it by hand", true);
    }
  } else if (act === "use") {
    $("api-token").value = apiPage.newKey.key;
    renderCurl();
    $("api-curl").scrollIntoView({ behavior: "smooth", block: "center" });
  } else {
    apiPage.newKey = null;
    renderApiPage();
  }
});
$("api-key-rows").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("[data-revoke]");
  if (!btn) return;
  const key = apiPage.data.keys.find((k) => k.id === btn.dataset.revoke);
  if (!key || !window.confirm(`Revoke the key “${key.description}” (${key.prefix}…)?\n\nClients using it are refused from now on.`)) return;
  try {
    await apiJson(`/v1/api/keys/${encodeURIComponent(key.id)}`, { method: "DELETE" });
    if (apiPage.newKey && apiPage.newKey.id === key.id) apiPage.newKey = null;
    toast("Key revoked");
    await loadApiPage();
  } catch (e) {
    if (!(e instanceof Unauthorized)) toast(e.message, true);
  }
});
$("set-filter").addEventListener("input", renderSettings);
$("api-list").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-ep]");
  if (!btn) return;
  settings.ep = Number(btn.dataset.ep);
  renderApi();
});
$("api-shell").addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-shell]");
  if (!btn) return;
  settings.shell = btn.dataset.shell;
  document.querySelectorAll("#api-shell button").forEach((b) => b.classList.toggle("on", b === btn));
  renderCurl();
});
["api-base", "api-token"].forEach((id) => $(id).addEventListener("input", renderCurl));
$("api-body").addEventListener("input", (ev) => {
  settings.bodies[ev.target.dataset.key] = ev.target.value;
  renderCurl();
});
$("api-copy").addEventListener("click", async () => {
  const btn = $("api-copy");
  try {
    await navigator.clipboard.writeText($("api-curl").textContent);
    btn.textContent = "Copied";
  } catch {
    btn.textContent = "Copy failed";
  }
  setTimeout(() => (btn.textContent = "Copy"), 1200);
});
$("models-filter").addEventListener("input", renderAvailable);
$("models-rows").addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr[data-path]");
  if (!tr) return;
  picker.sel = tr.dataset.path;
  renderAvailable();
  $("model-detail").scrollIntoView({ behavior: "smooth", block: "start" });
});

loadSettings();
boot();
