const $ = (id) => document.getElementById(id);
const LEVELS = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];

const state = {
  page: "dashboard",
  messages: loadMessages(),
  logLines: [],
  logSource: null,
};

function loadMessages() {
  try {
    return JSON.parse(localStorage.getItem("allpaca.chat") || "[]");
  } catch {
    return [];
  }
}

function saveMessages() {
  localStorage.setItem("allpaca.chat", JSON.stringify(state.messages.slice(-80)));
}

async function api(path, opts = {}) {
  const res = await fetch(path, { credentials: "include", ...opts });
  if (res.status === 401) {
    showLogin();
    throw new Error("unauthorized");
  }
  return res;
}

function showLogin() {
  $("login").classList.remove("hidden");
  $("app").classList.add("hidden");
}

function showApp() {
  $("login").classList.add("hidden");
  $("app").classList.remove("hidden");
}

function bytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  if (v < 1024 ** 2) return `${(v / 1024).toFixed(1)} KiB`;
  if (v < 1024 ** 3) return `${(v / 1024 ** 2).toFixed(1)} MiB`;
  return `${(v / 1024 ** 3).toFixed(2)} GiB`;
}

function meter(frac) {
  const pct = Math.max(0, Math.min(100, Math.round((frac || 0) * 100)));
  return `<div class="meter"><span style="width:${pct}%"></span></div>`;
}

function card(title, body, extra = "") {
  return `<article class="card"><h3>${title}</h3><div class="metric">${body}</div>${extra}</article>`;
}

function renderDashboard(data) {
  const s = data.stats || {};
  const h = data.host || {};
  const ramFrac = h.ram_total_bytes ? h.ram_used_bytes / h.ram_total_bytes : 0;
  const vramFrac = s.gpu_vram_total_bytes ? s.gpu_vram_used_bytes / s.gpu_vram_total_bytes : 0;
  const draftFrac = s.draft_gpu_vram_total_bytes
    ? s.draft_gpu_vram_used_bytes / s.draft_gpu_vram_total_bytes
    : 0;
  const gpuLabel = (id, fallback) => {
    const d = (data.devices || []).find((x) => x.id === id);
    if (d && d.name) return d.name.replace(/^AMD Radeon /, "");
    return fallback;
  };
  $("dash-cards").innerHTML = [
    card("Model", escapeHtml(s.model_id || "—"), `<p class="muted">${escapeHtml(s.backend || "")} · ${s.active_requests || 0} active</p>`),
    card("Host CPU", `${(h.cpu_usage || 0).toFixed(0)}%`, meter((h.cpu_usage || 0) / 100)),
    card("Host RAM", `${bytes(h.ram_used_bytes)} / ${bytes(h.ram_total_bytes)}`, meter(ramFrac)),
    card(`${escapeHtml(gpuLabel(s.target_device, "GPU 0"))} VRAM`, s.gpu_vram_total_bytes ? `${bytes(s.gpu_vram_used_bytes)} / ${bytes(s.gpu_vram_total_bytes)}` : "n/a", meter(vramFrac)),
    card(`${escapeHtml(gpuLabel(s.draft_device, "GPU 1"))} VRAM`, s.draft_gpu_vram_total_bytes ? `${bytes(s.draft_gpu_vram_used_bytes)} / ${bytes(s.draft_gpu_vram_total_bytes)}` : "n/a", meter(draftFrac)),
    card("GPU util", `${((s.gpu_util || 0) * 100).toFixed(0)}%`),
    card("TTFT / ITL", `${s.ttft_ms_last || 0} ms / ${(s.itl_ms_ema || 0).toFixed(1)} ms`),
    card("KV pages", `VRAM ${s.vram_pages || 0} · DRAM ${s.dram_pages || 0}`),
    card("Weights on device", bytes(s.weight_bytes_on_device)),
    card("Speculation", `${((s.acceptance_rate || 0) * 100).toFixed(1)}% accept`, `<p class="muted">${s.draft_accepted || 0} / ${s.draft_proposed || 0}</p>`),
    card("RPC", `${s.rpc_roundtrips || 0} roundtrips`, `<p class="muted">${bytes(s.rpc_activation_bytes)} activations</p>`),
  ].join("");
  const rows = (data.devices || [])
    .map(
      (d) =>
        `<tr><td>${d.id}</td><td>${escapeHtml(d.kind)}</td><td>${escapeHtml(d.name)}</td><td>${bytes(d.memory_bytes)}</td></tr>`
    )
    .join("");
  $("device-rows").innerHTML = rows || `<tr><td colspan="4" class="muted">No devices</td></tr>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMessages() {
  const box = $("messages");
  box.innerHTML = state.messages
    .map((m) => `<div class="bubble ${m.role}">${escapeHtml(m.content)}</div>`)
    .join("");
  box.scrollTop = box.scrollHeight;
}

function setPage(name) {
  state.page = name;
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  ["dashboard", "chat", "logs"].forEach((p) => {
    $(`page-${p}`).classList.toggle("hidden", p !== name);
  });
  if (name === "logs") startLogs();
}

async function pollStatus() {
  try {
    const res = await api("/v1/status");
    if (!res.ok) return;
    const data = await res.json();
    renderDashboard(data);
    const models = await api("/v1/models").then((r) => r.json()).catch(() => null);
    if (models && models.data && models.data[0]) {
      $("chat-model").textContent = models.data[0].id;
    }
  } catch {
    /* login gate handles 401 */
  }
}

async function sendChat(ev) {
  ev.preventDefault();
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  $("prompt").value = "";
  state.messages.push({ role: "user", content: prompt });
  state.messages.push({ role: "assistant", content: "" });
  saveMessages();
  renderMessages();
  const payload = {
    messages: state.messages.slice(0, -1),
    stream: true,
    max_tokens: Number($("max-tokens").value) || 128,
    temperature: Number($("temperature").value) || 0,
  };
  const res = await api("/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) {
    state.messages[state.messages.length - 1].content = `(error ${res.status})`;
    saveMessages();
    renderMessages();
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const assistant = state.messages[state.messages.length - 1];
  while (true) {
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
      try {
        const json = JSON.parse(data);
        const delta = json.choices && json.choices[0] && json.choices[0].delta;
        if (delta && delta.content) {
          assistant.content += delta.content;
          renderMessages();
        }
      } catch {
        /* ignore keepalives */
      }
    }
  }
  saveMessages();
}

function levelOk(line, min) {
  return LEVELS.indexOf((line.level || "INFO").toUpperCase()) >= LEVELS.indexOf(min);
}

function renderLogs() {
  const min = $("log-level").value;
  const text = state.logLines
    .filter((l) => levelOk(l, min))
    .map((l) => `${new Date(l.ts_ms).toISOString()} ${l.level.padEnd(5)} ${l.target} ${l.message}`)
    .join("\n");
  const view = $("log-view");
  const follow = $("log-follow").checked;
  view.textContent = text;
  if (follow) view.scrollTop = view.scrollHeight;
}

function startLogs() {
  if (state.logSource) return;
  api("/v1/logs")
    .then((r) => r.json())
    .then((body) => {
      state.logLines = body.lines || [];
      renderLogs();
    })
    .catch(() => {});
  const src = new EventSource("/v1/logs/stream");
  src.onmessage = (ev) => {
    try {
      const line = JSON.parse(ev.data);
      state.logLines.push(line);
      if (state.logLines.length > 4000) state.logLines.splice(0, 1000);
      if (state.page === "logs") renderLogs();
    } catch {
      /* ignore */
    }
  };
  src.onerror = () => {
    src.close();
    state.logSource = null;
  };
  state.logSource = src;
}

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("login-error").classList.add("hidden");
  const res = await fetch("/v1/auth/login", {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ password: $("password").value }),
  });
  if (!res.ok) {
    $("login-error").textContent = "Invalid password";
    $("login-error").classList.remove("hidden");
    return;
  }
  $("password").value = "";
  showApp();
  pollStatus();
});

$("logout").addEventListener("click", async () => {
  await fetch("/v1/auth/logout", { method: "POST", credentials: "include" });
  showLogin();
});

document.querySelectorAll("nav button").forEach((b) => {
  b.addEventListener("click", () => setPage(b.dataset.page));
});
$("chat-form").addEventListener("submit", sendChat);
$("log-level").addEventListener("change", renderLogs);
$("log-follow").addEventListener("change", renderLogs);

renderMessages();
pollStatus().then(() => {
  showApp();
  setInterval(pollStatus, 1000);
}).catch(() => {
  showLogin();
});
