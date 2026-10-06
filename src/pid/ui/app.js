// P&ID Digitizer review UI: pan/zoom viewer, confidence colouring, corrections.
"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const STRUCT = new Set(["crossing", "ankle"]);

const state = {
  jobId: null, graph: null, img: null, imgScale: 1,   // imgScale: original px -> image px
  view: { x: 0, y: 0, k: 1 },                         // screen = (orig * imgScale) * k + (x,y)
  sel: null,                                          // {kind: "node"|"edge", id}
  addFrom: null, thr: 1, showImg: true, poll: null,
};

// ---------- helpers ----------
function confColor(p) {
  p = Math.max(0, Math.min(1, p));
  return p < 0.5 ? `rgb(220,${Math.round(60 + 300 * p)},40)`
                 : `rgb(${Math.round(220 - 400 * (p - 0.5))},190,60)`;
}
function toScreen(x, y) {
  const s = state.imgScale * state.view.k;
  return [x * s + state.view.x, y * s + state.view.y];
}
function toOrig(sx, sy) {
  const s = state.imgScale * state.view.k;
  return [(sx - state.view.x) / s, (sy - state.view.y) / s];
}
function center(n) { return [(n.bbox.xmin + n.bbox.xmax) / 2, (n.bbox.ymin + n.bbox.ymax) / 2]; }
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (_) {}
    throw new Error(`${r.status}: ${msg}`);
  }
  return r;
}
function setStatus(t) { $("status").textContent = t; }

// ---------- drawing ----------
function resize() {
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio;
  canvas.height = r.height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}

function visible(score) { return score <= state.thr + 1e-9; }

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (state.img && state.showImg) {
    const [x0, y0] = toScreen(0, 0);
    ctx.globalAlpha = 0.55;
    ctx.drawImage(state.img, x0, y0, state.img.width * state.view.k, state.img.height * state.view.k);
    ctx.globalAlpha = 1;
  }
  const g = state.graph;
  if (!g) return;
  const byId = new Map(g.symbols.map((s) => [s.id, s]));
  const lw = Math.max(1.5, 2 * state.view.k);
  for (const e of g.edges) {
    const a = byId.get(e.source), b = byId.get(e.target);
    if (!a || !b) continue;
    const show = visible(e.score);
    const [x1, y1] = toScreen(...center(a)), [x2, y2] = toScreen(...center(b));
    ctx.strokeStyle = confColor(e.score);
    ctx.globalAlpha = show ? 1 : 0.12;
    ctx.lineWidth = state.sel && state.sel.id === e.id ? lw + 3 : lw;
    ctx.setLineDash(e.line_type === "non-solid" ? [6, 4] : []);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.setLineDash([]);
  for (const s of g.symbols) {
    const show = visible(s.score);
    ctx.globalAlpha = show ? 1 : 0.12;
    const isSel = state.sel && state.sel.id === s.id;
    const isFrom = state.addFrom === s.id;
    ctx.strokeStyle = ctx.fillStyle = isFrom ? "#2563eb" : confColor(s.score);
    if (STRUCT.has(s.cls)) {
      const [cx, cy] = toScreen(...center(s));
      ctx.beginPath(); ctx.arc(cx, cy, isSel ? 6 : 3.5, 0, 2 * Math.PI); ctx.fill();
    } else {
      const [x0, y0] = toScreen(s.bbox.xmin, s.bbox.ymin);
      const [x1, y1] = toScreen(s.bbox.xmax, s.bbox.ymax);
      ctx.lineWidth = isSel || isFrom ? 3 : 1.5;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }
  }
  ctx.globalAlpha = 1;
}

function fitView() {
  if (!state.img) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const k = Math.min(w / state.img.width, h / state.img.height) * 0.97;
  state.view = { k, x: (w - state.img.width * k) / 2, y: (h - state.img.height * k) / 2 };
  draw();
}

// ---------- hit testing ----------
function pick(sx, sy) {
  const g = state.graph;
  if (!g) return null;
  const [ox, oy] = toOrig(sx, sy);
  const tol = 8 / (state.imgScale * state.view.k);     // 8 screen px in original px
  let best = null, bestD = Infinity;
  for (const s of g.symbols) {
    if (!visible(s.score)) continue;
    const b = s.bbox;
    const inside = ox >= b.xmin - tol && ox <= b.xmax + tol && oy >= b.ymin - tol && oy <= b.ymax + tol;
    if (!inside) continue;
    const [cx, cy] = center(s);
    const d = Math.hypot(ox - cx, oy - cy) * (STRUCT.has(s.cls) ? 0.5 : 1);  // prefer small nodes
    if (d < bestD) { bestD = d; best = { kind: "node", id: s.id }; }
  }
  if (best) return best;
  const byId = new Map(g.symbols.map((s) => [s.id, s]));
  for (const e of g.edges) {
    if (!visible(e.score)) continue;
    const a = byId.get(e.source), b = byId.get(e.target);
    if (!a || !b) continue;
    const [x1, y1] = center(a), [x2, y2] = center(b);
    const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((ox - x1) * dx + (oy - y1) * dy) / L));
    const d = Math.hypot(ox - (x1 + t * dx), oy - (y1 + t * dy));
    if (d < tol && d < bestD) { bestD = d; best = { kind: "edge", id: e.id }; }
  }
  return best;
}

// ---------- selection panel ----------
function showSelection() {
  const g = state.graph, sel = state.sel;
  $("nodeActions").hidden = $("edgeActions").hidden = $("commonActions").hidden = true;
  if (!g || !sel) { $("sel").textContent = "Nothing selected."; draw(); return; }
  if (sel.kind === "node") {
    const s = g.symbols.find((x) => x.id === sel.id);
    if (!s) { state.sel = null; return showSelection(); }
    const deg = g.edges.filter((e) => e.source === s.id || e.target === s.id).length;
    $("sel").textContent = `node ${s.id}\nclass: ${s.cls}\nconfidence: ${s.score.toFixed(3)}\n` +
      `degree: ${deg}\nbbox: ${[s.bbox.xmin, s.bbox.ymin, s.bbox.xmax, s.bbox.ymax].map((v) => v.toFixed(0)).join(", ")}`;
    $("cls").value = s.cls;
    $("nodeActions").hidden = false;
  } else {
    const e = g.edges.find((x) => x.id === sel.id);
    if (!e) { state.sel = null; return showSelection(); }
    $("sel").textContent = `edge ${e.id}\n${e.source} — ${e.target}\nline: ${e.line_type}\n` +
      `confidence: ${e.score.toFixed(3)}`;
    $("edgeActions").hidden = false;
  }
  $("commonActions").hidden = false;
  draw();
}

function updateCounts() {
  const g = state.graph;
  if (!g) { $("counts").textContent = ""; return; }
  const low = g.edges.filter((e) => e.score < 0.5).length;
  const shown = g.edges.filter((e) => visible(e.score)).length;
  $("counts").textContent = `${g.symbols.length} nodes · ${g.edges.length} edges · ` +
    `${low} edges below 0.5 · ${shown} shown`;
}

// ---------- corrections ----------
async function correct(ops) {
  try {
    const r = await api(`/jobs/${state.jobId}/graph`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ops }),
    });
    state.graph = await r.json();
    setStatus(`saved ${ops.length} correction${ops.length > 1 ? "s" : ""}`);
    updateCounts(); showSelection();
  } catch (err) { setStatus(`correction failed: ${err.message}`); }
}

$("relabel").onclick = () => state.sel && correct([{ op: "relabel_node", id: state.sel.id, cls: $("cls").value }]);
$("confirm").onclick = () => state.sel && correct([{ op: "confirm", id: state.sel.id }]);
$("delete").onclick = () => {
  if (!state.sel) return;
  const op = state.sel.kind === "node" ? "delete_node" : "delete_edge";
  const id = state.sel.id; state.sel = null;
  correct([{ op, id }]);
};
$("toggleType").onclick = () => {
  const e = state.sel && state.graph.edges.find((x) => x.id === state.sel.id);
  if (e) correct([{ op: "set_line_type", id: e.id, line_type: e.line_type === "solid" ? "non-solid" : "solid" }]);
};
$("addEdge").onclick = () => {
  if (!state.sel || state.sel.kind !== "node") return;
  state.addFrom = state.sel.id; canvas.classList.add("adding");
  setStatus(`adding edge from ${state.addFrom}: click the target node (Esc cancels)`); draw();
};
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape") { state.addFrom = null; canvas.classList.remove("adding"); setStatus(""); draw(); }
  if ((ev.key === "Delete" || ev.key === "Backspace") && state.sel && document.activeElement === document.body) $("delete").onclick();
});

// ---------- pan / zoom / click ----------
let drag = null;
canvas.addEventListener("mousedown", (ev) => { drag = { x: ev.clientX, y: ev.clientY, moved: false }; });
window.addEventListener("mousemove", (ev) => {
  if (!drag) return;
  const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; canvas.classList.add("dragging"); }
  if (drag.moved) { state.view.x += dx; state.view.y += dy; drag.x = ev.clientX; drag.y = ev.clientY; draw(); }
});
window.addEventListener("mouseup", (ev) => {
  if (!drag) return;
  canvas.classList.remove("dragging");
  const wasClick = !drag.moved; drag = null;
  if (!wasClick || ev.target !== canvas) return;
  const r = canvas.getBoundingClientRect();
  const hit = pick(ev.clientX - r.left, ev.clientY - r.top);
  if (state.addFrom) {
    if (hit && hit.kind === "node" && hit.id !== state.addFrom) {
      const from = state.addFrom; state.addFrom = null; canvas.classList.remove("adding");
      correct([{ op: "add_edge", source: from, target: hit.id, line_type: "solid" }]);
    }
    return;
  }
  state.sel = hit; showSelection();
});
canvas.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const f = Math.exp(-ev.deltaY * 0.0015);
  const k = Math.max(0.05, Math.min(40, state.view.k * f));
  const real = k / state.view.k;
  state.view.x = mx - (mx - state.view.x) * real;
  state.view.y = my - (my - state.view.y) * real;
  state.view.k = k; draw();
}, { passive: false });

// ---------- controls ----------
$("thr").oninput = () => { state.thr = parseFloat($("thr").value); $("thrOut").textContent = state.thr.toFixed(2); updateCounts(); draw(); };
$("showImg").onchange = () => { state.showImg = $("showImg").checked; draw(); };
window.addEventListener("resize", resize);

// ---------- jobs ----------
async function refreshJobs() {
  try {
    const jobs = await (await api("/jobs?limit=30")).json();
    const sel = $("jobs"), cur = sel.value;
    sel.innerHTML = '<option value="">Recent jobs…</option>' + jobs.map((j) =>
      `<option value="${j.id}">${j.filename} · ${j.status} · ${new Date(j.created * 1000).toLocaleTimeString()}</option>`).join("");
    sel.value = state.jobId || cur;
  } catch (_) {}
}

function setDownloads(enabled) {
  for (const [id, href] of [["dlGraphml", `/jobs/${state.jobId}/result?format=graphml`],
                            ["dlDexpi", `/jobs/${state.jobId}/result?format=dexpi`],
                            ["dlOverlay", `/jobs/${state.jobId}/overlay`]]) {
    const a = $(id);
    a.classList.toggle("disabled", !enabled);
    a.href = enabled ? href : "#";
  }
}

async function openJob(jobId) {
  clearInterval(state.poll);
  Object.assign(state, { jobId, graph: null, sel: null, addFrom: null });
  setDownloads(false); showSelection(); updateCounts();
  const r = await api(`/jobs/${jobId}/image`);
  state.imgScale = parseFloat(r.headers.get("X-Scale") || "1");
  const blob = await r.blob();
  state.img = await createImageBitmap(blob);
  fitView();
  const poll = async () => {
    const j = await (await api(`/jobs/${jobId}`)).json();
    if (j.status === "done") {
      clearInterval(state.poll);
      state.graph = await (await api(`/jobs/${jobId}/graph`)).json();
      setStatus(`done · ${j.n_nodes} nodes, ${j.n_edges} edges` + (j.n_corrections ? ` · ${j.n_corrections} corrections` : ""));
      setDownloads(true); updateCounts(); draw(); refreshJobs();
    } else if (j.status === "failed") {
      clearInterval(state.poll); setStatus(`failed: ${j.error}`); refreshJobs();
    } else {
      setStatus(`${j.status} · ${j.stage} (${Math.round(100 * j.progress)}%)`);
    }
  };
  await poll();
  if (!state.graph && state.jobId === jobId) state.poll = setInterval(poll, 1500);
}

$("jobs").onchange = () => { if ($("jobs").value) openJob($("jobs").value).catch((e) => setStatus(e.message)); };
$("upload").onsubmit = async (ev) => {
  ev.preventDefault();
  const f = $("file").files[0];
  if (!f) return;
  const body = new FormData(); body.append("file", f);
  setStatus("uploading…");
  try {
    const job = await (await api("/jobs", { method: "POST", body })).json();
    await refreshJobs();
    await openJob(job.id);
  } catch (err) { setStatus(`upload failed: ${err.message}`); }
};

resize();
refreshJobs();
api("/health").then((r) => r.json()).then((h) => {
  if (!h.model_loaded) setStatus(`model not loaded: ${h.model_error || "unknown"}`);
}).catch(() => {});
