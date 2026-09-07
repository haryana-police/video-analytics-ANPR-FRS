// ==========================================================================
// VIDEO ANALYTICS ANPR FRS — v5 VERIFIABLE PIPELINE FRONTEND
//
// Architecture: MJPEG raw frames → hidden <img> → <canvas>; SSE detection
// events drive independent per-stage rendering (Raw / Objects / Plates /
// OCR / Tracking / All). Boxes are hit-tested → detail modal. Uploads run
// through /api/detect + /api/detect_video and open a per-track audit trail
// (/api/track_details) with character-level voting evidence.
// ==========================================================================

"use strict";

const $ = (id) => document.getElementById(id);

// --- Global state ---------------------------------------------------------
const state = {
    // live session
    sessionId: null,
    mjpegUrl: null,
    eventsUrl: null,
    stopUrl: null,
    eventSource: null,
    mjpegImg: null,
    busy: false,
    activeStage: "raw",

    // per-frame detection snapshots (canvas rendering)
    cocoDets: [],
    framePlates: [],           // plates visible in the CURRENT frame only
    trajectories: new Map(),   // track_id -> [{frame, cx, cy}, …] insertion-ordered

    // CUMULATIVE session history — every track ever seen stays until the
    // next Start. Ordered by first appearance so DOM rows never reshuffle.
    history: [],               // [{key, kind, trackId, …evidence}]
    historyIdx: new Map(),     // key -> record

    // canvas
    canvas: null,
    ctx: null,
    hoveredKey: null,
};

const VEHICLE_ICONS = { bicycle: "i-car", car: "i-car", motorcycle: "i-car", bus: "i-car", truck: "i-car" };
void VEHICLE_ICONS; // reserved for per-class icons on cards

// Canvas palette — mirrors the CSS stage identity colors
const C = {
    vehicle: "#fbbf24",
    person: "#38bdf8",
    plateOk: "#34d399",
    plateBad: "#f87171",
    trajLine: "#a78bfa",
    trajDot: "#f59e0b",
    trackText: "#a78bfa",
};

function escapeHtml(s) {
    return (s ?? "").toString().replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

function icon(name, cls = "") {
    return `<svg class="ic ${cls}"><use href="#${name}"/></svg>`;
}

// ==========================================================================
// Console panel (capped)
// ==========================================================================
function log(...args) {
    const line = args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" ");
    const el = $("console-log");
    if (el) {
        const t = new Date().toLocaleTimeString();
        const cls = /error|failed|⚠/i.test(line) ? "err" : (/✓|ok\b|done/i.test(line) ? "ok" : "");
        const div = document.createElement("div");
        div.innerHTML = `<span class="ts">${t}</span><span class="${cls}">${escapeHtml(line)}</span>`;
        el.appendChild(div);
        while (el.childElementCount > 250) el.removeChild(el.firstChild);
        el.scrollTop = el.scrollHeight;
    }
    console.log("[anpr]", ...args);
}

// ==========================================================================
// Init
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    setupCanvas();
    setupStageTabs();
    setupResultsTabs();
    setupDetailModal();
    setupAuditModal();
    setupBenchmark();
    setupModeToggle();
    setupUploadTabs();
    setupImageUpload();
    setupVideoUpload();

    $("btn-start").addEventListener("click", startLive);
    $("btn-stop").addEventListener("click", stopLive);
    $("btn-pause").addEventListener("click", togglePause);
    $("btn-refresh-sources").addEventListener("click", refreshSources);
    $("upload-results-clear").addEventListener("click", resetUploadResults);

    checkHealth();
    setInterval(checkHealth, 5000);
    refreshSources();
    log("Dashboard v5 loaded — pick a source and press Start live.");
});

// ==========================================================================
// Health + compute-device chip
// ==========================================================================
async function checkHealth() {
    try {
        const r = await fetch("/health/full");
        const d = await r.json();
        const dot = $("health-dot");
        dot.classList.remove("ready", "warn", "error");
        if (!d.awiros_loaded) {
            dot.classList.add("warn");
            $("health-text").textContent = "Degraded — OCR model not loaded";
        } else {
            dot.classList.add("ready");
            $("health-text").textContent = `Pipeline ready · ${d.cached_models?.length ?? 0} models cached`;
        }
        if (d.device) $("device-text").textContent = d.device;
        $("footer-status").textContent = "ONLINE";
    } catch {
        const dot = $("health-dot");
        dot.classList.remove("ready", "warn");
        dot.classList.add("error");
        $("health-text").textContent = "Pipeline offline";
        $("footer-status").textContent = "OFFLINE";
    }
}

// ==========================================================================
// Source picker
// ==========================================================================
async function refreshSources() {
    try {
        const r = await fetch("/api/live/sources");
        const d = await r.json();
        const sel = $("source-select");
        sel.innerHTML = "";
        if (!d.sources || d.sources.length === 0) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "(no sources found)";
            sel.appendChild(opt);
            return;
        }
        for (const s of d.sources) {
            const opt = document.createElement("option");
            opt.value = s.path;
            opt.textContent = s.kind === "camera"
                ? `Webcam (index 0)`
                : `${s.name} · ${s.size_mb} MB`;
            sel.appendChild(opt);
        }
        log(`Loaded ${d.sources.length} source(s) from ${d.sample_videos_dir}`);
    } catch (e) {
        log("refreshSources failed:", e.message);
    }
}

// ==========================================================================
// Live session lifecycle
// ==========================================================================
async function startLive() {
    if (state.busy) return;
    state.busy = true;
    setBtnBusy($("btn-start"), true);

    let source = $("source-url").value.trim();
    if (!source) source = $("source-select").value;
    if (!source) {
        log("ERROR: pick a sample or paste a RTSP/HTTP URL.");
        state.busy = false;
        setBtnBusy($("btn-start"), false);
        return;
    }

    const body = {
        source,
        target_fps: Math.min(30, Math.max(1, parseInt($("target-fps").value || "15", 10))),
        model_coco: $("model-coco").value,
        model_plate: $("model-plate").value,
        ocr_on_best_only: $("ocr-strategy").value === "best",
        best_crop_algo: $("crop-algo").value,
        loop: $("loop-toggle").checked,
    };

    log(`Starting session — source=${source} · models=(${body.model_coco} / ${body.model_plate}) · ocr=${body.ocr_on_best_only ? "best-crop" : "every-frame"} · criterion=${body.best_crop_algo}`);

    try {
        const r = await fetch("/api/live/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${r.status}`);
        }
        const d = await r.json();
        state.sessionId = d.session_id;
        state.mjpegUrl = d.mjpeg_url;
        state.eventsUrl = d.events_url;
        state.stopUrl = d.stop_url;

        resetLiveBuffers();          // fresh session → fresh history
        state.paused = false;
        setPauseUi(false);
        $("meta-session").textContent = `session ${state.sessionId.slice(0, 6)}…`;
        $("btn-stop").disabled = false;
        hideStreamAlert();
        $("viewport-overlay-empty").classList.add("hidden");

        // MJPEG → hidden img → canvas
        if (!state.mjpegImg) {
            state.mjpegImg = new Image();
            state.mjpegImg.onload = drawCanvas;
        }
        state.mjpegImg.src = state.mjpegUrl + "?t=" + Date.now();

        openEventSource();
        log(`Session ${state.sessionId} started — MJPEG + SSE connected.`);
    } catch (e) {
        log("startLive failed:", e.message);
        showStreamAlert(e.message);
    } finally {
        state.busy = false;
        setBtnBusy($("btn-start"), false);
    }
}

async function stopLive() {
    if (!state.sessionId) return;
    log("Stopping live session…");
    try {
        if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
        if (state.mjpegImg) state.mjpegImg.src = "";
        await fetch(state.stopUrl, { method: "POST" });
        log(`Live session stopped — ${state.history.length} tracks kept in the results panel.`);
    } catch (e) {
        log("stopLive failed:", e.message);
    } finally {
        state.sessionId = null;
        state.mjpegUrl = null;
        state.eventsUrl = null;
        // NOTE: history is deliberately NOT cleared here — accumulated
        // results persist until the next Start begins a new session.
        $("meta-session").textContent = "session ended";
        $("btn-stop").disabled = true;
        state.paused = false;
        setPauseUi(false);
        $("viewport-overlay-empty").classList.remove("hidden");
        if (state.ctx && state.canvas) {
            state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        }
    }
}

// ---------------------------------------------------------------------------
// Pause / resume — freezes server-side processing; viewport holds last frame
// ---------------------------------------------------------------------------
async function togglePause() {
    if (!state.sessionId) return;
    const target = !state.paused;
    try {
        const r = await fetch(`/api/live/pause/${state.sessionId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paused: target }),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        state.paused = target;
        setPauseUi(target);
        log(target ? "Paused — pipeline frozen at current frame." : "Resumed.");
    } catch (e) {
        log("Pause failed:", e.message);
    }
}

function setPauseUi(paused) {
    const btn = $("btn-pause");
    btn.disabled = !state.sessionId;
    $("btn-pause-label").textContent = paused ? "Resume" : "Pause";
    btn.querySelector("use").setAttribute("href", paused ? "#i-play" : "#i-pause");
    btn.title = paused ? "Resume processing" : "Freeze processing — the viewport holds its last frame";
}

function resetLiveBuffers() {
    // New session: wipe cumulative history + DOM rows + counters.
    state.cocoDets = [];
    state.framePlates = [];
    state.trajectories.clear();
    state.history = [];
    state.historyIdx.clear();
    state.hoveredKey = null;
    for (const id of ["vehicles-list", "plates-list", "persons-list"]) {
        $(id).innerHTML = `<p class="empty">No ${id.startsWith("vehicles") ? "vehicles" : id.startsWith("plates") ? "plates" : "persons"} tracked yet.</p>`;
    }
    ["sum-vehicles", "sum-valid-plates", "sum-plates", "sum-persons",
     "tab-count-vehicles", "tab-count-plates", "tab-count-persons"].forEach(id => $(id).textContent = "0");
}

// ==========================================================================
// SSE
// ==========================================================================
function openEventSource() {
    if (!state.eventsUrl) return;
    if (state.eventSource) state.eventSource.close();
    const es = new EventSource(state.eventsUrl);
    state.eventSource = es;

    es.addEventListener("frame", (e) => {
        try { handleFrameEvent(JSON.parse(e.data)); }
        catch (err) { log("bad frame event:", err.message); }
    });
    es.addEventListener("end", () => {
        log("Server signaled end of stream.");
        stopLive();
    });
    es.addEventListener("error", (e) => {
        let msg = "Pipeline error";
        try { msg = JSON.parse(e.data).message || msg; } catch { /* keep default */ }
        log("Pipeline error:", msg);
        showStreamAlert(msg);
    });
    es.onerror = () => {
        // EventSource auto-reconnects while the session exists; a dead
        // session ends with an explicit "end" event instead.
        log("SSE connection interrupted — retrying…");
    };
}

function showStreamAlert(msg) {
    $("stream-alert-text").textContent = msg;
    $("stream-alert").classList.remove("hidden");
}
function hideStreamAlert() {
    $("stream-alert").classList.add("hidden");
}

function handleFrameEvent(d) {
    hideStreamAlert();
    if (typeof d.fps === "number") $("meta-fps").textContent = `fps ${d.fps.toFixed(1)}`;
    $("meta-frame").textContent = `frame ${d.frame_index}`;
    if (d.frame_size) {
        $("meta-size").textContent = `${d.frame_size[0]} × ${d.frame_size[1]}`;
    }

    // timing bar — proportional fill + readout + HUD text
    if (d.timing) {
        const t = d.timing;
        const total = Math.max(t.total_ms || 1, 1);
        setSeg("fill-coco", t.coco_ms, total, `${t.coco_ms} ms`);
        setSeg("fill-plate", t.plate_ms, total, `${t.plate_ms} ms`);
        setSeg("fill-ocr", t.ocr_ms, total, t.ocr_ms > 0 ? `${t.ocr_ms} ms` : "—");
        $("timing-total").textContent = `${t.total_ms} ms`;
        $("timing-ocr-fired").textContent = t.ocr_fired ? "✓" : "—";
        state.hudText = true;
    } else {
        state.hudText = false;
    }

    // raw per-frame detections (canvas rendering)
    state.cocoDets = d.coco_detections || [];
    state.framePlates = (d.plates || []).filter(p => p.bbox_xyxy);

    // accumulate trajectory history client-side (server sends last 3 points)
    for (const [tidRaw, points] of Object.entries(d.trajectories || {})) {
        const tid = Number(tidRaw);
        if (!state.trajectories.has(tid)) state.trajectories.set(tid, []);
        const hist = state.trajectories.get(tid);
        const lastFrame = hist.length ? hist[hist.length - 1].frame : 0;
        for (const pt of points) {
            if (pt[0] > lastFrame) hist.push({ frame: pt[0], cx: pt[1], cy: pt[2] });
        }
        if (hist.length > 40) state.trajectories.set(tid, hist.slice(-40));
    }
    if (state.trajectories.size > 120) {
        const excess = state.trajectories.size - 100;
        for (const k of [...state.trajectories.keys()].slice(0, excess)) state.trajectories.delete(k);
    }

    // CUMULATIVE upsert — merge this frame's tracks into session history.
    // Records are created once, updated in place, and never dropped on stop.
    for (const v of d.vehicles || []) upsertRecord("vehicle", v);
    for (const p of d.persons || []) upsertRecord("person", p);
    for (const p of d.plates || []) upsertRecord("plate", p);
    trimHistory();
    syncCards();
}

function setSeg(id, valueMs, totalMs, label) {
    const el = $(id);
    el.style.width = `${Math.min(100, (valueMs / totalMs) * 100)}%`;
    el.title = label;
    const outId = { "fill-coco": "timing-coco", "fill-plate": "timing-plate", "fill-ocr": "timing-ocr" }[id];
    $(outId).textContent = label;
}

// ==========================================================================
// Canvas — per-stage rendering + hit testing
// ==========================================================================
function setupCanvas() {
    state.canvas = $("live-canvas");
    state.ctx = state.canvas.getContext("2d");

    state.canvas.addEventListener("click", (e) => {
        const hit = hitTest(canvasPoint(e));
        if (!hit) return;
        if (hit.kind === "vehicle") openVehicleDetail(hit.track_id);
        else if (hit.kind === "plate") openPlateDetail(hit.track_id);
        else openPersonDetail(hit.track_id);
    });

    state.canvas.addEventListener("mousemove", (e) => {
        const hit = hitTest(canvasPoint(e));
        const key = hit ? `${hit.kind}:${hit.track_id}` : null;
        state.canvas.style.cursor = hit ? "pointer" : "default";
        if (key !== state.hoveredKey) {           // redraw only on real change
            state.hoveredKey = key;
            drawCanvas();
        }
    });

    state.canvas.addEventListener("mouseleave", () => {
        if (state.hoveredKey !== null) {
            state.hoveredKey = null;
            drawCanvas();
        }
        state.canvas.style.cursor = "default";
    });
}

function canvasPoint(e) {
    const rect = state.canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) * (state.canvas.width / rect.width),
        y: (e.clientY - rect.top) * (state.canvas.height / rect.height),
    };
}

function hitTest({ x, y }) {
    // plates sit on top of vehicles visually — test them first
    for (const p of state.framePlates) {
        if (p.bbox_xyxy && inBox(x, y, p.bbox_xyxy)) return { kind: "plate", track_id: p.track_id };
    }
    for (const det of state.cocoDets) {
        if (inBox(x, y, det.bbox_xyxy)) {
            return { kind: det.class_name === "person" ? "person" : "vehicle", track_id: det.track_id };
        }
    }
    return null;
}
function inBox(x, y, [x1, y1, x2, y2]) {
    return x >= x1 && x <= x2 && y >= y1 && y <= y2;
}

function drawCanvas() {
    const ctx = state.ctx, img = state.mjpegImg;
    if (!ctx || !img || !img.naturalWidth || !img.naturalHeight) return;

    // Resize when EITHER dimension changed
    if (state.canvas.width !== img.naturalWidth || state.canvas.height !== img.naturalHeight) {
        state.canvas.width = img.naturalWidth;
        state.canvas.height = img.naturalHeight;
    }

    const W = state.canvas.width, H = state.canvas.height;
    ctx.drawImage(img, 0, 0, W, H);

    const stage = state.activeStage;
    const hovered = state.hoveredKey;

    if (stage === "objects" || stage === "all") drawObjects(ctx, hovered);
    if (stage === "plates" || stage === "all") drawPlates(ctx, hovered, false);
    if (stage === "ocr" || stage === "all") drawPlates(ctx, hovered, true);
    if (stage === "tracking" || stage === "all") drawTracking(ctx);

    if (stage !== "raw" && state.hudText) drawHud(ctx, W, H);
}

function drawObjects(ctx, hovered) {
    for (const det of state.cocoDets) {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const isPerson = det.class_name === "person";
        const color = isPerson ? C.person : C.vehicle;
        const key = `${isPerson ? "person" : "vehicle"}:${det.track_id}`;
        ctx.strokeStyle = color;
        ctx.lineWidth = hovered === key ? 3.5 : 1.8;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        labelBox(ctx, `${det.class_name} #${det.track_id} ${(det.confidence || 0).toFixed(2)}`,
                 x1, y1, color, 11);
    }
}

function drawPlates(ctx, hovered, ocrMode) {
    for (const p of state.framePlates) {
        if (!p.bbox_xyxy) continue;
        const [x1, y1, x2, y2] = p.bbox_xyxy;
        const hasText = p.text && p.text.length > 0;
        const color = !p.ocr_done ? "#94a3b8" : (hasText ? (p.valid ? C.plateOk : C.plateBad) : C.plateBad);
        const key = `plate:${p.track_id}`;

        ctx.strokeStyle = color;
        ctx.lineWidth = hovered === key ? 4 : 2.6;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        if (ocrMode) {
            const ocrText = hasText ? p.text : (p.ocr_done ? "(unreadable)" : "OCR pending…");
            const confTxt = hasText ? ` ${(p.ocr_conf || 0).toFixed(2)}` : "";
            const ly = Math.min(y2 + 22, state.canvas.height - 4);
            labelBox(ctx, `${ocrText}${confTxt}${p.valid ? "  ✓IND" : ""}`,
                     x1, ly - 18, color, 13, { below: true });        } else {
            labelBox(ctx, `plate #${p.track_id} ${(p.confidence || 0).toFixed(2)}`, x1, y1, color, 11);
        }
    }
}

function drawTracking(ctx) {
    // trajectories — most recently updated tracks first
    const entries = [...state.trajectories.entries()].slice(-30);
    for (const [tid, pts] of entries) {
        if (!pts || pts.length < 2) continue;
        ctx.strokeStyle = C.trajLine;
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((pt, i) => i === 0 ? ctx.moveTo(pt.cx, pt.cy) : ctx.lineTo(pt.cx, pt.cy));
        ctx.stroke();

        const last = pts[pts.length - 1];
        ctx.fillStyle = C.trajDot;
        ctx.beginPath();
        ctx.arc(last.cx, last.cy, 3.2, 0, Math.PI * 2);
        ctx.fill();
        labelBox(ctx, `#${tid}`, last.cx + 4, last.cy - 14, C.trackText, 10, { plainBg: true });
    }
    // dashed current boxes with IDs
    for (const det of state.cocoDets) {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const color = det.class_name === "person" ? C.person : C.vehicle;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);
        labelBox(ctx, `#${det.track_id}`, x1, y1, color, 10, { compact: true });
    }
}

function labelBox(ctx, text, x, y, color, fontSize, opts = {}) {
    ctx.font = `${opts.plainBg ? "" : "600 "} ${fontSize}px 'JetBrains Mono', monospace`;
    const tw = ctx.measureText(text).width;
    const bx = opts.below ? Math.max(x, 2) : x;
    const by = opts.below ? y : Math.max(y - 16, 0);
    if (!opts.plainBg) {
        ctx.fillStyle = "rgba(4,6,10,0.78)";
        ctx.fillRect(bx, by, tw + 10, 16);
    } else {
        ctx.fillStyle = "rgba(4,6,10,0.55)";
        ctx.fillRect(bx, by, tw + 8, 14);
    }
    ctx.fillStyle = color;
    ctx.fillText(text, bx + 4, by + 12);
}

function drawHud(ctx, W, H) {
    const fps = $("meta-fps").textContent;
    const frame = $("meta-frame").textContent;
    const t = $("timing-total").textContent;
    const txt = `${frame} · ${fps} · Σ ${t}`;
    ctx.font = "600 12px 'JetBrains Mono', monospace";
    const tw = ctx.measureText(txt).width;
    ctx.fillStyle = "rgba(4,6,10,0.72)";
    ctx.fillRect(W - tw - 26, 8, tw + 18, 24);
    ctx.fillStyle = "#f2f5fa";
    ctx.fillText(txt, W - tw - 17, 24);
}

// ==========================================================================
// Stage tabs
// ==========================================================================
function setupStageTabs() {
    document.querySelectorAll(".stage-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            const stage = btn.dataset.stage;
            document.querySelectorAll(".stage-tab").forEach(b => {
                const on = b.dataset.stage === stage;
                b.classList.toggle("active", on);
                b.setAttribute("aria-selected", on ? "true" : "false");
            });
            state.activeStage = stage;
            drawCanvas();
        });
    });
}

// ==========================================================================
// Results tabs
// ==========================================================================
function setupResultsTabs() {
    document.querySelectorAll(".results-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll(".results-tab").forEach(b => {
                const on = b.dataset.tab === tab;
                b.classList.toggle("active", on);
                b.setAttribute("aria-selected", on ? "true" : "false");
            });
            document.querySelectorAll(".results-content").forEach(c =>
                c.classList.toggle("active", c.dataset.tab === tab));
        });
    });
}

// ==========================================================================
// Track cards
// ==========================================================================
function plateReadoutHtml(text, done, conf, valid) {
    if (!text && !done) {
        return `<div class="plate-readout"><span class="plate-text muted">OCR pending…</span></div>`;
    }
    const cls = valid ? "valid" : "tentative";
    const confTxt = done ? `<span class="plate-conf">${(conf || 0).toFixed(2)}</span>` : "";
    const badge = valid ? `<span class="valid-badge">${icon("i-check")}IND</span>` : "";
    return `<div class="plate-readout ${cls}">
        <span class="plate-text">${text ? escapeHtml(text) : "(unreadable)"}</span>
        ${confTxt}${badge}
    </div>`;
}

// --- cumulative record model ---------------------------------------------
const MAX_HISTORY = 500;

const areaOf = (bbox) => bbox ? Math.max(0, bbox[2] - bbox[0]) * Math.max(0, bbox[3] - bbox[1]) : 0;

function upsertRecord(kind, t) {
    const prefix = kind === "vehicle" ? "v" : kind === "person" ? "s" : "p";
    const key = `${prefix}:${t.track_id}`;
    let r = state.historyIdx.get(key);
    if (!r) {
        r = {
            key, kind, trackId: t.track_id,
            className: t.class_name || (kind === "person" ? "person" : kind === "plate" ? "plate" : "vehicle"),
            firstSeenMs: Date.now(), lastSeenMs: Date.now(),
            nFrames: 0, confidence: 0,
            bboxLast: null, bboxBest: null, _bestArea: 0,
            ocrDone: false, plateText: "", plateConf: 0, plateValid: false,
            plateTrackId: null, linkedVehicleId: null,
            // stage evidence (JPEG dataURLs captured from live MJPEG frames)
            evRaw: null, evObj: null, _evArea: 0,
        };
        state.history.push(r);
        state.historyIdx.set(key, r);
    }
    r.lastSeenMs = Date.now();
    r.nFrames = Math.max(r.nFrames, t.n_frames || 0);
    r.confidence = Math.max(r.confidence, t.confidence || 0);
    if (t.bbox_xyxy) {
        r.bboxLast = t.bbox_xyxy;
        const a = areaOf(t.bbox_xyxy);
        if (a > r._bestArea) { r._bestArea = a; r.bboxBest = t.bbox_xyxy.slice(); }
    }
    if (kind === "vehicle") {
        if (t.ocr_done || t.plate_text) {
            r.ocrDone = !!t.ocr_done;
            r.plateText = t.plate_text || "";
            r.plateConf = t.plate_conf || 0;
            r.plateValid = !!t.plate_valid;
        }
        if (t.plate_track_id != null) r.plateTrackId = t.plate_track_id;
    } else if (kind === "plate") {
        r.ocrDone = !!t.ocr_done;
        if (t.text) r.plateText = t.text;
        r.plateConf = Math.max(r.plateConf, t.ocr_conf || 0);
        r.plateValid = !!t.valid;
        if (t.linked_vehicle_id != null) r.linkedVehicleId = t.linked_vehicle_id;
    }
    ensureEvidence(r);
}

// --- evidence snapshots (Raw / Object-crop / Plate-crop), client-side -----
let _snapCanvas = null;
let _rawCache = { url: null, at: 0 };

function snapDataURL(sx, sy, sw, sh, maxW, quality) {
    const img = state.mjpegImg;
    if (!img || !img.naturalWidth || !img.naturalHeight || sw < 2 || sh < 2) return null;
    const x = Math.max(0, Math.round(sx)), y = Math.max(0, Math.round(sy));
    const w = Math.min(img.naturalWidth - x, Math.round(sw));
    const h = Math.min(img.naturalHeight - y, Math.round(sh));
    if (w < 2 || h < 2) return null;
    const scale = Math.min(1, maxW / w);
    const cw = Math.max(2, Math.round(w * scale)), ch = Math.max(2, Math.round(h * scale));
    if (!_snapCanvas) _snapCanvas = document.createElement("canvas");
    _snapCanvas.width = cw; _snapCanvas.height = ch;
    const ctx = _snapCanvas.getContext("2d");
    ctx.drawImage(img, x, y, w, h, 0, 0, cw, ch);
    try { return _snapCanvas.toDataURL("image/jpeg", quality); } catch { return null; }
}

function rawSnapshot() {
    const now = performance.now();
    if (!_rawCache.url || now - _rawCache.at > 800) {
        const img = state.mjpegImg;
        _rawCache.url = (img && img.naturalWidth)
            ? snapDataURL(0, 0, img.naturalWidth, img.naturalHeight, 900, 0.6) : null;
        _rawCache.at = now;
    }
    return _rawCache.url;
}

function ensureEvidence(r) {
    if (!state.mjpegImg || !state.mjpegImg.naturalWidth) return;
    if (!r.evRaw) r.evRaw = rawSnapshot();
    const bb = r.bboxBest || r.bboxLast;
    if (!bb) return;
    const a = areaOf(bb);
    if (!r.evObj) {
        r.evObj = snapDataURL(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], 380, 0.82);
        r._evArea = a;
    } else if (a > r._evArea * 1.35) {
        // upgrade the crop when we've since seen a substantially bigger box
        const newer = snapDataURL(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1], 380, 0.82);
        if (newer) { r.evObj = newer; r._evArea = a; }
    }
}

function trimHistory() {
    while (state.history.length > MAX_HISTORY) {
        const old = state.history.shift();
        state.historyIdx.delete(old.key);
        document.querySelectorAll(`[data-key="${CSS.escape(old.key)}"]`).forEach(el => el.remove());
    }
}

// --- incremental card DOM (stable rows → no flicker, clicks always land) ---
function updateSummaryCounts() {
    let vehicles = 0, plates = 0, persons = 0, validPlates = 0;
    for (const r of state.history) {
        if (r.kind === "vehicle") vehicles++;
        else if (r.kind === "person") persons++;
        else { plates++; if (r.plateValid && r.plateText) validPlates++; }
    }
    $("sum-vehicles").textContent = vehicles;
    $("sum-valid-plates").textContent = validPlates;
    $("sum-plates").textContent = plates;
    $("sum-persons").textContent = persons;
    $("tab-count-vehicles").textContent = vehicles;
    $("tab-count-plates").textContent = plates;
    $("tab-count-persons").textContent = persons;
}

function syncCards() {
    syncKind("vehicles-list", "vehicle");
    syncKind("plates-list", "plate");
    syncKind("persons-list", "person");
    updateSummaryCounts();
}

function syncKind(listId, kind) {
    const root = $(listId);
    for (const r of state.history) {
        if (r.kind !== kind) continue;
        let el = root.querySelector(`[data-key="${CSS.escape(r.key)}"]`);
        if (!el) {
            root.querySelector(":scope > .empty")?.remove();
            el = buildCard(r);
            root.appendChild(el);           // history order is append-only → stable
        } else {
            updateCard(el, r);
        }
    }
}

function buildCard(r) {
    const el = document.createElement("div");
    el.className = `tcard tcard-kind-${r.kind}`;
    el.dataset.key = r.key;
    const ic = r.kind === "vehicle" ? "i-car" : r.kind === "person" ? "i-person" : "i-plate";
    const title = r.kind === "vehicle" ? escapeHtml(r.className) : r.kind === "plate" ? "Plate" : "Person";
    el.innerHTML = `
        <div class="tcard-head">
            <svg class="ic"><use href="#${ic}"/></svg>
            <span class="tcard-title">${title} <span class="mono">#${r.trackId}</span></span>
            <span class="f-conf tcard-conf"></span>
        </div>
        ${r.kind !== "person" ? `<div class="f-plate"></div>` : ""}
        <div class="tcard-meta"><span class="f-frames"></span><span class="f-link"></span></div>`;
    el.addEventListener("click", () => openTrackDetail(r.key));
    el._f = {
        conf: el.querySelector(".f-conf"),
        frames: el.querySelector(".f-frames"),
        plate: el.querySelector(".f-plate"),
        link: el.querySelector(".f-link"),
        sig: "",
    };
    updateCard(el, r);
    return el;
}

function updateCard(el, r) {
    const f = el._f;
    f.conf.textContent = `yolo ${(r.confidence || 0).toFixed(2)}`;
    f.frames.textContent = `${r.nFrames} frames`;
    if (f.plate) {
        const hasPlateInfo = r.kind === "plate" || r.ocrDone || r.plateTrackId != null;
        const sig = `${hasPlateInfo}|${r.plateText}|${r.plateValid}|${(r.plateConf || 0).toFixed(2)}`;
        if (sig !== f.sig) {
            f.sig = sig;
            f.plate.innerHTML = hasPlateInfo
                ? plateReadoutHtml(r.plateText, r.ocrDone || !!r.plateText, r.plateConf, r.plateValid)
                : `<div class="plate-readout"><span class="plate-text muted">OCR pending…</span></div>`;
        }
    }
    if (f.link) {
        f.link.textContent = "";
        if (r.kind === "plate" && r.linkedVehicleId != null) {
            f.link.innerHTML = `<span class="link-badge">↔ vehicle #${r.linkedVehicleId}</span>`;
        } else if (r.kind === "vehicle" && r.plateTrackId != null) {
            f.link.innerHTML = `<span class="link-badge">↔ plate #${r.plateTrackId}</span>`;
        }
    }
}

// ==========================================================================
// Detail modal (live tracks)
// ==========================================================================
function setupDetailModal() {
    $("detail-modal-close").addEventListener("click", closeDetailModal);
    $("detail-modal").addEventListener("click", (e) => {
        if (e.target.id === "detail-modal") closeDetailModal();
    });
}

function openModal(el) { el.classList.remove("hidden"); el.setAttribute("aria-hidden", "false"); }
function closeModal(el) { el.classList.add("hidden"); el.setAttribute("aria-hidden", "true"); }

function closeDetailModal() { closeModal($("detail-modal")); }

function chipRow(chips) {
    return chips.map(([k, v, cls]) =>
        `<div class="dchip ${cls || ""}">${k}:<b>${v}</b></div>`).join("");
}

function kvRows(rows) {
    return rows.map(([k, v]) =>
        `<div class="kv-row"><span>${k}</span><code>${v}</code></div>`).join("");
}

// --- pipeline stage-sequence rendering -------------------------------------
function stageImg(label, sub, url) {
    return `<div class="stage-card">
        <div class="stage-head"><span>${label}</span>${sub ? `<span class="sub">${sub}</span>` : ""}</div>
        <img src="${url}" alt="${label}">
    </div>`;
}

function stageOcr(text, conf, valid) {
    const cls = valid && text ? "ok" : (text ? "warn" : "bad");
    const verdict = valid && text ? "✓ VALID INDIAN PLATE"
                  : text ? "⚠ FORMAT REVIEW" : "NO TEXT READ";
    return `<div class="stage-card">
        <div class="stage-head"><span>04 · OCR OUTPUT</span><span class="sub">Awiros ANPR</span></div>
        <div class="ocr-out">
            <div class="ocr-text ${cls}">${escapeHtml(text || "(no read)")}</div>
            <div class="ocr-verdict" style="color:var(--${cls === "ok" ? "ok" : cls === "warn" ? "warn" : "bad"})">${verdict}</div>
            <div class="ocr-sub">ocr conf ${(conf || 0).toFixed(4)} · per-character votes in Upload → Video audit</div>
        </div>
    </div>`;
}

function openTrackDetail(key) {
    const r = state.historyIdx.get(key);
    if (!r) return;

    // linked plate record (for vehicles)
    const plateRec = r.kind === "vehicle" && r.plateTrackId != null
        ? state.historyIdx.get(`p:${r.plateTrackId}`) : null;
    const ocrSource = r.kind === "plate" ? r : plateRec;

    if (r.kind === "vehicle") {
        $("detail-modal-title").innerHTML = `${icon("i-car")} ${escapeHtml(r.className)} #${r.trackId}`;
        $("detail-modal-sub").textContent = `First seen ${new Date(r.firstSeenMs).toLocaleTimeString()} · last active ${new Date(r.lastSeenMs).toLocaleTimeString()}`;
    } else if (r.kind === "plate") {
        $("detail-modal-title").innerHTML = `${icon("i-plate")} Plate #${r.trackId}`;
        $("detail-modal-sub").textContent = `OCR ${r.ocrDone ? "complete" : "pending"} · first seen ${new Date(r.firstSeenMs).toLocaleTimeString()}`;
    } else {
        $("detail-modal-title").innerHTML = `${icon("i-person")} Person #${r.trackId}`;
        $("detail-modal-sub").textContent = `Tracked across ${r.nFrames} frames`;
    }

    $("detail-modal-summary").innerHTML = chipRow([
        ["track_id", r.trackId],
        ...(r.kind !== "person" ? [["frames seen", r.nFrames]] : []),
        ["yolo conf", (r.confidence || 0).toFixed(3)],
        ...(r.kind !== "plate" ? [] : [["linked vehicle", r.linkedVehicleId ?? "(none)"]]),
        ...(r.kind === "vehicle" ? [["plate track", r.plateTrackId ?? "(none)"]] : []),
        ...(ocrSource ? [
            ["text", escapeHtml(ocrSource.plateText || "—")],
            ["ocr conf", (ocrSource.plateConf || 0).toFixed(3)],
            ["valid indian", ocrSource.plateValid ? "✓ yes" : "✗ no", ocrSource.plateValid ? "ok" : "bad"],
        ] : []),
    ]);

    // ── the verifiable sequence: Raw → Object/Plate → OCR ──
    const stages = [];
    if (r.evRaw) stages.push(stageImg("01 · RAW FRAME", "live MJPEG frame", r.evRaw));
    else stages.push(`<div class="stage-card"><div class="stage-head"><span>01 · RAW FRAME</span></div><div class="placeholder-img"><p class="empty">Frame not captured.</p></div></div>`);

    if (r.kind === "plate") {
        if (r.evObj) stages.push(stageImg("02 · PLATE CROPPED", `yolo ${(r.confidence || 0).toFixed(2)}`, r.evObj));
    } else if (r.evObj) {
        stages.push(stageImg("02 · OBJECT DETECTED", `${escapeHtml(r.className)} · yolo ${(r.confidence || 0).toFixed(2)}`, r.evObj));
        if (plateRec?.evObj) stages.push(stageImg("03 · PLATE CROPPED", `track #${plateRec.trackId}`, plateRec.evObj));
    }

    if ((r.kind === "plate" || r.kind === "vehicle") && ocrSource && (ocrSource.ocrDone || ocrSource.plateText)) {
        stages.push(stageOcr(ocrSource.plateText, ocrSource.plateConf, ocrSource.plateValid));
    }

    $("detail-stage-flow").innerHTML = stages.join(
        `<svg class="ic stage-arrow"><use href="#i-chev"/></svg>`);

    $("detail-last-seen").innerHTML = kvRows([
        ["Last bbox (xyxy)", (r.bboxLast || []).join(", ")],
        ["Best bbox (xyxy)", (r.bboxBest || []).join(", ")],
        ...(r.kind !== "person" ? [["OCR done", r.ocrDone ? "yes" : "no"]] : []),
        ...(plateRec ? [["Plate track", `#${plateRec.trackId}`]] : []),
    ]);
    const meta = { ...r };
    delete meta._f; delete meta.evRaw; delete meta.evObj;
    $("detail-meta").textContent = JSON.stringify(meta, null, 2);
    openModal($("detail-modal"));
}

function openVehicleDetail(tid) { openTrackDetail(`v:${tid}`); }
function openPlateDetail(tid) { openTrackDetail(`p:${tid}`); }
function openPersonDetail(tid) { openTrackDetail(`s:${tid}`); }

// ==========================================================================
// Benchmark modal
// ==========================================================================
function setupBenchmark() {
    $("btn-benchmark").addEventListener("click", () => openModal($("benchmark-backdrop")));
    $("btn-benchmark-close").addEventListener("click", () => closeModal($("benchmark-backdrop")));
    $("benchmark-backdrop").addEventListener("click", (e) => {
        if (e.target.id === "benchmark-backdrop") closeModal($("benchmark-backdrop"));
    });
    $("btn-benchmark-run").addEventListener("click", runBenchmark);
}

async function runBenchmark() {
    const btn = $("btn-benchmark-run");
    setBtnBusy(btn, true, "Running…");
    log("Benchmark running — measures every local YOLO variant ×5…");
    try {
        const r = await fetch("/api/live/benchmark");
        const d = await r.json();
        renderBenchmark(d.results || []);
        log(`Benchmark done — ${(d.results || []).length} rows measured.`);
    } catch (e) {
        log("Benchmark failed:", e.message);
    } finally {
        setBtnBusy(btn, false);
    }
}

function renderBenchmark(results) {
    const tbody = $("benchmark-tbody");
    if (!results.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="empty">No models found.</td></tr>`;
        return;
    }
    tbody.innerHTML = results.map(r => {
        if (r.kind === "frame") {
            return `<tr><td><b>${escapeHtml(r.name)}</b></td><td>frame</td><td colspan="6" class="hint">${escapeHtml(r.source || "")}</td></tr>`;
        }
        if (r.error) {
            return `<tr><td>${escapeHtml(r.name)}</td><td>${escapeHtml(r.kind)}</td><td colspan="6" class="err">${escapeHtml(r.error)}</td></tr>`;
        }
        let verdict = "";
        const fps = r.max_fps || 0;
        if (r.kind === "tracker") verdict = verdictBadge("good", "negligible");
        else if (r.kind === "ocr")
            verdict = fps >= 5 ? verdictBadge("good", "fast")
                    : fps >= 1 ? verdictBadge("ok", "best-crop only")
                    : verdictBadge("bad", "bottleneck");
        else
            verdict = fps >= 10 ? verdictBadge("good", "live-ready")
                    : fps >= 4 ? verdictBadge("ok", "borderline")
                    : verdictBadge("bad", "too slow");
        const num = (v, unit) => `<td class="num">${v ?? "—"}${unit}</td>`;
        return `<tr>
            <td><b>${escapeHtml(r.name)}</b></td><td>${escapeHtml(r.kind)}</td>
            ${num(r.min_ms, " ms")}${num(r.median_ms, " ms")}${num(r.mean_ms, " ms")}${num(r.max_ms, " ms")}
            ${num(r.max_fps, " fps")}<td>${verdict}</td>
        </tr>`;
    }).join("");
}

function verdictBadge(level, label) {
    return `<span class="verdict verdict-${level}">${label}</span>`;
}

// ==========================================================================
// Mode toggle + upload tabs
// ==========================================================================
function setMode(mode) {
    document.querySelectorAll(".mode-btn").forEach(b =>
        b.classList.toggle("active", b.dataset.mode === mode));
    $("live-panel").classList.toggle("hidden", mode !== "live");
    $("upload-panel").classList.toggle("hidden", mode !== "upload");
}

function setupModeToggle() {
    document.querySelectorAll(".mode-btn").forEach(btn =>
        btn.addEventListener("click", () => setMode(btn.dataset.mode)));
}

function setupUploadTabs() {
    document.querySelectorAll(".upload-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.uploadTab;
            document.querySelectorAll(".upload-tab").forEach(b => {
                const on = b.dataset.uploadTab === tab;
                b.classList.toggle("active", on);
                b.setAttribute("aria-selected", on ? "true" : "false");
            });
            document.querySelectorAll("[data-upload-tab-pane]").forEach(pane =>
                pane.classList.toggle("hidden", pane.dataset.uploadTabPane !== tab));
            $("upload-results-title").textContent = tab === "image" ? "Image results" : "Video results";
        });
    });
}

// ==========================================================================
// Button busy helper
// ==========================================================================
function setBtnBusy(btn, busy, label) {
    if (busy) {
        btn.dataset.idleHtml = btn.innerHTML;
        btn.classList.add("is-busy");
        btn.disabled = true;
        if (label) {
            btn.classList.remove("is-busy");
            btn.innerHTML = `<span class="spinner"></span>${escapeHtml(label)}`;
            btn.dataset.busyLabel = true;
        }
    } else {
        btn.classList.remove("is-busy");
        delete btn.dataset.busyLabel;
        if (btn.dataset.idleHtml) btn.innerHTML = btn.dataset.idleHtml;
        btn.disabled = false;
    }
}

// ==========================================================================
// IMAGE UPLOAD
// ==========================================================================
let uploadImageFile = null;

function wireDropzone(dzId, inputId, contentId, previewSel, changeBtnId, acceptCheck, sizeFmt) {
    const dz = $(dzId), input = $(inputId), content = $(contentId);
    const preview = $(previewSel), changeBtn = $(changeBtnId);
    const detectBtn = dz.closest(".upload-card")?.querySelector(".btn-primary") ||
                      dz.parentElement.querySelector(".btn-primary");

    function pickFile() { input.click(); }
    dz.addEventListener("click", pickFile);
    changeBtn.addEventListener("click", (e) => { e.stopPropagation(); pickFile(); });

    ["dragenter", "dragover"].forEach(ev =>
        dz.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dz.classList.add("drag-active"); }));
    ["dragleave", "dragend"].forEach(ev =>
        dz.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); if (e.target === dz) dz.classList.remove("drag-active"); }));
    dz.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        dz.classList.remove("drag-active");
        const f = e.dataTransfer?.files?.[0];
        if (!f) return;
        if (!acceptCheck(f)) { log(`Rejected dropped file (wrong type): ${f.name}`); return; }
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
    });

    input.addEventListener("change", () => {
        const f = input.files[0];
        if (!f) return;
        const url = URL.createObjectURL(f);
        preview.src = url;
        preview.classList.remove("hidden");
        changeBtn.classList.remove("hidden");
        content.classList.add("hidden");
        detectBtn.disabled = false;
        $(dzId.replace("dropzone", "clear-btn")).disabled = false;
        log(`Selected: ${f.name} (${sizeFmt(f)})`);
        return f;
    });

    return { dz, input, content, preview, changeBtn, detectBtn };
}

function setupImageUpload() {
    const ui = wireDropzone(
        "image-dropzone", "image-file-input", "image-dropzone-content",
        "image-preview", "image-change-btn",
        (f) => f.type.startsWith("image/"),
        (f) => `${(f.size / 1024).toFixed(0)} KB`
    );
    const clearBtn = $("image-clear-btn");

    ui.input.addEventListener("change", () => { uploadImageFile = ui.input.files[0]; });
    clearBtn.addEventListener("click", () => {
        uploadImageFile = null;
        ui.input.value = "";
        ui.preview.src = "";
        ui.preview.classList.add("hidden");
        ui.changeBtn.classList.add("hidden");
        ui.content.classList.remove("hidden");
        ui.detectBtn.disabled = true;
        clearBtn.disabled = true;
        resetUploadResults();
    });
    ui.detectBtn.addEventListener("click", () => runImageDetect(ui.detectBtn));
}

async function runImageDetect(btn) {
    if (!uploadImageFile) return;
    setBtnBusy(btn, true, "Running…");
    showUploadProgress("Analyzing image through the verifiable pipeline…");
    try {
        const fd = new FormData();
        fd.append("image", uploadImageFile);
        fd.append("conf", "0.25");
        fd.append("model_coco", $("model-coco")?.value || "yolo11n");
        fd.append("model_plate", $("model-plate")?.value || "yolo11_plate");
        const r = await fetch("/api/detect", { method: "POST", body: fd });
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${r.status}`);
        }
        const d = await r.json();
        log(`Image done — ${d.num_plates} plates (${d.num_valid} valid), ${d.num_vehicles} vehicles, ${d.num_persons} persons · ${d.elapsed_seconds}s`);
        renderImageResults(d);
    } catch (e) {
        log("Image analysis failed:", e.message);
        showUploadError(e.message);
    } finally {
        setBtnBusy(btn, false);
    }
}

function renderImageResults(d) {
    showUploadBody();
    const html = `
        ${summaryTiles([
            [d.num_plates, "Plates"], [d.num_valid || 0, "Valid"],
            [d.num_vehicles, "Vehicles"], [d.num_persons, "Persons"],
        ])}
        <div class="engine-meta">
            ${escapeHtml(d.engine.detector_coco)} · ${escapeHtml(d.engine.detector_plate)} · ${escapeHtml(d.engine.ocr)}<br>
            yolo_coco ${d.inference_ms_yolo_coco}ms · yolo_plate ${d.inference_ms_yolo_plate}ms · ocr ${d.inference_ms_ocr_total}ms · total ${(d.elapsed_seconds * 1000).toFixed(0)}ms
        </div>
        <img class="result-media" src="${d.annotated_url}" alt="annotated image">
        ${renderUploadPlates(d.plates || [])}
        ${renderUploadVehicles(d.vehicles || [])}
    `;
    $("upload-results-body").innerHTML = html;
}

function renderUploadPlates(plates) {
    if (!plates.length) return "";
    const items = plates.map(p => {
        const ok = p.valid_format || p.readable;
        return `<div class="utrack">
            <div class="utrack-head">
                <span class="utrack-id mono">${escapeHtml(p.text || "(empty)")}</span>
                ${ok ? '<span class="badge badge-ok">' + icon("i-check") + 'Indian format</span>'
                     : '<span class="badge badge-warn">tentative</span>'}
            </div>
            <div class="utrack-meta">bbox ${(p.bbox_xyxy || []).join(", ")} · yolo ${(p.detection_confidence || 0).toFixed(2)} · ocr ${(p.ocr_confidence || 0).toFixed(2)}</div>
            ${p.crop_url ? `<div class="utrack-imgs"><img src="${p.crop_url}" alt="plate crop"></div>` : ""}
        </div>`;
    }).join("");
    return sectionBlock(`Plates (${plates.length})`, items);
}

function renderUploadVehicles(vehicles) {
    if (!vehicles.length) return "";
    const items = vehicles.map(v => {
        const pl = v.plate;
        const badge = pl && pl.text
            ? `<span class="utrack-id mono">${escapeHtml(pl.text)}</span>${
               (pl.readable ?? pl.valid)
                   ? '<span class="badge badge-ok">' + icon("i-check") + '</span>'
                   : '<span class="badge badge-warn">?</span>'}`
            : `<span class="tcard-conf">(no plate)</span>`;
        return `<div class="utrack">
            <div class="utrack-head">
                <svg class="ic"><use href="#i-car"/></svg>
                <span class="utrack-class">${escapeHtml(v.class_name)}</span>
                <span class="tcard-conf">yolo ${(v.confidence || 0).toFixed(2)}</span>
                ${badge}
            </div>
            <div class="utrack-meta">bbox ${(v.bbox_xyxy || []).join(", ")}</div>
            ${v.crop_url ? `<div class="utrack-imgs"><img src="${v.crop_url}" alt="vehicle crop"></div>` : ""}
        </div>`;
    }).join("");
    return sectionBlock(`Vehicles (${vehicles.length})`, items);
}

// ==========================================================================
// VIDEO UPLOAD
// ==========================================================================
let uploadVideoFile = null;

function setupVideoUpload() {
    const uv = wireDropzone(
        "video-dropzone", "video-file-input", "video-dropzone-content",
        "video-preview", "video-change-btn",
        (f) => f.type.startsWith("video/"),
        (f) => `${(f.size / 1e6).toFixed(1)} MB`
    );
    const clearBtn = $("video-clear-btn");
    const stopBtn = $("video-stop-btn");

    uv.input.addEventListener("change", () => { uploadVideoFile = uv.input.files[0]; });
    clearBtn.addEventListener("click", () => {
        uploadVideoFile = null;
        uv.input.value = "";
        uv.preview.src = "";
        uv.preview.classList.add("hidden");
        uv.changeBtn.classList.add("hidden");
        uv.content.classList.remove("hidden");
        uv.detectBtn.disabled = true;
        clearBtn.disabled = true;
        resetUploadResults();
    });
    uv.detectBtn.addEventListener("click", () => runVideoDetect(uv.detectBtn));

    stopBtn.addEventListener("click", async () => {
        stopBtn.disabled = true;
        log("Cancel requested — stopping after current frame…");
        try { await fetch("/api/cancel_video", { method: "POST" }); }
        catch { /* processing loop may be blocking; job will exit at next check */ }
    });
}

async function runVideoDetect(btn) {
    if (!uploadVideoFile) return;
    const stopBtn = $("video-stop-btn");
    setBtnBusy(btn, true, "Processing…");
    stopBtn.disabled = false;
    showUploadProgress("ByteTrack + Awiros OCR running — this can take a while (OCR ~2.5 s/crop on CPU). Press Stop to cancel.");
    try {
        const fd = new FormData();
        fd.append("video", uploadVideoFile);
        fd.append("frame_stride", $("upload-frame-stride").value || "2");
        const maxFrames = $("upload-max-frames").value;
        if (maxFrames) fd.append("max_frames", maxFrames);
        fd.append("write_video", $("upload-write-video").checked ? "1" : "0");
        fd.append("model_coco", $("model-coco")?.value || "yolo11n");
        fd.append("model_plate", $("model-plate")?.value || "yolo11_plate");
        const r = await fetch("/api/detect_video", { method: "POST", body: fd });
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${r.status}`);
        }
        const d = await r.json();
        log(`Video done — ${d.n_tracks} tracks, ${d.n_valid_plates} valid plates · ${d.fps_processed} fps processed · ${d.elapsed_seconds}s`);
        renderVideoResults(d);
    } catch (e) {
        log("Video processing failed:", e.message);
        showUploadError(e.message);
    } finally {
        setBtnBusy(btn, false);
        stopBtn.disabled = true;
    }
}

function renderVideoResults(d) {
    showUploadBody();
    const tracks = d.tracks || [];
    const videoBlock = d.annotated_video_url ? `
        <div style="margin-bottom:12px">
            <video controls class="result-media" src="${d.annotated_video_url}"></video>
            <a class="btn btn-ghost" href="${d.annotated_video_url}" download="annotated.mp4">
                ${icon("i-download")}Download annotated video
            </a>
        </div>` : "";
    const items = tracks.map(t => `
        <div class="utrack">
            <div class="utrack-head">
                <span class="utrack-id mono">#${t.track_id}</span>
                <span class="utrack-class">${escapeHtml(t.class_name || "?")}</span>
                <span class="tcard-conf">${t.n_frames} frames</span>
                ${t.final_text
                    ? `<span class="utrack-id mono">${escapeHtml(t.final_text)}</span>${
                       t.valid_indian ? '<span class="badge badge-ok">' + icon("i-check") + 'Indian format</span>'
                                      : '<span class="badge badge-warn">tentative</span>'}`
                    : '<span class="tcard-conf">(no plate OCR)</span>'}
            </div>
            <div class="utrack-meta">
                final conf ${(t.final_conf || 0).toFixed(2)} · avg yolo ${(t.avg_yolo_conf || 0).toFixed(2)} · unique reads ${t.n_unique_reads}<br>
                first @ frame ${t.first_seen} → last @ frame ${t.last_seen}
            </div>
            ${(t.best_crop_url || t.best_annotated_url || t.vehicle_crop_url) ? `
            <div class="utrack-imgs">
                ${t.vehicle_crop_url ? `<img src="${t.vehicle_crop_url}" title="Vehicle at best frame">` : ""}
                ${t.best_annotated_url ? `<img src="${t.best_annotated_url}" title="Annotated best frame">` : ""}
                ${t.best_crop_url ? `<img src="${t.best_crop_url}" title="Best plate crop">` : ""}
            </div>` : ""}
            <div class="utrack-actions">
                ${t.audit_url ? `<button class="btn btn-ghost" data-audit-url="${t.audit_url}" data-track="${t.track_id}">
                    ${icon("i-doc")}Track audit (${t.n_unique_reads} reads)</button>` : ""}
            </div>
        </div>`).join("");

    $("upload-results-body").innerHTML = `
        ${summaryTiles([
            [d.n_tracks, "Tracks"], [d.n_valid_plates, "Valid plates"],
            [d.fps_processed, "fps processed"], [`${d.elapsed_seconds}s`, "Elapsed"],
        ])}
        <div class="engine-meta">
            ${escapeHtml(d.engine?.detector_coco || "")} · ${escapeHtml(d.engine?.detector_plate || "")} · ${escapeHtml(d.engine?.ocr || "")}<br>
            ${d.n_frames_processed}/${d.n_total_frames} frames @ stride ${d.stride} · source ${d.fps} fps · ${escapeHtml(d.tracker)}
        </div>
        ${videoBlock}
        ${tracks.length ? sectionBlock(`Tracks (${tracks.length})`, items) : ""}
    `;
    $("upload-results-body").querySelectorAll("[data-audit-url]").forEach(btn =>
        btn.addEventListener("click", () => openAuditTrail(btn.dataset.auditUrl, btn.dataset.track)));
}

// ==========================================================================
// TRACK AUDIT MODAL (uploaded videos)
// ==========================================================================
function setupAuditModal() {
    $("audit-modal-close").addEventListener("click", () => closeModal($("audit-modal")));
    $("audit-modal").addEventListener("click", (e) => {
        if (e.target.id === "audit-modal") closeModal($("audit-modal"));
    });
}

async function openAuditTrail(auditUrl, trackHint) {
    openModal($("audit-modal"));
    $("audit-modal-title").innerHTML = `${icon("i-doc")} Track audit ${trackHint ? `#${escapeHtml(trackHint)}` : ""}`;
    $("audit-modal-summary").innerHTML = `<div class="dchip">loading…</div>`;
    $("audit-votes").innerHTML = "";
    $("audit-reads").innerHTML = `<p class="empty">Loading reads…</p>`;
    try {
        const r = await fetch(auditUrl);
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.error || `HTTP ${r.status}`);
        }
        const t = await r.json();
        renderAuditTrail(t);
    } catch (e) {
        $("audit-modal-summary").innerHTML = "";
        $("audit-reads").innerHTML = `<p class="empty">Failed to load audit: ${escapeHtml(e.message)}</p>`;
    }
}

function renderAuditTrail(t) {
    $("audit-modal-title").innerHTML =
        `${icon("i-doc")} Track #${t.track_id} — ${escapeHtml(t.class_name || "vehicle")}`;
    $("audit-modal-sub").textContent =
        `Frames ${t.first_seen} → ${t.last_seen} · ${t.n_frames} frames · ${t.n_unique_reads} unique OCR reads`;

    $("audit-modal-summary").innerHTML = chipRow([
        ["final text", escapeHtml(t.final_text || "—")],
        ["final conf", (t.final_conf || 0).toFixed(3)],
        ["valid indian", t.valid_indian ? "✓ yes" : "✗ no", t.valid_indian ? "ok" : "bad"],
        ["avg yolo", (t.avg_yolo_conf || 0).toFixed(3)],
        ["best frame", t.best_frame ?? "—"],
        ["best text", escapeHtml(t.best_text || "—")],
        ["best conf", (t.best_conf || 0).toFixed(3)],
    ]);

    // character voting visualization — one cell per position
    const votes = t.votes_per_pos || {};
    $("audit-votes").innerHTML = Object.keys(votes).length
        ? Object.entries(votes).map(([pos, bucket]) => {
            const entries = Object.entries(bucket).sort((a, b) => b[1] - a[1]);
            const winner = entries[0];
            return `<div class="vote-pos" title="${entries.slice(0, 4).map(([ch, sc]) => `${ch}:${sc}`).join("  ")}">
                <div class="vote-ch">${escapeHtml(winner ? winner[0] : "?")}</div>
                <div class="vote-score">${winner ? winner[1].toFixed(2) : "—"}</div>
            </div>`;
        }).join("")
        : `<p class="empty">No voting data for this track.</p>`;

    // per-frame reads with crops
    const reads = t.per_frame_reads || [];
    $("audit-reads").innerHTML = reads.length
        ? reads.map(rd => `
            <div class="read-row">
                <span class="read-frame">frame ${rd.frame}</span>
                ${rd.crop_url ? `<img src="${rd.crop_url}" alt="crop @ ${rd.frame}">` : ""}
                <span class="read-text">${rd.text ? escapeHtml(rd.text) : "(empty)"}</span>
                <span class="read-conf">ocr ${(rd.ocr_conf ?? 0).toFixed(2)} · yolo ${(rd.yolo_conf ?? 0).toFixed(2)}</span>
            </div>`).join("")
        : `<p class="empty">This track produced no plate reads.</p>`;
}

// ==========================================================================
// Upload helpers
// ==========================================================================
function summaryTiles(items) {
    return `<div class="upload-summary">${items.map(([num, lbl]) =>
        `<div class="stat"><div class="stat-num">${num}</div><div class="stat-lbl">${lbl}</div></div>`).join("")}</div>`;
}

function sectionBlock(title, inner) {
    return `<div class="section-h">${title}</div><div class="track-list">${inner}</div>`;
}

function showUploadProgress(text) {
    showUploadBody();
    $("upload-results-body").innerHTML =
        `<div class="progress-line"><span class="spinner"></span>${escapeHtml(text)}</div>`;
}

function showUploadError(msg) {
    showUploadBody();
    $("upload-results-body").innerHTML =
        `<div class="progress-line" style="color:var(--bad)">${icon("i-alert")}${escapeHtml(msg)}</div>`;
}

function showUploadBody() {
    $("upload-results-empty").classList.add("hidden");
    $("upload-results-body").classList.remove("hidden");
    $("upload-results-clear").classList.remove("hidden");
}

function resetUploadResults() {
    $("upload-results-empty").classList.remove("hidden");
    $("upload-results-body").classList.add("hidden");
    $("upload-results-body").innerHTML = "";
    $("upload-results-clear").classList.add("hidden");
}

// ==========================================================================
// Global keyboard handling — Esc closes the topmost modal
// ==========================================================================
document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    for (const id of ["audit-modal", "detail-modal", "benchmark-backdrop"]) {
        const el = $(id);
        if (el && !el.classList.contains("hidden")) { closeModal(el); break; }
    }
});
