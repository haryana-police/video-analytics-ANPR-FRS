// ==========================================================================
// TRAFFIC MANAGEMENT SYSTEM — v4 VERIFIABLE PIPELINE FRONTEND
// MJPEG raw frames → <canvas>, SSE detection data → per-stage rendering.
// Every pipeline step (Raw / Objects / Plates / OCR / Tracking) is rendered
// independently on a clickable canvas. Boxes are hit-tested → detail modal.
// ==========================================================================

const $ = (id) => document.getElementById(id);

// --- Global state ---
const state = {
    sessionId: null,
    mjpegUrl: null,
    eventsUrl: null,
    stopUrl: null,
    eventSource: null,
    busy: false,
    // Live track snapshots keyed by track_id per category
    vehicles: new Map(),
    plates: new Map(),
    persons: new Map(),
    // v4: raw per-frame detections from SSE (for canvas rendering)
    cocoDets: [],      // current frame COCO detections
    plateDets: [],     // current frame plate detections
    personDets: [],    // current frame person detections
    trajectories: {},  // { track_id: [[frame_idx, cx, cy], ...] } accumulated client-side
    timing: null,      // {coco_ms, plate_ms, ocr_ms, total_ms, ocr_fired}
    frameSize: [0, 0],
    // v4: active pipeline stage
    activeStage: "raw",
    // v4: hidden <img> to receive MJPEG stream, drawn to canvas
    mjpegImg: null,
    // v4: canvas + ctx
    canvas: null,
    ctx: null,
    // v4: hovered box for highlight
    hoveredBox: null,
    // Benchmark data
    benchmark: null,
};

const VEHICLE_EMOJI = {
    bicycle: "🚲",
    car: "🚗",
    motorcycle: "🏍️",
    bus: "🚌",
    truck: "🚚",
};

// Colors for each pipeline stage (canvas drawing)
const STAGE_COLORS = {
    objects: { vehicle: "#ff8c00", person: "#00a5ff", text: "#ff8c00" },
    plates: { box: "#00ff00", boxInvalid: "#ff0000", text: "#00ff00" },
    ocr: { box: "#00ff00", textOk: "#00ff00", textBad: "#ff4444", text: "#ffff00" },
    tracking: { line: "#00a5ff", dot: "#ff8c00", text: "#00a5ff" },
};

function log(...args) {
    const line = args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" ");
    const el = $("console-log");
    if (el) {
        const t = new Date().toLocaleTimeString();
        el.innerHTML += `<div><span class="ts">${t}</span> ${escapeHtml(line)}</div>`;
        el.scrollTop = el.scrollHeight;
    }
    console.log("[live]", ...args);
}

function escapeHtml(s) {
    return (s ?? "").toString().replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
}

// ==========================================================================
// Init
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    setupHealth();
    setupSourcePicker();
    setupControlDeck();
    setupStageTabs();
    setupCanvas();
    setupResultsTabs();
    setupDetailModal();
    setupBenchmark();

    log("Dashboard v4 loaded. Pick a source and click ▶ Start Live.");
    refreshSources();
    checkHealth();
    setInterval(checkHealth, 5000);
});

// ==========================================================================
// Health
// ==========================================================================
async function checkHealth() {
    try {
        const r = await fetch("/health/full");
        const d = await r.json();
        $("health-dot").classList.add("ready");
        $("health-dot").classList.remove("error");
        $("health-text").textContent = `Pipeline ready · ${d.cached_models?.length ?? 0} models cached`;
        $("footer-status").textContent = "ONLINE";
    } catch (e) {
        $("health-dot").classList.add("error");
        $("health-dot").classList.remove("ready");
        $("health-text").textContent = "Pipeline offline";
        $("footer-status").textContent = "OFFLINE";
    }
}

function setupHealth() { }

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
                ? `🎥 ${s.name}`
                : `📁 ${s.name} (${s.size_mb} MB)`;
            sel.appendChild(opt);
        }
        log(`Loaded ${d.sources.length} sources from ${d.sample_videos_dir}`);
    } catch (e) {
        log("refreshSources failed:", e.message);
    }
}

function setupSourcePicker() {
    $("btn-refresh-sources").addEventListener("click", refreshSources);
}

// ==========================================================================
// Control deck (start / stop)
// ==========================================================================
function setupControlDeck() {
    $("btn-start").addEventListener("click", startLive);
    $("btn-stop").addEventListener("click", stopLive);
}

async function startLive() {
    if (state.busy) return;
    state.busy = true;
    $("btn-start").disabled = true;

    let source = $("source-url").value.trim();
    if (!source) source = $("source-select").value;
    if (!source) {
        log("ERROR: pick a source or paste a URL");
        state.busy = false;
        $("btn-start").disabled = false;
        return;
    }

    const body = {
        source: source,
        target_fps: parseInt($("target-fps").value || "15"),
        model_coco: $("model-coco").value,
        model_plate: $("model-plate").value,
        ocr_on_best_only: $("ocr-strategy").value === "best",
        loop: $("loop-toggle").checked,
    };

    log(`Starting live session: source=${source} models=(${body.model_coco}, ${body.model_plate})`);

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
        $("meta-session").textContent = `session ${state.sessionId.slice(0, 6)}…`;
        $("btn-stop").disabled = false;

        // v4: hide empty state, set up MJPEG → hidden img → canvas
        $("viewport-overlay-empty").classList.add("hidden");
        if (!state.mjpegImg) {
            state.mjpegImg = new Image();
            state.mjpegImg.onload = () => {
                // When MJPEG frame loads, draw it to canvas with active stage annotations
                drawCanvas();
            };
        }
        state.mjpegImg.src = state.mjpegUrl + "?t=" + Date.now();

        // Connect SSE
        openEventSource();

        log(`Live session ${state.sessionId} started — MJPEG + SSE connected`);
    } catch (e) {
        log("startLive failed:", e.message);
    } finally {
        state.busy = false;
        $("btn-start").disabled = false;
    }
}

async function stopLive() {
    if (!state.sessionId) return;
    log("Stopping live session…");
    try {
        if (state.eventSource) {
            state.eventSource.close();
            state.eventSource = null;
        }
        if (state.mjpegImg) {
            state.mjpegImg.src = "";
        }
        $("viewport-overlay-empty").classList.remove("hidden");

        await fetch(state.stopUrl, { method: "POST" });
        log("Live session stopped");
    } catch (e) {
        log("stopLive failed:", e.message);
    } finally {
        state.sessionId = null;
        state.mjpegUrl = null;
        state.eventsUrl = null;
        state.cocoDets = [];
        state.plateDets = [];
        state.personDets = [];
        state.trajectories = {};
        state.timing = null;
        $("meta-session").textContent = "no session";
        $("btn-stop").disabled = true;
        // Clear canvas
        if (state.ctx) {
            state.ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
        }
    }
}

// ==========================================================================
// SSE — receive detection events
// ==========================================================================
function openEventSource() {
    if (!state.eventsUrl) return;
    if (state.eventSource) state.eventSource.close();
    const es = new EventSource(state.eventsUrl);
    state.eventSource = es;

    es.addEventListener("frame", (e) => {
        try {
            const data = JSON.parse(e.data);
            handleFrameEvent(data);
        } catch (err) {
            log("bad frame event:", err.message);
        }
    });
    es.addEventListener("end", () => {
        log("Server signaled end of stream");
        stopLive();
    });
    es.onerror = () => {
        log("SSE connection error");
    };
}

function handleFrameEvent(d) {
    // Update meta chips
    if (typeof d.fps === "number") {
        $("meta-fps").textContent = `fps ${d.fps.toFixed(1)}`;
    }
    $("meta-frame").textContent = `frame ${d.frame_index}`;

    // v4: frame size
    if (d.frame_size) {
        state.frameSize = d.frame_size;
        $("meta-size").textContent = `${d.frame_size[0]} × ${d.frame_size[1]}`;
    }

    // v4: timing bar
    if (d.timing) {
        state.timing = d.timing;
        $("timing-coco").textContent = `${d.timing.coco_ms} ms`;
        $("timing-plate").textContent = `${d.timing.plate_ms} ms`;
        $("timing-ocr").textContent = d.timing.ocr_ms > 0 ? `${d.timing.ocr_ms} ms` : "—";
        $("timing-total").textContent = `${d.timing.total_ms} ms`;
        $("timing-ocr-fired").textContent = d.timing.ocr_fired ? "✓ yes" : "—";
    }

    // v4: raw per-frame detections for canvas rendering
    state.cocoDets = d.coco_detections || [];
    state.plateDets = d.plate_detections || [];
    state.personDets = d.person_detections || [];

    // v4: accumulate trajectory points client-side.
    // Server sends only the last 3 points per track to keep payload small;
    // we merge them into our client-side trajectory history.
    const incomingTraj = d.trajectories || {};
    for (const [tid, points] of Object.entries(incomingTraj)) {
        if (!state.trajectories[tid]) state.trajectories[tid] = [];
        const existing = state.trajectories[tid];
        const lastFrame = existing.length > 0 ? existing[existing.length - 1][0] : 0;
        for (const pt of points) {
            if (pt[0] > lastFrame) existing.push(pt);
        }
        // Cap at 40 points per track to avoid unbounded growth
        if (existing.length > 40) {
            state.trajectories[tid] = existing.slice(-40);
        }
    }
    // Cap total number of tracked trajectory keys to avoid memory bloat
    const trajKeys = Object.keys(state.trajectories);
    if (trajKeys.length > 100) {
        // Delete oldest half
        for (const k of trajKeys.slice(0, 50)) {
            delete state.trajectories[k];
        }
    }

    // Replace track snapshots (for card lists)
    state.vehicles.clear();
    for (const v of d.vehicles || []) state.vehicles.set(v.track_id, v);
    state.plates.clear();
    for (const p of d.plates || []) state.plates.set(p.track_id, p);
    state.persons.clear();
    for (const p of d.persons || []) state.persons.set(p.track_id, p);

    // Update summary
    const validPlates = [...state.plates.values()].filter(p => p.valid).length;
    $("sum-vehicles").textContent = state.vehicles.size;
    $("sum-valid-plates").textContent = validPlates;
    $("sum-plates").textContent = state.plates.size;
    $("sum-persons").textContent = state.persons.size;
    $("tab-count-vehicles").textContent = state.vehicles.size;
    $("tab-count-plates").textContent = state.plates.size;
    $("tab-count-persons").textContent = state.persons.size;

    // Re-render card lists
    renderVehicleCards();
    renderPlateCards();
    renderPersonCards();

    // v4: draw canvas with active stage (MJPEG img.onload also triggers this,
    // but SSE data may arrive before/after the frame — redraw on data update too)
    drawCanvas();
}

// ==========================================================================
// v4: Canvas setup + drawing
// ==========================================================================
function setupCanvas() {
    state.canvas = $("live-canvas");
    state.ctx = state.canvas.getContext("2d");

    // Click handler — hit-test boxes and open detail modal
    state.canvas.addEventListener("click", (e) => {
        const rect = state.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (state.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (state.canvas.height / rect.height);
        const hit = hitTest(x, y);
        if (hit) {
            if (hit.kind === "vehicle") openVehicleDetail(hit.track_id);
            else if (hit.kind === "plate") openPlateDetail(hit.track_id);
            else if (hit.kind === "person") openPersonDetail(hit.track_id);
        }
    });

    // Mouse move — hover highlight
    state.canvas.addEventListener("mousemove", (e) => {
        const rect = state.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (state.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (state.canvas.height / rect.height);
        const hit = hitTest(x, y);
        state.canvas.style.cursor = hit ? "pointer" : "default";
        if (hit !== state.hoveredBox) {
            state.hoveredBox = hit;
            drawCanvas();
        }
    });

    state.canvas.addEventListener("mouseleave", () => {
        state.hoveredBox = null;
        state.canvas.style.cursor = "default";
        drawCanvas();
    });
}

function drawCanvas() {
    if (!state.ctx || !state.mjpegImg) return;

    const img = state.mjpegImg;
    if (!img.naturalWidth || !img.naturalHeight) return;

    // Set canvas size to match image (first frame or on resize)
    if (state.canvas.width !== img.naturalWidth) {
        state.canvas.width = img.naturalWidth;
        state.canvas.height = img.naturalHeight;
    }

    const ctx = state.ctx;
    const W = state.canvas.width;
    const H = state.canvas.height;

    // Always draw the raw frame first
    ctx.drawImage(img, 0, 0, W, H);

    const stage = state.activeStage;

    // Render the active stage's annotations
    if (stage === "raw") {
        // No annotations — just the raw frame
        return;
    }

    if (stage === "objects" || stage === "all") {
        drawObjectsStage(ctx, W, H);
    }

    if (stage === "plates" || stage === "all") {
        drawPlatesStage(ctx, W, H);
    }

    if (stage === "ocr" || stage === "all") {
        drawOcrStage(ctx, W, H);
    }

    if (stage === "tracking" || stage === "all") {
        drawTrackingStage(ctx, W, H);
    }

    // Draw FPS overlay (top-left, always on when streaming)
    if (state.timing) {
        drawFpsOverlay(ctx, W, H);
    }
}

function drawObjectsStage(ctx, W, H) {
    const colors = STAGE_COLORS.objects;
    for (const d of state.cocoDets) {
        const [x1, y1, x2, y2] = d.bbox_xyxy;
        const isPerson = d.class_name === "person";
        const color = isPerson ? colors.person : colors.vehicle;
        const isHovered = state.hoveredBox && state.hoveredBox.track_id === d.track_id && state.hoveredBox.kind === (isPerson ? "person" : "vehicle");

        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 4 : 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label background
        const label = `${d.class_name} #${d.track_id} (${(d.confidence || 0).toFixed(2)})`;
        ctx.font = "14px 'JetBrains Mono', monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(x1, Math.max(y1 - 20, 0), tw + 8, 18);
        ctx.fillStyle = color;
        ctx.fillText(label, x1 + 4, Math.max(y1 - 6, 12));
    }
}

function drawPlatesStage(ctx, W, H) {
    const colors = STAGE_COLORS.plates;
    for (const p of state.plateDets) {
        const [x1, y1, x2, y2] = p.bbox_xyxy;
        const valid = p.valid;
        const color = valid ? colors.box : colors.boxInvalid;
        const isHovered = state.hoveredBox && state.hoveredBox.track_id === p.track_id && state.hoveredBox.kind === "plate";

        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 5 : 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Label: plate track id + yolo conf
        const label = `Plate #${p.track_id} (${(p.confidence || 0).toFixed(2)})`;
        ctx.font = "13px 'JetBrains Mono', monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(x1, Math.max(y1 - 18, 0), tw + 8, 16);
        ctx.fillStyle = color;
        ctx.fillText(label, x1 + 4, Math.max(y1 - 5, 11));

        // Draw best-bbox (dashed) if different from current
        if (p.best_frame_count && p.ocr_done) {
            // indicate OCR was done at best frame
        }
    }
}

function drawOcrStage(ctx, W, H) {
    const colors = STAGE_COLORS.ocr;
    for (const p of state.plateDets) {
        const [x1, y1, x2, y2] = p.bbox_xyxy;
        const hasText = p.text && p.text.length > 0;
        const valid = p.valid;
        const color = valid ? colors.textOk : (hasText ? colors.textBad : colors.box);
        const isHovered = state.hoveredBox && state.hoveredBox.track_id === p.track_id && state.hoveredBox.kind === "plate";

        // Draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = isHovered ? 5 : 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // OCR text label below the box
        const ocrText = hasText ? p.text : (p.ocr_done ? "(empty)" : "OCR pending…");
        const confText = p.ocr_done ? ` ${(p.ocr_conf || 0).toFixed(2)}` : "";
        const label = `${ocrText}${confText}`;
        ctx.font = "bold 16px 'JetBrains Mono', monospace";
        const tw = ctx.measureText(label).width;
        const ly = Math.min(y2 + 20, H - 4);
        ctx.fillStyle = "rgba(0,0,0,0.8)";
        ctx.fillRect(Math.max(x1, 0), ly - 16, tw + 10, 20);
        ctx.fillStyle = color;
        ctx.fillText(label, Math.max(x1 + 4, 2), ly - 2);

        // Valid badge
        if (valid) {
            ctx.fillStyle = "#00ff00";
            ctx.fillText("✓", Math.max(x1 + tw + 14, 0), ly - 2);
        }
    }
}

function drawTrackingStage(ctx, W, H) {
    const colors = STAGE_COLORS.tracking;

    // Draw trajectory lines for tracks — limit to last 30 active tracks
    // to avoid drawing 700+ polylines every frame (perf)
    const trajEntries = Object.entries(state.trajectories).slice(-30);

    for (const [tid, points] of trajEntries) {
        if (!points || points.length < 2) continue;
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
            const [, cx, cy] = points[i];
            if (i === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
        }
        ctx.stroke();

        // Draw a dot at the last position only
        const last = points[points.length - 1];
        if (last) {
            const [, cx, cy] = last;
            ctx.fillStyle = colors.dot;
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, Math.PI * 2);
            ctx.fill();

            // Label with track ID
            ctx.font = "11px 'JetBrains Mono', monospace";
            ctx.fillStyle = "rgba(0,0,0,0.7)";
            ctx.fillRect(cx + 4, cy - 12, 36, 14);
            ctx.fillStyle = colors.text;
            ctx.fillText(`#${tid}`, cx + 6, cy - 1);
        }
    }

    // Also draw current bboxes with track IDs (dashed)
    for (const d of state.cocoDets) {
        const [x1, y1, x2, y2] = d.bbox_xyxy;
        const isPerson = d.class_name === "person";
        ctx.strokeStyle = isPerson ? STAGE_COLORS.objects.person : STAGE_COLORS.objects.vehicle;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);

        // Track ID label
        ctx.font = "12px 'JetBrains Mono', monospace";
        const label = `#${d.track_id}`;
        ctx.fillStyle = "rgba(0,0,0,0.7)";
        ctx.fillRect(x1, y1, 36, 16);
        ctx.fillStyle = colors.text;
        ctx.fillText(label, x1 + 3, y1 + 12);
    }
}

function drawFpsOverlay(ctx, W, H) {
    const t = state.timing;
    if (!t) return;
    const fpsText = $("meta-fps").textContent;
    const frameText = $("meta-frame").textContent;
    const label = `${frameText}  ${fpsText}  total=${t.total_ms}ms`;
    ctx.font = "16px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.fillRect(8, 6, ctx.measureText(label).width + 12, 22);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, 14, 22);
}

// --- Hit testing for clickable canvas ---
function hitTest(x, y) {
    // Check plates first (drawn on top), then vehicles, then persons
    for (const p of state.plateDets) {
        const [x1, y1, x2, y2] = p.bbox_xyxy;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
            return { kind: "plate", track_id: p.track_id };
        }
    }
    for (const d of state.cocoDets) {
        const [x1, y1, x2, y2] = d.bbox_xyxy;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
            return { kind: d.class_name === "person" ? "person" : "vehicle", track_id: d.track_id };
        }
    }
    return null;
}

// ==========================================================================
// v4: Stage tabs — switch active pipeline stage
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
            log(`Pipeline stage → ${stage}`);
            drawCanvas();
        });
    });
}

// ==========================================================================
// Card rendering — Vehicles, Plates, Persons
// ==========================================================================
function renderVehicleCards() {
    const root = $("vehicles-list");
    if (state.vehicles.size === 0) {
        root.innerHTML = `<p class="empty">No vehicles tracked yet.</p>`;
        return;
    }
    const sorted = [...state.vehicles.values()].sort((a, b) => {
        if (!!a.plate_valid !== !!b.plate_valid) return a.plate_valid ? -1 : 1;
        return (b.n_frames || 0) - (a.n_frames || 0);
    });
    root.innerHTML = sorted.map(v => {
        const emoji = VEHICLE_EMOJI[v.class_name] || "🚙";
        const plateHtml = v.plate_text
            ? `<div class="plate-readout ${v.plate_valid ? "valid" : "tentative"}">
                 <span class="plate-text">${escapeHtml(v.plate_text)}</span>
                 <span class="plate-conf">conf ${(v.plate_conf || 0).toFixed(2)}</span>
                 ${v.plate_valid ? '<span class="valid-badge">✓ Indian format</span>' : ""}
               </div>`
            : `<div class="plate-readout pending">
                 <span class="plate-text muted">${v.ocr_done ? "(no OCR read)" : "OCR pending…"}</span>
               </div>`;
        const bbox = v.bbox_xyxy ? `(${v.bbox_xyxy.join(", ")})` : "";
        return `<div class="card vehicle-card" data-tid="${v.track_id}" data-kind="vehicle">
            <div class="card-head">
                <span class="emoji">${emoji}</span>
                <span class="card-title">${escapeHtml(v.class_name)} #${v.track_id}</span>
                <span class="card-conf">yolo ${(v.confidence || 0).toFixed(2)}</span>
            </div>
            ${plateHtml}
            <div class="card-meta">
                <span>${v.n_frames} frames</span>
                <span>bbox ${bbox}</span>
            </div>
        </div>`;
    }).join("");
    root.querySelectorAll(".vehicle-card").forEach(el => {
        el.addEventListener("click", () => {
            const tid = parseInt(el.dataset.tid);
            openVehicleDetail(tid);
        });
    });
}

function renderPlateCards() {
    const root = $("plates-list");
    if (state.plates.size === 0) {
        root.innerHTML = `<p class="empty">No plates tracked yet.</p>`;
        return;
    }
    const sorted = [...state.plates.values()].sort((a, b) => {
        if (!!a.valid !== !!b.valid) return a.valid ? -1 : 1;
        return (b.ocr_conf || 0) - (a.ocr_conf || 0);
    });
    root.innerHTML = sorted.map(p => {
        const linked = p.linked_vehicle_id != null
            ? `<span class="link-badge" title="Associated with vehicle #${p.linked_vehicle_id}">↔ vehicle #${p.linked_vehicle_id}</span>`
            : "";
        const valid = p.valid ? "valid" : "tentative";
        return `<div class="card plate-card" data-tid="${p.track_id}" data-kind="plate">
            <div class="card-head">
                <span class="emoji">🔢</span>
                <span class="card-title">Plate #${p.track_id}</span>
                <span class="card-conf">yolo ${(p.confidence || 0).toFixed(2)}</span>
            </div>
            <div class="plate-readout ${valid}">
                <span class="plate-text">${p.text ? escapeHtml(p.text) : (p.ocr_done ? "(empty)" : "OCR pending…")}</span>
                ${p.ocr_done ? `<span class="plate-conf">conf ${(p.ocr_conf || 0).toFixed(2)}</span>` : ""}
                ${p.valid ? '<span class="valid-badge">✓ Indian format</span>' : ""}
            </div>
            <div class="card-meta">
                <span>${p.n_frames} frames</span>
                ${linked}
            </div>
        </div>`;
    }).join("");
    root.querySelectorAll(".plate-card").forEach(el => {
        el.addEventListener("click", () => {
            const tid = parseInt(el.dataset.tid);
            openPlateDetail(tid);
        });
    });
}

function renderPersonCards() {
    const root = $("persons-list");
    if (state.persons.size === 0) {
        root.innerHTML = `<p class="empty">No persons tracked yet.</p>`;
        return;
    }
    const sorted = [...state.persons.values()].sort((a, b) => (b.n_frames || 0) - (a.n_frames || 0));
    root.innerHTML = sorted.map(p => {
        return `<div class="card person-card" data-tid="${p.track_id}" data-kind="person">
            <div class="card-head">
                <span class="emoji">🚶</span>
                <span class="card-title">person #${p.track_id}</span>
                <span class="card-conf">yolo ${(p.confidence || 0).toFixed(2)}</span>
            </div>
            <div class="card-meta">
                <span>${p.n_frames} frames</span>
                <span>bbox (${(p.bbox_xyxy || []).join(", ")})</span>
            </div>
        </div>`;
    }).join("");
    root.querySelectorAll(".person-card").forEach(el => {
        el.addEventListener("click", () => {
            const tid = parseInt(el.dataset.tid);
            openPersonDetail(tid);
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
            document.querySelectorAll(".results-content").forEach(c => {
                c.classList.toggle("active", c.dataset.tab === tab);
            });
        });
    });
}

// ==========================================================================
// Detail modal — opened from card click or canvas click
// ==========================================================================
function setupDetailModal() {
    $("detail-modal-close").addEventListener("click", closeDetailModal);
    $("detail-modal").addEventListener("click", (e) => {
        if (e.target.id === "detail-modal") closeDetailModal();
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !$("detail-modal").classList.contains("hidden")) {
            closeDetailModal();
        }
    });
}

function openDetailModal(title, sub) {
    $("detail-modal-title").textContent = title;
    $("detail-modal-sub").textContent = sub || "";
    $("detail-modal").classList.remove("hidden");
    $("detail-modal").setAttribute("aria-hidden", "false");
    // Reset crop area
    $("detail-best-crop").innerHTML = `<p class="empty">Loading crop…</p>`;
    $("detail-best-crop-label").textContent = "Best crop (OCR'd at best-bbox frame)";
}

function closeDetailModal() {
    $("detail-modal").classList.add("hidden");
    $("detail-modal").setAttribute("aria-hidden", "true");
}

function openVehicleDetail(tid) {
    const v = state.vehicles.get(tid);
    if (!v) return;
    openDetailModal(`${VEHICLE_EMOJI[v.class_name] || "🚙"} ${v.class_name} #${tid}`,
                    `Tracked across ${v.n_frames} frames`);
    const linkedPlate = v.plate_track_id != null ? state.plates.get(v.plate_track_id) : null;
    $("detail-modal-summary").innerHTML = `
        <div class="detail-chip">track_id: <b>${tid}</b></div>
        <div class="detail-chip">class: <b>${escapeHtml(v.class_name)}</b></div>
        <div class="detail-chip">yolo conf: <b>${(v.confidence || 0).toFixed(3)}</b></div>
        <div class="detail-chip">frames seen: <b>${v.n_frames}</b></div>
        <div class="detail-chip">plate track: <b>${v.plate_track_id ?? "(none)"}</b></div>
        <div class="detail-chip">plate text: <b>${escapeHtml(v.plate_text || "—")}</b></div>
        <div class="detail-chip">plate valid: <b>${v.plate_valid ? "✓ yes" : "✗ no"}</b></div>
    `;
    // v4: trajectory timeline
    const traj = state.trajectories[tid] || [];
    $("detail-timeline").innerHTML = `
        <p class="hint">${traj.length} trajectory points recorded.</p>
        <p class="hint">OCR runs once per plate track at the best-crop frame (largest bbox area). For full frame-by-frame audit, use <b>Upload → Video</b>.</p>
    `;
    $("detail-last-seen").innerHTML = `
        <div class="kv-row"><span>Last bbox (xyxy):</span> <code>${(v.bbox_xyxy || []).join(", ")}</code></div>
        <div class="kv-row"><span>Best bbox (xyxy):</span> <code>${(v.best_bbox_xyxy || []).join(", ")}</code></div>
        <div class="kv-row"><span>Best frame index:</span> <code>${v.best_frame_count ?? "—"}</code></div>
        <div class="kv-row"><span>OCR done:</span> <code>${v.ocr_done ? "yes" : "no"}</code></div>
        ${linkedPlate ? `<div class="kv-row"><span>Linked plate text:</span> <code>${escapeHtml(linkedPlate.text || "—")}</code></div>` : ""}
    `;
    $("detail-meta").textContent = JSON.stringify(v, null, 2);

    // v4: fetch best crop if this vehicle has a linked plate track
    if (v.plate_track_id != null && state.sessionId) {
        fetchBestCrop(v.plate_track_id);
    } else {
        $("detail-best-crop").innerHTML = `<p class="empty">No plate track linked — no crop available.</p>`;
    }
}

function openPlateDetail(tid) {
    const p = state.plates.get(tid);
    if (!p) return;
    openDetailModal(`🔢 Plate #${tid}`,
                    `OCR ${p.ocr_done ? "complete" : "pending"} · ${p.n_frames} frames`);
    $("detail-modal-summary").innerHTML = `
        <div class="detail-chip">track_id: <b>${tid}</b></div>
        <div class="detail-chip">yolo conf: <b>${(p.confidence || 0).toFixed(3)}</b></div>
        <div class="detail-chip">ocr conf: <b>${(p.ocr_conf || 0).toFixed(3)}</b></div>
        <div class="detail-chip">text: <b>${escapeHtml(p.text || "—")}</b></div>
        <div class="detail-chip">valid Indian: <b>${p.valid ? "✓ yes" : "✗ no"}</b></div>
        <div class="detail-chip">linked vehicle: <b>${p.linked_vehicle_id ?? "(none)"}</b></div>
        <div class="detail-chip">frames seen: <b>${p.n_frames}</b></div>
    `;
    $("detail-timeline").innerHTML = `
        <p class="hint">OCR runs once per plate track, at the best-crop frame
        (largest bbox area for that track). For frame-by-frame OCR audit +
        persistent best crops, run the same source through <b>Upload → Video</b>.</p>
    `;
    $("detail-last-seen").innerHTML = `
        <div class="kv-row"><span>Last bbox (xyxy):</span> <code>${(p.bbox_xyxy || []).join(", ")}</code></div>
        <div class="kv-row"><span>Best bbox (xyxy):</span> <code>${(p.best_bbox_xyxy || []).join(", ")}</code></div>
        <div class="kv-row"><span>Best frame index:</span> <code>${p.best_frame_count ?? "—"}</code></div>
        <div class="kv-row"><span>OCR done at:</span> <code>${p.ocr_done ? "best frame" : "—"}</code></div>
    `;
    $("detail-meta").textContent = JSON.stringify(p, null, 2);

    // v4: fetch best crop from backend
    if (state.sessionId) {
        fetchBestCrop(tid);
    } else {
        $("detail-best-crop").innerHTML = `<p class="empty">No active session — crop not available.</p>`;
    }
}

function openPersonDetail(tid) {
    const p = state.persons.get(tid);
    if (!p) return;
    openDetailModal(`🚶 person #${tid}`, `Tracked across ${p.n_frames} frames`);
    $("detail-modal-summary").innerHTML = `
        <div class="detail-chip">track_id: <b>${tid}</b></div>
        <div class="detail-chip">yolo conf: <b>${(p.confidence || 0).toFixed(3)}</b></div>
        <div class="detail-chip">frames seen: <b>${p.n_frames}</b></div>
    `;
    $("detail-timeline").innerHTML = `<p class="hint">Persons have no OCR pipeline — only COCO detection + tracking.</p>`;
    $("detail-last-seen").innerHTML = `
        <div class="kv-row"><span>Last bbox (xyxy):</span> <code>${(p.bbox_xyxy || []).join(", ")}</code></div>
        <div class="kv-row"><span>Best bbox (xyxy):</span> <code>${(p.best_bbox_xyxy || []).join(", ")}</code></div>
    `;
    $("detail-meta").textContent = JSON.stringify(p, null, 2);
    $("detail-best-crop").innerHTML = `<p class="empty">Persons have no plate crop.</p>`;
}

// v4: Fetch best crop JPEG from backend and display in modal
async function fetchBestCrop(plateTrackId) {
    if (!state.sessionId) return;
    const url = `/api/live/crop/${state.sessionId}/${plateTrackId}`;
    try {
        const r = await fetch(url);
        if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            $("detail-best-crop").innerHTML = `<p class="empty">${escapeHtml(err.error || "Crop not yet captured.")}</p>`;
            return;
        }
        const blob = await r.blob();
        const objUrl = URL.createObjectURL(blob);
        $("detail-best-crop").innerHTML = `<img src="${objUrl}" alt="Best plate crop" style="max-width:100%;border-radius:6px;">`;
    } catch (e) {
        $("detail-best-crop").innerHTML = `<p class="empty">Error loading crop: ${escapeHtml(e.message)}</p>`;
    }
}

// ==========================================================================
// Benchmark
// ==========================================================================
function setupBenchmark() {
    $("btn-benchmark").addEventListener("click", () => {
        $("benchmark-panel").classList.remove("hidden");
    });
    $("btn-benchmark-close").addEventListener("click", () => {
        $("benchmark-panel").classList.add("hidden");
    });
    $("btn-benchmark-run").addEventListener("click", runBenchmark);
}

async function runBenchmark() {
    $("btn-benchmark-run").disabled = true;
    $("btn-benchmark-run").textContent = "Running…";
    log("Benchmark starting (5 measurements × ~warmup × 9 variants)…");
    try {
        const r = await fetch("/api/live/benchmark");
        const d = await r.json();
        renderBenchmark(d.results || []);
        log(`Benchmark done — ${d.results.length} variants measured`);
    } catch (e) {
        log("Benchmark failed:", e.message);
    } finally {
        $("btn-benchmark-run").disabled = false;
        $("btn-benchmark-run").textContent = "▶ Run benchmark";
    }
}

function renderBenchmark(results) {
    const tbody = $("benchmark-tbody");
    if (!results.length) {
        tbody.innerHTML = `<tr><td colspan="8" class="empty">No models found.</td></tr>`;
        return;
    }
    tbody.innerHTML = results.map(r => {
        if (r.error) {
            return `<tr><td>${r.name}</td><td>${r.kind}</td><td colspan="6" class="err">${escapeHtml(r.error)}</td></tr>`;
        }
        // v4: verdict column — is this fast enough for live?
        let verdict = "";
        if (r.kind === "coco" || r.kind === "plate") {
            const fps = r.max_fps || 0;
            if (fps >= 10) verdict = `<span class="verdict-good">✓ live-ready</span>`;
            else if (fps >= 4) verdict = `<span class="verdict-ok">~ borderline</span>`;
            else verdict = `<span class="verdict-bad">✗ too slow</span>`;
        } else if (r.kind === "ocr") {
            const fps = r.max_fps || 0;
            if (fps >= 5) verdict = `<span class="verdict-good">✓ fast</span>`;
            else if (fps >= 1) verdict = `<span class="verdict-ok">~ best-crop only</span>`;
            else verdict = `<span class="verdict-bad">✗ bottleneck</span>`;
        } else if (r.kind === "tracker") {
            verdict = `<span class="verdict-good">✓ negligible</span>`;
        } else if (r.kind === "frame") {
            return `<tr><td><b>${r.name}</b></td><td>${r.kind}</td><td colspan="6" class="hint">${escapeHtml(r.source || "")}</td></tr>`;
        }
        return `<tr>
            <td><b>${r.name}</b></td>
            <td>${r.kind}</td>
            <td>${r.min_ms ?? "—"} ms</td>
            <td>${r.median_ms ?? "—"} ms</td>
            <td>${r.mean_ms ?? "—"} ms</td>
            <td>${r.max_ms ?? "—"} ms</td>
            <td>${r.max_fps ?? "—"} fps</td>
            <td>${verdict}</td>
        </tr>`;
    }).join("");
}

// ==========================================================================
// MODE TOGGLE — Live vs Upload
// ==========================================================================
function setMode(mode) {
    document.querySelectorAll(".mode-btn").forEach(b => {
        b.classList.toggle("active", b.dataset.mode === mode);
    });
    if (mode === "live") {
        $("live-panel").classList.remove("hidden");
        $("upload-panel").classList.add("hidden");
    } else {
        $("live-panel").classList.add("hidden");
        $("upload-panel").classList.remove("hidden");
        $("upload-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    }
    log(`Mode → ${mode}`);
}

function setupModeToggle() {
    document.querySelectorAll(".mode-btn").forEach(btn => {
        btn.addEventListener("click", () => setMode(btn.dataset.mode));
    });
}

// ==========================================================================
// UPLOAD TABS (Image / Video)
// ==========================================================================
function setupUploadTabs() {
    document.querySelectorAll(".upload-tab").forEach(btn => {
        btn.addEventListener("click", () => {
            const tab = btn.dataset.uploadTab;
            document.querySelectorAll(".upload-tab").forEach(b => {
                b.classList.toggle("active", b.dataset.uploadTab === tab);
            });
            document.querySelectorAll("[data-upload-tab-pane]").forEach(pane => {
                pane.classList.toggle("hidden", pane.dataset.uploadTabPane !== tab);
            });
            $("upload-results-title").textContent =
                tab === "image" ? "Image results" : "Video results";
        });
    });
}

// ==========================================================================
// IMAGE UPLOAD
// ==========================================================================
let uploadImageFile = null;

function setupImageUpload() {
    const dz = $("image-dropzone");
    const input = $("image-file-input");
    const preview = $("image-preview");
    const detectBtn = $("image-detect-btn");
    const clearBtn = $("image-clear-btn");
    const changeBtn = $("image-change-btn");
    const content = $("image-dropzone-content");

    function pickFile() { input.click(); }
    dz.addEventListener("click", pickFile);
    changeBtn.addEventListener("click", (e) => { e.stopPropagation(); pickFile(); });

    // Drag-and-drop
    ["dragenter", "dragover"].forEach(ev => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add("drag-active");
        });
    });
    ["dragleave", "dragend"].forEach(ev => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.target === dz) dz.classList.remove("drag-active");
        });
    });
    dz.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.remove("drag-active");
        const f = e.dataTransfer?.files?.[0];
        if (!f) return;
        if (!f.type.startsWith("image/")) {
            log(`Rejected dropped file (not an image): ${f.name}`);
            return;
        }
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
    });

    input.addEventListener("change", () => {
        const f = input.files[0];
        if (!f) return;
        uploadImageFile = f;
        const url = URL.createObjectURL(f);
        preview.src = url;
        preview.classList.remove("hidden");
        changeBtn.classList.remove("hidden");
        content.classList.add("hidden");
        detectBtn.disabled = false;
        clearBtn.disabled = false;
        log(`Image selected: ${f.name} (${(f.size/1024).toFixed(0)} KB)`);
    });

    clearBtn.addEventListener("click", () => {
        uploadImageFile = null;
        input.value = "";
        preview.src = "";
        preview.classList.add("hidden");
        changeBtn.classList.add("hidden");
        content.classList.remove("hidden");
        detectBtn.disabled = true;
        clearBtn.disabled = true;
        resetUploadResults();
    });

    detectBtn.addEventListener("click", runImageDetect);
}

async function runImageDetect() {
    if (!uploadImageFile) return;
    const detectBtn = $("image-detect-btn");
    detectBtn.disabled = true;
    detectBtn.textContent = "Running…";
    log(`POST /api/detect (${uploadImageFile.name})`);
    showUploadProgress("Analyzing image…");
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
        log(`Image analysis done — ${d.num_plates} plates, ${d.num_vehicles} vehicles, ${d.num_persons} persons in ${d.elapsed_seconds}s`);
        renderImageResults(d);
    } catch (e) {
        log("Image analysis failed:", e.message);
        showUploadResultsError(e.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = '<span class="btn-icon">🔍</span><span class="btn-label">Run image analysis</span>';
    }
}

function renderImageResults(d) {
    $("upload-results-empty").classList.add("hidden");
    $("upload-results-body").classList.remove("hidden");
    $("upload-results-clear").classList.remove("hidden");
    const html = `
        <div class="upload-summary">
            <div class="summary-card"><div class="summary-num">${d.num_plates}</div><div class="summary-label">Plates</div></div>
            <div class="summary-card success"><div class="summary-num">${d.num_valid || 0}</div><div class="summary-label">Valid</div></div>
            <div class="summary-card vehicle"><div class="summary-num">${d.num_vehicles}</div><div class="summary-label">Vehicles</div></div>
            <div class="summary-card warn"><div class="summary-num">${d.num_persons}</div><div class="summary-label">Persons</div></div>
        </div>
        <div class="upload-meta">
            ${d.engine.detector_coco} · ${d.engine.detector_plate} · ${d.engine.ocr}<br>
            yolo_coco: ${d.inference_ms_yolo_coco}ms · yolo_plate: ${d.inference_ms_yolo_plate}ms · ocr: ${d.inference_ms_ocr_total}ms · total: ${(d.elapsed_seconds*1000).toFixed(0)}ms
        </div>
        <img class="upload-annotated" src="${d.annotated_url}" alt="annotated image">
        ${renderPlatesList(d.plates || [])}
        ${renderVehiclesList(d.vehicles || [])}
    `;
    $("upload-results-body").innerHTML = html;
}

function renderPlatesList(plates) {
    if (!plates.length) return "";
    const items = plates.map((p, i) => {
        const ok = p.valid_format || p.readable;
        return `<div class="upload-track-card" data-plate-idx="${i}">
            <span class="plate-text">${escapeHtml(p.text || '(empty)')}</span>
            ${ok ? '<span class="plate-valid">✓ Indian format</span>' : '<span class="plate-invalid">tentative</span>'}
            <div class="upload-meta">
                bbox ${(p.bbox_xyxy || []).join(", ")} · yolo ${(p.detection_confidence||0).toFixed(2)} · ocr ${(p.ocr_confidence||0).toFixed(2)}
            </div>
            ${p.crop_url ? `<img class="upload-annotated" src="${p.crop_url}" alt="plate crop">` : ""}
        </div>`;
    }).join("");
    return `<h4 style="margin-top:14px;margin-bottom:6px;font-size:12px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.05em;">Plates (${plates.length})</h4><div class="upload-track-list">${items}</div>`;
}

function renderVehiclesList(vehicles) {
    if (!vehicles.length) return "";
    const items = vehicles.map(v => {
        const plateBadge = v.plate && v.plate.text
            ? `<span class="plate-text">${escapeHtml(v.plate.text)}</span>${v.plate.valid ? '<span class="plate-valid">✓</span>' : '<span class="plate-invalid">?</span>'}`
            : '<span style="color:var(--text-mute);font-size:11px;">(no plate)</span>';
        return `<div class="upload-track-card">
            <b>${escapeHtml(v.class_name)}</b> <span style="color:var(--text-2);font-size:11px;">yolo ${(v.confidence||0).toFixed(2)}</span>
            · ${plateBadge}
            <div class="upload-meta">bbox ${(v.bbox_xyxy || []).join(", ")}</div>
            ${v.crop_url ? `<img class="upload-annotated" src="${v.crop_url}" alt="vehicle crop">` : ""}
        </div>`;
    }).join("");
    return `<h4 style="margin-top:14px;margin-bottom:6px;font-size:12px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.05em;">Vehicles (${vehicles.length})</h4><div class="upload-track-list">${items}</div>`;
}

// ==========================================================================
// VIDEO UPLOAD
// ==========================================================================
let uploadVideoFile = null;

function setupVideoUpload() {
    const dz = $("video-dropzone");
    const input = $("video-file-input");
    const preview = $("video-preview");
    const detectBtn = $("video-detect-btn");
    const clearBtn = $("video-clear-btn");
    const changeBtn = $("video-change-btn");
    const content = $("video-dropzone-content");
    const stopBtn = $("video-stop-btn");

    function pickFile() { input.click(); }
    dz.addEventListener("click", pickFile);
    changeBtn.addEventListener("click", (e) => { e.stopPropagation(); pickFile(); });

    // Drag-and-drop
    ["dragenter", "dragover"].forEach(ev => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add("drag-active");
        });
    });
    ["dragleave", "dragend"].forEach(ev => {
        dz.addEventListener(ev, (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.target === dz) dz.classList.remove("drag-active");
        });
    });
    dz.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dz.classList.remove("drag-active");
        const f = e.dataTransfer?.files?.[0];
        if (!f) return;
        if (!f.type.startsWith("video/")) {
            log(`Rejected dropped file (not a video): ${f.name}`);
            return;
        }
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
    });

    input.addEventListener("change", () => {
        const f = input.files[0];
        if (!f) return;
        uploadVideoFile = f;
        const url = URL.createObjectURL(f);
        preview.src = url;
        preview.classList.remove("hidden");
        changeBtn.classList.remove("hidden");
        content.classList.add("hidden");
        detectBtn.disabled = false;
        clearBtn.disabled = false;
        log(`Video selected: ${f.name} (${(f.size/1e6).toFixed(1)} MB)`);
    });

    clearBtn.addEventListener("click", () => {
        uploadVideoFile = null;
        input.value = "";
        preview.src = "";
        preview.classList.add("hidden");
        changeBtn.classList.add("hidden");
        content.classList.remove("hidden");
        detectBtn.disabled = true;
        clearBtn.disabled = true;
        resetUploadResults();
    });

    detectBtn.addEventListener("click", runVideoDetect);

    stopBtn.addEventListener("click", async () => {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping…";
        log("Sending cancel request…");
        try {
            await fetch("/api/cancel_video", { method: "POST" });
        } catch (_) { /* server may be busy in processing loop */ }
    });
}

async function runVideoDetect() {
    if (!uploadVideoFile) return;
    const detectBtn = $("video-detect-btn");
    const stopBtn = $("video-stop-btn");
    detectBtn.disabled = true;
    detectBtn.innerHTML = '<span class="spinner"></span><span class="btn-label">Processing…</span>';
    stopBtn.disabled = false;
    stopBtn.textContent = "■ Stop";
    log(`POST /api/detect_video (${uploadVideoFile.name})`);
    showUploadProgress("Processing video (ByteTrack + Awiros OCR). This may take a minute…");
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
        log(`Video processing done — ${d.n_tracks} tracks, ${d.n_valid_plates} valid plates, ${d.fps_processed} fps in ${d.elapsed_seconds}s`);
        renderVideoResults(d);
    } catch (e) {
        log("Video processing failed:", e.message);
        showUploadResultsError(e.message);
    } finally {
        detectBtn.disabled = false;
        detectBtn.innerHTML = '<span class="btn-icon">🎬</span><span class="btn-label">Run video tracking</span>';
        stopBtn.disabled = true;
        stopBtn.textContent = "■ Stop";
    }
}

function renderVideoResults(d) {
    $("upload-results-empty").classList.add("hidden");
    $("upload-results-body").classList.remove("hidden");
    $("upload-results-clear").classList.remove("hidden");
    const tracks = d.tracks || [];
    const html = `
        <div class="upload-summary">
            <div class="summary-card"><div class="summary-num">${d.n_tracks}</div><div class="summary-label">Tracks</div></div>
            <div class="summary-card success"><div class="summary-num">${d.n_valid_plates}</div><div class="summary-label">Valid plates</div></div>
            <div class="summary-card vehicle"><div class="summary-num">${d.fps_processed}</div><div class="summary-label">fps processed</div></div>
            <div class="summary-card time"><div class="summary-num">${d.elapsed_seconds}s</div><div class="summary-label">elapsed</div></div>
        </div>
        <div class="upload-meta">
            ${d.engine?.detector_coco || ''} · ${d.engine?.detector_plate || ''} · ${d.engine?.ocr || ''}<br>
            ${d.n_frames_processed}/${d.n_total_frames} frames @ stride=${d.stride} · ${d.fps} fps source · ${d.tracker}
        </div>
        ${d.annotated_video_url ? `<div class="upload-video-wrap">
            <video controls src="${d.annotated_video_url}"></video>
            <a class="btn-ghost" href="${d.annotated_video_url}" download="annotated.mp4">⬇ Download annotated.mp4</a>
        </div>` : ""}
        ${renderVideoTracksList(tracks, d)}
    `;
    $("upload-results-body").innerHTML = html;
}

function renderVideoTracksList(tracks, d) {
    if (!tracks.length) return "";
    const items = tracks.map(t => {
        const valid = t.valid_indian;
        const plateHtml = t.final_text
            ? `<span class="plate-text">${escapeHtml(t.final_text)}</span> ${valid ? '<span class="plate-valid">✓ Indian format</span>' : '<span class="plate-invalid">tentative</span>'}`
            : '<span style="color:var(--text-mute);font-size:11px;">(no plate OCR)</span>';
        const crop = t.best_crop_url
            ? `<img class="upload-annotated" src="${t.best_crop_url}" alt="best crop">`
            : "";
        const annotated = t.best_annotated_url
            ? `<img class="upload-annotated" src="${t.best_annotated_url}" alt="best annotated">`
            : "";
        const trackUrl = `${d.report_url || ""}/track_${t.track_id}`;
        return `<div class="upload-track-card" data-track-id="${t.track_id}">
            <b>#${t.track_id}</b> <span style="color:var(--text-1);">${escapeHtml(t.class_name || '?')}</span>
            <span style="color:var(--text-2);font-size:11px;">${t.n_frames} frames</span>
            · ${plateHtml}
            <div class="upload-meta">
                final conf ${(t.final_conf||0).toFixed(2)} · avg yolo ${(t.avg_yolo_conf||0).toFixed(2)} · n_unique_reads=${t.n_unique_reads}
                <br>first @ frame ${t.first_seen} → last @ frame ${t.last_seen}
            </div>
            ${crop}
            ${annotated}
            ${trackUrl && d.report_url ? `<a class="btn-ghost" href="${trackUrl}" target="_blank">📋 Detailed audit</a>` : ""}
        </div>`;
    }).join("");
    return `<h4 style="margin-top:14px;margin-bottom:6px;font-size:12px;color:var(--text-2);text-transform:uppercase;letter-spacing:0.05em;">Tracks (${tracks.length})</h4><div class="upload-track-list">${items}</div>`;
}

// ==========================================================================
// Upload helpers
// ==========================================================================
function showUploadProgress(text) {
    $("upload-results-empty").classList.add("hidden");
    $("upload-results-body").classList.remove("hidden");
    $("upload-results-body").innerHTML = `<div class="upload-progress"><span class="spinner"></span>${escapeHtml(text)}</div>`;
}

function showUploadResultsError(msg) {
    $("upload-results-body").innerHTML = `<div class="upload-progress" style="color:var(--red);">⚠ ${escapeHtml(msg)}</div>`;
}

function resetUploadResults() {
    $("upload-results-empty").classList.remove("hidden");
    $("upload-results-body").classList.add("hidden");
    $("upload-results-body").innerHTML = "";
    $("upload-results-clear").classList.add("hidden");
}

// Init on DOMContentLoaded — append to existing init
document.addEventListener("DOMContentLoaded", () => {
    setupModeToggle();
    setupUploadTabs();
    setupImageUpload();
    setupVideoUpload();
    $("upload-results-clear").addEventListener("click", resetUploadResults);
});
