"""
Live-stream endpoints for the Video Analytics ANPR FRS dashboard.

Endpoints:
    GET  /api/live/sources                  → list of available sample videos
    POST /api/live/start                    → create a session, returns session_id
    GET  /api/live/mjpeg/<sid>              → MJPEG stream of annotated frames
    GET  /api/live/events/<sid>             → Server-Sent Events (JSON detection events)
    POST /api/live/stop/<sid>               → stop + remove session
    GET  /api/live/benchmark                → benchmark all yolo11 variants + awiros OCR + bytetrack
    GET  /api/live/session/<sid>            → current session metadata

    All three benchmarks operate on the same representative 1280x720 CCTV
    frame (a pinned copy at static/benchmark_sample.jpg) so the numbers are
    directly comparable.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, stream_with_context

from core.engine import engine
from core.live import (
    LivePipeline,
    LiveSession,
    LiveSessionManager,
    LiveSource,
    _BENCHMARK_SAMPLE_REL,
    _load_benchmark_frame,
    benchmark_awiros,
    benchmark_model,
    benchmark_tracker,
)

log = logging.getLogger("live_routes")
live_bp = Blueprint("live", __name__)

HERE = Path(__file__).resolve().parent.parent
SESSION_MANAGER = LiveSessionManager()

# Default sample-videos directory (overridden by run.py if --sample-videos-dir is set).
# Drop .mp4 files into <repo>/sample_videos/ or pass --sample-videos-dir.
SAMPLE_VIDEOS_DIR = HERE / "sample_videos"


def set_sample_videos_dir(path: str):
    global SAMPLE_VIDEOS_DIR
    SAMPLE_VIDEOS_DIR = Path(path)
    log.info("Sample videos dir set to: %s", SAMPLE_VIDEOS_DIR)


# ---------------------------------------------------------------------------
# /api/live/sources — list local sample videos
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/sources")
def api_live_sources():
    """Return available video sources the user can stream from."""
    out = []
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    if SAMPLE_VIDEOS_DIR.exists():
        for p in sorted(
            p for ext in extensions for p in SAMPLE_VIDEOS_DIR.glob(ext)
        ):
            out.append({
                "name": p.stem,
                "path": str(p),
                "size_mb": round(p.stat().st_size / 1e6, 1),
                "kind": "local",
            })
        # also list subfolders (e.g. tests/)
        for sub in sorted(SAMPLE_VIDEOS_DIR.iterdir()):
            if sub.is_dir():
                for p in sorted(
                    p for ext in extensions for p in sub.glob(ext)
                ):
                    out.append({
                        "name": f"{sub.name}/{p.stem}",
                        "path": str(p),
                        "size_mb": round(p.stat().st_size / 1e6, 1),
                        "kind": "local",
                    })
    # Always include the camera placeholder
    out.append({
        "name": "Webcam (index 0)",
        "path": "0",
        "kind": "camera",
    })
    return jsonify(sources=out, sample_videos_dir=str(SAMPLE_VIDEOS_DIR))


# ---------------------------------------------------------------------------
# /api/live/start — create a new live session
# ---------------------------------------------------------------------------
@live_bp.post("/api/live/start")
def api_live_start():
    try:
        body = request.get_json(silent=True) or {}
        source = body.get("source", "")
        if not source:
            return jsonify(error="Missing 'source' (local path, URL, or '0' for webcam)."), 400
        try:
            target_fps = float(body.get("target_fps", 15.0))
        except (TypeError, ValueError):
            return jsonify(error="target_fps must be a number."), 400
        target_fps = min(60.0, max(1.0, target_fps))
        model_coco = body.get("model_coco", "yolo11n")
        model_plate = body.get("model_plate", "yolo11_plate")
        ocr_on_best_only = bool(body.get("ocr_on_best_only", True))
        best_crop_algo = body.get("best_crop_algo", "area")
        loop = bool(body.get("loop", True))

        # Resolve camera index — keep it an int so OpenCV opens the device,
        # not a file literally named "0".
        resolved_source = 0 if source == "0" else source

        # Restrict file sources to the configured sample-videos dir and URL
        # sources to expected camera schemes — /api/live/start must not become
        # an arbitrary local-file reader or internal-network prober.
        if isinstance(resolved_source, str) and not resolved_source.lower().startswith(
            ("rtsp://", "http://", "https://")
        ):
            src_path = Path(resolved_source).resolve()
            allowed_root = SAMPLE_VIDEOS_DIR.resolve()
            try:
                src_path.relative_to(allowed_root)
            except ValueError:
                return jsonify(error=(
                    "File sources must live inside the sample-videos directory "
                    f"({allowed_root}). Use a sample video, a webcam index ('0'), "
                    "or an rtsp/http(s) URL."
                )), 400
            if not src_path.exists():
                return jsonify(error=f"Source not found: {source}"), 404
            resolved_source = str(src_path)

        # Create source
        live_source = LiveSource(resolved_source, target_fps=target_fps, loop=loop)

        # Build FRESH YOLO instances for this session. ByteTrack's `persist=True`
        # state lives on the YOLO model object, so sharing the engine's cached
        # models across concurrent sessions would mix tracker state (and crash
        # with TypeError on the second `.track()` call after a tracker reset).
        try:
            engine._ensure_awiros()
            awiros = engine.awiros
            yolo_coco = engine._build_yolo(model_coco)
            yolo_plate = engine._build_yolo(model_plate)
        except Exception as e:
            traceback.print_exc()
            return jsonify(error=f"Model load failed: {e}"), 500

        # Allocate session
        import uuid
        sid = uuid.uuid4().hex[:12]
        pipeline = LivePipeline(
            session_id=sid,
            source=live_source,
            yolo_coco=yolo_coco,
            yolo_plate=yolo_plate,
            awiros=awiros,
            ocr_on_best_only=ocr_on_best_only,
            iou_thresh=0.20,
            device=engine._OPENVINO_DEVICE or "cpu",
            best_crop_algo=best_crop_algo,
        )
        session = LiveSession(source=live_source, pipeline=pipeline)
        session.meta = {
            "source": str(resolved_source),
            "target_fps": target_fps,
            "model_coco": model_coco,
            "model_plate": model_plate,
            "ocr_on_best_only": ocr_on_best_only,
            "best_crop_algo": pipeline.best_crop_algo,
            "loop": loop,
        }
        SESSION_MANAGER.add(sid, session)
        pipeline.start()
        log.info("Live session %s started: source=%s models=(coco:%s, plate:%s)",
                 sid, source, model_coco, model_plate)
        return jsonify(
            session_id=sid,
            mjpeg_url=f"/api/live/mjpeg/{sid}",
            events_url=f"/api/live/events/{sid}",
            stop_url=f"/api/live/stop/{sid}",
            session_url=f"/api/live/session/{sid}",
            **session.meta,
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500


# ---------------------------------------------------------------------------
# /api/live/mjpeg/<sid> — annotated MJPEG stream
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/mjpeg/<sid>")
def api_live_mjpeg(sid: str):
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found or stopped."), 404

    q = session.pipeline.jpeg_queue()
    boundary = b"--frame"

    def generate():
        # Send an initial empty frame so the <img> tag starts rendering
        SESSION_MANAGER.stream_open(sid)
        try:
            while True:
                s = SESSION_MANAGER.get(sid)
                if s is None:
                    return
                SESSION_MANAGER.touch(sid)
                try:
                    jpeg = q.get(timeout=2.0)
                except Exception:
                    continue
                if jpeg is None or jpeg == b"":
                    continue
                yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        except GeneratorExit:
            log.info("MJPEG client disconnected from session %s", sid)
            return
        finally:
            SESSION_MANAGER.stream_close(sid)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# /api/live/events/<sid> — Server-Sent Events stream
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/events/<sid>")
def api_live_events(sid: str):
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found or stopped."), 404

    q = session.pipeline.event_queue()

    def generate():
        SESSION_MANAGER.stream_open(sid)
        last_keepalive = time.time()
        try:
            while True:
                s = SESSION_MANAGER.get(sid)
                if s is None:
                    yield "event: end\ndata: {}\n\n"
                    return
                SESSION_MANAGER.touch(sid)
                try:
                    evt = q.get(timeout=0.5)
                except Exception:
                    evt = None
                if evt is not None:
                    payload = json.dumps(evt, default=str)
                    yield f"event: {evt.get('type', 'message')}\ndata: {payload}\n\n"
                    last_keepalive = time.time()
                # Periodic keep-alive — yields even on idle so the WSGI
                # server flushes the response (Flask dev server buffers
                # until the generator yields).
                if time.time() - last_keepalive > 5:
                    yield ": keep-alive\n\n"
                    last_keepalive = time.time()
        except GeneratorExit:
            log.info("SSE client disconnected from session %s", sid)
            return
        finally:
            SESSION_MANAGER.stream_close(sid)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# /api/live/crop/<sid>/<track_id> — best plate crop JPEG (v4)
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/crop/<sid>/<track_id>")
def api_live_crop(sid: str, track_id: str):
    """Return the best-crop JPEG for a plate track_id (for the detail modal)."""
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found."), 404
    try:
        tid = int(track_id)
    except ValueError:
        return jsonify(error="track_id must be an integer."), 400
    crop = session.pipeline._best_crops.get(tid)
    if not crop:
        return jsonify(error=f"No best crop cached for plate track {tid}."), 404
    return Response(crop, mimetype="image/jpeg")


# ---------------------------------------------------------------------------
# /api/live/pause/<sid> — freeze/resume frame processing
# ---------------------------------------------------------------------------
@live_bp.post("/api/live/pause/<sid>")
def api_live_pause(sid: str):
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found."), 404
    body = request.get_json(silent=True) or {}
    paused = bool(body.get("paused", True))
    session.pipeline.set_paused(paused)
    log.info("Session %s %s", sid, "paused" if paused else "resumed")
    return jsonify(session_id=sid, paused=paused)


# ---------------------------------------------------------------------------
# /api/live/stop/<sid>
# ---------------------------------------------------------------------------
@live_bp.post("/api/live/stop/<sid>")
def api_live_stop(sid: str):
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found."), 404
    try:
        session.pipeline.stop()
    except Exception:
        log.exception("Error stopping pipeline")
    SESSION_MANAGER.remove(sid)
    return jsonify(status="stopped", session_id=sid)


# ---------------------------------------------------------------------------
# /api/live/session/<sid>
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/session/<sid>")
def api_live_session(sid: str):
    session = SESSION_MANAGER.get(sid)
    if session is None:
        return jsonify(error="Session not found."), 404
    p = session.pipeline
    return jsonify(
        session_id=sid,
        meta=session.meta,
        frame_index=p.frame_index,
        fps_actual=round(p.fps_actual, 2),
        n_vehicles=len(p.vehicle_states),
        n_persons=len(p.person_states),
        n_plates=len(p.plate_states),
        active_streams=session.active_streams,
        paused=p.paused,
        elapsed_sec=round(time.time() - p.t_start, 1) if p.t_start else 0,
    )


# ---------------------------------------------------------------------------
# /api/live/benchmark — benchmark all yolo11 variants + awiros OCR + bytetrack
# ---------------------------------------------------------------------------
@live_bp.get("/api/live/benchmark")
def api_live_benchmark():
    """Time YOLO11 variants (n/s/m/l), Awiros OCR, and ByteTracker.

    Returns ms/frame (YOLO) or ms/crop (Awiros) or ms/update (ByteTrack) +
    max throughput for each. Useful to see at a glance which component is
    the live pipeline's bottleneck.
    """
    HERE = Path(__file__).resolve().parent.parent
    out = []

    # Load the shared representative frame ONCE. All three components
    # (YOLO detectors, Awiros OCR, ByteTrack) measure on this exact frame
    # so the numbers are directly comparable.
    bench_frame = _load_benchmark_frame()
    out.append({
        "kind": "frame",
        "name": "benchmark_sample",
        "source": _BENCHMARK_SAMPLE_REL,
        "shape": list(bench_frame.shape),
        "note": "pinned representative CCTV frame committed at static/benchmark_sample.jpg",
    })

    # 1) YOLO detectors (COCO + plate variants).
    # Measure exactly what the live pipeline would load for each name
    # (engine._resolve_yolo_source prefers *_openvino_model/ dirs), but only
    # include variants whose weights already exist locally so clicking
    # "Run benchmark" never triggers a surprise Hugging Face download.
    def _local_candidate(name: str) -> bool:
        return (HERE / f"{name}.pt").exists() or (HERE / f"{name}_openvino_model").is_dir()

    yolo_candidates = []
    for name in ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]:
        if _local_candidate(name):
            yolo_candidates.append(("coco", name))
    for name in ["yolo11_plate", "yolo11s_plate", "yolo11m_plate", "yolo11l_plate"]:
        if _local_candidate(name):
            yolo_candidates.append(("plate", name))

    if not yolo_candidates:
        out.append({"name": "yolo", "kind": "coco", "error": "No YOLO11 weights found in app/. Run install first."})
    else:
        for kind, name in yolo_candidates:
            try:
                src = engine._resolve_yolo_source(name)
                res = benchmark_model(str(src), frame=bench_frame)
                res["kind"] = kind
                res["name"] = f"{name}" + ("/openvino" if str(src).endswith("_openvino_model") else "")
                out.append(res)
            except Exception as e:
                out.append({"name": name, "kind": kind, "error": str(e)})

    # 2) Awiros OCR (primed at startup — no PaddlePaddle call from request thread)
    import core.live as _live_mod
    awiros_dir = HERE / "awiros_anpr"
    if (awiros_dir / "model.safetensors").exists() and (awiros_dir / "en_dict.txt").exists():
        primed = _live_mod._AWIROS_CACHE or _live_mod._AWIROS_PRIMED
        if primed is not None:
            out.append(primed)
        else:
            out.append({
                "name": "awiros_anpr", "kind": "ocr",
                "error": "Awiros benchmark not primed. Restart server to prime it.",
            })
    else:
        out.append({
            "name": "awiros_anpr", "kind": "ocr",
            "error": "model.safetensors or en_dict.txt missing. Run install first.",
        })

    # 3) ByteTrack tracker (no weights — pure-Python + ultralytics kalman)
    log.info("benchmark: starting ByteTrack")
    try:
        out.append(benchmark_tracker())
        log.info("benchmark: ByteTrack done")
    except Exception as e:
        log.exception("benchmark: ByteTrack failed")
        out.append({"name": "bytetrack", "kind": "tracker", "error": str(e)})

    return jsonify(results=out)