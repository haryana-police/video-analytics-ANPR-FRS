import sys
import time
import json
import traceback
import threading
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, request, send_from_directory

from core.engine import engine

video_bp = Blueprint("video", __name__)

HERE = Path(__file__).resolve().parent.parent
VIDEOS_DIR = HERE / "uploads_videos"
VIDEO_RESULTS_DIR = HERE / "video_results"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# One video job at a time: process_video() uses the shared cached YOLO models
# and resets their ByteTrack state mid-run, so two concurrent jobs would mix
# track IDs across videos (and the single global cancel flag would cancel both).
_VIDEO_JOB_LOCK = threading.Lock()

# Result dirs older than this are pruned on each new upload (audit artifacts
# are the UI's per-track crops/frames — keep a window, not forever).
RETENTION_DAYS = 7


def _prune_old_results():
    """Delete video-results dirs (and stray uploads) older than RETENTION_DAYS."""
    cutoff = time.time() - RETENTION_DAYS * 86400
    for root in (VIDEO_RESULTS_DIR, VIDEOS_DIR):
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for p in entries:
            try:
                if p.stat().st_mtime < cutoff:
                    if p.is_dir():
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)
            except OSError:
                continue


@video_bp.post("/api/detect_video")
def api_detect_video():
    # Non-blocking acquire: reject rather than queue silently for minutes.
    if not _VIDEO_JOB_LOCK.acquire(blocking=False):
        return jsonify(error="A video is already being processed. Wait for it to finish or stop it."), 409
    try:
        return _detect_video_impl()
    finally:
        _VIDEO_JOB_LOCK.release()


def _detect_video_impl():
    try:
        f = request.files.get("video")
        if f is None or not f.filename:
            return jsonify(error="No video uploaded. Send a file in the 'video' field."), 400

        try:
            stride = max(1, int(request.form.get("frame_stride", 2)))
        except (TypeError, ValueError):
            return jsonify(error="frame_stride must be an integer."), 400
        max_frames_raw = request.form.get("max_frames", "")
        try:
            max_frames = int(max_frames_raw) if max_frames_raw.strip() else None
        except (TypeError, ValueError):
            return jsonify(error="max_frames must be an integer."), 400
        if max_frames is not None and max_frames <= 0:
            return jsonify(error="max_frames must be positive."), 400
        write_video = request.form.get("write_video", "1") != "0"
        
        # New model configuration parameters
        model_coco = request.form.get("model_coco", "yolo11s")
        model_plate = request.form.get("model_plate", "yolo11_plate")

        suffix = Path(f.filename).suffix or ".mp4"
        ts_ms = int(time.time() * 1000)
        stem = Path(f.filename).stem
        video_name = f"{stem}_{ts_ms}{suffix}"
        video_path = VIDEOS_DIR / video_name
        f.save(str(video_path))

        out_dir = VIDEO_RESULTS_DIR / f"{stem}_awiros_{ts_ms}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from anpr_video_awiros import process_video

        _prune_old_results()

        t0 = time.time()
        try:
            summary = process_video(
                video_path=video_path,
                out_dir=out_dir,
                stride=stride,
                max_frames=max_frames,
                write_video=write_video,
                yolo_model=model_plate,
                yolo_coco_model=model_coco,
                device="cpu",
            )
        finally:
            # The uploaded source is only needed during processing; result
            # artifacts live under out_dir. Without this, every upload pins
            # its full size on disk forever (500 MB cap per file).
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass
        elapsed = round(time.time() - t0, 2)

        annotated_video_url = ""
        if write_video and (out_dir / "annotated.mp4").exists():
            annotated_video_url = f"/video-results/{out_dir.relative_to(VIDEO_RESULTS_DIR)}/annotated.mp4"
        report_url = f"/video-results/{out_dir.relative_to(VIDEO_RESULTS_DIR)}/report.html"

        tracks_payload = []
        rel_to_base = out_dir.relative_to(VIDEO_RESULTS_DIR)
        
        for t in summary["tracks"]:
            best_crop_url = ""
            if t.get("best_crop_file"):
                best_crop_url = f"/video-results/{rel_to_base}/best_frames/{t['best_crop_file']}"
                
            best_annotated_url = ""
            if t.get("best_annotated_file"):
                best_annotated_url = f"/video-results/{rel_to_base}/best_frames/{t['best_annotated_file']}"
                
            vehicle_crop_url = ""
            if t.get("vehicle_crop_file"):
                vehicle_crop_url = f"/video-results/{rel_to_base}/best_frames/{t['vehicle_crop_file']}"

            # In-app audit endpoint (per-frame OCR reads + crops) — the old
            # report_url pointed at a report.html that process_video never
            # generated, so links built from it 404'd.
            audit_url = f"/api/track_details/{rel_to_base}/track_{t['track_id']}"

            tracks_payload.append({
                "track_id": t["track_id"],
                "class_name": t.get("class_name", "car"),
                "final_text": t["final_text"],
                "final_conf": t["final_conf"],
                "valid_indian": t["valid_indian"],
                "n_frames": t["n_frames"],
                "first_seen": t["first_seen"],
                "last_seen": t["last_seen"],
                "avg_yolo_conf": t["avg_yolo_conf"],
                "n_unique_reads": t["n_unique_reads"],
                "votes_per_pos": t["votes_per_pos"],
                "best_frame": t.get("best_frame"),
                "best_text": t.get("best_text", ""),
                "best_conf": t.get("best_conf", 0.0),
                "best_crop_url": best_crop_url,
                "best_annotated_url": best_annotated_url,
                "vehicle_crop_url": vehicle_crop_url,
                "audit_url": audit_url,
                "all_frames": t.get("all_frames", []),
                "per_frame_reads": t.get("per_frame_reads", []),
                "crop_url": "",
            })

        return jsonify({
            "filename": f.filename,
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n_tracks": summary["n_tracks"],
            "n_valid_plates": summary["n_valid_plates"],
            "n_frames_processed": summary["n_frames_processed"],
            "n_total_frames": summary["n_total_frames"],
            "fps": summary["fps"],
            "stride": stride,
            "tracker": summary["tracker"],
            "elapsed_seconds": elapsed,
            "fps_processed": summary["fps_processed"],
            "tracks": tracks_payload,
            "annotated_video_url": annotated_video_url,
            "report_url": report_url,
            "engine": {
                "detector_coco": f"YOLO11 ({model_coco})",
                "detector_plate": f"YOLO11 ({model_plate})",
                "ocr": "Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)",
                "device": engine._OPENVINO_DEVICE or "cpu",
                "voting": "per-character position voting across the track's OCR reads",
            },
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@video_bp.post("/api/cancel_video")
def api_cancel_video():
    """Signal the running video processing loop to stop after the current frame."""
    from anpr_video_awiros import request_cancel
    request_cancel()
    return jsonify(status="cancel_requested")

@video_bp.get("/video-results/<path:relpath>")
def serve_video_result(relpath):
    target = (VIDEO_RESULTS_DIR / relpath).resolve()
    try:
        target.relative_to(VIDEO_RESULTS_DIR.resolve())
    except ValueError:
        return jsonify(error="Invalid path."), 400
    if target.is_dir():
        return jsonify(error="Directory listing disabled."), 404
    if not target.exists():
        return jsonify(error="Not found."), 404
    return send_from_directory(target.parent, target.name, as_attachment=False)

@video_bp.get("/api/track_details/<path:relpath>")
def api_track_details(relpath):
    target_dir = (VIDEO_RESULTS_DIR / relpath).resolve()
    try:
        target_dir.relative_to(VIDEO_RESULTS_DIR.resolve())
    except ValueError:
        return jsonify(error="Invalid path."), 400
    if not target_dir.is_dir():
        return jsonify(error="Not a video result dir."), 404

    track_id_raw = target_dir.name
    if not track_id_raw.startswith("track_"):
        return jsonify(error="Path must end with /track_<id>."), 400
    try:
        wanted_id = int(track_id_raw.split("_", 1)[1])
    except ValueError:
        return jsonify(error="Track id must be an integer."), 400

    video_dir = target_dir.parent
    summary_json = video_dir / "summary.json"

    track = None
    if summary_json.exists():
        try:
            data = json.loads(summary_json.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        tracks_list = data.get("tracks") or []
        for t in tracks_list:
            try:
                tid = int(t.get("track_id", -1))
            except (TypeError, ValueError):
                continue
            if tid == wanted_id:
                track = t
                break

    if track is None:
        return jsonify(error=f"Track {wanted_id} not found."), 404

    rel_to_base = str(video_dir.relative_to(VIDEO_RESULTS_DIR))
    crop_url_prefix = f"/video-results/{rel_to_base}/crops/"
    
    best_crop_url = ""
    if track.get("best_crop_file"):
        best_crop_url = f"/video-results/{rel_to_base}/best_frames/{track['best_crop_file']}"
        
    best_annotated_url = ""
    if track.get("best_annotated_file"):
        best_annotated_url = f"/video-results/{rel_to_base}/best_frames/{track['best_annotated_file']}"

    vehicle_crop_url = ""
    if track.get("vehicle_crop_file"):
        vehicle_crop_url = f"/video-results/{rel_to_base}/best_frames/{track['vehicle_crop_file']}"

    # Map frame crops
    formatted_reads = []
    for r in track.get("per_frame_reads", []):
        r_copy = r.copy()
        if r.get("crop_file"):
            r_copy["crop_url"] = crop_url_prefix + r["crop_file"]
        formatted_reads.append(r_copy)

    payload = {
        "track_id": track["track_id"],
        "class_name": track.get("class_name", "car"),
        "final_text": track["final_text"],
        "final_conf": track["final_conf"],
        "valid_indian": track["valid_indian"],
        "n_frames": track["n_frames"],
        "first_seen": track["first_seen"],
        "last_seen": track["last_seen"],
        "avg_yolo_conf": track["avg_yolo_conf"],
        "n_unique_reads": track["n_unique_reads"],
        "votes_per_pos": track["votes_per_pos"],
        "best_frame": track.get("best_frame"),
        "best_text": track.get("best_text", ""),
        "best_conf": track.get("best_conf", 0.0),
        "best_crop_url": best_crop_url,
        "best_annotated_url": best_annotated_url,
        "vehicle_crop_url": vehicle_crop_url,
        "all_frames": track.get("all_frames", []),
        "per_frame_reads": formatted_reads,
    }
    return jsonify(payload)

@video_bp.get("/walkthrough")
def api_walkthrough():
    p = (HERE / "walkthrough.html").resolve()
    if not p.exists():
        return jsonify(error="walkthrough.html not generated yet"), 404
    return send_from_directory(p.parent, p.name)
