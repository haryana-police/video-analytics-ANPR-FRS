import os
import time
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, render_template, request, redirect, send_from_directory

from core.engine import engine, _read_image, _decode_b64

image_bp = Blueprint("image", __name__)

HERE = Path(__file__).resolve().parent.parent
YOLO_MODEL = HERE / "yolo11_plate.pt"
UPLOAD_DIR = HERE / "uploads"
RESULTS_DIR = HERE / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_upload(file_storage) -> str:
    suffix = Path(file_storage.filename or "").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=suffix, delete=False)
    file_storage.save(tmp.name)
    tmp.close()
    return tmp.name

def _save_result_image(stem: str, img_bgr) -> str:
    import cv2
    fname = f"{stem}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(RESULTS_DIR / fname), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"/results/{fname}"

@image_bp.get("/")
def index():
    return render_template(
        "index.html",
        yolo_model=YOLO_MODEL.name,
    )

@image_bp.post("/api/predict")
def api_predict():
    try:
        if "file" not in request.files:
            return jsonify(error="No file uploaded. Send a file in the 'file' field."), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify(error="Empty filename."), 400
        
        model_coco = request.form.get("model_coco", "yolo11s")
        model_plate = request.form.get("model_plate", "yolo11_plate")
        conf = float(request.form.get("conf", 0.25))

        path = _save_upload(f)
        try:
            img = _read_image(path)
            t0 = time.time()
            out = engine.predict(
                img, conf=conf, model_coco_name=model_coco, model_plate_name=model_plate
            )
            out["total_ms"] = round((time.time() - t0) * 1000, 1)
            out["filename"] = f.filename
            out["timestamp"] = datetime.now().isoformat(timespec="seconds")
            return jsonify(out)
        finally:
            try: os.unlink(path)
            except OSError: pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@image_bp.post("/api/detect")
def api_detect():
    try:
        f = request.files.get("image")
        if f is None or not f.filename:
            return jsonify(error="No image uploaded. Send a file in the 'image' field."), 400

        conf = float(request.form.get("conf", 0.25))
        model_coco = request.form.get("model_coco", "yolo11s")
        model_plate = request.form.get("model_plate", "yolo11_plate")
        
        stem = Path(f.filename).stem
        path = _save_upload(f)
        try:
            img = _read_image(path)
            t0 = time.time()
            out = engine.predict(
                img, conf=conf, model_coco_name=model_coco, model_plate_name=model_plate
            )
            elapsed = round(time.time() - t0, 2)

            # Map plates to the flat format for retro-compatibility if needed
            plates_payload = []
            for p in out["plates"]:
                plates_payload.append({
                    "text": p["text"],
                    "valid_format": p["readable"],
                    "detection_confidence": p["confidence"],
                    "ocr_confidence": p["ocr_confidence"],
                    "crop_url": p["crop_url"],
                    "bbox_xyxy": p["bbox_xyxy"],
                    "raw_ocr": p["text"],
                    "fixes_applied": [],
                    "engine": "Awiros ANPR-OCR",
                })

            annotated_url = _save_result_image(stem, _decode_b64(out["annotated_jpeg_b64"]))
            response = {
                "filename": f.filename,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "size": out["size"],
                "num_plates": out["num_plates"],
                "num_valid": sum(1 for p in plates_payload if p["valid_format"]),
                "num_vehicles": out["num_vehicles"],
                "num_persons": out["num_persons"],
                "vehicles": out["vehicles"],
                "persons": out["persons"],
                "plates": plates_payload,
                "annotated_url": annotated_url,
                "elapsed_seconds": elapsed,
                "inference_ms_yolo_coco": out["inference_ms_yolo_coco"],
                "inference_ms_yolo_plate": out["inference_ms_yolo_plate"],
                "inference_ms_ocr_total": out["inference_ms_ocr_total"],
                "engine": {
                    "detector_coco": f"YOLO11 ({model_coco})",
                    "detector_plate": f"YOLO11 ({model_plate})",
                    "ocr": "Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)",
                    "device": "cpu",
                },
            }
            return jsonify(response)
        finally:
            try: os.unlink(path)
            except OSError: pass
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

@image_bp.get("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_DIR, filename)

@image_bp.get("/health")
def health():
    return jsonify(
        status="ok",
        yolo_loaded=len(engine.yolo_models) > 0,
        awiros_loaded=engine.awiros is not None,
    )


@image_bp.get("/health/full")
def health_full():
    """Health endpoint used by the live-dashboard to surface what the
    pipeline has cached. Includes per-model readiness flags."""
    return jsonify(
        status="ok",
        yolo_loaded=len(engine.yolo_models) > 0,
        awiros_loaded=engine.awiros is not None,
        cached_models=sorted(engine.yolo_models.keys()),
    )
