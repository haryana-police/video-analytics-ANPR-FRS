"""
YOLO 11 plate detection + Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC).

Workflow:
1. Run YOLO 11 plate detector on every image in the source folder.
2. Save annotated images + cropped plate regions.
3. For each crop, run Awiros ANPR-OCR (model.safetensors + en_dict.txt) to read
   the plate text. Uses PaddlePaddle (CPU) + the ppocr package shipped in the
   Awiros repo (PaddleOCR/).
4. Re-draw annotated images with detection bbox + OCR text label.
5. Write a final summary.json containing model info + detections + OCR text.

Usage:
    python detect_yolo11_awiros_ocr.py
    python detect_yolo11_awiros_ocr.py --src <input_dir> --conf 0.25
"""

import argparse
import copy
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from core.plate_validate import validate_indian_plate as _validate_plate

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
# Default paths are resolved relative to the repo root so the script is
# portable - no hardcoded user-specific folders.
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = str(HERE / "yolo11_plate.pt")
DEFAULT_SRC = str(HERE / "uploads")
DEFAULT_OUT_PARENT = str(HERE / "results")
DEFAULT_AWIROS_DIR = str(HERE / "awiros_anpr")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# PP-OCRv5 server rec / SVTR_HGNet architecture config (from Awiros test.py)
CTC_NUM_CLASSES = 64
NRTR_NUM_CLASSES = 67  # NRTRHead adds +1 internally -> 68 to match weights
IMAGE_SHAPE = [3, 48, 320]

# Crops shorter than this virtually never yield a usable read from Awiros.
# Enforced inside predict_crop() so EVERY pipeline path (video, live,
# image API, folder CLI) shares one policy instead of burning ~850 ms of
# CPU per hopeless crop. (Formerly the unused OCR_MIN_HEIGHT tunable in
# anpr_video_awiros.py.)
MIN_CROP_HEIGHT = 18

# PaddlePaddle's static graph is NOT thread-safe: a predict_crop() call from a
# second thread corrupts the shared program (compat meta-tensor invalidated).
# Serializing INSIDE predict_crop() protects every caller — live sessions,
# /api/detect, /api/detect_video, benchmarks — no matter which thread or how
# many concurrent Flask requests reach it.
AWIROS_PREDICT_LOCK = threading.RLock()

MODEL_CONFIG = {
    "Architecture": {
        "model_type": "rec",
        "algorithm": "SVTR_HGNet",
        "Transform": None,
        "Backbone": {"name": "PPHGNetV2_B4", "text_rec": True},
        "Head": {
            "name": "MultiHead",
            "out_channels_list": {
                "CTCLabelDecode": CTC_NUM_CLASSES,
                "NRTRLabelDecode": NRTR_NUM_CLASSES,
            },
            "head_list": [
                {
                    "CTCHead": {
                        "Neck": {
                            "name": "svtr",
                            "dims": 120,
                            "depth": 2,
                            "hidden_dims": 120,
                            "kernel_size": [1, 3],
                            "use_guide": True,
                        },
                        "Head": {"fc_decay": 1e-05},
                    }
                },
                {"NRTRHead": {"nrtr_dim": 384, "max_text_length": 25}},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Awiros ANPR-OCR wrapper
# ---------------------------------------------------------------------------
class AwirosANPR:
    """Lazy-loaded wrapper around Awiros ANPR-OCR (PaddlePaddle / ppocr).

    The model is a SVTR_HGNet PP-OCRv5-server recognition head that emits a CTC
    prediction over a 64-class character dictionary (Indian plates).
    """

    def __init__(self, awiros_dir: Path, device: str = "cpu"):
        self.awiros_dir = Path(awiros_dir)
        self.weights_path = self.awiros_dir / "model.safetensors"
        self.dict_path = self.awiros_dir / "en_dict.txt"
        self.paddleocr_dir = self.awiros_dir / "PaddleOCR"
        self.device = device
        self.model = None
        self.post_process = None
        self.paddle = None
        self._loaded = False

    def _ensure_paddleocr(self):
        """Insert the PaddleOCR repo (which provides `ppocr`) into sys.path."""
        if not self.paddleocr_dir.is_dir():
            raise FileNotFoundError(
                f"PaddleOCR repo not found at {self.paddleocr_dir}. "
                "Restore it with: git clone --depth 1 "
                "https://github.com/PaddlePaddle/PaddleOCR into that path."
            )
        root = str(self.paddleocr_dir)
        if root not in sys.path:
            sys.path.insert(0, root)

    def load(self):
        if self._loaded:
            return
        self._ensure_paddleocr()

        import paddle  # noqa: WPS433 (delayed import)

        from ppocr.modeling.architectures import build_model as ppocr_build_model
        from ppocr.postprocess import build_post_process
        from safetensors.numpy import load_file as st_load

        if self.device == "gpu" and not paddle.is_compiled_with_cuda():
            print("[AWIROS] CUDA not available, falling back to CPU.")
            paddle.set_device("cpu")
        else:
            paddle.set_device(self.device)

        # Build CTC post-processor
        self.post_process = build_post_process({
            "name": "CTCLabelDecode",
            "character_dict_path": str(self.dict_path),
            "use_space_char": True,
        })

        # Build model + load safetensors weights
        config = copy.deepcopy(MODEL_CONFIG)
        self.model = ppocr_build_model(config["Architecture"])
        self.model.eval()

        np_state = st_load(str(self.weights_path))
        state_dict = {k: paddle.to_tensor(v) for k, v in np_state.items()}
        self.model.set_state_dict(state_dict)
        self.paddle = paddle
        self._loaded = True
        print(f"[AWIROS] Loaded weights: {self.weights_path.name} "
              f"({self.weights_path.stat().st_size / 1e6:.1f} MB)")
        print(f"[AWIROS] Dict: {self.dict_path.name}")
        print(f"[AWIROS] Device: {self.device}")

    @staticmethod
    def _resize_for_rec(img_bgr, target_shape):
        _, h, w = target_shape
        img_h, img_w = img_bgr.shape[:2]
        ratio = h / img_h
        new_w = int(img_w * ratio)

        if new_w <= w:
            # Fits within max width — scale height to 48, pad width if needed.
            interp = cv2.INTER_AREA if img_h > h else cv2.INTER_CUBIC
            resized = cv2.resize(img_bgr, (new_w, h), interpolation=interp)
            if new_w < w:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:, :new_w, :] = resized
                resized = padded
        else:
            # Exceeds max width — scale to width=320 (height < 48) preserving
            # aspect ratio, then pad height.  Avoids horizontal squish that
            # distorts character shapes for wide plates (>6.67:1).
            w_ratio = w / img_w
            new_h = int(img_h * w_ratio)
            interp = cv2.INTER_AREA if img_h > new_h else cv2.INTER_CUBIC
            resized = cv2.resize(img_bgr, (w, new_h), interpolation=interp)
            if new_h < h:
                padded = np.zeros((h, w, 3), dtype=np.uint8)
                padded[:new_h, :, :] = resized
                resized = padded
        return resized

    @staticmethod
    def _preprocess(img_bgr, target_shape):
        img = AwirosANPR._resize_for_rec(img_bgr, target_shape)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return img.transpose((2, 0, 1))

    def predict_crop(self, crop_bgr) -> dict:
        """Run OCR on one cropped plate BGR image (thread-serialized).

        Crops shorter than MIN_CROP_HEIGHT are skipped without inference —
        Awiros virtually never reads them, and each call costs ~850 ms CPU.
        Returns the same shape as an unreadable crop so callers need no
        special-casing.
        """
        self.load()
        if (crop_bgr is None or crop_bgr.size == 0
                or crop_bgr.shape[0] < MIN_CROP_HEIGHT):
            return {"text": "", "confidence": 0.0, "readable": False}

        with AWIROS_PREDICT_LOCK:
            tensor = self.paddle.to_tensor(
                np.expand_dims(self._preprocess(crop_bgr, IMAGE_SHAPE), axis=0)
            )
            with self.paddle.no_grad():
                preds = self.model(tensor)

            if isinstance(preds, dict):
                pred_tensor = preds.get("ctc", next(iter(preds.values())))
            elif isinstance(preds, (list, tuple)):
                pred_tensor = preds[0]
            else:
                pred_tensor = preds

            post_result = self.post_process(pred_tensor.numpy())
        if isinstance(post_result, (list, tuple)) and len(post_result) > 0:
            text, confidence = post_result[0]
        else:
            text, confidence = "", 0.0

        text = (text or "").strip()
        # Heuristic: a plate with 4+ alphanumeric chars and conf > 0.2 is "readable".
        # Final "is it a real Indian plate?" check uses a regex grammar in
        # core.plate_validate (BH series + state-series). This keeps the
        # readable flag lenient (we still surface low-conf detections) while
        # giving callers a separate `valid_indian` boolean to filter on.
        alnum = sum(c.isalnum() for c in text)
        readable = alnum >= 4 and float(confidence) >= 0.20
        valid_indian, normalized = _validate_plate(text)
        return {
            "text": text,
            "normalized_text": normalized,
            "confidence": round(float(confidence), 4),
            "readable": bool(readable),
            "valid_indian": bool(valid_indian),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_images(src_dir: Path):
    if not src_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")
    found = []
    for p in sorted(src_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            found.append(p)
    return found


def draw_annotated(img: np.ndarray, detections) -> np.ndarray:
    """Draw bboxes + Awiros OCR text labels on a copy of the source image."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det["confidence"]
        ocr = det.get("ocr", {})
        text = ocr.get("text", "UNREADABLE") or "UNREADABLE"
        ocr_conf = ocr.get("confidence", 0.0)
        readable = ocr.get("readable", False)

        # Green if readable, red otherwise
        color = (0, 255, 0) if readable else (0, 0, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)

        lbl1 = f"YOLO {conf:.2f}"
        lbl2 = f"Awiros: {text} ({ocr_conf:.2f})"

        (tw1, th1), _ = cv2.getTextSize(lbl1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        (tw2, th2), _ = cv2.getTextSize(lbl2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tw = max(tw1, tw2)
        ty = max(y1 - 8, th1 + th2 + 8)

        cv2.rectangle(out, (x1, ty - th1 - th2 - 8), (x1 + tw + 8, ty + 2), color, -1)
        cv2.putText(out, lbl1, (x1 + 4, ty - th2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl2, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def detect_folder(
    src_dir: Path,
    out_dir: Path,
    yolo_model_path: str,
    awiros_dir: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    img_size: int = 640,
    device: str = "cpu",
):
    out_annotated = out_dir / "annotated"
    out_crops = out_dir / "crops"
    out_json = out_dir / "summary.json"
    out_annotated.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    print(f"[YOLO11] Loading model: {yolo_model_path}")
    yolo = YOLO(yolo_model_path)
    print(f"[YOLO11] Model class names: {yolo.names}")
    print(f"[YOLO11] Task: {yolo.task}")

    awiros = AwirosANPR(awiros_dir=awiros_dir, device=device)
    awiros.load()

    images = list_images(src_dir)
    if not images:
        print(f"[YOLO11] No images found in {src_dir}")
        return

    print(f"[YOLO11] Found {len(images)} image(s) in {src_dir}")
    print(f"[YOLO11] Output folder: {out_dir}")
    print(f"[OCR    ] Engine: Awiros ANPR-OCR (PP-OCRv5 SVTR_HGNet / CTC)")

    summary = {
        "detector": {
            "model_path": str(yolo_model_path),
            "model_class_names": {int(k): v for k, v in yolo.names.items()},
            "task": yolo.task,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "img_size": img_size,
            "framework": "ultralytics",
            "version": "yolo11",
        },
        "ocr": {
            "engine": "Awiros ANPR-OCR",
            "hf_repo": "https://huggingface.co/Awiros/anpr-ocr",
            "weights": str(awiros.weights_path),
            "dict": str(awiros.dict_path),
            "architecture": "PP-OCRv5 server rec / SVTR_HGNet (MultiHead: CTC + NRTR)",
            "framework": "paddlepaddle + ppocr",
            "device": device,
            "engine_description": (
                "License plate OCR done by the Awiros ANPR-OCR model "
                "(https://huggingface.co/Awiros/anpr-ocr). The model is "
                "PP-OCRv5 server recognition architecture (SVTR_HGNet, "
                "MultiHead with CTC + NRTR heads) and is run via PaddlePaddle "
                "with the ppocr package from PaddleOCR. The recognition is "
                "trained on Indian-style plates."
            ),
        },
        "source_folder": str(src_dir),
        "output_folder": str(out_dir),
        "run_started": datetime.now().isoformat(timespec="seconds"),
        "images": [],
        "totals": {
            "images": 0,
            "plates_detected": 0,
            "images_with_plates": 0,
            "plates_ocr_readable": 0,
            "plates_ocr_unreadable": 0,
        },
    }

    total_plates = 0
    images_with_plates = 0
    total_readable = 0
    total_unreadable = 0
    t_run = time.time()

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(images)}] SKIP (unreadable): {img_path.name}")
            summary["images"].append({"file": img_path.name, "error": "unreadable"})
            continue

        h, w = img.shape[:2]
        t0 = time.time()
        result = yolo.predict(img, conf=conf, iou=iou, imgsz=img_size, verbose=False)[0]
        dt_yolo = time.time() - t0

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf_val = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo.names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox_xyxy": xyxy,
                    "confidence": round(conf_val, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Save crops + run Awiros OCR on each
        crop_paths = []
        t_ocr_total = 0.0
        for n, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["bbox_xyxy"]
            # 6% padding around the plate bbox — matches the video pipeline
            # and engine.py. Gives OCR context beyond the tight crop edges.
            pw, ph = x2 - x1, y2 - y1
            pad_x, pad_y = int(pw * 0.06), int(ph * 0.06)
            x1c, y1c = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2c, y2c = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0:
                continue
            stem = img_path.stem
            # PNG: pixel-exact record of what the OCR engine saw
            # (no JPEG recompression of the evidence)
            crop_name = f"{stem}_plate{n}_{det['confidence']:.2f}.png"
            crop_path = out_crops / crop_name
            cv2.imwrite(str(crop_path), crop)
            det["crop_file"] = crop_name
            crop_paths.append(crop_name)

            # Run Awiros ANPR-OCR on the crop
            t1 = time.time()
            ocr = awiros.predict_crop(crop)
            dt_ocr = time.time() - t1
            t_ocr_total += dt_ocr
            det["ocr"] = {
                "text": ocr["text"],
                "confidence": ocr["confidence"],
                "readable": ocr["readable"],
                "engine": "Awiros ANPR-OCR",
                "inference_ms": round(dt_ocr * 1000, 1),
            }

        annotated = draw_annotated(img, detections)
        annotated_path = out_annotated / img_path.name
        cv2.imwrite(str(annotated_path), annotated)

        plates = len(detections)
        readable = sum(1 for d in detections if d.get("ocr", {}).get("readable"))
        unreadable = plates - readable
        total_plates += plates
        if plates > 0:
            images_with_plates += 1
        total_readable += readable
        total_unreadable += unreadable

        print(f"  [{idx}/{len(images)}] {img_path.name}  size={w}x{h}  "
              f"plates={plates}  readable={readable}  "
              f"({dt_yolo*1000:.0f}ms YOLO, {t_ocr_total*1000:.0f}ms OCR)")

        summary["images"].append({
            "file": img_path.name,
            "size": [w, h],
            "num_plates": plates,
            "num_ocr_readable": readable,
            "num_ocr_unreadable": unreadable,
            "inference_ms_yolo": round(dt_yolo * 1000, 1),
            "inference_ms_ocr_total": round(t_ocr_total * 1000, 1),
            "detections": detections,
            "annotated_file": img_path.name,
            "crop_files": crop_paths,
        })

    summary["totals"]["images"] = len(images)
    summary["totals"]["plates_detected"] = total_plates
    summary["totals"]["images_with_plates"] = images_with_plates
    summary["totals"]["plates_ocr_readable"] = total_readable
    summary["totals"]["plates_ocr_unreadable"] = total_unreadable
    summary["run_finished"] = datetime.now().isoformat(timespec="seconds")
    summary["total_seconds"] = round(time.time() - t_run, 2)

    out_json.write_text(json.dumps(summary, indent=2))

    print()
    print(f"[DONE] Run finished in {summary['total_seconds']}s")
    print(f"[DONE] {images_with_plates}/{len(images)} images had plates, "
          f"{total_plates} total plates detected by YOLO11")
    print(f"[DONE] OCR by Awiros ANPR-OCR: {total_readable}/{total_plates} readable, "
          f"{total_unreadable} unreadable")
    print(f"[DONE] Annotated -> {out_annotated}")
    print(f"[DONE] Crops     -> {out_crops}")
    print(f"[DONE] Summary   -> {out_json}")


def main():
    p = argparse.ArgumentParser(
        description="YOLO 11 plate detection + Awiros ANPR-OCR (PP-OCRv5)"
    )
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--yolo-model", default=DEFAULT_MODEL)
    p.add_argument("--awiros-dir", default=DEFAULT_AWIROS_DIR)
    p.add_argument("--out-parent", default=DEFAULT_OUT_PARENT)
    p.add_argument("--out-name", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = p.parse_args()

    out_parent = Path(args.out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    if args.out_name:
        out_dir = out_parent / args.out_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_parent / f"yolo11_awiros_ocr_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detect_folder(
        src_dir=Path(args.src),
        out_dir=out_dir,
        yolo_model_path=args.yolo_model,
        awiros_dir=Path(args.awiros_dir),
        conf=args.conf,
        iou=args.iou,
        img_size=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()