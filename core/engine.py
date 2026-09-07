import time
import base64
import threading
import urllib.request
import cv2
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

AWIROS_DIR = HERE / "awiros_anpr"
from core.plate_validate import validate_indian_plate as _validate_plate

def ensure_plate_model(name: str) -> Path:
    """Download plate model from Hugging Face if not present locally."""
    name_clean = name.strip().lower()
    
    if "yolo11s" in name_clean:
        filename = "yolo11s_plate.pt"
        hf_name = "license-plate-finetune-v1s.pt"
    elif "yolo11m" in name_clean:
        filename = "yolo11m_plate.pt"
        hf_name = "license-plate-finetune-v1m.pt"
    elif "yolo11l" in name_clean:
        filename = "yolo11l_plate.pt"
        hf_name = "license-plate-finetune-v1l.pt"
    else:
        # Default to nano
        filename = "yolo11_plate.pt"
        hf_name = "license-plate-finetune-v1n.pt"
        
    path = HERE / filename
    if not path.exists():
        url = f"https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/{hf_name}"
        print(f"[APP] Downloading plate model {hf_name} from Hugging Face...")
        try:
            urllib.request.urlretrieve(url, str(path))
            print(f"[APP] Downloaded {filename} successfully.")
        except Exception as e:
            print(f"[APP] Failed to download {hf_name}: {e}")
            # Fall back to yolo11_plate.pt if it exists
            fallback = HERE / "yolo11_plate.pt"
            if fallback.exists() and filename != "yolo11_plate.pt":
                print(f"[APP] Falling back to existing yolo11_plate.pt")
                return fallback
            raise e
    return path

class ANPREngine:
    """Loads and caches standard COCO + custom plate models on the fly.

    Prefers OpenVINO-exported models (``*_openvino_model/`` directories) for
    Intel GPU acceleration when available, falling back to PyTorch .pt files.
    """

    # Device string passed to ultralytics predict() for OpenVINO GPU.
    # Set to None if OpenVINO / GPU is unavailable.
    _OPENVINO_DEVICE = "intel:gpu"

    def __init__(self):
        self.yolo_models = {}
        self.awiros = None
        # Flask's dev server is threaded — lazy model loads are check-then-set
        # and must be serialized or two concurrent first requests each load
        # their own multi-hundred-MB copy (PaddleOCR double-init also leaks).
        self._cache_lock = threading.RLock()
        self._check_openvino()

    @classmethod
    def _check_openvino(cls):
        """Probe whether OpenVINO + GPU are available; disable if not."""
        if cls._OPENVINO_DEVICE is None:
            return
        try:
            from openvino import Core
            core = Core()
            if "GPU" not in core.available_devices:
                print("[APP] OpenVINO available but no GPU device — using CPU fallback.")
                cls._OPENVINO_DEVICE = None
            else:
                print(f"[APP] OpenVINO GPU detected — YOLO models will use GPU.")
        except ImportError:
            print("[APP] OpenVINO not installed — YOLO models will use PyTorch CPU.")
            cls._OPENVINO_DEVICE = None

    def _resolve_yolo_source(self, name: str):
        """Resolve a model name to its best on-disk source.

        Prefers the OpenVINO-exported directory (``<stem>_openvino_model/``)
        when GPU acceleration is available, otherwise the ``.pt`` weights.
        Downloads fine-tuned plate weights from Hugging Face if missing.
        No caching — see :meth:`_get_yolo_model` for the cached variant.
        """
        name_clean = name.strip()
        if "_plate" in name_clean or "plate" in name_clean:
            model_path = ensure_plate_model(name_clean)
            stem = model_path.stem  # e.g. "yolo11_plate"
            ov_dir = model_path.parent / f"{stem}_openvino_model"
        else:
            # Standard COCO model, e.g., yolo11s.pt
            filename = name_clean if name_clean.endswith(".pt") else f"{name_clean}.pt"
            local = HERE / filename
            model_path = local if local.exists() else filename
            stem = Path(filename).stem
            ov_dir = HERE / f"{stem}_openvino_model"
        if self._OPENVINO_DEVICE and ov_dir.is_dir():
            return ov_dir
        return model_path

    def _build_yolo(self, name: str):
        """Build a fresh, uncached YOLO instance (new tracker state)."""
        from ultralytics import YOLO

        src = self._resolve_yolo_source(name)
        print(f"[APP] Loading YOLO model: {Path(src).name}")
        return YOLO(str(src), task="detect")

    def _get_yolo_model(self, name: str):
        name_clean = name.strip()
        with self._cache_lock:
            if name_clean not in self.yolo_models:
                self.yolo_models[name_clean] = self._build_yolo(name_clean)
            return self.yolo_models[name_clean]

    def _ensure_awiros(self):
        with self._cache_lock:
            if self.awiros is None:
                print(f"[APP] Loading Awiros ANPR-OCR...")
                from detect_yolo11_awiros_ocr import AwirosANPR
                self.awiros = AwirosANPR(awiros_dir=AWIROS_DIR, device="cpu")
                self.awiros.load()
                print(f"[APP] Awiros loaded | dict: {self.awiros.dict_path.name}")

    def warmup(self):
        """Warm up default models."""
        self._get_yolo_model("yolo11s")
        self._get_yolo_model("yolo11_plate")
        self._ensure_awiros()

    def predict(self, img_bgr: np.ndarray, conf: float = 0.25, 
                model_coco_name: str = "yolo11s", model_plate_name: str = "yolo11_plate") -> dict:
        """Run COCO object detection + Plate detection + Awiros OCR."""
        yolo_coco = self._get_yolo_model(model_coco_name)
        yolo_plate = self._get_yolo_model(model_plate_name)
        self._ensure_awiros()

        h, w = img_bgr.shape[:2]

        # 1. Run COCO Detector (vehicles + persons)
        t0 = time.time()
        # Classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        _dev = self._OPENVINO_DEVICE
        coco_result = yolo_coco.predict(
            img_bgr, conf=conf, iou=0.45, imgsz=640, classes=[0, 1, 2, 3, 5, 7], verbose=False,
            **({"device": _dev} if _dev else {}),
        )[0]
        dt_coco = time.time() - t0

        # Parse COCO detections
        vehicles = []
        persons = []
        
        if coco_result.boxes is not None and len(coco_result.boxes) > 0:
            for box in coco_result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                c = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = yolo_coco.names.get(cls_id, str(cls_id))
                
                det_obj = {
                    "bbox_xyxy": xyxy,
                    "confidence": round(c, 4),
                    "class_id": cls_id,
                    "class_name": cls_name,
                }
                
                if cls_name == "person":
                    persons.append(det_obj)
                else:
                    det_obj["plate"] = None  # placeholder for association
                    vehicles.append(det_obj)

        # 2. Run Plate Detector
        t1 = time.time()
        plate_result = yolo_plate.predict(
            img_bgr, conf=conf, iou=0.45, imgsz=640, verbose=False,
            **({"device": self._OPENVINO_DEVICE} if self._OPENVINO_DEVICE else {}),
        )[0]
        dt_plate = time.time() - t1

        # Parse Plate detections
        plates = []
        t_ocr_total = 0.0
        
        if plate_result.boxes is not None and len(plate_result.boxes) > 0:
            for box in plate_result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                c = float(box.conf[0].cpu().numpy())
                
                # Crop with padding
                px1, py1, px2, py2 = xyxy
                pw, ph = px2 - px1, py2 - py1
                pad_x = int(pw * 0.06)
                pad_y = int(ph * 0.06)
                x1c = max(0, px1 - pad_x)
                y1c = max(0, py1 - pad_y)
                x2c = min(w, px2 + pad_x)
                y2c = min(h, py2 + pad_y)
                
                crop = img_bgr[y1c:y2c, x1c:x2c]
                
                if crop.size == 0:
                    ocr_res = {"text": "", "confidence": 0.0, "readable": False, "inference_ms": 0}
                else:
                    t_ocr_start = time.time()
                    ocr = self.awiros.predict_crop(crop)
                    dt_ocr = time.time() - t_ocr_start
                    t_ocr_total += dt_ocr
                    ocr_res = {
                        "text": ocr["text"],
                        "confidence": ocr["confidence"],
                        "readable": ocr["readable"],
                        "engine": "Awiros ANPR-OCR",
                        "inference_ms": round(dt_ocr * 1000, 1),
                    }
                
                plates.append({
                    "bbox_xyxy": xyxy,
                    "confidence": round(c, 4),
                    "ocr": ocr_res,
                    "crop_bgr": crop,
                })

        # 3. Associate Plates with Vehicles
        # Containment check: check if plate bbox is inside vehicle bbox
        for p in plates:
            px1, py1, px2, py2 = p["bbox_xyxy"]
            best_vehicle = None
            best_overlap = 0.0
            
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["bbox_xyxy"]
                ix1 = max(px1, vx1)
                iy1 = max(py1, vy1)
                ix2 = min(px2, vx2)
                iy2 = min(py2, vy2)
                
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                
                p_area = (px2 - px1) * (py2 - py1)
                if p_area <= 0:
                    continue
                
                overlap = inter_area / p_area
                if overlap > 0.70 and overlap > best_overlap:
                    best_overlap = overlap
                    best_vehicle = v
            
            if best_vehicle is not None:
                # Link plate to vehicle
                best_vehicle["plate"] = p

        # 4. Generate base64 crops and serialize payloads
        vehicles_payload = []
        for i, v in enumerate(vehicles):
            vx1, vy1, vx2, vy2 = v["bbox_xyxy"]
            v_crop = img_bgr[max(0, vy1):min(h, vy2), max(0, vx1):min(w, vx2)]
            
            plate_payload = None
            if v["plate"] is not None:
                p = v["plate"]
                plate_payload = {
                    "bbox_xyxy": p["bbox_xyxy"],
                    "confidence": p["confidence"],
                    "text": p["ocr"]["text"],
                    "ocr_confidence": p["ocr"]["confidence"],
                    "readable": p["ocr"]["readable"],
                    "crop_url": _crop_to_b64(p["crop_bgr"], PLATE_PREVIEW_MAX_W),
                }

            vehicles_payload.append({
                "id": i + 1,
                "class_name": v["class_name"],
                "confidence": v["confidence"],
                "bbox_xyxy": v["bbox_xyxy"],
                "crop_url": _crop_to_b64(v_crop, 240),
                "plate": plate_payload,
            })

        persons_payload = []
        for i, p in enumerate(persons):
            px1, py1, px2, py2 = p["bbox_xyxy"]
            p_crop = img_bgr[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
            persons_payload.append({
                "id": i + 1,
                "confidence": p["confidence"],
                "bbox_xyxy": p["bbox_xyxy"],
                "crop_url": _crop_to_b64(p_crop, 160),
            })

        plates_payload = []
        for i, p in enumerate(plates):
            plates_payload.append({
                "id": i + 1,
                "bbox_xyxy": p["bbox_xyxy"],
                "confidence": p["confidence"],
                "text": p["ocr"]["text"],
                "ocr_confidence": p["ocr"]["confidence"],
                "readable": p["ocr"]["readable"],
                "crop_url": _crop_to_b64(p["crop_bgr"], PLATE_PREVIEW_MAX_W),
            })

        # 5. Draw Annotated Result Image
        annotated_img = img_bgr.copy()
        # Draw vehicles (blue/cyan)
        for v in vehicles_payload:
            x1, y1, x2, y2 = v["bbox_xyxy"]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 128, 0), 2)
            lbl = f"{v['class_name']} {v['confidence']:.2f}"
            cv2.putText(annotated_img, lbl, (x1, max(y1 - 5, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1, cv2.LINE_AA)

        # Draw persons (amber)
        for p in persons_payload:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(annotated_img, "person", (x1, max(y1 - 5, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # Draw plates (green/red based on readability)
        for p in plates_payload:
            x1, y1, x2, y2 = p["bbox_xyxy"]
            color = (0, 255, 0) if p["readable"] else (0, 0, 255)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            
            lbl = f"Plate: {p['text'] or '?'}"
            cv2.putText(annotated_img, lbl, (x1, max(y2 + 15, h - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        return {
            "size": [w, h],
            "inference_ms_yolo_coco": round(dt_coco * 1000, 1),
            "inference_ms_yolo_plate": round(dt_plate * 1000, 1),
            "inference_ms_ocr_total": round(t_ocr_total * 1000, 1),
            "num_plates": len(plates_payload),
            "num_vehicles": len(vehicles_payload),
            "num_persons": len(persons_payload),
            "vehicles": vehicles_payload,
            "persons": persons_payload,
            "plates": plates_payload,
            "annotated_jpeg_b64": _img_to_b64(annotated_img, ".jpg"),
        }

engine = ANPREngine()

def _img_to_b64(img: np.ndarray, ext: str = ".jpg") -> str:
    """Encode OpenCV BGR image to a base64 data-URI."""
    if img is None or img.size == 0:
        return ""
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

# Max width for base64 plate-crop previews embedded in JSON payloads.
# Aligned with core/live.py's _encode_crop_jpeg (max_w=400) so the image-API
# UI shows enough detail for a human to verify what the OCR engine saw.
PLATE_PREVIEW_MAX_W = 400


def _crop_to_b64(crop: np.ndarray, max_w: int = 320) -> str:
    """Resize crop if width > max_w to keep JSON payload lightweight."""
    if crop is None or crop.size == 0:
        return ""
    h, w = crop.shape[:2]
    if w > max_w:
        h_new = int(h * (max_w / w))
        crop = cv2.resize(crop, (max_w, h_new))
    return _img_to_b64(crop, ".jpg")

def _read_image(path: str) -> np.ndarray:
    """Unicode-safe image read for Windows paths."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {path}")
    return img

def _decode_b64(data_uri: str) -> np.ndarray:
    """Decode a base64 image (data URI or raw) back to a BGR numpy array."""
    if not data_uri:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw = base64.b64decode(data_uri)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
