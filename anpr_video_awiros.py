"""
Video ANPR using YOLO11 COCO (for vehicles + persons) + YOLO11 plate detector + Awiros ANPR-OCR.
Tracks objects across frames and aggregates plate OCR results under the correct vehicle track.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import shutil
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# PaddleOCR + protobuf workaround (must be set BEFORE paddle imports).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core.engine import engine

log = logging.getLogger("anpr_video_awiros")
# Configure root logging only for standalone CLI runs. When imported by the
# Flask app, defer to whatever the app/entry point has configured instead of
# hijacking root logging at import time.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
IMGSZ = 640
DET_CONF = 0.25          # YOLO plate detection confidence threshold
# NOTE: the old OCR_MIN_HEIGHT tunable is now enforced where it belongs —
# AwirosANPR.predict_crop() short-circuits crops shorter than MIN_CROP_HEIGHT
# (detect_yolo11_awiros_ocr.py), so video/live/image-API paths share one policy.

# Duplicate-plate-track merging (batch pipeline, applied after the frame loop).
# ByteTrack can hand out two IDs to one physical plate (ID switch after an
# occlusion, or the track aged out during a detection gap and the plate got
# re-detected as new). Tracks whose character-voted text agrees are folded
# back together under safety gates.
MERGE_MIN_CONF = 0.55    # BOTH tracks must vote a VALID Indian plate ≥ this conf
MERGE_NEAR_FRAC = 0.12   # "ever spatially adjacent": min bbox-center distance
                         # below this fraction of the frame diagonal

# ---------------------------------------------------------------------------
# ByteTracker wrapper (ultralytics BYTETracker, SimpleTracker-compatible API)
# ---------------------------------------------------------------------------
class ByteTracker:
    """ByteTrack via ultralytics, exposing the SimpleTracker interface.

    Input:  update(detections) where detections = [(bbox_xyxy, conf), ...]
    Output: [(tid, bbox_xyxy, conf), ...] with persistent track IDs across frames.
    """

    class _ResultsView:
        def __init__(self, xyxy, conf, cls_):
            xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
            self.xyxy = xyxy
            self.xywh = self._xyxy_to_xywh(xyxy)
            self.conf = np.asarray(conf, dtype=float)
            self.cls  = np.asarray(cls_, dtype=int)
        def __getattr__(self, name):
            raise AttributeError(
                f"_ResultsView has no attribute '{name}'."
            )
        @staticmethod
        def _xyxy_to_xywh(xyxy):
            x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            return np.stack([cx, cy, w, h], axis=1)
        def __len__(self): return len(self.conf)
        def __getitem__(self, key):
            if isinstance(key, np.ndarray) and key.dtype == bool:
                return ByteTracker._ResultsView(self.xyxy[key], self.conf[key], self.cls[key])
            raise TypeError("BYTETracker._ResultsView only supports boolean-mask indexing")

    def __init__(self, frame_rate: int = 30, max_age: int = 30,
                 high_thresh: float = 0.5, low_thresh: float = 0.10,
                 match_thresh: float = 0.8):
        from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETracker
        args = argparse.Namespace()
        args.tracker_yaml        = ''
        args.track_high_thresh   = high_thresh
        args.track_low_thresh    = low_thresh
        args.new_track_thresh    = high_thresh + 0.1
        args.track_buffer        = max_age
        args.match_thresh        = match_thresh
        args.fuse_score          = False
        args.min_box_area        = 10
        args.min_consecutive_frames = 1
        self._bt = _BYTETracker(args=args)
        self.max_age = max_age

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / max(ua, 1)

    def update(self, detections: list) -> list:
        if not detections:
            # Still step the internal tracker on empty frames so existing
            # tracks age out (track_buffer) instead of living forever and
            # re-matching stale boxes on the next detection.
            empty = ByteTracker._ResultsView(
                np.zeros((0, 4), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=int),
            )
            self._bt.update(empty)
            return []
        xyxy = np.array([[*d[0]] for d in detections], dtype=float)
        conf = np.array([d[1] for d in detections], dtype=float)
        cls  = np.zeros(len(detections), dtype=int)
        results = self._ResultsView(xyxy, conf, cls)
        tracks = self._bt.update(results)
        out = []
        for t in tracks:
            tx1, ty1, tx2, ty2, tid = float(t[0]), float(t[1]), float(t[2]), float(t[3]), int(t[4])
            # Match the tracked box back to the detection it came from.
            # Skip the track entirely when nothing overlaps — blindly
            # attributing detections[0] recorded wrong boxes/confs.
            best_iou, best_idx = 0.0, -1
            for i, (bbox, _) in enumerate(detections):
                iou = self._iou_xyxy((tx1, ty1, tx2, ty2), bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx < 0 or best_iou <= 0.0:
                continue
            out.append((tid, tuple(detections[best_idx][0]), detections[best_idx][1]))
        return out

# ---------------------------------------------------------------------------
# Character-position voting
# ---------------------------------------------------------------------------
def vote_track_text(reads: list) -> dict:
    if not reads:
        return {"text": "", "conf": 0.0, "valid": False, "votes": {}}

    # Length = most-common read length
    len_counter = Counter(len(t) for t, _ in reads if t)
    if not len_counter:
        return {"text": "", "conf": 0.0, "valid": False, "votes": {}}
    best_len, _ = len_counter.most_common(1)[0]

    pos_buckets = defaultdict(lambda: defaultdict(float))
    for text, conf in reads:
        if not text:
            continue
        text = text[:best_len] if len(text) >= best_len else text
        for i, ch in enumerate(text):
            pos_buckets[i][ch] += conf

    chars = []
    confs = []
    votes_per_pos = {}
    for i in range(best_len):
        bucket = pos_buckets.get(i, {})
        if not bucket:
            chars.append("?")
            confs.append(0.0)
            votes_per_pos[i] = {}
            continue
        best_ch, best_score = max(bucket.items(), key=lambda kv: kv[1])
        chars.append(best_ch)
        total = sum(bucket.values())
        confs.append(round(best_score / total, 4) if total > 0 else 0.0)
        votes_per_pos[i] = {ch: round(s, 3) for ch, s in bucket.items()}

    text = "".join(chars)
    real = [c for c in confs if c > 0]
    final_conf = round(sum(real) / len(real), 4) if real else 0.0
    # Use regex grammar for "valid Indian plate" instead of the old hardcoded
    # length-6 / conf-0.40 heuristic. We only require at least 1 voting read
    # per character position (no "?"s) and the format to match.
    from core.plate_validate import validate_indian_plate as _vp
    ok_format, normalized = _vp(text)
    valid = ok_format and "?" not in text and final_conf >= 0.30

    return {"text": text, "normalized": normalized, "conf": final_conf, "valid": valid, "votes": votes_per_pos}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pad_bbox(x1, y1, x2, y2, W, H, pad: float = 0.06):
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad), int(bh * pad)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(W, x2 + px),
        min(H, y2 + py),
    )

# ---------------------------------------------------------------------------
# Duplicate plate-track merging (post-loop de-duplication)
# ---------------------------------------------------------------------------
def _sample_seq(seq, cap: int = 150):
    """Evenly down-sample a list to at most `cap` items (bounds O(n·m)
    proximity checks on long tracks without changing the answer materially)."""
    if len(seq) <= cap:
        return seq
    step = len(seq) / cap
    return [seq[int(i * step)] for i in range(cap)]


def _merge_two_plate_tracks(dst: dict, src: dict) -> None:
    """Fold plate-history `src` into `dst`.

    Unions the per-frame records (reads / yolo confs / bboxes / crop files)
    sorted by processed frame index, and keeps whichever track produced the
    better composite best-crop score — the same score formula used in-loop,
    so the stored values are directly comparable.
    """
    rows = list(zip(dst["frames"], dst["reads"], dst["confs"],
                    dst["bboxes"], dst["crop_files"]))
    rows += list(zip(src["frames"], src["reads"], src["confs"],
                     src["bboxes"], src["crop_files"]))
    rows.sort(key=lambda r: r[0])
    dst["frames"]     = [r[0] for r in rows]
    dst["reads"]      = [r[1] for r in rows]
    dst["confs"]      = [r[2] for r in rows]
    dst["bboxes"]     = [r[3] for r in rows]
    dst["crop_files"] = [r[4] for r in rows]

    if float(src.get("best_score") or 0.0) > float(dst.get("best_score") or 0.0):
        dst["best_frame"] = src["best_frame"]
        dst["best_bbox"] = src["best_bbox"]
        dst["best_text"] = src["best_text"]
        dst["best_conf"] = src["best_conf"]
    dst["best_score"] = max(float(dst.get("best_score") or 0.0),
                            float(src.get("best_score") or 0.0))


def _tracks_compatible(a: dict, b: dict, near_px: float) -> bool:
    """Time/space gate between two same-text plate tracks.

    True when their sightings never overlap in time (the car left and came
    back) or when they were ever spatially adjacent (the classic tracker
    ID-switch signature). Overlapping-in-time AND never adjacent smells like
    two different vehicles wearing the same number — those stay separate.
    """
    if b["frames"][0] > a["frames"][-1]:      # time-disjoint → safe to merge
        return True
    # Windows overlap: merge only if the tracks were ever close.
    for ba in _sample_seq(a["bboxes"]):
        ax, ay = (ba[0] + ba[2]) / 2.0, (ba[1] + ba[3]) / 2.0
        for bb in _sample_seq(b["bboxes"]):
            bx, by = (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
            if math.hypot(ax - bx, ay - by) <= near_px:
                return True
    return False


def _merge_duplicate_plate_tracks(
    plate_history: defaultdict,
    plate_to_vehicle_map: defaultdict,
    frame_diag: float,
    min_conf: float = MERGE_MIN_CONF,
    near_frac: float = MERGE_NEAR_FRAC,
) -> tuple:
    """De-duplicate plate tracks that converged on the same plate number.

    Runs ONCE after the frame loop, before results are compiled. Tracks whose
    character-voted text normalizes identically — and where BOTH pass the
    valid-Indian-plate grammar at >= min_conf confidence — are merged when
    they are either time-disjoint or were ever spatially adjacent. The
    survivor keeps the lowest original track id and inherits the absorbed
    tracks' vehicle votes.

    Returns:
      merge_map  {survivor_tid: [absorbed_tids]} — merges applied
      dup_groups {text: [separate tids]}         — same-text tracks NOT merged
                                                   (overlap in time, never
                                                   adjacent → possible cloned
                                                   plates; surfaced instead of
                                                   silently hidden)
    """
    votes = {tid: vote_track_text(ph["reads"])
             for tid, ph in plate_history.items()}

    # Group by canonical (normalized) text; garbage/low-conf reads never group.
    groups = defaultdict(list)
    for tid, v in votes.items():
        if not v["valid"] or float(v["conf"]) < min_conf:
            continue
        key = (v.get("normalized") or v["text"]).strip().upper()
        if key:
            groups[key].append(tid)

    near_px = near_frac * frame_diag
    merge_map: dict = {}
    dup_groups: dict = {}

    for key, tids in groups.items():
        if len(tids) < 2:
            continue
        # Chronological greedy clustering under the compatibility gate: a
        # track joins the first cluster it is compatible with ANY member of.
        tids.sort(key=lambda t: plate_history[t]["frames"][0])
        clusters = [[tids[0]]]
        for tid in tids[1:]:
            ph = plate_history[tid]
            target = None
            for cl in clusters:
                if any(_tracks_compatible(plate_history[m], ph, near_px)
                       for m in cl):
                    target = cl
                    break
            if target is not None:
                target.append(tid)
            else:
                clusters.append([tid])

        if len(clusters) > 1:
            # Same text survived as distinct co-temporal clusters — flag it.
            dup_groups[key] = [t for cl in clusters for t in cl]

        for cl in clusters:
            if len(cl) < 2:
                continue
            cl.sort()
            survivor = cl[0]
            for m in cl[1:]:
                _merge_two_plate_tracks(plate_history[survivor],
                                        plate_history[m])
                del plate_history[m]
                # Absorbed track's vehicle votes move to the survivor so the
                # downstream majority vote sees the full history.
                for veh, cnt in plate_to_vehicle_map.pop(m, {}).items():
                    plate_to_vehicle_map[survivor][veh] += cnt
            merge_map[survivor] = cl[1:]

    return merge_map, dup_groups

# ---------------------------------------------------------------------------
# Cancel flag
# ---------------------------------------------------------------------------
# Module-level cancel flag — set by the /api/cancel_video endpoint. Checked
# once per frame so a long video can be aborted mid-flight without killing
# the Flask worker thread.
# NOTE: this is a single global flag, so it assumes one video processed at a
# time (the UI enforces this). Cancelling also cancels any other in-flight
# video job; _reset_cancel() at the start of each run clears stale flags.
_cancel_flag = False

def request_cancel():
    """Signal the running process_video() to stop after the current frame."""
    global _cancel_flag
    _cancel_flag = True

def _is_cancelled() -> bool:
    return _cancel_flag

def _reset_cancel():
    global _cancel_flag
    _cancel_flag = False


# ---------------------------------------------------------------------------
# Sharpness / clarity metric
# ---------------------------------------------------------------------------
def _crop_sharpness(crop_bgr) -> float:
    """Blur metric: Laplacian variance on a fixed-size grayscale version.

    Resizing to a canonical 100x40 normalises the metric across crops of
    different dimensions so it reflects *intrinsic* blur rather than just
    pixel count. Higher = sharper.  Returns 0.0 for degenerate inputs.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 40), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# Main per-video pipeline
# ---------------------------------------------------------------------------
def process_video(
    video_path: Path,
    out_dir: Path,
    stride: int = 2,
    max_frames: int = None,
    write_video: bool = True,
    yolo_model: str = None,         # Plate detector model name
    yolo_coco_model: str = None,    # COCO detector model name
    device: str = "cpu",
) -> dict:
    """Run dual-YOLO tracking + Awiros OCR on a video, associate plates, and vote."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"
    best_frames_dir = out_dir / "best_frames"
    
    frames_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    best_frames_dir.mkdir(exist_ok=True)

    # ── Load models via core.engine (handles downloads & caching) ──
    plate_model_name = yolo_model or "yolo11_plate"
    coco_model_name = yolo_coco_model or "yolo11s"
    
    yolo_plate = engine._get_yolo_model(plate_model_name)
    yolo_coco = engine._get_yolo_model(coco_model_name)
    engine._ensure_awiros()
    awiros = engine.awiros

    # Use OpenVINO GPU device if available, otherwise the caller's device.
    ov_dev = engine._OPENVINO_DEVICE
    infer_device = ov_dev or device

    # Force reset tracking state for a new video
    if hasattr(yolo_coco, "predictor") and yolo_coco.predictor is not None:
         yolo_coco.predictor.trackers = None

    # ── Open video ──
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Video: %s %dx%d fps=%.2f frames=%d stride=%d",
             video_path, W, H, fps, total, stride)

    out_video = None
    if write_video:
        out_video_path = out_dir / "annotated.mp4"
        # H.264 (avc1) plays in browser <video> tags; the mp4v fallback does
        # not decode in Chrome/Firefox/Edge (download still works). Use avc1
        # when this OpenCV build can open it, else fall back.
        for fourcc_tag in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(out_video_path), cv2.VideoWriter_fourcc(*fourcc_tag), fps, (W, H)
            )
            if writer.isOpened():
                out_video = writer
                break
            writer.release()

    # Plate tracker
    plate_tracker = ByteTracker(frame_rate=int(fps), max_age=max(30, int(fps * 2)))

    # Track histories
    vehicle_history = defaultdict(lambda: {
        "class_name": "",
        "frames": [],
        "bboxes": [],
        "confs": [],
        "best_frame": None,
        "best_bbox": None,
        "best_crop_file": "",
        "best_annotated_file": "",
        "best_score": 0.0,
        "crop_files": [],
    })
    
    person_history = defaultdict(lambda: {
        "frames": [],
        "bboxes": [],
        "confs": [],
        "best_frame": None,
        "best_bbox": None,
        "best_crop_file": "",
        "best_score": 0.0,
        "crop_files": [],
    })

    plate_history = defaultdict(lambda: {
        "reads": [],       # list[(text, conf)]
        "frames": [],      # list[int]
        "bboxes": [],      # list[(x1,y1,x2,y2)]
        "confs": [],       # list[float]
        "crop_files": [],
        "best_frame": None,
        "best_bbox": None,
        "best_text": "",
        "best_conf": 0.0,
        "best_crop_file": "",
        "best_score": 0.0,
    })

    # Association mapping: plate_tid -> Counter(vehicle_track_ids)
    plate_to_vehicle_map = defaultdict(lambda: Counter())

    n_proc = 0
    t0 = time.time()
    last_log_t = t0
    _reset_cancel()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Honor limits BEFORE counting the frame as processed — n_proc must
            # equal the number of frames actually pulled through the pipeline.
            if _is_cancelled():
                log.info("Cancel requested — stopping at frame %d", n_proc)
                break
            if max_frames is not None and n_proc >= max_frames:
                break
            n_proc += 1
            if stride > 1 and (n_proc - 1) % stride != 0:
                if out_video is not None:
                    out_video.write(frame)
                continue

            # ── 1. Run COCO Tracker (vehicles + persons) ──
            # Classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
            coco_results = yolo_coco.track(
                frame,
                persist=True,
                classes=[0, 1, 2, 3, 5, 7],
                conf=DET_CONF,
                tracker="bytetrack.yaml",
                verbose=False,
                device=infer_device
            )[0]

            coco_dets = []
            if coco_results.boxes is not None and coco_results.boxes.id is not None:
                for box in coco_results.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    tid = int(box.id[0].cpu().numpy())
                    c = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = yolo_coco.names.get(cls_id, str(cls_id))
                    coco_dets.append({
                        "track_id": tid,
                        "bbox_xyxy": xyxy,
                        "confidence": c,
                        "class_name": cls_name,
                    })

            # Save COCO tracks history
            for d in coco_dets:
                tid = d["track_id"]
                bbox = d["bbox_xyxy"]
                conf = d["confidence"]
                cls_name = d["class_name"]
            
                x1, y1, x2, y2 = bbox
                bx1, by1 = max(0, x1), max(0, y1)
                bx2, by2 = min(W, x2), min(H, y2)
                crop = frame[by1:by2, bx1:bx2]
            
                if crop.size == 0:
                    continue

                if cls_name == "person":
                    hist = person_history[tid]
                    hist["frames"].append(n_proc)
                    hist["bboxes"].append(bbox)
                    hist["confs"].append(conf)
                
                    # Best crop = highest clarity (sharpness * area), not just area.
                    sharp = _crop_sharpness(crop)
                    area = max(1, (x2 - x1) * (y2 - y1))
                    score = sharp * area
                    if hist["best_bbox"] is None or score > float(hist.get("best_score") or 0.0) * 1.05:
                        hist["best_frame"] = n_proc
                        hist["best_bbox"] = bbox
                        hist["best_score"] = score
                        # Save crop file
                        p_crop_name = f"person{tid:03d}_best.jpg"
                        cv2.imwrite(str(best_frames_dir / p_crop_name), crop)
                        hist["best_crop_file"] = p_crop_name
                else:
                    hist = vehicle_history[tid]
                    hist["class_name"] = cls_name
                    hist["frames"].append(n_proc)
                    hist["bboxes"].append(bbox)
                    hist["confs"].append(conf)
                
                    # Best crop = highest clarity (sharpness * area), not just area.
                    sharp = _crop_sharpness(crop)
                    area = max(1, (x2 - x1) * (y2 - y1))
                    score = sharp * area
                    if hist["best_bbox"] is None or score > float(hist.get("best_score") or 0.0) * 1.05:
                        hist["best_frame"] = n_proc
                        hist["best_bbox"] = bbox
                        hist["best_score"] = score
                        # Save crop file
                        v_crop_name = f"vehicle{tid:03d}_best.jpg"
                        cv2.imwrite(str(best_frames_dir / v_crop_name), crop)
                        hist["best_crop_file"] = v_crop_name

            # ── 2. Run Plate Detector ──
            plate_results = yolo_plate.predict(
                frame, conf=DET_CONF, iou=0.45, imgsz=IMGSZ, verbose=False, device=infer_device
            )[0]
            raw_plate_dets = []
            if plate_results.boxes is not None and len(plate_results.boxes) > 0:
                for box in plate_results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    yconf = float(box.conf[0].cpu().item())
                
                    # Filter weird aspect ratios
                    bw, bh = x2 - x1, y2 - y1
                    if bh <= 0 or bw <= 0: continue
                    ar = bw / bh
                    if ar < 1.5 or ar > 6.5: continue
                    if bh < 16 or bh > 320: continue
                    raw_plate_dets.append(((int(x1), int(y1), int(x2), int(y2)), yconf))

            # ── 3. Track Plates ──
            tracked_plates = plate_tracker.update(raw_plate_dets)

            # ── 4. OCR on Plate Crops & BBox containment checks ──
            frame_annotations = [] # for rendering annotated video
        
            for plate_tid, bbox, yconf in tracked_plates:
                x1, y1, x2, y2 = bbox
                x1c, y1c, x2c, y2c = _pad_bbox(x1, y1, x2, y2, W, H, pad=0.06)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                ocr = awiros.predict_crop(crop)
                text = ocr["text"] or ""
                conf = ocr["confidence"]

                hist = plate_history[plate_tid]
                hist["reads"].append((text, conf))
                hist["frames"].append(n_proc)
                hist["bboxes"].append(bbox)
                hist["confs"].append(yconf)
            
                # Save crop frame-by-frame. PNG keeps a pixel-exact audit
                # record of exactly what the OCR engine saw (JPEG would
                # recompress the evidence); plate crops are small so the
                # size penalty is negligible.
                crop_fname = f"plate{plate_tid:03d}_f{n_proc:06d}.png"
                cv2.imwrite(str(crops_dir / crop_fname), crop)
                hist["crop_files"].append(crop_fname)

                # Pick best crop — composite clarity score so the saved image
                # genuinely looks like the clearest frame of this plate track.
                #   sharpness : normalised Laplacian variance (primary "looks clear")
                #   area      : bbox pixel area, log-weighted (more pixels = detail)
                #   ocr_bonus : confirms characters are legible (text frames get ~2x
                #               the weight of empty-text frames, which can still win
                #               as a fallback when no readable frame ever appears)
                prefer = False
                best_score = float(hist.get("best_score") or 0.0)
                best_text = hist.get("best_text") or ""

                sharp = _crop_sharpness(crop)
                area = max(1, (x2 - x1) * (y2 - y1))
                ocr_bonus = (0.5 + 0.5 * conf) if text else 0.25
                score = sharp * (1.0 + math.log(area)) * ocr_bonus

                if hist["best_bbox"] is None:
                    # First detection always seeds the best so we have a fallback.
                    prefer = True
                elif text and not best_text:
                    # We finally got a non-empty OCR where the current best is empty.
                    prefer = True
                elif score > best_score * 1.05:
                    # 5% hysteresis — avoids flicker between near-equal frames.
                    prefer = True

                if prefer:
                    hist["best_frame"] = n_proc
                    hist["best_bbox"] = bbox
                    hist["best_text"] = text
                    hist["best_conf"] = conf
                    hist["best_score"] = score
                
                    # Save best annotated frame for this plate track
                    best_ann = frame.copy()
                    ann_color = (0, 255, 0) if (text and conf >= 0.20) else (0, 0, 255)
                    cv2.rectangle(best_ann, (x1, y1), (x2, y2), ann_color, 3)
                    ann_label = f"ID{plate_tid} {text or '?'} Conf={conf:.2f}"
                    cv2.putText(best_ann, ann_label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    best_ann_file = f"plate{plate_tid:03d}_best_annotated_f{n_proc:06d}.jpg"
                    cv2.imwrite(str(best_frames_dir / best_ann_file), best_ann)
                    hist["best_annotated_file"] = best_ann_file

                # ── 5. Associate Plate track with Vehicle track ──
                # Look for tracked vehicles in the same frame
                best_vehicle_id = None
                best_overlap = 0.0
            
                for v in coco_dets:
                    if v["class_name"] == "person":
                        continue
                    vx1, vy1, vx2, vy2 = v["bbox_xyxy"]
                    ix1 = max(x1, vx1)
                    iy1 = max(y1, vy1)
                    ix2 = min(x2, vx2)
                    iy2 = min(y2, vy2)
                
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter_area = iw * ih
                
                    p_area = (x2 - x1) * (y2 - y1)
                    if p_area <= 0: continue
                
                    overlap = inter_area / p_area
                    if overlap > 0.70 and overlap > best_overlap:
                        best_overlap = overlap
                        best_vehicle_id = v["track_id"]
                    
                if best_vehicle_id is not None:
                    plate_to_vehicle_map[plate_tid][best_vehicle_id] += 1
                
                frame_annotations.append((plate_tid, bbox, text, conf, best_vehicle_id))

            # ── 6. Render Frame and write video ──
            ann_frame = frame.copy()
        
            # Draw vehicle bboxes (blue)
            for v in coco_dets:
                tx1, ty1, tx2, ty2 = v["bbox_xyxy"]
                if v["class_name"] == "person":
                    cv2.rectangle(ann_frame, (tx1, ty1), (tx2, ty2), (0, 165, 255), 2)
                    cv2.putText(ann_frame, f"person #{v['track_id']}", (tx1, max(ty1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(ann_frame, (tx1, ty1), (tx2, ty2), (255, 128, 0), 2)
                    cv2.putText(ann_frame, f"{v['class_name']} #{v['track_id']}", (tx1, max(ty1 - 5, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1, cv2.LINE_AA)
                            
            # Draw plate bboxes (green/red)
            for plate_tid, bbox, text, conf, v_tid in frame_annotations:
                tx1, ty1, tx2, ty2 = bbox
                color = (0, 255, 0) if (text and conf >= 0.20) else (0, 0, 255)
                cv2.rectangle(ann_frame, (tx1, ty1), (tx2, ty2), color, 3)
            
                lbl = f"Plate #{plate_tid}: {text or '?'}"
                if v_tid is not None:
                    lbl += f" (Veh #{v_tid})"
                cv2.putText(ann_frame, lbl, (tx1, max(ty2 + 15, H - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

            cv2.putText(ann_frame, f"frame {n_proc}/{total}  YOLO11 + Awiros ANPR-OCR",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(ann_frame, f"frame {n_proc}/{total}  YOLO11 + Awiros ANPR-OCR",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            if out_video is not None:
                out_video.write(ann_frame)
            
            # Save one annotated frame per second for timeline previews
            if n_proc % max(1, int(fps)) == 0 or n_proc == 1:
                cv2.imwrite(str(frames_dir / f"frame_{n_proc:06d}.jpg"), ann_frame)

            if time.time() - last_log_t > 3.0:
                elapsed = time.time() - t0
                log.info("  frame %d/%d  active_tracks=%d  (%.1fs elapsed)",
                         n_proc, total, len(coco_dets), elapsed)
                last_log_t = time.time()

    finally:
        cap.release()
        if out_video is not None:
            out_video.release()

    # ── 7. Compile unified tracking results and run Voting ──
    
    # De-duplicate plate tracks that converged on the same number (tracker
    # ID switches / re-detections). Runs BEFORE the vehicle-majority vote so
    # absorbed tracks transfer their vehicle links to the survivor.
    merge_map, dup_groups = _merge_duplicate_plate_tracks(
        plate_history, plate_to_vehicle_map, math.hypot(W, H),
    )
    dup_of = {tid: key for key, tids in dup_groups.items() for tid in tids}

    # Map plate tracks to vehicle tracks using majority vote
    vehicle_to_plate_map = {}
    for plate_tid, counter in plate_to_vehicle_map.items():
        if not counter:
            continue
        best_veh_id, _ = counter.most_common(1)[0]
        vehicle_to_plate_map[best_veh_id] = plate_tid

    tracks_summary = []
    
    # Process Vehicle tracks
    for tid in sorted(vehicle_history.keys()):
        vh = vehicle_history[tid]
        if not vh["frames"]:
            continue
            
        # Check if this vehicle has an associated plate track
        plate_tid = vehicle_to_plate_map.get(tid)
        
        # Copy vehicle best crop file
        best_crop_file = ""
        if vh["best_crop_file"]:
            src = best_frames_dir / vh["best_crop_file"]
            if src.exists():
                best_crop_file = f"vehicle{tid:03d}_best.jpg"
                dst = best_frames_dir / best_crop_file
                if src.resolve() != dst.resolve():
                    shutil.copy2(str(src), str(dst))

        if plate_tid is not None:
            ph = plate_history[plate_tid]
            voted = vote_track_text(ph["reads"])
            
            # Format per-frame reads
            per_frame_reads = []
            for i, fnum in enumerate(ph["frames"]):
                text, ocr_conf = ph["reads"][i]
                per_frame_reads.append({
                    "frame": int(fnum),
                    "text": text,
                    "ocr_conf": round(float(ocr_conf), 4),
                    "yolo_conf": round(float(ph["confs"][i]), 4),
                    "bbox": [int(v) for v in ph["bboxes"][i]],
                    "crop_file": ph["crop_files"][i] if i < len(ph["crop_files"]) else "",
                })
                
            # Copy best plate crop to best_frames/ directory
            best_plate_crop = ""
            if ph["best_crop_file"] == "" and ph["crop_files"]:
                best_frame_num = int(ph["best_frame"])
                for i, fnum in enumerate(ph["frames"]):
                    if int(fnum) == best_frame_num:
                        src_crop = crops_dir / ph["crop_files"][i]
                        if src_crop.exists():
                            # Keep the source extension — crops are PNG now
                            best_plate_crop = f"plate{plate_tid:03d}_best_f{best_frame_num:06d}{src_crop.suffix}"
                            shutil.copy2(str(src_crop), str(best_frames_dir / best_plate_crop))
                        break
            
            tracks_summary.append({
                "track_id": int(tid),
                "class_name": vh["class_name"],
                "n_frames": len(vh["frames"]),
                "first_seen": int(vh["frames"][0]),
                "last_seen": int(vh["frames"][-1]),
                "frames": vh["frames"][:50],
                "all_frames": vh["frames"],
                "per_frame_reads": per_frame_reads,
                "best_frame": int(ph["best_frame"] or vh["frames"][0]),
                "best_text": ph["best_text"] or "",
                "best_conf": round(float(ph["best_conf"]), 4),
                "best_crop_file": best_plate_crop,
                "best_annotated_file": ph.get("best_annotated_file", ""),
                "final_text": voted["text"],
                "final_conf": voted["conf"],
                "valid_indian": voted["valid"],
                "avg_yolo_conf": round(sum(vh["confs"]) / len(vh["confs"]), 4),
                "n_unique_reads": len(set(t for t, _ in ph["reads"] if t)),
                "votes_per_pos": voted["votes"],
                "vehicle_crop_file": best_crop_file,
                "merged_from": sorted(merge_map.get(plate_tid, [])),
                "duplicate_text_group": dup_of.get(plate_tid, ""),
            })
        else:
            # Vehicle with no plate associated
            tracks_summary.append({
                "track_id": int(tid),
                "class_name": vh["class_name"],
                "n_frames": len(vh["frames"]),
                "first_seen": int(vh["frames"][0]),
                "last_seen": int(vh["frames"][-1]),
                "frames": vh["frames"][:50],
                "all_frames": vh["frames"],
                "per_frame_reads": [],
                "best_frame": int(vh["best_frame"]),
                "best_text": "",
                "best_conf": 0.0,
                "best_crop_file": "",
                "best_annotated_file": "",
                "final_text": "",
                "final_conf": 0.0,
                "valid_indian": False,
                "avg_yolo_conf": round(sum(vh["confs"]) / len(vh["confs"]), 4),
                "n_unique_reads": 0,
                "votes_per_pos": {},
                "vehicle_crop_file": best_crop_file,
            })

    # Process Person tracks
    for tid in sorted(person_history.keys()):
        ph = person_history[tid]
        if not ph["frames"]:
            continue
            
        best_crop_file = ""
        if ph["best_crop_file"]:
            src = best_frames_dir / ph["best_crop_file"]
            if src.exists():
                best_crop_file = f"person{tid:03d}_best.jpg"
                dst = best_frames_dir / best_crop_file
                if src.resolve() != dst.resolve():
                    shutil.copy2(str(src), str(dst))
                
        tracks_summary.append({
            "track_id": int(tid),
            "class_name": "person",
            "n_frames": len(ph["frames"]),
            "first_seen": int(ph["frames"][0]),
            "last_seen": int(ph["frames"][-1]),
            "frames": ph["frames"][:50],
            "all_frames": ph["frames"],
            "per_frame_reads": [],
            "best_frame": int(ph["best_frame"]),
            "best_text": "",
            "best_conf": 0.0,
            "best_crop_file": "",
            "best_annotated_file": "",
            "final_text": "",
            "final_conf": 0.0,
            "valid_indian": False,
            "avg_yolo_conf": round(sum(ph["confs"]) / len(ph["confs"]), 4),
            "n_unique_reads": 0,
            "votes_per_pos": {},
            "vehicle_crop_file": best_crop_file,
        })

    # Process Standalone Plate tracks (if any plate wasn't associated with a vehicle)
    for plate_tid in sorted(plate_history.keys()):
        # Check if plate_tid was mapped to any vehicle
        mapped = False
        for tid, p_tid in vehicle_to_plate_map.items():
            if p_tid == plate_tid:
                mapped = True
                break
        if mapped:
            continue
            
        ph = plate_history[plate_tid]
        voted = vote_track_text(ph["reads"])
        
        per_frame_reads = []
        for i, fnum in enumerate(ph["frames"]):
            text, ocr_conf = ph["reads"][i]
            per_frame_reads.append({
                "frame": int(fnum),
                "text": text,
                "ocr_conf": round(float(ocr_conf), 4),
                "yolo_conf": round(float(ph["confs"][i]), 4),
                "bbox": [int(v) for v in ph["bboxes"][i]],
                "crop_file": ph["crop_files"][i] if i < len(ph["crop_files"]) else "",
            })
            
        best_plate_crop = ""
        if ph["best_crop_file"] == "" and ph["crop_files"]:
            best_frame_num = int(ph["best_frame"])
            for i, fnum in enumerate(ph["frames"]):
                if int(fnum) == best_frame_num:
                    src_crop = crops_dir / ph["crop_files"][i]
                    if src_crop.exists():
                        # Keep the source extension — crops are PNG now
                        best_plate_crop = f"plate{plate_tid:03d}_best_f{best_frame_num:06d}{src_crop.suffix}"
                        shutil.copy2(str(src_crop), str(best_frames_dir / best_plate_crop))
                    break
                    
        # Add as standalone plate
        tracks_summary.append({
            "track_id": 1000 + int(plate_tid),
            "class_name": "plate",
            "n_frames": len(ph["frames"]),
            "first_seen": int(ph["frames"][0]),
            "last_seen": int(ph["frames"][-1]),
            "frames": ph["frames"][:50],
            "all_frames": ph["frames"],
            "per_frame_reads": per_frame_reads,
            "best_frame": int(ph["best_frame"] or ph["frames"][0]),
            "best_text": ph["best_text"] or "",
            "best_conf": round(float(ph["best_conf"]), 4),
            "best_crop_file": best_plate_crop,
            "best_annotated_file": ph.get("best_annotated_file", ""),
            "final_text": voted["text"],
            "final_conf": voted["conf"],
            "valid_indian": voted["valid"],
            "avg_yolo_conf": round(sum(ph["confs"]) / len(ph["confs"]), 4),
            "n_unique_reads": len(set(t for t, _ in ph["reads"] if t)),
            "votes_per_pos": voted["votes"],
            "vehicle_crop_file": "",
            "merged_from": sorted(merge_map.get(plate_tid, [])),
            "duplicate_text_group": dup_of.get(plate_tid, ""),
        })

    # Sort tracks: Vehicles with valid plates first, then general vehicles, then persons, then standalone plates
    def sort_key(t):
        is_veh = t["class_name"] != "person" and t["class_name"] != "plate"
        is_person = t["class_name"] == "person"
        is_valid = t.get("valid_indian", False)
        final_conf = t.get("final_conf", 0.0)
        return (
            -int(is_veh and is_valid),  # valid vehicle plates first
            -int(is_veh),              # vehicle tracks second
            -int(is_person),           # persons third
            -final_conf,               # higher confidence plates
            -t["n_frames"]
        )
        
    tracks_summary.sort(key=sort_key)

    elapsed = round(time.time() - t0, 2)
    summary = {
        "video": str(video_path),
        "video_name": video_path.name,
        "n_total_frames": total,
        "n_frames_processed": n_proc,
        "fps": round(fps, 2),
        "stride": stride,
        "tracker": f"ByteTrack (COCO: {coco_model_name}, Plate: {plate_model_name})",
        "elapsed_sec": elapsed,
        "fps_processed": round(n_proc / elapsed, 2) if elapsed > 0 else 0.0,
        "n_tracks": len(tracks_summary),
        "n_valid_plates": sum(1 for t in tracks_summary if t.get("valid_indian")),
        "n_merged_plate_tracks": sum(len(v) for v in merge_map.values()),
        "duplicate_plate_groups": dup_groups,
        "detector": f"COCO: {coco_model_name}, Plate: {plate_model_name}",
        "ocr": "Awiros ANPR-OCR",
        "device": device,
        "tracks": tracks_summary,
        "output_dir": str(out_dir),
    }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done in %.1fs — %d tracks, %d valid plates",
             elapsed, summary["n_tracks"], summary["n_valid_plates"])
    return summary

def main():
    p = argparse.ArgumentParser(description="ANPR video pipeline")
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--yolo-model", type=str, default="yolo11_plate")
    p.add_argument("--yolo-coco-model", type=str, default="yolo11s")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if not args.video.exists():
        log.error("Video not found: %s", args.video)
        return 2

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = args.video.parent / f"{args.video.stem}_awiros_{ts}"

    process_video(
        video_path=args.video,
        out_dir=args.out,
        stride=args.stride,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        yolo_model=args.yolo_model,
        yolo_coco_model=args.yolo_coco_model,
        device=args.device,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
