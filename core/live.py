"""
Live-stream pipeline for Video Analytics ANPR FRS.

Architecture:
    LiveSource       — wraps cv2.VideoCapture for either a local video file
                        (looped at a target FPS) or a network camera URL
                        (RTSP / HTTP / index). Runs in its own daemon thread
                        and yields BGR frames into a thread-safe queue.
    LivePipeline     — for each frame: dual-YOLO detection (COCO for
                        vehicles/persons + plate detector) + ByteTrack
                        association + Awiros OCR. Pushes annotated frames
                        + JSON detection events to per-session queues that
                        the Flask routes consume.
    LiveSessionManager — owns the live sessions dict, hands out session_ids,
                        and cleans up on stop.

Performance strategy (verified on this CPU-only Windows box):
    - YOLO11n COCO ~146 ms/frame + YOLO11_plate ~130 ms/frame = ~280 ms
    - Awiros OCR ~844 ms/crop is the bottleneck. We run OCR ONLY when a
      tracked plate reaches its best-crop frame (largest area for that
      track_id). That keeps live preview at ~3-4 fps while still giving
      every plate track a single readable OCR read.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("live")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# LiveSource
# ---------------------------------------------------------------------------
class LiveSource:
    """Reads frames from a local video file (looped) or a network camera URL."""

    def __init__(self, source: str | int, target_fps: float = 25.0, loop: bool = True):
        self.source = source  # int camera index, or str file path / URL
        self.target_fps = max(1.0, float(target_fps))
        self.loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._frame_interval = 1.0 / self.target_fps
        self._reader_t0 = 0.0
        self._frame_count = 0
        self._fps_native = 25.0
        self._width = 0
        self._height = 0
        self._is_url = isinstance(source, str) and source.lower().startswith(
            ("rtsp://", "http://", "https://")
        )

    def start(self):
        if self._thread is not None:
            return
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")
        self._fps_native = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._thread = threading.Thread(target=self._reader, daemon=True, name="LiveSource")
        self._thread.start()
        print(f"[live] {_now()} LiveSource started: source={self.source} target_fps={self.target_fps:.1f} native={self._fps_native:.1f} {self._width}x{self._height}",
              file=sys.stderr, flush=True)

    def stop(self):
        # Only the reader thread touches self._cap (read/reopen/release) —
        # releasing a VideoCapture from another thread while the reader is in
        # cap.read() can crash natively. Signal, wait for the reader to exit,
        # then drain the frame queue.
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def native_fps(self) -> float:
        return self._fps_native

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Block until the next frame is available, or return None on stop."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _reader(self):
        self._reader_t0 = time.monotonic()
        try:
            self._reader_loop()
        finally:
            # The reader thread owns the capture lifecycle — release here,
            # never from another thread (see LiveSource.stop).
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def _reader_loop(self):
        while not self._stop_event.is_set():
            ok, frame = self._cap.read() if self._cap else (False, None)
            if not ok or frame is None:
                if self._is_url:
                    time.sleep(0.2)
                    if self._cap:
                        try:
                            self._cap.release()
                        except Exception:
                            pass
                    self._cap = cv2.VideoCapture(self.source)
                    continue
                if not self.loop:
                    self._stop_event.set()
                    return
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = cv2.VideoCapture(self.source)
                continue

            self._frame_count += 1
            now = time.monotonic()
            expected = self._frame_count * self._frame_interval
            elapsed = now - self._reader_t0
            if elapsed < expected:
                time.sleep(expected - elapsed)
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Per-track state
# ---------------------------------------------------------------------------
@dataclass
class TrackState:
    """Live state for a tracked object (vehicle, person, or plate)."""
    track_id: int
    class_name: str
    first_seen_ms: int
    last_seen_ms: int
    last_bbox_xyxy: list
    last_conf: float
    n_frames: int = 0
    plate_text: str = ""
    plate_conf: float = 0.0
    plate_valid: bool = False
    plate_track_id: Optional[int] = None
    ocr_done: bool = False
    ocr_scheduled: bool = False
    best_bbox_area: int = 0
    best_bbox_xyxy: Optional[list] = None
    best_frame_count: int = 0
    best_score: float = 0.0


# ---------------------------------------------------------------------------
# LivePipeline
# ---------------------------------------------------------------------------
class LivePipeline:
    """Runs YOLO + OCR on each frame from a LiveSource."""

    def __init__(
        self,
        session_id: str,
        source: LiveSource,
        yolo_coco,
        yolo_plate,
        awiros,
        ocr_on_best_only: bool = True,
        iou_thresh: float = 0.20,
        device: str = "cpu",
        best_crop_algo: str = "area",
    ):
        self.session_id = session_id
        self.source = source
        self.yolo_coco = yolo_coco
        self.yolo_plate = yolo_plate
        self.awiros = awiros
        self.ocr_on_best_only = ocr_on_best_only
        self.iou_thresh = iou_thresh
        self.device = device
        # Best-crop selection criterion for the OCR trigger frame:
        #   "area"      — largest plate bbox seen for the track (default)
        #   "sharpness" — sharpest crop by Laplacian variance
        self.best_crop_algo = best_crop_algo if best_crop_algo in ("area", "sharpness") else "area"

        self._jpeg_q: "queue.Queue[bytes]" = queue.Queue(maxsize=2)
        self._event_q: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.vehicle_states: dict[int, TrackState] = {}
        self.person_states: dict[int, TrackState] = {}
        self.plate_states: dict[int, TrackState] = {}
        self._plate_to_vehicle: dict[int, int] = {}
        self._plate_tracker = self._make_plate_tracker()

        # v4: best crop JPEG bytes per plate track_id (for modal inspection)
        self._best_crops: dict[int, bytes] = {}
        # v4: per-track trajectory points for Tracking stage (deque of (frame_idx, cx, cy))
        self._trajectories: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

        self.fps_actual: float = 0.0
        self.frame_index: int = 0
        self.t_start: float = 0.0
        self.t_last_frame: float = 0.0
        # v4: per-frame component timings (ms)
        self.last_timing: dict = {}

    @staticmethod
    def _make_plate_tracker():
        from anpr_video_awiros import ByteTracker
        return ByteTracker(frame_rate=30, max_age=60)

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"LivePipeline-{self.session_id}"
        )
        self._thread.start()
        print(f"[live] {_now()} LivePipeline {self.session_id} thread started",
              file=sys.stderr, flush=True)

    def stop(self):
        self._stop_event.set()
        self._pause_event.clear()
        self.source.stop()

    def set_paused(self, paused: bool):
        """Pause/resume frame processing. While paused the pipeline stops
        consuming frames: MJPEG freezes on the last frame, SSE goes quiet,
        and file sources simply hold their position (the reader's 2-slot
        queue drops frames it can't deliver)."""
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    @property
    def paused(self) -> bool:
        return self._pause_event.is_set()

    def jpeg_queue(self) -> "queue.Queue[bytes]":
        return self._jpeg_q

    def event_queue(self) -> "queue.Queue[dict]":
        return self._event_q

    def _emit_event(self, event_type: str, data: dict):
        evt = {"type": event_type, "ts_ms": int(time.time() * 1000), **data}
        try:
            self._event_q.put_nowait(evt)
        except queue.Full:
            try:
                self._event_q.get_nowait()
                self._event_q.put_nowait(evt)
            except Exception:
                pass

    def _encode_jpeg_raw(self, frame: np.ndarray) -> bytes:
        """Encode a RAW (un-annotated) frame as JPEG for MJPEG — v4."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return b""
        return buf.tobytes()

    def _encode_crop_jpeg(self, crop: np.ndarray, max_w: int = 400) -> bytes:
        """Encode a plate crop as JPEG bytes (for _best_crops cache)."""
        if crop is None or crop.size == 0:
            return b""
        h, w = crop.shape[:2]
        if w > max_w:
            h_new = int(h * (max_w / w))
            crop = cv2.resize(crop, (max_w, h_new))
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return b""
        return buf.tobytes()

    def _plate_score(self, frame: np.ndarray, bbox) -> float:
        """Score a plate bbox by the session's best-crop criterion.

        "area"      → bbox pixel area (cheap)
        "sharpness" → Laplacian variance of the crop resized to 100x40
                      (normalizes for crop size so it measures intrinsic
                      blur — same metric as anpr_video_awiros._crop_sharpness)
        """
        x1, y1, x2, y2 = bbox
        if self.best_crop_algo != "sharpness":
            return float((x2 - x1) * (y2 - y1))
        try:
            from anpr_video_awiros import _crop_sharpness
            H, W = frame.shape[:2]
            crop = frame[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
            return _crop_sharpness(crop)
        except Exception:
            return float((x2 - x1) * (y2 - y1))

    @staticmethod
    def _crop_with_padding(frame: np.ndarray, bbox: list) -> np.ndarray:
        """Crop a bbox region with 6% padding, clamped to frame bounds."""
        H, W = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw, bh = bx2 - bx1, by2 - by1
        px, py = int(bw * 0.06), int(bh * 0.06)
        cx1 = max(0, int(bx1) - px)
        cy1 = max(0, int(by1) - py)
        cx2 = min(W, int(bx2) + px)
        cy2 = min(H, int(by2) + py)
        return frame[cy1:cy2, cx1:cx2]

    def _run_ocr(self, crop: np.ndarray) -> tuple[str, float, bool, float]:
        """Run Awiros OCR on a crop.

        Thread-safety lives inside AwirosANPR.predict_crop() itself
        (AWIROS_PREDICT_LOCK), so every caller — live sessions, upload
        endpoints, benchmarks — is serialized against PaddlePaddle's
        non-thread-safe static graph.
        Returns (text, confidence, valid_indian, elapsed_ms). elapsed_ms is
        0.0 when OCR did not actually run (empty crop or error).
        """
        if crop is None or crop.size == 0:
            return "", 0.0, False, 0.0
        t0 = time.time()
        try:
            res = self.awiros.predict_crop(crop)
            text = res.get("text", "") or ""
            conf = float(res.get("confidence", 0.0))
            valid = bool(res.get("valid_indian", False))
            return text, conf, valid, (time.time() - t0) * 1000
        except Exception as e:
            print(f"[live] {_now()} OCR error: {e}", file=sys.stderr, flush=True)
            return "", 0.0, False, 0.0

    def _link_ocr_to_vehicle(self, plate_tid: int, text: str, conf: float, valid: bool):
        """Propagate an OCR result onto the associated vehicle's track state."""
        v_tid = self._plate_to_vehicle.get(plate_tid)
        if v_tid is None:
            return
        vst = self.vehicle_states.get(v_tid)
        if vst is not None:
            vst.plate_text = text
            vst.plate_conf = conf
            vst.plate_valid = valid
            vst.plate_track_id = plate_tid

    def _run(self):
        try:
            self._run_inner()
        except Exception as e:
            print(f"[live] {_now()} _run crashed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._emit_event("error", {"message": str(e)})

    def _run_inner(self):
        self.source.start()
        self.t_start = time.time()
        self.t_last_frame = time.time()

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                # Frozen: hold last MJPEG frame, emit nothing.
                time.sleep(0.15)
                continue
            try:
                frame = self.source.read(timeout=1.0)
                if frame is None:
                    if self.source._stop_event.is_set():
                        break
                    continue
                self._process_frame(frame)
            except Exception as e:
                print(f"[live] {_now()} frame loop error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                continue

        self._emit_event("end", {"frame_index": self.frame_index})
        print(f"[live] {_now()} _run_inner exiting sid={self.session_id} frames={self.frame_index}",
              file=sys.stderr, flush=True)

    def _process_frame(self, frame):
        """Process one frame through the full pipeline — v4 verifiable mode.

        Sends RAW (un-annotated) frames via MJPEG. All detection data goes
        through SSE so the frontend can render each pipeline stage independently
        on a clickable canvas.
        """
        import time as _time
        self.frame_index += 1
        ts_ms = int(_time.time() * 1000)
        H, W = frame.shape[:2]

        # ── Timing: COCO detection + tracking ──
        t_coco_start = _time.time()
        coco_res = self.yolo_coco.track(
            frame,
            persist=True,
            classes=[0, 1, 2, 3, 5, 7],
            conf=0.25,
            tracker="bytetrack.yaml",
            verbose=False,
            device=self.device,
        )[0]
        dt_coco_ms = (_time.time() - t_coco_start) * 1000

        coco_dets = []
        if coco_res.boxes is not None and coco_res.boxes.id is not None:
            for box in coco_res.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                tid = int(box.id[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo_coco.names.get(cls_id, str(cls_id))
                coco_dets.append({
                    "track_id": tid,
                    "bbox_xyxy": xyxy,
                    "confidence": conf,
                    "class_name": cls_name,
                })

        # ── Timing: Plate detection ──
        t_plate_start = _time.time()
        plate_res = self.yolo_plate.predict(
            frame, conf=0.25, iou=0.45, imgsz=640, verbose=False, device=self.device
        )[0]
        dt_plate_ms = (_time.time() - t_plate_start) * 1000

        raw_plates = []
        if plate_res.boxes is not None and len(plate_res.boxes) > 0:
            for box in plate_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                yconf = float(box.conf[0].cpu().item())
                bw, bh = x2 - x1, y2 - y1
                if bh <= 0 or bw <= 0:
                    continue
                ar = bw / bh
                if ar < 1.5 or ar > 6.5:
                    continue
                if bh < 16 or bh > 320:
                    continue
                raw_plates.append(((int(x1), int(y1), int(x2), int(y2)), yconf))

        tracked_plates = self._plate_tracker.update(raw_plates)
        dt_track_ms = 0.0  # tracked_plates update is negligible; could time separately

        # ── Update state dicts + trajectories ──
        for d in coco_dets:
            tid = d["track_id"]
            bbox = d["bbox_xyxy"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            self._trajectories[tid].append((self.frame_index, cx, cy))
            if d["class_name"] == "person":
                st = self.person_states.get(tid)
                if st is None:
                    st = TrackState(
                        track_id=tid, class_name="person",
                        first_seen_ms=ts_ms, last_seen_ms=ts_ms,
                        last_bbox_xyxy=bbox, last_conf=d["confidence"],
                        best_bbox_area=area, best_bbox_xyxy=bbox,
                        best_frame_count=self.frame_index,
                    )
                    self.person_states[tid] = st
                else:
                    st.last_seen_ms = ts_ms
                    st.last_bbox_xyxy = bbox
                    st.last_conf = d["confidence"]
                    st.n_frames += 1
                    if area > st.best_bbox_area:
                        st.best_bbox_area = area
                        st.best_bbox_xyxy = bbox
                        st.best_frame_count = self.frame_index
            else:
                st = self.vehicle_states.get(tid)
                if st is None:
                    st = TrackState(
                        track_id=tid, class_name=d["class_name"],
                        first_seen_ms=ts_ms, last_seen_ms=ts_ms,
                        last_bbox_xyxy=bbox, last_conf=d["confidence"],
                        best_bbox_area=area, best_bbox_xyxy=bbox,
                        best_frame_count=self.frame_index,
                    )
                    self.vehicle_states[tid] = st
                else:
                    st.last_seen_ms = ts_ms
                    st.last_bbox_xyxy = bbox
                    st.last_conf = d["confidence"]
                    st.n_frames += 1
                    if area > st.best_bbox_area:
                        st.best_bbox_area = area
                        st.best_bbox_xyxy = bbox
                        st.best_frame_count = self.frame_index

        # Plates: associate with vehicles by containment
        current_plate_tids: set[int] = set()
        for plate_tid, bbox, yconf in tracked_plates:
            px1, py1, px2, py2 = bbox
            p_area = (px2 - px1) * (py2 - py1)
            if p_area <= 0:
                continue
            current_plate_tids.add(plate_tid)
            score = self._plate_score(frame, bbox)
            # trajectory for plate track
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            self._trajectories[plate_tid].append((self.frame_index, pcx, pcy))
            pst = self.plate_states.get(plate_tid)
            if pst is None:
                pst = TrackState(
                    track_id=plate_tid, class_name="plate",
                    first_seen_ms=ts_ms, last_seen_ms=ts_ms,
                    last_bbox_xyxy=list(bbox), last_conf=yconf,
                    best_bbox_area=p_area, best_bbox_xyxy=list(bbox),
                    best_frame_count=self.frame_index, best_score=score,
                )
                self.plate_states[plate_tid] = pst
            else:
                pst.last_seen_ms = ts_ms
                pst.last_bbox_xyxy = list(bbox)
                pst.last_conf = yconf
                pst.n_frames += 1
                if score > pst.best_score:
                    pst.best_score = score
                    pst.best_bbox_area = p_area
                    pst.best_bbox_xyxy = list(bbox)
                    pst.best_frame_count = self.frame_index

            best_v = None
            best_overlap = 0.0
            for d in coco_dets:
                if d["class_name"] == "person":
                    continue
                vx1, vy1, vx2, vy2 = d["bbox_xyxy"]
                ix1, iy1 = max(px1, vx1), max(py1, vy1)
                ix2, iy2 = min(px2, vx2), min(py2, vy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                overlap = ((ix2 - ix1) * (iy2 - iy1)) / p_area
                if overlap > 0.7 and overlap > best_overlap:
                    best_overlap = overlap
                    best_v = d["track_id"]
            if best_v is not None:
                self._plate_to_vehicle[plate_tid] = best_v

        # ── OCR stage — per-session strategy ──
        #   "best"  (ocr_on_best_only=True): run Awiros once per plate track,
        #           at the frame where its bbox is largest (best crop).
        #           Recommended — Awiros is ~2.5 s/crop on CPU.
        #   "every" (ocr_on_best_only=False): run OCR on every visible plate
        #           each frame and keep the highest-confidence read.
        #           Spec's "every frame" strategy; very slow on CPU by design.
        dt_ocr_ms = 0.0
        ocr_fired_this_frame = False
        if self.awiros is not None:
            if self.ocr_on_best_only:
                for plate_tid, pst in list(self.plate_states.items()):
                    if pst.ocr_done or pst.ocr_scheduled:
                        continue
                    if self.frame_index == pst.best_frame_count and pst.best_bbox_xyxy is not None:
                        pst.ocr_scheduled = True
                        crop = self._crop_with_padding(frame, pst.best_bbox_xyxy)
                        text, conf, valid, ms = self._run_ocr(crop)
                        if ms > 0:
                            dt_ocr_ms += ms
                            ocr_fired_this_frame = True
                        pst.plate_text = text
                        pst.plate_conf = conf
                        pst.plate_valid = valid
                        pst.ocr_done = True
                        self._best_crops[plate_tid] = self._encode_crop_jpeg(crop)
                        self._link_ocr_to_vehicle(plate_tid, text, conf, valid)
            else:
                for plate_tid in current_plate_tids:
                    pst = self.plate_states.get(plate_tid)
                    if pst is None or pst.last_bbox_xyxy is None:
                        continue
                    crop = self._crop_with_padding(frame, pst.last_bbox_xyxy)
                    text, conf, valid, ms = self._run_ocr(crop)
                    if ms > 0:
                        dt_ocr_ms += ms
                        ocr_fired_this_frame = True
                    # Keep the best read so far for this track
                    if (not pst.ocr_done) or conf >= pst.plate_conf:
                        pst.plate_text = text
                        pst.plate_conf = conf
                        pst.plate_valid = valid
                        self._best_crops[plate_tid] = self._encode_crop_jpeg(crop)
                        self._link_ocr_to_vehicle(plate_tid, pst.plate_text,
                                                  pst.plate_conf, pst.plate_valid)
                    pst.ocr_done = True

        # ── Send RAW frame via MJPEG (no annotation) ──
        jpeg = self._encode_jpeg_raw(frame)
        try:
            self._jpeg_q.put_nowait(jpeg)
        except queue.Full:
            try:
                self._jpeg_q.get_nowait()
                self._jpeg_q.put_nowait(jpeg)
            except Exception:
                pass

        # ── FPS + timing ──
        now = _time.time()
        self.fps_actual = 1.0 / max(now - self.t_last_frame, 1e-6)
        self.t_last_frame = now
        total_ms = dt_coco_ms + dt_plate_ms + dt_ocr_ms
        self.last_timing = {
            "coco_ms": round(dt_coco_ms, 1),
            "plate_ms": round(dt_plate_ms, 1),
            "ocr_ms": round(dt_ocr_ms, 1),
            "total_ms": round(total_ms, 1),
            "ocr_fired": ocr_fired_this_frame,
        }

        # ── Emit enriched SSE event (every frame — annotations must stay in
        # sync with the raw MJPEG frames for the verifiable-pipeline view;
        # slow clients are protected by the queue's drop-oldest policy) ──
        self._emit_frame_event(coco_dets, W, H)

        # ── Prune stale tracks so long-running sessions stay bounded ──
        self._prune_stale(ts_ms)

    # Drop track state not seen for this long (ms). Live cards/overlays only
    # make sense for currently-active objects anyway.
    PRUNE_AFTER_MS = 15_000
    MAX_BEST_CROPS = 60

    def _prune_stale(self, ts_ms: int):
        def prune_dict(d: dict):
            dead = [tid for tid, st in d.items()
                    if ts_ms - st.last_seen_ms > self.PRUNE_AFTER_MS]
            for tid in dead:
                del d[tid]
            return set(dead)

        dead_total = set()
        dead_total |= prune_dict(self.vehicle_states)
        dead_total |= prune_dict(self.person_states)
        dead_total |= prune_dict(self.plate_states)
        for tid in dead_total:
            self._trajectories.pop(tid, None)

        # Cap cached crops (dict preserves insertion order → drop oldest)
        excess = len(self._best_crops) - self.MAX_BEST_CROPS
        if excess > 0:
            for tid in list(self._best_crops.keys())[:excess]:
                del self._best_crops[tid]

    def _emit_frame_event(self, coco_dets: list, W: int, H: int):
        """Build and enqueue one enriched 'frame' SSE event."""
        # Build raw per-frame detection lists for client-side canvas rendering
        coco_dets_payload = [
            {
                "track_id": d["track_id"],
                "class_name": d["class_name"],
                "bbox_xyxy": d["bbox_xyxy"],
                "confidence": round(d["confidence"], 3),
            }
            for d in coco_dets
        ]
        plate_dets_payload = [
            {
                "track_id": pst.track_id,
                "bbox_xyxy": pst.last_bbox_xyxy,
                "confidence": round(pst.last_conf, 3),
                "text": pst.plate_text,
                "ocr_conf": round(pst.plate_conf, 3),
                "valid": pst.plate_valid,
                "linked_vehicle_id": self._plate_to_vehicle.get(pst.track_id),
                "ocr_done": pst.ocr_done,
                "best_frame_count": pst.best_frame_count,
            }
            for pst in self.plate_states.values()
            if pst.last_bbox_xyxy is not None
        ]
        person_dets_payload = [
            {
                "track_id": d["track_id"],
                "bbox_xyxy": d["bbox_xyxy"],
                "confidence": round(d["confidence"], 3),
            }
            for d in coco_dets if d["class_name"] == "person"
        ]
        # Trajectories for Tracking stage — only send CURRENT frame's active
        # track centroids (not full history) to keep payload small.
        # The frontend accumulates trajectory points client-side.
        trajectories_payload = {
            str(tid): list(self._trajectories[tid])[-3:]  # last 3 points only
            for tid in list(self._trajectories.keys())[-50:]  # last 50 tracks
        }

        self._emit_event("frame", {
            "frame_index": self.frame_index,
            "frame_size": [W, H],
            "fps": round(self.fps_actual, 2),
            "timing": self.last_timing,
            "coco_detections": coco_dets_payload,
            "plate_detections": plate_dets_payload,
            "person_detections": person_dets_payload,
            "trajectories": trajectories_payload,
            "vehicles": [
                {
                    "track_id": st.track_id,
                    "class_name": st.class_name,
                    "bbox_xyxy": st.last_bbox_xyxy,
                    "confidence": round(st.last_conf, 3),
                    "n_frames": st.n_frames,
                    "plate_text": st.plate_text,
                    "plate_conf": st.plate_conf,
                    "plate_valid": st.plate_valid,
                    "plate_track_id": st.plate_track_id,
                    "best_bbox_xyxy": st.best_bbox_xyxy,
                    "best_frame_count": st.best_frame_count,
                    "ocr_done": st.ocr_done,
                }
                for st in self.vehicle_states.values()
            ],
            "persons": [
                {
                    "track_id": st.track_id,
                    "bbox_xyxy": st.last_bbox_xyxy,
                    "confidence": round(st.last_conf, 3),
                    "n_frames": st.n_frames,
                    "best_bbox_xyxy": st.best_bbox_xyxy,
                    "best_frame_count": st.best_frame_count,
                }
                for st in self.person_states.values()
            ],
            "plates": [
                {
                    "track_id": st.track_id,
                    "bbox_xyxy": st.last_bbox_xyxy,
                    "confidence": round(st.last_conf, 3),
                    "n_frames": st.n_frames,
                    "text": st.plate_text,
                    "ocr_conf": st.plate_conf,
                    "valid": st.plate_valid,
                    "linked_vehicle_id": self._plate_to_vehicle.get(st.track_id),
                    "best_bbox_xyxy": st.best_bbox_xyxy,
                    "best_frame_count": st.best_frame_count,
                    "ocr_done": st.ocr_done,
                }
                for st in self.plate_states.values()
            ],
        })


# ---------------------------------------------------------------------------
# LiveSession + LiveSessionManager
# ---------------------------------------------------------------------------
class LiveSession:
    def __init__(self, source: LiveSource, pipeline: LivePipeline):
        self.source = source
        self.pipeline = pipeline
        self.created_ms = int(time.time() * 1000)
        self.last_active_ms = self.created_ms
        self.active_streams = 0  # attached MJPEG + SSE clients
        self.meta: dict = {}


class LiveSessionManager:
    def __init__(self):
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._janitor_started = False

    def get(self, sid: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.get(sid)

    def add(self, sid: str, session: LiveSession):
        with self._lock:
            self._sessions[sid] = session
            self._ensure_janitor_locked()

    def remove(self, sid: str):
        with self._lock:
            self._sessions.pop(sid, None)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    # -- stream connection tracking -------------------------------------
    def stream_open(self, sid: str):
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.active_streams += 1

    def stream_close(self, sid: str):
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.active_streams = max(0, s.active_streams - 1)

    def touch(self, sid: str):
        with self._lock:
            s = self._sessions.get(sid)
            if s is not None:
                s.last_active_ms = int(time.time() * 1000)

    def _ensure_janitor_locked(self):
        """Start the background reaper once. It auto-stops sessions whose
        MJPEG + SSE clients have all disconnected (e.g. the browser tab was
        closed without pressing Stop) so pipelines/threads don't leak."""
        if self._janitor_started:
            return
        self._janitor_started = True

        def _janitor():
            grace_ms = 120_000  # 2 min with zero attached streams
            while True:
                time.sleep(30)
                now_ms = int(time.time() * 1000)
                dead = []
                with self._lock:
                    for sid, s in self._sessions.items():
                        if s.active_streams <= 0 and now_ms - s.last_active_ms > grace_ms:
                            dead.append(sid)
                for sid in dead:
                    s = self.get(sid)
                    if s is None:
                        continue
                    try:
                        s.pipeline.stop()
                    except Exception:
                        log.exception("janitor: error stopping pipeline %s", sid)
                    self.remove(sid)
                    log.info("janitor: reaped idle session %s (no streams for %.0fs)",
                             sid, grace_ms / 1000)

        threading.Thread(target=_janitor, daemon=True, name="LiveSessionJanitor").start()


# ---------------------------------------------------------------------------
# Benchmark sample-frame loader
# ---------------------------------------------------------------------------
# All three benchmarks (YOLO / Awiros OCR / ByteTrack) operate on the same
# representative 1280x720 CCTV frame so the numbers are directly comparable.
# The frame is loaded once per /api/live/benchmark call and threaded through
# to each component.
#
# Source priority:
#   1. static/benchmark_sample.jpg  (pinned copy committed in the repo)
#   2. A 720x1280 zero array (last-resort fallback if the pinned copy is missing)
_BENCHMARK_SAMPLE_REL = "static/benchmark_sample.jpg"


def _load_benchmark_frame() -> np.ndarray:
    """Return a single representative 1280x720 BGR frame for benchmarking.

    Pinned source: static/benchmark_sample.jpg (committed copy). Falls back
    to a zero array if it is missing.
    """
    # 1) Pinned repo copy (resolved relative to the repo root)
    sample_path = Path(__file__).resolve().parent.parent / _BENCHMARK_SAMPLE_REL
    if sample_path.exists():
        frame = cv2.imread(str(sample_path))
        if frame is not None:
            return frame
    # 2) Last resort
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def benchmark_model(yolo_model_path: str, frame: np.ndarray, imgsz: int = 640,
                    n_warmup: int = 2, n_measure: int = 5) -> dict:
    """Run a YOLO model N times on a representative frame and return stats.

    `frame` is the shared benchmark frame loaded by _load_benchmark_frame()
    so all three components measure on the exact same input.
    """
    from ultralytics import YOLO
    yolo = YOLO(yolo_model_path)
    for _ in range(n_warmup):
        yolo.predict(frame, conf=0.25, imgsz=imgsz, verbose=False)
    times = []
    for _ in range(n_measure):
        t0 = time.time()
        yolo.predict(frame, conf=0.25, imgsz=imgsz, verbose=False)
        times.append(time.time() - t0)
    times.sort()
    return {
        "model_path": yolo_model_path,
        "imgsz": imgsz,
        "n_measure": n_measure,
        "min_ms": round(min(times) * 1000, 1),
        "median_ms": round(times[len(times) // 2] * 1000, 1),
        "mean_ms": round(sum(times) / len(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "max_fps": round(1.0 / max(times), 2),
    }


# Module-level lock: PaddlePaddle's static-graph program is not thread-safe.
# The Flask dev server runs requests in threads (threaded=True), so any
# AwirosANPR.predict_crop() call from a request handler must hold this lock
# for the duration of the call. The LivePipeline does not need it because
# each session is a single daemon thread that owns the engine instance.
_AWIROS_LOCK = threading.Lock()

# Cache the first successful Awiros benchmark result. PaddlePaddle's static
# graph breaks on the second request thread (compat meta-tensor invalidated
# in conv2d), so we return the cached value for subsequent /api/live/benchmark
# hits. To get a fresh measurement, restart the server (or hit the endpoint
# with a fresh process).
_AWIROS_CACHE: dict | None = None

# v4: Priming result captured at server startup (run.py calls this in the
# main thread before Flask starts). PaddlePaddle's static graph is thread-
# local — the first predict_crop() must run in the thread that loaded the
# model. Once primed, the result is cached here and returned by
# benchmark_awiros() so the benchmark endpoint works from any request thread.
_AWIROS_PRIMED: dict | None = None


def prime_awiros_benchmark(awiros_dir: Path, frame: np.ndarray = None) -> dict:
    """Run Awiros OCR once in the main thread at startup and cache the result.

    Called by run.py before Flask starts. The cached result is returned by
    benchmark_awiros() so the /api/live/benchmark endpoint works even though
    PaddlePaddle's static graph can't be called from Flask request threads.
    """
    global _AWIROS_PRIMED, _AWIROS_CACHE
    if _AWIROS_PRIMED is not None:
        return _AWIROS_PRIMED
    if frame is None:
        frame = _load_benchmark_frame()
    from core.engine import engine as _engine
    import gc, time as _time
    _engine._ensure_awiros()
    awiros = _engine.awiros
    # Warmup
    awiros.predict_crop(frame)
    gc.collect()
    _time.sleep(0.2)
    # Measure
    times = []
    for _ in range(3):
        t0 = _time.time()
        awiros.predict_crop(frame)
        times.append(_time.time() - t0)
    times.sort()
    result = {
        "kind": "ocr",
        "name": "awiros_anpr",
        "model_path": str(Path(awiros_dir) / "model.safetensors"),
        "source": "shared benchmark frame (1280x720)",
        "n_measure": 3,
        "min_ms": round(min(times) * 1000, 1),
        "median_ms": round(times[len(times) // 2] * 1000, 1),
        "mean_ms": round(sum(times) / len(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "max_fps": round(1.0 / max(times), 2),
    }
    _AWIROS_PRIMED = result
    _AWIROS_CACHE = result  # also set the old cache
    print(f"[startup] Awiros benchmark primed: median={result['median_ms']}ms max_fps={result['max_fps']}")
    return result


def benchmark_awiros(awiros_dir: Path, frame: np.ndarray, n_measure: int = 1) -> dict:
    """Return the primed Awiros OCR benchmark result.

    v4: PaddlePaddle's static graph cannot be called from Flask request
    threads. The benchmark is primed at server startup by run.py calling
    prime_awiros_benchmark() in the main thread. This function returns
    that cached result. To get a fresh measurement, restart the server.
    """
    # Check both caches (prime_awiros_benchmark sets both)
    if _AWIROS_CACHE is not None:
        return _AWIROS_CACHE
    if _AWIROS_PRIMED is not None:
        return _AWIROS_PRIMED
    # No primed result available — can't run from request thread
    return {
        "kind": "ocr",
        "name": "awiros_anpr",
        "error": "Awiros benchmark not primed. Restart server to prime it.",
    }


def benchmark_tracker(frame_rate: int = 30, max_age: int = 30,
                      n_warmup: int = 2, n_measure: int = 20) -> dict:
    """Time ByteTracker.update on a synthetic batch of N=6 detections.

    Uses a fixed-but-jittered detection list (slight per-iteration noise so
    the tracker actually has work to do) and a fresh tracker per run so the
    cold-start cost is included. Reports ms/update + max updates/sec.
    """
    from anpr_video_awiros import ByteTracker
    # Build a jittered detection batch: 6 boxes scattered across a 1280x720 frame
    rng = np.random.default_rng(seed=42)
    base_boxes = np.array([
        [100, 200, 220, 280],
        [400, 300, 520, 380],
        [700, 250, 820, 330],
        [900, 400, 1020, 480],
        [200, 500, 320, 580],
        [600, 150, 720, 230],
    ], dtype=float)
    base_confs = np.array([0.85, 0.78, 0.92, 0.65, 0.71, 0.88])

    def make_batch() -> list:
        jitter = rng.normal(0, 2.0, base_boxes.shape)  # ~2px jitter
        boxes = base_boxes + jitter
        return [(tuple(b), float(c)) for b, c in zip(boxes, base_confs)]

    # Warmup (also primes the internal Kalman state)
    for _ in range(n_warmup):
        bt = ByteTracker(frame_rate=frame_rate, max_age=max_age)
        for _ in range(5):  # 5 consecutive frames to build tracks
            bt.update(make_batch())

    times = []
    for _ in range(n_measure):
        bt = ByteTracker(frame_rate=frame_rate, max_age=max_age)
        # Pre-seed a few frames so the tracker has established tracks
        for _ in range(5):
            bt.update(make_batch())
        t0 = time.time()
        bt.update(make_batch())
        times.append(time.time() - t0)
    times.sort()
    max_fps = round(1.0 / max(times), 2) if max(times) > 0 else None
    return {
        "kind": "tracker",
        "name": "bytetrack",
        "n_detections": len(base_boxes),
        "frame_rate": frame_rate,
        "max_age": max_age,
        "n_measure": n_measure,
        "min_ms": round(min(times) * 1000, 3),
        "median_ms": round(times[len(times) // 2] * 1000, 3),
        "mean_ms": round(sum(times) / len(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "max_fps": max_fps,
    }