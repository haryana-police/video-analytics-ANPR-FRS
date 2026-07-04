"""
Benchmark the traffic-plates ANPR pipeline components on 50 sampled frames from
a CCTV video. Reports per-stage timing for:

  * Object detection  (COCO YOLO11 variants: n, s, m)            -> ms/frame
  * Plate detection   (YOLO11 plate-finetune variants: n, s)      -> ms/frame
  * ByteTrack         (associating plate detections frame-to-frame) -> ms/frame
  * Awiros ANPR-OCR   (reading cropped plates)                    -> ms/crop

Frame source: app/benchmarks/frames_50/*.jpg  (pre-extracted)
Models       : app/yolo11{n,s,m}.pt and app/yolo11{,_s}_plate.pt
OCR          : app/awiros_anpr/{model.safetensors, en_dict.txt, PaddleOCR/}

Device       : CPU (the same as the live pipeline default).

Output:
  app/benchmarks/benchmark_results.json
  app/benchmarks/benchmark_report.txt
"""
import os, sys, json, time, argparse, statistics, importlib
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
APP  = HERE.parent
sys.path.insert(0, str(APP))

FRAMES_DIR = HERE / "frames_50"
OUT_JSON   = HERE / "benchmark_results.json"
OUT_REPORT = HERE / "benchmark_report.txt"

COCO_CLASSES = [0, 1, 2, 3, 5, 7]   # person, bicycle, car, motorcycle, bus, truck
DET_CONF = 0.25
IOU      = 0.45
IMGSZ    = 640


def _stats(times_ms):
    if not times_ms:
        return {}
    s = sorted(times_ms)
    n = len(s)
    def pct(p): return s[min(n-1, int(round(p*(n-1))))]
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s), 2),
        "median_ms": round(statistics.median(s), 2),
        "p90_ms": round(pct(0.90), 2),
        "p99_ms": round(pct(0.99), 2),
        "min_ms": round(min(s), 2),
        "max_ms": round(max(s), 2),
        "fps": round(1000.0 / statistics.mean(s), 2) if statistics.mean(s) > 0 else 0.0,
    }


def bench_object_detection(model_paths):
    """model_paths: list[(label, abs_path, imgsz)] — runs each, returns dict."""
    from ultralytics import YOLO
    results = {}
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"\n[object-detection] frames={len(frames)} variants={[l for l,_,_ in model_paths]}")

    for label, path, imgsz in model_paths:
        if not Path(path).exists():
            results[label] = {"error": f"weights not found: {path}"}
            print(f"  {label}: SKIP (no weights)")
            continue
        # warmup
        m = YOLO(str(path))
        _ = m.predict(frames[0], imgsz=imgsz, verbose=False)
        times = []
        for fp in frames:
            t0 = time.perf_counter()
            _ = m.predict(str(fp), imgsz=imgsz, classes=COCO_CLASSES,
                          conf=DET_CONF, iou=IOU, verbose=False, device="cpu")
            times.append((time.perf_counter() - t0) * 1000.0)
        results[label] = {
            "weights": str(path),
            "imgsz": imgsz,
            **_stats(times),
        }
        print(f"  {label}: mean={results[label].get('mean_ms')} ms  fps={results[label].get('fps')}")
        del m
    return results


def bench_plate_detection(model_paths):
    """Plate YOLO detectors — run on full frames, no class filter."""
    from ultralytics import YOLO
    results = {}
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"\n[plate-detection] frames={len(frames)} variants={[l for l,_,_ in model_paths]}")

    for label, path, imgsz in model_paths:
        if not Path(path).exists():
            results[label] = {"error": f"weights not found: {path}"}
            print(f"  {label}: SKIP (no weights)")
            continue
        m = YOLO(str(path))
        _ = m.predict(frames[0], imgsz=imgsz, verbose=False)
        times = []
        total_dets = 0
        for fp in frames:
            t0 = time.perf_counter()
            r = m.predict(str(fp), imgsz=imgsz, conf=DET_CONF, iou=IOU,
                          verbose=False, device="cpu")
            times.append((time.perf_counter() - t0) * 1000.0)
            total_dets += len(r[0].boxes) if r[0].boxes is not None else 0
        results[label] = {
            "weights": str(path),
            "imgsz": imgsz,
            "total_detections": total_dets,
            "detections_per_frame": round(total_dets / max(1, len(frames)), 3),
            **_stats(times),
        }
        print(f"  {label}: mean={results[label].get('mean_ms')} ms  fps={results[label].get('fps')}  "
              f"dets/frame={results[label].get('detections_per_frame')}")
        del m
    return results


def bench_bytetrack():
    """Run yolo11s_plate (best available plate model) with persist + bytetrack
    on the 50 frames. We measure full frame throughput (which is the cost the
    pipeline actually pays) AND isolate tracker update time per frame.
    """
    from ultralytics import YOLO
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"\n[bytetrack] frames={len(frames)}  model=yolo11s_plate  tracker=bytetrack.yaml")
    weights = APP / "yolo11s_plate.pt"
    if not weights.exists():
        return {"error": f"plate weights missing: {weights}"}

    m = YOLO(str(weights))
    # warmup (ultralytics initializes tracker on first .track call)
    _ = m.track(frames[0], persist=True, tracker="bytetrack.yaml",
                conf=DET_CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device="cpu")

    full_times = []     # end-to-end: predict + tracker.update
    track_ids_total = 0
    for fp in frames:
        t0 = time.perf_counter()
        r = m.track(str(fp), persist=True, tracker="bytetrack.yaml",
                    conf=DET_CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device="cpu")[0]
        full_times.append((time.perf_counter() - t0) * 1000.0)
        if r.boxes is not None and r.boxes.id is not None:
            track_ids_total += int(r.boxes.id.shape[0])

    # Isolate tracker-only cost: time just the association pass on synthetic
    # detections. We approximate by creating a 1-frame tracker and feeding it
    # detections extracted from the run above. The tracker is called inside
    # ultralytics' predict(), so a clean measurement is hard without forking
    # the code; we instead report full predict+track time which is what
    # matters for pipeline throughput.
    return {
        "weights": str(weights),
        "tracker_config": "bytetrack.yaml (ultralytics built-in)",
        "imgsz": IMGSZ,
        "total_track_assignments": track_ids_total,
        "note": "Time includes both YOLO plate detection AND ByteTrack "
                "association per frame — that is the actual pipeline cost.",
        **_stats(full_times),
    }


def bench_awiros_ocr(limit_crops: int = 200):
    """Awiros OCR — feed cropped plate regions to the recognizer. We mine
    crops from the 50 frames using yolo11s_plate first (best plate detector
    locally), then run OCR on each crop.
    """
    from ultralytics import YOLO
    sys.path.insert(0, str(APP / "awiros_anpr"))
    from detect_yolo11_awiros_ocr import AwirosANPR  # type: ignore

    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"\n[awiros-ocr] mining plate crops from {len(frames)} frames via yolo11s_plate…")
    det = YOLO(str(APP / "yolo11s_plate.pt"))

    crops = []
    crops_dir = HERE / "crops"
    crops_dir.mkdir(exist_ok=True)
    for fp in frames:
        img = cv2.imread(str(fp))
        r = det.predict(img, conf=0.15, iou=0.45, imgsz=IMGSZ,
                        verbose=False, device="cpu")[0]
        if r.boxes is None:
            continue
        H, W = img.shape[:2]
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            # 6% padding like the pipeline
            bw, bh = x2 - x1, y2 - y1
            px, py = int(bw * 0.06), int(bh * 0.06)
            x1c = max(0, x1 - px); y1c = max(0, y1 - py)
            x2c = min(W, x2 + px); y2c = min(H, y2 + py)
            crop = img[y1c:y2c, x1c:x2c]
            if crop.size == 0 or (y2c - y1c) < 8:
                continue
            crops.append(crop)
            if len(crops) >= limit_crops:
                break
        if len(crops) >= limit_crops:
            break

    print(f"[awiros-ocr] collected {len(crops)} crops, loading Awiros…")
    awiros = AwirosANPR(awiros_dir=APP / "awiros_anpr", device="cpu")
    awiros.load()

    # warmup
    if crops:
        awiros.predict_crop(crops[0])

    times = []
    texts = []
    confs = []
    for c in crops:
        t0 = time.perf_counter()
        out = awiros.predict_crop(c)
        times.append((time.perf_counter() - t0) * 1000.0)
        texts.append(out.get("text", "") or "")
        confs.append(float(out.get("confidence", 0.0) or 0.0))

    return {
        "weights": str(APP / "awiros_anpr" / "model.safetensors"),
        "dict": str(APP / "awiros_anpr" / "en_dict.txt"),
        "device": "cpu",
        "n_crops": len(crops),
        "n_non_empty": sum(1 for t in texts if t),
        "avg_confidence_non_empty": round(
            statistics.mean([c for c, t in zip(confs, texts) if t]) if any(texts) else 0.0, 4),
        "samples": [{"text": t, "conf": round(c, 4)} for t, c in zip(texts, confs) if t][:10],
        **_stats(times),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit-crops", type=int, default=200)
    args = p.parse_args()

    coco_variants = [
        ("yolo11n", APP / "yolo11n.pt", 640),
        ("yolo11s", APP / "yolo11s.pt", 640),
        ("yolo11m", APP / "yolo11m.pt", 640),
    ]
    plate_variants = [
        ("yolo11_plate (n)",  APP / "yolo11_plate.pt", 640),
        ("yolo11s_plate",     APP / "yolo11s_plate.pt", 640),
    ]

    print("=" * 60)
    print(f"Benchmarking on {len(list(FRAMES_DIR.glob('*.jpg')))} frames")
    print(f"Frame source: {FRAMES_DIR}")
    print("=" * 60)

    obj = bench_object_detection(coco_variants)
    plate = bench_plate_detection(plate_variants)
    bt = bench_bytetrack()
    ocr = bench_awiros_ocr(limit_crops=args.limit_crops)

    summary = {
        "device": "cpu",
        "frames_dir": str(FRAMES_DIR),
        "n_frames": len(list(FRAMES_DIR.glob("*.jpg"))),
        "image_size": IMGSZ,
        "object_detection": obj,
        "plate_detection": plate,
        "bytetrack": bt,
        "awiros_ocr": ocr,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Plain-text report
    lines = []
    def add(s=""): lines.append(s)
    add("=" * 64)
    add(" TRAFFIC-PLATES PIPELINE BENCHMARK")
    add(f" Device        : {summary['device']}")
    add(f" Frames        : {summary['n_frames']}  (from app/benchmarks/frames_50/)")
    add(f" Image size    : {summary['image_size']}x{summary['image_size']}")
    add("=" * 64)

    def block(title, payload, cols=("variant", "mean ms", "median ms", "p90 ms", "fps")):
        add("")
        add(f" {title}")
        add(" " + "-" * 60)
        if not payload:
            add("   (no results)")
            return
        if "error" in payload and len(payload) <= 2:
            add(f"   error: {payload['error']}")
            return
        add("   " + " | ".join(f"{c:>14}" for c in cols))
        rows = list(payload.items()) if isinstance(payload, dict) else []
        for label, stats in rows:
            if not isinstance(stats, dict):
                continue
            row = [label,
                   f"{stats.get('mean_ms', '—')}",
                   f"{stats.get('median_ms', '—')}",
                   f"{stats.get('p90_ms', '—')}",
                   f"{stats.get('fps', '—')}"]
            add("   " + " | ".join(f"{str(v):>14}" for v in row))
            if "total_detections" in stats:
                add(f"      total_detections={stats['total_detections']}  "
                    f"det/frame={stats['detections_per_frame']}")

    block("OBJECT DETECTION (COCO, classes=vehicle+person)", obj)
    block("PLATE DETECTION (full-frame)", plate)

    add("")
    add(" BYTETRACK (plate detector + tracker, per-frame)")
    add(" " + "-" * 60)
    if "error" in bt:
        add(f"   error: {bt['error']}")
    else:
        add(f"   mean     : {bt['mean_ms']} ms/frame   ({bt['fps']} fps)")
        add(f"   median   : {bt['median_ms']} ms       p90: {bt['p90_ms']} ms")
        add(f"   tracker  : {bt['tracker_config']}")
        add(f"   total track assignments across {bt['n']} frames: {bt['total_track_assignments']}")
        add(f"   note     : {bt['note']}")

    add("")
    add(" AWIROS ANPR-OCR  (per plate crop)")
    add(" " + "-" * 60)
    if "error" in ocr:
        add(f"   error: {ocr['error']}")
    else:
        add(f"   crops OCR'd      : {ocr['n_crops']}")
        add(f"   non-empty OCR    : {ocr['n_non_empty']}")
        add(f"   avg conf (texts) : {ocr['avg_confidence_non_empty']}")
        add(f"   mean latency     : {ocr['mean_ms']} ms/crop    ({ocr['fps']} crops/s)")
        add(f"   median           : {ocr['median_ms']} ms      p90: {ocr['p90_ms']} ms")
        add(f"   sample OCR reads :")
        for s in ocr["samples"]:
            add(f"      \"{s['text']}\"   conf={s['conf']}")

    add("")
    add("=" * 64)
    add(f" Full JSON: {OUT_JSON}")
    add("=" * 64)

    OUT_REPORT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
