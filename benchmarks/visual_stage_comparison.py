"""
Visual pipeline stage-comparison test — Video Analytics ANPR FRS.

Runs the REAL pipeline components (core.engine YOLO models + Awiros ANPR OCR)
over the CCTV sample videos and saves SIDE-BY-SIDE evidence images for manual
inspection:

  1. fXXXXXX_objects.jpg : RAW FRAME  |  OBJECTS DETECTED (YOLO11 COCO)
  2. fXXXXXX_plates.jpg  : RAW FRAME (plate boxes)  |  PLATE CROPS EXTRACTED
  3. fXXXXXX_ocr.jpg     : PLATE CROP  |  OCR OUTPUT (text · conf · validity)

plus report.json / report.md / index.html summarizing everything.

Usage:
    env\\Scripts\\python.exe benchmarks\\visual_stage_comparison.py
        [--videos-dir sample_videos] [--out benchmarks/stage_comparison]
        [--frames 3] [--max-plates 2] [--conf 0.25]

NOTE: Awiros OCR costs ~2 s per plate crop on CPU — keep --frames/--max-plates
modest. All OCR calls go through the thread-safe AWIROS_PREDICT_LOCK.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from core.engine import engine  # noqa: E402  (repo-root aware)

FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_SIMPLEX  # cv2 has no true mono face; simplex reads fine at small sizes

BG = (12, 16, 22)          # panel background (BGR)
BG_ALT = (18, 24, 34)
SEP = (46, 61, 86)         # separator color
TXT = (242, 245, 250)      # near-white
TXT_DIM = (133, 148, 171)
GREEN = (52, 211, 153)
AMBER = (94, 191, 251)
RED = (113, 113, 248)
BLUE = (248, 189, 56)


# ---------------------------------------------------------------------------
# Image composition helpers
# ---------------------------------------------------------------------------
def with_header(img, title, subtitle=""):
    """Dark header bar above a panel."""
    h, w = img.shape[:2]
    bar = np.full((40, w, 3), BG, dtype=np.uint8)
    cv2.line(bar, (0, 39), (w, 39), SEP, 1)
    cv2.putText(bar, title, (12, 26), FONT, 0.62, TXT, 1, cv2.LINE_AA)
    if subtitle:
        tw = cv2.getTextSize(title, FONT, 0.62, 1)[0][0]
        cv2.putText(bar, subtitle, (w - 12 - len(subtitle) * 11, 27), FONT, 0.48, TXT_DIM, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def resize_to_height(img, target_h):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    s = target_h / h
    return cv2.resize(img, (max(1, int(w * s)), target_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def hstack_panels(panels, gap=10):
    """Panels already carry their own headers — align heights, insert separators."""
    H = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        h, w = p.shape[:2]
        canvas = np.full((H, w, 3), BG, dtype=np.uint8)
        canvas[:h, :w] = p
        padded.append(canvas)
    sep_col = np.full((H, gap, 3), SEP, dtype=np.uint8)
    out = []
    for i, p in enumerate(padded):
        out.append(p)
        if i < len(padded) - 1:
            out.append(sep_col)
    return np.hstack(out)


def placeholder_panel(w, h, lines):
    panel = np.full((h, w, 3), BG_ALT, dtype=np.uint8)
    y = h // 2 - 14 * len(lines) // 2
    for i, ln in enumerate(lines):
        tw = cv2.getTextSize(ln, FONT, 0.55, 1)[0][0]
        cv2.putText(panel, ln, ((w - tw) // 2, y + i * 26 + 14), FONT, 0.55, TXT_DIM, 1, cv2.LINE_AA)
    return panel


def draw_det_box(img, bbox, color, label):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    ty = max(y1 - 7, 15)
    cv2.putText(img, label, (x1, ty), FONT, 0.48, (4, 6, 10), 3, cv2.LINE_AA)
    cv2.putText(img, label, (x1, ty), FONT, 0.48, color, 1, cv2.LINE_AA)


def crops_collage(crops, target_h=170):
    """Tile plate crops horizontally at a common height."""
    if not crops:
        return placeholder_panel(560, target_h, ["no plate crops extracted"])
    tiles = [resize_to_height(c, target_h) for c in crops]
    W = sum(t.shape[1] for t in tiles) + 6 * (len(tiles) - 1)
    canvas = np.full((target_h, W, 3), SEP, dtype=np.uint8)
    x = 0
    for t in tiles:
        canvas[:, x:x + t.shape[1]] = t
        x += t.shape[1] + 6
    return canvas


def ocr_row(crop, ocr, plate_meta):
    """PLATE CROP | OCR OUTPUT panel, side by side."""
    ch, cw = crop.shape[:2]
    panel_h = max(ch, 150)
    crop_r = resize_to_height(crop, panel_h)
    pw = max(430, int(cw * (panel_h / ch)) + 60)
    panel = np.full((panel_h, pw, 3), BG, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (pw - 1, panel_h - 1), SEP, 1)

    text = (ocr.get("text") or "").strip()
    conf = float(ocr.get("confidence", 0.0))
    valid = bool(ocr.get("valid_indian", False))

    color = GREEN if valid and text else (AMBER if text else RED)
    disp = text if text else "(no text)"
    scale = 1.05 if len(disp) <= 10 else (0.85 if len(disp) <= 14 else 0.65)
    cv2.putText(panel, disp, (18, 58), FONT, scale, (4, 6, 10), 5, cv2.LINE_AA)
    cv2.putText(panel, disp, (18, 58), FONT, scale, color, 2, cv2.LINE_AA)

    verdict = "VALID INDIAN PLATE" if valid else ("REVIEW — format mismatch" if text else "NO READ")
    vcolor = GREEN if valid else (AMBER if text else RED)
    cv2.putText(panel, verdict, (18, 92), FONT, 0.55, vcolor, 1, cv2.LINE_AA)

    lines = [
        f"ocr conf   : {conf:.4f}",
        f"yolo conf  : {plate_meta['yolo_conf']:.4f}",
        f"bbox xyxy  : {plate_meta['bbox_xyxy']}",
        f"crop size  : {plate_meta['crop_w']}x{plate_meta['crop_h']} px (+6% pad)",
    ]
    y = 122
    for ln in lines:
        cv2.putText(panel, ln, (18, y), FONT, 0.46, TXT_DIM, 1, cv2.LINE_AA)
        y += 21

    return hstack_panels([with_header(crop_r, "PLATE CROP"), with_header(panel, "OCR OUTPUT — Awiros ANPR")])


# ---------------------------------------------------------------------------
# Pipeline runners (use the app's own cached models)
# ---------------------------------------------------------------------------
COCO_CLASSES = [0, 1, 2, 3, 5, 7]


def run_coco(frame, conf):
    yolo = engine._get_yolo_model("yolo11n")
    dev = engine._OPENVINO_DEVICE
    t0 = time.time()
    res = yolo.predict(frame, conf=conf, iou=0.45, imgsz=640, classes=COCO_CLASSES,
                       verbose=False, **({"device": dev} if dev else {}))[0]
    ms = (time.time() - t0) * 1000
    dets = []
    if res.boxes is not None:
        for b in res.boxes:
            dets.append({
                "class_name": yolo.names.get(int(b.cls[0]), "?"),
                "confidence": round(float(b.conf[0]), 4),
                "bbox_xyxy": [int(v) for v in b.xyxy[0].tolist()],
            })
    return dets, ms


PLATE_AR_MIN, PLATE_AR_MAX, PLATE_H_MIN, PLATE_H_MAX = 1.5, 6.5, 16, 320


def run_plates(frame, conf):
    yolo = engine._get_yolo_model("yolo11_plate")
    dev = engine._OPENVINO_DEVICE
    t0 = time.time()
    res = yolo.predict(frame, conf=conf, iou=0.45, imgsz=640,
                       verbose=False, **({"device": dev} if dev else {}))[0]
    ms = (time.time() - t0) * 1000
    plates, filtered = [], 0
    if res.boxes is not None:
        for b in res.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            bw, bh = x2 - x1, y2 - y1
            ar = bw / bh if bh > 0 else 0
            if not (PLATE_AR_MIN <= ar <= PLATE_AR_MAX and PLATE_H_MIN <= bh <= PLATE_H_MAX):
                filtered += 1
                continue
            plates.append({
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "yolo_conf": round(float(b.conf[0]), 4),
                "area": bw * bh,
                "crop_w": int(bw), "crop_h": int(bh),
            })
    plates.sort(key=lambda p: -p["area"])
    return plates, filtered, ms


def crop_with_padding(frame, bbox, pad=0.06):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return frame[max(0, y1 - py):min(H, y2 + py), max(0, x1 - px):min(W, x2 + px)]


def run_ocr(crop):
    engine._ensure_awiros()
    t0 = time.time()
    res = engine.awiros.predict_crop(crop)
    ms = (time.time() - t0) * 1000
    res["inference_ms"] = round(ms, 1)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Side-by-side pipeline stage comparison evidence")
    ap.add_argument("--videos-dir", default=str(HERE / "sample_videos"))
    ap.add_argument("--out", default=str(HERE / "benchmarks" / "stage_comparison"))
    ap.add_argument("--frames", type=int, default=3, help="frames sampled per video")
    ap.add_argument("--max-plates", type=int, default=2, help="plates OCR'd per frame (largest first)")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    videos = sorted(list(Path(args.videos_dir).glob("*.mp4")))
    if not videos:
        print(f"[!] no .mp4 files found in {args.videos_dir}")
        return 2

    device = engine._OPENVINO_DEVICE or "cpu"
    print(f"[i] device={device} · models=yolo11n + yolo11_plate + Awiros ANPR · "
          f"{args.frames} frames/video · top {args.max_plates} plates OCR'd/frame")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "models": {"coco": "yolo11n", "plate": "yolo11_plate", "ocr": "Awiros ANPR (PP-OCRv5 SVTR_HGNet)"},
        "settings": {"frames_per_video": args.frames, "max_plates_per_frame": args.max_plates,
                     "conf": args.conf, "plate_ar_filter": [PLATE_AR_MIN, PLATE_AR_MAX]},
        "videos": [],
    }

    for video in videos:
        stem = video.stem
        vdir = out_root / f"video{stem}"
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {video.name} ===")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"[!] cannot open {video}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        targets = sorted({int(total * f) for f in (0.3, 0.55, 0.8)[:args.frames]})
        ventry = {"video": video.name, "total_frames": total, "frames": []}
        ti = 0
        got = 0
        idx = -1
        while got < args.frames and ti < len(targets):
            idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            if idx != targets[ti]:
                continue
            ti += 1
            tag = f"f{idx:06d}"
            H, W = frame.shape[:2]

            # ── stage 1: objects ──
            dets, coco_ms = run_coco(frame, args.conf)
            ann = frame.copy()
            for d in dets:
                col = BLUE if d["class_name"] == "person" else AMBER
                draw_det_box(ann, d["bbox_xyxy"], col,
                             f"{d['class_name']} {d['confidence']:.2f}")
            n_v = sum(1 for d in dets if d["class_name"] != "person")
            img_objects = hstack_panels([
                with_header(frame.copy(), "RAW FRAME", f"{W}x{H}"),
                with_header(ann, "OBJECTS DETECTED",
                            f"yolo11n · {coco_ms:.0f}ms · {n_v} veh / {len(dets)-n_v} pers"),
            ])
            cv2.imwrite(str(vdir / f"{tag}_objects.jpg"), img_objects,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            # ── stage 2: plate extraction ──
            plates, filtered, plate_ms = run_plates(frame, args.conf)
            pann = frame.copy()
            for p in plates:
                draw_det_box(pann, p["bbox_xyxy"], GREEN,
                             f"plate {p['yolo_conf']:.2f}")
            crops = [crop_with_padding(frame, p["bbox_xyxy"]) for p in plates]
            img_plates = hstack_panels([
                with_header(pann, "RAW FRAME — PLATE BOXES",
                            f"yolo11_plate · {plate_ms:.0f}ms · kept {len(plates)} (AR-filtered {filtered})"),
                with_header(crops_collage(crops), f"PLATES CROPPED ({len(crops)})"),
            ])
            cv2.imwrite(str(vdir / f"{tag}_plates.jpg"), img_plates,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            # ── stage 3: crop → OCR ──
            rows, fmeta = [], []
            for p in plates[:args.max_plates]:
                crop = crop_with_padding(frame, p["bbox_xyxy"])
                if crop.size == 0:
                    continue
                ocr = run_ocr(crop)
                rows.append(ocr_row(crop, ocr, p))
                fmeta.append({**{k: p[k] for k in ("bbox_xyxy", "yolo_conf")},
                              "ocr_text": ocr.get("text", ""),
                              "ocr_conf": round(float(ocr.get("confidence", 0)), 4),
                              "valid_indian": bool(ocr.get("valid_indian")),
                              "ocr_ms": ocr["inference_ms"]})
                print(f"  {tag} plate @{p['bbox_xyxy']} -> '{ocr.get('text','')}' "
                      f"conf={ocr.get('confidence')} valid={ocr.get('valid_indian')} ({ocr['inference_ms']:.0f}ms)")
            if rows:
                # Rows differ in width (crops differ) — pad to common width
                w_max = max(r.shape[1] for r in rows)

                def pad_to_w(img):
                    if img.shape[1] >= w_max:
                        return img
                    pad = np.full((img.shape[0], w_max - img.shape[1], 3), BG, dtype=np.uint8)
                    return np.hstack([img, pad])

                sep = np.full((8, w_max, 3), SEP, dtype=np.uint8)
                stack = []
                for r_i, r in enumerate(rows):
                    stack.append(pad_to_w(r))
                    if r_i < len(rows) - 1:
                        stack.append(sep)
                img_ocr = np.vstack(stack)
            else:
                img_ocr = hstack_panels([
                    with_header(placeholder_panel(640, 200, ["no plates survived the geometry filter"]),
                                "PLATE CROP"),
                    with_header(placeholder_panel(640, 200, ["—"]), "OCR OUTPUT"),
                ])
            cv2.imwrite(str(vdir / f"{tag}_ocr.jpg"), img_ocr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            ventry["frames"].append({
                "frame_index": idx, "size": [W, H],
                "coco_ms": round(coco_ms, 1), "plate_ms": round(plate_ms, 1),
                "n_detections": len(dets),
                "n_vehicles": n_v, "n_persons": len(dets) - n_v,
                "n_plates_kept": len(plates), "n_plates_filtered": filtered,
                "plates": fmeta,
                "images": [f"{tag}_objects.jpg", f"{tag}_plates.jpg", f"{tag}_ocr.jpg"],
            })
            got += 1
            print(f"  {tag}: {n_v} veh / {len(dets)-n_v} pers · {len(plates)} plates "
                  f"(coco {coco_ms:.0f}ms, plate {plate_ms:.0f}ms)")
        cap.release()
        report["videos"].append(ventry)

    (out_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_root / "report.md")
    write_gallery(report, out_root / "index.html")
    print(f"\n[i] done — results in {out_root}")
    return 0


def write_markdown(report, path):
    L = ["# Pipeline stage-comparison results",
         "",
         f"*Generated {report['generated_at']} · device `{report['device']}` · "
         f"coco=`{report['models']['coco']}` plate=`{report['models']['plate']}` ocr=`{report['models']['ocr']}`*",
         ""]
    for v in report["videos"]:
        L.append(f"## {v['video']} — {v['total_frames']} frames total")
        L.append("")
        for f in v["frames"]:
            L.append(f"### frame {f['frame_index']} — {f['n_vehicles']} vehicles / {f['n_persons']} persons / "
                     f"{f['n_plates_kept']} plates kept ({f['n_plates_filtered']} AR-filtered) · "
                     f"coco {f['coco_ms']}ms · plate {f['plate_ms']}ms")
            L.append("| comparison | image |")
            L.append("|---|---|")
            L.append(f"| raw ↔ objects | `{f['images'][0]}` |")
            L.append(f"| raw ↔ plate crops | `{f['images'][1]}` |")
            L.append(f"| plate crop ↔ OCR output | `{f['images'][2]}` |")
            if f["plates"]:
                L.append("")
                L.append("| OCR text | ocr conf | valid | yolo conf | bbox | ms |")
                L.append("|---|---|---|---|---|---|")
                for p in f["plates"]:
                    L.append(f"| **{p['ocr_text'] or '(none)'}** | {p['ocr_conf']} | "
                             f"{'✅' if p['valid_indian'] else '⚠️'} | {p['yolo_conf']} | {p['bbox_xyxy']} | {p['ocr_ms']} |")
            L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_gallery(report, path):
    css = ("body{background:#0c1016;color:#f2f5fa;font-family:system-ui;margin:0;padding:28px}"
           "h1{font-size:22px}h2{margin:26px 0 10px;color:#f59e0b;font-size:17px}"
           "h3{margin:18px 0 8px;font-size:13px;font-weight:600;color:#c3cdde}"
           "img{width:100%;max-width:1400px;border-radius:8px;border:1px solid #2e3d56;display:block;margin:6px 0}"
           ".muted{color:#8494ab;font-size:13px}")
    parts = [f"<html><head><meta charset='utf-8'><title>Pipeline stage comparison</title><style>{css}</style></head><body>",
             "<h1>Pipeline stage comparison — manual visual verification</h1>",
             f"<p class='muted'>Generated {report['generated_at']} · device {report['device']} · "
             f"raw↔objects · raw↔plate-crops · crop↔OCR-output</p>"]
    for v in report["videos"]:
        parts.append(f"<h2>{v['video']}</h2>")
        for f in v["frames"]:
            parts.append(f"<h3>frame {f['frame_index']} · {f['n_vehicles']} vehicles · {f['n_persons']} persons · "
                         f"{f['n_plates_kept']} plates</h3>")
            for img in f["images"]:
                parts.append(f"<img src=\"{v['video'].replace('.mp4','')}/{img}\" alt='{img}'>")
            for p in f["plates"]:
                verdict = "✅ valid" if p["valid_indian"] else "⚠️ review"
                parts.append(f"<p class='muted'>OCR “<b>{p['ocr_text'] or '—'}</b>” · conf {p['ocr_conf']} · "
                             f"{verdict} · yolo {p['yolo_conf']} @ {p['bbox_xyxy']}</p>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
