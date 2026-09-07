# Video Analytics ANPR FRS

Police video analytics system — Phase 1: ANPR (Automatic Number Plate
Recognition) web app. Upload an image or video, or stream live CCTV/webcam,
and get back detected plates with bounding boxes, track IDs, and OCR text.
(Phase 2 roadmap: FRS — see the formulation document below.)

## Formulation document (living spec)

The project's goals and roadmap are maintained in an external Google Doc that
is actively updated:

> **https://docs.google.com/document/d/1k2KNIAj4Gpe_0jikH3_xZLYFpH1xCj1dCFLdg3WTXlA/edit?usp=sharing**

A local mirror + refresh instructions live in [`docs/FORMULATION.md`](docs/FORMULATION.md)
(snapshot: `docs/formulation_snapshot.txt`). Re-fetch the snapshot before
starting new work — the canonical link always wins.

## Pipeline

- **Detection**: YOLO11 (`yolo11_plate.pt` — fine-tuned on Indian plates,
  single class `License_Plate`) + a COCO YOLO11 model for vehicles/persons.
  Prefers OpenVINO-exported models (`*_openvino_model/`) for Intel GPU when
  available; falls back to PyTorch `.pt` on CPU automatically.
- **Tracking**: ByteTrack (ultralytics built-in, motion-only — no ReID model).
  Live sessions get fresh YOLO instances so tracker state never leaks between
  sessions.
- **OCR**: Awiros ANPR (PP-OCRv5 SVTR_HGNet / CTC), vendored `ppocr` package
  under `awiros_anpr/PaddleOCR` (the `paddleocr` pip package is NOT used).
- **Plate text**: per-character-position voting across each track's OCR reads
  collapses noise into a final plate text + confidence, validated against the
  Indian plate grammar (`core/plate_validate.py`).
- **Web**: Flask on port 8766. UI = verifiable pipeline inspector: every stage
  (Raw → Objects → Plates → OCR → Tracking) renders independently on a
  clickable canvas; per-track audit trails for uploaded videos come from
  `/api/track_details`.

## Per-session configuration

Live sessions accept: COCO model variant, plate model variant, **OCR
strategy** (`best-crop only` vs `every frame`), and the **best-crop
criterion** (`largest area` or `sharpest Laplacian`). The same model picks
apply to image/video upload runs.

## Operational notes

- One video job at a time — a second concurrent `/api/detect_video` gets
  `409` (shared tracker state must not be reset mid-run).
- Uploaded videos are deleted right after processing; `video_results/` and
  `results/` artifacts are pruned after 7 days.
- Live sessions whose MJPEG + SSE clients all disconnect are reaped
  automatically after ~2 minutes.
- File sources for live streams must live inside the sample-videos dir;
  URLs are restricted to `rtsp://` / `http(s)://`.

## Run

```bash
# Python 3.11 venv recommended
pip install -r requirements.txt

python run.py             # http://127.0.0.1:8766
python run.py --port 9000
python run.py --no-warmup # skip YOLO warmup (Awiros OCR still preloads — PaddlePaddle thread-safety)
python run.py --sample-videos-dir "D:\path\to\clips"   # videos listed in the live UI
```

Required artifacts before first run (already present / LFS-tracked):

- `yolo11_plate.pt` (+ optional `yolo11s.pt` COCO detector) — auto-downloaded
  from Hugging Face if missing
- `awiros_anpr/model.safetensors` + `awiros_anpr/en_dict.txt`
- `awiros_anpr/PaddleOCR/` — trimmed vendored repo providing `ppocr`

Sample clips for the live dashboard go in `sample_videos/` (or pass
`--sample-videos-dir`); the UI also supports webcam (`0`) and RTSP/HTTP URLs.

## Layout

```
run.py                       # entry point — Flask app + warmup + OCR priming
api/
  __init__.py                # Flask app factory (create_app)
  routes_image.py            # /api/detect, /api/predict, /health
  routes_video.py            # /api/detect_video, /api/cancel_video, /video-results/*
  routes_live.py             # /api/live/* (sessions, MJPEG, SSE, benchmark)
core/
  engine.py                  # ANPR engine (model cache, OpenVINO pick, association)
  live.py                    # LiveSource/LivePipeline/session manager + benchmarks
  plate_validate.py          # Indian plate format validator (BH + state series)
anpr_video_awiros.py         # video pipeline + ByteTracker wrapper + OCR voting
detect_yolo11_awiros_ocr.py  # AwirosANPR class + folder-batch CLI
templates/index.html         # upload + live dashboard UI
static/                      # JS + CSS + pinned benchmark frame
benchmarks/                  # benchmark.py + pinned frames + last report
docs/                        # FORMULATION.md (spec link) + snapshot
requirements.txt
```

Model weights, runtime uploads, and per-video result folders are gitignored;
weights are tracked via Git LFS.

## Notes

- The live pipeline runs OCR only at each plate track's best-crop frame —
  Awiros (~2.5 s/crop CPU) is the bottleneck by design; see
  `/api/live/benchmark` for per-component numbers.
- Swapping trackers (BoT-SORT / DeepOC-SORT) is a one-line change inside
  `anpr_video_awiros.ByteTracker`.
- Dev server binds `127.0.0.1` and is single-process + threaded; there is no
  auth — do not expose it publicly.
