import argparse
import os
from pathlib import Path

# Must be set BEFORE any PaddlePaddle import chain runs (paddle 2.6 needs the
# pure-python proto runtime). Applies to both the warmup and --no-warmup paths.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from api import create_app
from core.engine import engine

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description="Video Analytics ANPR FRS — ANPR + live dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the full YOLO warmup. Awiros OCR is still loaded "
                        "in the main thread (required for PaddlePaddle "
                        "thread-safety); YOLO models load on first request.")
    p.add_argument("--sample-videos-dir", default=str(HERE / "sample_videos"),
                   help="Directory of local sample videos exposed to the live UI "
                        "(default: <repo>/sample_videos)")
    args = p.parse_args()

    app = create_app(sample_videos_dir=args.sample_videos_dir)

    print("=" * 60)
    print(f"Video Analytics ANPR FRS")
    print(f"  YOLO11 plate detector : {HERE / 'yolo11_plate.pt'}")
    print(f"  Awiros OCR            : {HERE / 'awiros_anpr' / 'model.safetensors'}")
    print(f"  Sample videos         : {args.sample_videos_dir}")
    print(f"  Listening on          : http://{args.host}:{args.port}")
    print("=" * 60)

    # Awiros OCR must be LOADED and PRIMED in the main thread either way:
    # PaddlePaddle's static graph initializes on first inference, and if that
    # happens inside a Flask request thread the graph state gets corrupted
    # for subsequent cross-thread calls.
    from core.live import prime_awiros_benchmark
    awiros_dir = HERE / "awiros_anpr"

    if not args.no_warmup:
        engine.warmup()
        prime_awiros_benchmark(awiros_dir)
    else:
        print("[startup] Pre-loading Awiros OCR in main thread (PaddlePaddle thread-safety)…")
        engine._ensure_awiros()
        prime_awiros_benchmark(awiros_dir)
        print("[startup] Awiros OCR primed ✓")

    # threaded=True lets the dev server serve the SSE/MJPEG streaming
    # requests + the polling /api/live/session requests concurrently with
    # the live-pipeline worker thread (which is CPU-bound on YOLO).
    # process=1 keeps the model instances isolated — a second worker would
    # duplicate the in-memory YOLO state and risk OOM.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            threaded=True, processes=1)

if __name__ == "__main__":
    main()