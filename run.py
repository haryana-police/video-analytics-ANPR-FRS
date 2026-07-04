import argparse
from pathlib import Path
from api import create_app
from core.engine import engine

HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description="traffic-plates ANPR + live dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip loading models at startup (loads on first request)")
    p.add_argument("--sample-videos-dir", default=r"C:\Users\harsh\Downloads\cctv samples",
                   help="Directory of local sample videos exposed to the live UI")
    args = p.parse_args()

    app = create_app(sample_videos_dir=args.sample_videos_dir)

    print("=" * 60)
    print(f"Traffic Management System")
    print(f"  YOLO11 plate detector : {HERE / 'yolo11_plate.pt'}")
    print(f"  Awiros OCR            : {HERE / 'awiros_anpr' / 'model.safetensors'}")
    print(f"  Sample videos         : {args.sample_videos_dir}")
    print(f"  Listening on          : http://{args.host}:{args.port}")
    print("=" * 60)

    if not args.no_warmup:
        engine.warmup()
    else:
        # v4: Even with --no-warmup, we must load Awiros in the MAIN thread
        # before the Flask dev server starts spawning request threads.
        # PaddlePaddle's static graph is initialized on first inference —
        # if that first call happens inside a Flask request thread, the
        # graph's CompatMetaTensor state gets corrupted for all subsequent
        # cross-thread calls. Loading + running one inference here in the
        # main thread primes the graph correctly.
        import os
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        print("[startup] Pre-loading Awiros OCR in main thread (PaddlePaddle thread-safety)…")
        engine._ensure_awiros()
        # v4: Prime the benchmark measurement now (in the main thread) so
        # /api/live/benchmark can return real numbers from any request thread.
        from core.live import prime_awiros_benchmark
        awiros_dir = HERE / "awiros_anpr"
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