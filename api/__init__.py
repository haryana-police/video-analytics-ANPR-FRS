from flask import Flask


def create_app(sample_videos_dir: str = None):
    from pathlib import Path
    HERE = Path(__file__).resolve().parent.parent

    app = Flask(
        __name__,
        template_folder=str(HERE / "templates"),
        static_folder=str(HERE / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload cap (videos)

    from .routes_image import image_bp
    from .routes_video import video_bp
    from .routes_live import live_bp, set_sample_videos_dir

    app.register_blueprint(image_bp)
    app.register_blueprint(video_bp)
    app.register_blueprint(live_bp)

    if sample_videos_dir:
        set_sample_videos_dir(sample_videos_dir)

    return app