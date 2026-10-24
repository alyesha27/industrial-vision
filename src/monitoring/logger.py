import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    app = RotatingFileHandler(
        "logs/application.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    app.setFormatter(fmt)

    cam = RotatingFileHandler(
        "logs/camera_errors.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    cam.setFormatter(fmt)

    root = logging.getLogger("vision")
    root.setLevel(logging.INFO)
    root.addHandler(app)
    root.addHandler(logging.StreamHandler())  # дублируем в консоль

    camera = logging.getLogger("vision.camera")
    camera.addHandler(cam)  # propagate=True, поэтому попадёт и в application.log