import argparse
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import load_config
from src.capture.video_capture import VideoSource
from src.inference.detector import Detector
from src.events.event_engine import EventEngine
from src.zones.geometry import bottom_center, to_pixels, point_in_polygon
from src.monitoring.metrics import Metrics


def row(name, values):
    a = np.array(values)
    return f"| {name} | {a.mean():.1f} | {np.percentile(a, 95):.1f} | {a.max():.1f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="0 для камеры или путь к видео")
    ap.add_argument("--model", default=None)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="docs/benchmark.md")
    args = ap.parse_args()

    cfg = load_config()
    if args.source is None:
        src_value = cfg["camera"]["source"]
    else:
        src_value = int(args.source) if args.source.isdigit() else args.source
    model_path = args.model or cfg["models"]["detector_path"]

    source = VideoSource(src_value)
    detector = Detector(model_path, confidence=cfg["models"]["person_confidence"],
                        tracker=cfg["models"]["tracker"])
    metrics = Metrics()
    zone_norm = cfg["zones"]["restricted_01"]
    stages = {"capture": [], "inference": [], "events": [], "total": []}

    source.open()
    with tempfile.TemporaryDirectory() as tmp:
        engine = EventEngine(cfg["camera"]["id"], root=tmp,
                             n=cfg["events"]["confirmation_n"],
                             m=cfg["events"]["confirmation_m"],
                             cooldown_sec=cfg["events"]["cooldown_sec"])
        wall_start = None

        for i in range(args.warmup + args.frames):
            if i == args.warmup:
                metrics.get_cpu()  # сбрасываем счётчик CPU после прогрева
                wall_start = perf_counter()

            t0 = perf_counter()
            frame = source.read()
            t1 = perf_counter()
            detections = detector.predict(frame)
            t2 = perf_counter()

            h, w = frame.shape[:2]
            poly = to_pixels(zone_norm, w, h)
            active = {}
            for o in detections:
                if o["class_name"] == "person" and o["track_id"] is not None:
                    p = bottom_center(o)
                    if point_in_polygon(p, poly):
                        active[o["track_id"]] = (o["confidence"], {"x": p[0] / w, "y": p[1] / h})
            engine.process("person_in_restricted_zone", active, frame)
            t3 = perf_counter()

            if i >= args.warmup:
                stages["capture"].append((t1 - t0) * 1000)
                stages["inference"].append((t2 - t1) * 1000)
                stages["events"].append((t3 - t2) * 1000)
                stages["total"].append((t3 - t0) * 1000)

        elapsed = perf_counter() - wall_start

    source.close()
    cpu, ram = metrics.get_cpu(), metrics.get_ram()
    fps = args.frames / elapsed
    size_mb = Path(model_path).stat().st_size / 1e6 if Path(model_path).exists() else 0

    lines = [
        f"Модель: {model_path} ({size_mb:.1f} МБ), кадров: {args.frames}, источник: {src_value}",
        f"Сквозной FPS: {fps:.1f}; CPU процесса: {cpu:.1f}%; RAM: {ram:.0f} МБ",
        "",
        "| Компонент | Среднее, мс | P95, мс | Максимум, мс |",
        "|---|---|---|---|",
        row("Capture", stages["capture"]),
        row("Inference + postprocess", stages["inference"]),
        row("Event processing", stages["events"]),
        row("Total pipeline", stages["total"]),
    ]
    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()