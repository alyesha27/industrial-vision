import json
import time
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config
from src.inference.detector import Detector
from src.capture.video_capture import VideoSource, FPSMeter
from src.events.event_engine import EventEngine
from src.zones.geometry import bottom_center, to_pixels, point_in_polygon
from src.monitoring.metrics import Metrics
from src.monitoring.logger import setup_logging

setup_logging()
log = logging.getLogger("vision")
cam_log = logging.getLogger("vision.camera")

cfg = load_config()
CAM_ID = cfg["camera"]["id"]
STATUS_FILE = Path(cfg["storage"]["root"]) / "status.json"
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)


def write_status(camera, fps, inference_ms):
    data = {
        "camera": camera,
        "fps": round(fps, 1),
        "inference_ms": round(inference_ms, 1),
        "model": "loaded",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


source = VideoSource(source=cfg["camera"]["source"], reconnect_sec=cfg["camera"]["reconnect_sec"])
detector = Detector(
    cfg["models"]["detector_path"],
    confidence=cfg["models"]["person_confidence"],
    tracker=cfg["models"]["tracker"],
)
fps_meter = FPSMeter()
metrics = Metrics()
engine = EventEngine(
    CAM_ID,
    root=cfg["storage"]["root"],
    n=cfg["events"]["confirmation_n"],
    m=cfg["events"]["confirmation_m"],
    cooldown_sec=cfg["events"]["cooldown_sec"],
)
zone_norm = cfg["zones"]["restricted_01"]

camera_lost = False
last_status = 0.0
fps, inference_ms = 0.0, 0.0

try:
    source.open()
    log.info("Камера подключена")
    print("Камера подключена. Нажмите Q для выхода.")

    while True:
        # --- получение кадра с обработкой потери потока ---
        try:
            frame = source.read()
            if camera_lost:
                camera_lost = False
                cam_log.warning("Поток восстановлен")
                engine.emit("camera_stream_restored", 1.0, None, {})
        except Exception as error:
            if not camera_lost:
                camera_lost = True
                cam_log.error("Потеря потока: %s", error)
                engine.emit("camera_stream_lost", 1.0, None, {})
                write_status("lost", 0.0, 0.0)
            time.sleep(cfg["camera"]["reconnect_sec"])
            continue

        h, w = frame.shape[:2]
        polygon_px = to_pixels(zone_norm, w, h)

        # --- inference + tracking ---
        start = metrics.start_timer()
        detections = detector.predict(frame)
        inference_ms = metrics.get_inference_time(start)

        # --- аналитика: кто из людей сейчас в зоне ---
        active = {}
        for obj in detections:
            if obj["class_name"] != "person" or obj["track_id"] is None:
                continue
            point = bottom_center(obj)
            if point_in_polygon(point, polygon_px):
                active[obj["track_id"]] = (
                    obj["confidence"],
                    {"x": round(point[0] / w, 3), "y": round(point[1] / h, 3)},
                )

        # --- Event Engine решает, создавать ли событие ---
        engine.process("person_in_restricted_zone", active, frame)

        # --- отрисовка ---
        cv2.polylines(frame, [np.array(polygon_px, dtype=np.int32)], True, (255, 0, 0), 2)

        for obj in detections:
            in_zone = obj["track_id"] in active
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            label = f'{obj["class_name"]} #{obj["track_id"]} {obj["confidence"]:.2f}'
            cv2.rectangle(frame, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), color, 2)
            cv2.putText(frame, label, (obj["x1"], max(obj["y1"] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        fps = fps_meter.tick()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Inference: {inference_ms:.1f} ms", (20, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"CPU: {metrics.get_cpu():.1f}% RAM: {metrics.get_ram():.0f} MB",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # статус для API раз в секунду
        if time.time() - last_status > 1:
            write_status("connected", fps, inference_ms)
            last_status = time.time()

        cv2.imshow("Industrial Vision - Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception:
    log.exception("Критическая ошибка")

finally:
    source.close()
    cv2.destroyAllWindows()
    write_status("stopped", 0.0, 0.0)