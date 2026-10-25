import argparse
import json
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from src.analytics.spill import detect_spill, draw_spill_overlay
from src.config import load_config
from src.inference.detector import Detector
from src.capture.video_capture import VideoSource, FPSMeter
from src.events.event_engine import EventEngine
from src.zones.geometry import bottom_center, to_pixels, point_in_polygon
from src.monitoring.metrics import Metrics
from src.monitoring.logger import setup_logging
from src.analytics.ppe import ppe_violations

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None,
                    help="0 — камера, либо путь к видеофайлу (по умолчанию из конфига)")
args = parser.parse_args()

setup_logging()
log = logging.getLogger("vision")
cam_log = logging.getLogger("vision.camera")

cfg = load_config()
spill_cfg = cfg.get("spill", {})
spill_enabled = spill_cfg.get("enabled", False)
CAM_ID = cfg["camera"]["id"]
STATUS_FILE = Path(cfg["storage"]["root"]) / "status.json"
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

PPE_EVENTS = ("ppe_helmet_missing", "ppe_vest_missing")


def write_status(camera, fps, inference_ms):
    data = {
        "camera": camera,
        "fps": round(fps, 1),
        "inference_ms": round(inference_ms, 1),
        "model": "loaded",
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    STATUS_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# Источник видео: параметр --source важнее конфига
if args.source is None:
    src_value = cfg["camera"]["source"]
else:
    src_value = int(args.source) if args.source.isdigit() else args.source
log.info("Источник видео: %s", src_value)

source = VideoSource(source=src_value, reconnect_sec=cfg["camera"]["reconnect_sec"])

# Основной детектор: люди + Track ID
detector = Detector(
    cfg["models"]["detector_path"],
    confidence=cfg["models"]["person_confidence"],
    tracker=cfg["models"]["tracker"],
)

# PPE-детектор (каска / голова без каски / жилет): без tracking.
# Если весов ещё нет, контроль СИЗ просто отключается.
ppe_path = cfg["models"].get("ppe_path")
ppe_detector = None
if ppe_path and Path(ppe_path).exists():
    ppe_detector = Detector(ppe_path, confidence=cfg["models"]["ppe_confidence"], track=False)
    log.info("PPE-модель загружена: %s", ppe_path)
else:
    log.warning("PPE-модель не найдена (%s), контроль СИЗ отключён", ppe_path)

# Детектор дыма и огня (без tracking)
fire_path = cfg["models"].get("fire_smoke_path")
fire_detector = None
fire_conf = {}
if fire_path and Path(fire_path).exists():
    fire_conf = {
        "fire": cfg["models"]["fire_confidence"],
        "smoke": cfg["models"]["smoke_confidence"],
    }
    fire_detector = Detector(fire_path, confidence=min(fire_conf.values()), track=False)
    log.info("Модель дыма/огня загружена: %s", fire_path)
else:
    log.warning("Модель дыма/огня не найдена (%s), контроль отключён", fire_path)

extra_every = cfg["models"].get("extra_every", 1)

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
min_person_height = cfg["models"].get("min_person_height", 120)

camera_lost = False
last_status = 0.0
fps, inference_ms = 0.0, 0.0
frame_idx = 0
ppe_dets, fire_dets = [], []
spill_mask = None
spill_candidates = []

try:
    source.open()
    log.info("Источник подключён")
    print(f"Источник подключён ({src_value}). Нажмите Q для выхода.")

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

        # --- inference: люди (каждый кадр), СИЗ и дым/огонь (каждый extra_every-й) ---
        start = metrics.start_timer()
        detections = detector.predict(frame)
        if frame_idx % extra_every == 0:
            ppe_dets = ppe_detector.predict(frame) if ppe_detector else []
            fire_dets = fire_detector.predict(frame) if fire_detector else []
        frame_idx += 1
        inference_ms = metrics.get_inference_time(start)

        # --- аналитика 1: кто из людей сейчас в зоне ---
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

        engine.process("person_in_restricted_zone", active, frame)

        # --- аналитика 2: нарушения СИЗ ---
        if ppe_detector:
            ppe_active = {event_type: {} for event_type in PPE_EVENTS}
            for obj in detections:
                if obj["class_name"] != "person" or obj["track_id"] is None:
                    continue
                px, py = bottom_center(obj)
                for violation in ppe_violations(obj, ppe_dets, min_person_height):
                    ppe_active[violation][obj["track_id"]] = (
                        obj["confidence"],
                        {"x": round(px / w, 3), "y": round(py / h, 3)},
                    )
            # вызываем каждый кадр, чтобы окна N из M обновлялись и без нарушений
            for event_type, act in ppe_active.items():
                engine.process(event_type, act, frame)

        # --- аналитика 3: огонь и дым (по кадру, без Track ID) ---
        if fire_detector:
            for cls in ("fire", "smoke"):
                hits = [d for d in fire_dets
                        if d["class_name"] == cls and d["confidence"] >= fire_conf[cls]]
                act = {}
                if hits:
                    best = max(hits, key=lambda d: d["confidence"])
                    cx = (best["x1"] + best["x2"]) // 2
                    cy = (best["y1"] + best["y2"]) // 2
                    act[0] = (best["confidence"], {"x": round(cx / w, 3), "y": round(cy / h, 3)})
                # вызываем каждый кадр, чтобы окно N из M обновлялось
                engine.process(f"{cls}_detected", act, frame)

        # --- аналитика 4: экспериментальное обнаружение разливов ---
        spill_frame = frame

        if spill_enabled:
            spill_mask, spill_candidates = detect_spill(
                frame,
                min_area=spill_cfg.get("min_area", 1200),
                min_area_ratio=spill_cfg.get("min_area_ratio", 0.0005),
                max_area_ratio=spill_cfg.get("max_area_ratio", 0.20),
                min_saturation=spill_cfg.get("min_saturation", 35),
                max_value=spill_cfg.get("max_value", 175),
                ground_start_ratio=spill_cfg.get("ground_start_ratio", 0.45),
            )

            spill_active = {}

            if spill_candidates:
                # Пока используем самую большую найденную область.
                best = max(spill_candidates, key=lambda c: c["area"])

                spill_active[0] = (
                    0.5,  # Условная оценка, НЕ вероятность обнаружения.
                    {
                        "x": round(best["center_x"] / w, 3),
                        "y": round(best["center_y"] / h, 3),
                    },
                )

            # Вызываем на каждом кадре, в том числе без кандидатов.
            engine.process("spill_detected", spill_active, frame)

            if spill_candidates:
                spill_frame = draw_spill_overlay(
                    frame, spill_mask, spill_candidates
                )

        # --- отрисовка ---
        if spill_enabled and spill_candidates:
            frame = draw_spill_overlay(
                frame, spill_mask, spill_candidates
            )

        cv2.polylines(
            frame,
            [np.array(polygon_px, dtype=np.int32)],
            True,
            (255, 0, 0),
            2,
        )
        for obj in detections:
            in_zone = obj["track_id"] in active
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            label = f'{obj["class_name"]} #{obj["track_id"]} {obj["confidence"]:.2f}'
            cv2.rectangle(frame, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), color, 2)
            cv2.putText(frame, label, (obj["x1"], max(obj["y1"] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for p in ppe_dets:
            cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 200, 255), 1)
            cv2.putText(frame, f'{p["class_name"]} {p["confidence"]:.2f}',
                        (p["x1"], max(p["y1"] - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for d in fire_dets:
            if d["confidence"] >= fire_conf.get(d["class_name"], 1.0):
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 0, 255), 2)
                cv2.putText(frame, f'{d["class_name"]} {d["confidence"]:.2f}',
                            (d["x1"], max(d["y1"] - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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