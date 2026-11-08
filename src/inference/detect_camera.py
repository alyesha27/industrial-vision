import argparse
import json
import os
import signal
import time
import logging
from pathlib import Path

import cv2
import numpy as np
from src.analytics.spill import detect_spill, draw_spill_overlay
from src.analytics.object_events import prepare_object_events
from src.config import load_config
from src.inference.detector import Detector
from src.capture.video_capture import VideoSource, FPSMeter
from src.events.event_engine import EventEngine
from src.zones.geometry import bottom_center, to_pixels
from src.monitoring.metrics import Metrics
from src.monitoring.logger import setup_logging
from src.analytics.ppe import ppe_violations

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None,
                    help="0 — камера, либо путь к видеофайлу (по умолчанию из конфига)")
parser.add_argument("--no-show", action="store_true",
                    help="без окна с видео (Docker, плата без экрана)")
args = parser.parse_args()
HEADLESS = args.no_show or os.environ.get("HEADLESS") == "1"

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
PERSON_VEHICLE_CLASSES = {"person", "car", "truck", "bus"}
PPE_CLASSES = {"helmet", "head", "vest"}
# класс модели -> тип события
HAZARD_EVENT = {"fire": "fire_detected", "smoke": "smoke_detected", "water": "spill_detected"}

# Корректная остановка по SIGTERM (docker stop) и по Q
stop_requested = False


def _request_stop(signum, _frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, _request_stop)


def write_status(camera, fps, inference_ms, model="loaded"):
    data = {
        "camera": camera,
        "fps": round(fps, 1),
        "inference_ms": round(inference_ms, 1),
        "model": model,
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

# ---------------------------------------------------------------- модели
models_cfg = cfg["models"]
unified_path = models_cfg.get("unified_path")
UNIFIED = bool(unified_path and Path(unified_path).exists())

detector = ppe_detector = fire_detector = unified = None
hazard_conf = {}

if UNIFIED:
    # Одна модель на все классы: person, car, truck, bus, helmet, head, vest, fire, smoke, water
    class_conf = dict(models_cfg["class_confidence"])
    unified = Detector(unified_path, tracker=models_cfg["tracker"], class_conf=class_conf)
    hazard_conf = {c: class_conf[c] for c in HAZARD_EVENT if c in class_conf}
    ppe_enabled = True
    hazard_enabled = True
    log.info("Единая модель загружена: %s (классы: %s)", unified_path,
             ", ".join(unified.model.names.values()))
else:
    # Режим из трёх моделей (если единая модель ещё не обучена)
    detector = Detector(
        models_cfg["detector_path"],
        confidence=models_cfg["person_confidence"],
        tracker=models_cfg["tracker"],
    )
    ppe_path = models_cfg.get("ppe_path")
    if ppe_path and Path(ppe_path).exists():
        ppe_detector = Detector(ppe_path, confidence=models_cfg["ppe_confidence"], track=False)
        log.info("PPE-модель загружена: %s", ppe_path)
    else:
        log.warning("PPE-модель не найдена (%s), контроль СИЗ отключён", ppe_path)

    fire_path = models_cfg.get("fire_smoke_path")
    if fire_path and Path(fire_path).exists():
        hazard_conf = {"fire": models_cfg["fire_confidence"],
                       "smoke": models_cfg["smoke_confidence"]}
        fire_detector = Detector(fire_path, confidence=min(hazard_conf.values()), track=False)
        log.info("Модель дыма/огня загружена: %s", fire_path)
    else:
        log.warning("Модель дыма/огня не найдена (%s), контроль отключён", fire_path)
    ppe_enabled = ppe_detector is not None
    hazard_enabled = fire_detector is not None

# Разлив: если единая модель умеет класс water, эвристика не нужна
use_spill_heuristic = spill_enabled and not (UNIFIED and "water" in hazard_conf)

extra_every = models_cfg.get("extra_every", 1)

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
anchor_mode = cfg["zones"].get("anchor_mode", {})
min_person_height = models_cfg.get("min_person_height", 120)

camera_lost = False
last_status = 0.0
fps, inference_ms = 0.0, 0.0
frame_idx = 0
detections, ppe_dets, hazard_dets = [], [], []
spill_mask = None
spill_candidates = []
EVENT_LABEL_SECONDS = 3

try:
    source.open()
    log.info("Источник подключён")
    print(f"Источник подключён ({src_value}). "
          + ("Остановка: Ctrl+C." if HEADLESS else "Нажмите Q для выхода."))

    while not stop_requested:
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

        # --- inference ---
        start = metrics.start_timer()
        if UNIFIED:
            # один проход модели на кадре, затем разбор по классам
            all_dets = unified.predict(frame)
            detections = [d for d in all_dets if d["class_name"] in PERSON_VEHICLE_CLASSES]
            ppe_dets = [d for d in all_dets if d["class_name"] in PPE_CLASSES]
            hazard_dets = [d for d in all_dets if d["class_name"] in HAZARD_EVENT]
        else:
            detections = detector.predict(frame)
            if frame_idx % extra_every == 0:
                ppe_dets = ppe_detector.predict(frame) if ppe_detector else []
                hazard_dets = fire_detector.predict(frame) if fire_detector else []
            frame_idx += 1
        inference_ms = metrics.get_inference_time(start)

        # --- аналитика 1: люди и транспорт ---
        person_active, person_zone_active, vehicle_zone_active = prepare_object_events(
            detections, polygon_px, w, h, anchor_mode=anchor_mode,
        )
        engine.process("person_detected", person_active, frame)
        engine.process("person_in_restricted_zone", person_zone_active, frame)
        engine.process("vehicle_in_restricted_zone", vehicle_zone_active, frame)

        # Для отрисовки красным отмечаем объекты в запретной зоне
        active = person_zone_active

        # --- аналитика 2: нарушения СИЗ ---
        if ppe_enabled:
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

        # --- аналитика 3: огонь, дым, вода/разлив (по кадру, без Track ID) ---
        if hazard_enabled:
            for cls, event_type in HAZARD_EVENT.items():
                if cls not in hazard_conf:
                    continue
                hits = [d for d in hazard_dets
                        if d["class_name"] == cls and d["confidence"] >= hazard_conf[cls]]
                act = {}
                if hits:
                    best = max(hits, key=lambda d: d["confidence"])
                    cx = (best["x1"] + best["x2"]) // 2
                    cy = (best["y1"] + best["y2"]) // 2
                    act[0] = (best["confidence"], {"x": round(cx / w, 3), "y": round(cy / h, 3)})
                # вызываем каждый кадр, чтобы окно N из M обновлялось
                engine.process(event_type, act, frame)

        # --- аналитика 4: эвристика разливов (только если нет класса water) ---
        if use_spill_heuristic:
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

        # --- отрисовка (без окна не нужна, но снимки событий уже сохранены) ---
        if use_spill_heuristic and spill_candidates:
            frame = draw_spill_overlay(frame, spill_mask, spill_candidates)

        cv2.polylines(frame, [np.array(polygon_px, dtype=np.int32)], True, (255, 0, 0), 2)

        for obj in detections:
            track_id = obj["track_id"]
            in_zone = track_id in active
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            label = f'{obj["class_name"]} #{track_id} {obj["confidence"]:.2f}'

            cv2.rectangle(frame, (obj["x1"], obj["y1"]), (obj["x2"], obj["y2"]), color, 2)
            cv2.putText(frame, label, (obj["x1"], max(obj["y1"] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Последнее событие для этого Track ID
            now = time.time()
            for event in reversed(engine.recent_events):
                if event["track_id"] != track_id:
                    continue
                if now - event["timestamp"] > EVENT_LABEL_SECONDS:
                    continue
                text_y = min(obj["y2"] + 22, h - 10)
                cv2.putText(frame, event["event_type"], (obj["x1"], text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                break

        for p in ppe_dets:
            cv2.rectangle(frame, (p["x1"], p["y1"]), (p["x2"], p["y2"]), (0, 200, 255), 1)
            cv2.putText(frame, f'{p["class_name"]} {p["confidence"]:.2f}',
                        (p["x1"], max(p["y1"] - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        for d in hazard_dets:
            if d["confidence"] >= hazard_conf.get(d["class_name"], 1.0):
                color = (255, 128, 0) if d["class_name"] == "water" else (0, 0, 255)
                cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, 2)
                cv2.putText(frame, f'{d["class_name"]} {d["confidence"]:.2f}',
                            (d["x1"], max(d["y1"] - 8, 15)),
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

        if not HEADLESS:
            cv2.imshow("Industrial Vision - Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

except Exception:
    log.exception("Критическая ошибка")

finally:
    source.close()
    if not HEADLESS:
        cv2.destroyAllWindows()
    write_status("stopped", 0.0, 0.0)
