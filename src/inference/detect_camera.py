import cv2

from src.inference.detector import Detector
from src.capture.video_capture import VideoSource, FPSMeter
from src.events.event_engine import EventEngine
from src.zones.roi import ROI
from src.monitoring.metrics import Metrics

source = VideoSource(source=0)
detector = Detector(confidence=0.5)
fps_meter = FPSMeter()
metrics = Metrics()

# Создаём контролируемую зону и обработчик событий
roi = ROI(100, 100, 500, 400)
event_engine = EventEngine(confirmation_time=2)

try:
    source.open()
    print("Камера подключена. Нажмите Q для выхода.")

    while True:
        frame = source.read()

        # Запускаем обнаружение объектов
        start = metrics.start_timer()
        detections = detector.predict(frame)
        inference_ms = metrics.get_inference_time(start)

        cpu = metrics.get_cpu()
        ram = metrics.get_ram()

        # Рисуем контролируемую зону
        cv2.rectangle(
            frame,
            (roi.x1, roi.y1),
            (roi.x2, roi.y2),
            (255, 0, 0),
            2
        )

        # Проверяем, находится ли человек в зоне
        person_in_roi = False

        for obj in detections:
            if obj["class_name"] == "person":
                center_x = (obj["x1"] + obj["x2"]) // 2
                center_y = (obj["y1"] + obj["y2"]) // 2

                if roi.contains(center_x, center_y):
                    person_in_roi = True
                    break

        # Передаём результат в обработчик событий
        event_engine.update(person_in_roi)

        # Рисуем рамки и подписи обнаруженных объектов
        for obj in detections:
            x1 = obj["x1"]
            y1 = obj["y1"]
            x2 = obj["x2"]
            y2 = obj["y2"]

            label = (
                f'{obj["class_name"]} '
                f'{obj["confidence"]:.2f}'
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Показываем FPS
        fps = fps_meter.tick()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Inference: {inference_ms:.1f} ms",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"CPU: {cpu:.1f}% RAM: {ram:.0f} MB",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("Industrial Vision - Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as error:
    print(f"Ошибка: {error}")

finally:
    source.close()
    cv2.destroyAllWindows()
