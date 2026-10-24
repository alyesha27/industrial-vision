
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo26n.pt", confidence=0.5):
        # Загружаем готовую модель
        self.model = YOLO(model_path)
        self.confidence = confidence

    def predict(self, frame):
        # Ищем объекты на одном кадре
        results = self.model.predict(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        # Получаем найденные объекты
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            detections.append({
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        return detections
