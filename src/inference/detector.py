from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo26n.pt", confidence=0.5, tracker="bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker

    def predict(self, frame):
        # track() = detect + ByteTrack; persist=True сохраняет ID между кадрами
        results = self.model.track(
            frame,
            conf=self.confidence,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id.item()) if box.id is not None else None

            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": self.model.names[class_id],
                "confidence": float(box.conf.item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        return detections