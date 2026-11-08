from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolo26n.pt", confidence=0.5,
                 tracker="bytetrack.yaml", track=True, class_conf=None):
        """class_conf: {имя_класса: порог}. Если задан, модель запускается с
        минимальным порогом, а затем каждая детекция фильтруется по порогу
        своего класса (нужно для единой модели с разными классами)."""
        self.model = YOLO(model_path)
        self.class_conf = dict(class_conf) if class_conf else {}
        self.confidence = min(self.class_conf.values()) if self.class_conf else confidence
        self.tracker = tracker
        self.track = track  # True: с Track ID, False: просто детекция

    def predict(self, frame):
        if self.track:
            # track() = detect + ByteTrack; persist=True сохраняет ID между кадрами
            results = self.model.track(
                frame,
                conf=self.confidence,
                tracker=self.tracker,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.confidence,
                verbose=False,
            )

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            confidence = float(box.conf.item())

            if self.class_conf and confidence < self.class_conf.get(class_name, self.confidence):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id.item()) if box.id is not None else None

            detections.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

        return detections
