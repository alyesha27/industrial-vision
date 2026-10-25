import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2

from src.analytics.temporal import TemporalConfirm

log = logging.getLogger("vision.events")


@dataclass
class Event:
    event_id: str
    event_type: str
    timestamp: str
    camera_id: str
    confidence: float
    track_id: int | None
    coordinates: dict
    snapshot: str | None
    status: str = "new"


class EventEngine:
    def __init__(self, camera_id, root="events", n=3, m=5, cooldown_sec=10):
        self.camera_id = camera_id
        self.root = Path(root)
        self.n, self.m = n, m
        self.cooldown_sec = cooldown_sec
        self.recent_events = []

        self.windows = {}    # event_type -> {track_id: TemporalConfirm}
        self.fired = set()   # (event_type, track_id), по которым событие уже создано
        self.last_emit = {}  # event_type -> время последнего события

    def process(self, event_type, active, frame):
        """active: {track_id: (confidence, coordinates)} для объектов,
        у которых условие выполнено в ЭТОМ кадре."""
        windows = self.windows.setdefault(event_type, {})

        for track_id in active:
            windows.setdefault(track_id, TemporalConfirm(self.n, self.m))

        for track_id, tc in list(windows.items()):
            present = track_id in active
            confirmed = tc.update(present)

            # Трек давно не в зоне: забываем его, он сможет создать новое событие
            if not present and not any(tc.window):
                del windows[track_id]
                self.fired.discard((event_type, track_id))
                continue

            key = (event_type, track_id)
            if present and confirmed and key not in self.fired:
                if time.perf_counter() - self.last_emit.get(event_type, -1e9) < self.cooldown_sec:
                    continue  # cooldown
                conf, coords = active[track_id]
                self.emit(event_type, conf, track_id, coords, frame)
                self.fired.add(key)

    def emit(self, event_type, confidence, track_id, coordinates, frame=None):
        event = Event(
            event_id=str(uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            camera_id=self.camera_id,
            confidence=round(float(confidence), 3),
            track_id=track_id,
            coordinates=coordinates,
            snapshot=None,
        )

        # events/YYYY/MM/DD/<uuid>/
        now = datetime.now()
        folder = self.root / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / event.event_id
        folder.mkdir(parents=True, exist_ok=True)

        if frame is not None:
            snapshot_path = folder / "snapshot.jpg"
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                buf.tofile(str(snapshot_path))  # безопасно для не-ASCII путей
                event.snapshot = snapshot_path.as_posix()

        with (folder / "event.json").open("w", encoding="utf-8") as file:
            json.dump(asdict(event), file, ensure_ascii=False, indent=4)

        self.last_emit[event_type] = time.perf_counter()
        log.info("EVENT %s track=%s conf=%.2f", event_type, track_id, confidence)
        self.recent_events.append({
            "event_type": event_type,
            "track_id": track_id,
            "coordinates": coordinates,
            "timestamp": time.time(),
        })

        # Храним только последние 100 событий
        self.recent_events = self.recent_events[-100:]
        return event