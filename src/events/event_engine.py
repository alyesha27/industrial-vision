
import json
import time
from pathlib import Path
from datetime import datetime


class EventEngine:
    def __init__(self, confirmation_time=2):
        self.confirmation_time = confirmation_time
        self.start_time = None
        self.event_created = False

        self.events_file = Path("events/events.json")
        self.events_file.parent.mkdir(parents=True, exist_ok=True)

    def update(self, person_in_roi):
        current_time = time.perf_counter()

        # Если человек находится в зоне
        if person_in_roi:
            if self.start_time is None:
                self.start_time = current_time

            duration = current_time - self.start_time

            # Подтверждаем событие после 2 секунд
            if duration >= self.confirmation_time and not self.event_created:
                event = {
                    "event_type": "person_in_roi",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "duration_seconds": round(duration, 2),
                    "description": "Человек находится в контролируемой зоне"
                }

                # Читаем предыдущие события
                if self.events_file.exists():
                    with self.events_file.open("r", encoding="utf-8") as file:
                        events = json.load(file)
                else:
                    events = []

                events.append(event)

                # Сохраняем обновлённый список
                with self.events_file.open("w", encoding="utf-8") as file:
                    json.dump(events, file, ensure_ascii=False, indent=4)

                print("Событие зарегистрировано:", event)
                self.event_created = True

        else:
            # Если человек вышел из зоны, начинаем отсчёт заново
            self.start_time = None
            self.event_created = False
