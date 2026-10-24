
import json
from pathlib import Path

from fastapi import FastAPI

app = FastAPI(title="Industrial Vision API")

# Путь к файлу событий в корне проекта
EVENTS_FILE = Path("events/events.json")


@app.get("/")
def home():
    return {
        "message": "Industrial Vision API работает"
    }


@app.get("/events")
def get_events():
    # Если событий пока нет, возвращаем пустой список
    if not EVENTS_FILE.exists():
        return []

    # Читаем сохранённые события
    with EVENTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }
