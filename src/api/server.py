import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Industrial Vision API")

EVENTS_DIR = Path("events")


def load_events():
    events = []
    for file in EVENTS_DIR.glob("*/*/*/*/event.json"):
        with file.open("r", encoding="utf-8") as f:
            events.append(json.load(f))
    return sorted(events, key=lambda e: e["timestamp"])


@app.get("/")
def home():
    return {"message": "Industrial Vision API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    file = EVENTS_DIR / "status.json"
    if not file.exists():
        return {"camera": "unknown", "model": "unknown"}
    return json.loads(file.read_text(encoding="utf-8"))


@app.get("/events")
def get_events():
    return load_events()


@app.get("/events/{event_id}")
def event_by_id(event_id: str):
    for item in load_events():
        if item["event_id"] == event_id:
            return item
    raise HTTPException(status_code=404, detail="Event not found")
