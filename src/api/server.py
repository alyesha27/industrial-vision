import json
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Industrial Vision API")

EVENTS_DIR = Path("events")
CONFIG_FILE = Path("configs/configuration.yaml")


def load_events():
    events = []

    for file in EVENTS_DIR.glob("*/*/*/*/event.json"):
        with file.open("r", encoding="utf-8") as f:
            events.append(json.load(f))

    return sorted(events, key=lambda e: e["timestamp"])


def load_config():
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config):
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False
        )


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


@app.get("/cameras")
def cameras():
    status_file = EVENTS_DIR / "status.json"

    if not status_file.exists():
        return [{"camera_id": "cam_01", "status": "unknown"}]

    camera_status = json.loads(
        status_file.read_text(encoding="utf-8")
    )

    return [{
        "camera_id": camera_status.get("camera_id", "cam_01"),
        "status": camera_status.get("camera", "unknown")
    }]


@app.post("/config/thresholds")
def update_thresholds(payload: dict):
    config = load_config()

    models = config.setdefault("models", {})

    allowed = {
        "person_confidence",
        "ppe_confidence",
        "fire_confidence",
        "smoke_confidence"
    }

    for name, value in payload.items():
        if name not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown threshold: {name}"
            )

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be a number"
            )

        if not 0 <= value <= 1:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be between 0 and 1"
            )

        models[name] = value

    save_config(config)

    return {"updated": payload}


@app.post("/config/zones")
def update_zones(payload: dict):
    config = load_config()

    zones = payload.get("zones")

    if not isinstance(zones, dict) or not zones:
        raise HTTPException(
            status_code=400,
            detail="Expected a non-empty zones object"
        )

    for zone_name, points in zones.items():
        if not isinstance(points, list) or len(points) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Zone {zone_name} must contain at least 3 points"
            )

        for point in points:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and 0 <= v <= 1
                    for v in point
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid coordinates in zone {zone_name}"
                )

    config["zones"] = zones
    save_config(config)

    return {"updated": {"zones": zones}}