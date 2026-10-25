import json

from src.events.event_engine import EventEngine

ZONE = "person_in_restricted_zone"
ACTIVE = {1: (0.9, {"x": 0.5, "y": 0.5})}


def make(tmp_path, cooldown=10):
    return EventEngine("cam_test", root=str(tmp_path), n=3, m=5, cooldown_sec=cooldown)


def files(tmp_path):
    return list(tmp_path.glob("*/*/*/*/event.json"))


def test_no_event_before_confirmation(tmp_path):
    engine = make(tmp_path)
    for _ in range(4):
        engine.process(ZONE, ACTIVE, None)
    assert files(tmp_path) == []


def test_event_created_once_per_track(tmp_path):
    engine = make(tmp_path)
    for _ in range(20):
        engine.process(ZONE, ACTIVE, None)
    assert len(files(tmp_path)) == 1


def test_event_json_has_required_fields(tmp_path):
    engine = make(tmp_path)
    for _ in range(5):
        engine.process(ZONE, ACTIVE, None)
    data = json.loads(files(tmp_path)[0].read_text(encoding="utf-8"))
    for key in ("event_id", "event_type", "timestamp", "camera_id",
                "confidence", "track_id", "coordinates", "snapshot", "status"):
        assert key in data
    assert data["event_type"] == ZONE
    assert data["camera_id"] == "cam_test"
    assert data["track_id"] == 1


def test_cooldown_blocks_second_track(tmp_path):
    engine = make(tmp_path, cooldown=10)
    two = {1: (0.9, {}), 2: (0.8, {})}
    for _ in range(5):
        engine.process(ZONE, two, None)
    assert len(files(tmp_path)) == 1


def test_new_event_after_leaving_zone(tmp_path):
    engine = make(tmp_path, cooldown=0)
    for _ in range(5):
        engine.process(ZONE, ACTIVE, None)
    for _ in range(5):
        engine.process(ZONE, {}, None)  # человек вышел
    for _ in range(5):
        engine.process(ZONE, ACTIVE, None)  # вошёл снова
    assert len(files(tmp_path)) == 2


def test_system_event_without_snapshot(tmp_path):
    engine = make(tmp_path)
    event = engine.emit("camera_stream_lost", 1.0, None, {})
    assert event.snapshot is None
    assert len(files(tmp_path)) == 1