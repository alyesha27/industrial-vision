from src.zones.anchors import in_zone, zone_hit
from src.analytics.object_events import prepare_object_events

# зона: прямоугольник x 100..300, y 100..300
POLY = [(100, 100), (300, 100), (300, 300), (100, 300)]


def box(x1, y1, x2, y2, name="person", track_id=1, conf=0.9):
    return {"track_id": track_id, "class_name": name, "confidence": conf,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# центр (200, 250) внутри, низ (200, 350) снаружи
CENTER_IN_BOTTOM_OUT = box(150, 150, 250, 350)
# центр (200, 50) снаружи, низ (200, 150) внутри
BOTTOM_IN_CENTER_OUT = box(150, -50, 250, 150)


def test_bottom_center_ignores_center():
    assert not in_zone(CENTER_IN_BOTTOM_OUT, POLY, "bottom_center")
    assert in_zone(BOTTOM_IN_CENTER_OUT, POLY, "bottom_center")


def test_center_ignores_bottom():
    assert in_zone(CENTER_IN_BOTTOM_OUT, POLY, "center")
    assert not in_zone(BOTTOM_IN_CENTER_OUT, POLY, "center")


def test_any_triggers_on_either_point():
    assert in_zone(CENTER_IN_BOTTOM_OUT, POLY, "any")
    assert in_zone(BOTTOM_IN_CENTER_OUT, POLY, "any")


def test_object_far_outside_never_triggers():
    far = box(500, 500, 600, 600)
    for mode in ("bottom_center", "center", "any"):
        assert zone_hit(far, POLY, mode) is None


def test_prepare_events_person_mode_and_vehicle_mode():
    dets = [box(150, 150, 250, 350, "person", 1),
            box(150, 150, 250, 350, "car", 2)]
    _, person_zone, vehicle_zone = prepare_object_events(
        dets, POLY, 640, 480,
        anchor_mode={"person": "bottom_center", "vehicle": "any"})
    assert 1 not in person_zone      # человек: низ снаружи, режим bottom_center
    assert 2 in vehicle_zone         # машина: режим any, центр внутри

    _, person_zone, _ = prepare_object_events(
        dets, POLY, 640, 480, anchor_mode={"person": "center"})
    assert 1 in person_zone
