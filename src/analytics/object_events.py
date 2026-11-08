from src.zones.anchors import bottom_center_point, zone_hit

VEHICLE_CLASSES = ("car", "truck", "bus")
DEFAULT_ANCHOR_MODE = {"person": "bottom_center", "vehicle": "any"}


def prepare_object_events(detections, polygon_px, w, h,
                          anchor_mode=None, vehicle_classes=VEHICLE_CLASSES):
    """Готовит словари {track_id: (confidence, {"x", "y"})} для Event Engine.

    Возвращает: person_active, person_zone_active, vehicle_zone_active.
    anchor_mode: {"person": режим, "vehicle": режим}, см. src/zones/anchors.py
    """
    modes = dict(DEFAULT_ANCHOR_MODE)
    modes.update(anchor_mode or {})

    person_active = {}
    person_zone_active = {}
    vehicle_zone_active = {}

    def norm(point):
        return {"x": round(point[0] / w, 3), "y": round(point[1] / h, 3)}

    for obj in detections:
        track_id = obj.get("track_id")
        if track_id is None:
            continue
        name = obj["class_name"]
        conf = obj["confidence"]

        if name == "person":
            person_active[track_id] = (conf, norm(bottom_center_point(obj)))
            point = zone_hit(obj, polygon_px, modes["person"])
            if point is not None:
                person_zone_active[track_id] = (conf, norm(point))
        elif name in vehicle_classes:
            point = zone_hit(obj, polygon_px, modes["vehicle"])
            if point is not None:
                vehicle_zone_active[track_id] = (conf, norm(point))

    return person_active, person_zone_active, vehicle_zone_active
