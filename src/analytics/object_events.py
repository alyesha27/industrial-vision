from src.zones.geometry import bottom_center, point_in_polygon


VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}


def prepare_object_events(detections, polygon_px, width, height):
    """Подготавливает события обнаружения человека и транспорта.

    Возвращает:
    1. person_active — все обнаруженные люди с Track ID;
    2. person_zone_active — люди внутри запретной зоны;
    3. vehicle_zone_active — транспорт внутри запретной зоны.

    Формат значений совместим с EventEngine:
    track_id: (confidence, {"x": ..., "y": ...})
    """
    person_active = {}
    person_zone_active = {}
    vehicle_zone_active = {}

    for obj in detections:
        track_id = obj.get("track_id")
        if track_id is None:
            continue

        class_name = str(obj.get("class_name", "")).lower()
        point = bottom_center(obj)

        coordinates = {
            "x": round(point[0] / width, 3),
            "y": round(point[1] / height, 3),
        }
        event_data = (obj["confidence"], coordinates)

        inside_zone = point_in_polygon(point, polygon_px)

        if class_name == "person":
            person_active[track_id] = event_data

            if inside_zone:
                person_zone_active[track_id] = event_data

        elif class_name in VEHICLE_CLASSES and inside_zone:
            vehicle_zone_active[track_id] = event_data

    return person_active, person_zone_active, vehicle_zone_active
