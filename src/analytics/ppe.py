def _center(d):
    return ((d["x1"] + d["x2"]) // 2, (d["y1"] + d["y2"]) // 2)


def _inside(point, person):
    x, y = point
    return person["x1"] <= x <= person["x2"] and person["y1"] <= y <= person["y2"]


def ppe_violations(person, ppe_dets, min_person_height=120):
    """Нарушения для ОДНОГО человека. ppe_dets: детекции классов helmet / head / vest.
    СИЗ относится к человеку, если центр её bbox лежит внутри bbox человека."""
    # Слишком маленький человек: проверка ненадёжна, события не создаём
    if person["y2"] - person["y1"] < min_person_height:
        return set()

    names = {d["class_name"] for d in ppe_dets if _inside(_center(d), person)}
    violations = set()

    # Каска: нарушение, только если видна голова без каски (класс head)
    if "head" in names and "helmet" not in names:
        violations.add("ppe_helmet_missing")

    # Жилет: нарушение, если жилета нет внутри bbox человека
    if "vest" not in names:
        violations.add("ppe_vest_missing")

    return violations