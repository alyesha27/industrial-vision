from src.analytics.ppe import ppe_violations


def person(h=300):
    return {"x1": 0, "y1": 0, "x2": 100, "y2": h}


def det(name, x=50, y=50):
    return {"class_name": name, "x1": x - 5, "y1": y - 5, "x2": x + 5, "y2": y + 5}


def test_all_ppe_present():
    assert ppe_violations(person(), [det("helmet"), det("vest")]) == set()


def test_helmet_missing_when_bare_head_seen():
    result = ppe_violations(person(), [det("head"), det("vest")])
    assert result == {"ppe_helmet_missing"}


def test_vest_missing():
    result = ppe_violations(person(), [det("helmet")])
    assert result == {"ppe_vest_missing"}


def test_small_person_is_skipped():
    assert ppe_violations(person(h=80), [det("head")]) == set()


def test_ppe_of_another_person_is_ignored():
    result = ppe_violations(person(), [det("helmet"), det("vest", x=500)])
    assert "ppe_vest_missing" in result