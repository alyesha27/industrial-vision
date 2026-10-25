from src.zones.geometry import point_in_polygon, bottom_center, to_pixels

POLY = [(0, 0), (100, 0), (100, 100), (0, 100)]


def test_point_inside():
    assert point_in_polygon((50, 50), POLY)


def test_point_outside():
    assert not point_in_polygon((150, 50), POLY)


def test_point_on_border_counts_as_inside():
    assert point_in_polygon((100, 50), POLY)


def test_bottom_center():
    obj = {"x1": 10, "y1": 20, "x2": 50, "y2": 120}
    assert bottom_center(obj) == (30, 120)


def test_to_pixels():
    assert to_pixels([[0.5, 0.25]], 200, 100) == [(100, 25)]