"""Опорные точки рамки объекта и проверка попадания в зону.

Режимы (mode):
  bottom_center - нижняя центральная точка (точка контакта с полом)
  center        - геометрический центр рамки
  any           - срабатывает, если в зоне хотя бы одна из двух точек
"""
import cv2
import numpy as np

MODES = ("bottom_center", "center", "any")


def bottom_center_point(obj):
    return ((obj["x1"] + obj["x2"]) / 2.0, float(obj["y2"]))


def bbox_center(obj):
    return ((obj["x1"] + obj["x2"]) / 2.0, (obj["y1"] + obj["y2"]) / 2.0)


def _inside(point, polygon_px):
    poly = np.array(polygon_px, dtype=np.int32)
    # >= 0: граница полигона считается внутренней областью
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def zone_hit(obj, polygon_px, mode="bottom_center"):
    """Возвращает точку (x, y), которая оказалась в зоне, или None."""
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим привязки к зоне: {mode}")
    candidates = {
        "bottom_center": [bottom_center_point(obj)],
        "center": [bbox_center(obj)],
        "any": [bottom_center_point(obj), bbox_center(obj)],
    }[mode]
    for point in candidates:
        if _inside(point, polygon_px):
            return point
    return None


def in_zone(obj, polygon_px, mode="bottom_center"):
    return zone_hit(obj, polygon_px, mode) is not None
