import cv2
import numpy as np


def bottom_center(obj):
    # Нижняя центральная точка bbox: примерно там, где человек стоит на полу
    return ((obj["x1"] + obj["x2"]) // 2, obj["y2"])


def to_pixels(polygon_norm, width, height):
    # Переводим нормированные координаты [0;1] в пиксели текущего кадра
    return [(int(x * width), int(y * height)) for x, y in polygon_norm]


def point_in_polygon(point, polygon_px):
    contour = np.array(polygon_px, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0