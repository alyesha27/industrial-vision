
import cv2
import numpy as np


def detect_spill(
    frame,
    min_area=1200,
    min_area_ratio=0.0005,
    max_area_ratio=0.20,
    min_saturation=35,
    max_value=175,
    ground_start_ratio=0.45,
):
    """
    Экспериментальное выделение кандидатов в разливы.

    Используются цветовые признаки и морфологическая обработка.
    Это эвристика OpenCV, а не обученная модель.
    """

    height, width = frame.shape[:2]
    frame_area = height * width

    # Переводим изображение в HSV:
    # H — оттенок, S — насыщенность, V — яркость.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Ищем тёмные малонасыщенные и цветные области.
    dark = (value < max_value) & (saturation < 100)
    colored = (saturation >= min_saturation) & (value < 235)

    mask = ((dark | colored).astype(np.uint8)) * 255

    # Анализируем преимущественно нижнюю часть кадра,
    # где обычно находится пол.
    ground_start = int(height * ground_start_ratio)
    mask[:ground_start, :] = 0

    # Закрываем небольшие пробелы и удаляем мелкий шум.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_pixels = max(min_area, int(frame_area * min_area_ratio))
    max_pixels = int(frame_area * max_area_ratio)

    result_mask = np.zeros((height, width), dtype=np.uint8)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Отбрасываем слишком маленькие и слишком большие области.
        if area < min_pixels or area > max_pixels:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 10 or h < 10:
            continue

        # Оставляем маску только для принятых областей.
        cv2.drawContours(
            result_mask, [contour], -1, 255, thickness=cv2.FILLED
        )

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": int(area),
            "center_x": x + w // 2,
            "center_y": y + h // 2,
        })

    return result_mask, candidates


def draw_spill_overlay(frame, mask, candidates):
    """Рисует найденные области на кадре."""

    result = frame.copy()
    overlay = result.copy()

    # Полупрозрачная зелёная маска.
    overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)

    for candidate in candidates:
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

        cv2.rectangle(
            result, (x, y), (x + w, y + h), (0, 255, 0), 2
        )
        cv2.putText(
            result,
            "spill candidate",
            (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return result
