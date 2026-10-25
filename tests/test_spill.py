
import cv2
import numpy as np

from src.analytics.spill import detect_spill


def test_detect_spill_on_dark_area():
    # Светлый кадр имитирует пол.
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)

    # Тёмная область в нижней части кадра имитирует разлив.
    cv2.ellipse(
        frame,
        (320, 350),
        (90, 35),
        0,
        0,
        360,
        (50, 50, 50),
        -1,
    )

    mask, candidates = detect_spill(frame)

    assert mask.shape == frame.shape[:2]
    assert len(candidates) > 0
    assert np.count_nonzero(mask) > 0


def test_detect_spill_on_uniform_frame():
    # Однородный серый кадр: нет заметного кандидата на разлив.
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)

    mask, candidates = detect_spill(frame)

    assert mask.shape == frame.shape[:2]
    assert len(candidates) == 0
    assert np.count_nonzero(mask) == 0
