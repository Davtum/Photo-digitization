import numpy as np
import pytest

from facade_digitizer.geometry.vanishing import detect_segments
from tests.test_synth import make_scene


def test_detects_segments_on_synthetic_facade():
    img = make_scene().render()
    segs = detect_segments(img)
    assert segs.shape[1] == 4
    assert len(segs) >= 8  # четыре стороны стены плюс контуры проёма


def test_filters_short_segments():
    img = make_scene().render()
    long_only = detect_segments(img, min_length_px=500.0)
    all_segs = detect_segments(img, min_length_px=10.0)
    assert len(long_only) < len(all_segs)


def test_returns_empty_on_blank_image():
    blank = np.full((400, 400), 128, dtype=np.uint8)
    assert len(detect_segments(blank)) == 0


def test_empty_result_keeps_four_column_shape():
    """Пустой результат обязан быть формы (0, 4), а не просто пустым.

    Потребитель складывает выходы разных снимков через np.vstack и читает
    столбцы [x1, y1, x2, y2]. Массив формы (0, 2) проходит проверку len() == 0,
    но ломает конкатенацию и разбор столбцов уже у потребителя.
    """
    blank = np.full((400, 400), 128, dtype=np.uint8)
    assert detect_segments(blank).shape == (0, 4)


def test_rejects_multichannel_image():
    """Трёхканальный вход отвергается своей ошибкой, а не ошибкой OpenCV.

    Без явной проверки размерности LSD падает внутренней cv2.error, и вызывающий
    код не отличает неверный формат входа от сбоя детектора.
    """
    colour = np.zeros((400, 400, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="одноканальное"):
        detect_segments(colour)
