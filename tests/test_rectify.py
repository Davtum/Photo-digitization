import numpy as np
import pytest

from facade_digitizer.geometry.homography import homography_from_vanishing_points
from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from facade_digitizer.pipeline.rectify import rectify
from tests.test_homography import CORNERS, apply
from tests.test_synth import SIZE, K, make_scene


def prepared(mm_per_px=4.0):
    sc = make_scene()
    vh, vv, _ = estimate_vanishing_points(detect_segments(sc.render()), SIZE, K)
    H = homography_from_vanishing_points(vh.point, vv.point, K, SIZE)
    r = apply(H, sc.project(CORNERS))
    mmu = 20000.0 / np.linalg.norm(r[1] - r[0])
    return sc, rectify(sc.render(), H, mmu, mm_per_px), H, mmu


def test_output_size_follows_the_convention():
    """Размер задаётся миллиметрами на пиксель, а не безразмерным размахом."""
    _, r4, _, _ = prepared(mm_per_px=4.0)
    _, r8, _, _ = prepared(mm_per_px=8.0)
    assert r4.image.shape[0] > 1000
    assert r4.image.shape[0] / r8.image.shape[0] == pytest.approx(2.0, rel=0.02)


def test_known_distance_measures_correctly_in_the_raster():
    """Мера длины: 20 м фасада занимают 20000/mm_per_px пикселей."""
    sc, rect, _, _ = prepared(mm_per_px=4.0)
    p = np.column_stack([sc.project(CORNERS), np.ones(4)]) @ rect.H.T
    p = p[:, :2] / p[:, 2:3]
    assert np.linalg.norm(p[1] - p[0]) * rect.mm_per_px == pytest.approx(20000.0, rel=0.01)


def test_absurd_scale_is_rejected_loudly():
    sc, _, H, mmu = prepared()
    with pytest.raises(ValueError):
        rectify(sc.render(), H, mmu, mm_per_px=1e-4)


def test_valid_mask_matches_image():
    _, rect, _, _ = prepared()
    assert rect.valid_mask.shape == rect.image.shape
    assert 0.1 < rect.valid_mask.mean() <= 1.0


# Причины отказа. Подстроки подобраны так, чтобы КАЖДАЯ подходила только к своему
# сообщению: `pytest.raises(ValueError)` без `match=` принял бы любой ValueError,
# в том числе пришедший совсем из другого места.
REFUSAL_NON_POSITIVE = "mm_per_px должен быть положительным"
REFUSAL_IMPOSSIBLE_SIZE = "невозможный размер ректифицированного растра"

#: Метасимволы регулярного выражения. `match=` — это не подстрока, а шаблон поиска:
#: вертикальная черта в причине читалась бы как альтернатива и подошла бы к чужому
#: сообщению. `re.escape` в роли этой проверки не годится: он экранирует и пробел.
REGEX_META = set(r"\^$.|?*+()[]{}")


def frame_image_in_raster(scene, rect):
    """Образ четырёх углов кадра в ректифицированных ПИКСЕЛЯХ.

    Меритель независим от внутренностей `rectify`: гомография `rect.H` применяется
    как чёрный ящик, формула перехода из единиц в пиксели здесь не повторяется.
    """
    h, w = scene.render().shape[:2]
    return apply(rect.H, np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]]))


def test_raster_exactly_frames_the_image_content():
    """Растр — ограничивающий прямоугольник образа кадра: ни обрезки, ни полей.

    Тесты брифа переживали две мутации, потому что проверяли только высоту растра
    и её отношение при удвоении масштаба. Перестановка ширины с высотой оставляет и
    то, и другое в силе (растр 5406x4349 вместо 4349x5406), а потерянный сдвиг
    начала координат уводит содержимое в отрицательные координаты, не трогая
    расстояния. Здесь проверяется само свойство растра: образ кадра касается всех
    четырёх его сторон.
    """
    sc, rect, _, _ = prepared(mm_per_px=4.0)
    corners = frame_image_in_raster(sc, rect)
    out_h, out_w = rect.image.shape[:2]
    assert corners.min(axis=0) == pytest.approx([0.0, 0.0], abs=1.0)
    assert corners.max(axis=0) == pytest.approx([out_w, out_h], abs=1.0)


def test_valid_mask_covers_exactly_the_image_quadrilateral():
    """Доля истины в маске равна доле площади четырёхугольника-образа кадра.

    `0.1 < mask.mean() <= 1.0` из брифа пропускает маску, заполненную целиком
    истиной. Образ кадра — наклонный четырёхугольник, он занимает лишь часть своего
    ограничивающего прямоугольника (на синтетической сцене около 0.76), и маска
    обязана это показывать: охват не равен растру.
    """
    sc, rect, _, _ = prepared(mm_per_px=4.0)
    x, y = frame_image_in_raster(sc, rect).T
    area = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))     # формула площади Гаусса
    out_h, out_w = rect.image.shape[:2]
    assert rect.valid_mask.mean() == pytest.approx(area / (out_w * out_h), rel=0.02)
    assert rect.valid_mask.mean() < 1.0


def test_impossible_size_is_refused_by_its_own_reason_at_both_ends():
    """Отказ по размеру наступает ДО варпа, называет свою причину и ловит оба конца.

    `pytest.raises(ValueError)` из брифа отличает отказ лишь от полного молчания.
    Снятая охрана «убивалась» им случайно: абсурдный масштаб доходил до
    `cv2.warpPerspective`, и тот падал нехваткой памяти — исходом, зависящим от
    объёма памяти машины, а не от кода. Здесь проверяется именно отказ по своей
    причине, и с обеих сторон: растр, раздутый за MAX_SIDE_PX, и растр, схлопнутый
    в ноль, — тот самый случай 16x16 из редакции 1.
    """
    sc, _, H, mmu = prepared()
    for bad in (1e-4, 1e6):
        with pytest.raises(ValueError, match=REFUSAL_IMPOSSIBLE_SIZE):
            rectify(sc.render(), H, mmu, mm_per_px=bad)


def test_non_positive_mm_per_px_is_refused_by_its_own_reason():
    """Неположительный масштаб отвергается своей причиной, а не размером растра."""
    sc, _, H, mmu = prepared()
    for bad in (0.0, -4.0):
        with pytest.raises(ValueError, match=REFUSAL_NON_POSITIVE):
            rectify(sc.render(), H, mmu, mm_per_px=bad)


def test_refusal_reasons_are_literal_and_mutually_exclusive():
    """Подстроки причин не пересекаются: ни одна не подойдёт к чужому сообщению.

    Вертикальная черта в `match=` читается как альтернатива регулярного выражения,
    поэтому причины проверяются и на отсутствие метасимволов.
    """
    assert REFUSAL_NON_POSITIVE not in REFUSAL_IMPOSSIBLE_SIZE
    assert REFUSAL_IMPOSSIBLE_SIZE not in REFUSAL_NON_POSITIVE
    for reason in (REFUSAL_NON_POSITIVE, REFUSAL_IMPOSSIBLE_SIZE):
        assert not set(reason) & REGEX_META
