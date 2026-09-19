import re

import numpy as np
import pytest

from facade_digitizer.geometry.homography import homography_from_vanishing_points
from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from facade_digitizer.pipeline.rectify import MAX_SIDE_PX, attainable_mm_per_px, rectify
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
REFUSAL_MM_PER_RECT_UNIT = "mm_per_rect_unit должен быть положительным конечным числом"
REFUSAL_IMPOSSIBLE_SIZE = "невозможный размер ректифицированного растра"
REFUSAL_NOT_NUMERIC = "изображение: ожидался числовой массив numpy"
REFUSAL_NOT_GRAYSCALE = "изображение: ожидался двумерный полутоновый кадр"
REFUSAL_EMPTY_FRAME = "изображение: кадр пуст"

REFUSAL_PATTERNS = (REFUSAL_NON_POSITIVE, REFUSAL_MM_PER_RECT_UNIT, REFUSAL_IMPOSSIBLE_SIZE,
                    REFUSAL_NOT_NUMERIC, REFUSAL_NOT_GRAYSCALE, REFUSAL_EMPTY_FRAME)

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


#: Шаг наружу от границы — тысячная доля её самой. МНОЖИТЕЛЬ здесь не годится, и это
#: не придирка к стилю: прежняя проверка округлённых размеров принимала масштаб вплоть
#: до УДВОЕННОЙ верхней границы, поэтому проба «в десять раз больше» проскакивала
#: мягкую зону целиком и объявляла границу жёсткой, когда та таковой не была.
BOUNDARY_STEP = 1e-3


def test_range_boundaries_are_hard_and_symmetric():
    """Шаг внутрь — успех, шаг наружу — отказ по своей причине. С ОБЕИХ сторон.

    Нижняя граница была жёсткой и раньше: убавить тысячную — и отказ. Верхняя была
    мягкой: проверялись ОКРУГЛЁННЫЕ стороны растра, поэтому короткая сторона в 0.5
    единицы округлялась до единицы, и `rectify` молча возвращал растр 1x1 при
    масштабе почти вдвое выше заявленного потолка. Растр в один пиксель — то же
    молчаливое схлопывание, ради устранения которого переписывалась задача, только с
    другого конца диапазона.

    Заодно проверяется, что сами границы достижимы и означают именно то, что
    объявлено: на нижней длинная сторона растра равна ровно MAX_SIDE_PX, на верхней
    короткая — ровно одному пикселю.
    """
    sc, _, H, mmu = prepared()
    img = sc.render()
    lo, hi = attainable_mm_per_px(img, H, mmu)

    assert max(rectify(img, H, mmu, lo).image.shape) == MAX_SIDE_PX
    assert min(rectify(img, H, mmu, hi).image.shape) == 1
    assert max(rectify(img, H, mmu, lo * (1 + BOUNDARY_STEP)).image.shape) <= MAX_SIDE_PX
    assert min(rectify(img, H, mmu, hi * (1 - BOUNDARY_STEP)).image.shape) >= 1

    for outside in (lo * (1 - BOUNDARY_STEP), hi * (1 + BOUNDARY_STEP)):
        with pytest.raises(ValueError, match=REFUSAL_IMPOSSIBLE_SIZE):
            rectify(img, H, mmu, outside)


def test_non_positive_mm_per_px_is_refused_by_its_own_reason():
    """Неположительный масштаб отвергается своей причиной, а не размером растра."""
    sc, _, H, mmu = prepared()
    for bad in (0.0, -4.0):
        with pytest.raises(ValueError, match=REFUSAL_NON_POSITIVE):
            rectify(sc.render(), H, mmu, mm_per_px=bad)


def test_non_positive_mm_per_rect_unit_is_refused_by_its_own_reason():
    """Негодный mm_per_rect_unit называет СЕБЯ, а не размер растра.

    Раньше нечисловое значение доезжало до проверки размера: `nan` проваливал
    `1 <= out_w` (сравнение с `nan` ложно), и отказ отправлял вызывающего искать
    неисправность в гомографии и в доверии к плоскости — то есть не туда. Строка и
    None не добирались и до этого: `float("4.0") <= 0` поднимает TypeError, а отказ
    получался про внутренности функции.
    """
    sc, _, H, _ = prepared()
    for bad in (0.0, -11987.5, float("nan"), float("inf"), "11987.5", None, True):
        with pytest.raises(ValueError, match=REFUSAL_MM_PER_RECT_UNIT):
            rectify(sc.render(), H, bad, 4.0)


def test_impossible_size_message_names_the_attainable_range():
    """Отказ по размеру называет диапазон, а не оставляет вычислять его вызывающему.

    Сообщение «сторона превышает предел» вынуждало подбирать масштаб вслепую —
    ровно тот подбор, из которого вырос растр 16x16. Проба берётся ВЫШЕ верхней
    границы: там отказ ничего не выделяет, тогда как проба ниже нижней границы
    заставляла бы мутанта без охраны просить десятки петабайт и падать нехваткой
    памяти вместо того, чтобы отличаться поведением.
    """
    sc, _, H, mmu = prepared()
    img = sc.render()
    lo, hi = attainable_mm_per_px(img, H, mmu)
    assert lo < hi

    with pytest.raises(ValueError) as excinfo:
        rectify(img, H, mmu, hi * 4.0)
    assert f"{lo:.4g}" in str(excinfo.value)
    assert f"{hi:.4g}" in str(excinfo.value)


def test_degenerate_transform_says_that_no_scale_fits_at_all():
    """Вырожденный образ кадра: диапазон перевёрнут, и отказ говорит об этом прямо.

    Преобразование, схлопывающее всю высоту кадра в ноль, не допускает НИ ОДНОГО
    масштаба: нижняя граница выше верхней. Сообщать в этом случае «годится от 1.08
    до 0» значило бы отправить вызывающего перебирать несуществующие значения.
    """
    sc, _, _, mmu = prepared()
    flat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    lo, hi = attainable_mm_per_px(sc.render(), flat, mmu)
    assert hi < lo
    with pytest.raises(ValueError, match=REFUSAL_IMPOSSIBLE_SIZE) as excinfo:
        rectify(sc.render(), flat, mmu, 4.0)
    assert "нет вовсе" in str(excinfo.value)


def test_origin_px_is_where_the_given_rectified_point_lands():
    """origin_rect_px — образ origin_rect_units в пикселях растра, и ничто иное.

    Смысл аргумента закрепляется тестом, потому что перепутанное начало отсчёта
    сдвигает ВСЕ измеренные положения разом и на одну величину: результат остаётся
    правдоподобным и молча неверным. Меритель независим: та же точка пропускается
    через итоговую `rect.H` как через чёрный ящик.
    """
    sc, _, H, mmu = prepared()
    corner_units = apply(H, sc.project(CORNERS))[0]     # образ левого нижнего угла фасада
    rect = rectify(sc.render(), H, mmu, 4.0, origin_rect_units=tuple(corner_units))
    landed = apply(rect.H, sc.project(CORNERS))[0]
    assert rect.origin_rect_px == pytest.approx(tuple(landed), abs=1e-3)


def test_default_origin_is_the_rectified_zero_not_the_facade_corner():
    """Умолчание (0, 0) — нуль РЕКТИФИЦИРОВАННОЙ системы, а не угол фасада.

    Гомография ставит свой нуль туда, куда ей велят точки схода; на этой сцене он
    отстоит от левого нижнего угла фасада на тысячи пикселей. Принять умолчание за
    угол фасада — значит сдвинуть весь результат на это расстояние.
    """
    sc, _, H, mmu = prepared()
    corner_units = apply(H, sc.project(CORNERS))[0]
    default = rectify(sc.render(), H, mmu, 4.0)
    at_corner = rectify(sc.render(), H, mmu, 4.0, origin_rect_units=tuple(corner_units))
    apart = np.linalg.norm(np.array(default.origin_rect_px) - np.array(at_corner.origin_rect_px))
    assert apart > 1000.0


def test_colour_and_non_image_inputs_are_refused():
    """Конвейер одноканален: иной кадр отвергается, а не приводится молча.

    На трёхканальном кадре `np.full_like` дал бы трёхканальную маску охвата, и
    потребитель, ожидающий двумерную, получил бы её без единого слова.
    """
    _, _, H, mmu = prepared()
    for bad, reason in ((np.zeros((32, 32, 3), np.uint8), REFUSAL_NOT_GRAYSCALE),
                        (np.zeros(32, np.uint8), REFUSAL_NOT_GRAYSCALE),
                        (np.zeros((0, 0), np.uint8), REFUSAL_EMPTY_FRAME),
                        ([[1, 2], [3, 4]], REFUSAL_NOT_NUMERIC),
                        (None, REFUSAL_NOT_NUMERIC)):
        with pytest.raises(ValueError, match=reason):
            rectify(bad, H, mmu, 4.0)


def refusal_messages():
    """Все шесть сообщений отказа, каждое — вместе со своим вызовом.

    Вызовы перечислены здесь, а не разбросаны по тестам, чтобы взаимная
    исключительность проверялась на ПОЛНОМ наборе: пара «положительный масштаб»
    против «невозможный размер» сличалась и раньше, а причины, добавленные после
    рецензии, не сличались ни с чем.
    """
    sc, _, H, mmu = prepared()
    img = sc.render()

    def calls():
        yield REFUSAL_NON_POSITIVE, lambda: rectify(img, H, mmu, 0.0)
        yield REFUSAL_MM_PER_RECT_UNIT, lambda: rectify(img, H, 0.0, 4.0)
        # Выше верхней границы, а не ниже нижней: отказ обязан быть отказом, а не
        # падением на попытке выделить растр в сотни миллионов пикселей.
        yield REFUSAL_IMPOSSIBLE_SIZE, lambda: rectify(img, H, mmu, 1e6)
        yield REFUSAL_NOT_NUMERIC, lambda: rectify(None, H, mmu, 4.0)
        yield REFUSAL_NOT_GRAYSCALE, lambda: rectify(np.zeros((8, 8, 3), np.uint8), H, mmu, 4.0)
        yield REFUSAL_EMPTY_FRAME, lambda: rectify(np.zeros((0, 0), np.uint8), H, mmu, 4.0)

    out = {}
    for pattern, call in calls():
        with pytest.raises(ValueError) as excinfo:
            call()
        out[pattern] = str(excinfo.value)
    return out


def test_refusal_reasons_are_literal_and_mutually_exclusive():
    """Охрана самих охран: каждая подстрока подходит ровно к ОДНОМУ сообщению.

    Вертикальная черта в `match=` читается как альтернатива регулярного выражения и
    делает проверку почти всегда успешной; любой другой метасимвол размывает
    подстроку так же. Сверх литеральности проверяется главное — что `match=` не
    поймает соседний отказ и не объявит охрану работающей, когда сработала чужая.
    """
    for pattern in REFUSAL_PATTERNS:
        assert not set(pattern) & REGEX_META

    messages = refusal_messages()
    assert set(messages) == set(REFUSAL_PATTERNS)
    assert len(set(messages.values())) == len(REFUSAL_PATTERNS)

    for pattern in REFUSAL_PATTERNS:
        matched = [own for own, text in messages.items() if re.search(pattern, text)]
        assert matched == [pattern], f"подстрока {pattern!r} подошла к {matched}"
