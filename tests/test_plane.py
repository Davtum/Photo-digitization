import math

import numpy as np
import pytest

from facade_digitizer.pipeline.plane import estimate_plane, estimate_plane_manual
from tests.test_homography import CORNERS, apply
from tests.test_synth import K, make_scene


def test_plane_is_estimated_on_a_good_scene():
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    assert res.method == "vanishing_points"
    assert not res.needs_operator
    assert res.confidence.value > 0.5


def test_estimated_homography_rectifies_the_facade():
    """Гомография проверяется по существу, а не по факту непустоты."""
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    r = apply(res.H, sc.project(CORNERS))
    a, b = r[1] - r[0], r[3] - r[0]
    ang = np.degrees(np.arccos(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert ang == pytest.approx(90.0, abs=0.5)


def test_low_confidence_requests_the_operator_instead_of_guessing():
    """Спецификация, п. 4.2: ниже порога доверия система не гадает."""
    noise = np.random.default_rng(1).integers(0, 255, (600, 600), dtype=np.uint8)
    res = estimate_plane(noise, K)
    assert res.needs_operator
    assert res.confidence.reasons
    assert res.H is None


def test_vanishing_points_are_carried_out_for_the_pose_step():
    """Позу считает сборка конвейера, поэтому точки схода обязаны выйти наружу."""
    sc = make_scene()
    res = estimate_plane(sc.render(), K)
    assert res.vh is not None and res.vv is not None
    assert np.asarray(res.vh).shape == (3,)


def test_manual_requires_disambiguation():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    with pytest.raises(ValueError, match="четырёх точек недостаточно"):
        estimate_plane_manual(pts)


def test_manual_with_sizes_produces_those_sizes():
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, size_mm=(1460.0, 1900.0))
    assert res.method == "manual_four_point"
    assert not res.needs_operator
    r = apply(res.H, pts)
    assert np.linalg.norm(r[1] - r[0]) == pytest.approx(1460.0, rel=1e-6)


def test_manual_confidence_is_not_borrowed_from_the_automatic_path():
    """Ручной вариант не должен изображать измеренное доверие."""
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, aspect_ratio=2.0)
    assert res.confidence.reasons == []
    assert res.confidence.support_h == 0 and res.confidence.support_v == 0


def test_threshold_is_honoured():
    """Порог доверия действительно решает, а не декоративен."""
    sc = make_scene()
    strict = estimate_plane(sc.render(), K, min_confidence=0.999)
    assert strict.needs_operator
    assert strict.H is None


# Три проверки ниже добавлены сверх набора брифа: мутационная проверка показала, что
# восемь тестов брифа переживают мутации решений, которые бриф не описывал, — «порог
# отключается аргументом», «отказ всё равно отдаёт точки схода», «ручное доверие
# записывает правдоподобные измеренные числа».


def test_a_named_reason_refuses_at_any_threshold():
    """Контракт п. 4.2 не отключается аргументом `min_confidence=0.0`.

    Порог — не единственная охрана: непустой список причин означает, что метрика
    доверия отвергла вход по существу, и такой вход не годится ни при каком пороге.
    Иначе один аргумент вызывающего кода снимал бы весь запрет на догадку.
    """
    noise = np.random.default_rng(1).integers(0, 255, (600, 600), dtype=np.uint8)
    res = estimate_plane(noise, K, min_confidence=0.0)
    assert res.confidence.reasons
    assert res.needs_operator
    assert res.H is None


def test_refusal_does_not_hand_out_untrusted_vanishing_points():
    """Отказ не оставляет обходного пути: точек схода при отказе наружу нет.

    Иначе вызывающий код взял бы `res.vh`, `res.vv` напрямую и построил гомографию
    сам — ровно ту догадку, ради запрета которой отказ и существует.
    """
    noise = np.random.default_rng(1).integers(0, 255, (600, 600), dtype=np.uint8)
    res = estimate_plane(noise, K)
    assert res.needs_operator
    assert res.vh is None and res.vv is None


def test_manual_confidence_does_not_fake_measured_numbers():
    """Величины, которых ручной путь не измерял, записаны как NaN.

    Нулевая невязка и ровно 90° — правдоподобные ИЗМЕРЕННЫЕ значения: потребитель
    не отличил бы их от настоящих и решил бы, что плоскость подтверждена
    измерением. NaN отличим сразу. Пустого списка причин для этого мало — он
    одинаков у ручного пути и у безупречного автоматического.
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, aspect_ratio=2.0)
    assert math.isnan(res.confidence.residual_px)
    assert math.isnan(res.confidence.orthogonality_deg)
