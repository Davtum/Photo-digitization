import math

import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.parallax import (
    apparent_position,
    correct_for_depth,
    parallax_offset,
    reveal_depth,
    reveal_depth_sigma,
    visible_reveal_side,
)

CAM = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)


def test_point_in_plane_is_not_displaced():
    """Наружная кромка лежит в Π и параллаксом не искажается. Спецификация, п. 5.3."""
    ax, ay = apparent_position(CAM, 5000.0, 3000.0, depth=0.0)
    assert ax == pytest.approx(5000.0, abs=1e-9)
    assert ay == pytest.approx(3000.0, abs=1e-9)


def test_offset_magnitude_equals_d_tan_theta():
    d = 150.0
    dx, dy = parallax_offset(CAM, 5000.0, 3000.0, depth=d)
    expected = d * math.tan(math.radians(CAM.theta_deg(5000.0, 3000.0, depth=d)))
    assert math.hypot(dx, dy) == pytest.approx(expected, rel=1e-9)


def test_offset_points_toward_foot_for_recessed():
    """Заглублённая точка смещается К опорной точке. Ошибка знака в редакции 2 спеки."""
    x, y, d = 5000.0, 3000.0, 150.0
    dx, dy = parallax_offset(CAM, x, y, depth=d)
    assert dx < 0  # опорная точка левее (cx=3236 < 5000)
    assert dy < 0  # опорная точка ниже (cy=-2823 < 3000)


def test_offset_points_away_from_foot_for_protruding():
    dx, dy = parallax_offset(CAM, 5000.0, 3000.0, depth=-150.0)
    assert dx > 0
    assert dy > 0


def test_correction_is_inverse_of_displacement():
    x, y, d = 4200.0, 5100.0, 220.0
    ax, ay = apparent_position(CAM, x, y, d)
    bx, by = correct_for_depth(CAM, ax, ay, d)
    assert bx == pytest.approx(x, abs=1e-9)
    assert by == pytest.approx(y, abs=1e-9)


def test_near_side_inner_edge_is_occluded():
    """Внутреннее ребро на ближней стороне проецируется за стену. Спецификация, п. 5.5."""
    cam = CameraOnPlane(cx=1000.0, cy=3000.0, cz=10000.0)
    x_left = 3500.0
    ax, _ = apparent_position(cam, x_left, 3000.0, depth=150.0)
    assert ax < x_left


@pytest.mark.parametrize("depth,theta_deg,expected_mm", [
    (150.0, 5.0, 13.1), (150.0, 15.0, 40.2), (150.0, 30.0, 86.6), (150.0, 45.0, 150.0),
    (300.0, 15.0, 80.4), (300.0, 30.0, 173.2),
])
def test_table_5_2_of_spec(depth, theta_deg, expected_mm):
    """Таблица смещений п. 5.2 проверяется ПРОТИВ РЕАЛИЗАЦИИ, а не против самой себя."""
    cz = 10000.0
    x_edge, y_edge = 5000.0, 3000.0
    cam = CameraOnPlane(
        cx=x_edge - (cz + depth) * math.tan(math.radians(theta_deg)),
        cy=y_edge,
        cz=cz,
    )
    dx, dy = parallax_offset(cam, x_edge, y_edge, depth=depth)
    assert math.hypot(dx, dy) == pytest.approx(expected_mm, abs=0.1)


def test_apparent_position_rejects_point_behind_camera():
    """Охрана согласована с camera.tan_theta: без неё деление на ноль либо неверный знак."""
    with pytest.raises(ValueError):
        apparent_position(CAM, 5000.0, 3000.0, depth=-CAM.cz)
    with pytest.raises(ValueError):
        apparent_position(CAM, 5000.0, 3000.0, depth=-CAM.cz - 5000.0)


def test_correction_matches_independently_computed_value():
    """Ожидаемые значения посчитаны вручную по п. 5.2 и заморожены литералами.

    Формула в теле теста повторяла бы формулу реализации и ловила бы только
    рассогласование копий, а не ошибку в самой формуле.
    """
    d = 220.0
    observed = (4150.0, 4900.0)
    got = correct_for_depth(CAM, observed[0], observed[1], d)
    assert got[0] == pytest.approx(4170.108, abs=1e-3)
    assert got[1] == pytest.approx(5069.906, abs=1e-3)


def test_correct_for_depth_rejects_point_behind_camera():
    """Область определения обратной функции обязана совпадать с прямой."""
    with pytest.raises(ValueError):
        correct_for_depth(CAM, 5000.0, 3000.0, depth=-CAM.cz)
    with pytest.raises(ValueError):
        correct_for_depth(CAM, 5000.0, 3000.0, depth=-CAM.cz - 5000.0)


# Опорная геометрия для проверок чувствительности: камера 10 м над плоскостью,
# заглубление 150 мм, компонента угла визирования 30°. Сценарий строится ПРЯМОЙ
# моделью (w = d · tg θ_⊥, |u| = tg θ_⊥ · (C_z + d)), а не формулой оценки.
X_EDGE, Y_EDGE = 5000.0, 3000.0
CZ_BASE, DEPTH_BASE = 10000.0, 150.0
TAN_BASE = math.tan(math.radians(30.0))
U_BASE = TAN_BASE * (CZ_BASE + DEPTH_BASE)
W_BASE = DEPTH_BASE * TAN_BASE


def _depth_from(width_mm, u_mm, cz_mm):
    """Глубина по РЕАЛИЗАЦИИ `reveal_depth` при независимых w, |u|, C_z.

    Порог θ_min снят намеренно: функция используется для численного
    дифференцирования, где важна гладкость, а не пригодность грани.
    """
    cam = CameraOnPlane(cx=X_EDGE - u_mm, cy=Y_EDGE, cz=cz_mm)
    return reveal_depth(
        cam, X_EDGE, Y_EDGE, width_mm, edge_normal=(1.0, 0.0), theta_min_deg=0.0
    )


def test_reveal_depth_recovers_known_depth():
    """Прямая проверка: породить ширину откоса из известной глубины и вернуть её.

    Порог снят: у этой позы θ_⊥ = 9.86°, грань для реального измерения непригодна,
    но арифметику восстановления она проверяет.
    """
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 150.0
    dx, _ = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    w = abs(dx)  # ширина ВЕРТИКАЛЬНОГО откоса — только X-компонента
    got = reveal_depth(cam, x_edge, y_edge, w, edge_normal=(1.0, 0.0), theta_min_deg=0.0)
    assert got == pytest.approx(d_true, rel=1e-9)


def test_horizontal_reveal_uses_y_component():
    """Горизонтальная грань: θ_⊥ = 29.7°, порог по умолчанию не мешает."""
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 220.0
    _, dy = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    got = reveal_depth(cam, x_edge, y_edge, abs(dy), edge_normal=(0.0, 1.0))
    assert got == pytest.approx(d_true, rel=1e-9)


def test_reveal_depth_is_invariant_to_edge_normal_length():
    """Нормаль задаёт направление, её длина на результат влиять не должна.

    Без нормировки edge_normal проекция u масштабируется вместе с нормалью,
    и глубина искажается во столько же раз.
    """
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    x_edge, y_edge, d_true = 5000.0, 3000.0, 150.0
    dx, _ = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    got = reveal_depth(
        cam, x_edge, y_edge, abs(dx), edge_normal=(3.0, 0.0), theta_min_deg=0.0
    )
    assert got == pytest.approx(d_true, rel=1e-9)


def test_reveal_depth_ignores_edge_normal_orientation():
    """Знак нормали безразличен: проекция берётся по модулю.

    Без `abs` нормаль, направленная к опорной точке, даёт отрицательную проекцию
    и вырожденную геометрию вместо глубины.
    """
    outward = _depth_from(W_BASE, U_BASE, CZ_BASE)
    cam = CameraOnPlane(cx=X_EDGE - U_BASE, cy=Y_EDGE, cz=CZ_BASE)
    inward = reveal_depth(cam, X_EDGE, Y_EDGE, W_BASE, edge_normal=(-1.0, 0.0))
    assert outward == pytest.approx(DEPTH_BASE, rel=1e-9)
    assert inward == pytest.approx(outward, rel=1e-12)


def test_reveal_depth_rejects_zero_normal():
    """Нулевая нормаль не задаёт направления ребра.

    Сообщение проверяется: при подмене охраны на `norm = 1.0` срабатывает другая
    охрана, и вызывающий код получает диагноз «вырожденная геометрия» вместо
    указания на неверную нормаль.
    """
    cam = CameraOnPlane(cx=3236.0, cy=-2823.0, cz=10000.0)
    with pytest.raises(ValueError, match="нормаль к ребру откоса не может быть нулевой"):
        reveal_depth(cam, 5000.0, 3000.0, 26.0, edge_normal=(0.0, 0.0))


def test_reveal_depth_rejects_nonpositive_width():
    """Неположительная ширина — не измерение, а сбой детектора.

    Без охраны нулевая ширина даёт глубину 0, отрицательная — отрицательную
    глубину, и то и другое молча.
    """
    for bad_width in (0.0, -10.0):
        with pytest.raises(ValueError, match="ширина откоса должна быть строго положительной"):
            _depth_from(bad_width, U_BASE, CZ_BASE)


@pytest.mark.parametrize("theta_x_deg,theta_y_deg,ratio", [
    (10.0, 30.0, 3.4236), (5.0, 25.0, 5.4229), (15.0, 15.0, 1.4142), (10.0, 0.0, 1.0000),
])
def test_scalar_formula_underestimates(theta_x_deg, theta_y_deg, ratio):
    """Скалярная запись по полному углу занижает глубину. Спецификация, п. 5.4.

    Тест ВЫЗЫВАЕТ реализацию: считает ширину откоса из известной глубины, затем
    восстанавливает глубину правильной компонентной формулой и ошибочной скалярной,
    и сравнивает. В редакции 1 плана этот тест считал арифметику над собственными
    входами и проходил при полностью удалённом модуле.

    Коэффициенты заморожены для ТОЧНЫХ углов: hypot(tg θ_x, tg θ_y) / tg θ_x.
    Прежние 3.45 и 5.43 относились к 9.86°/29.84°, а не к 10°/30°.
    Порог θ_min снят: ракурсы с θ_x = 5° и 10° ниже порога по построению —
    в них и состоит проверяемый эффект.
    """
    cz, d_true = 10000.0, 150.0
    x_edge, y_edge = 5000.0, 3000.0
    cam = CameraOnPlane(
        cx=x_edge - (cz + d_true) * math.tan(math.radians(theta_x_deg)),
        cy=y_edge - (cz + d_true) * math.tan(math.radians(theta_y_deg)),
        cz=cz,
    )
    dx, dy = parallax_offset(cam, x_edge, y_edge, depth=d_true)
    w = abs(dx)

    correct = reveal_depth(
        cam, x_edge, y_edge, w, edge_normal=(1.0, 0.0), theta_min_deg=0.0
    )
    assert correct == pytest.approx(d_true, rel=1e-6)

    # Скалярная запись: полный угол вместо компоненты.
    tx, ty = cam.tan_theta(x_edge, y_edge, depth=d_true)
    scalar = w / math.hypot(tx, ty)
    assert correct / scalar == pytest.approx(ratio, rel=1e-3)


def test_reveal_depth_rejects_degenerate_geometry():
    """Ширина откоса не может превышать расстояние до опорной точки.

    Сообщение проверяется: без этой охраны срабатывает порог θ_min, и диагноз
    подменяется — убийство мутанта оказалось бы фиктивным.
    """
    cam = CameraOnPlane(cx=4990.0, cy=3000.0, cz=10000.0)
    with pytest.raises(ValueError, match="не меньше расстояния до опорной точки"):
        reveal_depth(cam, 5000.0, 3000.0, reveal_width_mm=50.0, edge_normal=(1.0, 0.0))


def test_reveal_depth_rejects_shallow_angle():
    """Порог θ_⊥ ≥ 15°: при малом угле глубина уходит в бесконечность. П. 5.4.

    При |u| = 1000 и w = 999.9999 замкнутая форма даёт 1.0e11 мм — сто километров
    заглубления без всякого признака неисправности. Порог обязан это отсечь.
    """
    cam = CameraOnPlane(cx=X_EDGE - 1000.0, cy=Y_EDGE, cz=CZ_BASE)
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(cam, X_EDGE, Y_EDGE, 999.9999, edge_normal=(1.0, 0.0))

    # Порог перекрываем — и тогда виден масштаб бедствия.
    unguarded = _depth_from(999.9999, 1000.0, CZ_BASE)
    assert unguarded > 1e10

    # Штатная геометрия (θ_⊥ = 30°) порогом по умолчанию не отсекается.
    cam_ok = CameraOnPlane(cx=X_EDGE - U_BASE, cy=Y_EDGE, cz=CZ_BASE)
    got = reveal_depth(cam_ok, X_EDGE, Y_EDGE, W_BASE, edge_normal=(1.0, 0.0))
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)


def test_reveal_depth_threshold_boundary_is_enforced():
    """Порог отсекает ровно то, что ниже него, и пропускает то, что выше.

    Сравнение строгое: грань РОВНО на пороге принимается. Порог берётся тем самым
    значением угла, которое вычисляет сама реализация по возвращённой ею глубине,
    поэтому равенство точное, а не приблизительное.
    """
    cam = CameraOnPlane(cx=X_EDGE - U_BASE, cy=Y_EDGE, cz=CZ_BASE)
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(
            cam, X_EDGE, Y_EDGE, W_BASE, edge_normal=(1.0, 0.0), theta_min_deg=30.5
        )
    got = reveal_depth(
        cam, X_EDGE, Y_EDGE, W_BASE, edge_normal=(1.0, 0.0), theta_min_deg=29.5
    )
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)

    theta_exact = math.degrees(math.atan(abs(X_EDGE - cam.cx) / (cam.cz + got)))
    on_threshold = reveal_depth(
        cam, X_EDGE, Y_EDGE, W_BASE, edge_normal=(1.0, 0.0), theta_min_deg=theta_exact
    )
    assert on_threshold == pytest.approx(got, rel=1e-12)


def test_threshold_uses_angle_at_inner_edge():
    """Порог считается по ВНУТРЕННЕМУ ребру, а не по плоскости стены. П. 5.4.

    Угол на внутреннем ребре arctg(|u| / (C_z + d)) = 30.00°, а на плоскости стены
    arctg(|u| / C_z) = 30.37°: те же полтора процента по тангенсу и того же
    происхождения, что и занижение в бюджете погрешности. Порог 30.2° лежит между
    ними, поэтому различает две версии: по внутреннему ребру грань отвергается,
    по плоскости стены — принималась бы. Прежний тест с 29.5° и 30.5° обе версии
    пропускал.
    """
    cam = CameraOnPlane(cx=X_EDGE - U_BASE, cy=Y_EDGE, cz=CZ_BASE)

    at_edge = math.degrees(math.atan(U_BASE / (CZ_BASE + DEPTH_BASE)))
    at_wall = math.degrees(math.atan(U_BASE / CZ_BASE))
    assert at_edge == pytest.approx(30.0, abs=1e-9)
    assert at_wall == pytest.approx(30.3708, abs=1e-4)

    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(
            cam, X_EDGE, Y_EDGE, W_BASE, edge_normal=(1.0, 0.0), theta_min_deg=30.2
        )


def test_width_sigma_matches_numeric_sensitivity():
    """Вклад ширины сверяется с РЕАЛИЗАЦИЕЙ `reveal_depth`, а не с формулой. П. 6.2.

    Прежний тест повторял формулу кода и ловил только рассогласование копий.
    Здесь ширина варьируется как независимый аргумент, глубина пересчитывается
    `reveal_depth`, и центральная разность сравнивается с ∂d/∂w из σ.

    Литерал 1.75803 — число независимого рецензента; малоугловое приближение
    ctg 30° = 1.73205 занижает его ровно в 1 + d/C_z = 1.015.
    """
    h = 1e-4
    numeric = (
        _depth_from(W_BASE + h, U_BASE, CZ_BASE) - _depth_from(W_BASE - h, U_BASE, CZ_BASE)
    ) / (2.0 * h)

    analytic = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 1.0, 0.0, 0.0)
    assert analytic == pytest.approx(numeric, rel=1e-6)
    assert analytic == pytest.approx(1.75803, rel=1e-5)
    assert analytic > 1.0 / math.tan(math.radians(30.0))


def test_u_sigma_matches_numeric_sensitivity():
    """Вклад позы через проекцию |u| сверяется с реализацией. П. 6.2."""
    h = 1e-2
    numeric = abs(
        _depth_from(W_BASE, U_BASE + h, CZ_BASE) - _depth_from(W_BASE, U_BASE - h, CZ_BASE)
    ) / (2.0 * h)

    analytic = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 0.0, 1.0, 0.0)
    assert analytic == pytest.approx(numeric, rel=1e-6)


def test_cz_sigma_matches_numeric_sensitivity():
    """Вклад высоты камеры сверяется с реализацией. П. 6.2."""
    h = 1e-2
    numeric = abs(
        _depth_from(W_BASE, U_BASE, CZ_BASE + h) - _depth_from(W_BASE, U_BASE, CZ_BASE - h)
    ) / (2.0 * h)

    analytic = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 0.0, 0.0, 1.0)
    assert analytic == pytest.approx(numeric, rel=1e-6)
    assert analytic == pytest.approx(DEPTH_BASE / CZ_BASE, rel=1e-9)


def test_sigma_combines_by_rss_not_linear_sum():
    """Независимые источники складываются квадратично. Спецификация, п. 6.2.

    На опорной геометрии при σ_w = 5 мм и σ_u = 175 мм вклады равны 8.7902 и
    4.5466 мм: RSS даёт 9.8964 мм, линейная сумма — 13.3368 мм, завышение на 35 %.
    """
    from_width = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 5.0, 0.0, 0.0)
    from_u = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 0.0, 175.0, 0.0)
    both = reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, 5.0, 175.0, 0.0)

    assert from_width == pytest.approx(8.7902, rel=1e-4)
    assert from_u == pytest.approx(4.5466, rel=1e-4)
    assert both == pytest.approx(9.8964, rel=1e-4)
    assert both == pytest.approx(math.hypot(from_width, from_u), rel=1e-12)
    assert both < from_width + from_u
    assert (from_width + from_u) / both == pytest.approx(1.3476, rel=1e-3)


def test_sigma_grows_as_angle_shrinks():
    """σ_d резко растёт при малом угле визирования. Спецификация, п. 6.2."""
    sigma_w, sigma_u = 5.0, 175.0

    def sigma_at(theta_deg):
        tan = math.tan(math.radians(theta_deg))
        return reveal_depth_sigma(
            DEPTH_BASE * tan, tan * (CZ_BASE + DEPTH_BASE), CZ_BASE, sigma_w, sigma_u
        )

    wide = sigma_at(30.0)
    narrow = sigma_at(5.0)
    assert narrow > 3 * wide


def test_sigma_rejects_negative_uncertainty():
    """Отрицательная σ на входе — сбой вызывающего кода, а не данные.

    Без охраны `hypot` съедает знак и возвращает правдоподобное число.
    """
    for bad in ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)):
        with pytest.raises(ValueError, match="отрицательными"):
            reveal_depth_sigma(W_BASE, U_BASE, CZ_BASE, *bad)


def test_sigma_rejects_degenerate_inputs():
    """Производные не определены при вырожденной геометрии и неверных входах."""
    with pytest.raises(ValueError, match="не меньше проекции"):
        reveal_depth_sigma(W_BASE, W_BASE, CZ_BASE, 5.0, 175.0)
    with pytest.raises(ValueError, match="ширина откоса должна быть строго положительной"):
        reveal_depth_sigma(0.0, U_BASE, CZ_BASE, 5.0, 175.0)
    with pytest.raises(ValueError, match="высота камеры"):
        reveal_depth_sigma(W_BASE, U_BASE, 0.0, 5.0, 175.0)


def test_visible_reveal_side_is_far_side():
    """Виден откос на ДАЛЬНЕЙ от опорной точки стороне. Спецификация, п. 5.5."""
    cam = CameraOnPlane(cx=1000.0, cy=1000.0, cz=10000.0)
    vertical, horizontal = visible_reveal_side(cam, 3500.0, 5000.0, 3000.0, 5000.0)
    assert vertical == "right"
    assert horizontal == "top"

    cam2 = CameraOnPlane(cx=9000.0, cy=9000.0, cz=10000.0)
    vertical2, horizontal2 = visible_reveal_side(cam2, 3500.0, 5000.0, 3000.0, 5000.0)
    assert vertical2 == "left"
    assert horizontal2 == "bottom"


def test_visible_reveal_side_splits_at_opening_midpoint():
    """Граница — середина проёма, а не его край. Спецификация, п. 5.5.

    Опорная точка внутри диапазона проёма: cx = 4000 левее середины 4250, но правее
    левого края 3500; cy = 3500 ниже середины 4000, но выше нижнего края 3000.
    Сравнение с краем вместо середины дало бы обе грани наоборот.
    """
    cam = CameraOnPlane(cx=4000.0, cy=3500.0, cz=10000.0)
    vertical, horizontal = visible_reveal_side(cam, 3500.0, 5000.0, 3000.0, 5000.0)
    assert vertical == "right"
    assert horizontal == "top"
