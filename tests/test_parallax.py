import math

import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.parallax import (
    REFERENCE_DEPTH_MM,
    REFERENCE_GSD_MM_PER_PX,
    REFERENCE_SIGMA_CZ_REL,
    REFERENCE_SIGMA_PX,
    REFERENCE_SIGMA_THETA_DEG,
    REFERENCE_TOLERANCE_MM,
    apparent_position,
    correct_for_depth,
    parallax_offset,
    reveal_depth,
    reveal_depth_sigma,
    reveal_depth_theta_min_deg,
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
    # Нормаль к ребру здесь (1, 0), поэтому в ширину откоса входит только x-компонента.
    dx, _dy = parallax_offset(cam, x_edge, y_edge, depth=d_true)
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
    """При малом угле глубина уходит в бесконечность, и порог обязан это отсечь. П. 5.4.

    При |u| = 1000 и w = 999.9999 замкнутая форма даёт 1.0e11 мм — сто километров
    заглубления без всякого признака неисправности.

    Порог здесь берётся по умолчанию, то есть ВЫЧИСЛЯЕТСЯ (п. 6.2), а не равен
    прежней константе 15°. Угол этой геометрии — 6e-7°, он ниже любого порога
    из таблицы п. 6.2, поэтому проверка не зависит от того, какое качество
    кромки принято опорным.
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


# --- Порог применимости θ_⊥ как вычисляемая величина. Спецификация, п. 6.2. ---
#
# Порог отозван как константа: он зависит от σ_px и меняется вчетверо на заявленном
# диапазоне режимов п. 6.1. Проверки ниже охраняют три вещи: совпадение со
# спецификацией, множитель √2 в σ_w и согласованность порога с той самой функцией σ,
# которую система выдаёт потребителю.

SIGMA_THETA_REF_DEG = 0.344   # P95 направления на разреженной сцене, п. 6.2
TOLERANCE_REF_MM = 20.0       # допуск на глубину, п. 2.2
SIGMA_CZ_REF_MM = 0.004 * CZ_BASE   # 0.4 % дистанции — то же, что в расчёте п. 6.2


def _sigma_d_at_theta(theta_deg, sigma_w_mm, cz_mm=CZ_BASE, depth_mm=DEPTH_BASE,
                      sigma_theta_deg=SIGMA_THETA_REF_DEG, sigma_cz_mm=SIGMA_CZ_REF_MM):
    """σ_d на заданном θ_⊥ по формулам СПЕЦИФИКАЦИИ, независимо от реализации порога.

    Геометрия п. 6.2: w = d · tg θ_⊥, |u| = (C_z + d) · tg θ_⊥.
    Перевод угловой погрешности позы: σ_u = C_z · sec²φ · σ_θ, φ = arctg(|u| / C_z).
    """
    tan = math.tan(math.radians(theta_deg))
    w = depth_mm * tan
    u = (cz_mm + depth_mm) * tan
    phi = math.atan(u / cz_mm)
    sigma_u = cz_mm / math.cos(phi) ** 2 * math.radians(sigma_theta_deg)
    return reveal_depth_sigma(w, u, cz_mm, sigma_w_mm, sigma_u, sigma_cz_mm)


def _theta_min(**over):
    args = {
        "sigma_px": 1.0,
        "gsd_mm_per_px": 5.0,
        "cz_mm": CZ_BASE,
        "depth_mm": DEPTH_BASE,
        "tolerance_mm": TOLERANCE_REF_MM,
        "sigma_theta_deg": SIGMA_THETA_REF_DEG,
        "sigma_cz_mm": SIGMA_CZ_REF_MM,
    }
    args.update(over)
    return reveal_depth_theta_min_deg(**args)


@pytest.mark.parametrize("sigma_px,expected_deg", [
    (0.5, 10.5), (1.0, 19.9), (2.0, 35.8), (3.0, 47.3),
])
def test_theta_min_reproduces_spec_table(sigma_px, expected_deg):
    """Таблица порогов п. 6.2: четыре режима качества кромки — четыре порога.

    Допуск 0.1° — последний знак таблицы. Это охрана от расхождения формулы
    со спецификацией: константа на этом месте скрывала бы зависимость,
    определяющую применимость метода.
    """
    assert _theta_min(sigma_px=sigma_px) == pytest.approx(expected_deg, abs=0.1)


def test_theta_min_carries_sqrt2_of_two_edge_rule():
    """σ_w = √2·σ_px·GSD, а не σ_px·GSD. Именно отсутствие √2 породило 15°.

    Ширина откоса — разность ДВУХ кромок (наружной кромки проёма и внутренней
    кромки грани), поэтому подчиняется правилу п. 6.1 для двухкромочных величин.

    Числа заморожены из п. 6.2: при σ_w = σ_px·GSD = 5 мм порог 14.49° (откуда и
    взялись записанные ранее 15°), по правилу п. 6.1 при σ_w = 7.07 мм — 19.93°.
    Первый вызов подаёт σ_px = 1/√2, чтобы получить σ_w = 5 мм ИМЕННО через
    множитель реализации: если √2 убрать, первый вызов даст 10.51° вместо 14.49°,
    а второй — 14.49° вместо 19.93°. Тест падает с обоих концов.
    """
    old_assumption = _theta_min(sigma_px=1.0 / math.sqrt(2.0))
    by_rule = _theta_min(sigma_px=1.0)

    assert old_assumption == pytest.approx(14.49, abs=0.005)
    assert by_rule == pytest.approx(19.93, abs=0.005)
    assert by_rule > old_assumption + 5.0


@pytest.mark.parametrize("sigma_px", [0.5, 1.0, 2.0, 3.0])
def test_sigma_at_threshold_equals_tolerance_exactly(sigma_px):
    """На самом пороге σ_d равна допуску РОВНО. П. 6.2.

    Если порог считается не по той функции, по которой выдаётся σ, здесь будет
    расхождение. Это худший из возможных дефектов на этом месте: потребитель
    получил бы принятую грань с σ выше заявленного допуска.
    """
    theta = _theta_min(sigma_px=sigma_px)
    sigma_w = math.sqrt(2.0) * sigma_px * 5.0
    assert _sigma_d_at_theta(theta, sigma_w) == pytest.approx(TOLERANCE_REF_MM, rel=1e-9)

    # Порог — НАИМЕНЬШИЙ подходящий угол: чуть ниже него допуск уже нарушен,
    # чуть выше — соблюдён. Без этой пары равенство выше выполнялось бы и для
    # верхней ветви σ_d, где допуск нарушается с ростом угла.
    assert _sigma_d_at_theta(theta - 0.05, sigma_w) > TOLERANCE_REF_MM
    assert _sigma_d_at_theta(theta + 0.05, sigma_w) < TOLERANCE_REF_MM


def test_theta_min_grows_with_edge_uncertainty():
    """Порог монотонно растёт с σ_px: вчетверо на диапазоне режимов п. 6.1."""
    values = [_theta_min(sigma_px=s) for s in (0.5, 1.0, 2.0, 3.0)]
    assert values == sorted(values)
    assert values[3] / values[0] == pytest.approx(4.5, abs=0.2)


def test_theta_min_depends_on_product_of_sigma_px_and_gsd():
    """σ_px и GSD входят только произведением: σ_w = √2·σ_px·GSD.

    Вдвое худшая кромка при вдвое лучшем GSD даёт тот же порог. Тест падает,
    если один из множителей потерян или возведён в степень.
    """
    a = _theta_min(sigma_px=1.0, gsd_mm_per_px=2.5)
    b = _theta_min(sigma_px=0.5, gsd_mm_per_px=5.0)
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(10.5, abs=0.1)


def test_theta_min_refuses_unreachable_tolerance():
    """Недостижимый допуск — отказ с названной причиной, не None и не бесконечность.

    σ_d имеет МИНИМУМ по θ_⊥: при больших углах растёт вклад позы, σ_u ∝ sec²φ.
    Ниже этого минимума решения нет ни при каком угле. На опорной геометрии
    минимум ≈ 3.93 мм, поэтому допуск 3 мм недостижим, а 4.5 мм достижим —
    значит отказ вызван допуском, а не порчей входов.
    """
    with pytest.raises(ValueError, match="недостижим"):
        _theta_min(tolerance_mm=3.0)

    got = _theta_min(tolerance_mm=4.5)
    assert 0.0 < got < 90.0
    assert _sigma_d_at_theta(got, math.sqrt(2.0) * 5.0) == pytest.approx(4.5, rel=1e-9)


def test_theta_min_names_the_unreachable_minimum():
    """Причина отказа названа числом: минимум σ_d и угол, на котором он достигнут.

    Сообщение «недостижим» без величины минимума не позволяет вызывающему коду
    понять, насколько допуск занижен.
    """
    with pytest.raises(ValueError) as excinfo:
        _theta_min(tolerance_mm=3.0)
    text = str(excinfo.value)
    # Дизъюнкция здесь была бы мнимой проверкой: «3.9» — подстрока «3.93»,
    # и такое условие выполнялось бы при вдвое отличающемся минимуме.
    assert "3.93" in text
    assert "70.3" in text


def test_theta_min_is_zero_when_nothing_is_uncertain():
    """Без погрешностей порога нет: допуск соблюдён при любом угле."""
    assert _theta_min(sigma_px=0.0, sigma_theta_deg=0.0, sigma_cz_mm=0.0) == 0.0


@pytest.mark.parametrize("bad,match", [
    ({"gsd_mm_per_px": 0.0}, "GSD"),
    ({"gsd_mm_per_px": -5.0}, "GSD"),
    ({"cz_mm": 0.0}, "высота камеры"),
    ({"cz_mm": -1.0}, "высота камеры"),
    ({"depth_mm": 0.0}, "глубина"),
    ({"depth_mm": -150.0}, "глубина"),
    ({"tolerance_mm": 0.0}, "допуск"),
    ({"tolerance_mm": -1.0}, "допуск"),
    ({"sigma_px": -1.0}, "отрицательными"),
    ({"sigma_theta_deg": -0.1}, "отрицательными"),
    ({"sigma_cz_mm": -1.0}, "отрицательными"),
])
def test_theta_min_rejects_invalid_inputs(bad, match):
    """Неверные входы — отказ с диагнозом, а не правдоподобное число."""
    with pytest.raises(ValueError, match=match):
        _theta_min(**bad)


def _cam_at_theta(theta_deg, cz_mm=CZ_BASE, depth_mm=DEPTH_BASE):
    """Камера, дающая ровно θ_⊥ на кромке (X_EDGE, Y_EDGE) при заглублении depth."""
    tan = math.tan(math.radians(theta_deg))
    cam = CameraOnPlane(cx=X_EDGE - (cz_mm + depth_mm) * tan, cy=Y_EDGE, cz=cz_mm)
    return cam, depth_mm * tan


def test_reveal_depth_default_threshold_is_computed_not_fifteen():
    """Порог по умолчанию — вычисленные ≈19.93°, а не константа 15°. П. 6.2.

    Грань с θ_⊥ = 17° прежним порогом принималась, новым — отвергается. Тест
    падает при любой константе в диапазоне [15°, 17°], в том числе при прежней.
    """
    cam, w = _cam_at_theta(17.0)
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0))

    cam_ok, w_ok = _cam_at_theta(21.0)
    got = reveal_depth(cam_ok, X_EDGE, Y_EDGE, w_ok, edge_normal=(1.0, 0.0))
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)


def test_reveal_depth_default_threshold_equals_reference_computation():
    """Значение по умолчанию прослеживается до названного допущения, а не до числа.

    Порог при σ_px = 1, GSD 5 мм/px и опорной геометрии п. 6.2 — ровно то, что
    возвращает `reveal_depth_theta_min_deg` на опубликованных допущениях модуля.
    Проверяется по обе стороны от него, шагом 0.01°.
    """
    theta_ref = reveal_depth_theta_min_deg(
        sigma_px=REFERENCE_SIGMA_PX,
        gsd_mm_per_px=REFERENCE_GSD_MM_PER_PX,
        cz_mm=CZ_BASE,
        depth_mm=REFERENCE_DEPTH_MM,
        tolerance_mm=REFERENCE_TOLERANCE_MM,
        sigma_theta_deg=REFERENCE_SIGMA_THETA_DEG,
        sigma_cz_mm=REFERENCE_SIGMA_CZ_REL * CZ_BASE,
    )
    assert theta_ref == pytest.approx(19.93, abs=0.005)

    cam_below, w_below = _cam_at_theta(theta_ref - 0.01)
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(cam_below, X_EDGE, Y_EDGE, w_below, edge_normal=(1.0, 0.0))

    cam_above, w_above = _cam_at_theta(theta_ref + 0.01)
    got = reveal_depth(cam_above, X_EDGE, Y_EDGE, w_above, edge_normal=(1.0, 0.0))
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)


def test_reveal_depth_default_threshold_follows_camera_distance():
    """C_z в порог по умолчанию берётся ФАКТИЧЕСКИЙ, а не опорные 10 м.

    C_z — известный вход позы, и спецификация называет его аргументом порога.
    На 20 м порог равен 19.80° против 19.93° на 10 м: сдвиг мал, но он есть,
    и шаг проверки 0.02° его различает. Реализация, подставляющая опорные 10 м
    вместо фактических 20 м, отвергнет здесь заведомо пригодную грань.
    """
    cz_far = 20000.0
    theta_far = reveal_depth_theta_min_deg(
        sigma_px=REFERENCE_SIGMA_PX, gsd_mm_per_px=REFERENCE_GSD_MM_PER_PX,
        cz_mm=cz_far, depth_mm=REFERENCE_DEPTH_MM,
        tolerance_mm=REFERENCE_TOLERANCE_MM,
        sigma_theta_deg=REFERENCE_SIGMA_THETA_DEG,
        sigma_cz_mm=REFERENCE_SIGMA_CZ_REL * cz_far,
    )
    assert theta_far == pytest.approx(19.80, abs=0.02)
    assert theta_far < _theta_min()

    cam_below, w_below = _cam_at_theta(theta_far - 0.02, cz_mm=cz_far)
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(cam_below, X_EDGE, Y_EDGE, w_below, edge_normal=(1.0, 0.0))

    cam_above, w_above = _cam_at_theta(theta_far + 0.02, cz_mm=cz_far)
    got = reveal_depth(cam_above, X_EDGE, Y_EDGE, w_above, edge_normal=(1.0, 0.0))
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)


def test_reveal_depth_accepts_callers_edge_quality():
    """Вызывающий код передаёт свои σ_px и GSD, и порог смягчается или ужесточается.

    Та же грань 17°: при субпиксельном уточнении (σ_px = 0.5, порог 10.5°) она
    пригодна, при автоматическом режиме (σ_px = 2, порог 35.8°) — нет.
    """
    cam, w = _cam_at_theta(17.0)
    got = reveal_depth(cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0), sigma_px=0.5)
    assert got == pytest.approx(DEPTH_BASE, rel=1e-9)

    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0), sigma_px=2.0)

    # GSD входит наравне с σ_px: вчетверо крупнее пиксель — вчетверо хуже кромка.
    with pytest.raises(ValueError, match="ниже порога"):
        reveal_depth(
            cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0),
            sigma_px=0.5, gsd_mm_per_px=20.0,
        )


def test_reveal_depth_rejects_conflicting_threshold_arguments():
    """Явный порог и качество кромки вместе — противоречие, а не молчаливый приоритет.

    Без охраны σ_px, переданная вместе с theta_min_deg, была бы тихо
    проигнорирована, и вызывающий код считал бы, что задал режим.
    """
    cam, w = _cam_at_theta(30.0)
    with pytest.raises(ValueError, match="одновременно"):
        reveal_depth(
            cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0),
            theta_min_deg=0.0, sigma_px=1.0,
        )
    with pytest.raises(ValueError, match="одновременно"):
        reveal_depth(
            cam, X_EDGE, Y_EDGE, w, edge_normal=(1.0, 0.0),
            theta_min_deg=0.0, gsd_mm_per_px=5.0,
        )
