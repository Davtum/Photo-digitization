import cv2
import numpy as np
import pytest

from facade_digitizer.geometry.vanishing import detect_segments, estimate_vanishing_points
from tests.test_synth import SIZE, K, make_rich_scene, make_scene


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


def test_vanishing_points_are_found_on_tilted_facade():
    scene = make_scene()
    segs = detect_segments(scene.render())
    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.value > 0.5
    assert conf.support_h >= 2 and conf.support_v >= 2


def test_confidence_is_low_without_structure():
    noise = np.random.default_rng(0).integers(0, 255, (600, 600), dtype=np.uint8)
    segs = detect_segments(noise)
    _, _, conf = estimate_vanishing_points(segs, (600, 600), K)
    assert conf.value < 0.5
    assert conf.reasons


def test_orthogonality_close_to_ninety_on_valid_scene():
    scene = make_scene()
    segs = detect_segments(scene.render())
    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=6.0)


def _direction(v, K_):
    """Направление в системе камеры, отвечающее точке схода."""
    d = np.linalg.inv(K_) @ np.asarray(v, dtype=float)
    return d / np.linalg.norm(d)


def _angle_to_truth_deg(v, scene, d_world):
    """Угол между оценённым направлением и истинным, взятым из позы сцены.

    Истинная точка схода направления d считается точно: K · (R_wc · d).
    """
    truth = scene.K @ (scene.R_wc @ np.asarray(d_world, dtype=float))
    cos = abs(float(_direction(v, scene.K) @ _direction(truth, scene.K)))
    return float(np.degrees(np.arccos(min(1.0, cos))))


@pytest.mark.parametrize("build", [make_scene, make_rich_scene],
                         ids=["бедная", "обогащённая"])
def test_confidence_holds_on_poor_and_rich_scene(build):
    """Контракт доверия обязан держаться и на бедной сцене, и на обогащённой.

    Различитель: вертикальный пучок бедной сцены — четыре прямые, три из которых
    короткие и сгрудились на одном проёме; обогащённая даёт ему разнесённые
    вертикали. Если оценщик проходит только на богатой — дело в данных.
    """
    scene = build()
    segs = detect_segments(scene.render())
    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.value > 0.5
    assert conf.support_h >= 2 and conf.support_v >= 2
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=6.0)
    assert conf.reasons == []


@pytest.mark.parametrize("build", [make_scene, make_rich_scene],
                         ids=["бедная", "обогащённая"])
def test_vanishing_points_agree_with_true_pose(build):
    """Оценка сверяется с точной истиной K · (R_wc · d), а не сама с собой.

    Без этой сверки метрика доверия может уверенно подтверждать неверную
    плоскость: согласованный пучок обломков одной прямой даёт и поддержку,
    и малую невязку, и ортогональность.
    """
    scene = build()
    segs = detect_segments(scene.render())
    vh, vv, _ = estimate_vanishing_points(segs, scene.image_size, K)
    assert _angle_to_truth_deg(vh.point, scene, [1.0, 0.0, 0.0]) < 2.0
    assert _angle_to_truth_deg(vv.point, scene, [0.0, 1.0, 0.0]) < 2.0


def _single_bundle_image(vp=(-2200.0, 1978.0), size=(1400, 900)):
    """Кадр, где есть только горизонтальный пучок: вертикальной точке схода не на чем стоять."""
    w, h = size
    img = np.full((h, w), 200, dtype=np.uint8)
    for y in range(120, h - 40, 80):
        p0 = np.array([float(w - 20), float(y)])
        d = p0 - np.array(vp, dtype=float)
        d = d / np.linalg.norm(d)
        p1 = p0 - d * 1300.0
        cv2.line(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 90, 5)
    return img


def test_confidence_is_zero_when_one_bundle_has_no_support():
    """Один пучок из двух — не плоскость, каким бы согласованным он ни был.

    Пучок из 45 сходящихся горизонталей даёт покрытие 1.0 и невязку 6 пикселей:
    по этим числам результат выглядит отличным. Вертикальной точки схода при
    этом нет вовсе, и доверие обязано быть нулевым, а не высоким, — иначе
    оператор не будет вызван там, где спецификация (п. 4.2) этого требует.
    """
    img = _single_bundle_image()
    segs = detect_segments(img)
    _, _, conf = estimate_vanishing_points(segs, (1400, 900), K)
    assert conf.support_v < 2
    assert conf.reasons
    assert conf.value == 0.0


def test_thin_support_is_not_trusted_even_on_exact_geometry():
    """Восемь точных прямых из истинной позы: невязка 1e-13, покрытие 1.0 — и всё же отказ.

    Точки схода тут восстановлены идеально, ортогональность ровно 90°, но четыре
    прямые на пучок — слишком тонкая опора, чтобы на реальном снимке отличить
    верную плоскость от случайно согласованной. Проверка суммарного числа
    поддерживающих отрезков — единственное, что удерживает этот случай от
    доверия 1.00.
    """
    scene = make_scene()
    rows = []
    for y in (2000.0, 6000.0, 10000.0, 14000.0):
        p = scene.project(np.array([[1000.0, y], [19000.0, y]]))
        rows.append([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])
    for x in (2000.0, 7000.0, 12000.0, 18000.0):
        p = scene.project(np.array([[x, 1000.0], [x, 14000.0]]))
        rows.append([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])
    segs = np.array(rows)

    _, _, conf = estimate_vanishing_points(segs, SIZE, K)
    assert conf.support_h >= 2 and conf.support_v >= 2
    assert conf.coverage == pytest.approx(1.0)
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=1.0)
    assert any("мало поддерживающих" in r for r in conf.reasons)
    assert conf.value == 0.0


def test_confidence_is_not_bare_coverage():
    """Доверие — не доля поддержавших отрезков, а величина со своей шкалой.

    У признанного годным результата есть базовый уровень 0.3, к которому
    покрытие добавляет остальное. Порог передачи оператору откалиброван по этой
    шкале, поэтому подмена доверия голым покрытием сдвигает порог.
    """
    scene = make_scene()
    segs = detect_segments(scene.render())
    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert conf.reasons == []
    assert conf.coverage < 1.0
    assert conf.value > conf.coverage
    assert conf.value == pytest.approx(0.3 + 0.7 * conf.coverage)


def test_residual_is_measured_not_a_constant():
    """Невязка измеряется по инлайерам, а не подставляется как threshold * размер кадра.

    В редакции 1 здесь стояла константа 105.6 при любой сцене. Признак подмены —
    равенство невязки этой константе и её независимость от качества входа.
    """
    scene = make_scene()
    conf = estimate_vanishing_points(detect_segments(scene.render()), scene.image_size, K)[2]
    assert 0.0 < conf.residual_px < 0.02 * max(scene.image_size) / 2.0

    exact = []
    for y in (1000.0, 4000.0, 7000.0, 10000.0, 13000.0):
        p = scene.project(np.array([[500.0, y], [19500.0, y]]))
        exact.append([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])
    for x in (1000.0, 5000.0, 9000.0, 13000.0, 17000.0, 19000.0):
        p = scene.project(np.array([[x, 500.0], [x, 14500.0]]))
        exact.append([p[0, 0], p[0, 1], p[1, 0], p[1, 1]])
    conf_exact = estimate_vanishing_points(np.array(exact), SIZE, K)[2]
    assert conf_exact.residual_px < conf.residual_px
