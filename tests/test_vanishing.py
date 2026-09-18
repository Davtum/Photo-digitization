import cv2
import numpy as np
import pytest

from facade_digitizer.geometry.vanishing import (
    detect_segments,
    endpoint_deviation_px,
    estimate_vanishing_points,
)
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
    # Причина названа своими словами: отказ «пучок плохо обусловлен» формально тоже
    # сработал бы, но оператору он сообщил бы не то, чего не хватает.
    assert "недостаточная поддержка одной из точек схода" in conf.reasons
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


def _fan_segments(vanishing_px, midpoints, length_px, deviations_px):
    """Отрезки, смотрящие в заданную точку схода и отклонённые от неё на заданные пиксели.

    Построение независимо от реализации: отклонение задаётся поворотом отрезка
    вокруг его середины на угол asin(d / (L/2)), так что отклонение его концов от
    направления на точку схода равно ровно d пикселей — по определению синуса, а не
    по формуле из модуля.
    """
    vp = np.asarray(vanishing_px, dtype=float)
    out = []
    for m, d in zip(np.asarray(midpoints, dtype=float), deviations_px):
        u = m - vp
        u = u / np.linalg.norm(u)
        a = np.arcsin(np.clip(d / (length_px / 2.0), -1.0, 1.0))
        c, s = np.cos(a), np.sin(a)
        h = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]]) * (length_px / 2.0)
        out.append([m[0] - h[0], m[1] - h[1], m[0] + h[0], m[1] + h[1]])
    return np.array(out)


_MID_H = [(1200.0 + 300.0 * i, 400.0 + 320.0 * i) for i in range(8)]
_MID_V = [(500.0 + 420.0 * i, 1200.0 + 120.0 * i) for i in range(8)]
_FRAME = (4000, 3000)


def _bundles_towards(vp_h, vp_v, dev_h=(0.0,) * 8, dev_v=(0.0,) * 8, length=900.0):
    """Пара пучков, сходящихся в две заданные точки схода."""
    return np.vstack([_fan_segments(vp_h, _MID_H, length, dev_h),
                      _fan_segments(vp_v, _MID_V, length, dev_v)])


def _near_parallel_bundles(seed=0, jitter_deg=1.0, n=12, size=(2000, 1500), length=420.0):
    """Два пучка почти параллельных, но заведомо не сходящихся прямых.

    Реалистичный шум детектора на почти фронтальном фасаде: направления гуляют в
    пределах ±1°, поэтому никакой общей точки схода у пучка нет, а есть точка,
    целиком порождённая шумом и убегающая на 1e5 пикселей и дальше.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    segs = []
    for nominal, anchors in ((0.0, [(w * 0.5, 120.0 + i * (h - 240.0) / (n - 1))
                                    for i in range(n)]),
                             (90.0, [(120.0 + i * (w - 240.0) / (n - 1), h * 0.5)
                                     for i in range(n)])):
        for (cx, cy), j in zip(anchors, rng.uniform(-jitter_deg, jitter_deg, n)):
            a = np.radians(nominal + j)
            d = np.array([np.cos(a), np.sin(a)]) * (length / 2.0)
            segs.append([cx - d[0], cy - d[1], cx + d[0], cy + d[1]])
    return np.array(segs), size


def _direction_pair(angle_deg):
    """Пара единичных направлений в системе камеры с заданным углом между ними.

    Обе имеют ненулевую компоненту вдоль оптической оси, поэтому обе дают конечные
    точки схода. Угол задаётся построением, а не измеряется модулем.
    """
    u = np.array([1.0, 0.10, 0.25])
    u = u / np.linalg.norm(u)
    w = np.array([0.05, 1.0, 0.20])
    w = w - (w @ u) * u
    w = w / np.linalg.norm(w)
    a = np.radians(angle_deg)
    return u, np.cos(a) * u + np.sin(a) * w


def _vanishing_of(direction):
    """Истинная точка схода направления: K·d в пикселях."""
    v = K @ np.asarray(direction, dtype=float)
    return v[:2] / v[2]



@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_near_parallel_bundles_get_no_confidence(seed):
    """Главный случай, ради которого метрика и введена: плоскость не определена.

    Два пучка почти параллельных прямых с дрожанием в 1° дают поддержку, покрытие,
    невязку и ортогональность, каждая из которых выглядит благополучно, — а точка
    схода при этом целиком порождена шумом и стоит в 1e5 пикселей от кадра.
    Прежняя редакция выдавала здесь доверие до 1.00 при пустом списке причин: отказ
    становился молчаливым ровно там, где спецификация (п. 4.2) требует позвать
    оператора.
    """
    segs, size = _near_parallel_bundles(seed)
    vh, _, conf = estimate_vanishing_points(segs, size, K)

    assert np.linalg.norm(vh.point[:2]) / abs(vh.point[2]) > 1e4   # точка схода убежала
    assert conf.reasons, "плоскость не определена, а причин отказа нет"
    assert any("не определена" in r for r in conf.reasons)
    assert conf.value == 0.0


def test_inlier_tolerance_does_not_grow_with_distance_to_vanishing_point():
    """Допуск инлайера — в пикселях кадра и не зависит от удалённости точки схода.

    Прежний критерий нормировал дистанцию на ‖v‖, из-за чего фактический допуск
    равнялся threshold·‖v‖: у точки схода в 1.6e5 пикселей он составлял 3253 пикселя,
    и инлайером становился любой отрезок. Проверяется прямо: один и тот же отрезок,
    отклонённый на 10 пикселей, не должен становиться инлайером оттого, что точка
    схода отодвинулась на два порядка.
    """
    mids = [(2000.0, 500.0 + 300.0 * i) for i in range(6)]
    for vp in ((-9000.0, 1500.0), (-900000.0, 1500.0)):
        segs = _fan_segments(vp, mids, 900.0, [0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
        dev = endpoint_deviation_px(segs, np.array([vp[0], vp[1], 1.0]))
        assert dev[:4] == pytest.approx(0.0, abs=1e-6)
        assert dev[4:] == pytest.approx(10.0, rel=1e-6)


def test_residual_is_the_mean_endpoint_deviation_in_pixels():
    """Невязка — среднее отклонение концов инлайеров в настоящих пикселях кадра.

    Ожидаемое значение считается из построения входа, а не по формуле модуля:
    отклонения заданы явно и симметрично, так что подгонка остаётся в истинной точке
    схода. Проверка различает среднее и максимум (0.60 против 1.65) и ловит любой
    посторонний множитель — прежняя редакция домножала безразмерную величину на
    max(image_size), расходясь с истинной дистанцией в 34 раза.
    """
    dev_h = (0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.8, -1.8)
    dev_v = (0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 1.5, -1.5)
    segs = _bundles_towards((-9000.0, 1500.0), (2000.0, -14000.0), dev_h, dev_v)

    _, _, conf = estimate_vanishing_points(segs, _FRAME, K)
    expected = (np.mean(np.abs(dev_h)) + np.mean(np.abs(dev_v))) / 2.0
    assert expected == pytest.approx(0.6)
    assert conf.support_h == len(dev_h) and conf.support_v == len(dev_v)
    assert conf.residual_px == pytest.approx(expected, rel=0.03)
    assert conf.residual_px < 0.5 * max(np.max(np.abs(dev_h)), np.max(np.abs(dev_v)))


def test_non_orthogonal_directions_are_refused_by_name():
    """Направления под 60° к друг другу — не пара осей фасада, и отказ обязан это назвать.

    Угол задан построением: точки схода получены как K·d для двух направлений с
    известным углом между ними, так что 60.0° — не то, что посчитал модуль, а то,
    что было заложено во вход. Проверяется именно причина, а не нулевое доверие:
    доверие обнулилось бы и само собой, и снятие проверки осталось бы незамеченным.
    """
    d1, d2 = _direction_pair(60.0)
    segs = _bundles_towards(_vanishing_of(d1), _vanishing_of(d2))

    _, _, conf = estimate_vanishing_points(segs, _FRAME, K)
    assert conf.orthogonality_deg == pytest.approx(60.0, abs=0.5)
    assert conf.reasons == [f"направления не ортогональны: {conf.orthogonality_deg:.1f}°"]
    assert conf.value == 0.0


def test_orthogonal_directions_of_the_same_shape_are_accepted():
    """Контроль к предыдущему тесту: то же построение под прямым углом проходит.

    Без него отказ по неортогональности неотличим от отказа по самому построению.
    """
    d1, d2 = _direction_pair(90.0)
    _, _, conf = estimate_vanishing_points(
        _bundles_towards(_vanishing_of(d1), _vanishing_of(d2)), _FRAME, K)
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=0.5)
    assert conf.reasons == []
    assert conf.value > 0.5


def test_too_few_segments_is_refused_before_fitting_anything():
    """Четырёх отрезков мало, и отказ наступает до подгонки, а не после неё.

    Проверяется, что причина ровно одна и что ничего не подгонялось: поддержка
    нулевая, невязка не определена. Сдвиг границы раннего отказа пропустил бы этот
    вход в основной путь, где он получил бы другую причину и ненулевую поддержку.
    """
    segs = _bundles_towards((-9000.0, 1500.0), (2000.0, -14000.0))[:4]
    _, _, conf = estimate_vanishing_points(segs, _FRAME, K)
    assert conf.reasons == ["слишком мало отрезков"]
    assert conf.support_h == 0 and conf.support_v == 0
    assert not np.isfinite(conf.residual_px)
    assert conf.value == 0.0


def test_confidence_falls_when_residual_grows_and_when_angle_leaves_ninety():
    """Доверие зависит от невязки и от отклонения от прямого угла, а не от покрытия одного.

    Три входа с одинаковым покрытием 1.0 и одинаковой поддержкой: точный, с
    увеличенной невязкой и с уведённым от 90° углом. Если доверие — функция одного
    лишь покрытия, все три дадут одно и то же число.
    """
    vp_h, vp_v = (-9000.0, 1500.0), (2000.0, -14000.0)
    exact = estimate_vanishing_points(_bundles_towards(vp_h, vp_v), _FRAME, K)[2]

    noisy_dev = (0.0, 0.0, 1.2, -1.2, 1.5, -1.5, 1.8, -1.8)
    noisy = estimate_vanishing_points(
        _bundles_towards(vp_h, vp_v, noisy_dev, noisy_dev), _FRAME, K)[2]

    skew = estimate_vanishing_points(
        _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(80.0)]), _FRAME, K)[2]

    assert exact.reasons == [] and noisy.reasons == [] and skew.reasons == []
    assert exact.coverage == noisy.coverage == skew.coverage == pytest.approx(1.0)
    assert noisy.residual_px > exact.residual_px
    assert noisy.value < exact.value, "рост невязки обязан снижать доверие"
    assert skew.orthogonality_deg == pytest.approx(80.0, abs=0.5)
    assert skew.value < exact.value, "уход угла от прямого обязан снижать доверие"


def test_confidence_is_not_bare_coverage():
    """Доверие не равно доле поддержавших отрезков и не совпадает с ней численно.

    На корректной сцене покрытие достигает 1.0, а доверие — нет: в него входят ещё
    невязка и отклонение от прямого угла, и ни то, ни другое не бывает идеальным на
    настоящем снимке.
    """
    scene = make_scene()
    _, _, conf = estimate_vanishing_points(detect_segments(scene.render()),
                                           scene.image_size, K)
    assert conf.reasons == []
    assert conf.coverage == pytest.approx(1.0)
    assert 0.5 < conf.value < conf.coverage


def _rotate_about_principal_point(segs, degrees):
    """Поворот всех отрезков вокруг главной точки кадра.

    Такому повороту отвечает поворот камеры вокруг оптической оси: взаимный угол
    между восстановленными направлениями он сохраняет, а углы отрезков в кадре —
    сдвигает. Удобный способ проверить, что классификация пучков не привязана к
    случайной ориентации снимка.
    """
    a = np.radians(degrees)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pp = np.array([K[0, 2], K[1, 2]])
    out = segs.copy()
    for k in (0, 2):
        out[:, k:k + 2] = (segs[:, k:k + 2] - pp) @ rot.T + pp
    return out


def test_bundle_split_survives_camera_roll():
    """Поворот камеры вокруг оптической оси не должен разрушать разделение пучков.

    После поворота на 50° одно семейство лежит в кадре около 57°, другое около 146°.
    Оба остаются парой осей фасада, и результат обязан остаться годным. Смещение
    границы разделения свалило бы оба семейства в один пучок, и второй точки схода
    не осталось бы вовсе.
    """
    segs = _rotate_about_principal_point(
        _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)]), 50.0)
    angles = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1],
                                   segs[:, 2] - segs[:, 0])) % 180.0
    assert angles.min() > 50.0 and angles.max() < 150.0   # ни одно не лежит у 0/180

    _, _, conf = estimate_vanishing_points(segs, (6000, 5000), K)
    assert conf.reasons == []
    assert conf.support_h == 8 and conf.support_v == 8
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=0.5)
    assert conf.value > 0.9


def _wide_fan_bundles(deviations, spacing=1400.0, length=350.0):
    """Пара сильно расходящихся пучков: обусловленность высока даже при крупной невязке.

    Веер нужен широкий намеренно. Обусловленность есть отношение веера к угловому
    шуму, поэтому на тесном пучке крупная невязка сама роняет обусловленность, и
    проверка величины невязки оказалась бы неотличима от проверки обусловленности.
    Здесь веер около 40°, и великой остаётся только невязка.
    """
    d1, d2 = _direction_pair(90.0)
    n = len(deviations)
    mid_h = [(1200.0 + 0.3 * spacing * i, 400.0 + spacing * i) for i in range(n)]
    mid_v = [(400.0 + spacing * i, 1200.0 + 0.3 * spacing * i) for i in range(n)]
    return np.vstack([_fan_segments(_vanishing_of(d1), mid_h, length, deviations),
                      _fan_segments(_vanishing_of(d2), mid_v, length, deviations)])


def test_large_residual_is_refused_relative_to_the_tolerance_in_force():
    """Средняя невязка инлайеров выше трёх четвертей допуска — сама по себе отказ.

    Порог задан долей действующего допуска, а не абсолютным числом пикселей: инлайер
    по определению лежит ближе допуска, поэтому абсолютный порог выше допуска был бы
    недостижим — ровно тот мёртвый код, что уже дважды находился в этой метрике.
    Проверяются обе стороны при допуске 8 px: невязка 1.7 px проходит, 6.2 px — нет,
    и при этом поддержка, покрытие и угол у обоих входов одинаковы.
    """
    mild = (0.0, 0.0, 2.0, -2.0, 2.2, -2.2, 2.4, -2.4, 2.6, -2.6)
    heavy = (0.0, 0.0, 7.7, -7.7, 7.7, -7.7, 7.7, -7.7, 7.7, -7.7)
    frame = (20000, 15000)

    ok = estimate_vanishing_points(_wide_fan_bundles(mild), frame, K, threshold_px=8.0)[2]
    bad = estimate_vanishing_points(_wide_fan_bundles(heavy), frame, K, threshold_px=8.0)[2]

    assert ok.support_h == bad.support_h == 10
    assert ok.support_v == bad.support_v == 10
    assert ok.coverage == bad.coverage == pytest.approx(1.0)
    assert ok.orthogonality_deg == pytest.approx(bad.orthogonality_deg, abs=0.5)
    assert ok.residual_px < 0.75 * 8.0 < bad.residual_px
    assert ok.reasons == []
    assert bad.reasons == [f"велика невязка инлайеров: {bad.residual_px:.2f} px"]
    assert bad.value == 0.0


def test_low_coverage_is_refused_even_with_a_perfect_core():
    """Шестнадцать безупречных прямых среди сотни посторонних — не основание доверять.

    Ядро согласовано идеально: невязка нулевая, угол ровно прямой, поддержки хватает.
    Но поддержала гипотезу лишь восьмая часть кадра, а остальное детектор увидел иначе,
    и такую плоскость подтверждать нельзя.
    """
    core = _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)])
    junk = np.random.default_rng(7).uniform(200.0, 3800.0, size=(104, 4))

    _, _, conf = estimate_vanishing_points(np.vstack([core, junk]), _FRAME, K)
    assert conf.support_h == 8 and conf.support_v == 8
    assert conf.residual_px == pytest.approx(0.0, abs=1e-6)
    assert conf.orthogonality_deg == pytest.approx(90.0, abs=0.5)
    assert conf.coverage < 0.2
    assert conf.reasons == ["малая доля отрезков поддержала точки схода"]
    assert conf.value == 0.0


def test_support_crammed_into_a_strip_of_a_large_frame_is_refused():
    """Тот же пучок в маленьком кадре годен, а в большом занимает полоску — и отвергается.

    Отрезки не меняются вовсе, меняется только заявленный размер кадра. Охват
    считается по каждой оси отдельно: полоска в четверть ширины и четверть высоты
    двадцатитысячного кадра не свидетельствует о плоскости всего снимка.
    """
    segs = _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)])

    small = estimate_vanishing_points(segs, _FRAME, K)[2]
    large = estimate_vanishing_points(segs, (16000, 12000), K)[2]

    assert small.support_h == large.support_h == 8
    assert small.reasons == []
    assert large.reasons == ["поддержавшие отрезки занимают малую долю кадра"]
    assert large.value == 0.0


def test_confidence_tracks_the_angle_when_nothing_else_changes():
    """Три входа, различающиеся только углом между направлениями: доверие строго падает.

    Невязка у всех трёх нулевая, покрытие единица, поддержка одинаковая. Если угол в
    доверие не входит, все три дадут одно и то же число.
    """
    values = []
    for angle in (90.0, 85.0, 80.0):
        conf = estimate_vanishing_points(
            _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(angle)]),
            _FRAME, K)[2]
        assert conf.reasons == []
        assert conf.coverage == pytest.approx(1.0)
        assert conf.residual_px == pytest.approx(0.0, abs=1e-6)
        assert conf.orthogonality_deg == pytest.approx(angle, abs=0.5)
        values.append(conf.value)

    assert values[0] > values[1] > values[2]
    assert values[0] == pytest.approx(1.0, abs=1e-6)


def test_confidence_tracks_the_residual_when_nothing_else_changes():
    """Два входа, различающиеся только невязкой: доверие строго падает.

    Угол прямой у обоих, покрытие единица, поддержка одинаковая.
    """
    vps = [_vanishing_of(d) for d in _direction_pair(90.0)]
    dev = (0.0, 0.0, 1.2, -1.2, 1.5, -1.5, 1.8, -1.8)

    exact = estimate_vanishing_points(_bundles_towards(*vps), _FRAME, K)[2]
    noisy = estimate_vanishing_points(_bundles_towards(*vps, dev, dev), _FRAME, K)[2]

    assert exact.reasons == [] and noisy.reasons == []
    assert exact.coverage == noisy.coverage == pytest.approx(1.0)
    assert exact.support_h == noisy.support_h
    assert exact.orthogonality_deg == pytest.approx(noisy.orthogonality_deg, abs=0.5)
    assert noisy.residual_px > exact.residual_px
    assert noisy.value < exact.value


def _near_parallel_horizontals(n=12, jitter_deg=1.0, x=2000.0, half=210.0, seed=3):
    """Почти параллельный горизонтальный пучок, чьи направления лежат по обе стороны 0°.

    Именно этот случай разоблачает наивный подсчёт веера: направления 179.9° и 0.03°
    отстоят друг от друга на 0.13°, а по величине — почти на 180°.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = np.radians(rng.uniform(-jitter_deg, jitter_deg))
        d = np.array([np.cos(a), np.sin(a)]) * half
        cy = 300.0 + i * 220.0
        out.append([x - d[0], cy - d[1], x + d[0], cy + d[1]])
    return np.array(out)


def test_one_ill_conditioned_bundle_is_enough_to_refuse():
    """Сходящийся вертикальный пучок не выкупает почти параллельный горизонтальный.

    Дополнительно проверяется склейка направлений через 0°: углы пучка лежат и у
    179.9°, и у 0.03°, то есть отстоят на десятые доли градуса, а не на 180. Наивный
    размах max - min насчитал бы здесь веер почти в 180° и объявил бы безнадёжно
    параллельный пучок превосходно обусловленным.
    """
    segs = np.vstack([
        _near_parallel_horizontals(),
        _fan_segments(_vanishing_of(_direction_pair(90.0)[1]), _MID_V, 900.0, (0.0,) * 8),
    ])
    angles = np.degrees(np.arctan2(segs[:12, 3] - segs[:12, 1],
                                   segs[:12, 2] - segs[:12, 0])) % 180.0
    assert angles.min() < 1.0 and angles.max() > 179.0     # пучок лежит по обе стороны 0°

    _, _, conf = estimate_vanishing_points(segs, _FRAME, K)
    assert conf.support_h >= 8 and conf.support_v >= 8
    assert any("не определена" in r for r in conf.reasons)
    assert conf.value == 0.0


@pytest.mark.parametrize("build", [make_scene, make_rich_scene],
                         ids=["бедная", "обогащённая"])
def test_direction_error_on_synthetic_scenes_stays_below_a_quarter_degree(build):
    """Числовая планка точности против истины K·(R_wc·d), а не против самой оценки.

    Прежняя редакция давала здесь 0.671° на горизонтальном пучке бедной сцены;
    пиксельный критерий инлайера довёл ту же величину до 0.005°. Планка в 0.25°
    закрепляет достигнутое и падает при любом возврате к нормировке на ‖v‖.
    """
    scene = build()
    vh, vv, _ = estimate_vanishing_points(detect_segments(scene.render()),
                                          scene.image_size, K)
    assert _angle_to_truth_deg(vh.point, scene, [1.0, 0.0, 0.0]) < 0.25
    assert _angle_to_truth_deg(vv.point, scene, [0.0, 1.0, 0.0]) < 0.25
