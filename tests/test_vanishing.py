import itertools

import cv2
import numpy as np
import pytest

from facade_digitizer.geometry.vanishing import (
    CONFIDENCE_THRESHOLD,
    INLIER_PX,
    INSTABILITY_REF_DEG,
    MAX_INSTABILITY_DEG,
    _fit_vanishing_point,
    _hypothesis_pairs,
    _split_bundles,
    detect_segments,
    direction_instability_deg,
    endpoint_deviation_px,
    estimate_vanishing_points,
)
from tests.test_synth import SIZE, K, make_rich_scene, make_scene


def test_detects_segments_on_synthetic_facade():
    img = make_scene().render()
    segs = detect_segments(img)
    assert segs.shape[1] == 4
    assert len(segs) >= 8  # четыре стороны стены плюс контуры проёма


def _segment_lengths(segs):
    return np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])


def test_filters_short_segments():
    """Фильтр длины отсекает ИМЕННО короткие, а не просто прореживает выборку.

    Прежде проверялось одно `len(long_only) < len(all_segs)` — условие, которому
    удовлетворяет любое прореживание, включая выброс каждого второго отрезка или
    обрезку списка по счёту. Здесь проверяется сам порог: в отфильтрованном
    наборе нет ни одного отрезка короче него, а в исходном такие есть (иначе
    фильтру нечего было бы делать и проверка зеленела бы вхолостую).
    """
    img = make_scene().render()
    threshold = 500.0
    long_only = detect_segments(img, min_length_px=threshold)
    all_segs = detect_segments(img, min_length_px=10.0)

    assert len(long_only) > 0
    assert _segment_lengths(long_only).min() >= threshold
    assert _segment_lengths(all_segs).min() < threshold
    assert len(long_only) < len(all_segs)
    # Отсечены ровно те, что короче порога, а не «столько же, но другие»:
    # число выживших совпадает с числом длинных в полном наборе.
    assert len(long_only) == int(np.count_nonzero(
        _segment_lengths(all_segs) >= threshold))


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


THRESHOLD_PREFIX = "доверие ниже порога"


def _gate_reasons(conf):
    """Причины, названные дискретными проверками, без итогового пояснения порога.

    Пояснение «доверие ниже порога» добавляется структурно, когда произведение
    непрерывных множителей уходит под порог при пройденных проверках. Тесты,
    проверяющие сами проверки, смотрят на дискретные причины.
    """
    return [r for r in conf.reasons if not r.startswith(THRESHOLD_PREFIX)]


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
    assert any("неустойчиво" in r for r in conf.reasons)
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


def test_inlier_tolerance_is_two_pixels_and_cuts_exactly_there():
    """Сам ДОПУСК инлайера, а не только его размерность, — прямой проверкой.

    Отсечка эта уже порождала дефект проекта: в прежней редакции фактический допуск
    достигал 3253 px, и инлайером становился любой отрезок. Одной охраны на неё было
    мало и после починки: мутация «удвоить допуск» давала `1 failed, 77 passed`,
    роняя только точный список отвергнутых ракурсов, а все числовые границы замера
    её пропускали — их полоса шире, чем эффект удвоения.

    Проверка двусторонняя и написана в АБСОЛЮТНЫХ пикселях, а не в долях `INLIER_PX`:
    отклонения, заданные через саму константу, выросли бы вместе с ней, и удвоение
    снова прошло бы незамеченным. Отклонения заданы построением `_fan_segments`
    (поворот на asin(d/(L/2))), то есть независимо от формулы модуля.
    """
    assert INLIER_PX == pytest.approx(2.0), (
        "допуск инлайера — часть контракта метрики доверия, а не свободный параметр; "
        "его изменение обязано быть замечено здесь, а не растворено в границах замера"
    )

    vp = (-9000.0, 1500.0)
    mids = [(2000.0, 500.0 + 300.0 * i) for i in range(10)]
    #            шесть точных        внутри 2 px   снаружи 2 px
    deviations = [0.0] * 6 + [1.5, -1.5] + [3.5, -3.5]
    segs = _fan_segments(vp, mids, 900.0, deviations)

    _, mask, _ = _fit_vanishing_point(segs, INLIER_PX, np.random.default_rng(0))
    assert list(mask) == [True] * 8 + [False] * 2, (
        "при допуске 2 px инлайерами обязаны быть ровно восемь отрезков, "
        "отклонённых на 0.0 и 1.5 px"
    )

    # Обе стороны отсечки. Без нижней проверка прошла бы и при допуске,
    # схлопнутом до нуля; без верхней — при допуске, распухшем до бесконечности.
    _, tight, _ = _fit_vanishing_point(segs, 1.0, np.random.default_rng(0))
    _, wide, _ = _fit_vanishing_point(segs, 4.0, np.random.default_rng(0))
    assert (int(tight.sum()), int(mask.sum()), int(wide.sum())) == (7, 8, 10), (
        f"число инлайеров при 1, 2 и 4 px: {int(tight.sum())}, {int(mask.sum())}, "
        f"{int(wide.sum())} — отсечка перестала быть восьмёркой ровно на 2 px")


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
    # Именно бесконечность, а не «что угодно нечисловое»: nan тоже не конечен, но при
    # сравнении с порогом ведёт себя как «меньше». Живого пути к nan здесь нет, это
    # профилактика — та же, что и в проверке охран меры устойчивости.
    assert conf.residual_px == float("inf")
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

    assert _gate_reasons(exact) == _gate_reasons(noisy) == _gate_reasons(skew) == []
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
    assert _gate_reasons(ok) == []
    # Крупная невязка заодно расшатывает и направление, поэтому причин у отказа две;
    # проверяется, что своя названа. Без проверки величины невязки осталась бы только
    # причина про неустойчивость, и отказ не назвал бы того, что действительно не так.
    assert f"велика невязка инлайеров: {bad.residual_px:.2f} px" in _gate_reasons(bad)
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
        assert _gate_reasons(conf) == []
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

    assert _gate_reasons(exact) == _gate_reasons(noisy) == []
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
    """Безупречный вертикальный пучок не выкупает расшатанный горизонтальный.

    Вертикали здесь сходятся точно, неустойчивость их направления равна нулю. Но оси
    фасада восстанавливаются парой, и негодной одной из них достаточно, чтобы
    плоскость была не определена. Поэтому неустойчивость берётся как худшая из двух,
    а не как лучшая и не как средняя.
    """
    segs = np.vstack([
        _near_parallel_horizontals(jitter_deg=2.0),
        _fan_segments(_vanishing_of(_direction_pair(90.0)[1]), _MID_V, 900.0, (0.0,) * 8),
    ])
    horizontal, vertical = segs[:12], segs[12:]
    assert direction_instability_deg(vertical, K) == pytest.approx(0.0, abs=1e-9)
    assert direction_instability_deg(horizontal, K) > 0.6

    _, _, conf = estimate_vanishing_points(segs, _FRAME, K)
    assert conf.support_h >= 6 and conf.support_v == 8
    assert any("неустойчиво" in r for r in _gate_reasons(conf))
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


def test_frontal_view_is_accepted():
    """Съёмка в упор — наилучший для задачи случай, а не худший.

    При фронтальном ракурсе точка схода уходит в бесконечность: её пиксельное
    положение не определено по построению, тогда как направление K⁻¹v определено
    превосходно. Мера устойчивости обязана отвечать на вопрос о направлении, иначе
    кадр с ошибкой в тысячные доли градуса получает отказ.
    """
    scene = make_scene(dx=0.0, dy=0.0)
    vh, vv, conf = estimate_vanishing_points(detect_segments(scene.render()),
                                             scene.image_size, K)
    assert _angle_to_truth_deg(vh.point, scene, [1.0, 0.0, 0.0]) < 0.05
    assert _angle_to_truth_deg(vv.point, scene, [0.0, 1.0, 0.0]) < 0.05
    assert conf.reasons == []
    assert conf.value > 0.9


@pytest.mark.parametrize("dx, dy", [(0.0, 0.0), (200.0, -150.0), (700.0, -450.0),
                                    (1500.0, -1000.0), (3000.0, -2000.0)],
                         ids=["фронт", "почти фронт", "слегка", "умеренно", "наклонно"])
def test_near_frontal_views_are_accepted_when_both_families_are_present(dx, dy):
    """Весь околофронтальный диапазон принимается, когда вертикали есть в кадре.

    Отказы в этом диапазоне на бедной сцене вызваны не фронтальностью, а тем, что
    вертикальный пучок там состоит из трёх-пяти обломков одного проёма. Измерено на
    шаге 100 мм: бедная сцена отвергает dx = 100 (ошибка 1.705°, отказ заслуженный),
    300 (1.494°), 400, 500, 600, а также 1100 и 5000 — то есть это не сплошная полоса
    и не только околофронтальная область. Обогащённая сцена на тех же ракурсах не
    отвергает ни одного, минимальное доверие среди принятых 0.687 по ряду до dx = 5000
    и 0.741 по околофронтальному ряду до 1500. Различитель тот же, что и в работе B, и
    говорит он о данных, а не об оценщике.
    """
    scene = make_rich_scene(dx=dx, dy=dy)
    vh, vv, conf = estimate_vanishing_points(detect_segments(scene.render()),
                                             scene.image_size, K)
    assert _angle_to_truth_deg(vh.point, scene, [1.0, 0.0, 0.0]) < 0.25
    assert _angle_to_truth_deg(vv.point, scene, [0.0, 1.0, 0.0]) < 0.25
    assert conf.reasons == []
    assert conf.value > 0.5


def test_direction_is_well_determined_where_the_point_itself_is_not():
    """Положение точки схода и направление на неё — разные по устойчивости величины.

    На фронтальном кадре точка схода стоит в сотнях тысяч пикселей от кадра и
    сдвигается на порядки от ничтожного шума, а направление при этом определено с
    точностью до сотых долей градуса. Проверяется обе половины утверждения сразу.
    """
    scene = make_scene(dx=0.0, dy=0.0)
    vh, _, conf = estimate_vanishing_points(detect_segments(scene.render()),
                                            scene.image_size, K)
    assert np.linalg.norm(vh.point[:2]) > 1e5 * abs(vh.point[2])   # точка убежала
    assert _angle_to_truth_deg(vh.point, scene, [1.0, 0.0, 0.0]) < 0.05
    assert conf.value > 0.5


def _confidence_population():
    """Разнородная выборка входов: годные, пограничные и заведомо негодные."""
    out = []
    vps90 = [_vanishing_of(d) for d in _direction_pair(90.0)]
    for angle in (90.0, 88.0, 85.0, 82.0, 80.0, 78.0, 76.0):
        out.append((_bundles_towards(*[_vanishing_of(d) for d in _direction_pair(angle)]),
                    _FRAME, 2.0))
    for scale in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8):
        dev = tuple(scale * d for d in (0.0, 0.0, 0.7, -0.7, 0.9, -0.9, 1.0, -1.0))
        out.append((_bundles_towards(*vps90, dev, dev), _FRAME, 2.0))
    core = _bundles_towards(*vps90)
    for extra in (0, 4, 8, 16, 24, 40, 60):
        junk = np.random.default_rng(3).uniform(200.0, 3800.0, size=(extra, 4))
        out.append((np.vstack([core, junk]) if extra else core, _FRAME, 2.0))
    for seed in range(6):
        for jitter in (0.2, 0.6, 1.0, 2.0):
            segs, size = _near_parallel_bundles(seed, jitter_deg=jitter)
            out.append((segs, size, 2.0))
    for width in (4000, 8000, 12000, 16000, 24000):
        out.append((core, (width, int(width * 0.75)), 2.0))
    return out


def test_low_confidence_always_names_a_reason():
    """Структурный инвариант контракта: доверие ниже порога обязано быть объяснено.

    Оператора вызывают не молча. Дискретные проверки покрывают не всё: доверие есть
    произведение непрерывных множителей и может уйти под порог, когда каждая проверка
    порознь пройдена. Раньше так получалось больше чем в половине случаев.
    """
    population = _confidence_population()
    assert len(population) >= 50

    silent = []
    for segs, size, tol in population:
        conf = estimate_vanishing_points(segs, size, K, threshold_px=tol)[2]
        if conf.value < CONFIDENCE_THRESHOLD and not conf.reasons:
            silent.append(conf)
        if conf.value >= CONFIDENCE_THRESHOLD:
            assert conf.reasons == [], "доверие выше порога не объясняют отказами"
    assert silent == [], f"{len(silent)} входов дали низкое доверие без единой причины"


def test_confidence_falls_when_coverage_falls():
    """Доля поддержавших отрезков входит в доверие наравне с невязкой и углом.

    Два входа с одним и тем же ядром: во втором к нему добавлены посторонние отрезки,
    которых ядро не объясняет. Поддержка, невязка, угол и устойчивость у них
    совпадают — различается только покрытие.
    """
    core = _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)])
    junk = np.random.default_rng(11).uniform(200.0, 3800.0, size=(16, 4))

    full = estimate_vanishing_points(core, _FRAME, K)[2]
    diluted = estimate_vanishing_points(np.vstack([core, junk]), _FRAME, K)[2]

    assert full.support_h == diluted.support_h and full.support_v == diluted.support_v
    assert full.residual_px == pytest.approx(diluted.residual_px, abs=1e-9)
    assert full.orthogonality_deg == pytest.approx(diluted.orthogonality_deg, abs=1e-6)
    assert diluted.coverage == pytest.approx(0.5)
    assert _gate_reasons(diluted) == []
    assert diluted.value < 0.75 * full.value


def test_one_pixel_of_residual_costs_a_third_of_the_confidence():
    """Чувствительность доверия к невязке задана в пикселях кадра, а не номинально.

    Невязка около пикселя — это уже заметно рыхлая подгонка, и доверие обязано
    отозваться на неё существенно, а не в третьем знаке. Проверяется величина отклика,
    а не формула: при прочих равных доверие должно потерять не меньше трети.
    """
    vps = [_vanishing_of(d) for d in _direction_pair(90.0)]
    dev = (0.0, 0.0, 1.2, -1.2, 1.5, -1.5, 1.8, -1.8)

    exact = estimate_vanishing_points(_bundles_towards(*vps), _FRAME, K)[2]
    noisy = estimate_vanishing_points(_bundles_towards(*vps, dev, dev), _FRAME, K)[2]

    assert noisy.residual_px == pytest.approx(1.125, rel=0.05)
    assert noisy.value < 0.67 * exact.value


def test_hypothesis_pairs_are_exhaustive_below_the_budget():
    """Ниже бюджета перебор пар исчерпывающий и без повторов — на этом стоит детерминизм.

    Свойство объявлено в докстринге и потому обязано проверяться: подмена перебора
    выборкой сохранила бы все видимые результаты на простых входах и тихо вернула бы
    зависимость от seed.
    """
    rng = np.random.default_rng(0)
    for n in (2, 5, 20, 63):
        pairs = _hypothesis_pairs(n, 2000, rng)
        assert len(pairs) == n * (n - 1) // 2
        assert {tuple(sorted(p)) for p in pairs} == set(itertools.combinations(range(n), 2))

    sampled = _hypothesis_pairs(200, 2000, rng)
    assert len(sampled) == 2000                      # выше бюджета включается выборка
    assert all(i != j for i, j in sampled)


def _ambiguous_bundle():
    """Вход, где верную гипотезу даёт малая доля пар: семь точных прямых среди тридцати трёх."""
    rng = np.random.default_rng(5)
    good = _fan_segments((-9000.0, 1500.0),
                         [(1500.0 + 260.0 * i, 500.0 + 300.0 * i) for i in range(7)],
                         700.0, (0.0,) * 7)
    noise = []
    for _ in range(33):
        a = np.radians(rng.uniform(-35.0, 35.0))
        d = np.array([np.cos(a), np.sin(a)]) * 320.0
        cx, cy = rng.uniform(600.0, 3400.0), rng.uniform(400.0, 2600.0)
        noise.append([cx - d[0], cy - d[1], cx + d[0], cy + d[1]])
    vertical = _fan_segments(_vanishing_of(_direction_pair(90.0)[1]), _MID_V, 900.0, (0.0,) * 8)
    return np.vstack([good, np.array(noise), vertical])


def test_result_does_not_depend_on_seed():
    """На входе, где верную гипотезу даёт одна пара из ста, ответ обязан быть один.

    Семь точных прямых среди тридцати трёх посторонних: случайная выборка гипотез
    находит верную не всегда, и ответ начинает зависеть от seed. Полный перебор пар
    снимает эту зависимость, и именно он здесь и проверяется — поведением, а не
    заглядыванием в реализацию.
    """
    segs = _ambiguous_bundle()
    results = set()
    for seed in range(6):
        conf = estimate_vanishing_points(segs, _FRAME, K, seed=seed)[2]
        results.add((conf.support_h, conf.support_v, round(conf.residual_px, 9),
                     round(conf.value, 9)))
    assert len(results) == 1, f"ответ зависит от seed: {sorted(results)}"
    only = next(iter(results))
    assert only[0] == 7                      # найдены ровно семь точных прямых
    assert only[2] == pytest.approx(0.0, abs=1e-9)


def test_tightest_hypothesis_wins_among_equally_supported():
    """Когда гипотезы собирают одинаковую поддержку, побеждает та, что села плотнее.

    Отклонения заданы построением, поэтому средняя невязка относительно номинальной
    точки схода известна точно и равна 1.84 px. При равной поддержке подгонка обязана
    найти не худшую точку, то есть отчитаться строго меньшим числом; без доразрешения
    ничьей она остановилась бы на первой попавшейся и вернула ровно 1.84.
    """
    deviations = (0.0, 0.0, 2.0, -2.0, 2.2, -2.2, 2.4, -2.4, 2.6, -2.6)
    about_nominal = float(np.mean(np.abs(deviations)))
    assert about_nominal == pytest.approx(1.84)

    conf = estimate_vanishing_points(_wide_fan_bundles(deviations), (20000, 15000), K,
                                     threshold_px=8.0)[2]
    assert conf.support_h == conf.support_v == len(deviations)
    assert conf.residual_px < about_nominal


def test_deviation_is_the_half_length_when_the_point_falls_on_the_midpoint():
    """Вырожденный случай достижим и обработан: точка схода села на середину отрезка.

    Направление на неё тогда не определено вовсе, и отрезок такую гипотезу не
    поддерживает — отклонение объявляется равным полудлине, то есть заведомо больше
    любого разумного допуска. Величина именно полудлина: концы отстоят от середины на
    неё, а не на полную длину.
    """
    segs = _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)])
    midpoint = np.array([(segs[0, 0] + segs[0, 2]) / 2.0,
                         (segs[0, 1] + segs[0, 3]) / 2.0, 1.0])
    half_length = float(np.hypot(segs[0, 2] - segs[0, 0], segs[0, 3] - segs[0, 1]) / 2.0)

    deviations = endpoint_deviation_px(segs, midpoint)
    assert half_length == pytest.approx(450.0)
    assert deviations[0] == pytest.approx(half_length)
    assert deviations[0] > 2.0                      # заведомо не инлайер


def _bundles_with_fan(spread, deviations=(0.0, 0.0, 0.5, -0.5, 0.7, -0.7, 0.9, -0.9)):
    """Пара пучков с заданным разносом середин: веер шире или уже при том же всём остальном.

    Отклонения заданы построением и потому одинаковы, точки схода те же, поддержка и
    покрытие те же. Меняется только то, насколько широко пучок расходится, — а от
    этого и зависит, насколько устойчиво определено направление.
    """
    d1, d2 = _direction_pair(90.0)
    n = len(deviations)
    mid_h = [(1800.0 + 0.3 * spread * i, 1200.0 + spread * i) for i in range(n)]
    mid_v = [(1200.0 + spread * i, 1500.0 + 0.3 * spread * i) for i in range(n)]
    return np.vstack([_fan_segments(_vanishing_of(d1), mid_h, 900.0, deviations),
                      _fan_segments(_vanishing_of(d2), mid_v, 900.0, deviations)])


def test_confidence_tracks_stability_when_nothing_else_changes():
    """Устойчивость направления входит в доверие наравне с покрытием, невязкой и углом.

    Два входа с одними и теми же точками схода и одними и теми же отклонениями: в
    первом середины отрезков разнесены на 1200 px, во втором на 300. Поддержка,
    покрытие, невязка и угол совпадают; узкий веер определяет направление хуже
    (неустойчивость 0.036° против 0.149°), и доверие обязано это учесть.
    """
    wide = estimate_vanishing_points(_bundles_with_fan(1200.0), _FRAME, K)[2]
    tight = estimate_vanishing_points(_bundles_with_fan(300.0), _FRAME, K)[2]

    assert wide.reasons == [] and tight.reasons == []
    assert wide.support_h == tight.support_h and wide.support_v == tight.support_v
    assert wide.coverage == tight.coverage == pytest.approx(1.0)
    assert wide.residual_px == pytest.approx(tight.residual_px, abs=0.01)
    assert wide.orthogonality_deg == pytest.approx(tight.orthogonality_deg, abs=0.1)
    assert tight.value < 0.9 * wide.value


def test_confidence_threshold_is_the_contract_value():
    """Порог передачи оператору закреплён числом, а не ссылкой на саму константу.

    Тест, импортирующий ту величину, которую призван закрепить, подтверждает лишь
    собственную непротиворечивость: подмена 0.5 на 0.05 прошла бы незамеченной.
    """
    assert CONFIDENCE_THRESHOLD == 0.5


def test_below_the_contract_threshold_a_reason_appears_and_above_it_does_not():
    """Граница инварианта проверяется числом 0.5, взятым из контракта, а не из кода.

    Два входа по разные стороны порога: у первого доверие 0.33 и причина названа, у
    второго 0.67 и список причин пуст.
    """
    low = estimate_vanishing_points(
        _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(80.0)]), _FRAME, K)[2]
    high = estimate_vanishing_points(
        _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(85.0)]), _FRAME, K)[2]

    assert low.value < 0.5 and low.reasons
    assert high.value > 0.5 and high.reasons == []


def _stability_dominated_input():
    """Вход, где под порог доверие уводит именно неустойчивость, а отсечка не сработала.

    Отсечка по неустойчивости стоит раньше, чем её множитель успевает стать
    наименьшим, поэтому величины подобраны совместно: покрытие 1.00, невязка 0.29 px,
    угол 85.9°, неустойчивость 0.30° при пороге отсечки 0.40. Перебор нашёл 206 таких
    конфигураций, так что область не игольное ушко.
    """
    spread, scale = 200.0, 0.6
    dev = tuple(scale * x for x in (0.0, 0.0, 0.5, -0.5, 0.7, -0.7, 0.9, -0.9))
    d1, d2 = _direction_pair(86.0)
    n = len(dev)
    mid_h = [(700.0 + 0.3 * spread * i, 400.0 + spread * i) for i in range(n)]
    mid_v = [(400.0 + spread * i, 700.0 + 0.3 * spread * i) for i in range(n)]
    return np.vstack([_fan_segments(_vanishing_of(d1), mid_h, 500.0, dev),
                      _fan_segments(_vanishing_of(d2), mid_v, 500.0, dev)]), (3000, 2400)


@pytest.mark.parametrize("case", ["угол", "невязка", "покрытие", "устойчивость"])
def test_reason_names_the_factor_that_actually_dropped(case):
    """Причина называет просевший множитель, а не первый попавшийся из списка.

    Оператору сообщают, чего именно не хватило. Четыре входа, в каждом под порог
    доверие уводит свой множитель; проверяется, что назван именно он.
    """
    vps90 = [_vanishing_of(d) for d in _direction_pair(90.0)]
    if case == "угол":
        segs, frame = _bundles_towards(
            *[_vanishing_of(d) for d in _direction_pair(80.0)]), _FRAME
        expected = "не ортогональны"
    elif case == "невязка":
        dev = (0.0, 0.0, 1.3, -1.3, 1.5, -1.5, 1.7, -1.7)
        segs, frame = _bundles_towards(*vps90, dev, dev), _FRAME
        expected = "велика невязка"
    elif case == "покрытие":
        junk = np.random.default_rng(3).uniform(200.0, 3800.0, size=(45, 4))
        segs, frame = np.vstack([_bundles_towards(*vps90), junk]), _FRAME
        expected = "малая доля отрезков"
    else:
        segs, frame = _stability_dominated_input()
        expected = "неустойчиво"

    conf = estimate_vanishing_points(segs, frame, K)[2]
    assert conf.value < CONFIDENCE_THRESHOLD
    assert _gate_reasons(conf) == [], "должен сработать инвариант, а не дискретная проверка"
    assert len(conf.reasons) == 1
    assert expected in conf.reasons[0]


def test_instability_is_infinite_when_the_direction_is_not_determined_at_all():
    """Обе охраны меры достижимы и возвращают бесконечность, а не число.

    Одна прямая направления не задаёт вовсе; шесть копий одной прямой не задают его
    тоже, сколько бы их ни было — матрица связей вырождена, и второе снизу
    сингулярное число обращается в ноль.
    """
    segs = _bundles_towards(*[_vanishing_of(d) for d in _direction_pair(90.0)])
    # Именно бесконечность, а не «что угодно нечисловое»: снятая охрана даёт 0/0, то
    # есть nan, который тоже не конечен, но при сравнении с порогом ведёт себя как
    # «меньше», и вырожденный пучок молча проходит отсечку.
    assert direction_instability_deg(segs[:1], K) == float("inf")
    assert direction_instability_deg(np.empty((0, 4)), K) == float("inf")

    six_copies = np.repeat(segs[:1], 6, axis=0)
    assert len(six_copies) == 6
    assert direction_instability_deg(six_copies, K) == float("inf")


def test_degenerate_bundle_is_reported_in_words_not_as_infinity():
    """Оператору не показывают «±inf°»: вырожденный пучок называется словами.

    Бесконечность в сообщении — не диагностика, а сбой форматирования: она не
    говорит, что именно случилось и что с этим делать.
    """
    good = _fan_segments(_vanishing_of(_direction_pair(90.0)[1]), _MID_V, 900.0, (0.0,) * 8)
    horizontal = _fan_segments((-9000.0, 1500.0), [(1500.0, 800.0)], 900.0, (0.0,))
    segs = np.vstack([np.repeat(horizontal, 6, axis=0), good])

    conf = estimate_vanishing_points(segs, _FRAME, K)[2]
    joined = " | ".join(conf.reasons)
    assert "inf" not in joined.lower()
    assert "вырожден" in joined
    assert conf.value == 0.0


def test_instability_cutoff_decides_at_the_chosen_value():
    """Выбранное значение отсечки закреплено поведением на реальном ракурсе.

    Ракурс dx = 4000 мм, dy = −2800 мм, дистанция 12 м даёт неустойчивость 0.401° —
    чуть выше выбранной отсечки 0.40 и заметно ниже 0.60, которая стояла до выбора.
    При 0.40 он отвергается отсечкой, при 0.60 принимается без единой причины, так что
    откат отсечки виден здесь исходом, а не оттенком формулировки.

    Отказ тут ложный: фактическая ошибка направления 0.238°. Это и есть цена выбора —
    она платится ложными вызовами, а не промахами, — и она сознательная: см. кривую
    компромисса в тесте ниже. Тест закрепляет не то, что отказ правильный, а то, что
    граница стоит там, где её поставили.
    """
    scene = make_scene(dx=4000.0, dy=-2800.0)
    segs = detect_segments(scene.render())
    horizontal, vertical = _split_bundles(segs)
    rng = np.random.default_rng(0)
    inst = max(direction_instability_deg(horizontal[_fit_vanishing_point(
                   horizontal, INLIER_PX, rng)[1]], K),
               direction_instability_deg(vertical[_fit_vanishing_point(
                   vertical, INLIER_PX, np.random.default_rng(0))[1]], K))
    assert 0.40 < inst < 0.60, f"ракурс перестал попадать в вилку: {inst:.4f}"

    _, _, conf = estimate_vanishing_points(segs, scene.image_size, K)
    assert any("неустойчиво" in r for r in _gate_reasons(conf))
    assert conf.value == 0.0


def test_instability_cutoff_trade_off_is_documented():
    """Кривая компромисса записана здесь, чтобы порог не подбирали заново вслепую.

    Замер на 315 ракурсах бедной сцены — шаг 100 мм до dx = 1500, три дистанции, три
    соглашения о dy, — перебор 126 пар «отсечка T, масштаб R»:

        T / R              | отказов из 315 | ложных | промахов | худший принятый
        0.60 / 1.50 (кр.4) | 23             | 15     | 4        | 1.556°
        0.60 / 0.60        | 41             | 27     | 3        | 1.460°
        0.40 / 0.60 (наш)  | 51             | 35     | 3        | 1.460°
        0.25 / 0.45        | 88 (28 %)      | 62     | 0        | 0.905°

    **Отсечка 0.40 выбрана не ради меньшего числа промахов.** При том же масштабе 0.60
    подъём отсечки до 0.60 не добавляет ни одного промаха — промахов те же три, — а
    снимает 10 отказов и 8 ложных вызовов. По целевой функции «равные промахи, затем
    максимум принятых» выигрывала бы пара 0.60 / 0.60.

    Выбор сделан по третьему приоритету — устойчивости самой отсечки. У пары 0.60 / 0.60
    произведение непрерывных множителей на границе отсечки равно ровно 0.500, то есть
    она решает точно на структурном пороге, где исход определяется численным шумом.
    Параметр, решающий на самой границе, хрупок ровно тем способом, который в этой
    задаче уже дважды давал дефекты: мёртвый масштаб влияния в редакции 4 и три отсечки,
    формирующие только объяснение. У выбранной пары запас есть: произведение на границе
    0.600. **Плата названа прямо: 8 дополнительных ложных вызовов на 315 кадров, то есть
    2.5 процентных пункта.**

    Ноль промахов достижим — его дают 78 пар из 126, — но дешевейшая из них зовёт
    оператора на 28 % кадров, и 70 % этих вызовов впустую. Переход 0.40 / 0.60 →
    0.25 / 0.45 убирает три промаха ценой 27 дополнительных ложных вызовов, то есть
    примерно девять к одному; при этом меняется не только отсечка, но и масштаб, так
    что сдвигать их порознь нельзя. Оставшиеся три промаха лежат в 1.0…1.46°, и все три
    метрика ранжирует в верхнем хвосте годного распределения (0.26…0.35 при медиане
    годных 0.0899 и максимуме 0.3945) — как подозрительные, просто не выше порога.
    """
    assert MAX_INSTABILITY_DEG == 0.40
    assert INSTABILITY_REF_DEG == 0.60
