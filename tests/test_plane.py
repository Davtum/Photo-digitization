import cv2
import numpy as np
import pytest

from facade_digitizer.geometry.homography import camera_pose
from facade_digitizer.pipeline.plane import estimate_plane, estimate_plane_manual
from tests.test_homography import CORNERS, apply
from tests.test_synth import SIZE, K, make_rich_scene, make_scene

# «Хороший случай» берётся на ОБОГАЩЁННОЙ сцене. На бедной доверие измерено равным
# 0.5597 при пороге 0.5, а неустойчивость направления — 0.355° при отсечке 0.40°,
# то есть бедная сцена сидит вплотную к решающей отсечке задачи 7: такой тест упал
# бы от любого ухудшения оценки точек схода, не сказав, что сломалась не
# оркестрация. На обогащённой доверие 0.7389 и неустойчивость 0.158°. Близость
# бедной сцены к отсечке зафиксирована отдельной характеризующей проверкой ниже.


def test_plane_is_estimated_on_a_good_scene():
    sc = make_rich_scene()
    res = estimate_plane(sc.render(), K)
    assert res.method == "vanishing_points"
    assert not res.needs_operator
    assert res.confidence.value > 0.5


def test_estimated_homography_rectifies_the_facade():
    """Гомография проверяется по существу, а не по факту непустоты."""
    sc = make_rich_scene()
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
    sc = make_rich_scene()
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


def _direction_angle_deg(a, b):
    """Угол между направлениями K⁻¹a и K⁻¹b, без учёта знака точки схода."""
    Kinv = np.linalg.inv(K)
    da = Kinv @ np.asarray(a, dtype=float)
    db = Kinv @ np.asarray(b, dtype=float)
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    return float(np.degrees(np.arccos(min(1.0, abs(float(da @ db))))))


def test_plane_result_labels_which_bundle_each_vanishing_point_came_from():
    """`vh` — точка схода ГОРИЗОНТАЛЬНОГО пучка, `vv` — вертикального.

    Различитель написан здесь потому, что через `camera_pose` перестановка этих
    двух полей НЕ наблюдаема, и это свойство постановки, а не недосмотр: опорная
    точка есть образ точки схода нормали, нормаль есть векторное произведение двух
    направлений, ортогонализация Грама — Шмидта сохраняет их линейную ОБОЛОЧКУ, а
    перестановка меняет у нормали только знак — знак же исчезает при делении на
    третью компоненту. Измерено на реальном детекторе: сдвиг опорной точки от
    перестановки равен 0.000000 мм на всех проверенных ракурсах, включая тот, где
    восстановленные направления расходятся с прямым углом на 0.33°.

    Наблюдаема перестановка по тому, с каким направлением фасада согласуется каждое
    поле. Эталон берётся из истины сцены, а не из кода оценщика.
    """
    sc = make_rich_scene()
    res = estimate_plane(sc.render(), K)
    true_h = K @ (sc.R_wc @ np.array([1.0, 0.0, 0.0]))
    true_v = K @ (sc.R_wc @ np.array([0.0, 1.0, 0.0]))

    assert _direction_angle_deg(res.vh, true_h) < 1.0
    assert _direction_angle_deg(res.vv, true_v) < 1.0
    # И не согласуется с чужим направлением: без этой пары перестановка прошла бы,
    # если бы оценщик выдал оба поля близкими к одной и той же оси.
    assert _direction_angle_deg(res.vh, true_v) > 80.0
    assert _direction_angle_deg(res.vv, true_h) > 80.0


def test_manual_plane_cannot_be_turned_into_a_pose_silently():
    """Сборка конвейера (задача 15) зовёт `camera_pose` на ОБОИХ путях п. 4.2.

    На ручном пути `PlaneResult` не несёт ни точек схода, ни калибровки, а
    ректифицированная система имеет произвольные начало и масштаб. Прежняя редакция
    `camera_pose` этого не замечала: она читала только `mm_per_rect_unit` и
    `origin_rect` и выдавала позу, в которой расстояние до плоскости равнялось
    миллиметру. Здесь закреплено, что вместо этого будет названный отказ.
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, size_mm=(1460.0, 1900.0))
    assert res.vh is None and res.vv is None

    with pytest.raises(ValueError, match="поза не восстановима без точек схода"):
        camera_pose(res.H, res.vh, res.vv, K, 1.0, (0.0, 0.0))


def test_manual_confidence_is_not_borrowed_from_the_automatic_path():
    """Ручной вариант не должен изображать измеренное доверие."""
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, aspect_ratio=2.0)
    assert res.confidence.reasons == []
    assert res.confidence.support_h == 0 and res.confidence.support_v == 0


def test_threshold_is_honoured():
    """Порог доверия действительно решает, а не декоративен."""
    sc = make_rich_scene()
    strict = estimate_plane(sc.render(), K, min_confidence=0.999)
    assert strict.needs_operator
    assert strict.H is None


# Проверки ниже добавлены сверх набора брифа: мутационная проверка показала, что
# восемь тестов брифа переживают мутации решений, которых бриф не описывал, — «порог
# отключается аргументом», «отказ всё равно отдаёт точки схода», «ручное доверие
# записывает правдоподобные измеренные числа».

# Шаг сетки и размер угла фасада для сгрудившегося пучка. Подобраны измерением:
# при них метрика доверия называет ровно одну причину — «поддержавшие отрезки
# занимают малую долю кадра», — а сами точки схода остаются безупречными
# (ортогональность 89.9°), и гомография по ним СТРОИТСЯ. Это и нужно: мутант,
# снявший причины из условия отказа, доживает до утверждения, а не падает раньше.
CLUSTER_CORNER_MM = 3000.0
CLUSTER_STEP_MM = 500.0


def render_clustered_bundle():
    """Кадр с геометрически безупречными отрезками, сгрудившимися в одном углу.

    Прямые идут вдоль осей фасада и спроецированы той же сценой, поэтому их точки
    схода — истинные точки схода фасада, а не приближение. Порочен здесь не пучок,
    а его расположение: середины поддержавших отрезков занимают малую долю кадра,
    и метрика доверия отказывает по НАЗВАННОЙ причине, не вырождаясь.
    """
    sc = make_scene()
    img = np.full((SIZE[1], SIZE[0]), 200, dtype=np.uint8)
    for t in np.arange(CLUSTER_STEP_MM, CLUSTER_CORNER_MM + 1.0, CLUSTER_STEP_MM):
        for line in (np.array([[0.0, t], [CLUSTER_CORNER_MM, t]]),
                     np.array([[t, 0.0], [t, CLUSTER_CORNER_MM]])):
            p = sc.project(line)
            cv2.line(img, tuple(np.round(p[0]).astype(int)),
                     tuple(np.round(p[1]).astype(int)), 60, 1)
    return img


def test_the_clustered_bundle_is_a_valid_probe():
    """Охрана самой пробы: пучок отвергается ПО ПРИЧИНЕ, а не по вырождению.

    Без этой проверки соседний тест мог бы «работать» на входе, где точки схода
    вырождены: тогда он подтверждал бы не запрет догадки, а невозможность её
    построить. Здесь же точки схода истинные — ортогональность держится у 90°.
    """
    res = estimate_plane(render_clustered_bundle(), K)
    assert res.confidence.reasons == ["поддержавшие отрезки занимают малую долю кадра"]
    assert res.confidence.support_h >= 2 and res.confidence.support_v >= 2
    assert res.confidence.orthogonality_deg == pytest.approx(90.0, abs=1.0)


def test_a_named_reason_refuses_at_any_threshold():
    """Контракт п. 4.2 не отключается аргументом `min_confidence=0.0`.

    Порог — не единственная охрана: непустой список причин означает, что метрика
    доверия отвергла вход по существу, и такой вход не годится ни при каком пороге.
    Иначе один аргумент вызывающего кода снимал бы весь запрет на догадку.

    Вход подобран так, чтобы утверждение сработало НА ДЕЛЕ: на шумовом кадре код,
    снявший причины из условия, падал бы внутри построения гомографии по
    вырожденным точкам схода — убийство мутанта случалось бы по исключению, а не по
    проверяемой причине, и уцелело бы только до ближайшей смены пути отказа.
    """
    res = estimate_plane(render_clustered_bundle(), K, min_confidence=0.0)
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


def test_manual_confidence_cannot_slip_through_a_threshold_guard():
    """Неизмеренные величины ручного пути равны `None`, и это проверяется охраной.

    Нулевая невязка и ровно 90° — правдоподобные ИЗМЕРЕННЫЕ значения: потребитель
    не отличил бы их от настоящих. `NaN` отличим глазом, но не охраной: сравнение с
    ним ложно, и проверка вида `residual_px > X` пропустила бы ручной путь молча.
    Поэтому проверяется не только само значение, но и ПОВЕДЕНИЕ под охраной —
    сравнение обязано падать, а не возвращать ложь.
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    res = estimate_plane_manual(pts, aspect_ratio=2.0)
    assert res.confidence.residual_px is None
    assert res.confidence.orthogonality_deg is None
    with pytest.raises(TypeError):
        _ = res.confidence.residual_px > 0.5        # охрана «велика невязка»
    with pytest.raises(TypeError):
        _ = res.confidence.orthogonality_deg < 75.0  # охрана «не ортогональны"


def test_manual_confidence_is_not_shared_between_results():
    """Доверие ручного пути строится заново на каждый вызов.

    Общая на модуль константа была бы изменяемой через `reasons`: дописанная одним
    потребителем причина появилась бы у всех результатов, включая уже отданные.
    """
    pts = np.array([[100.0, 100.0], [900.0, 120.0], [880.0, 700.0], [120.0, 690.0]])
    first = estimate_plane_manual(pts, aspect_ratio=2.0).confidence
    second = estimate_plane_manual(pts, aspect_ratio=2.0).confidence
    first.reasons.append("дописано потребителем")
    assert second.reasons == []


def test_the_sparse_scene_sits_close_to_the_cutoff():
    """ХАРАКТЕРИЗУЮЩАЯ проверка: не требование к системе, а зафиксированный факт.

    Бедная сцена проходит порог, но еле-еле. Измерено: доверие 0.5597 против 0.7389
    у обогащённой при пороге 0.5; неустойчивость направления 0.355° при отсечке
    0.40°; вертикальный пучок поддержан 8 отрезками против 48. Причина видна:
    вертикальных прямых на бедной сцене почти нет — только кромки единственного
    проёма.

    Тест существует ради того, чтобы близость к отсечке была записана, а не
    обнаружилась когда-нибудь падением «хорошего случая». Если он упадёт, это
    означает не поломку оркестрации, а смену поведения оценщика точек схода —
    и число здесь надо ПЕРЕИЗМЕРИТЬ, а не подогнать.
    """
    sparse = estimate_plane(make_scene().render(), K).confidence
    rich = estimate_plane(make_rich_scene().render(), K).confidence
    assert 0.5 < sparse.value < 0.6            # измерено 0.5597: запас меньше 0.1
    assert rich.value - sparse.value > 0.1     # измерено 0.7389 против 0.5597
    assert sparse.support_v * 3 < rich.support_v   # измерено 8 против 48
