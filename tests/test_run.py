"""Свойства сборки конвейера: снимок -> модель фасада одним проходом.

Каждая проверка названа свойством, а не номером шага. Сцена и внутренние параметры
берутся из `tests/test_synth.py`, роль оператора играют проекции истинных углов
фасада (`tests/test_homography.CORNERS`) — ровно как в `tests/test_integration.py`.
Из истины сцены берутся ТОЛЬКО начало отсчёта и одна опорная длина; гомография,
точки схода, поза, поле углов и разрешение приходят из кадра.

Снимок пишется на диск через `save_image` без EXIF-блока, поэтому EXIF в нём нет:
путь «внутренние параметры из таблицы» входит в проверку, а не обходится
подстановкой.
"""
import json
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
import piexif
import pytest

from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import HALF_TURN_UNRESOLVED, camera_pose
from facade_digitizer.pipeline.calib import (
    CalibrationProfile,
    intrinsics_from_meta,
    save_profile,
)
from facade_digitizer.pipeline.io import load_image, save_image
from facade_digitizer.pipeline.plane import estimate_plane
from facade_digitizer.pipeline.quality import DEFAULT as QUALITY_THRESHOLDS
from facade_digitizer.pipeline.quality import THETA_MAX_DEG
from facade_digitizer.pipeline.run import (
    SIGMA_REL_TOLERANCE,
    OperatorReference,
    _apply,
    _camera_record,
    _field_grid,
    _frame_fields,
    _gsd_at,
    main,
    process,
)
from facade_digitizer.schema import FacadeModel
from tests.test_homography import CORNERS
from tests.test_synth import SIZE, K, make_scene

FACADE_W_MM = 20000.0

#: Масштаб выходного растра. Свободный параметр визуализации, не GSD и не масштаб
#: ректифицированной системы: ровно то же значение, на котором считает
#: `tests/test_integration.py`, и оно лежит внутри достижимых границ.
RASTER_MM_PER_PX = 10.0

#: Дистанция съёмки. Не умолчание `make_scene`, и причина существенная: при 12 м
#: двадцатиметровая сторона фасада в кадр 5280x3956 при f = 3600 px НЕ помещается,
#: её углы проецируются за край снимка. Оператор такую базу указать не может, а
#: конвейер обязан такой вход отвергнуть (см. проверку про точки вне кадра).
#: 20 м — ближайшая рабочая дистанция, с которой вся сторона видна.
DISTANCE_MM = 20000.0



def _reference(scene, span_mm=FACADE_W_MM, span_fraction=1.0):
    """Опорные указания оператора: начало отсчёта и одна база известной длины.

    `span_fraction` укорачивает базу вдоль нижней стороны фасада, оставляя её левый
    конец на месте: так меняется ровно одна величина — длина базы.
    """
    far = np.array([[FACADE_W_MM * span_fraction, 0.0]])
    pts = scene.project(np.vstack([CORNERS[:1], far]))
    return OperatorReference(
        origin_px=(float(pts[0][0]), float(pts[0][1])),
        span_px=((float(pts[0][0]), float(pts[0][1])),
                 (float(pts[1][0]), float(pts[1][1]))),
        span_mm=span_mm * span_fraction,
    )


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    """Каталог под снимок. Фикстура намеренно не делает ничего сверх этого.

    Сам проход конвейера вынесен из фикстуры в `baseline`, вызываемый ИЗ ТЕЛА теста.
    Разница не косметическая: исключение, поднятое в фикстуре, pytest записывает в
    junit-XML как `error`, а поднятое в теле — как `failure`. При мутационном
    испытании это ровно то различие, которым отделяют пойманную мутацию от
    сломавшегося модуля, и складывать их в одну корзину значит потерять критерий.
    """
    return tmp_path_factory.mktemp("run")


_BASELINE = {}


def baseline(workdir):
    """Один проход конвейера на наклонном ракурсе. Считается один раз на модуль.

    Результат — или исключение — запоминается и переиспользуется: повторный прогон
    полной цепочки на кадре 20 Мп стоит около полутора секунд на тест.
    """
    if not _BASELINE:
        scene = make_scene(dist=DISTANCE_MM)
        image = scene.render()
        path = workdir / "facade.png"
        save_image(path, image)
        reference = _reference(scene)
        try:
            model = process(path, operator_reference=reference,
                            raster_mm_per_px=RASTER_MM_PER_PX)
        except BaseException as error:
            _BASELINE["error"] = error
            raise
        _BASELINE["value"] = SimpleNamespace(scene=scene, image=image, path=path,
                                             reference=reference, model=model)
    if "error" in _BASELINE:
        raise _BASELINE["error"]
    return _BASELINE["value"]


def _numbers(node, out):
    """Все числа разобранного JSON, включая лежащие во вложенных списках."""
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _numbers(value, out)
    elif isinstance(node, list):
        for value in node:
            _numbers(value, out)


def test_model_survives_round_trip_through_json(workdir):
    """Схема валидна на обратном разборе, версия схемы — 1.1."""
    data = baseline(workdir)
    payload = data.model.model_dump_json()
    restored = FacadeModel.model_validate_json(payload)
    assert restored.schema_version == "1.1"
    assert restored.mode == "assisted"
    assert restored.coverage == "partial"
    assert restored.elements == []


def test_output_carries_no_infinities_or_nan(workdir):
    """В выходе нет бесконечностей и NaN — по разобранному JSON, а не по подстроке.

    `json.dumps` пишет бесконечность словом `Infinity`, а `NaN` встречается и внутри
    вложенных списков гомографии; проверка вхождения подстроки в текст прошла бы при
    любом входе. Число проверенных чисел утверждается отдельно: обход, ничего не
    нашедший, — это молчаливый успех, а не чистый выход.
    """
    data = baseline(workdir)
    found = []
    _numbers(json.loads(data.model.model_dump_json()), found)
    assert len(found) > 20
    assert all(math.isfinite(value) for value in found)


def test_pose_is_recorded_together_with_its_origin(workdir):
    """Поза записана вместе с происхождением: координаты и признак поворота на 180°."""
    data = baseline(workdir)
    pose = data.model.images[0].pose_to_facade
    assert set(pose) >= {"cx", "cy", "cz", "half_turn"}
    assert pose["half_turn"]["resolved"] is False
    assert pose["half_turn"]["reason"] == HALF_TURN_UNRESOLVED
    assert pose["half_turn"]["assumption"]


def test_pose_matches_scene_truth(workdir):
    """Поза близка к истине сцены.

    Допуск 2 % расстояния взят с запасом к измеренному в задаче 14 (P95 0.601 % по
    опорной точке на разреженной сцене). Цель — не повторить тот замер, а поймать
    сборку, потерявшую или переставившую аргументы: именно такая ошибка ставила
    камеру в десятки километров от фасада и проходила все прочие проверки.
    """
    data = baseline(workdir)
    truth = data.scene.camera_on_plane()
    pose = data.model.images[0].pose_to_facade
    foot_err = math.hypot(pose["cx"] - truth.cx, pose["cy"] - truth.cy)
    assert foot_err < 0.02 * truth.cz
    assert abs(pose["cz"] - truth.cz) < 0.02 * truth.cz


def test_camera_tilt_is_not_the_angle_field(workdir):
    """`theta_cam_deg` строго внутри поля углов, а не совпадает с его краем."""
    data = baseline(workdir)
    record = data.model.images[0]
    field = record.theta_field_deg
    assert field["min"] < record.theta_cam_deg < field["max"]
    assert field["min"] < field["p95"] <= field["max"]
    # Ракурс обязан быть наклонным, иначе строгие неравенства выше ничего не значат.
    assert field["max"] - field["min"] > 5.0


def test_resolution_is_measured_not_copied_from_the_argument(workdir):
    """GSD измерен: границы различны и ни одна не равна масштабу растра."""
    data = baseline(workdir)
    quality = data.model.images[0].quality
    assert quality.gsd_mm_px_max > quality.gsd_mm_px_min
    assert quality.gsd_mm_px_min != pytest.approx(RASTER_MM_PER_PX)
    assert quality.gsd_mm_px_max != pytest.approx(RASTER_MM_PER_PX)


def test_low_confidence_hands_control_to_operator(tmp_path):
    """Низкое доверие выражено данными: отказ, а не догадка и не исключение."""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(900, 1200), dtype=np.uint8)
    path = tmp_path / "noise.png"
    save_image(path, noise)
    reference = OperatorReference(origin_px=(100.0, 800.0),
                                  span_px=((100.0, 800.0), (1100.0, 800.0)),
                                  span_mm=5000.0)

    model = process(path, operator_reference=reference,
                    raster_mm_per_px=RASTER_MM_PER_PX)

    record = model.images[0]
    assert record.rectification.needs_operator is True
    assert isinstance(record.rectification.confidence, float)
    assert record.homography is None
    assert record.pose_to_facade is None
    assert record.theta_cam_deg is None
    assert record.theta_field_deg is None
    assert record.quality.verdict != "ok"
    assert record.quality.reasons
    # Отказ обязан быть выражен данными: модель всё равно валидна по схеме.
    assert FacadeModel.model_validate_json(model.model_dump_json()).mode == "assisted"


def test_needs_operator_records_the_sentinel_angle_not_a_measurement(tmp_path):
    """Угол визирования на отказном пути — предельные 90°, а не измеренное число.

    `_needs_operator_model` не может измерить угол — он требует позы, а плоскость не
    восстановлена, — и записывает 90° как названный предел, не как измерение
    (`run.py`, докстринг `_needs_operator_model`). Ноль в семантике поля означает
    «идеально фронтальный, отличные условия», то есть противоположное по смыслу
    утверждение: мутация 90.0 -> 0.0 превратила бы честный отказ в самый лестный
    возможный результат.
    """
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(900, 1200), dtype=np.uint8)
    path = tmp_path / "noise.png"
    save_image(path, noise)
    reference = OperatorReference(origin_px=(100.0, 800.0),
                                  span_px=((100.0, 800.0), (1100.0, 800.0)),
                                  span_mm=5000.0)

    model = process(path, operator_reference=reference, raster_mm_per_px=RASTER_MM_PER_PX)

    assert model.images[0].rectification.needs_operator is True
    assert model.images[0].quality.theta_field_deg_p95 == 90.0
    assert model.images[0].quality.verdict == "degraded"


def test_needs_operator_never_reports_a_measured_tolerance(tmp_path):
    """На отказном пути `meets_tolerance` — всегда `False`, даже когда σ мала сама по себе.

    Плоскость не восстановлена (`needs_operator = True`, `homography = None`), и
    допуск не достигнут НИЧЕМ — это утверждает докстринг `_needs_operator_model`
    дословно. Без принудительного `False` признак стал бы результатом голого
    сравнения `sigma_rel <= SIGMA_REL_TOLERANCE`, и вход ниже подобран так, чтобы
    именно это сравнение обмануло: опорная база во всю ширину кадра (~3000 px) при
    span_mm = 5000 даёт σ около 0.22 %, что МЕНЬШЕ порога 0.28 % — то есть без
    принуждения признак стал бы `True` на записи, где ничего не измерено.
    """
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(2000, 3000), dtype=np.uint8)
    path = tmp_path / "noise_wide.png"
    save_image(path, noise)
    reference = OperatorReference(origin_px=(50.0, 1000.0),
                                  span_px=((50.0, 1000.0), (2950.0, 1000.0)),
                                  span_mm=5000.0)

    model = process(path, operator_reference=reference, raster_mm_per_px=RASTER_MM_PER_PX)

    record = model.images[0]
    assert record.rectification.needs_operator is True
    assert record.homography is None
    scale = model.facade.scale
    # Голое сравнение сочло бы допуск достигнутым — это и есть западня входа.
    assert scale.sigma_rel < SIGMA_REL_TOLERANCE
    assert scale.meets_tolerance is False


def test_needs_operator_rejects_a_uniformly_flat_frame(tmp_path):
    """Строго однородный кадр на отказном пути получает `reject`, а не `degraded`.

    Ветка `verdict == "reject"` внутри `_needs_operator_model` достижима только при
    нулевой дисперсии яркости: `sharpness` считает вариацию лапласиана, нормированную
    на дисперсию, и на постоянном кадре она равна ровно нулю. Проверяются обе ветви:
    здесь — `reject` с названной причиной, а `test_low_confidence_hands_control_to_operator`
    и `test_needs_operator_records_the_sentinel_angle_not_a_measurement` закрепляют
    `degraded` на кадре с достаточной резкостью.
    """
    uniform = np.full((900, 1200), 128, dtype=np.uint8)
    path = tmp_path / "uniform.png"
    save_image(path, uniform)
    reference = OperatorReference(origin_px=(100.0, 800.0),
                                  span_px=((100.0, 800.0), (1100.0, 800.0)),
                                  span_mm=5000.0)

    model = process(path, operator_reference=reference, raster_mm_per_px=RASTER_MM_PER_PX)

    record = model.images[0]
    assert record.rectification.needs_operator is True
    assert record.quality.verdict == "reject"
    assert record.quality.sharpness == 0.0
    assert any("резкост" in reason for reason in record.quality.reasons)


def test_gnss_reaches_the_output_when_present_in_exif(tmp_path):
    """Координаты съёмки из EXIF доходят до JSON, а не теряются молча.

    `pipeline/io.py` разбирает GNSS в `CameraMeta.gnss`, но до задачи 15 `ImageRecord`
    такого поля не нёс вовсе, а `Strict` запрещает лишние поля — координаты читались
    и тут же выбрасывались. Снимок без EXIF (весь модуль работает с такими) даёт
    `gnss = None`, а не выдуманный ноль — это проверяет
    `test_pose_is_recorded_together_with_its_origin` и соседние тесты на `baseline`
    заодно, поскольку тот же путь используется для остальных полей записи.
    """
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(900, 1200), dtype=np.uint8)
    path = tmp_path / "gnss.jpg"
    save_image(path, noise)
    blob = piexif.dump({
        "0th": {}, "Exif": {}, "1st": {}, "Interop": {}, "thumbnail": None,
        "GPS": {
            piexif.GPSIFD.GPSLatitude: ((12, 1), (30, 1), (3600, 100)),
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLongitude: ((45, 1), (15, 1), (1800, 100)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSAltitude: (12345, 100),
            piexif.GPSIFD.GPSAltitudeRef: 0,
        },
    })
    piexif.insert(blob, str(path))
    reference = OperatorReference(origin_px=(100.0, 800.0),
                                  span_px=((100.0, 800.0), (1100.0, 800.0)),
                                  span_mm=5000.0)

    model = process(path, operator_reference=reference, raster_mm_per_px=RASTER_MM_PER_PX)

    gnss = model.images[0].gnss
    assert gnss is not None
    assert gnss["lat"] == pytest.approx(12.51)
    assert gnss["lon"] == pytest.approx(45.255)
    assert gnss["alt_m"] == pytest.approx(123.45)


def test_gnss_is_none_without_exif(workdir):
    """Снимок без EXIF не выдумывает координаты: `gnss = None`, а не нуль."""
    data = baseline(workdir)
    assert data.model.images[0].gnss is None


def test_gnss_reaches_the_output_on_the_resolved_plane_path_too(tmp_path):
    """GNSS доходит до JSON и на пути с восстановленной плоскостью, не только на отказном.

    `ImageRecord` собирается в сборке дважды — в `_needs_operator_model` и в основном
    пути `process()`, и оба места прокидывают `meta.gnss` порознь. Проверка на
    отказном пути (`test_gnss_reaches_the_output_when_present_in_exif`) не покрывает
    вторую запись: мутация, вернувшая `gnss=None` только здесь, тем прогоном не
    ловится. Снимок пишется как JPEG (а не PNG, как у `baseline`), чтобы нести EXIF
    и при этом остаться настоящей сценой — плоскость обязана восстановиться.
    """
    scene = make_scene(dist=DISTANCE_MM)
    image = scene.render()
    path = tmp_path / "facade.jpg"
    save_image(path, image)
    blob = piexif.dump({
        "0th": {}, "Exif": {}, "1st": {}, "Interop": {}, "thumbnail": None,
        "GPS": {
            piexif.GPSIFD.GPSLatitude: ((12, 1), (30, 1), (3600, 100)),
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLongitude: ((45, 1), (15, 1), (1800, 100)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSAltitude: (12345, 100),
            piexif.GPSIFD.GPSAltitudeRef: 0,
        },
    })
    piexif.insert(blob, str(path))
    reference = _reference(scene)

    model = process(path, operator_reference=reference, raster_mm_per_px=RASTER_MM_PER_PX)

    record = model.images[0]
    assert record.rectification.needs_operator is False
    gnss = record.gnss
    assert gnss is not None
    assert gnss["lat"] == pytest.approx(12.51)
    assert gnss["lon"] == pytest.approx(45.255)
    assert gnss["alt_m"] == pytest.approx(123.45)


def test_camera_record_rejects_a_non_finite_k(workdir):
    """`_camera_record` проверяет K поэлементно, как и почти все прочие числа сборки.

    Ниже по потоку K уходит в линейную алгебру `camera_pose`, и нечисловой элемент
    упал бы раньше даже без этой проверки — рецензент утечки не нашёл. Но это
    страховка по совпадению работы `camera_pose`, а не по контракту самой сборки, и
    шаг 12 задания требует проверять КАЖДОЕ число, а не полагаться на то, что
    негодное значение будет с чем-то перемножено достаточно рано.
    """
    _, meta = load_image(baseline(workdir).path)
    bad_K = np.array([[3600.0, 0.0, 2640.0], [0.0, float("nan"), 1978.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match=r"camera\.K\[1\]\[1\]"):
        _camera_record(meta, bad_K, "database", [0.0] * 5)


def test_camera_record_rejects_a_non_finite_distortion(workdir):
    """Коэффициенты дисторсии теперь попадают в выход и проверяются так же, как K."""
    _, meta = load_image(baseline(workdir).path)
    with pytest.raises(ValueError, match=r"camera\.dist\[2\]"):
        _camera_record(meta, K, "target", [0.0, 0.0, float("inf"), 0.0, 0.0])


def test_unreachable_raster_scale_is_rejected_by_naming_the_range(workdir):
    """Негодный масштаб растра отвергается с названием достижимых границ."""
    data = baseline(workdir)
    with pytest.raises(ValueError, match="годится mm_per_px от"):
        process(data.path, operator_reference=data.reference,
                raster_mm_per_px=1.0e-6)


def test_scale_sigma_follows_the_length_of_the_reference_base(workdir, tmp_path):
    """σ масштаба считается из длины опорной базы, а не берётся из таблицы.

    База вдесятеро короче даёт σ примерно вдесятеро больше. Точного множителя 10
    здесь быть не может, и это не небрежность допуска: σ пропорциональна ещё и
    ИЗМЕРЕННОМУ локальному разрешению в районе опорных точек, а короткая база лежит
    у левого края фасада, где разрешение иное, чем в среднем по всей двадцатиметровой
    стороне. Измерено: множитель 10.95 при σ 0.183 % на полной базе и 2.00 % на
    укороченной. Промежуток допуска поставлен вокруг измеренного, а не «с запасом»:
    множитель 1 (σ вовсе не зависит от базы) им отвергается, как и множитель,
    отличающийся от измеренного более чем в полтора раза.

    Заодно проверяется, что `meets_tolerance` — результат сравнения, а не константа:
    та же сцена, тот же источник, но короткая база допуска п. 6.3 уже не достигает.
    """
    data = baseline(workdir)
    short = process(data.path,
                    operator_reference=_reference(data.scene, span_fraction=0.1),
                    raster_mm_per_px=RASTER_MM_PER_PX)

    long_sigma = data.model.facade.scale.sigma_rel
    short_sigma = short.facade.scale.sigma_rel
    assert long_sigma > 0.0
    assert 7.0 < short_sigma / long_sigma < 16.0
    assert data.model.facade.scale.meets_tolerance is True
    assert short.facade.scale.meets_tolerance is False


def test_default_sigma_px_agrees_with_the_specification(workdir):
    """На базе во всю двадцатиметровую сторону σ лежит в 0.14–0.28 % (п. 6.3).

    Проверка обязательная: умолчание `sigma_px` выбрано в `run.py` самостоятельно, и
    величина, выводящая σ за диапазон спецификации для источника 1, противоречит ей.
    """
    data = baseline(workdir)
    scale = data.model.facade.scale
    assert scale.source == "operator_reference"
    assert 0.0014 <= scale.sigma_rel <= 0.0028
    assert scale.meets_tolerance is True


def test_operator_points_outside_the_frame_are_refused_by_name(workdir):
    """Точка, указанная за краем кадра, отвергается поимённо, а не подтягивается.

    Без этой охраны индекс ближайшего узла поля разрешения упирался бы в край
    сетки: разрешение «в районе опорной точки» бралось бы из другого места кадра.
    Измерено на умолчании `make_scene` (дистанция 12 м, где двадцатиметровая
    сторона в кадр не помещается): за точку, вышедшую за край на 1267 px, бралось
    наименьшее разрешение по всему кадру, то есть самое выгодное из возможных. Ни
    одна величина при этом не становилась ни бесконечной, ни отрицательной —
    молчаливый успех в чистом виде.
    """
    data = baseline(workdir)
    outside = OperatorReference(
        origin_px=data.reference.origin_px,
        span_px=(data.reference.span_px[0], (float(SIZE[0] + 500), 100.0)),
        span_mm=data.reference.span_mm)
    with pytest.raises(ValueError, match=r"operator_reference\.span_px\[1\].*вне кадра"):
        process(data.path, operator_reference=outside,
                raster_mm_per_px=RASTER_MM_PER_PX)


_FIELDS = {}


def _frame_fields_of(data):
    """Поля кадра на той же сцене. Плоскость оценивается заново: сборка их не хранит.

    Сводки `usable` и `theta_field_deg` в выходной файл попадают не полностью —
    `usable` лишь через причины вердикта, — поэтому проверять их приходится на самих
    полях. Это единственное новое вычисление задачи, и оставить его без прямой
    проверки значило бы поверить доле и квантилю на слово.

    Внутренние параметры берутся тем же путём, что и в конвейере, — через
    `intrinsics_from_meta`, а НЕ истинной матрицей сцены. Разница не умозрительная:
    снимок записан без EXIF, поэтому конвейер берёт K из типового поля зрения и
    получает f = 3663.6 вместо истинных 3600, отчего поле углов сдвигается на треть
    градуса. Помощник, взявший истинную K, проверял бы другую величину, чем та, что
    попала в выходной файл, и расхождение между ними выглядело бы как ошибка сборки.
    """
    if not _FIELDS:
        image, meta = load_image(data.path)
        K_pipeline, _ = intrinsics_from_meta(meta)
        plane = estimate_plane(image, K_pipeline)
        span_rect = _apply(plane.H, np.asarray(data.reference.span_px, dtype=float))
        mm_per_unit = data.reference.span_mm / float(
            np.linalg.norm(span_rect[1] - span_rect[0]))
        origin_rect = tuple(_apply(plane.H, [data.reference.origin_px])[0])
        camera, _ = camera_pose(plane.H, plane.vh, plane.vv, K_pipeline,
                                mm_per_unit, origin_rect)
        _FIELDS["value"] = _frame_fields(
            plane.H, camera, mm_per_unit, origin_rect,
            (image.shape[1], image.shape[0]), THETA_MAX_DEG)
    return _FIELDS["value"]


def test_usable_fraction_needs_both_conditions(workdir):
    """Доля пригодных узлов лежит строго между нулём и единицей на наклонном кадре.

    Порог углов разрезает именно это поле: минимум поля ниже порога, максимум выше,
    значит пригодны часть узлов, а не все и не ни один. Доля, посчитанная по одному
    лишь признаку видимости (за линией схода узлов здесь нет), дала бы ровно 1.0 и
    этой проверкой отвергается.
    """
    fields = _frame_fields_of(baseline(workdir))
    assert fields.gsd.shape == fields.theta_deg.shape
    assert fields.theta_field_deg["min"] < THETA_MAX_DEG < fields.theta_field_deg["max"]
    assert 0.0 < fields.usable < 1.0


def test_p95_of_the_angle_field_is_a_high_quantile(workdir):
    """P95 поля углов — квантиль, а не произвольная сводка.

    Выше P95 обязана лежать примерно двадцатая часть видимых узлов. Ровно 5 % не
    достигается и не должно: `np.percentile` интерполирует между соседними
    отсчётами, и на 4096 узлах выше квантиля оказывается 205 из них, то есть
    5.005 %. Допуск поставлен вокруг двадцатой части, а не «не более 5 %»: медиана
    даёт около 50 % и отвергается, а проверка одним лишь `min < p95 <= max` её
    пропускает.
    """
    fields = _frame_fields_of(baseline(workdir))
    above = float(np.mean(fields.theta_deg[fields.visible]
                          > fields.theta_field_deg["p95"]))
    assert 0.04 <= above <= 0.06


def test_raster_carries_the_origin_of_the_facade(workdir):
    """Растр несёт начало отсчёта фасада, иначе в нём нечего мерить.

    `H` доводит до пикселей растра, но не говорит, где в растре нуль фасада: без
    `origin_rect_px` измеренное в растре положение в миллиметры фасада не
    переводится вовсе. Проверяется, что обе величины взяты от ОДНОЙ точки — от той,
    которую указал оператор, а не от угла растра: пропуск `origin_rect_units` при
    ректификации сдвигает все положения разом и на одну и ту же величину.
    """
    data = baseline(workdir)
    homography = data.model.images[0].homography
    H = np.asarray(homography["H"], dtype=float)
    origin_px = np.asarray(homography["origin_rect_px"], dtype=float)
    mm_per_px = data.model.facade.mm_per_rectified_px

    raster = _apply(H, [data.reference.origin_px])[0]
    assert (raster[0] - origin_px[0]) * mm_per_px == pytest.approx(0.0, abs=1.0)
    assert (raster[1] - origin_px[1]) * mm_per_px == pytest.approx(0.0, abs=1.0)

    # Дальний верхний угол фасада в растре: величина, в задание масштаба не входившая.
    far = _apply(H, data.scene.project(CORNERS[3:4]))[0]
    assert (far[0] - origin_px[0]) * mm_per_px == pytest.approx(0.0, abs=150.0)
    assert -(far[1] - origin_px[1]) * mm_per_px == pytest.approx(15000.0, abs=150.0)


_OK_FRAME = {}


def _ok_frame(directory):
    """Кадр, на котором вердикт обязан быть `ok`, — на РАБОЧЕЙ аппаратуре.

    Камера та же, что у базового кадра: поле зрения 84° по диагонали, внутренние
    параметры берутся из типовой таблицы, профиль калибровки не передаётся. Это
    существенно после снятия порога P95 (спецификация, п. 4.1): пока порог
    действовал, `ok` был недостижим ни при каком ракурсе широкоугольной камеры, и
    проверку положительного пути приходилось ставить на узкий объектив. Теперь
    вердикт `ok` получает та самая аппаратура, ради которой метод и создаётся, и
    проверка перестала быть про исключение.

    От базового кадра отличаются дистанция и ракурс, и оба отличия названы числом.
    Дистанция 12 м вместо 20: разрешение есть `Z / f`, и при f = 3663.7 двадцать
    метров дают 5.46 мм/px ещё до всякого наклона, то есть выше требования п. 2.2
    само по себе (рабочее ограничение дистанции — около 18 м). Ракурс умереннее
    (смещения 1500 и -1000 мм вместо 3000 и -2000): доля пригодной области выходит
    0.625 против требуемых 0.5, тогда как на базовом ракурсе она падает к самому
    порогу. Опорная база — 4 м по фасаду, целиком внутри кадра.
    """
    if not _OK_FRAME:
        scene = make_scene(dx=1500.0, dy=-1000.0, dist=12000.0)
        path = directory / "ok_frame.png"
        save_image(path, scene.render())
        points = scene.project(np.array([[8000.0, 6000.0], [12000.0, 6000.0]]))
        reference = OperatorReference(
            origin_px=(float(points[0][0]), float(points[0][1])),
            span_px=((float(points[0][0]), float(points[0][1])),
                     (float(points[1][0]), float(points[1][1]))),
            span_mm=4000.0)
        _OK_FRAME["value"] = (scene, path, reference)
    return _OK_FRAME["value"]


def test_good_frame_gets_an_ok_verdict(workdir):
    """Положительный путь существует: на исправном кадре вердикт `ok` без причин.

    Без этой проверки вердикт `ok` не достигался бы ни на каком входе, и все
    проверки отбраковки проходили бы при шлюзе, который отбраковывает всё подряд.

    Заодно это охрана от возвращения снятого порога: P95 углового поля на этом кадре
    равен примерно 43°, и порог `P95 <= 30°`, будь он на месте, увёл бы вердикт в
    `degraded`. Утверждается ОБА факта сразу — и что P95 велик, и что вердикт `ok`.
    """
    _, path, reference = _ok_frame(workdir)
    model = process(path, operator_reference=reference, raster_mm_per_px=5.0)

    quality = model.images[0].quality
    assert quality.reasons == []
    assert quality.verdict == "ok"
    assert quality.gsd_mm_px_max <= QUALITY_THRESHOLDS.gsd_max_mm_px
    assert quality.theta_field_deg_p95 > THETA_MAX_DEG
    assert model.images[0].rectification.needs_operator is False


def _frontal_angle_field(focal_px):
    """Поле углов визирования фронтальной камеры: `arctg(r / f)`. Сцены здесь нет.

    Именно в этом суть: при фронтальном ракурсе угол визирования в точке кадра
    зависит только от расстояния пикселя до главной точки и от фокусного, то есть
    от ОБЪЕКТИВА. Считается на той же сетке, на какой сборка считает поле углов.
    """
    gx, gy = _field_grid(SIZE, (64, 64))
    radius = np.hypot(gx - (SIZE[0] - 1) / 2.0, gy - (SIZE[1] - 1) / 2.0)
    return np.degrees(np.arctan(radius / focal_px))


def test_angle_p95_belongs_to_the_lens_and_decides_nothing(workdir):
    """P95 углового поля определяется фокусным, а не сценой, и вердикта не решает.

    Это охрана от возвращения снятого порога по недосмотру, и она держит обе
    половины утверждения.

    Первая — расчётная, без всякой сцены: P95 поля `arctg(r / f)` по кадру 5280x3956
    убывает с фокусным и пересекает 30° между f = 4500 и f = 6000, то есть порог
    `P95 <= 30°` выполним, лишь когда диагональное поле зрения не шире примерно 68°.
    У рабочей камеры оно около 84° (f = 3600), и P95 равен 38.5° при ИДЕАЛЬНО
    фронтальном ракурсе. Доля пригодной области при этом те же 0.63 — она порог
    проходит и остаётся рабочим признаком.

    Вторая — измеренная: на базовом кадре P95 равен 44.5°, и тем не менее среди
    причин вердикта нет ни одной про угол. Вернись порог в шлюз — причин станет две,
    и проверка упадёт.
    """
    p95 = [float(np.percentile(_frontal_angle_field(f), 95))
           for f in (3000.0, 3600.0, 4500.0, 6000.0)]
    assert p95 == sorted(p95, reverse=True)
    assert p95[1] > THETA_MAX_DEG            # 84° по диагонали: порог недостижим
    assert p95[2] > THETA_MAX_DEG            # 72°: всё ещё недостижим
    assert p95[3] < THETA_MAX_DEG            # 58°: достижим — пересечение найдено
    assert float(np.mean(_frontal_angle_field(3600.0) <= THETA_MAX_DEG)) > \
        QUALITY_THRESHOLDS.usable_min_fraction

    data = baseline(workdir)
    quality = data.model.images[0].quality
    assert quality.theta_field_deg_p95 > THETA_MAX_DEG
    assert not any("угол" in reason for reason in quality.reasons)
    assert len(quality.reasons) == 1
    assert "разрешение" in quality.reasons[0]
    assert quality.verdict == "reject"


def test_resolution_is_measured_over_the_usable_area(workdir):
    """Границы разрешения берутся по пригодной области, а не по всему кадру.

    Спецификация, п. 2.2: требование к GSD относится к худшему локальному значению
    в пределах пригодной области. Проверка не вырожденная: на базовом кадре
    ограничение маской реально срезает худшее значение — 6.55 против 8.21 мм/px, —
    поэтому равенство границы максимуму по всем видимым узлам ею отвергается.
    """
    data = baseline(workdir)
    fields = _frame_fields_of(data)
    over_all_visible = float(fields.gsd[fields.visible].max())

    assert fields.gsd_max < over_all_visible
    assert fields.gsd_max == pytest.approx(
        data.model.images[0].quality.gsd_mm_px_max)
    assert fields.usable_mask.sum() < fields.visible.sum()


def test_angle_field_is_not_restricted_to_the_usable_area(workdir):
    """Поле углов остаётся по видимым узлам, иначе сверка P95 становится тавтологией.

    Внутри пригодной области угол по определению не превышает порога, поэтому P95 по
    ней не мог бы порог превысить никогда — величина сообщала бы лишь о том, как её
    посчитали. Здесь утверждается обратное: максимум и P95 поля берутся по всем
    видимым узлам и порог превышают, то есть поле описывает кадр целиком, как того
    требует п. 4.1 от справочной величины.
    """
    data = baseline(workdir)
    fields = _frame_fields_of(data)

    assert fields.theta_deg[fields.usable_mask].max() <= THETA_MAX_DEG
    assert fields.theta_field_deg["max"] > THETA_MAX_DEG
    assert fields.theta_field_deg["p95"] > THETA_MAX_DEG
    assert data.model.images[0].quality.theta_field_deg_p95 == pytest.approx(
        fields.theta_field_deg["p95"])


def test_assumed_floor_height_changes_sigma_and_nothing_else(workdir):
    """Типовая высота этажа — тот же путь оператора, отличающийся только σ.

    Геометрия обязана совпасть до последнего знака: оператор указал те же точки, и
    `span_mm` вошёл в расчёт тем же способом. Различие ровно одно — происхождение
    `span_mm`, отчего σ берётся из таблицы п. 6.3 (~5 %), а не из длины базы. Режим
    остаётся `assisted`: оператор кликал и здесь.
    """
    data = baseline(workdir)
    assumed = process(data.path, operator_reference=data.reference,
                      raster_mm_per_px=RASTER_MM_PER_PX,
                      scale_source="assumed_floor_height")

    assert assumed.facade.scale.source == "assumed_floor_height"
    assert assumed.facade.scale.sigma_rel == 0.05
    assert assumed.facade.scale.meets_tolerance is False
    assert assumed.mode == "assisted"
    assert assumed.images[0].homography["H"] == data.model.images[0].homography["H"]
    assert assumed.images[0].pose_to_facade == data.model.images[0].pose_to_facade
    # Источник 1 на том же кадре допуска достигает — иначе сравнение ни о чём.
    assert data.model.facade.scale.meets_tolerance is True


@pytest.mark.parametrize("source, expected", [
    ("photogrammetry", "раздел 18"),
    ("exif_range", "бортовом дальномере"),
    ("неизвестно", "неизвестный источник масштаба"),
])
def test_unavailable_scale_sources_are_refused_by_name(workdir, source, expected):
    """Источник, которого ядро не получает, отвергается с названной причиной.

    Словарь источников в схеме перечисляет все четыре, и два из них ядру недоступны.
    Молчаливое отсутствие оставило бы потребителя выяснять это по `TypeError`;
    причины подобраны так, чтобы каждая подходила только к своему источнику.
    """
    data = baseline(workdir)
    with pytest.raises(ValueError, match=expected):
        process(data.path, operator_reference=data.reference,
                raster_mm_per_px=RASTER_MM_PER_PX, scale_source=source)


def test_gsd_is_read_at_the_node_nearest_to_the_point():
    """Разрешение берётся у ближайшего узла, а не у первого попавшегося.

    Ожидание выписано вручную: поле 2x3 на кадре 5x3 задано в узлах x = 0, 2, 4 и
    y = 0, 2. Точка (4, 0) обязана попасть в правый верхний узел, (0, 2) — в левый
    нижний, (1.9, 0.9) — в средний верхний. Перестановка строки со столбцом и
    постоянный нулевой индекс этой проверкой отвергаются порознь.
    """
    gsd = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    assert _gsd_at(gsd, (5, 3), (4.0, 0.0)) == 30.0
    assert _gsd_at(gsd, (5, 3), (0.0, 2.0)) == 40.0
    assert _gsd_at(gsd, (5, 3), (1.9, 0.9)) == 20.0


def test_reference_resolution_is_taken_where_the_base_is(workdir):
    """σ зависит от того, ГДЕ указана база, а не только от её длины.

    Две базы одинаковой длины — у левого и у правого края фасада — лежат в местах с
    разным локальным разрешением (6.37 и 5.03 мм/px по краям двадцатиметровой
    стороны), и σ обязана различаться в ту же сторону. Измерено: 2.00 % против
    1.61 %, отношение 1.24. Проверка отвергает разрешение, взятое в постоянной
    точке кадра: там обе σ совпали бы до последнего знака при равной длине базы.
    """
    data = baseline(workdir)

    def sigma(x_from, x_to):
        points = data.scene.project(np.array([[x_from, 0.0], [x_to, 0.0]]))
        reference = OperatorReference(
            origin_px=(float(points[0][0]), float(points[0][1])),
            span_px=((float(points[0][0]), float(points[0][1])),
                     (float(points[1][0]), float(points[1][1]))),
            span_mm=float(x_to - x_from))
        return process(data.path, operator_reference=reference,
                       raster_mm_per_px=RASTER_MM_PER_PX).facade.scale.sigma_rel

    left = sigma(0.0, 2000.0)
    right = sigma(FACADE_W_MM - 2000.0, FACADE_W_MM)
    assert 1.1 < left / right < 1.45


def test_empty_usable_area_is_refused_not_summarised(workdir):
    """Пустая пригодная область — отказ с названной причиной, а не пустая сводка.

    Камера, поставленная в миллиметре от плоскости, видит каждый узел кадра под
    углом, неотличимым от прямого; пригодных узлов не остаётся ни одного. Без
    отказа `min` и `max` по пустой выборке подняли бы невнятную ошибку numpy, а
    доля вышла бы нулевой — цикл по пустой выборке в чистом виде.
    """
    data = baseline(workdir)
    image, meta = load_image(data.path)
    K_pipeline, _ = intrinsics_from_meta(meta)
    plane = estimate_plane(image, K_pipeline)
    span_rect = _apply(plane.H, np.asarray(data.reference.span_px, dtype=float))
    mm_per_unit = data.reference.span_mm / float(
        np.linalg.norm(span_rect[1] - span_rect[0]))
    origin_rect = tuple(_apply(plane.H, [data.reference.origin_px])[0])
    grazing = CameraOnPlane(cx=0.0, cy=0.0, cz=1.0)

    with pytest.raises(ValueError, match="нет ни одного пригодного узла"):
        _frame_fields(plane.H, grazing, mm_per_unit, origin_rect,
                      (image.shape[1], image.shape[0]), THETA_MAX_DEG)


def test_field_grid_covers_the_frame_with_both_ends_included():
    """Сетка поля углов совпадает с сеткой `local_gsd_field` узел в узел.

    Ожидание выписано вручную, а не переписано из кода: сетка 3x2 на кадре 5x3
    обязана дать столбцы 0 и 4 и строки 0, 1, 2. Сдвиг на полшага или исключённый
    правый конец разошёлся бы с полем разрешения, и два поля описывали бы разные
    точки при совпадающих размерностях.
    """
    gx, gy = _field_grid((5, 3), (3, 2))
    assert gx.tolist() == [[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]]
    assert gy.tolist() == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_output_carries_the_profile_distortion(workdir, tmp_path):
    """`camera.dist` в выходе — коэффициенты, которые конвейер действительно снял.

    Прежде `_camera_record` писал `[0.0]*5` при любом профиле: файл утверждал, что
    дисторсии не было, тогда как координаты в нём уже лежат в исправленном кадре, и
    обратная связь с исходным снимком (спецификация, п. 7) была разорвана.
    """
    data = baseline(workdir)
    dist = [-0.01, 0.002, 0.0, 0.0, 0.0]
    profile_path = tmp_path / "profile.json"
    save_profile(CalibrationProfile(model="unknown", K=K.tolist(), dist=dist,
                                    rms_px=0.3, image_size=SIZE), profile_path)
    model = process(data.path, operator_reference=data.reference,
                    raster_mm_per_px=RASTER_MM_PER_PX, profile_path=profile_path)
    camera = model.images[0].camera
    assert camera.calibration == "target"
    assert camera.dist == pytest.approx(dist)


def test_output_without_profile_declares_zero_distortion(workdir):
    """Без профиля дисторсия не снимается, и выход говорит ровно это — нули."""
    data = baseline(workdir)
    camera = data.model.images[0].camera
    assert camera.calibration != "target"
    assert camera.dist == [0.0] * 5


def test_missing_profile_file_is_refused_by_process(workdir, tmp_path):
    data = baseline(workdir)
    with pytest.raises(FileNotFoundError, match="missing_profile"):
        process(data.path, operator_reference=data.reference,
                raster_mm_per_px=RASTER_MM_PER_PX,
                profile_path=tmp_path / "missing_profile.json")


def test_incompatible_profile_is_rejected_per_file_and_the_batch_goes_on(workdir, tmp_path, capsys):
    """Несовпадение профиля со снимком отвергается по этому снимку, а не роняет пакет."""
    data = baseline(workdir)
    profile_path = tmp_path / "profile.json"
    save_profile(CalibrationProfile(model="unknown", K=K.tolist(), dist=[0.0] * 5,
                                    rms_px=0.3, image_size=SIZE), profile_path)

    wrong_size = tmp_path / "wrong_size.png"
    save_image(wrong_size, cv2.resize(data.image, (SIZE[0] // 2, SIZE[1] // 2)))
    out_dir = tmp_path / "out"

    ref = data.reference
    code = main([
        str(wrong_size), str(data.path),
        "--profile", str(profile_path),
        "--raster-mm-per-px", str(RASTER_MM_PER_PX),
        "--origin-px", str(ref.origin_px[0]), str(ref.origin_px[1]),
        "--span-px", str(ref.span_px[0][0]), str(ref.span_px[0][1]),
        str(ref.span_px[1][0]), str(ref.span_px[1][1]),
        "--span-mm", str(ref.span_mm),
        "--out-dir", str(out_dir),
    ])

    captured = capsys.readouterr()
    assert code != 0
    assert "wrong_size.png" in captured.err
    assert "разрешении" in captured.err
    # Второй снимок обработан, несмотря на отказ по первому.
    produced = sorted(p.name for p in out_dir.glob("*.json"))
    assert produced == ["facade.json"]
    model = FacadeModel.model_validate_json((out_dir / "facade.json").read_text("utf-8"))
    assert model.images[0].camera.calibration == "target"


def test_pipeline_fits_the_time_budget(workdir):
    """Бюджет времени: не более 15 с на снимок 20 Мп (спецификация, раздел 16)."""
    data = baseline(workdir)
    assert data.image.shape == (SIZE[1], SIZE[0])
    started = time.perf_counter()
    process(data.path, operator_reference=data.reference,
            raster_mm_per_px=RASTER_MM_PER_PX)
    assert time.perf_counter() - started < 15.0


def test_cli_accepts_an_image_at_an_absolute_cyrillic_path(tmp_path, capsys):
    """CLI принимает снимок по АБСОЛЮТНОМУ пути с кириллицей.

    Каталог этого проекта называется «Оцифровка фото». До починки `load_image`
    такой путь `cv2.imread` не открывал вовсе: файл существовал, но снимок
    считался ненайденным. Тест гоняет настоящий `main()` — точку входа CLI,
    разбирающую argv, — а не только внутренний `process()`, чтобы поймать и
    дефект на уровне разбора путей в самом `main` (`Path(image).stem` и т. п.).
    """
    scene = make_scene(dist=DISTANCE_MM)
    image = scene.render()
    directory = tmp_path / "Оцифровка фото" / "снимки"
    directory.mkdir(parents=True)
    path = directory / "фасад.png"
    save_image(path, image)
    reference = _reference(scene)
    out_dir = tmp_path / "Оцифровка фото" / "выход"

    code = main([
        str(path),
        "--raster-mm-per-px", str(RASTER_MM_PER_PX),
        "--origin-px", str(reference.origin_px[0]), str(reference.origin_px[1]),
        "--span-px", str(reference.span_px[0][0]), str(reference.span_px[0][1]),
                    str(reference.span_px[1][0]), str(reference.span_px[1][1]),
        "--span-mm", str(reference.span_mm),
        "--out-dir", str(out_dir),
    ])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    out_path = out_dir / "фасад.json"
    assert out_path.exists()
    model = FacadeModel.model_validate_json(out_path.read_text("utf-8"))
    assert model.images[0].quality is not None
