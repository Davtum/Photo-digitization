"""Сборка геометрического ядра: снимок -> модель фасада одним проходом, плюс CLI.

Последнее звено ядра. Оно связывает уже проверенные модули и добавляет **ровно одно**
новое вычисление — пригодность кадра (`_frame_fields`), которую ни один модуль не
считает целиком. САМ конвейер элементов фасада не выделяет: детекция окон и дверей —
следующий этап, и делать вид, что он уже есть, здесь нечем. `elements` наполняется
только разметкой оператора (`marks`, задача 18), и тогда каждый элемент несёт
`origin = "operator"`.

`coverage` следует из начала координат (п. 7): `"full"`, когда оператор указал
началом левый нижний угол фасада (`OperatorReference.origin_is_facade_corner`), и
`"partial"`, когда начало — его произвольная точка (план 3, задача 11).

**Двух «мм на пиксель» здесь действительно два, и смешивать их нельзя.**

* `mm_per_unit` — масштаб ректифицированной системы координат, он же расстояние от
  камеры до плоскости. Из кадра невыводим: снимок плоскости знает её с точностью до
  подобия. Его задаёт оператор через `OperatorReference`.
* `raster_mm_per_px` — разрешение выходного растра, свободный параметр визуализации,
  обязанный лежать внутри `attainable_mm_per_px`.

Ни один из них не есть GSD. GSD — измеренное поле, меняющееся по кадру, и берётся оно
из `local_gsd_field`. Подстановка одного аргумента во все три роли — дефект, ради
устранения которого задача переписывалась.

**Почему `mode` всегда `"assisted"`.** Оба рабочих источника масштаба — и измеренный
оператором размер, и типовое допущение о высоте этажа — берут одни и те же указания
оператора: те же две точки на снимке и ту же точку начала отсчёта. Различие между ними
не геометрическое, а в происхождении `span_mm` и, как следствие, в σ. Оператор в
проходе участвовал в обоих случаях, поэтому режим `"assisted"`, и ветки `"auto"` здесь
нет: она потребовала бы источника масштаба, не нуждающегося в операторе, а таких
геометрическое ядро не получает (см. `UNAVAILABLE_SCALE_SOURCES`).
"""
import argparse
import dataclasses
import math
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from facade_digitizer.geometry.angles import FIELD_SHAPE, local_gsd_field, nearest_node
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import (
    HALF_TURN_FROM_OPERATOR_ORDER,
    apply_homography,
    camera_pose,
    pose_from_homography,
    rectified_to_facade_mm,
)
from facade_digitizer.pipeline import quality as quality_gate
from facade_digitizer.pipeline.elements import (
    WALL_FLATNESS_DEVIATION_MM,
    ElementMark,
    digitize_elements,
    load_marks,
)
from facade_digitizer.pipeline.frame import Frame, load_frame
from facade_digitizer.pipeline.io import save_image
from facade_digitizer.pipeline.plane import (
    ManualPlane,
    estimate_plane,
    estimate_plane_manual,
)
from facade_digitizer.pipeline.rectify import (
    RasterGeometry,
    Rectified,
    attainable_mm_per_px,
    raster_geometry,
    warp,
)
from facade_digitizer.schema import (
    CameraIntrinsics,
    FacadeModel,
    FacadeRecord,
    ImageRecord,
    QualityReport,
    Rectification,
    ScaleEstimate,
)

try:
    SOFTWARE_VERSION = f"facade-digitizer {version('facade-digitizer')}"
except PackageNotFoundError:          # пакет не установлен — версия неизвестна
    SOFTWARE_VERSION = "facade-digitizer (не установлен)"

# `FIELD_SHAPE` — сетка узлов КАДРА, общая для поля разрешения и поля углов. Общая
# обязательно: два поля, посчитанные на разных сетках, описывают разные точки, и
# доля пригодных узлов считалась бы по одной выборке, а признак «за линией схода» —
# по другой. Значение ИМПОРТИРУЕТСЯ у `geometry.angles`, где объявлено один раз и
# служит умолчанием самой `local_gsd_field`. Своего объявления здесь больше нет:
# второе такое же было у `pipeline.elements`, которая считала поле заново, и
# расхождение двух копий развело бы σ элементов и `quality.gsd_mm_px_max` одного
# файла молча.

#: Верхняя граница σ масштаба для источника 1 (спецификация, п. 6.3: 0.14–0.28 %).
#: С ней сравнивается ПОСЧИТАННАЯ σ — `meets_tolerance` есть результат сравнения,
#: а не переписанный из таблицы ответ.
SIGMA_REL_TOLERANCE = 0.0028

#: Точность указания точки оператором, в пикселях ИСХОДНОГО снимка.
#:
#: Величина выбрана здесь, а не взята из таблицы, и вот из чего. Оператор указывает
#: кромку не на снимке, а на экране: кадр 20 Мп (5280x3956) в окне шириной 1920 px
#: показан с уменьшением в 2.75 раза, поэтому один экранный пиксель стоит 2.75
#: пикселей снимка. Попадание по кромке на экране — порядка полутора пикселей
#: (положение курсора плюс размытие самой кромки), то есть около 4 пикселей снимка.
#:
#: Согласованность с п. 6.3 ПРОВЕРЕНА, а не постулирована. Диапазон 0.14–0.28 %
#: спецификации относится к базе во всю двадцатиметровую сторону фасада, и такая
#: база помещается в кадр 5280x3956 при f = 3600 px лишь с дистанции около 14 м;
#: рабочие дистанции — от 20 до 25 м.
#:
#: **Протокол замера**, чтобы следующая проверка воспроизвела те же числа, а не
#: третьи. Сцена — `tests/test_synth.make_scene` с умолчательными смещениями
#: (3000 и -2000 мм), опорная база — вся нижняя сторона фасада (20000 мм).
#: Снимок пишется на диск через `save_image` и потому без EXIF, поэтому K — НЕ
#: истинная K сцены (f = 3600), а та, что даёт сама сборка в этом случае: типовое
#: поле зрения 84° по диагонали (f ≈ 3663.6 px), путь `intrinsics_from_meta` при
#: отсутствующем EXIF. Величина измерена самим проходом `process()`, тем же путём,
#: каким её получает конвейер, а не отдельной формулой:
#:
#:     дистанция 20 м   среднеквадратичное разрешение при базе 5.74 мм/px   σ = 0.183 %
#:     дистанция 25 м   то же                             7.07 мм/px        σ = 0.225 %
#:
#: Обе величины внутри диапазона; крайние значения `sigma_px`, при которых это ещё
#: так на обеих дистанциях, — 3.45 и 5.60, и 4.5 лежит между ними, а не у края.
#: Проверка закреплена в
#: `tests/test_run.py::test_default_sigma_px_agrees_with_the_specification`.
#:
#: **Эта величина относится ТОЛЬКО к опорной базе масштаба.** Точность клика по
#: углу проёма — отдельная величина, `DEFAULT_MARK_SIGMA_PX`; см. её обоснование.
DEFAULT_OPERATOR_SIGMA_PX = 4.5

#: Точность указания УГЛА ПРОЁМА оператором, в пикселях исходного снимка.
#:
#: Это не та же величина, что `DEFAULT_OPERATOR_SIGMA_PX`, и слияние их в одну —
#: подмена. Разведены они по РЕЖИМУ ПРОСМОТРА, в котором оператор делает клик.
#:
#: * Концы опорной базы оператор указывает на кадре целиком: база тянется во всю
#:   двадцатиметровую сторону фасада, и увидеть её концы одновременно можно только
#:   при уменьшении 2.75x (кадр 5280 px в окне 1920 px). Один экранный пиксель
#:   стоит там 2.75 пикселей снимка, отсюда 4.5.
#: * Угол проёма оператор указывает с УВЕЛИЧЕНИЕМ, и иначе быть не может. При том
#:   же уменьшении 2.75x полутораметровый проём при локальном GSD 5 мм/px занимает
#:   на экране около 106 px, попадание по кромке стоит те же ~1.5 экранных пикселя,
#:   то есть ~4 пикселя снимка, а одно слагаемое локализации п. 6.1 даёт тогда
#:   √2·4.5·GSD: 31.8 мм при GSD 5 и 12.7 мм при GSD 2 — при допуске п. 2.2 в
#:   10 мм. То есть на уменьшенном экране допуск недостижим ни при каком угле, и
#:   инструмент разметки, показывающий проём так, непригоден по построению
#:   (п. 8.1: разметчик работает в инструменте с увеличением).
#:
#: **Значение.** При увеличении 1:1 и выше один экранный пиксель стоит не более
#: одного пикселя снимка, и точность клика упирается в саму кромку, а не в
#: масштаб показа. Спецификация называет для этого случая свою строку: п. 6.1,
#: режим `assisted`, σ_px 0.5–1, где 1.0 — НИЖНЯЯ граница качества режима, то есть
#: худший случай режима, в котором допуск ещё заявлен достижимым. Берётся она, а
#: не 0.5: субпиксельное уточнение кромки (п. 6.5) геометрическим ядром не
#: выполняется, и заявлять его точность было бы нечем. То же значение служит
#: опорным для порога глубины (`parallax.REFERENCE_SIGMA_PX`), и совпадение здесь
#: не случайно — обе величины суть точность локализации одной и той же кромки.
#:
#: Перекрывается ключом CLI `--mark-sigma-px`: инструмент разметки с
#: субпиксельным уточнением вправе заявить 0.5, а разметка по мелкому растру —
#: 2 и выше, и тогда σ вырастет сама.
DEFAULT_MARK_SIGMA_PX = 1.0

#: Что шлюз качества ДЕЛАЕТ, вынеся вердикт `reject`. Спецификация, п. 4.1
#: называет модуль «шлюзом» и «отбраковкой», и до сих пор он не был ни тем, ни
#: другим: `process()` считала вердикт, клала его в файл и на него не смотрела,
#: CLI печатала его и возвращала 0. На наборе CMP вердикт `ok` не получил ни один
#: снимок из 378 — и все 378 были обработаны так же, как получившие бы его.
#:
#: **Решение: шлюз не отбраковывает снимок, а снимает метрическую интерпретацию
#: с его элементов.** `meets_tolerance` — единственное поле выхода, которое
#: УТВЕРЖДАЕТ соответствие требованиям п. 2.2; вердикт `reject` ставится ровно
#: тогда, когда кадр требованию п. 2.2 к разрешению НЕ удовлетворяет (худшее
#: локальное значение по пригодной области выше `gsd_max_mm_px`). Выпускать в
#: одном файле «кадр не годен по п. 2.2» и «габарит в допуске п. 2.2» — то же
#: самое противоречие, что уже устранено для `edge_reference != "wall_plane"`, и
#: устраняется оно тем же способом: геометрия выдаётся, утверждение о ней — нет.
#: Отсутствие, а не `False`: «на таком кадре не проверяем» и «проверили и не
#: прошло» — разные утверждения.
#:
#: Почему НЕ отказ по снимку. `process()` отказывает исключением, а исключение не
#: пишет выходного файла вовсе: оператор потерял бы `quality.reasons` — ровно то
#: единственное, что называет, ЧТО с кадром не так. Сам модуль шлюза объявляет о
#: себе «отбраковывает, но не улучшает»: он помечает, а не уничтожает улику.
#: К тому же вердикт сегодня определяется порогом GSD, а GSD считается от
#: масштаба, заданного оператором, и на пробе CMP этот масштаб ВЫДУМАН
#: (`scripts/smoke_cmp.py`); отказ отдал бы выдуманной величине право решать,
#: будет ли у снимка выход вообще.
#:
#: Почему НЕ «не ветвиться и убрать слово «шлюз»». Тогда `quality.verdict`
#: остался бы величиной, которую конвейер вычисляет, записывает и ни на что не
#: употребляет, — тем самым классом дефекта, который этот проект ловит из задачи
#: в задачу.
GATE_REJECT_WITHHOLDS_TOLERANCE = (
    "вердикт reject: признак соответствия допуску (meets_tolerance) у элементов "
    "не выпускается — кадр не удовлетворяет требованию п. 2.2 к разрешению, и "
    "утверждать по нему соответствие тому же п. 2.2 нечем")

#: σ масштаба для источника 4 п. 6.3 — типового допущения о высоте этажа (~5 %).
#: Геометрия пути та же, что у источника 1: оператор указывает те же две точки и то же
#: начало отсчёта. Отличается ОДНО — `span_mm` не измерен, а принят по типовому
#: значению, поэтому σ из длины базы не выводится и берётся из таблицы. Допуска эта
#: величина не достигает, и это видно из самого сравнения, а не из отдельной записи
#: «нет» рядом с ней: 0.05 больше SIGMA_REL_TOLERANCE на порядок с лишним.
SIGMA_REL_ASSUMED_FLOOR_HEIGHT = 0.05

#: Источники масштаба, которых геометрическое ядро не получает, и причина по каждому.
#: Отказ здесь громкий и названный, а не молчаливое отсутствие: словарь источников в
#: схеме перечисляет все четыре, и потребитель вправе узнать, почему два из них не
#: работают, а не обнаружить это по `TypeError` где-то внутри.
UNAVAILABLE_SCALE_SOURCES = {
    "photogrammetry": (
        "масштаб облёта приходит из внешнего уравнивания снимков (спецификация, "
        "раздел 18), а геометрическое ядро работает по одиночному снимку и такого "
        "входа не имеет"),
    "exif_range": (
        "источник применим только при бортовом дальномере либо известном положении "
        "здания (спецификация, п. 6.3); телеметрия EXIF расстояния до фасада по "
        "нормали не содержит, а вывод дальности из позы сам требует масштаба"),
}

#: Источник матрицы K в терминах схемы. `intrinsics_from_meta` уточняет путь через
#: EXIF ("exif:crop_factor" и подобные), схема же различает лишь три происхождения.
#: Уточнение отбрасывается ЗДЕСЬ и осознанно, а не теряется по дороге.
_CALIBRATION_BY_SOURCE = {"target": "target", "database": "database"}


@dataclass(frozen=True)
class OperatorReference:
    """Указания оператора, без которых масштаб не определён.

    `origin_px` — точка кадра, которая станет началом координат фасада.
    `span_px` — две точки кадра, между которыми оператор знает истинное расстояние.
    `span_mm` — это расстояние в миллиметрах.
    `sigma_px` — точность указания точки; см. `DEFAULT_OPERATOR_SIGMA_PX`.
    `end_sigma_px` — точность указания КАЖДОГО конца базы порознь (план 3,
    задача 6); `None` — оба конца с `sigma_px`. Концы указываются при разном
    масштабе просмотра, и одна σ на оба приписала бы точному концу промах грубого.
    """

    origin_px: tuple
    span_px: tuple
    span_mm: float
    sigma_px: float = DEFAULT_OPERATOR_SIGMA_PX
    end_sigma_px: tuple | None = None
    #: Погрешность САМОЙ длины базы в мм — рулетки, дальномера (план 3, задача 11).
    #: `None` — не передана: в σ масштаба входит только промах указания концов.
    span_sigma_mm: float | None = None
    #: Начало отсчёта — левый нижний угол фасада, а не произвольная точка (п. 7):
    #: тогда `facade.origin = "bottom_left"` и `coverage = "full"`.
    origin_is_facade_corner: bool = False

    @property
    def end_sigmas(self) -> tuple[float, float]:
        if self.end_sigma_px is None:
            return (self.sigma_px, self.sigma_px)
        first, second = self.end_sigma_px
        return (first, second)


def _finite(name: str, value) -> float:
    """Конечное число либо отказ, называющий поле.

    Бесконечность, просочившаяся в выход, ломает обратный разбор JSON, то есть
    обнаруживается у потребителя, а не здесь. Тип проверяется до значения: и
    `math.isfinite` на строке, и сравнение с `None` дали бы сообщение про
    внутренности сборки вместо негодной величины.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: ожидалось вещественное число, получено {value!r}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name}: значение неопределено ({v})")
    return v


def _finite_or_none(value):
    """Число, если оно конечно, иначе отсутствие. Для полей, где схема допускает None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _apply(H, points):
    """Точки кадра в ректифицированные единицы. Тонкая обёртка над `apply_homography`.

    Реализация переехала в `geometry.homography` (задача 18), где та же функция
    нужна и `pipeline.elements`: разметка проёмов переводится в миллиметры фасада
    ТОЙ ЖЕ гомографией, что и поля кадра здесь, а вторая независимая реализация
    деления могла разойтись бы с этой незаметно. Имя и сигнатура сохранены —
    от них зависят существующие тесты этого модуля.
    """
    return apply_homography(H, points)


def _field_grid(image_size, shape):
    """Узлы сетки КАДРА. Обязана совпадать с сеткой `local_gsd_field` узел в узел.

    Повторена здесь потому, что `local_gsd_field` сетку наружу не отдаёт, а поле
    углов считается в тех же точках. Совпадение закреплено проверкой в тестах:
    сетка покрывает [0, w-1] x [0, h-1] с включёнными концами.
    """
    w_px, h_px = image_size
    rows, cols = shape
    return np.meshgrid(np.linspace(0, w_px - 1, cols), np.linspace(0, h_px - 1, rows))


def _gsd_at(gsd: np.ndarray, image_size, point_px) -> float:
    """Измеренное локальное разрешение в районе точки кадра.

    Берётся значение ближайшего узла того же поля, а не пересчитывается заново:
    вторая реализация якобиана разошлась бы с первой незаметно. Узел выбирает
    `geometry.angles.nearest_node` — та же функция, какой его выбирает
    `pipeline.elements._gsd_at_point`; прежде формула была выписана в обоих
    модулях порознь. Узел за линией схода помечен в поле как `nan` и здесь
    возвращается как есть — вызывающий обязан отказать, а не усреднить.

    Точка обязана лежать В КАДРЕ, и это проверено вызывающим (`_points_in_frame`).
    Без такой проверки индекс ближайшего узла упирался бы в край сетки, и разрешение
    «в районе опорной точки» бралось бы из совсем другого места кадра. Измерено: на
    точке, вышедшей за правый край кадра на 1267 px, так бралось НАИМЕНЬШЕЕ
    разрешение по всему кадру (2.73 мм/px при разбросе поля 2.73…6.87), то есть
    самое выгодное из возможных; при этом ни одна величина не становилась ни
    бесконечной, ни отрицательной, и отказа не наступало.
    """
    rows, cols = gsd.shape
    w_px, h_px = image_size
    col = nearest_node(float(point_px[0]), w_px, cols)
    row = nearest_node(float(point_px[1]), h_px, rows)
    return float(gsd[row, col])


def _points_in_frame(reference: "OperatorReference", image_size) -> None:
    """Указанные оператором точки обязаны лежать в кадре.

    Оператор указывает точку НА СНИМКЕ; точка вне кадра означает, что переданы
    координаты из другой системы либо от другого снимка. Отказ поимённый: молча
    подтянуть такую точку к краю значит взять разрешение и начало отсчёта из места,
    на которое оператор не показывал.
    """
    w_px, h_px = image_size
    named = (("origin_px", reference.origin_px),
             ("span_px[0]", reference.span_px[0]),
             ("span_px[1]", reference.span_px[1]))
    for name, point in named:
        x = _finite(f"operator_reference.{name}.x", point[0])
        y = _finite(f"operator_reference.{name}.y", point[1])
        if not (0.0 <= x <= w_px - 1 and 0.0 <= y <= h_px - 1):
            raise ValueError(
                f"operator_reference.{name} = ({x:.1f}, {y:.1f}) лежит вне кадра "
                f"{w_px}x{h_px}: оператор указывает точки на снимке")


def _scale_sigma_rel(span_mm: float, end_sigmas, end_gsds,
                     span_sigma_mm: float | None = None) -> float:
    """σ масштаба из ДЛИНЫ ОПОРНОЙ БАЗЫ и измеренного разрешения при ней.

        sigma_rel = sqrt((σ1·g1)² + (σ2·g2)²) / span_mm

    Оператор указывает ДВЕ точки, погрешности их указания независимы, и в
    миллиметрах каждая равна своей σ на своё локальное разрешение. Прежде σ была
    одна на оба конца и разрешение бралось среднеквадратичным по концам —
    `sqrt(2)·σ·g_ск`, что есть тот же ответ ровно при равных σ (план 3, задача 6:
    концы указываются при разном увеличении).

    `span_sigma_mm` — погрешность самой измеренной длины (план 3, задача 11):
    независима от промаха указания и складывается с ним квадратично. Прежде она не
    учитывалась вовсе, и σ масштаба при рулетке с погрешностью 5 мм на базе 3 м
    занижалась ровно на это слагаемое.

    Оговорка задачи 14 о том, что реальный оператор даст базу вдесятеро короче
    двадцатиметровой стороны фасада, перестаёт быть текстом в спецификации и
    становится числом в выходном файле: короткая база сама поднимает σ.
    """
    span_mm = _finite("operator_reference.span_mm", span_mm)
    sigmas = [_finite("operator_reference.sigma_px", v) for v in end_sigmas]
    if span_mm <= 0 or min(sigmas) <= 0:
        raise ValueError("опорная база и точность указания точки должны быть положительны: "
                         f"span_mm={span_mm}, sigma_px={sigmas}")
    gsds = [_finite("gsd_at_reference", g) for g in end_gsds]
    click = math.hypot(sigmas[0] * gsds[0], sigmas[1] * gsds[1]) / span_mm
    if span_sigma_mm is None:
        return click
    span_sigma_mm = _finite("operator_reference.span_sigma_mm", span_sigma_mm)
    if span_sigma_mm < 0:
        raise ValueError(f"погрешность длины базы не может быть отрицательной: {span_sigma_mm}")
    return math.hypot(click, span_sigma_mm / span_mm)


def _checked_scale_source(scale_source: str) -> str:
    """Источник масштаба, который ядро способно обеспечить, либо отказ с причиной."""
    unavailable = UNAVAILABLE_SCALE_SOURCES.get(scale_source)
    if unavailable is not None:
        raise ValueError(f"источник масштаба «{scale_source}» геометрическому ядру "
                         f"недоступен: {unavailable}")
    if scale_source not in ("operator_reference", "assumed_floor_height"):
        raise ValueError(f"неизвестный источник масштаба: {scale_source!r}; "
                         "спецификация, п. 6.3 перечисляет четыре")
    return scale_source


def _origin(reference: OperatorReference) -> str:
    """Начало отсчёта (п. 7): угол фасада, если оператор его указал, иначе его точка."""
    return "bottom_left" if reference.origin_is_facade_corner else "operator_reference"


def _coverage(reference: OperatorReference) -> str:
    """Охват (п. 7): полный — ровно когда начало в левом нижнем углу фасада.

    П. 7 сопоставляет началу в опорной точке оператора частичный охват («если угол
    не в кадре»), а началу в угле фасада — полный. Пара согласована: одно без
    другого выпустить нельзя (план 3, задача 11).
    """
    return "full" if reference.origin_is_facade_corner else "partial"


def _span_sigma(reference: OperatorReference, scale_source: str) -> float | None:
    """Погрешность длины базы — в выход, только когда длина измерена (источник 1)."""
    return reference.span_sigma_mm if scale_source == "operator_reference" else None


def _scale_estimate(source: str, sigma_rel: float,
                    meets_tolerance: bool | None = None,
                    span_sigma_mm: float | None = None) -> ScaleEstimate:
    """Оценка масштаба с происхождением. `meets_tolerance` — сравнение, а не константа.

    Сравнение одно на все источники, и таблица п. 6.3 воспроизводится им сама:
    измеренная оператором база даёт σ порядка 0.18 % и допуска достигает, типовое
    допущение о высоте этажа — 5 % и не достигает. Записывать ответ «нет» отдельной
    константой значило бы держать в коде два источника истины об одном и том же.
    """
    sigma_rel = _finite("facade.scale.sigma_rel", sigma_rel)
    if meets_tolerance is None:
        meets_tolerance = sigma_rel <= SIGMA_REL_TOLERANCE
    return ScaleEstimate(source=source, sigma_rel=sigma_rel,
                         meets_tolerance=bool(meets_tolerance),
                         span_sigma_mm=(None if span_sigma_mm is None
                                        else _finite("facade.scale.span_sigma_mm",
                                                     span_sigma_mm)))


@dataclass(frozen=True)
class FrameFields:
    """Поля, посчитанные в узлах КАДРА, и выведенная из них пригодность кадра.

    Поля `gsd`, `theta_deg` и `visible` заданы на ОДНОЙ сетке и возвращаются
    наружу целиком, а не только своими сводками. Иначе `usable` и `theta_field_deg`
    нечем проверить: доля и квантиль по выборке, которую никто не видит, — это
    ровно та величина, которая соглашается с любым утверждением о себе.
    """

    gsd: np.ndarray
    theta_deg: np.ndarray
    visible: np.ndarray
    usable_mask: np.ndarray
    behind_vanishing_line: float
    usable: float
    theta_field_deg: dict
    gsd_min: float
    gsd_max: float


def _frame_fields(H, camera, mm_per_unit, origin_rect, image_size, theta_max_deg):
    """Пригодность кадра: поле углов, поле разрешения и доля пригодных узлов.

    Единственное новое вычисление сборки. Считается на сетке узлов **кадра**, а не
    фасада, потому что доля нормируется по кадру: `angle_map` и `usable_mask` заданы
    на габарите фасада, часть которого в кадр не попала, и доля по ним отвечала бы на
    другой вопрос.

    Узел пригоден, когда выполнены ОБА условия: угол визирования не превышает порог
    И узел лежит перед камерой, а не за линией схода. Второй признак берётся из
    `local_gsd_field`, где такие узлы помечены `nan`; сетка у обоих полей одна.
    Охрана знака знаменателя из `local_gsd_field` на рабочих ракурсах не
    срабатывает — из сборки путь за линию схода недостижим, потому что оценка
    плоскости отказывает раньше, — и мёртвой от этого не становится: без неё узлы,
    камерой не видимые, входили бы в границы разрешения наравне с видимыми, а
    верхняя грань по всему кадру на таких ракурсах бесконечна (спецификация, п. 2.2).
    Испытана она своими тестами, `tests/test_angles.py`.

    **Границы разрешения берутся по ПРИГОДНОЙ ОБЛАСТИ, а не по всем видимым узлам.**
    Спецификация, п. 2.2: «Требование к GSD относится к худшему локальному значению
    в пределах пригодной области, а не по всему кадру». Разница не формальная: на
    кадре 20 Мп с полем зрения 84° углы кадра отстоят от опорной точки на 40° с
    лишним, и худшее разрешение по всему кадру описывает те места, где измерять
    заведомо нельзя.

    **`theta_field_deg` на ту же маску НЕ переносится, и это существенно.** Маска
    задана условием `theta <= theta_max_deg`, поэтому P95 углов внутри неё не может
    превысить порога никогда — вышла бы справочная величина, сообщающая лишь о том,
    как её посчитали. Поле углов остаётся по ВИДИМЫМ узлам той же сетки и описывает
    кадр целиком; по пригодной области берётся только разрешение.
    """
    field = local_gsd_field(H, mm_per_unit, image_size, shape=FIELD_SHAPE)
    gx, gy = _field_grid(image_size, FIELD_SHAPE)
    facade_mm = rectified_to_facade_mm(
        _apply(H, np.column_stack([gx.ravel(), gy.ravel()])), origin_rect, mm_per_unit)

    theta = np.array([camera.theta_deg(float(x), float(y)) for x, y in facade_mm])
    theta = theta.reshape(FIELD_SHAPE)

    visible = ~np.isnan(field.gsd)
    if not visible.any():
        raise ValueError("ни один узел кадра не смотрит на плоскость фасада: "
                         "линия схода прошла по всему кадру")
    visible_theta = theta[visible]
    if not np.all(np.isfinite(visible_theta)):
        raise ValueError("поле углов визирования неопределено в видимой части кадра")

    usable_mask = visible & (theta <= theta_max_deg)
    if not usable_mask.any():
        raise ValueError(
            "в кадре нет ни одного пригодного узла: угол визирования всюду превышает "
            f"{theta_max_deg}° либо узлы лежат за линией схода, и худшее разрешение "
            "по пригодной области не определено")

    theta_field = {
        "min": _finite("theta_field_deg.min", visible_theta.min()),
        "max": _finite("theta_field_deg.max", visible_theta.max()),
        "p95": _finite("theta_field_deg.p95", np.percentile(visible_theta, 95)),
    }
    # `nan` в выборке невозможен по построению: пригодный узел прежде всего видим,
    # а нечисловым помечен ровно невидимый. Поэтому берётся обычный min и max —
    # `nanmin` здесь был бы охраной от того, чего маска уже не пропускает.
    gsd_usable = field.gsd[usable_mask]
    return FrameFields(
        gsd=field.gsd, theta_deg=theta, visible=visible, usable_mask=usable_mask,
        behind_vanishing_line=field.behind_vanishing_line,
        usable=float(usable_mask.mean()),
        theta_field_deg=theta_field,
        gsd_min=_finite("quality.gsd_mm_px_min", gsd_usable.min()),
        gsd_max=_finite("quality.gsd_mm_px_max", gsd_usable.max()))


def _theta_at(H, camera, mm_per_unit, origin_rect, point_px) -> float:
    """Угол визирования в точке кадра. Тем же путём, каким считается поле."""
    facade_mm = rectified_to_facade_mm(_apply(H, [point_px]), origin_rect, mm_per_unit)[0]
    return _finite("theta_cam_deg", camera.theta_deg(float(facade_mm[0]),
                                                     float(facade_mm[1])))


def _camera_record(meta, K, source, dist) -> CameraIntrinsics:
    """Внутренние параметры с происхождением. Источник обязан дойти до выхода.

    `K` проверяется поэлементно на конечность, как и практически все прочие числа
    сборки (шаг 12 задания). Ниже по потоку та же матрица уходит в линейную алгебру
    `camera_pose`, где нечисловой элемент упал бы раньше и без этой проверки —
    но это страховка по совпадению, а не по контракту: K читается из EXIF или из
    файла профиля калибровки, и обе цепочки не гарантируют конечность на входе сюда.

    `dist` — коэффициенты, которые конвейер ДЕЙСТВИТЕЛЬНО снял со снимка (`undistort`
    в `process`), а не константа. Прежде здесь стояло `[0.0]*5` при любом профиле:
    файл утверждал, что дисторсии не было, тогда как координаты в нём уже лежат в
    исправленном кадре, и обратная связь с исходным снимком (п. 7) была разорвана.
    """
    calibration = _CALIBRATION_BY_SOURCE.get(source)
    if calibration is None:
        if not source.startswith("exif"):
            raise ValueError(f"неизвестное происхождение внутренних параметров: {source}")
        calibration = "exif"
    K_checked = [[_finite(f"camera.K[{row}][{col}]", v) for col, v in enumerate(values)]
                for row, values in enumerate(K)]
    dist_checked = [_finite(f"camera.dist[{i}]", v) for i, v in enumerate(dist)]
    return CameraIntrinsics(model=meta.model, K=K_checked,
                            dist=dist_checked, calibration=calibration)


def _needs_operator_model(path, image_id, meta, K, dist, source, plane, reference,
                          sharpness_raw,
                          raster_mm_per_px, scale_source) -> FacadeModel:
    """Модель отказа: плоскость не восстановлена, управление у оператора.

    Ректификации здесь нет вовсе, и это контракт п. 4.2, а не экономия: единичная
    гомография вместо отказа неотличима от настоящей на стороне потребителя.

    Что при этом можно записать честно, а что нельзя.

    * Разрешение — можно: опорная база оператора сама есть измерение, `span_mm` на
      её длину в пикселях кадра даёт среднее разрешение вдоль базы. Поле разрешения
      без гомографии не строится, поэтому границы совпадают: разброс НЕ измерен, и
      равенство границ об этом и говорит.
    * Угол визирования — нельзя: он требует позы. Записано предельное значение 90°,
      названное в причинах. Вердикта эта величина не решает (спецификация, п. 4.1),
      но до потребителя она доходит, и тот вправе приложить к ней свой порог: нуль
      или правдоподобное число прошли бы такую проверку молча, 90° не проходит ни
      одной.
    * `meets_tolerance` — `False` независимо от σ: плоскость не восстановлена, и
      допуск не достигнут ничем.
    """
    span = np.asarray(reference.span_px, dtype=float)
    span_px_len = float(np.linalg.norm(span[1] - span[0]))
    if span_px_len <= 0:
        raise ValueError("опорные точки оператора совпали: база нулевой длины")
    gsd_along_base = _finite("quality.gsd_mm_px_min", reference.span_mm / span_px_len)

    reasons = list(plane.confidence.reasons)
    reasons.append("разрешение оценено по опорной базе оператора: поле разрешения "
                   "без гомографии не строится, разброс не измерен")
    reasons.append("угол визирования не измерен: поза требует плоскости; записано "
                   "предельное значение 90°")

    # Резкость посчитана фазой кадра (план 3, задача 4): от опорной базы она не
    # зависит, и считать её второй раз — до 0.7 с на снимке 20 Мп без пользы.
    sharpness = _finite("quality.sharpness", sharpness_raw)
    if sharpness < quality_gate.DEFAULT.sharpness_min:
        reasons.append(f"недостаточная резкость: {sharpness:.2e} < "
                       f"{quality_gate.DEFAULT.sharpness_min:.2e}")
        verdict = "reject"
    else:
        verdict = "degraded"

    record = ImageRecord(
        id=image_id,
        path=str(path),
        camera=_camera_record(meta, K, source, dist),
        captured_at=meta.captured_at,
        gnss=meta.gnss,
        pose_to_facade=None,
        theta_cam_deg=None,
        theta_field_deg=None,
        quality=QualityReport(verdict=verdict, sharpness=sharpness,
                              gsd_mm_px_min=gsd_along_base,
                              gsd_mm_px_max=gsd_along_base,
                              theta_field_deg_p95=90.0, reasons=reasons),
        rectification=Rectification(method=plane.method,
                                    confidence=_finite("rectification.confidence",
                                                       plane.confidence.value),
                                    residual_px=_finite_or_none(plane.confidence.residual_px),
                                    needs_operator=True),
        homography=None,
    )
    sigma_rel = (_scale_sigma_rel(reference.span_mm, reference.end_sigmas,
                                  (gsd_along_base, gsd_along_base), reference.span_sigma_mm)
                 if scale_source == "operator_reference"
                 else SIGMA_REL_ASSUMED_FLOOR_HEIGHT)
    return FacadeModel(
        software_version=SOFTWARE_VERSION,
        coverage=_coverage(reference),
        mode="assisted",
        images=[record],
        facade=FacadeRecord(
            origin=_origin(reference),
            # Габарит охваченной части фасада без гомографии не определён. Пустой
            # список — отсутствие; четыре нуля были бы правдоподобным габаритом.
            bounds_mm=[],
            mm_per_rectified_px=_finite("facade.mm_per_rectified_px", raster_mm_per_px),
            scale=_scale_estimate(scale_source, sigma_rel, meets_tolerance=False,
                                  span_sigma_mm=_span_sigma(reference, scale_source)),
            # Не измеряется по той же причине, что и на основном пути (см. там):
            # это отклонение точек фасада от плоскости, а точек в пространстве
            # у ядра нет.
            plane_residual_mm=None,
        ),
        elements=[],
    )


# ---------------------------------------------------------------------------
# Фазы конвейера. План 3, задача 4.
#
# Граница проведена по ЗАВИСИМОСТЯМ, а не по шагам алгоритма. Замер на снимке
# 20.9 Мп: оценка плоскости 1.7-4.1 с и резкость 0.3-0.7 с от опорной базы не
# зависят; поза, поля разрешения и углов и вердикт — 6-44 мс, зависят; пиксели
# растра — 34-48 мс при 10 мм/px и до 4 с при наибольшем разрешении; разметка —
# около 0.5 мс на элемент. Прежде всё это было одним проходом `process()`, и
# интерфейс оператора пересчитывал бы оценку плоскости при каждом вводе длины
# опорной базы и растр — при каждом сдвиге угла проёма.
#
# `process()` осталась фасадом над фазами с прежними подписью и поведением;
# совпадение её выхода с выходом до распила проверено побайтно на пяти сценариях.
# ---------------------------------------------------------------------------

#: Отказ фазы масштаба на кадре без восстановленной плоскости. Путь `process()`
#: на таком кадре выдаёт модель отказа (`_needs_operator_model`), а фаза масштаба
#: отказывает по имени: считать масштаб без гомографии не на чем. Ручное задание
#: плоскости (п. 4.2) — план 3, задача 5.
SCALE_NEEDS_PLANE = (
    "масштаб не вычисляется: плоскость фасада не восстановлена (needs_operator), "
    "гомографии нет; нужна плоскость, заданная оператором (спецификация, п. 4.2)")


@dataclass(frozen=True)
class FrameStage:
    """Фаза кадра: всё, что от опорной базы НЕ зависит. Секунды; в фоне.

    `sharpness` хранится сырой, без проверки конечности: проверяют её потребители —
    `quality.assess` и модель отказа, — каждый со своей прежней формулировкой.
    """

    path: Path
    frame: Frame
    camera_record: CameraIntrinsics
    plane: object                 # PlaneResult: H, vh, vv, confidence, needs_operator, method
    sharpness: float

    @property
    def image_size(self) -> tuple[int, int]:
        return self.frame.size


@dataclass(frozen=True)
class ScaleStage:
    """Фаза масштаба: всё, что зависит от опорной базы. Миллисекунды.

    Вердикт `quality` здесь — окончательный и единственный экземпляр: от длины базы
    он зависит (разрешение считается от `mm_per_unit`), поэтому живёт в этой фазе,
    а не в фазе кадра.
    """

    reference: OperatorReference
    scale_source: str
    mm_per_unit: float
    origin_rect: tuple
    camera: CameraOnPlane
    half_turn: object
    fields: object                # _FrameFields
    quality: QualityReport
    attainable_mm_per_px: tuple
    sigma_rel: float
    bounds_mm: list
    theta_cam_deg: float


def frame_stage(image_path, *, profile_path=None, with_color: bool = False,
                operator_reference: OperatorReference | None = None,
                plane_override: ManualPlane | None = None, roi=None) -> FrameStage:
    """Кадр, внутренние параметры, плоскость, резкость.

    `operator_reference`, если передан, проверяется на попадание в кадр ДО оценки
    плоскости: опечатка в координатах не должна стоить двух секунд расчёта, а
    `process()` проверяла её именно в этом месте.

    `plane_override` — плоскость, заданная оператором (п. 4.2, план 3, задача 5):
    точки схода тогда не оцениваются вовсе, и путь `needs_operator` получает выход.
    `roi` — область кадра, по отрезкам которой оцениваются точки схода. Вместе они
    не передаются: области оценки у ручной плоскости нет, и молча проигнорированный
    аргумент выглядел бы применённым.
    """
    path = Path(image_path)
    frame = load_frame(path, profile_path, with_color=with_color)
    if operator_reference is not None:
        _points_in_frame(operator_reference, frame.size)
    camera_record = _camera_record(frame.meta, frame.K, frame.source, frame.dist)
    plane = _plane_for(frame, plane_override, roi)
    return FrameStage(path=path, frame=frame, camera_record=camera_record, plane=plane,
                      sharpness=float(quality_gate.sharpness(frame.gray)))


def _plane_for(frame: Frame, plane_override, roi):
    """Плоскость кадра: ручная (мгновенно) либо по точкам схода (секунды)."""
    if plane_override is not None:
        if roi is not None:
            raise ValueError("область оценки точек схода (roi) к ручной плоскости не "
                             "применяется: точки схода при ней не оцениваются")
        _manual_points_in_frame(plane_override.image_pts, frame.size)
        return estimate_plane_manual(
            plane_override.image_pts, aspect_ratio=plane_override.aspect_ratio,
            size_mm=plane_override.size_mm,
            assume_calibrated=plane_override.assume_calibrated,
            K=frame.K, image_size=frame.size)
    return estimate_plane(frame.gray, frame.K, roi=roi)


def replace_plane(fs: FrameStage, *, plane_override: ManualPlane | None = None,
                  roi=None) -> FrameStage:
    """Та же фаза кадра с другой плоскостью — без повторного чтения снимка и резкости.

    Нужна интерфейсу оператора (план 3, задача 10): ручная плоскость считается
    мгновенно, и перечитывать ради неё снимок и резкость (до 0.7 с на 20 Мп) незачем.
    С `roi` точки схода оцениваются заново — это секунды, и вызывающий делает это в
    фоне. Без обоих аргументов — снова автоматическая плоскость по всему кадру.
    """
    return dataclasses.replace(fs, plane=_plane_for(fs.frame, plane_override, roi))


def _manual_points_in_frame(points, image_size) -> None:
    """Углы ручной плоскости — ровно четыре конечные точки в кадре `[0, w−1]×[0, h−1]`."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        raise ValueError("ручная плоскость: нужны ровно четыре конечные точки (x, y)")
    w, h = image_size
    for i, (x, y) in enumerate(pts):
        if not (0.0 <= x <= w - 1 and 0.0 <= y <= h - 1):
            raise ValueError(f"ручная плоскость: угол {i + 1} ({x:.1f}, {y:.1f}) лежит "
                             f"вне кадра {w}x{h}: оператор указывает точки на снимке")


def scale_stage(fs: FrameStage, operator_reference: OperatorReference, *,
                scale_source: str = "operator_reference") -> ScaleStage:
    """Масштаб, поза, поля разрешения и углов, окончательный вердикт, σ масштаба."""
    scale_source = _checked_scale_source(scale_source)
    image = fs.frame.gray
    image_size = fs.image_size
    _points_in_frame(operator_reference, image_size)
    plane = fs.plane
    if plane.needs_operator:
        raise ValueError(SCALE_NEEDS_PLANE)
    K = fs.frame.K

    span_rect = _apply(plane.H, np.asarray(operator_reference.span_px, dtype=float))
    span_units = float(np.linalg.norm(span_rect[1] - span_rect[0]))
    if not math.isfinite(span_units) or span_units <= 0:
        raise ValueError("опорная база оператора вырождена в ректифицированных "
                         f"координатах: длина {span_units}")
    mm_per_unit = _finite("mm_per_unit", operator_reference.span_mm / span_units)
    origin_rect = tuple(_apply(plane.H, [operator_reference.origin_px])[0])
    if not all(math.isfinite(v) for v in origin_rect):
        raise ValueError("начало отсчёта оператора лежит на линии схода плоскости")

    camera, half_turn = _pose(plane, K, mm_per_unit, origin_rect)

    fields = _frame_fields(plane.H, camera, mm_per_unit, origin_rect, image_size,
                           quality_gate.THETA_MAX_DEG)

    report = quality_gate.assess(image, fields.gsd_min, fields.gsd_max,
                                 fields.theta_field_deg["p95"], fields.usable,
                                 sharpness_value=fs.sharpness)
    # Доля кадра за линией схода — причина деградации, а не повод молча усреднить.
    if fields.behind_vanishing_line > 1.0 - quality_gate.DEFAULT.usable_min_fraction:
        report = report.model_copy(update={
            "verdict": "degraded" if report.verdict == "ok" else report.verdict,
            "reasons": [*report.reasons,
                        (f"за линией схода оказалось {fields.behind_vanishing_line:.0%} "
                         "кадра: эта часть на плоскость фасада не смотрит")],
        })

    attainable = attainable_mm_per_px(image, plane.H, mm_per_unit)

    if scale_source == "operator_reference":
        end_gsds = [_finite("gsd_at_reference", _gsd_at(fields.gsd, image_size, point))
                    for point in operator_reference.span_px]
        sigma_rel = _scale_sigma_rel(operator_reference.span_mm,
                                     operator_reference.end_sigmas, end_gsds,
                                     operator_reference.span_sigma_mm)
    else:
        sigma_rel = SIGMA_REL_ASSUMED_FLOOR_HEIGHT

    h_px, w_px = image.shape[:2]
    corners_mm = rectified_to_facade_mm(
        _apply(plane.H, [[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]]),
        origin_rect, mm_per_unit)
    bounds_mm = [_finite("facade.bounds_mm", v) for v in
                 (corners_mm[:, 0].min(), corners_mm[:, 1].min(),
                  corners_mm[:, 0].max(), corners_mm[:, 1].max())]

    theta_cam = _theta_at(plane.H, camera, mm_per_unit, origin_rect,
                          (float(K[0, 2]), float(K[1, 2])))
    return ScaleStage(reference=operator_reference, scale_source=scale_source,
                      mm_per_unit=mm_per_unit, origin_rect=origin_rect, camera=camera,
                      half_turn=half_turn, fields=fields, quality=report,
                      attainable_mm_per_px=tuple(attainable), sigma_rel=sigma_rel,
                      bounds_mm=bounds_mm, theta_cam_deg=theta_cam)


def _pose(plane, K, mm_per_unit, origin_rect):
    """Поза камеры для любой плоскости. Одно место выбора — для фазы масштаба и
    для предпросмотра углов, чтобы выбор не разошёлся в двух местах."""
    if plane.vh is not None and plane.vv is not None:
        return camera_pose(plane.H, plane.vh, plane.vv, K, mm_per_unit, origin_rect)
    # Ручная плоскость по отношению сторон или двум размерам (п. 4.2): точек схода
    # нет, гомография — в единицах самого прямоугольника. Поза — разложением;
    # ориентацию осей задал оператор порядком углов.
    return pose_from_homography(plane.H, K, mm_per_unit, origin_rect,
                                half_turn=HALF_TURN_FROM_OPERATOR_ORDER)


@dataclass(frozen=True)
class ThetaPreview:
    """Поле углов визирования до опорной базы. План 3, задачи 9 и 12.

    `theta_deg` и `usable_mask` — на сетке узлов кадра (`FIELD_SHAPE`), как у фазы
    масштаба; просмотрщик растягивает маску на кадр.
    """

    theta_deg: np.ndarray
    usable_mask: np.ndarray
    usable_fraction: float
    theta_p95: float
    half_turn: object             # HalfTurnAmbiguity: от масштаба не зависит


#: Условный масштаб предпросмотра: 1 мм на ректифицированную единицу, начало — нуль.
#: Значение не важно по построению — см. `theta_preview`.
_PREVIEW_MM_PER_UNIT = 1.0


def theta_preview(fs: FrameStage) -> ThetaPreview:
    """Углы визирования и пригодная зона θ ≤ 30° БЕЗ опорной базы.

    Оператор должен видеть, где проём мерить можно, до разметки (п. 2.4), а база к
    тому времени может быть не указана. Углы от масштаба и начала отсчёта не зависят:
    равномерное растяжение сцены и сдвиг начала углов не меняют (проверено: θ_cam и
    θ_p95 при длинах базы 20 000 и 15 000 мм совпадают до шестого знака). Поэтому
    поле считается при условном масштабе тем же `_frame_fields`, что и в фазе
    масштаба, и совпадает с ним (тест).
    """
    if fs.plane.needs_operator:
        raise ValueError(SCALE_NEEDS_PLANE)
    origin = (0.0, 0.0)
    camera, half_turn = _pose(fs.plane, fs.frame.K, _PREVIEW_MM_PER_UNIT, origin)
    fields = _frame_fields(fs.plane.H, camera, _PREVIEW_MM_PER_UNIT, origin,
                           fs.image_size, quality_gate.THETA_MAX_DEG)
    return ThetaPreview(theta_deg=fields.theta_deg, usable_mask=fields.usable_mask,
                        usable_fraction=fields.usable,
                        theta_p95=fields.theta_field_deg["p95"], half_turn=half_turn)


def auto_raster_mm_per_px(fs: FrameStage, ss: ScaleStage) -> float:
    """Разрешение растра, при котором в нём примерно столько пикселей, сколько в кадре.

    `raster_mm_per_px` у CLI обязателен и ограничен `attainable_mm_per_px`, а
    интерфейсу оператора его взять неоткуда. Площадь растра обратна квадрату
    разрешения, поэтому оно подбирается по площади образа кадра на фасаде и
    зажимается в достижимые границы: растр не бывает ни больше `MAX_SIDE_PX`, ни
    меньше пикселя.
    """
    lo, hi = ss.attainable_mm_per_px
    x0, y0, x1, y1 = ss.bounds_mm
    area_mm2 = abs((x1 - x0) * (y1 - y0))
    w, h = fs.image_size
    guess = math.sqrt(area_mm2 / float(w * h)) if area_mm2 > 0 else hi
    return float(min(max(guess, lo), hi))


def raster_geometry_for(fs: FrameStage, ss: ScaleStage,
                        mm_per_px: float | None = None) -> RasterGeometry:
    """Геометрия растра (дёшево). `None` — разрешение подбирается автоматически."""
    if mm_per_px is None:
        mm_per_px = auto_raster_mm_per_px(fs, ss)
    return raster_geometry(fs.frame.gray, fs.plane.H, ss.mm_per_unit, mm_per_px,
                           origin_rect_units=ss.origin_rect)


def raster_stage(fs: FrameStage, geometry: RasterGeometry, *,
                 color: bool = False) -> Rectified:
    """Пиксели растра по готовой геометрии — той же, что уйдёт в выходной файл."""
    source = fs.frame.color if color else fs.frame.gray
    if source is None:
        raise ValueError("цветной кадр не загружен: frame_stage(..., with_color=True)")
    return warp(source, geometry)


def elements_stage(fs: FrameStage, ss: ScaleStage, geometry: RasterGeometry,
                   marks: list[ElementMark] | None = None, *, image_id: str = "img_0",
                   mark_sigma_px: float = DEFAULT_MARK_SIGMA_PX,
                   wall_flatness_mm: float = WALL_FLATNESS_DEVIATION_MM) -> FacadeModel:
    """Разметка → модель фасада. Не вызывает ни оценки плоскости, ни растра."""
    plane, camera = fs.plane, ss.camera
    report = ss.quality
    elements = []
    if marks is not None:
        elements = digitize_elements(
            marks, H=plane.H, camera=camera, mm_per_unit=ss.mm_per_unit,
            origin_rect=ss.origin_rect, image_size=fs.image_size, gsd_field=ss.fields.gsd,
            sigma_px=mark_sigma_px, sigma_rel=ss.sigma_rel,
            residual_px=_finite_or_none(plane.confidence.residual_px),
            calibration=fs.camera_record.calibration, image_id=image_id,
            wall_flatness_mm=wall_flatness_mm)

    # Шлюз качества ветвится на собственный вердикт. Обоснование выбора из трёх
    # возможных ветвлений — в `GATE_REJECT_WITHHOLDS_TOLERANCE`. Причина
    # дописывается в тот же `quality.reasons` только вместе с самим действием:
    # без элементов утверждать о них нечего.
    if report.verdict == "reject" and elements:
        elements = [element.model_copy(update={"meets_tolerance": None})
                    for element in elements]
        report = report.model_copy(update={
            "reasons": [*report.reasons, GATE_REJECT_WITHHOLDS_TOLERANCE]})

    meta, half_turn = fs.frame.meta, ss.half_turn
    record = ImageRecord(
        id=image_id,
        path=str(fs.path),
        camera=fs.camera_record,
        captured_at=meta.captured_at,
        gnss=meta.gnss,
        pose_to_facade={
            "cx": _finite("pose_to_facade.cx", camera.cx),
            "cy": _finite("pose_to_facade.cy", camera.cy),
            "cz": _finite("pose_to_facade.cz", camera.cz),
            "units": "mm",
            "half_turn": {"resolved": bool(half_turn.resolved),
                          "assumption": half_turn.assumption,
                          "reason": half_turn.reason,
                          "resolver": half_turn.resolver},
        },
        theta_cam_deg=ss.theta_cam_deg,
        theta_field_deg=ss.fields.theta_field_deg,
        quality=report,
        rectification=Rectification(
            method=plane.method,
            confidence=_finite("rectification.confidence", plane.confidence.value),
            residual_px=_finite_or_none(plane.confidence.residual_px),
            needs_operator=False),
        homography={"from": "image_px", "to": "rectified_px",
                    "H": [[_finite("homography.H", v) for v in row]
                          for row in geometry.H],
                    "origin_rect_px": [_finite("homography.origin_rect_px", v)
                                       for v in geometry.origin_rect_px],
                    "attainable_mm_per_px": [_finite("attainable_mm_per_px", v)
                                             for v in ss.attainable_mm_per_px]},
    )
    return FacadeModel(
        software_version=SOFTWARE_VERSION,
        coverage=_coverage(ss.reference),
        mode="assisted",
        images=[record],
        facade=FacadeRecord(
            origin=_origin(ss.reference),
            bounds_mm=ss.bounds_mm,
            mm_per_rectified_px=_finite("facade.mm_per_rectified_px", geometry.mm_per_px),
            scale=_scale_estimate(ss.scale_source, ss.sigma_rel,
                                  span_sigma_mm=_span_sigma(ss.reference, ss.scale_source)),
            plane_residual_mm=None,
        ),
        elements=elements,
    )


def process(image_path, *, operator_reference: OperatorReference,
            raster_mm_per_px: float, profile_path=None,
            scale_source: str = "operator_reference",
            image_id: str = "img_0",
            marks: list[ElementMark] | None = None,
            mark_sigma_px: float = DEFAULT_MARK_SIGMA_PX,
            wall_flatness_mm: float = WALL_FLATNESS_DEVIATION_MM,
            save_rectified_to=None, plane_override: ManualPlane | None = None,
            roi=None) -> FacadeModel:
    """Снимок -> модель фасада. Один проход геометрического ядра.

    `operator_reference` обязателен: без опорного размера расстояние до плоскости
    неизвестно, а `camera_pose` требует именно его. `raster_mm_per_px` — разрешение
    выходного растра; выход за достижимые границы поднимает `ValueError`, называющий
    их, и глушить его нельзя — это ровно то, что нужно вызывающему.

    `scale_source` различает два пути, геометрически ОДИНАКОВЫХ: оператор указывает
    те же две точки, и `span_mm` попадает в расчёт тем же способом. Различие в
    происхождении `span_mm` и потому в σ — измеренный размер даёт σ из длины базы,
    типовая высота этажа берёт σ из таблицы п. 6.3. Два оставшихся источника
    спецификации ядру недоступны и отвергаются поимённо, а не отсутствуют молча.

    Несовпадение профиля калибровки со снимком тоже поднимает `ValueError`: профиль
    передан явным действием оператора, и противоречие означает чужой файл.

    `marks` — разметка проёмов оператором (задача 18, `pipeline.elements.ElementMark`),
    необязательная: без неё `elements` остаётся пустым списком. Разметка требует
    восстановленной плоскости фасада: на пути `needs_operator` она поднимает
    `ValueError`, а не молча игнорируется — размечать элементы без гомографии
    нечем. Различаются `None` (разметки не передавали) и переданный список: пустой
    список до сюда не доходит, его отвергает `parse_marks`, называя причину.

    `mark_sigma_px` — точность клика ПО УГЛУ ПРОЁМА
    (`DEFAULT_MARK_SIGMA_PX`). Это не `operator_reference.sigma_px`: та величина
    описывает клик по концам опорной базы масштаба, выполняемый на уменьшенном
    экране, и в бюджете габарита она даёт 31.8 мм при GSD 5 — втрое больше
    допуска п. 2.2. `wall_flatness_mm` — названное допущение о неплоскостности
    стены (`elements.WALL_FLATNESS_DEVIATION_MM`), тоже перекрываемое: конвейер
    эту величину не измеряет.

    **Вердикт шлюза качества здесь не декоративен.** При `verdict = "reject"`
    элементы выпускаются с габаритом и σ, но БЕЗ `meets_tolerance`, и причина
    этого дописывается в `quality.reasons` того же файла. Выбор именно такого
    ветвления (а не отказа по снимку и не отказа от слова «шлюз») обоснован в
    `GATE_REJECT_WITHHOLDS_TOLERANCE`.

    **`coverage` следует из начала координат (п. 7).** Начало в левом нижнем углу
    фасада (`OperatorReference.origin_is_facade_corner`) даёт `facade.origin =
    "bottom_left"` и `coverage = "full"`, начало в произвольной точке оператора —
    `"operator_reference"` и `"partial"`. Пара согласована и выпускается только
    вместе (план 3, задача 11).

    `save_rectified_to` — путь для выровненного растра (`--save-rectified` CLI).
    Сохраняется ТОТ ЖЕ объект `Rectified`, что и породил `homography.H` этой же
    записи: второго, независимого вызова `rectify()` здесь нет, поэтому записанная
    гомография и сохранённый файл не могут разойтись (спецификация задачи 18,
    требование к шагу 7 плана). Рядом сохраняется маска охвата (`_mask` в имени).
    Растр — для просмотра и работы оператора, а не носитель измерения: сама
    разметка выполняется по ИСХОДНОМУ снимку (п. 6.5).
    """
    scale_source = _checked_scale_source(scale_source)
    fs = frame_stage(image_path, profile_path=profile_path,
                     operator_reference=operator_reference,
                     plane_override=plane_override, roi=roi)
    if fs.plane.needs_operator:
        if marks is not None:
            raise ValueError(
                "разметка элементов недоступна: доверие к плоскости ниже порога, "
                "плоскость не восстановлена, и размечать элементы не на чём "
                "(сначала нужно подтверждение оператора, спецификация, п. 4.2)")
        if save_rectified_to is not None:
            raise ValueError(
                "растр нечего сохранять: плоскость фасада не восстановлена "
                "(needs_operator), ректификация не выполнялась")
        frame = fs.frame
        return _needs_operator_model(fs.path, image_id, frame.meta, frame.K, frame.dist,
                                     frame.source, fs.plane, operator_reference,
                                     fs.sharpness, raster_mm_per_px, scale_source)

    ss = scale_stage(fs, operator_reference, scale_source=scale_source)
    geometry = raster_geometry_for(fs, ss, raster_mm_per_px)
    if save_rectified_to is not None:
        rectified = raster_stage(fs, geometry)
        raster_path = Path(save_rectified_to)
        save_image(raster_path, rectified.image)
        mask_path = raster_path.with_name(f"{raster_path.stem}_mask{raster_path.suffix}")
        save_image(mask_path, (rectified.valid_mask.astype(np.uint8) * 255))
    return elements_stage(fs, ss, geometry, marks, image_id=image_id,
                          mark_sigma_px=mark_sigma_px, wall_flatness_mm=wall_flatness_mm)


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="facade-digitize",
        description="Снимок фасада -> модель фасада (JSON схемы 1.1).")
    parser.add_argument("images", nargs="+", help="снимки фасада")
    parser.add_argument("--raster-mm-per-px", type=float, required=True,
                        help="разрешение выходного растра, мм на пиксель")
    parser.add_argument("--origin-px", type=float, nargs=2, required=True,
                        metavar=("X", "Y"),
                        help="точка кадра, которая станет началом координат фасада")
    parser.add_argument("--span-px", type=float, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="две точки кадра, между которыми известно расстояние")
    parser.add_argument("--span-mm", type=float, required=True,
                        help="истинное расстояние между точками --span-px, мм")
    parser.add_argument("--sigma-px", type=float, default=DEFAULT_OPERATOR_SIGMA_PX,
                        help="точность указания КОНЦОВ ОПОРНОЙ БАЗЫ масштаба, "
                             "пикселей снимка (клик по кадру целиком)")
    parser.add_argument("--mark-sigma-px", type=float, default=DEFAULT_MARK_SIGMA_PX,
                        help="точность указания УГЛА ПРОЁМА оператором, пикселей "
                             "снимка (клик с увеличением). Отдельная от --sigma-px "
                             "величина: разное увеличение — разная точность")
    parser.add_argument("--wall-flatness-mm", type=float,
                        default=WALL_FLATNESS_DEVIATION_MM,
                        help="принимаемое отклонение стены от плоскости, мм "
                             "(п. 6.1: ±20–50). Допущение, а не измерение: "
                             "конвейер неплоскостность не измеряет")
    parser.add_argument("--scale-source", default="operator_reference",
                        help="источник масштаба (п. 6.3): operator_reference либо "
                             "assumed_floor_height; photogrammetry и exif_range "
                             "геометрическому ядру недоступны")
    parser.add_argument("--profile", default=None,
                        help="профиль калибровки камеры по мишени (JSON)")
    parser.add_argument("--marks", default=None,
                        help="JSON-разметка проёмов оператором (задача 18): список "
                             "объектов class/mounting/edge_type/corners_px[/reveal]; "
                             "координаты — в ИСХОДНОМ снимке, не в выровненном растре")
    parser.add_argument("--save-rectified", action="store_true",
                        help="сохранить выровненный растр и маску охвата рядом с "
                             "JSON (<имя>_rectified.png, <имя>_rectified_mask.png). "
                             "Растр — для просмотра оператором и работы с ним, а НЕ "
                             "носитель измерения: сама разметка — по исходному "
                             "снимку (спецификация, п. 6.5)")
    parser.add_argument("--out-dir", default=None,
                        help="каталог для выходных JSON; по умолчанию рядом со снимком")
    manual = parser.add_argument_group(
        "ручная плоскость (спецификация, п. 4.2; план 3, задача 5)",
        "Четыре угла заведомого прямоугольника в плоскости стены — верхний левый, "
        "верхний правый, нижний правый, нижний левый, в пикселях кадра — и РОВНО одно "
        "доопределение. Точки схода тогда не оцениваются.")
    manual.add_argument("--manual-plane", type=float, nargs=8, default=None,
                        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"))
    manual.add_argument("--manual-aspect", type=float, default=None,
                        help="отношение сторон прямоугольника, ширина / высота")
    manual.add_argument("--manual-size-mm", type=float, nargs=2, default=None,
                        metavar=("W", "H"), help="ширина и высота прямоугольника, мм")
    manual.add_argument("--manual-calibrated", action="store_true",
                        help="камера откалибрована, стороны прямоугольника ортогональны")
    parser.add_argument("--roi", type=float, nargs="+", default=None,
                        help="область кадра для оценки точек схода: x1 y1 x2 y2 x3 y3 ... "
                             "(не менее трёх точек) — для снимка угла здания")
    args = parser.parse_args(argv)
    constraints = (args.manual_aspect is not None, args.manual_size_mm is not None,
                   args.manual_calibrated)
    if args.manual_plane is None and any(constraints):
        parser.error("доопределение ручной плоскости задано без --manual-plane")
    if args.manual_plane is not None and sum(constraints) != 1:
        parser.error("к --manual-plane нужно РОВНО одно из: --manual-aspect, "
                     "--manual-size-mm, --manual-calibrated")
    if args.roi is not None and (len(args.roi) < 6 or len(args.roi) % 2):
        parser.error("--roi: чётное число координат, не менее трёх точек")
    if args.roi is not None and args.manual_plane is not None:
        parser.error("--roi к ручной плоскости не применяется: точки схода при ней не "
                     "оцениваются")
    return args


def _manual_plane_from_args(args) -> ManualPlane | None:
    if args.manual_plane is None:
        return None
    pts = np.asarray(args.manual_plane, dtype=float).reshape(4, 2)
    return ManualPlane(image_pts=pts, aspect_ratio=args.manual_aspect,
                       size_mm=tuple(args.manual_size_mm) if args.manual_size_mm else None,
                       assume_calibrated=bool(args.manual_calibrated))


def main(argv=None) -> int:
    """CLI. Отказ по одному снимку не роняет пакет и называет виноватый файл.

    Возвращает ненулевой код, если хотя бы один снимок обработать не удалось, —
    но лишь после того, как обработаны остальные: пакетный проход, брошенный на
    первом же негодном файле, заставляет оператора запускать его заново по одному.
    """
    args = _parse_args(argv)
    x1, y1, x2, y2 = args.span_px
    reference = OperatorReference(origin_px=tuple(args.origin_px),
                                  span_px=((x1, y1), (x2, y2)),
                                  span_mm=args.span_mm, sigma_px=args.sigma_px)
    # `--marks` задаётся в пикселях ОДНОГО снимка, поэтому пакет с разметкой
    # отвергается целиком и до всякой обработки. Прежде файл читался один раз и
    # подавался на каждый снимок пакета без сверки чего бы то ни было — даже
    # размеров кадра: координаты окна с первого снимка давали уверенные
    # миллиметры на втором, где этого окна нет. Охрана на уровне элемента
    # (`elements._points_in_frame`) ловит лишь тот случай, когда точка вылезла за
    # край чужого кадра; совпади размеры — не поймала бы и она.
    if args.marks and len(args.images) > 1:
        print("--marks задаётся в пикселях одного снимка и не переносится на "
              f"пакет из {len(args.images)}: запустите по снимку на разметку",
              file=sys.stderr)
        return 1
    # Файл читается и проверяется на строгость формата ровно один раз, до цикла:
    # негодный `--marks` должен назвать причину сразу, а не при обработке первого
    # снимка пакета.
    try:
        marks = load_marks(args.marks) if args.marks else None
    except (ValueError, FileNotFoundError) as error:
        print(f"--marks {args.marks}: {error}", file=sys.stderr)
        return 1
    failures = 0
    for index, image in enumerate(args.images):
        source = Path(image)
        out_dir = Path(args.out_dir) if args.out_dir else source.parent
        save_rectified_to = (out_dir / f"{source.stem}_rectified.png"
                             if args.save_rectified else None)
        if save_rectified_to is not None:
            # `save_image` пишет байты сама и каталог не создаёт (задача 17,
            # починка `cv2.imwrite` на путях с кириллицей); каталог обязан
            # существовать ДО того, как `process()` попытается в него писать.
            out_dir.mkdir(parents=True, exist_ok=True)
        try:
            model = process(source, operator_reference=reference,
                            raster_mm_per_px=args.raster_mm_per_px,
                            profile_path=args.profile,
                            scale_source=args.scale_source,
                            image_id=f"img_{index}",
                            marks=marks,
                            mark_sigma_px=args.mark_sigma_px,
                            wall_flatness_mm=args.wall_flatness_mm,
                            save_rectified_to=save_rectified_to,
                            plane_override=_manual_plane_from_args(args),
                            roi=(np.asarray(args.roi, dtype=float).reshape(-1, 2)
                                 if args.roi is not None else None))
        # Отказ по снимку — свойство этого снимка либо переданных для него
        # аргументов; он называется по имени файла и не прекращает пакет.
        except (ValueError, FileNotFoundError) as error:
            failures += 1
            print(f"{source.name}: {error}", file=sys.stderr)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source.stem}.json"
        out_path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        verdict = model.images[0].quality.verdict if model.images[0].quality else "—"
        print(f"{source.name}: {out_path} ({verdict})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
