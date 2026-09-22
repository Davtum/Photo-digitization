"""Оцифровка проёмов оператором. Спецификация, разделы 5, 6, 10.

Задача 18 — первая, после которой `elements` в выходном файле не пуст. Ядро (задачи
1-17) даёт метрическую систему координат на плоскости фасада и доказательство её
точности; этот модуль — тонкая обвязка над ним: оператор указывает углы проёма и,
при необходимости, внутреннюю кромку грани откоса, а модуль переводит их в
миллиметры с погрешностью и решает, что из этого годится в выход.

**Автоматической детекции здесь нет и быть не должно.** `ElementMark` не несёт ни
`confidence`, ни `size_mm`, ни `recess.value_mm` — всё вычисляемое вычисляется, а не
принимается на вход. `digitize_elements` всегда ставит `origin = "operator"`:
значения `auto`, `auto_confirmed`, `auto_edited` здесь не выдаются, потому что
автоматики, которая бы их заслужила, нет (спецификация вводит их для этапа 5).

**Где оператор указывает точки.** В ИСХОДНОМ снимке, а не в выровненном растре.
Спецификация, п. 6.5: ректифицированное изображение интерполировано, а на дальней
стороне косого кадра — апсемплировано, поэтому измерение по нему было бы измерением
по вычисленным пикселям. `corners_px` и `reveal.inner_edge_px` переводятся в
миллиметры фасада ТОЙ ЖЕ гомографией `H`, что записана в выходной файл
(`facade_digitizer.geometry.homography.apply_homography`) — второй, независимой
реализации этого перевода в модуле сознательно нет: разойдись она с первой,
расхождение было бы невидимо. Растр, который сохраняет CLI (`--save-rectified`),
служит оператору для ПРОСМОТРА и не является носителем измерения.

**Откуда берётся σ направления позы для перевода в миллиметры глубины (п. 6.2).**
Функция `reveal_depth_theta_min_deg` и, вслед за ней, `_recess_sigma_mm` берут
σ_θ = `REFERENCE_SIGMA_THETA_DEG` (0.344°) — измеренный P95 ошибки направления на
РАЗРЕЖЕННОЙ синтетической сцене (спецификация, п. 2.2). Это осознанный выбор из
трёх кандидатов:

1. **Худшее из двух измеренных значений (0.344° разреженной сцены против 0.045°
   обогащённой) — выбрано.** Косвенных признаков «эта сцена обогащённая» конвейер
   в момент оцифровки не имеет: `plane.confidence` измеряет устойчивость точек
   схода, а не наличие вертикальных членений фасада, и подставлять по нему одно
   из двух измеренных значений значило бы решать вопрос пунктом 3 ниже под другим
   именем.
2. Взять типичное фасадное членение как признак обогащённости сцены и переключать
   σ_θ по нему — отклонено: это требовало бы отдельного классификатора фасада
   внутри геометрического модуля, которого модуль не строит, и подменяло бы
   измеренную σ гипотезой о сцене.
3. **Взять σ_θ по измеренному доверию к плоскости (`plane.confidence`) — отклонено
   явно.** Спецификация, п. 4.2 прямо фиксирует: доверие «предсказателем точности
   НЕ является» (измеренная корреляция с фактической ошибкой направления
   −0.37…−0.43). Использовать его как источник σ_θ значило бы выдать заведомо
   слабый предиктор за измерение точности — ровно то, что спецификация запрещает.

Итог: `REFERENCE_SIGMA_THETA_DEG` — консервативная константа, а не переменная по
доверию. Она завышает σ_u (а с ним и σ_depth) на фасадах с выраженными вертикальными
членениями примерно в 7.6 раза (0.344° / 0.045°) против достижимого там запаса, но
не занижает её нигде: ошибка — в БЕЗОПАСНУЮ сторону, как и требует общий принцип
проекта (`geometry/parallax.py`, докстринг `reveal_depth_sigma`).
"""
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from facade_digitizer.geometry.angles import local_gsd_field
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import apply_homography, rectified_to_facade_mm
from facade_digitizer.geometry.parallax import (
    REFERENCE_SIGMA_CZ_REL,
    REFERENCE_SIGMA_THETA_DEG,
    reveal_depth,
    reveal_depth_sigma,
    visible_reveal_side,
)
from facade_digitizer.pipeline.quality import THETA_MAX_DEG
from facade_digitizer.schema import Element, Point, Recess, SizeMM, Strict, ThetaDeg

#: Сетка узлов для локального разрешения. То же значение, что `run.py:FIELD_SHAPE`,
#: но совпадение не обязательно: `local_gsd_field` — чистая функция от (H,
#: mm_per_unit, image_size), и достаточно мелкая сетка даёт устойчивый ближайший
#: узел у любого угла проёма. Отдельная константа — чтобы `elements.py` оставался
#: самостоятельным модулем и не зависел от приватных имён `run.py`.
FIELD_SHAPE = (64, 64)

#: Цель п. 2.2 для режима `assisted`: 1σ ≤ 10 мм на габарит по наружному контуру.
#: `meets_tolerance` сравнивает с этим значением ПОСЧИТАННУЮ σ (RSS четырёх
#: слагаемых п. 6.1 при фактических угле и GSD), а не берёт готовый ответ по
#: режиму — ровно то, что требует спецификация («`meets_tolerance` вычисляется по
#: фактическому углу, а не по режиму», п. 6.1).
ASSISTED_SIZE_TOLERANCE_MM = 10.0

#: Неплоскостность стены, п. 6.1: диапазон ±20–50 мм, эффект = отклонение·tg θ.
#: Взята ВЕРХНЯЯ граница диапазона — консервативная оценка, а не «типичное»
#: значение: недооценка здесь означала бы заниженную σ, то есть ошибку в опасную
#: сторону (общий принцип проекта). Совпадает с верхней строкой таблицы п. 6.1: при
#: θ = 30° даёт 50·tg30° ≈ 28.9 мм — ровно верхний край названного там «11.5–29 мм».
WALL_FLATNESS_DEVIATION_MM = 50.0

#: Остаточная проективная невязка, когда `residual_px` не измерена (путь
#: `manual_four_point`: `ManualPlaneConfidence.residual_px is None`, оператор задал
#: плоскость четырьмя точками, и невязку относительно точек схода мерить не от
#: чего). Середина диапазона `assisted` таблицы п. 6.1 (3–7 мм) — не измерение, а
#: названное допущение на месте отсутствующего.
RESIDUAL_REFERENCE_MM = 5.0

#: Датум по умолчанию для измеренной глубины: ребро четверти, а не плоскость рамы.
#: Спецификация, п. 5.5: по ГОСТ 30971 рама в проёмах с четвертью заходит за неё не
#: менее чем на 10 мм и снаружи не видна, поэтому наблюдаемая внутренняя граница —
#: ребро четверти. Разметка в этой задаче не различает датумы, и это единственный,
#: который соответствует тому, что оператор физически видит и кликает.
MEASURED_DATUM = "quarter_edge"

_ID_PREFIX = {"window": "w", "door": "d", "facade_boundary": "fb"}


class RevealMark(Strict):
    """Разметка внутренней кромки видимой грани откоса. Спецификация, п. 5.4-5.5.

    `inner_edge_px` — ровно две точки на границе видимой (дальней от опорной точки
    камеры) грани откоса в ИСХОДНОМ снимке. `side` называет, какая это грань:
    `left`/`right` — вертикальный откос, `top`/`bottom` — перемычка или подоконная
    часть. Правило видимости («видна дальняя от опорной точки грань», п. 5.5)
    здесь не проверяется — это делает `digitize_elements`, у которого есть поза
    камеры; здесь только формат.
    """

    side: Literal["left", "right", "top", "bottom"]
    inner_edge_px: list[Point]

    @model_validator(mode="after")
    def _exactly_two_points(self):
        if len(self.inner_edge_px) != 2:
            raise ValueError(
                "reveal.inner_edge_px обязан нести ровно две точки: внутренняя "
                "кромка грани откоса задаётся отрезком")
        return self


class ElementMark(Strict):
    """Разметка одного элемента оператором. Спецификация, раздел 10, задание 18.

    Чего в формате НЕТ намеренно: `size_mm`, `recess.value_mm`, `confidence`. Всё
    вычисляемое вычисляется `digitize_elements`, а не принимается на вход — иначе
    разметка смогла бы продиктовать результат, и проверить его стало бы нечем
    (спецификация задачи 18, раздел «Формат разметки»).

    **Порядок `corners_px`.** Четыре угла проёма обходятся в ТОМ ЖЕ порядке, в
    каком их задаёт `synth.scene.Opening.corners_mm()` и весь остальной код этого
    проекта: нижний левый, нижний правый, верхний правый, верхний левый (в
    миллиметрах фасада; на экране при типичной съёмке это соответствует обходу по
    часовой стрелке от нижнего левого угла). Порядок используется только для
    расчёта ширины и высоты по пáрам противоположных сторон
    (`_quad_size_mm`) — определение видимой грани откоса от него не зависит,
    оно берётся по осевым границам контура.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_: Literal["window", "door", "facade_boundary"] = Field(alias="class")
    mounting: Literal["embedded", "flush", "protruding"]
    edge_type: Literal["sharp_wall_edge", "surround", "cladding_edge", "unknown"]
    corners_px: list[Point]
    reveal: RevealMark | None = None

    @model_validator(mode="after")
    def _exactly_four_corners(self):
        if len(self.corners_px) != 4:
            raise ValueError(
                "corners_px обязан нести ровно четыре точки: проём — четырёхугольник "
                "(спецификация, п. 2.3)")
        return self


def parse_marks(raw) -> list[ElementMark]:
    """Разбор уже загруженного JSON (список объектов) в разметку с проверкой формата.

    Лишние поля отвергаются `ElementMark`/`RevealMark` (`extra="forbid"`) — формат
    проверяется на строгость, поэтому подмена поля, которого в нём нет, невозможна
    по построению (спецификация задачи 18, свойство 8).
    """
    # ValueError, а не TypeError (вопреки TRY004): негодная разметка --marks
    # обязана ловиться той же веткой, что и остальные отказы валидации входа
    # конвейера (`main()` перехватывает `(ValueError, FileNotFoundError)`), а не
    # завести для формата разметки отдельный тип исключения.
    if not isinstance(raw, list):
        raise ValueError(  # noqa: TRY004
            f"разметка --marks обязана быть JSON-списком элементов, получено "
            f"{type(raw).__name__}")
    return [ElementMark.model_validate(item) for item in raw]


def load_marks(path) -> list[ElementMark]:
    """Файл `--marks` в список `ElementMark`."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_marks(json.loads(text))


def _nearest_node(value_px: float, extent_px: int, count: int) -> int:
    """Индекс ближайшего узла регулярной сетки, покрывающей `[0, extent_px - 1]`.

    Тот же приём, что `pipeline.run._nearest_node` (задача 15): узел сетки
    `local_gsd_field`, ближайший к точке кадра. Реализация продублирована
    сознательно, а не импортирована из `run.py` — `elements.py` не зависит от
    приватных имён модуля, который сам импортирует `elements.py`
    (`run.py` — оркестратор конвейера, `elements.py` — его часть); обе копии
    вычисляют одно и то же по одной и той же формуле и проверены порознь своими
    тестами.
    """
    if extent_px <= 1 or count <= 1:
        return 0
    idx = round(value_px / (extent_px - 1) * (count - 1))
    return max(0, min(count - 1, idx))


def _gsd_at_point(gsd_field: np.ndarray, image_size, point_px) -> float:
    rows, cols = gsd_field.shape
    w_px, h_px = image_size
    col = _nearest_node(float(point_px[0]), w_px, cols)
    row = _nearest_node(float(point_px[1]), h_px, rows)
    return float(gsd_field[row, col])


def _gsd_near(gsd_field: np.ndarray, image_size, points_px, label: str) -> float:
    """Худшее (наибольшее) локальное разрешение среди узлов у заданных точек.

    Бюджет п. 6.1 нормируется по худшему случаю В ПРЕДЕЛАХ проёма, а не по
    среднему по нему — тем же принципом, каким `_frame_fields` в `run.py` берёт
    границы разрешения по пригодной области, а не усредняет.
    """
    values = [_gsd_at_point(gsd_field, image_size, p) for p in points_px]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(
            f"{label}: локальное разрешение у контура не определено — узел лежит "
            "за линией схода плоскости фасада")
    return max(values)


def _in_front_of_camera(H, image_size, points_px) -> np.ndarray:
    """Точки кадра, для которых знаменатель гомографии того же знака, что в центре.

    То же условие, что использует `local_gsd_field`/`run.py:_frame_fields` для
    маскирования узлов за линией схода: такие точки камерой не видны, и переводить
    их в миллиметры фасада нельзя — результат был бы координатами точки, которой
    камера не видит (спецификация, п. 2.2, «пригодная область»).
    """
    w_px, h_px = image_size
    Harr = np.asarray(H, dtype=float)
    w_ref = Harr[2, 0] * (w_px / 2.0) + Harr[2, 1] * (h_px / 2.0) + Harr[2, 2]
    if not np.isfinite(w_ref) or w_ref == 0.0:
        raise ValueError(
            "знаменатель гомографии в центре кадра вырожден: видимая часть кадра "
            "не определена")
    pts = np.atleast_2d(np.asarray(points_px, dtype=float))
    w = Harr[2, 0] * pts[:, 0] + Harr[2, 1] * pts[:, 1] + Harr[2, 2]
    return np.isfinite(w) & (np.sign(w) == np.sign(w_ref))


def _quad_size_mm(contour_mm: np.ndarray) -> tuple[float, float]:
    """Ширина и высота проёма по четырём точкам контура (см. докстринг `ElementMark`).

    Берутся как СРЕДНЕЕ пары противоположных сторон — устойчиво к небольшой
    непрямоугольности реального клика оператора (в отличие от одной пары сторон,
    которая отдала бы результат на волю того, какую сторону оператор кликнул
    точнее).
    """
    p = np.asarray(contour_mm, dtype=float)
    width = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2.0
    height = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2.0
    return float(width), float(height)


def _size_sigma_mm(size_mm: float, theta_deg: float, gsd_local: float, sigma_px: float,
                    sigma_rel: float, residual_mm: float, wall_flatness_mm: float) -> float:
    """σ габарита: сумма RSS четырёх слагаемых п. 6.1 при ФАКТИЧЕСКИХ угле и GSD.

    Слагаемые и их источник в спецификации:

    1. **Локализация двух кромок** — `√2 · σ_px · GSD_local`. Габарит есть разность
       положений двух независимо кликнутых кромок (левой/правой либо верхней/
       нижней), поэтому несёт множитель √2 правила «двухкромочной величины»
       (п. 6.1, та же логика, что и у ширины откоса в п. 6.2). `GSD_local` —
       измеренное локальное разрешение У ЭТОГО проёма (`_gsd_near`), а не табличное
       значение режима.
    2. **Масштаб** — `size_mm · sigma_rel`. `sigma_rel` уже посчитана СБОРКОЙ
       конвейера из длины опорной базы оператора (`run.py:_scale_sigma_rel`,
       задача 14) и передаётся сюда готовой: пересчитывать её заново значило бы
       завести второй источник истины, способный разойтись с `facade.scale.sigma_rel`
       того же выходного файла.
    3. **Остаточная проективная невязка** — `residual_mm`, уже переведённая в
       миллиметры вызывающим кодом (`residual_px · GSD_local`, либо
       `RESIDUAL_REFERENCE_MM`, когда невязка не измерена).
    4. **Неплоскостность стены** — `wall_flatness_mm · tg θ` при ФАКТИЧЕСКОМ угле
       визирования ЭТОГО элемента в его собственной точке (не θ_cam и не сводка
       поля углов кадра — задание 18, требование 3).

    Сумма — RSS: источники ПРЕДПОЛАГАЮТСЯ независимыми (то же допущение, что и в
    `reveal_depth_sigma`, п. 6.2). Это не консервативно: GSD и невязка происходят
    из одной оценки плоскости и в общем случае коррелированы, а при положительной
    корреляции RSS даёт НИЖНЮЮ оценку σ. Строгий бюджет потребовал бы ковариационной
    матрицы позы (та же оговорка, что в п. 6.2).

    **Что сюда намеренно не входит.** Ошибка фокусного расстояния (`ε·sin²θ`) равна
    нулю при откалиброванной камере и в остальных случаях требует знания источника
    K на уровне ЭЛЕМЕНТА, которого эта функция не получает; остаточная дисторсия
    устраняется шагом `calib.undistort` до всякой разметки. Оба — предмет
    отдельного уточнения (см. отчёт задачи 18), а не слагаемые, забытые здесь.
    """
    localisation = math.sqrt(2.0) * sigma_px * gsd_local
    scale = size_mm * sigma_rel
    flatness = wall_flatness_mm * math.tan(math.radians(theta_deg))
    return math.hypot(localisation, scale, residual_mm, flatness)


def _element_id(class_name: str, index: int) -> str:
    return f"{_ID_PREFIX.get(class_name, 'e')}_{index:03d}"


def _digitize_recess(mark: ElementMark, contour_mm: np.ndarray, H, camera: CameraOnPlane,
                      mm_per_unit: float, origin_rect, gsd_field: np.ndarray, image_size,
                      sigma_px: float) -> Recess:
    """Глубина заглубления по разметке внутренней кромки грани откоса. П. 5.4-5.5, 6.2.

    Три исхода, каждый — с названным происхождением, а не тихим числом:

    * оператор разметил грань, которая по позе камеры НЕ видна (правило п. 5.5:
      видна дальняя от опорной точки F грань) — `origin = "unavailable"`, задача 3
      плана («закрой отложенное замечание задачи 3: вертикальный откос виден на
      дальней от F стороне, размеченная ближняя сторона распознаётся, а не
      считается по невидимой грани»);
    * грань видна, но фактический θ_⊥ ниже вычисляемого порога применимости
      (`reveal_depth` вызывает `reveal_depth_theta_min_deg` внутри себя, порог не
      константа — п. 6.2) — тоже `"unavailable"`, отказ `reveal_depth` ловится, а
      не роняет весь снимок;
    * грань видна и выше порога — `origin = "measured_from_reveal"` с `value_mm`,
      `sigma_mm`, `theta_perp_deg` и `datum = "quarter_edge"`.
    """
    reveal = mark.reveal
    p = np.asarray(contour_mm, dtype=float)
    x_left, x_right = float(p[:, 0].min()), float(p[:, 0].max())
    y_bottom, y_top = float(p[:, 1].min()), float(p[:, 1].max())
    cx_edge, cy_edge = float(p[:, 0].mean()), float(p[:, 1].mean())

    visible_vertical, visible_horizontal = visible_reveal_side(
        camera, x_left, x_right, y_bottom, y_top)
    side = reveal.side
    is_vertical = side in ("left", "right")
    expected_side = visible_vertical if is_vertical else visible_horizontal

    if side != expected_side:
        return Recess(origin="unavailable", reveal_side=side)

    if is_vertical:
        edge_x, edge_y = (x_left if side == "left" else x_right), cy_edge
        normal = (1.0, 0.0)
        component = 0
    else:
        edge_x, edge_y = cx_edge, (y_bottom if side == "bottom" else y_top)
        normal = (0.0, 1.0)
        component = 1

    inner_rect = apply_homography(H, reveal.inner_edge_px)
    inner_mm = rectified_to_facade_mm(inner_rect, origin_rect, mm_per_unit)
    if not np.all(np.isfinite(inner_mm)):
        return Recess(origin="unavailable", reveal_side=side)
    inner_component = float(inner_mm[:, component].mean())
    outer_component = edge_x if component == 0 else edge_y
    width_mm = abs(outer_component - inner_component)
    if width_mm <= 0:
        return Recess(origin="unavailable", reveal_side=side)

    gsd_local = _gsd_near(gsd_field, image_size,
                          [*mark.corners_px, *reveal.inner_edge_px],
                          label=f"reveal[{side}]")

    try:
        depth = reveal_depth(camera, edge_x, edge_y, width_mm, normal,
                             sigma_px=sigma_px, gsd_mm_per_px=gsd_local)
    except ValueError:
        # Грань видна, но фактический θ_⊥ ниже вычисляемого порога применимости
        # (либо иная вырожденность геометрии) — отказ ловится и становится
        # названным происхождением, а не роняет весь снимок из-за одного проёма.
        return Recess(origin="unavailable", reveal_side=side)

    nx, ny = normal
    u_mm = abs((edge_x - camera.cx) * nx + (edge_y - camera.cy) * ny)
    theta_perp_deg = math.degrees(math.atan(u_mm / (camera.cz + depth)))

    sigma_width_mm = math.sqrt(2.0) * sigma_px * gsd_local
    phi = math.atan(u_mm / camera.cz)
    sigma_u_mm = camera.cz / math.cos(phi) ** 2 * math.radians(REFERENCE_SIGMA_THETA_DEG)
    sigma_cz_mm = REFERENCE_SIGMA_CZ_REL * camera.cz
    sigma_mm = reveal_depth_sigma(width_mm, u_mm, camera.cz, sigma_width_mm,
                                  sigma_u_mm, sigma_cz_mm)

    return Recess(
        value_mm=depth,
        sigma_mm=sigma_mm,
        origin="measured_from_reveal",
        datum=MEASURED_DATUM,
        reveal_side=side,
        theta_perp_deg=theta_perp_deg,
        consistency="unchecked",
    )


def _digitize_one(mark: ElementMark, index: int, *, H, camera: CameraOnPlane,
                   mm_per_unit: float, origin_rect, image_size, gsd_field: np.ndarray,
                   sigma_px: float, sigma_rel: float, residual_px, theta_max_deg: float,
                   wall_flatness_mm: float) -> Element:
    label = f"{mark.class_}[{index}]"
    corners_px = np.asarray(mark.corners_px, dtype=float)

    visible = _in_front_of_camera(H, image_size, corners_px)
    if not bool(np.all(visible)):
        raise ValueError(
            f"{label}: элемент лежит за линией схода плоскости фасада — измерение "
            "недоступно (спецификация, п. 2.2)")

    contour_rect = apply_homography(H, corners_px)
    contour_mm = rectified_to_facade_mm(contour_rect, origin_rect, mm_per_unit)
    if not np.all(np.isfinite(contour_mm)):
        raise ValueError(f"{label}: контур не переводится в миллиметры фасада")

    centroid = contour_mm.mean(axis=0)
    cx_mm, cy_mm = float(centroid[0]), float(centroid[1])
    # Угол визирования В ТОЧКЕ ЭЛЕМЕНТА, тем же путём, каким сборка конвейера
    # считает поле углов (`run.py:_theta_at`, `camera.theta_deg`). Не θ_cam и не
    # сводка поля кадра — задание 18, требование 3.
    theta_full_deg = camera.theta_deg(cx_mm, cy_mm)
    if theta_full_deg > theta_max_deg:
        raise ValueError(
            f"{label}: угол визирования {theta_full_deg:.1f}° превышает порог "
            f"{theta_max_deg:.1f}°: элемент вне пригодной области (спецификация, "
            "п. 2.2)")

    tan_x, tan_y = camera.tan_theta(cx_mm, cy_mm)
    theta = ThetaDeg(x_deg=math.degrees(math.atan(tan_x)),
                     y_deg=math.degrees(math.atan(tan_y)),
                     full_deg=theta_full_deg)

    width_mm, height_mm = _quad_size_mm(contour_mm)
    gsd_local = _gsd_near(gsd_field, image_size, corners_px, label=label)
    residual_mm = (residual_px * gsd_local if residual_px is not None
                   else RESIDUAL_REFERENCE_MM)

    sigma_width = _size_sigma_mm(width_mm, theta_full_deg, gsd_local, sigma_px,
                                 sigma_rel, residual_mm, wall_flatness_mm)
    sigma_height = _size_sigma_mm(height_mm, theta_full_deg, gsd_local, sigma_px,
                                  sigma_rel, residual_mm, wall_flatness_mm)
    meets_tolerance = max(sigma_width, sigma_height) <= ASSISTED_SIZE_TOLERANCE_MM

    recess = None
    if mark.reveal is not None:
        recess = _digitize_recess(mark, contour_mm, H, camera, mm_per_unit, origin_rect,
                                  gsd_field, image_size, sigma_px)

    return Element(
        id=_element_id(mark.class_, index),
        class_name=mark.class_,
        mounting=mark.mounting,
        edge_reference="wall_plane",
        edge_type=mark.edge_type,
        contour_mm=[(float(x), float(y)) for x, y in contour_mm],
        theta=theta,
        size_mm=SizeMM(width=width_mm, height=height_mm,
                       sigma_width=sigma_width, sigma_height=sigma_height),
        origin="operator",
        recess=recess,
        meets_tolerance=bool(meets_tolerance),
    )


def digitize_elements(marks: list[ElementMark], *, H, camera: CameraOnPlane,
                       mm_per_unit: float, origin_rect, image_size,
                       sigma_px: float, sigma_rel: float, residual_px,
                       theta_max_deg: float = THETA_MAX_DEG,
                       wall_flatness_mm: float = WALL_FLATNESS_DEVIATION_MM,
                       gsd_shape=FIELD_SHAPE) -> list[Element]:
    """Разметка оператора -> список `Element` с миллиметрами и погрешностью.

    `H`, `camera`, `mm_per_unit`, `origin_rect`, `image_size` — та же геометрия,
    что сборка конвейера (`pipeline.run.process`) уже вычислила и записала в выход;
    вызывающий обязан передать ИМЕННО её, а не оценивать заново (докстринг модуля,
    требование 1 задачи 18). `sigma_px` — точность указания точки оператором,
    `sigma_rel` — σ масштаба (уже посчитанная сборкой из длины опорной базы),
    `residual_px` — невязка оценки плоскости в пикселях (`PlaneConfidence.residual_px`
    либо `None` на ручном пути `manual_four_point`).

    Отказ по ОДНОМУ элементу (за линией схода либо вне углового условия,
    требование 4 задачи 18) поднимает `ValueError`, называющий этот элемент, и
    останавливает весь вызов: по умолчанию в этом модуле нет частичного пропуска
    негодных меток — оператор поправляет разметку и запускает оцифровку заново
    (тем же принципом, каким `process()` целиком отказывает на негодном входе).
    """
    gsd_field = local_gsd_field(H, mm_per_unit, image_size, shape=gsd_shape).gsd
    return [
        _digitize_one(mark, index, H=H, camera=camera, mm_per_unit=mm_per_unit,
                      origin_rect=origin_rect, image_size=image_size, gsd_field=gsd_field,
                      sigma_px=sigma_px, sigma_rel=sigma_rel, residual_px=residual_px,
                      theta_max_deg=theta_max_deg, wall_flatness_mm=wall_flatness_mm)
        for index, mark in enumerate(marks)
    ]
