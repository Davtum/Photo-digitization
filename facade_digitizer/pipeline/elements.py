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
расхождение было бы невидимо. По той же причине модуль НЕ считает поле локального
разрешения сам: `gsd_field` передаётся вызывающим — тем самым полем, по которому
он же записал `quality.gsd_mm_px_min/max` этого же файла. Второй, независимо
посчитанный экземпляр поля совпадал бы с первым ровно до тех пор, пока кто-нибудь
не изменит сетку у одного из двух. Растр, который сохраняет CLI
(`--save-rectified`), служит оператору для ПРОСМОТРА и не является носителем
измерения.

**Точность указания точки здесь СВОЯ.** `sigma_px` этого модуля — точность клика по
углу проёма, а не по концам опорной базы масштаба: это разные действия оператора,
выполняемые на разном увеличении, и сливать их в одну величину нельзя (см.
`run.py:DEFAULT_MARK_SIGMA_PX` против `run.py:DEFAULT_OPERATOR_SIGMA_PX`).

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

**Чем эта величина является и чем не является.** Доказано следующее:
`REFERENCE_SIGMA_THETA_DEG` — P95 измеренного распределения ошибки направления на
разреженной сцене, то есть она мажорирует типичный случай и на обогащённых фасадах
завышает σ_u (а с ним и σ_depth) примерно в 7.6 раза (0.344° / 0.045°) против
достижимого там запаса. Границей ошибки она при этом НЕ является: P95 по
построению оставляет двадцатую часть выборки выше себя, и сама выборка собрана
ПОСЛЕ отбраковки шлюзом доверия, тогда как п. 4.2 прямо говорит, что доверие
предсказателем точности не является — то есть шлюз недостающей границы не
восстанавливает. Утверждение «не занижает нигде» из этого не следует и снято;
верно лишь то, что величина консервативна в типичном случае и что более сильного
утверждения измерениями проекта не обеспечено.
"""
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from facade_digitizer.geometry.angles import THETA_MAX_DEG, nearest_node
from facade_digitizer.geometry.camera import CameraOnPlane
from facade_digitizer.geometry.homography import apply_homography, rectified_to_facade_mm
from facade_digitizer.geometry.parallax import (
    REFERENCE_SIGMA_THETA_DEG,
    reveal_depth,
    reveal_depth_sigma,
    signed_reveal_width_mm,
    visible_reveal_side,
)
from facade_digitizer.schema import Element, Point, Recess, SizeMM, Strict, ThetaDeg

#: Цель п. 2.2 для режима `assisted`: 1σ ≤ 10 мм на габарит по наружному контуру.
#: `meets_tolerance` сравнивает с этим значением ПОСЧИТАННУЮ σ (RSS слагаемых
#: п. 6.1 при фактических угле и GSD), а не берёт готовый ответ по режиму — ровно
#: то, что требует спецификация («`meets_tolerance` вычисляется по фактическому
#: углу, а не по режиму», п. 6.1).
ASSISTED_SIZE_TOLERANCE_MM = 10.0

#: Неплоскостность стены, п. 6.1: диапазон ±20–50 мм, эффект = отклонение·tg θ.
#:
#: **Это ДОПУЩЕНИЕ, а не измерение**, и поэтому оно названо и перекрывается
#: (`digitize_elements(wall_flatness_mm=...)`, CLI `--wall-flatness-mm`). Все
#: прочие слагаемые п. 6.1 берутся по факту: разрешение — измеренное поле, невязка —
#: измеренная невязка, угол — фактический угол ЭТОГО элемента. Неплоскостность
#: стены конвейер не измеряет вовсе, и единственное, что он может сделать честно, —
#: назвать принятую величину и дать её заменить тому, кто про эту стену знает
#: больше.
#:
#: **Умолчание ВЫВЕДЕНО из собственных утверждений п. 6.1**, а не выбрано из
#: диапазона на глаз. Спецификация делает про этот бюджет два количественных
#: заявления, и оба обязаны воспроизводиться:
#:
#: 1. RSS режима `assisted` равна 6.0–13.7 мм при θ ≤ 8°. Нижний край берётся по
#:    нижним концам ВСЕХ строк таблицы (σ_px 0.5, σ_rel 0.14 %, невязка 3 мм,
#:    дисторсия 2.5 мм, неплоскостность 20 мм), верхний — по верхним (σ_px 1,
#:    σ_rel 0.28 %, невязка 7 мм, дисторсия 5 мм, неплоскостность 50 мм).
#:    Пересчитано: 6.33 и 13.82 мм — ветви опознаны верно.
#: 2. «Цель 1σ ≤ 10 мм достижима при θ ≲ 15–20°», и диапазон до 30°
#:    СОХРАНЯЕТСЯ как рабочий.
#:
#: Заявление 2 достижимо только на благоприятной ветви заявления 1 (на
#: неблагоприятной σ превышает 10 мм уже при θ = 0). Считается эта ветвь
#: КОНСТАНТАМИ САМОГО МОДУЛЯ, а не нижними концами таблицы: дисторсия у модуля
#: взята по верхнему краю (`RESIDUAL_DISTORTION_MM` = 5 мм, см. там же), и
#: выводить отсюда отклонение по её нижнему краю значило бы вывести его из
#: бюджета, которого модуль не реализует. С σ_px 0.5, GSD 5, σ_rel 0.14 %,
#: невязкой 3 мм и дисторсией 5 мм остальные слагаемые дают 7.14 мм, на
#: неплоскостность остаётся 7.01 мм бюджета, и угол переключения равен
#: `arctg(7.01 / отклонение)`. Отсюда:
#:
#:     θ* = 20° требует отклонения 19.25 мм
#:     θ* = 15° требует отклонения 26.15 мм
#:
#: Взято **25 мм** — внутри этого отрезка и строго выше оптимистического края
#: диапазона ±20–50: угол переключения выходит 15.66°, то есть собственный вывод
#: п. 6.1 о рабочем диапазоне воспроизводится.
#:
#: Почему не 50 мм, как было. Это верхний край диапазона, выбор в безопасную
#: сторону, и цена его измерена: слагаемое `50·tg θ` даёт 96 % дисперсии на
#: крутом ракурсе (1.2 % на пологом), то есть `meets_tolerance` переставал
#: вычисляться «по фактическому углу» и начинал вычисляться по константе модуля,
#: умноженной на тангенс фактического угла. Угол переключения при 50 мм равен
#: 7.98° — вне рабочего диапазона, который спецификация прямо сохраняет.
#: Почему не 20 мм: это оптимистический край, и брать его без измерения значило
#: бы занижать σ.
#:
#: Величина перекрывается, и это существенно: 25 мм — допущение о стене вообще,
#: а не измерение ЭТОЙ стены. Тому, кто про свою стену знает больше, ключ
#: `--wall-flatness-mm` даёт подставить измеренное.
WALL_FLATNESS_DEVIATION_MM = 25.0

#: Остаток бюджета допуска п. 2.2 на слагаемое неплоскостности при благоприятной
#: ветви остальных диапазонов п. 6.1 и константах этого модуля (вывод — см.
#: `WALL_FLATNESS_DEVIATION_MM`). Угол переключения `meets_tolerance` равен
#: `arctg(WALL_FLATNESS_BUDGET_MM / wall_flatness_mm)`. Вынесен в имя, чтобы
#: проверка воспроизводимости вывода не переписывала число у себя.
WALL_FLATNESS_BUDGET_MM = 7.0064

#: Остаточная дисторсия, п. 6.1: 2.5–5 мм, одинаково в обеих колонках таблицы.
#:
#: Слагаемое называется ОСТАТОЧНЫМ: оно описывает то, что остаётся ПОСЛЕ снятия
#: дисторсии, и стоит в колонке `assisted`, где калибровка предполагается. Ссылка
#: на `calib.undistort` его не снимает — тот шаг и есть коррекция, после которой
#: этот остаток назван; более того, при нулевых коэффициентах `undistort`
#: возвращает исходный массив, то есть не делает ничего вовсе.
#:
#: Взята ВЕРХНЯЯ граница, в отличие от `WALL_FLATNESS_DEVIATION_MM`, и различие не
#: произвольно: здесь диапазон узок (2.5 мм размаха против 30) и, главное, не
#: умножается на тангенс угла, поэтому слагаемое не способно ни доминировать в
#: дисперсии, ни сделать `meets_tolerance` функцией одной константы. Цена
#: консервативного края — не более 1.9 мм в квадратичной сумме.
RESIDUAL_DISTORTION_MM = 5.0

#: Относительная ошибка фокусного расстояния без калибровки по мишени, п. 6.1
#: (ε = 2 %). Эффект `ε·L·sin²θ`: на полутораметровом окне 7.5 мм при θ = 30° и
#: 15 мм при θ = 45° — ровно те числа, что названы в таблице.
FOCAL_RELATIVE_ERROR = 0.02

#: Ошибка фокусного по ПРОИСХОЖДЕНИЮ матрицы K (`camera.calibration` выходного
#: файла). Таблица п. 6.1 различает два состояния — «при калибровке» и «без
#: калибровки», — и калибровкой она называет разовую калибровку по мишени
#: (п. 4.2: «камеры собственного парка калибруются по мишени, а не по EXIF»).
#: Поэтому нулю равна только строка `target`. Фокусное из EXIF — номинальная
#: величина объектива, а не измеренная у этого экземпляра, и основания считать её
#: точной у конвейера нет; типовое поле зрения (`database`) тем более. Обе
#: получают ε таблицы.
#:
#: Источник K есть свойство СНИМКА и записан в выход именно затем, чтобы этот
#: вклад можно было оценить (докстринг `calib.intrinsics_from_meta`). Задача 18 —
#: первый его потребитель.
FOCAL_ERROR_BY_CALIBRATION = {
    "target": 0.0,
    "exif": FOCAL_RELATIVE_ERROR,
    "database": FOCAL_RELATIVE_ERROR,
}

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

#: Классы `edge_type`, при которых кромка заведомо НЕ лежит в плоскости стены:
#: спецификация, п. 5.3 — «при значении `surround` или `cladding_edge` элемент
#: обрабатывается как выступающий (п. 5.6) и требует `offset_mm`».
_OFFSET_EDGE_TYPES = frozenset({"surround", "cladding_edge"})

#: Способ установки, при котором действует то же правило п. 5.6 независимо от
#: `edge_type`: «для карниза, цоколя, выступающих окон, подоконников и обрамлений
#: d < 0 и видимая кромка не лежит в Π».
_OFFSET_MOUNTING = "protruding"

_ID_PREFIX = {"window": "w", "door": "d"}


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

    **`facade_boundary` этот формат не принимает.** Спецификация, п. 2.3 задаёт
    границу фасада ПОЛИЛИНИЕЙ с атрибутом выноса `offset_mm`, а не
    четырёхугольником: у вертикального ребра, линии кровли и линии цоколя нет ни
    ширины, ни высоты. Выпустить её отсюда значило бы приписать полилинии габарит
    с σ и объявить его в допуске — величины, которых п. 2.3 для этого класса не
    определяет вовсе, — да ещё и без `offset_mm`, которого нет и в схеме
    (оценка выноса — следующий этап, см. п. 5.6).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_: Literal["window", "door", "facade_boundary"] = Field(alias="class")
    mounting: Literal["embedded", "flush", "protruding"]
    edge_type: Literal["sharp_wall_edge", "surround", "cladding_edge", "unknown"]
    corners_px: list[Point]
    reveal: RevealMark | None = None

    @model_validator(mode="after")
    def _openings_only(self):
        if self.class_ == "facade_boundary":
            raise ValueError(
                "класс facade_boundary этой разметкой не оцифровывается: "
                "спецификация, п. 2.3 задаёт границу фасада полилинией с "
                "атрибутом выноса offset_mm, а не четырёхугольником, и габарита "
                "с допуском у неё нет; оценка выноса — следующий этап (п. 5.6)")
        return self

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

    Пустой список отвергается отдельно и по имени. Молча приравняв его к
    отсутствию разметки, конвейер уравнял бы два разных указания оператора: «я
    ничего не размечал» и «я передал файл разметки». Разница не косметическая —
    на пути `needs_operator` непустая разметка обязана поднять отказ, а пустая
    прошла бы насквозь, и оператор получил бы файл без единого элемента и без
    единого слова о том, почему.
    """
    # ValueError, а не TypeError (вопреки TRY004): негодная разметка --marks
    # обязана ловиться той же веткой, что и остальные отказы валидации входа
    # конвейера (`main()` перехватывает `(ValueError, FileNotFoundError)`), а не
    # завести для формата разметки отдельный тип исключения.
    if not isinstance(raw, list):
        raise ValueError(  # noqa: TRY004
            f"разметка --marks обязана быть JSON-списком элементов, получено "
            f"{type(raw).__name__}")
    if not raw:
        raise ValueError(
            "разметка --marks пуста: список без единого элемента неотличим от "
            "отсутствия разметки; уберите ключ, если размечать нечего")
    return [ElementMark.model_validate(item) for item in raw]


def load_marks(path) -> list[ElementMark]:
    """Файл `--marks` в список `ElementMark`."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_marks(json.loads(text))


def _gsd_at_point(gsd_field: np.ndarray, image_size, point_px) -> float:
    """Разрешение в ближайшем к точке узле поля.

    Узел выбирает `geometry.angles.nearest_node` — ТА ЖЕ функция, какой его
    выбирает сборка конвейера (`pipeline.run._gsd_at`). Прежде формула была
    выписана здесь вторично, со ссылкой на цикл импорта как на причину; цикла
    нет — оба модуля уже импортируют `geometry.angles`, — и общее объявление
    переехало туда, к самой сетке `FIELD_SHAPE`.
    """
    rows, cols = gsd_field.shape
    w_px, h_px = image_size
    col = nearest_node(float(point_px[0]), w_px, cols)
    row = nearest_node(float(point_px[1]), h_px, rows)
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


def _points_in_frame(points_px, image_size, label: str) -> None:
    """Точки разметки обязаны лежать в кадре. Тот же контракт, что `run._points_in_frame`.

    Оператор кликает ПО СНИМКУ; точка вне кадра означает, что переданы координаты
    из другой системы либо, что реальнее, от другого снимка — `--marks` читается
    один раз и подаётся на весь пакет. Перенесённая на чужой кадр разметка без
    этой охраны даёт уверенные миллиметры для окна, которого там нет: индекс
    ближайшего узла упирается в край сетки, и ни одна величина не становится ни
    бесконечной, ни отрицательной.
    """
    w_px, h_px = image_size
    for index, point in enumerate(points_px):
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{label}: точка разметки [{index}] не определена "
                             f"({x}, {y})")
        if not (0.0 <= x <= w_px - 1 and 0.0 <= y <= h_px - 1):
            raise ValueError(
                f"{label}: точка разметки [{index}] = ({x:.1f}, {y:.1f}) лежит "
                f"вне кадра {w_px}x{h_px}: оператор указывает точки на снимке, и "
                "разметка привязана к тому снимку, на котором сделана")


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


def _edge_reference(mark: ElementMark) -> str:
    """К чему отнесён измеренный контур. Спецификация, пп. 5.3, 5.6.

    Три исхода, и ни один из них не выдумывает выноса:

    * кромка острая и элемент не выступает — п. 5.3 применима, контур лежит в
      плоскости стены, `"wall_plane"`;
    * `surround` / `cladding_edge` либо `mounting = "protruding"` — п. 5.3 сама
      называет эти случаи нарушением своего допущения, элемент обрабатывается как
      выступающий (п. 5.6), и кромка отстоит от Π на НЕ ИЗМЕРЕННЫЙ вынос:
      `"offset_plane"`;
    * `edge_type = "unknown"` — про кромку не известно ничего, и сказать, лежит
      она в Π или нет, не на чем: `"unknown"`.

    Порядок проверок не произволен: `mounting = "protruding"` решает раньше
    `edge_type = "unknown"`, потому что «элемент выступает» — утверждение более
    сильное, чем «род кромки не распознан», и оно верно независимо от рода.
    """
    if mark.edge_type in _OFFSET_EDGE_TYPES or mark.mounting == _OFFSET_MOUNTING:
        return "offset_plane"
    if mark.edge_type == "unknown":
        return "unknown"
    return "wall_plane"


def _size_sigma_mm(size_mm: float, theta_deg: float, gsd_local: float, sigma_px: float,
                    sigma_rel: float, residual_mm: float, wall_flatness_mm: float,
                    distortion_mm: float, focal_rel: float) -> float:
    """σ габарита: сумма RSS слагаемых п. 6.1 при ФАКТИЧЕСКИХ угле и GSD.

    Слагаемые и их источник в спецификации — все шесть строк таблицы п. 6.1,
    кроме последней (параллакс), которая не разброс, а смещение (см. ниже):

    1. **Локализация двух кромок** — `√2 · σ_px · GSD_local`. Габарит есть разность
       положений двух независимо кликнутых кромок (левой/правой либо верхней/
       нижней), поэтому несёт множитель √2 правила «двухкромочной величины»
       (п. 6.1, та же логика, что и у ширины откоса в п. 6.2). `GSD_local` —
       измеренное локальное разрешение У ЭТОГО проёма (`_gsd_near`), а не табличное
       значение режима. `σ_px` — точность клика ПО УГЛУ ПРОЁМА, своя величина
       (докстринг модуля).
    2. **Масштаб** — `size_mm · sigma_rel`. `sigma_rel` уже посчитана СБОРКОЙ
       конвейера из длины опорной базы оператора (`run.py:_scale_sigma_rel`,
       задача 14) и передаётся сюда готовой: пересчитывать её заново значило бы
       завести второй источник истины, способный разойтись с `facade.scale.sigma_rel`
       того же выходного файла.
    3. **Остаточная проективная невязка** — `residual_mm`, уже переведённая в
       миллиметры вызывающим кодом (`residual_px · GSD_local`, либо
       `RESIDUAL_REFERENCE_MM`, когда невязка не измерена).
    4. **Остаточная дисторсия** — `distortion_mm`, см. `RESIDUAL_DISTORTION_MM`.
       Строка названа остаточной и описывает то, что остаётся ПОСЛЕ коррекции;
       шаг `calib.undistort` её не снимает.
    5. **Ошибка фокусного** — `focal_rel · size_mm · sin²θ` при фактическом угле.
       `focal_rel` берётся вызывающим по `camera.calibration` того же выходного
       файла (`FOCAL_ERROR_BY_CALIBRATION`): при калибровке по мишени слагаемое
       обращается в нуль само, без отдельной ветки.
    6. **Неплоскостность стены** — `wall_flatness_mm · tg θ` при ФАКТИЧЕСКОМ угле
       визирования ЭТОГО элемента в его собственной точке (не θ_cam и не сводка
       поля углов кадра — задание 18, требование 3). Сама величина — названное
       допущение, см. `WALL_FLATNESS_DEVIATION_MM`.

    **Чего здесь нет и почему.** Последняя строка таблицы п. 6.1 — параллакс при
    ошибке идентификации кромки, 10–45 мм — в эту сумму не входит, и спецификация
    сама говорит почему: это **смещение, а не разброс**. Складывать систематический
    сдвиг с квадратичной суммой случайных вкладов нельзя, он бы там растворился.
    Случай, в котором эта строка не равна нулю, выражается не числом в σ, а полем
    `edge_reference` (`_edge_reference`) и отсутствием `meets_tolerance`: габарит
    выдаётся, метрическая интерпретация — нет (п. 5.6).

    Сумма — RSS: источники ПРЕДПОЛАГАЮТСЯ независимыми (то же допущение, что и в
    `reveal_depth_sigma`, п. 6.2). Это не консервативно: GSD и невязка происходят
    из одной оценки плоскости и в общем случае коррелированы, а при положительной
    корреляции RSS даёт НИЖНЮЮ оценку σ. Строгий бюджет потребовал бы ковариационной
    матрицы позы (та же оговорка, что в п. 6.2).
    """
    theta_rad = math.radians(theta_deg)
    localisation = math.sqrt(2.0) * sigma_px * gsd_local
    scale = size_mm * sigma_rel
    focal = focal_rel * size_mm * math.sin(theta_rad) ** 2
    flatness = wall_flatness_mm * math.tan(theta_rad)
    return math.hypot(localisation, scale, residual_mm, distortion_mm, focal, flatness)


def _element_id(class_name: str, index: int) -> str:
    return f"{_ID_PREFIX.get(class_name, 'e')}_{index:03d}"


def _worst_corner_theta(camera: CameraOnPlane, contour_mm: np.ndarray):
    """Угол визирования элемента — по ХУДШЕЙ точке контура, а не по центроиду.

    Спецификация, п. 2.2 нормирует пригодную область условием на ТОЧКИ, а не на
    центр элемента: «требование к GSD относится к худшему локальному значению в
    пределах пригодной области». Тем же правилом берёт своё значение `_gsd_near`,
    и брать угол по центроиду рядом с ней было бы непоследовательно: измерено, что
    центроид даёт 29.66° там, где худший угол контура равен 34.46° при пороге 30°,
    то есть проём, четверть которого лежит ВНЕ пригодной области, принимался как
    годный.

    Возвращаются компоненты и полный угол ОДНОЙ И ТОЙ ЖЕ точки: тройка (x, y,
    full), собранная из разных точек, не была бы углом ни одной из них.
    """
    p = np.asarray(contour_mm, dtype=float)
    worst_index, worst_theta = 0, -1.0
    for index, (x, y) in enumerate(p):
        theta = camera.theta_deg(float(x), float(y))
        if theta > worst_theta:
            worst_index, worst_theta = index, theta
    x, y = float(p[worst_index][0]), float(p[worst_index][1])
    tan_x, tan_y = camera.tan_theta(x, y)
    return worst_theta, tan_x, tan_y


def _digitize_recess(mark: ElementMark, contour_mm: np.ndarray, H, camera: CameraOnPlane,
                      mm_per_unit: float, origin_rect, gsd_field: np.ndarray, image_size,
                      sigma_px: float, sigma_rel: float, edge_reference: str) -> Recess:
    """Глубина заглубления по разметке внутренней кромки грани откоса. П. 5.4-5.5, 6.2.

    Исходы, каждый — с названным происхождением, а не тихим числом:

    * `edge_reference != "wall_plane"` — наружная кромка, от которой отсчитывалась
      бы глубина, заведомо не лежит в плоскости стены либо про неё не известно
      ничего. `origin = "unavailable"`;
    * оператор разметил грань, которая по позе камеры НЕ видна (правило п. 5.5:
      видна дальняя от опорной точки F грань) — `origin = "unavailable"`;
    * внутренняя кромка смещена НЕ В ТУ сторону — `origin = "unavailable"`;
    * грань видна, но фактический θ_⊥ ниже вычисляемого порога применимости
      (`reveal_depth` вызывает `reveal_depth_theta_min_deg` внутри себя, порог не
      константа — п. 6.2) — тоже `"unavailable"`, отказ `reveal_depth` ловится, а
      не роняет весь снимок;
    * всё сошлось — `origin = "measured_from_reveal"` с `value_mm`, `sigma_mm`,
      `theta_perp_deg` и `datum = "quarter_edge"`.

    **Почему `edge_reference` решает здесь, а не только у габарита.** Глубина
    отсчитывается ОТ НАРУЖНОЙ КРОМКИ проёма, и `edge_reference` — это и есть
    утверждение о том, где та кромка лежит. При `"offset_plane"` п. 5.3 сама
    объявляет своё допущение нарушенным: элемент обрабатывается как выступающий
    (п. 5.6) и требует НЕ ИЗМЕРЕННОГО выноса, а `recess` помечается
    `unavailable` (п. 17). При `"unknown"` про кромку не известно ничего, и
    отсчитывать от неё нечего. Прежде эта связь существовала только для
    `edge_reference` и `meets_tolerance`, а `recess` считался безусловно — и один
    и тот же элемент выходного файла утверждал разом «контур не лежит в плоскости
    стены» (`edge_reference: "offset_plane"`) и «глубина измерена по откосу до
    ребра четверти» (`origin: "measured_from_reveal"`, `datum: "quarter_edge"`).

    **Направление смещения внутренней кромки проверяется, а не берётся по
    модулю.** Внутренняя кромка обязана лежать МЕЖДУ наружной кромкой и опорной
    точкой F: при заглублении параллакс сносит её к F, при выносе — от неё
    (`signed_reveal_width_mm`). Разметка «с другой стороны» — отказ, а не число:
    прежде здесь стоял `abs(наружная − внутренняя)`, и истинно выступающий
    элемент (истина −150 мм) выдавал уверенные +154.6 мм в поле, где числу
    полагается быть глубиной. Возвращать вместо этого отрицательную глубину
    нельзя по той же п. 5.6: вынос — отдельная величина `offset_mm` следующего
    этапа, а не заглубление с другим знаком, и `Recess` поля под неё не имеет.
    Второй конец того же неравенства (кромка не перешла ЗА F) проверяет сама
    `reveal_depth` условием `u > w`.

    **Точки грани откоса проходят ту же проверку видимости, что и углы контура.**
    `_in_front_of_camera` прежде применялась только к `corners_px`; узел за линией
    схода даёт КОНЕЧНЫЕ, но бессмысленные миллиметры, поэтому проверка на
    `isfinite` его не ловит.
    """
    reveal = mark.reveal
    side = reveal.side

    if edge_reference != "wall_plane":
        return Recess(origin="unavailable", reveal_side=side)

    p = np.asarray(contour_mm, dtype=float)
    x_left, x_right = float(p[:, 0].min()), float(p[:, 0].max())
    y_bottom, y_top = float(p[:, 1].min()), float(p[:, 1].max())
    cx_edge, cy_edge = float(p[:, 0].mean()), float(p[:, 1].mean())

    visible_vertical, visible_horizontal = visible_reveal_side(
        camera, x_left, x_right, y_bottom, y_top)
    is_vertical = side in ("left", "right")
    expected_side = visible_vertical if is_vertical else visible_horizontal

    if side != expected_side:
        return Recess(origin="unavailable", reveal_side=side)

    if is_vertical:
        edge_x, edge_y = (x_left if side == "left" else x_right), cy_edge
        normal = (1.0, 0.0)
    else:
        edge_x, edge_y = cx_edge, (y_bottom if side == "bottom" else y_top)
        normal = (0.0, 1.0)

    inner_rect = apply_homography(H, reveal.inner_edge_px)
    inner_mm = rectified_to_facade_mm(inner_rect, origin_rect, mm_per_unit)
    if not np.all(np.isfinite(inner_mm)):
        return Recess(origin="unavailable", reveal_side=side)
    inner_x = float(inner_mm[:, 0].mean())
    inner_y = float(inner_mm[:, 1].mean())

    try:
        width_mm = signed_reveal_width_mm(camera, edge_x, edge_y, inner_x, inner_y,
                                          normal)
    except ValueError:
        # Опорная точка села на само ребро откоса: сторона смещения не определена.
        return Recess(origin="unavailable", reveal_side=side)
    if width_mm <= 0:
        # Внутренняя кромка смещена ОТ опорной точки либо не смещена вовсе:
        # элемент не заглублён, и глубины у него нет — есть вынос, величина
        # следующего этапа (п. 5.6).
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
    # σ дистанции до плоскости берётся ИЗМЕРЕННОЙ, а не по опорному допущению
    # `parallax.REFERENCE_SIGMA_CZ_REL` (0.4 %). `camera_pose` кладёт в `cz`
    # ровно `mm_per_unit` (`geometry.homography.camera_pose`), а `mm_per_unit`
    # задан опорным размером оператора: его относительная погрешность — это и
    # есть `facade.scale.sigma_rel` того же выходного файла (на рабочем проходе
    # 0.18-0.23 %). Допущение 0.4 % верно в модуле-источнике, где фактической
    # величины нет; у потребителя, который её знает, оно названо бы не своим
    # происхождением. Влияние мало (0.60 мм из 27 на опорной геометрии), но
    # «мало» — не основание держать в бюджете чужое число.
    sigma_cz_mm = sigma_rel * camera.cz
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
                   wall_flatness_mm: float, distortion_mm: float, focal_rel: float,
                   image_id: str) -> Element:
    label = f"{mark.class_}[{index}]"
    corners_px = np.asarray(mark.corners_px, dtype=float)

    marked_px = list(mark.corners_px)
    if mark.reveal is not None:
        marked_px += list(mark.reveal.inner_edge_px)
    _points_in_frame(marked_px, image_size, label)

    # Проверяются ВСЕ точки разметки, а не только углы контура. Точки грани
    # откоса переводятся в миллиметры той же гомографией и той же строкой кода,
    # что и углы, поэтому за линией схода они так же дают конечные бессмысленные
    # координаты; охрана на `isfinite` в `_digitize_recess` их не ловит. Набор
    # точек здесь тот же, что у `_points_in_frame` выше, — два разных набора у
    # двух охран одного и того же входа расходились бы молча.
    visible = _in_front_of_camera(H, image_size, corners_px)
    if not bool(np.all(visible)):
        raise ValueError(
            f"{label}: элемент лежит за линией схода плоскости фасада — измерение "
            "недоступно (спецификация, п. 2.2)")

    contour_rect = apply_homography(H, corners_px)
    contour_mm = rectified_to_facade_mm(contour_rect, origin_rect, mm_per_unit)
    if not np.all(np.isfinite(contour_mm)):
        raise ValueError(f"{label}: контур не переводится в миллиметры фасада")

    # Угол визирования В ТОЧКЕ ЭЛЕМЕНТА, тем же путём, каким сборка конвейера
    # считает поле углов (`run.py:_theta_at`, `camera.theta_deg`). Не θ_cam и не
    # сводка поля кадра — задание 18, требование 3. Берётся ХУДШАЯ точка контура,
    # тем же правилом, каким `_gsd_near` берёт худшее разрешение.
    theta_full_deg, tan_x, tan_y = _worst_corner_theta(camera, contour_mm)
    if theta_full_deg > theta_max_deg:
        raise ValueError(
            f"{label}: угол визирования {theta_full_deg:.1f}° превышает порог "
            f"{theta_max_deg:.1f}°: элемент вне пригодной области (спецификация, "
            "п. 2.2)")

    theta = ThetaDeg(x_deg=math.degrees(math.atan(tan_x)),
                     y_deg=math.degrees(math.atan(tan_y)),
                     full_deg=theta_full_deg)

    width_mm, height_mm = _quad_size_mm(contour_mm)
    gsd_local = _gsd_near(gsd_field, image_size, corners_px, label=label)
    residual_mm = (residual_px * gsd_local if residual_px is not None
                   else RESIDUAL_REFERENCE_MM)

    sigma_width = _size_sigma_mm(width_mm, theta_full_deg, gsd_local, sigma_px,
                                 sigma_rel, residual_mm, wall_flatness_mm,
                                 distortion_mm, focal_rel)
    sigma_height = _size_sigma_mm(height_mm, theta_full_deg, gsd_local, sigma_px,
                                  sigma_rel, residual_mm, wall_flatness_mm,
                                  distortion_mm, focal_rel)

    edge_reference = _edge_reference(mark)
    # П. 5.6: при отсутствии оценки выноса габарит выдаётся БЕЗ метрической
    # интерпретации. Признак допуска и есть метрическая интерпретация габарита,
    # поэтому у кромки, про которую неизвестно, лежит ли она в плоскости стены,
    # его нет вовсе. Отсутствие, а не `False`: «не проверено» и «проверено и не
    # прошло» — разные утверждения, и второе из них здесь необоснованно, ведь σ
    # разброса допуск как раз проходит; не проходит несведённое СМЕЩЕНИЕ, которое
    # в σ не входит (см. `_size_sigma_mm`).
    meets_tolerance = None
    if edge_reference == "wall_plane":
        meets_tolerance = bool(
            max(sigma_width, sigma_height) <= ASSISTED_SIZE_TOLERANCE_MM)

    recess = None
    if mark.reveal is not None:
        recess = _digitize_recess(mark, contour_mm, H, camera, mm_per_unit, origin_rect,
                                  gsd_field, image_size, sigma_px, sigma_rel,
                                  edge_reference)

    return Element(
        id=_element_id(mark.class_, index),
        class_name=mark.class_,
        mounting=mark.mounting,
        edge_reference=edge_reference,
        edge_type=mark.edge_type,
        contour_mm=[(float(x), float(y)) for x, y in contour_mm],
        contour_px=[{"image_id": image_id,
                     "points": [[float(x), float(y)] for x, y in mark.corners_px]}],
        theta=theta,
        size_mm=SizeMM(width=width_mm, height=height_mm,
                       sigma_width=sigma_width, sigma_height=sigma_height),
        origin="operator",
        recess=recess,
        meets_tolerance=meets_tolerance,
    )


def digitize_elements(marks: list[ElementMark], *, H, camera: CameraOnPlane,
                       mm_per_unit: float, origin_rect, image_size,
                       gsd_field: np.ndarray, sigma_px: float, sigma_rel: float,
                       residual_px, calibration: str,
                       image_id: str = "img_0",
                       theta_max_deg: float = THETA_MAX_DEG,
                       wall_flatness_mm: float = WALL_FLATNESS_DEVIATION_MM,
                       distortion_mm: float = RESIDUAL_DISTORTION_MM) -> list[Element]:
    """Разметка оператора -> список `Element` с миллиметрами и погрешностью.

    `H`, `camera`, `mm_per_unit`, `origin_rect`, `image_size`, `gsd_field` — та же
    геометрия и то же поле разрешения, что сборка конвейера (`pipeline.run.process`)
    уже вычислила и записала в выход; вызывающий обязан передать ИМЕННО их, а не
    оценивать заново (докстринг модуля, требование 1 задачи 18). `gsd_field`
    передаётся, а не считается здесь, по той же причине, по какой здесь нет второй
    реализации гомографии: два независимо посчитанных поля разошлись бы молча, и
    σ элементов описывала бы не тот кадр, про который `quality.gsd_mm_px_max`
    говорит в том же файле.

    `sigma_px` — точность указания ТОЧКИ ПРОЁМА оператором (не точность указания
    опорной базы масштаба, см. докстринг модуля), `sigma_rel` — σ масштаба (уже
    посчитанная сборкой из длины опорной базы), `residual_px` — невязка оценки
    плоскости в пикселях (`PlaneConfidence.residual_px` либо `None` на ручном пути
    `manual_four_point` и там, где невязка неконечна).

    `sigma_rel` входит в ДВА бюджета, а не в один. В бюджете габарита (п. 6.1) это
    строка «масштаб», `L · σ_rel`. В бюджете глубины (п. 6.2) — относительная
    погрешность дистанции до плоскости: `camera_pose` кладёт в `cz` ровно
    `mm_per_unit`, а `mm_per_unit` задан тем же опорным размером оператора, поэтому
    σ_Cz / C_z и есть σ_rel. Опорное допущение `parallax.REFERENCE_SIGMA_CZ_REL`
    (0.4 %) остаётся только там, где фактической величины нет, — во внутреннем
    расчёте порога применимости `reveal_depth`, где и глубина, и допуск тоже
    опорные.

    `calibration` — происхождение матрицы K этого СНИМКА в терминах схемы
    (`camera.calibration`). Аргумент обязательный и без умолчания: от него зависит
    слагаемое ошибки фокусного (п. 6.1), и умолчание здесь означало бы тихо
    выбранный ответ на вопрос, который знает только вызывающий.

    Отказ по ОДНОМУ элементу (вне кадра, за линией схода либо вне углового условия,
    требование 4 задачи 18) поднимает `ValueError`, называющий этот элемент, и
    останавливает весь вызов: по умолчанию в этом модуле нет частичного пропуска
    негодных меток — оператор поправляет разметку и запускает оцифровку заново
    (тем же принципом, каким `process()` целиком отказывает на негодном входе).
    """
    focal_rel = FOCAL_ERROR_BY_CALIBRATION.get(calibration)
    if focal_rel is None:
        raise ValueError(
            f"неизвестное происхождение внутренних параметров: {calibration!r}; "
            "слагаемое ошибки фокусного (п. 6.1) по нему не определено")
    return [
        _digitize_one(mark, index, H=H, camera=camera, mm_per_unit=mm_per_unit,
                      origin_rect=origin_rect, image_size=image_size, gsd_field=gsd_field,
                      sigma_px=sigma_px, sigma_rel=sigma_rel, residual_px=residual_px,
                      theta_max_deg=theta_max_deg, wall_flatness_mm=wall_flatness_mm,
                      distortion_mm=distortion_mm, focal_rel=focal_rel,
                      image_id=image_id)
        for index, mark in enumerate(marks)
    ]
