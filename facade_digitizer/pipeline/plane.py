"""Оркестрация плоскости фасада с передачей управления оператору. Спецификация, п. 4.2.

Шаг собирает готовые части — детектор отрезков, оценку точек схода с мерой доверия и
построение гомографии — в одно решение: **строить приведение к плоскости или звать
оператора**. Ничего нового здесь не вычисляется, и в этом смысл: решение должно
приниматься в одном месте, а не расползаться по вызывающему коду.

**Почему результат не несёт позы.** Восстановление позы требует масштаба
ректифицированных координат — числа миллиметров на единицу, — а оно становится
известно только после того, как оператор задал опорный размер. Поэтому наружу
выдаются гомография и точки схода, а позу считает сборка конвейера, где масштаб уже
есть. Выдумать масштаб здесь значило бы выдать за измерение то, что измерением не
является.

**Ниже порога доверия система не гадает.** Это контракт, а не диагностика: при низком
доверии `H` равна `None`, признак `needs_operator` поднят, а причины взяты у метрики
доверия дословно. Единичная гомография вместо `None` запрещена — вызывающий код не
отличил бы её от настоящей и ректифицировал бы снимок тождественным преобразованием,
получив «результат» без единого признака беды.
"""
from dataclasses import dataclass, field

import numpy as np

from facade_digitizer.geometry.homography import (
    homography_from_four_points,
    homography_from_vanishing_points,
)
from facade_digitizer.geometry.vanishing import (
    CONFIDENCE_THRESHOLD,
    PlaneConfidence,
    detect_segments,
    estimate_vanishing_points,
)

METHOD_VANISHING_POINTS = "vanishing_points"
METHOD_MANUAL_FOUR_POINT = "manual_four_point"


@dataclass(frozen=True)
class ManualPlaneConfidence:
    """Доверие ручного пути: ОБЪЯВЛЕННОЕ оператором, а не измеренное.

    Отдельный тип, а не заполненный `PlaneConfidence`, потому что соглашение
    «поля доверия объясняют, а не решают» обязано держаться типом, а не
    комментарием в коде.

    Величины, которых ручной путь не измерял, равны `None`, и это не украшение.
    Нуль невязки и ровно 90° — правдоподобные ИЗМЕРЕННЫЕ значения: потребитель не
    отличил бы их от настоящих. `NaN` отличим глазом, но не охраной — сравнение с
    ним ложно, и проверка вида `residual_px > X` пропустила бы ручной путь молча,
    ровно как нечисловое разрешение в задаче 9 давало вердикт «годен» без единой
    причины. `None` пропустить молча нельзя: `None > 0.5` возбуждает `TypeError`,
    и охрана, которой ручной путь не по зубам, падает громко и сразу.

    `value = 1.0` — не измерение, а запись того, что плоскость задал оператор:
    звать его второй раз некуда. Поддержки и покрытие нулевые, потому что отрезков
    не искали и поддерживать гипотезу нечему.
    """

    value: float
    support_h: int
    support_v: int
    residual_px: float | None
    orthogonality_deg: float | None
    coverage: float
    reasons: list[str] = field(default_factory=list)


def _manual_confidence() -> ManualPlaneConfidence:
    """Свежий экземпляр на каждый вызов.

    Общая на модуль константа была бы изменяемой через `reasons`: один потребитель
    дописал бы причину в список, и она появилась бы у всех последующих результатов,
    включая уже отданные.
    """
    return ManualPlaneConfidence(value=1.0, support_h=0, support_v=0,
                                 residual_px=None, orthogonality_deg=None,
                                 coverage=0.0, reasons=[])


@dataclass(frozen=True)
class PlaneResult:
    """Плоскость фасада: приведение, доверие к нему и происхождение.

    `H` равна `None` ровно тогда, когда поднят `needs_operator`: отказ и отсутствие
    гомографии — одно и то же событие, записанное дважды, чтобы потребитель не мог
    прочитать только одно из двух.

    `vh` и `vv` — однородные векторы точек схода, а не объекты `VanishingPoint`:
    сборка конвейера передаёт их в `camera_pose`, который ждёт векторы. При отказе
    они равны `None` вместе с `H` — точки схода, которым не доверяют, наружу не
    выходят, иначе отказ можно было бы обойти, взяв их напрямую.
    """

    H: np.ndarray | None
    confidence: PlaneConfidence | ManualPlaneConfidence
    method: str
    needs_operator: bool
    vh: np.ndarray | None = None
    vv: np.ndarray | None = None


def estimate_plane(image, K, min_confidence: float = CONFIDENCE_THRESHOLD) -> PlaneResult:
    """Плоскость фасада по точкам схода, с отказом вместо догадки.

    Порог `min_confidence` решает, а не украшает: при доверии ниже порога гомография
    не строится вовсе. Сверх порога связывает и НАЗВАННАЯ причина: список `reasons`
    непуст только тогда, когда метрика доверия отвергла вход по существу, и такой
    вход отвергается при любом пороге, в том числе при нулевом. Иначе вызывающий код
    отключал бы контракт п. 4.2 одним аргументом `min_confidence=0.0`, а гомография
    строилась бы по точкам схода, которые метрика уже объявила негодными.

    Исключения из `homography_from_vanishing_points` (вырожденная ориентация осей,
    центр кадра за линией схода) намеренно НЕ превращаются в `needs_operator`: это
    отказ с названной причиной, а не низкое доверие, и заглушить его признаком
    доверия значило бы выдать геометрическое вырождение за нехватку измерений.
    """
    segments = detect_segments(image)          # он же проверяет вид изображения
    image_size = (image.shape[1], image.shape[0])

    vh, vv, confidence = estimate_vanishing_points(segments, image_size, K)

    if confidence.reasons or confidence.value < min_confidence:
        return PlaneResult(H=None, confidence=confidence,
                           method=METHOD_VANISHING_POINTS, needs_operator=True)

    H = homography_from_vanishing_points(vh.point, vv.point, K, image_size)
    return PlaneResult(H=H, confidence=confidence, method=METHOD_VANISHING_POINTS,
                       needs_operator=False, vh=vh.point, vv=vv.point)


def estimate_plane_manual(image_pts, aspect_ratio=None, size_mm=None,
                          assume_calibrated=False, K=None, image_size=None) -> PlaneResult:
    """Плоскость по четырём точкам оператора плюс одно доопределение.

    Четырёх точек мало: они задают гомографию лишь с точностью до подобия целевого
    четырёхугольника. Спецификация п. 4.2 перечисляет три способа доопределения —
    отношение сторон, два размера в миллиметрах, подтверждение калибровки, — и все
    три передаются в `homography_from_four_points` как есть, вместе с его отказами.
    Подставлять доопределение по умолчанию запрещено: оператор, не задавший размер,
    получил бы выдуманный вместо вопроса.

    `needs_operator` опущен по построению: оператор уже вмешался, и звать его
    второй раз некуда.
    """
    H = homography_from_four_points(image_pts, aspect_ratio=aspect_ratio,
                                    size_mm=size_mm,
                                    assume_calibrated=assume_calibrated,
                                    K=K, image_size=image_size)
    return PlaneResult(H=H, confidence=_manual_confidence(),
                       method=METHOD_MANUAL_FOUR_POINT, needs_operator=False)
