"""Меж-разметчиковая согласованность: сведение экспортов. План 3, задача 26; п. 8.2.

Каждый разметчик размечает те же снимки сам и экспортирует JSON. Сравнивать
миллиметры разных файлов напрямую нельзя: базу и начало отсчёта каждый ставит
свои, и различие баз смешалось бы с различием разметки. Поэтому кромки всех
разметчиков переводятся в миллиметры **одной общей геометрией** — гомографией и
масштабом файла-эталона (`reference`), а в сравнение идут пиксели кадра
(`contour_px`), которые от базы не зависят.

Кромки проёма: левая и правая (x по средним соответствующих углов), нижняя и
верхняя (y). Элементы разных разметчиков сопоставляются по центру контура в
кадре (`MATCH_RADIUS_PX`); проём, не найденный хотя бы у одного, в согласие не
входит и считается отдельно — пропуск проёма тоже расхождение.

**Выбор кромки** (рама вместо кромки стены, п. 5.3) отдельного поля не имеет и
проявляется как выброс положения одной кромки: расхождение с медианой по
разметчикам больше `CHOICE_THRESHOLD_MM` считается расхождением в выборе, а не
в точности указания. Порог — половина типового выноса рамы в четверти (п. 5.2).
"""
from dataclasses import dataclass

import numpy as np

from facade_digitizer.geometry.homography import apply_homography, rectified_to_facade_mm
from facade_digitizer.metrics import (
    MultiRaterAgreement,
    disagreement_fraction,
    fleiss_kappa,
    limits_of_agreement,
)

MATCH_RADIUS_PX = 50.0
CHOICE_THRESHOLD_MM = 40.0
EDGES = ("left", "right", "bottom", "top")


def edges_mm(corners_px, reference: dict) -> dict[str, float]:
    """Кромки проёма в мм фасада по геометрии файла-эталона."""
    record = reference["images"][0]["homography"]
    raster = apply_homography(np.asarray(record["H"], dtype=float),
                              np.asarray(corners_px, dtype=float))
    mm = rectified_to_facade_mm(raster, record["origin_rect_px"],
                                reference["facade"]["mm_per_rectified_px"])
    bl, br, tr, tl = mm
    return {"left": (bl[0] + tl[0]) / 2, "right": (br[0] + tr[0]) / 2,
            "bottom": (bl[1] + br[1]) / 2, "top": (tl[1] + tr[1]) / 2}


def _centre(element) -> np.ndarray:
    return np.asarray(element["contour_px"][0]["points"], dtype=float).mean(axis=0)


@dataclass(frozen=True)
class MatchedItem:
    elements: tuple        # по элементу каждого разметчика


def match_elements(models: list[dict]) -> tuple[list[MatchedItem], int]:
    """Проёмы, найденные у ВСЕХ разметчиков; второе — число не найденных у кого-то."""
    base = models[0]["elements"]
    items, missing = [], 0
    for element in base:
        c = _centre(element)
        found = [element]
        for other in models[1:]:
            candidates = [(np.linalg.norm(_centre(e) - c), e) for e in other["elements"]]
            best = min(candidates, key=lambda t: t[0], default=(np.inf, None))
            if best[0] > MATCH_RADIUS_PX:
                break
            found.append(best[1])
        if len(found) == len(models):
            items.append(MatchedItem(tuple(found)))
        else:
            missing += 1
    total = max(len(m["elements"]) for m in models)
    return items, missing + max(0, total - len(base))


@dataclass(frozen=True)
class AgreementReport:
    items: int
    missing: int
    edges: dict            # кромка → MultiRaterAgreement
    kappa_edge_type: float | None
    kappa_mounting: float | None
    edge_type_disagreement: float
    choice_disagreement: float


def _kappa(labels) -> float | None:
    try:
        return fleiss_kappa(labels)
    except ValueError:
        return None          # все согласны на одной категории: κ не определена


def _collect(models: list[dict], reference: dict | None):
    if len(models) < 2:
        raise ValueError("для согласия нужно не менее двух разметчиков")
    reference = reference or models[0]
    items, missing = match_elements(models)
    table = {edge: np.array([[edges_mm(e["contour_px"][0]["points"], reference)[edge]
                              for e in item.elements] for item in items]).reshape(
                                  len(items), len(models))
             for edge in EDGES}
    edge_types = [[e["edge_type"] for e in item.elements] for item in items]
    mountings = [[e["mounting"] for e in item.elements] for item in items]
    return table, edge_types, mountings, missing


def _report(table, edge_types, mountings, missing) -> AgreementReport:
    n = table[EDGES[0]].shape[0]
    if n < 2:
        raise ValueError(f"сопоставлено проёмов {n}: для пределов согласия мало")
    choice = np.zeros(n, dtype=bool)
    for values in table.values():
        choice |= (np.abs(values - np.median(values, axis=1, keepdims=True))
                   > CHOICE_THRESHOLD_MM).any(axis=1)
    edges: dict[str, MultiRaterAgreement] = {e: limits_of_agreement(v)
                                             for e, v in table.items()}
    return AgreementReport(
        items=n, missing=missing, edges=edges,
        kappa_edge_type=_kappa(edge_types), kappa_mounting=_kappa(mountings),
        edge_type_disagreement=disagreement_fraction(edge_types),
        choice_disagreement=float(choice.mean()))


def agreement(models: list[dict], reference: dict | None = None) -> AgreementReport:
    """Согласие по одному снимку; геометрия — у `reference` (по умолчанию первый)."""
    return _report(*_collect(models, reference))


def agreement_many(groups: list[list[dict]]) -> AgreementReport:
    """Согласие по нескольким снимкам: у каждого снимка своя эталонная геометрия
    (первый разметчик), таблицы кромок объединяются по проёмам."""
    tables, edge_types, mountings, missing = {e: [] for e in EDGES}, [], [], 0
    for models in groups:
        table, types, mounts, miss = _collect(models, None)
        for edge in EDGES:
            tables[edge].append(table[edge])
        edge_types += types
        mountings += mounts
        missing += miss
    return _report({e: np.vstack(v) for e, v in tables.items()}, edge_types,
                   mountings, missing)
