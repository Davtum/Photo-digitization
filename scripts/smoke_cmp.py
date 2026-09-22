"""Проба геометрического ядра на реальных снимках публичного набора CMP.

Задача 17 плана `2026-09-17-geometric-core`. Скрипт не входит в пакет и ничего не
пишет в репозиторий — только читает снимки из переданной папки и печатает сводку
в stdout. Результат прогона переносится в
`docs/superpowers/plans/results/geometric-core-smoke.md` вручную.

**Что здесь проверяется, а что нет.** CMP — набор уже РЕКТИФИЦИРОВАННЫХ фасадов:
пучки вертикальных и горизонтальных линий на снимке параллельны, точки схода лежат
на бесконечности, и главный интересный показатель — `theta_cam_deg`, угол между
оптической осью и восстановленной нормалью к плоскости. На таком наборе он обязан
быть близок к нулю; систематическое отклонение указывает на дефект оценки точек
схода или выбора внутренней калибровки. Точность в миллиметрах эта проба НЕ
проверяет: истинных размеров у CMP нет, опорный размер выдуман (см. `_reference`).

**Три исхода, не два.** Каждый снимок относится ровно к одному из трёх:
плоскость оценена; плоскость не оценена и вызван оператор (штатный отказ по
контракту п. 4.2 спецификации); исключение (неожиданное). Второй и третий исходы
не складываются — это разные события, и смешение их теряет единственный
интересный различитель прогона.
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from facade_digitizer.geometry.vanishing import detect_segments
from facade_digitizer.pipeline.calib import intrinsics_from_meta, undistort
from facade_digitizer.pipeline.io import load_image
from facade_digitizer.pipeline.plane import estimate_plane
from facade_digitizer.pipeline.run import OperatorReference, process

#: Опорный размер ВЫДУМАН — у CMP нет истинных размеров фасадов. Порядок величины
#: (порядок десяти метров на ширину фасада) выбран по заданию задачи 17, а не
#: измерен. Любая величина в миллиметрах, посчитанная от него ниже по конвейеру
#: (вердикт качества, GSD), унаследует эту выдуманность, а не станет измерением.
ASSUMED_FACADE_WIDTH_MM = 10000.0

#: Опорная база — доля ширины кадра между двумя точками, которые оператор
#: «указал бы» на снимке. Начало отсчёта совпадает с левой из них (тот же приём,
#: что в tests/test_run.py::_reference: меняется ровно одна величина).
SPAN_FRACTION = 0.8
X0_FRACTION = 0.1
Y_FRACTION = 0.9

#: Источник масштаба, для которого спецификация сама говорит «допуск не
#: достигается» (п. 6.3) — именно поэтому он годится для честной пробы без
#: измеренного размера: невозможность подогнать σ под допуск не путают с
#: намеренным допущением.
SCALE_SOURCE = "assumed_floor_height"

#: Разбор сообщения `rectify()` о недостижимом mm_per_px (facade_digitizer/
#: pipeline/rectify.py): "годится mm_per_px от X до Y мм". Наша ПЕРВАЯ догадка
#: разрешения растра (натуральный масштаб исходных пикселей) на выборке всегда
#: попадала в достижимый диапазон (см. отчёт задачи), но это не гарантия для
#: каждого возможного кадра — при промахе скрипт разбирает границы из ЭТОГО ЖЕ
#: сообщения (оно для того и называет их, см. docstring `process()`) и повторяет
#: попытку один раз со средним геометрическим границ, не изобретая числа заново.
_RASTER_BOUNDS_RE = re.compile(r"годится mm_per_px от ([0-9.eE+\-]+) до ([0-9.eE+\-]+) мм")

#: Известные шаблоны причин отказа (из geometry/vanishing.py и pipeline/quality.py)
#: с вырезанными числовыми хвостами — для группировки «причина: сколько случаев»,
#: а не «уникальная строка с числом: один случай» (числа делают почти каждую
#: причину неповторимой, а группировать нужно её СМЫСЛ).
_REASON_PREFIXES = [
    "слишком мало отрезков",
    "недостаточная поддержка одной из точек схода",
    "направления не ортогональны",
    "малая доля отрезков поддержала точки схода",
    "велика невязка инлайеров",
    "поддержавшие отрезки занимают малую долю кадра",
    "направление точки схода не определено",
    "направление точки схода определено неустойчиво",
    "мало поддерживающих отрезков",
    "недостаточная резкость",
    "недостаточное разрешение",
    "угловому условию удовлетворяет лишь",
    "за линией схода оказалось",
    "разрешение оценено по опорной базе оператора",
    "угол визирования не измерен",
]


#: Обёртка `estimate_vanishing_points` (geometry/vanishing.py) вокруг текста
#: «просевшего сильнее прочих» множителя: "доверие ниже порога 0.5: <текст>".
#: Без снятия обёртки причина внутри неё группировалась бы отдельно от ТОЙ ЖЕ
#: причины, пришедшей по дискретной проверке напрямую — то есть 204 отказа
#: дробились бы на десятки корзин по одному снимку вместо слияния по смыслу.
#: Найдено при самопроверке скрипта, см. отчёт задачи 17.
_THRESHOLD_WRAPPER_RE = re.compile(r"^доверие ниже порога [0-9.]+: (.+)$")


def _bucket_reason(reason: str) -> str:
    """Причина без числового хвоста — для группировки по смыслу, а не по значению."""
    match = _THRESHOLD_WRAPPER_RE.match(reason)
    if match is not None:
        reason = match.group(1)
    for prefix in _REASON_PREFIXES:
        if reason.startswith(prefix):
            return prefix
    return reason  # неизвестный шаблон — показывается как есть, чтобы не потерять


def _reference(w: int, h: int) -> tuple[OperatorReference, float]:
    """Указания оператора: выдуманный опорный размер, привязанный к КАДРУ.

    Точки — доля ширины кадра, а не абсолютные пиксели: снимки CMP разного
    размера (241..1024 px по стороне), и абсолютные координаты вышли бы за кадр
    одних снимков и сжались бы в угол других.
    """
    x0, x1, y = X0_FRACTION * w, (X0_FRACTION + SPAN_FRACTION) * w, Y_FRACTION * h
    span_mm = SPAN_FRACTION * ASSUMED_FACADE_WIDTH_MM
    reference = OperatorReference(origin_px=(x0, y), span_px=((x0, y), (x1, y)), span_mm=span_mm)
    return reference, span_mm


def _raster_guess(span_mm: float, w: int) -> float:
    """Первая догадка разрешения растра: натуральный масштаб исходных пикселей."""
    return span_mm / (SPAN_FRACTION * w)


def _run_process(path, reference, raster_guess, scale_source):
    """`process()` с одной подстраховочной попыткой при недостижимом raster_mm_per_px.

    Возвращает (model, exc, raster_used, retried). `exc` — не None ровно тогда,
    когда model равна None: смешивать «нет модели» с «есть модель, но и есть
    исключение» здесь нечем, это взаимоисключающие случаи.
    """
    try:
        model = process(path, operator_reference=reference, raster_mm_per_px=raster_guess,
                        scale_source=scale_source)
        return model, None, raster_guess, False
    except ValueError as error:
        match = _RASTER_BOUNDS_RE.search(str(error))
        if match is None:
            return None, error, raster_guess, False
        lo, hi = float(match.group(1)), float(match.group(2))
        retry_value = math.sqrt(lo * hi)
        try:
            model = process(path, operator_reference=reference, raster_mm_per_px=retry_value,
                            scale_source=scale_source)
            return model, None, retry_value, True
        except Exception as retry_error:  # noqa: BLE001 - причина сохраняется, не глушится
            return None, retry_error, retry_value, True
    except Exception as error:  # noqa: BLE001 - причина сохраняется, не глушится
        return None, error, raster_guess, False


def _diagnostics(image: np.ndarray, K: np.ndarray) -> dict:
    """Инструментальные измерения, НЕ влияющие на официальный исход.

    Официальный исход даёт исключительно `process()` (см. `_run_process`). Здесь —
    отдельно посчитанные число отрезков ДО и ПОСЛЕ фильтра длины (задание просит
    сверить порог `min_length_px = 40`, настроенный на кадре 5280x3956, с кадром
    на порядок мельче) и доля третьей однородной компоненты точек схода
    (проверка пути «точка схода на бесконечности» из задания). Ошибка здесь —
    не поломка прогона, а отсутствующая диагностика по этому кадру; она попадает
    в сводку отдельной строкой, а не проглатывается молча.
    """
    diag = {"raw_segments": None, "filtered_segments": None,
            "vh_w_ratio": None, "vv_w_ratio": None, "diag_error": None}
    try:
        lsd = cv2.createLineSegmentDetector()
        lines, _, _, _ = lsd.detect(image)
        diag["raw_segments"] = 0 if lines is None else len(lines)
        diag["filtered_segments"] = len(detect_segments(image))
    except Exception as error:  # noqa: BLE001 - диагностика необязательна, причина видна
        diag["diag_error"] = f"отрезки: {type(error).__name__}: {error}"
        return diag
    try:
        plane = estimate_plane(image, K)
        if not plane.needs_operator:
            vh_norm = float(np.linalg.norm(plane.vh))
            vv_norm = float(np.linalg.norm(plane.vv))
            if vh_norm > 0:
                diag["vh_w_ratio"] = abs(float(plane.vh[2])) / vh_norm
            if vv_norm > 0:
                diag["vv_w_ratio"] = abs(float(plane.vv[2])) / vv_norm
    except Exception as error:  # noqa: BLE001
        diag["diag_error"] = f"точки схода: {type(error).__name__}: {error}"
    return diag


def _percentile_line(label, values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if values.size == 0:
        return f"{label}: нет данных (n=0)"
    return (f"{label}: n={values.size}  median={np.median(values):.3f}  "
           f"p10={np.percentile(values, 10):.3f}  p90={np.percentile(values, 90):.3f}  "
           f"min={values.min():.3f}  max={values.max():.3f}")


def run(folder: Path, limit: int | None) -> int:
    files = sorted(folder.glob("*.jpg"))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"снимков *.jpg в {folder} не найдено", file=sys.stderr)
        return 1

    rows = []
    exceptions = []          # (file, stage, type, message)
    diag_failures = 0
    raster_retries = 0

    t_start = time.perf_counter()
    for path in files:
        row = {"file": path.name}
        try:
            image, meta = load_image(path)
            K, _source = intrinsics_from_meta(meta, None)
            image = undistort(image, K, [0.0] * 5)
        except Exception as error:  # noqa: BLE001 - один битый снимок не роняет прогон
            exceptions.append((path.name, "load", type(error).__name__, str(error)))
            rows.append({**row, "outcome": "exception"})
            continue

        diag = _diagnostics(image, K)
        if diag["diag_error"] is not None:
            diag_failures += 1
        row.update(diag)

        h, w = image.shape[:2]
        row["width_px"], row["height_px"] = w, h
        reference, span_mm = _reference(w, h)
        raster_guess = _raster_guess(span_mm, w)

        t0 = time.perf_counter()
        model, error, raster_used, retried = _run_process(path, reference, raster_guess,
                                                          SCALE_SOURCE)
        row["elapsed_s"] = time.perf_counter() - t0
        row["raster_mm_per_px"] = raster_used
        if retried:
            raster_retries += 1

        if error is not None:
            exceptions.append((path.name, "process", type(error).__name__, str(error)))
            row["outcome"] = "exception"
            rows.append(row)
            continue

        rec = model.images[0]
        row["outcome"] = "needs_operator" if rec.rectification.needs_operator else "plane"
        row["confidence"] = rec.rectification.confidence
        row["verdict"] = rec.quality.verdict
        row["reasons"] = list(rec.quality.reasons)
        row["theta_cam_deg"] = rec.theta_cam_deg
        rows.append(row)

    total_elapsed = time.perf_counter() - t_start

    # Самопроверка счётчика: каждый снимок обязан попасть ровно в один исход.
    outcomes = {"plane": 0, "needs_operator": 0, "exception": 0}
    for row in rows:
        outcomes[row["outcome"]] += 1
    assert sum(outcomes.values()) == len(files) == len(rows), (
        f"счётчик исходов разошёлся с числом снимков: {outcomes} vs {len(files)}")

    print("=" * 72)
    print(f"CMP smoke: {folder}")
    print(f"снимков найдено: {len(files)}; обработано: {len(rows)}; "
         f"время прогона: {total_elapsed:.1f} с "
         f"({total_elapsed / max(len(rows), 1):.2f} с/снимок)")
    print()
    print("Три исхода (доли от общего числа):")
    n = len(rows)
    for key in ("plane", "needs_operator", "exception"):
        print(f"  {key:16s}: {outcomes[key]:4d}  ({outcomes[key] / n:.1%})")
    print()

    plane_rows = [r for r in rows if r["outcome"] == "plane"]
    print("theta_cam_deg — только по снимкам, где плоскость оценена "
         f"(n={len(plane_rows)} из {n}; needs_operator и exception сюда не входят):")
    print("  " + _percentile_line("theta_cam_deg", [r["theta_cam_deg"] for r in plane_rows]))
    print("  " + _percentile_line("confidence (plane)", [r["confidence"] for r in plane_rows]))
    print()

    no_rows = [r for r in rows if r["outcome"] == "needs_operator"]
    if no_rows:
        print(f"confidence на отказных снимках (n={len(no_rows)}):")
        print("  " + _percentile_line("confidence (needs_operator)",
                                      [r["confidence"] for r in no_rows]))
        print()

    print("Вердикты качества (описательно — критерий не в масштабе, "
         "он выдуман; см. docstring):")
    verdict_counts = {}
    for r in rows:
        if "verdict" in r:
            verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1
    for verdict, count in sorted(verdict_counts.items()):
        print(f"  {verdict:10s}: {count:4d}  ({count / n:.1%})")
    print()

    print("Причины (сгруппированы по шаблону, число снимков, у которых причина "
         "встретилась хотя бы раз):")
    reason_counts = {}
    for r in rows:
        for reason in r.get("reasons", []):
            bucket = _bucket_reason(reason)
            reason_counts[bucket] = reason_counts.get(bucket, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {reason}")
    print()

    seg_rows = [r for r in rows if r.get("filtered_segments") is not None]
    print(f"Отрезки детектора (min_length_px=40), кадр CMP на порядок мельче "
         f"5280x3956 (n={len(seg_rows)} из {n}):")
    print("  " + _percentile_line("найдено (до фильтра)",
                                  [r["raw_segments"] for r in seg_rows]))
    print("  " + _percentile_line("после фильтра >=40px",
                                  [r["filtered_segments"] for r in seg_rows]))
    ratios = [r["filtered_segments"] / r["raw_segments"] for r in seg_rows
             if r["raw_segments"]]
    print("  " + _percentile_line("доля пережившая фильтр", ratios))
    print()

    inf_rows = [r for r in rows if r.get("vh_w_ratio") is not None]
    near_inf = [r for r in inf_rows
               if max(r["vh_w_ratio"], r["vv_w_ratio"]) < 1e-3]
    print(f"Точки схода на бесконечности (диагностика, n={len(inf_rows)} из {n} "
         "снимков с оценённой плоскостью, где диагностика удалась):")
    print(f"  снимков с обеими |w|/|v| < 1e-3 (вырождение в бесконечность "
         f"по построению): {len(near_inf)} из {len(inf_rows)}")
    if inf_rows:
        both = [max(r["vh_w_ratio"], r["vv_w_ratio"]) for r in inf_rows]
        print("  " + _percentile_line("max(|w_h|/|v_h|, |w_v|/|v_v|)", both))
    print()

    sizes = [(r["width_px"], r["height_px"]) for r in rows if "width_px" in r]
    if sizes:
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"Размеры кадров: ширина {min(ws)}..{max(ws)} px, "
             f"высота {min(hs)}..{max(hs)} px "
             f"(для сравнения — синтетика настроена на 5280x3956)")
        print()

    if exceptions:
        print(f"Исключения по отдельным снимкам ({len(exceptions)}):")
        for name, stage, etype, msg in exceptions:
            print(f"  [{stage}] {name}: {etype}: {msg[:200]}")
        print()

    print("Самопроверка скрипта:")
    print(f"  диагностика (отрезки/точки схода) не удалась на {diag_failures} снимках "
         f"из {n} — официальный исход это не затронуло, т.к. его даёт только process()")
    print(f"  подстраховочный повтор raster_mm_per_px сработал на {raster_retries} "
         f"снимках из {n}")
    print(f"  сумма трёх исходов == числу обработанных снимков: "
         f"{sum(outcomes.values())} == {len(rows)}: OK")
    print("=" * 72)
    return 0


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="smoke_cmp",
        description="Проба геометрического ядра на снимках CMP Facade DB (задача 17).")
    parser.add_argument("folder", type=Path, help="папка со снимками *.jpg (набора CMP)")
    parser.add_argument("--limit", type=int, default=None,
                        help="предел числа снимков (по умолчанию — все найденные)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass  # старый Python без reconfigure — вывод останется в кодировке консоли
    args = _parse_args(argv)
    return run(args.folder, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
