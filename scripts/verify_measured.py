"""Проверка по снимкам с измеренными размерами. План 3, задача 27; спецификация, п. 12–13.

    python scripts/verify_measured.py измерения.csv выход1.json [выход2.json ...]

Таблица измерений — CSV (UTF-8, разделитель «,»), по строке на измерение:

    image,element_id,quantity,value_mm,sigma_ref_mm,is_base,note

* `image` — имя снимка без расширения (как `<снимок>.json` экспорта);
* `element_id` — идентификатор проёма в экспорте (w_000, …); у базы — пусто;
* `quantity` — `width`, `height`, `recess`, `x`, `y` (нижний левый угол от начала
  отсчёта фасада), `base`;
* `sigma_ref_mm` — погрешность самого измерения (рулетка, дальномер);
* `is_base` — 1 у ОДНОЙ строки на снимок: размер, ставший опорной базой. Он в
  проверку НЕ входит — иначе проверка замкнута на себя. Снимок без строки базы —
  отказ: неизвестно, что исключать.

Печатается раздельно по величинам (габарит, положение, глубина) и по режимам:
`error_stats` — RMSE, P95, смещение ОТДЕЛЬНО от разброса; покрытие 1σ и 2σ
вместе с `mean_sharpness` (покрытие без остроты завышается раздутой σ);
interval score. σ интервала — `√(σ_модели² + σ_измерения²)`.

На нескольких снимках это «первые цифры точности» (п. 13), а не проверка
статистического допуска п. 2.2 — вывод говорит это прямо.
"""
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from facade_digitizer.metrics import coverage, error_stats, interval_score, mean_sharpness

GROUPS = {"width": "габарит", "height": "габарит", "x": "положение", "y": "положение",
          "recess": "глубина"}
MIN_FOR_STATISTICS = 30


@dataclass(frozen=True)
class Pair:
    group: str
    mode: str
    measured: float        # результат конвейера
    truth: float           # измерение заказчика
    sigma: float           # σ интервала: модель и измерение


def read_table(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    required = {"image", "element_id", "quantity", "value_mm", "sigma_ref_mm", "is_base"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path}: нужны столбцы {sorted(required)}")
    return rows


def _model_value(element: dict, quantity: str) -> tuple[float, float] | None:
    size = element["size_mm"]
    if quantity == "width":
        return size["width"], size["sigma_width"]
    if quantity == "height":
        return size["height"], size["sigma_height"]
    if quantity in ("x", "y"):
        if element.get("position_sigma_mm") is None:
            return None
        x, y = element["contour_mm"][0]
        return (x if quantity == "x" else y), element["position_sigma_mm"]
    if quantity == "recess":
        recess = element.get("recess") or {}
        if recess.get("value_mm") is None or recess.get("sigma_mm") is None:
            return None
        return recess["value_mm"], recess["sigma_mm"]
    raise ValueError(f"неизвестная величина: {quantity}")


def pairs(rows: list[dict], models: dict[str, dict]) -> tuple[list[Pair], list[str]]:
    """Пары «результат — измерение» без строк базы; второе — пропущенные с причиной."""
    bases = {r["image"] for r in rows if r["is_base"].strip() in ("1", "true", "да")}
    out, skipped = [], []
    for image in sorted({r["image"] for r in rows}):
        if image not in bases:
            raise ValueError(f"{image}: в таблице нет строки опорной базы (is_base = 1) — "
                             "неизвестно, какое измерение исключить из проверки")
    for r in rows:
        if r["is_base"].strip() in ("1", "true", "да"):
            continue
        model = models.get(r["image"])
        if model is None:
            skipped.append(f"{r['image']}: нет выходного JSON")
            continue
        element = next((e for e in model["elements"] if e["id"] == r["element_id"]), None)
        if element is None:
            skipped.append(f"{r['image']}/{r['element_id']}: элемента нет в выходе")
            continue
        value = _model_value(element, r["quantity"])
        if value is None:
            skipped.append(f"{r['image']}/{r['element_id']}/{r['quantity']}: "
                           "величина не выпущена конвейером")
            continue
        measured, sigma_model = value
        sigma = math.hypot(sigma_model, float(r["sigma_ref_mm"] or 0.0))
        out.append(Pair(group=GROUPS[r["quantity"]], mode=model["mode"],
                        measured=measured, truth=float(r["value_mm"]), sigma=sigma))
    return out, skipped


def summary(selected: list[Pair]) -> dict:
    m = np.array([p.measured for p in selected])
    t = np.array([p.truth for p in selected])
    s = np.array([p.sigma for p in selected])
    stats = error_stats(m, t)
    return {"n": stats.n, "rmse": stats.rmse, "p95": stats.p95, "bias": stats.bias,
            "sigma": stats.sigma, "coverage_1": coverage(m, t, s, 1.0),
            "coverage_2": coverage(m, t, s, 2.0), "sharpness": mean_sharpness(s),
            "interval_score": interval_score(m, t, s)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="verify_measured", description=__doc__.splitlines()[0])
    p.add_argument("table", type=Path)
    p.add_argument("outputs", nargs="+", type=Path)
    args = p.parse_args(argv)
    models = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in args.outputs}
    try:
        found, skipped = pairs(read_table(args.table), models)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print("Первые цифры точности (п. 13), НЕ проверка статистического допуска п. 2.2: "
          f"снимков {len(models)}, сравнений {len(found)}. Опорная база в проверку не входит.")
    for group in ("габарит", "положение", "глубина"):
        for mode in sorted({x.mode for x in found}):
            selected = [x for x in found if x.group == group and x.mode == mode]
            if not selected:
                continue
            s = summary(selected)
            few = " (мало для статистики)" if s["n"] < MIN_FOR_STATISTICS else ""
            print(f"{group}, режим {mode}: n = {s['n']}{few}\n"
                  f"  RMSE {s['rmse']:.1f} мм, P95 {s['p95']:.1f} мм, смещение "
                  f"{s['bias']:+.1f} мм, разброс {s['sigma']:.1f} мм\n"
                  f"  покрытие 1σ {s['coverage_1']:.0%} (норма 68 %), 2σ "
                  f"{s['coverage_2']:.0%} (95 %) при средней σ {s['sharpness']:.1f} мм; "
                  f"interval score {s['interval_score']:.1f} мм")
    for line in skipped:
        print(f"пропущено: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
