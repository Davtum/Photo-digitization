"""Замер σ клика на операторах: выдать сцены и разобрать сессии. План 3, задача 25.

    python scripts/click_study.py make    --out-dir замер_клика
    python scripts/click_study.py analyze --out-dir замер_клика [--repeat-dir повторы]

`make` пишет по каталогу на уровень размытия (`размытие_0.8/…`), в каждом — сцены и
`истина.json`. Оператор размечает углы проёмов при масштабах из протокола
(«Вид → Масштаб …», Ctrl+5 / Ctrl+1 / Ctrl+2 / Ctrl+4) и сохраняет СЕССИЮ
(«Файл → Сохранить сессию…») рядом со снимком: в ней масштаб каждого клика.
`analyze` сводит ошибки кликов по масштабам, подгоняет модель `ui.zoom` и печатает
смещение отдельно от разброса; с `--repeat-dir` — повторяемость внутри оператора
по повторным сессиям одного снимка. Протокол —
`docs/superpowers/plans/results/click-sigma-protocol.md`.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from facade_digitizer.click_study import (
    click_errors,
    clicks_of_session,
    edge_width_px,
    fit_sigma_model,
    per_scale,
    repeatability,
)
from facade_digitizer.pipeline.io import load_image
from facade_digitizer.training import SceneSpec, write_block

BLUR_LEVELS = (0.8, 1.5, 2.5)
NOMINAL_SCALES = (0.5, 1.0, 2.0, 4.0)
STUDY_SEED = 25


def _make(args) -> int:
    for blur in BLUR_LEVELS:
        root = Path(args.out_dir) / f"размытие_{blur}"
        for block in range(args.blocks):
            for path in write_block(root, block, seed=STUDY_SEED, blur_sigma_px=blur,
                                    prefix="набор"):
                print(path)
    return 0


def _truth_corners(record: dict) -> np.ndarray:
    fields = {k: record[k] for k in ("name", "dx", "dy", "dist")}
    spec = SceneSpec(**fields, openings=tuple(record["openings"]))
    sc = spec.scene()
    return np.vstack([sc.project(o.corners_mm()) for o in sc.openings])


def _analyze(args) -> int:
    for blur_dir in sorted(Path(args.out_dir).glob("размытие_*")):
        errors, widths, total = [], [], 0
        for set_dir in sorted(blur_dir.glob("набор*")):
            truth = {t["name"]: t for t in json.loads(
                (set_dir / "истина.json").read_text(encoding="utf-8"))}
            for session_path in sorted(set_dir.glob("*.session.json")):
                raw = json.loads(session_path.read_text(encoding="utf-8"))
                name = Path(raw["image"]["path"]).stem
                if name not in truth:
                    continue
                corners = _truth_corners(truth[name])
                errors += click_errors(raw, corners)
                total += len(clicks_of_session(raw))
                image, _meta = load_image(set_dir / f"{name}.png")
                # Все четыре стороны: к одной из них примыкает грань откоса, и
                # профиль там — не чистая ступенька; медиана по сторонам устойчива.
                for q in corners.reshape(-1, 4, 2):
                    widths += [edge_width_px(image, q[i], q[(i + 1) % 4]) for i in range(4)]
        if not errors:
            continue
        rows = per_scale(errors, NOMINAL_SCALES)
        print(f"{blur_dir.name}: кликов {len(errors)} (вне 25 px от угла — {total - len(errors)},"
              f" в σ не входят), ширина кромки 10–90 % {np.median(widths):.2f} px")
        for r in rows:
            print(f"  масштаб {r.scale:g}: n={r.n}, смещение ({r.bias_x:+.2f}, "
                  f"{r.bias_y:+.2f}) px, σ {r.sigma:.2f} px")
        try:
            fit = fit_sigma_model(rows)
            print(f"  σ_экрана {fit.sigma_screen_px:.2f} px, σ_кромки {fit.sigma_edge_px:.2f} px,"
                  f" невязка модели {fit.residual_rel:.0%}")
        except ValueError as error:
            print(f"  {error}")
    if args.repeat_dir:
        sessions = [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(Path(args.repeat_dir).glob("*.session.json"))]
        sigma, n = repeatability(sessions)
        print(f"Повторяемость внутри оператора: σ {sigma:.2f} px по {n} углам, "
              f"{len(sessions)} повторов")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="click_study", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make")
    make.add_argument("--out-dir", type=Path, default=Path("замер_клика"))
    make.add_argument("--blocks", type=int, default=2)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--out-dir", type=Path, default=Path("замер_клика"))
    analyze.add_argument("--repeat-dir", type=Path, default=None)
    args = p.parse_args(argv)
    return _make(args) if args.command == "make" else _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
