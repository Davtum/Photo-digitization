"""Блоки обучения оператора: выдать и оценить. План 3, задача 24.

    python scripts/training_blocks.py make  --out-dir обучение --blocks 6
    python scripts/training_blocks.py score --out-dir обучение

`make` пишет `обучение/блокNN/` — снимки и `истина.json`. Оператор размечает каждый
снимок в `facade-digitize-ui --operator op-…`, базу — по подсказке из `истина.json`,
и экспортирует в `обучение/блокNN/экспорт/`. `score` сверяет экспорт с истиной по
блокам и печатает, закончено ли обучение (`training.converged`).
"""
import argparse
import json
import sys
from pathlib import Path

from facade_digitizer.training import block_stats, converged, score_scene, write_block


def _make(args) -> int:
    for block in range(args.first, args.first + args.blocks):
        for path in write_block(args.out_dir, block, seed=args.seed):
            print(path)
    return 0


def _score(args) -> int:
    stats = []
    for block_dir in sorted(Path(args.out_dir).glob("блок*")):
        truth = {t["name"]: t for t in json.loads(
            (block_dir / "истина.json").read_text(encoding="utf-8"))}
        scores = []
        for name, record in truth.items():
            exported = block_dir / "экспорт" / f"{name}.json"
            if not exported.is_file():
                print(f"{block_dir.name}: нет экспорта {exported.name}", file=sys.stderr)
                continue
            scores += score_scene(json.loads(exported.read_text(encoding="utf-8")), record)
        if not scores:
            continue
        block = int(block_dir.name.removeprefix("блок"))
        s = block_stats(block, scores)
        stats.append(s)
        seconds = f"{s.median_seconds:.0f} с" if s.median_seconds is not None else "—"
        print(f"блок {block:02d}: элементов {s.elements}, медиана времени на элемент "
              f"{seconds}, медианный |промах| габарита {s.median_abs_error_mm:.1f} мм")
    done, why = converged(stats)
    print(("Обучение закончено: " if done else "Обучение продолжается: ") + why)
    return 0 if done else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="training_blocks", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="выдать блоки")
    make.add_argument("--out-dir", type=Path, default=Path("обучение"))
    make.add_argument("--blocks", type=int, default=6)
    make.add_argument("--first", type=int, default=0)
    make.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score", help="оценить экспорт")
    score.add_argument("--out-dir", type=Path, default=Path("обучение"))
    args = p.parse_args(argv)
    return _make(args) if args.command == "make" else _score(args)


if __name__ == "__main__":
    raise SystemExit(main())
