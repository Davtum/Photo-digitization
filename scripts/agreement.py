"""Согласие разметчиков по их экспортам. План 3, задача 26.

    python scripts/agreement.py согласие/op-1a2b3c4d согласие/op-5e6f7a8b согласие/op-9c0d1e2f

Каждый каталог — экспорт одного разметчика (`<снимок>.json`). Снимки,
экспортированные всеми, сводятся: кромки — в мм общей геометрией (первого
разметчика), пределы согласия по каждой кромке, каппа Флейса для edge_type и
mounting, доля расхождений в выборе кромки и пропущенных проёмов. Протокол —
`docs/superpowers/plans/results/agreement-protocol.md`.
"""
import argparse
import json
from pathlib import Path

from facade_digitizer.agreement_study import CHOICE_THRESHOLD_MM, agreement_many


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agreement", description=__doc__.splitlines()[0])
    p.add_argument("raters", nargs="+", type=Path, help="каталоги экспорта разметчиков")
    args = p.parse_args(argv)
    names = sorted(set.intersection(*[{f.name for f in d.glob("*.json")
                                       if not f.name.endswith((".marks.json",
                                                               ".session.json"))}
                                      for d in args.raters]))
    groups = [[json.loads((d / n).read_text(encoding="utf-8")) for d in args.raters]
              for n in names]
    report = agreement_many(groups)
    print(f"Снимков {len(names)}, разметчиков {len(args.raters)}, сопоставлено проёмов "
          f"{report.items}, пропущено хотя бы одним — {report.missing}")
    for edge, s in report.edges.items():
        bias = ", ".join(f"{b:+.1f}" for b in s.rater_bias)
        print(f"  кромка {edge}: σ внутри проёма {s.sigma_within:.1f} мм, предел согласия "
              f"пары ±{s.loa_half_width:.1f} мм; смещения разметчиков [{bias}] мм")

    def kappa(v):
        return "не определена (все согласны)" if v is None else f"{v:.3f}"

    print(f"  каппа Флейса: edge_type {kappa(report.kappa_edge_type)}, "
          f"mounting {kappa(report.kappa_mounting)}")
    print(f"  нет единодушия по edge_type: {report.edge_type_disagreement:.0%} проёмов")
    print(f"  расхождение в выборе кромки (> {CHOICE_THRESHOLD_MM:.0f} мм от медианы): "
          f"{report.choice_disagreement:.0%} проёмов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
