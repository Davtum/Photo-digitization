"""Интерфейс оператора по записанному сценарию. План 3, задача 19; веб-интерфейс, задача 10.

Сценарий — тот же путь, который проходит человек по чек-листу приёмки
(`docs/operator-acceptance-checklist.md`): открыть снимок → предварительный вердикт →
опорная база → проём с откосом → результат → правка точки → выгрузка JSON и DXF.
Клики записаны заранее: это проекции ИСТИННЫХ углов синтетической сцены
(`scripts/demo_synthetic.py`), поэтому в конце габарит сверяется с истиной.

Сценарий идёт через `Workbench` и `Desk` — ту же логику, что стоит за страницей в
браузере, — без браузера. Рабочая папка демо — `--out-dir`: снимок, сессия (пишется
сама), выгрузка в `экспорт/`.

* по умолчанию — сценарий и итог в консоли;
* `--show` — затем запускается интерфейс над этой папкой, и браузер открывается на
  размеченном снимке: так показывают работу вживую и продолжают руками.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Ракурс ближе ракурса `demo_synthetic`: на нём разрешение проходит шлюз качества,
#: и вердикт — не отказ. Тот же ракурс у тестов (`tests/web_scenes.NEAR`).
VIEW = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0, "depth": 150.0}
#: Увеличение, при котором «оператор» указывает углы и кромку откоса.
MARK_SCALE = 3.0
IMAGE = "facade_demo.png"


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="demo_ui", description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("demo_ui_out"),
                   help="рабочая папка демо: снимок, сессия, выгрузка JSON/DXF")
    p.add_argument("--show", action="store_true",
                   help="затем открыть интерфейс в браузере над этой папкой")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from demo_synthetic import build_scene, operator_base

    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from facade_digitizer.web.jobs import InlineRunner
    from facade_digitizer.web.workbench import Workbench

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sc = build_scene(**VIEW)
    save_image(out / IMAGE, sc.render())
    base_px, span_mm, _origin = operator_base(sc)
    op = sc.openings[0]
    corners = sc.project(op.corners_mm())
    side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                   op.y, op.y + op.height)
    x_edge = op.x if side == "left" else op.x + op.width
    inner = sc.project(np.array([[x_edge, op.y], [x_edge, op.y + op.height]]),
                       depth=op.depth)

    workbench = Workbench(out, runner=InlineRunner())
    session_path = out / "facade_demo.session.json"
    if session_path.exists():
        session_path.unlink()                    # демо каждый раз начинается заново
    workbench.open(IMAGE)
    desk = workbench.desk
    step = [0]

    def report(note: str) -> None:
        step[0] += 1
        notice = desk.state()["notice"]
        if notice and notice["kind"] == "error":
            raise SystemExit(f"шаг {step[0]} ({note}): {notice['text']}")
        print(f"[{step[0]}] {note}")

    q = desk.state()["quality"]
    report(f"Снимок открыт: предварительный вердикт «{q['verdict']}»")

    desk.act("set_step", step="scale")
    for x, y in base_px:
        desk.act("click", x=float(x), y=float(y), view_scale=1.0)
    desk.act("set_base", span_mm=float(span_mm), span_sigma_mm=None,
             scale_source="operator_reference", origin_is_facade_corner=False)
    report(f"Опорная база {span_mm:.0f} мм: {desk.state()['scale']['result']}")

    desk.act("new_mark", class_="window")
    mark_id = desk.current_mark
    for x, y in corners:
        desk.act("click", x=float(x), y=float(y), view_scale=MARK_SCALE)
    desk.act("set_reveal_side", side=side)
    desk.act("start_reveal")
    for x, y in inner:
        desk.act("click", x=float(x), y=float(y), view_scale=MARK_SCALE)
    report(f"Проём {mark_id} и кромка откоса ({side}) при увеличении {MARK_SCALE:g}:1")

    row = desk.state()["result"]["rows"][0]
    report(f"Результат: ширина {row['width']}, высота {row['height']}, "
           f"заглубление {row['recess']}")

    x, y = corners[1]
    desk.act("drop", key=f"{mark_id}:corner:1", x=float(x + 12.0), y=float(y),
             view_scale=MARK_SCALE, moved=True)
    moved = desk.state()["result"]["rows"][0]["width"]
    desk.act("drop", key=f"{mark_id}:corner:1", x=float(x), y=float(y),
             view_scale=MARK_SCALE, moved=True)
    report(f"Угол сдвинут на 12 px и возвращён: ширина {moved} → "
           f"{desk.state()['result']['rows'][0]['width']} (полный пересчёт)")

    exported = desk.act("export")
    report(f"Выгружено в {exported.json_path.parent}")

    element = next(e for e in desk.session.model.elements if e.id == mark_id)
    size = element.size_mm
    print()
    print(f"Истина сцены: {op.width:.0f} × {op.height:.0f} мм, заглубление {op.depth:.0f} мм")
    print(f"Измерено:     {size.width:.1f} ± {size.sigma_width:.1f} × "
          f"{size.height:.1f} ± {size.sigma_height:.1f} мм", end="")
    if element.recess and element.recess.value_mm is not None:
        print(f", заглубление {element.recess.value_mm:.1f} ± {element.recess.sigma_mm:.1f} мм")
    else:
        print()
    print(f"Сессия: {session_path}")
    print(f"Выгрузка: {exported.json_path}\n          {exported.dxf_path}")
    print(f"Команда CLI, воспроизводящая тот же JSON:\n  {exported.cli_line}")
    workbench.close()
    if args.show:
        from facade_digitizer.web.__main__ import main as serve

        print("\nОткрываю интерфейс над папкой демо — можно продолжать руками.")
        return serve(["--data-dir", str(out), "--open", IMAGE])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
