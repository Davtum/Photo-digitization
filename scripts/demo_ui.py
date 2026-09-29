"""Интерфейс оператора по записанному сценарию: снимки экрана каждого шага. План 3, задача 19.

Сценарий — тот же путь, который проходит человек по чек-листу приёмки
(`docs/operator-quickstart.md`): открыть снимок → предварительный вердикт → опорная
база → проём с откосом → панель результата → правка точки → экспорт JSON и DXF.
Клики записаны заранее: это проекции ИСТИННЫХ углов синтетической сцены
(`scripts/demo_synthetic.py`), поэтому в конце габарит сверяется с истиной.

Два режима:

* по умолчанию — без экрана (offscreen), снимки окна после каждого шага в
  `--shots-dir`; это материал для отчётов, а НЕ замена ручного запуска;
* `--show` — тот же сценарий в видимом окне, с паузой между шагами, после чего окно
  остаётся открытым: так показывают работу вживую и продолжают руками.

Рядом с результатом пишется файл сессии: «Файл → Открыть сессию…» восстанавливает
разметку в окне, запущенном командой `facade-digitize-ui`.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Ракурс ближе ракурса `demo_synthetic`: на нём разрешение проходит шлюз качества,
#: и вердикт — не отказ. Тот же ракурс у тестов окна (`tests/test_ui_marks.VIEW`).
VIEW = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0, "depth": 150.0}
#: Увеличение, при котором «оператор» указывает углы и кромку откоса.
MARK_SCALE = 3.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="demo_ui", description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("demo_ui_out"),
                   help="снимок сцены, сессия, экспорт JSON/DXF")
    p.add_argument("--shots-dir", type=Path, default=None,
                   help="снимки экрана по шагам (по умолчанию <out-dir>/shots)")
    p.add_argument("--show", action="store_true",
                   help="видимое окно с паузами; окно остаётся открытым в конце")
    p.add_argument("--pause", type=float, default=1.5, help="пауза между шагами при --show, с")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from demo_synthetic import build_scene, operator_base
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QApplication, QScrollArea

    from facade_digitizer.geometry.parallax import visible_reveal_side
    from facade_digitizer.pipeline.io import save_image
    from facade_digitizer.ui.window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    out = args.out_dir.resolve()
    shots = (args.shots_dir or out / "shots").resolve()
    out.mkdir(parents=True, exist_ok=True)
    shots.mkdir(parents=True, exist_ok=True)

    sc = build_scene(**VIEW)
    image = out / "facade_demo.png"
    save_image(image, sc.render())
    base_px, span_mm, _origin = operator_base(sc)
    op = sc.openings[0]
    corners = sc.project(op.corners_mm())
    side, _h = visible_reveal_side(sc.camera_on_plane(), op.x, op.x + op.width,
                                   op.y, op.y + op.height)
    x_edge = op.x if side == "left" else op.x + op.width
    inner = sc.project(np.array([[x_edge, op.y], [x_edge, op.y + op.height]]),
                       depth=op.depth)

    w = MainWindow()
    w.resize(1600, 950)
    w.show()
    step = [0]

    def settle(seconds=0.0):
        end = time.monotonic() + seconds
        while True:
            app.processEvents()
            if time.monotonic() >= end:
                break
            time.sleep(0.02)

    column = w.findChild(QScrollArea)

    def shot(name: str, note: str, panel=None) -> None:
        if panel is not None:
            settle()
            column.ensureWidgetVisible(panel, 0, 0)
        settle(args.pause if args.show else 0.05)
        step[0] += 1
        path = shots / f"{step[0]:02d}-{name}.png"
        w.grab().save(str(path))
        print(f"[{step[0]}] {note}\n    {path}")

    def wait(predicate, what: str, seconds=120.0) -> None:
        deadline = time.monotonic() + seconds
        while not predicate():
            if time.monotonic() > deadline:
                raise SystemExit(f"не дождались: {what}; статус — {w.status_label.text()}")
            settle(0.05)

    def zoom_to(x, y, scale):
        w.canvas.set_view_scale(scale)
        w.canvas.centerOn(QPointF(x + 0.5, y + 0.5))

    w.open_image(image)
    wait(lambda: w.session.frame is not None, "фаза кадра")
    w.canvas.fit_to_window()
    shot("frame-and-preliminary-verdict",
         "Снимок открыт: кадр конвейера, пригодная по углу область, предварительный вердикт")

    w.set_mode("base")
    for x, y in base_px:
        zoom_to(x, y, 1.0)
        w.handle_click(x, y, 1.0)
    w.side.base.set_values(span_mm=span_mm)
    w.apply_base()
    w.canvas.fit_to_window()
    shot("base", f"Опорная база {span_mm:.0f} мм: σ масштаба и начало отсчёта на панели",
         w.side.base)
    wait(lambda: w.rectified is not None, "выровненный растр")
    shot("rectified", "Выровненный вид рядом с кадром")

    w.side.marks.new_mark.click()
    for x, y in corners:
        zoom_to(x, y, MARK_SCALE)
        w.handle_click(x, y, MARK_SCALE)
    w.side.marks.side.setCurrentIndex(w.side.marks.side.findData(side))
    w.side.marks.pick_reveal.click()
    for x, y in inner:
        zoom_to(x, y, MARK_SCALE)
        w.handle_click(x, y, MARK_SCALE)
    cx, cy = corners.mean(axis=0)
    zoom_to(cx, cy, 0.6)
    shot("opening-with-reveal",
         f"Проём и кромка откоса ({side}) при увеличении {MARK_SCALE:g}:1", w.side.marks)

    model = w.compute()
    if model is None:
        raise SystemExit(f"модель не посчитана: {w.session.error}")
    shot("result-panel", "Панель результата: миллиметры с σ", w.side.result)

    mark_id = w.current_mark
    x, y = corners[1]
    w.drop_point(f"{mark_id}:corner:1", x + 12.0, y, MARK_SCALE)
    shot("edited-corner", "Угол сдвинут на 12 px: полный пересчёт, габарит изменился",
         w.side.result)
    w.drop_point(f"{mark_id}:corner:1", x, y, MARK_SCALE)

    session_path = w.save_session(out / "facade_demo.session.json")
    exported = w.export(out)
    shot("exported", "Экспорт JSON, marks.json, DXF и строка CLI", w.side.result)

    element = next(e for e in w.session.model.elements if e.id == mark_id)
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
    print(f"Экспорт: {exported.json_path}\n         {exported.dxf_path}")
    print(f"Команда CLI, воспроизводящая тот же JSON:\n  {exported.cli_line}")
    if args.show:
        print("\nОкно остаётся открытым — можно продолжать руками.")
        return app.exec()
    w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
