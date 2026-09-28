"""Демонстрация геометрического ядра одной командой: косой кадр → миллиметры фасада.

Скрипт собирает синтетическую сцену с ИЗВЕСТНОЙ геометрией, рендерит косой кадр,
пишет разметку проёма и прогоняет поставляемый CLI `facade-digitize`, после чего
сверяет выход с истиной сцены и печатает сводку. Нужен он затем, чтобы показать
результат ядра без ручного переноса шести чисел из одной команды в другую:
воспроизведение отчёта `docs/superpowers/plans/results/task-18-oblique-demo.md`
требовало сперва напечатать спроецированные точки, а потом вписать их в аргументы.

**Прогоняется именно CLI, а не параллельный путь.** Скрипт вызывает
`facade_digitizer.pipeline.run.main` — ту же точку входа, что и команда
`facade-digitize`. Второй, независимой сборки конвейера здесь сознательно нет:
разойдись она с поставляемой, расхождение было бы невидимо, а демонстрация
показывала бы работу кода, которым никто не пользуется.

**Чего эта демонстрация НЕ показывает — и это главное.** Точности на реальных
данных. Геометрия сцены известна точно; шума матрицы, дисторсии объектива и
неплоскостности стены в ней нет вовсе. Совпадение габарита с истиной до
полуметра миллиметра — свойство сцены, а не измерение по фотографии. Сверх того,
разметка здесь получается ПРОЕЦИРОВАНИЕМ истины, то есть оператор в ней идеален:
σ в выходном файле моделирует его промах, но на входе промаха нет. Для реальных
снимков есть `scripts/smoke_cmp.py` и отчёты по CMP; там, наоборот, нет истинных
размеров. Ни то, ни другое не заменяет натурных измерений с эталоном, которых у
проекта пока нет.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from facade_digitizer.geometry.parallax import visible_reveal_side
from facade_digitizer.pipeline.io import load_image, save_image
from facade_digitizer.pipeline.run import main as cli_main
from facade_digitizer.synth.scene import Opening, SyntheticScene

#: Камера и кадр сцены — те же, на которых работают тесты
#: (`tests/test_synth.py`). Значения повторены здесь, а не импортированы из
#: тестов: скрипт поставляется вместе с пакетом, а тесты — нет, и зависимость
#: поставляемого кода от тестового модуля сломала бы его при установке пакета
#: без исходников. Расхождение с тестами безвредно: сцена здесь нужна как
#: предъявляемый пример, а не как проверка.
K = np.array([[3600.0, 0.0, 2640.0], [0.0, 3600.0, 1978.0], [0.0, 0.0, 1.0]])
SIZE = (5280, 3956)

FACADE_WIDTH_MM = 20000.0
FACADE_HEIGHT_MM = 15000.0

#: Проём сцены: те же габариты, что у тестов стыка `run → elements`.
OPENING = {"x": 4000.0, "y": 3000.0, "width": 1460.0, "height": 1900.0}

#: Ракурс по умолчанию — ракурс отчёта задачи 18: камера смещена на 3 м вправо и
#: 2 м вниз от центра фасада, дистанция 20 м. На нём кадр заведомо косой, а шлюз
#: качества заведомо отбраковывает снимок по разрешению — обе ветви видны сразу.
DEFAULT_VIEW = {"dx": 3000.0, "dy": -2000.0, "dist": 20000.0, "depth": 150.0}

#: Разрешение выровненного растра, мм на пиксель. Только для просмотра
#: оператором: измерение выполняется по ИСХОДНОМУ снимку (спецификация, п. 6.5).
RASTER_MM_PER_PX = 10.0


def build_scene(dx: float, dy: float, dist: float, depth: float) -> SyntheticScene:
    """Сцена с одним проёмом заданного заглубления."""
    ops = [Opening(depth=depth, **OPENING)]
    return SyntheticScene.looking_at_centre(FACADE_WIDTH_MM, FACADE_HEIGHT_MM, ops,
                                            dist, dx, dy, K, SIZE)


def reveal_marks(sc: SyntheticScene) -> tuple[dict, str]:
    """Разметка видимой грани откоса — той, которую камера с этой позы реально видит.

    Сторону выбирает не скрипт, а `geometry.parallax.visible_reveal_side` по позе
    камеры: с ближней стороны проёма грань откоса ЗАКРЫТА собственной стеной, и
    разметить её нельзя. Тот же вызов делает `pipeline.elements`, поэтому
    подсунуть здесь не ту сторону и получить уверенное число невозможно —
    конвейер отдаст `origin = "unavailable"`.
    """
    op = sc.openings[0]
    cam = sc.camera_on_plane()
    x_left, x_right = op.x, op.x + op.width
    y_bottom, y_top = op.y, op.y + op.height
    side, _ = visible_reveal_side(cam, x_left, x_right, y_bottom, y_top)

    x = x_left if side == "left" else x_right
    edge_mm = np.array([[x, y_bottom], [x, y_top]])
    inner_px = sc.project(edge_mm, depth=op.depth)
    return {"side": side,
            "inner_edge_px": [[float(u), float(v)] for u, v in inner_px]}, side


def operator_base(sc: SyntheticScene) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Опорная база оператора: САМАЯ ДЛИННАЯ видимая горизонтальная хорда фасада.

    Брать углы фасада нельзя, и нижнюю сторону тоже нельзя. С близкой дистанции
    фасад в кадр не влезает, и угол проецируется за его пределы — такую точку
    конвейер отвергает по контракту («оператор указывает точки на снимке»), и
    отвергает правильно: опорный размер, снятый с точки, которой на снимке нет,
    есть выдумка. Нижней стороны при взгляде на центр фасада с близкой дистанции
    может не быть в кадре целиком, поэтому перебирается высота: оператор указал бы
    две точки на том уровне, который видит.

    Берётся самая длинная из видимых хорд, потому что σ масштаба обратна длине
    базы (п. 6.3): короткая база при том же промахе клика даёт худший масштаб.

    Возвращается пара точек в пикселях, ДЕЙСТВИТЕЛЬНОЕ расстояние между ними в
    миллиметрах и левый конец хорды в координатах фасада. Расстояние — не ширина
    фасада: подставить ширину фасада под базу, которая короче, значило бы завысить
    масштаб ровно во столько же раз.

    Левый конец возвращается затем, что он и есть НАЧАЛО ОТСЧЁТА выходного файла
    (`--origin-px`, `facade.origin = "operator_reference"`). Система координат
    фасада привязана к точке оператора, а не к углу стены, поэтому сверять
    абсолютные координаты выхода с истиной сцены можно только с этим сдвигом.
    Без него поза камеры расходится с истиной на всю высоту базы — измерено: при
    базе на высоте 13250 мм C_y «ошибается» на 296 %, хотя ошибки нет вовсе.
    """
    w_px, h_px = SIZE
    xs = np.linspace(0.0, FACADE_WIDTH_MM, 401)
    best: tuple[np.ndarray, float, tuple[float, float]] | None = None
    for y in np.linspace(0.0, FACADE_HEIGHT_MM, 61):
        pts = sc.project(np.column_stack([xs, np.full_like(xs, y)]))
        inside = ((pts[:, 0] >= 0.0) & (pts[:, 0] <= w_px - 1)
                  & (pts[:, 1] >= 0.0) & (pts[:, 1] <= h_px - 1))
        idx = np.flatnonzero(inside)
        if idx.size < 2:
            continue
        lo, hi = int(idx[0]), int(idx[-1])
        span = float(xs[hi] - xs[lo])
        if best is None or span > best[1]:
            best = (pts[[lo, hi]], span, (float(xs[lo]), float(y)))
    if best is None:
        raise SystemExit(
            "ни одна горизонтальная хорда фасада не попадает в кадр двумя точками: "
            "опорную базу задать нечем. Отодвиньте камеру (--dist) или уменьшите "
            "смещение (--dx, --dy).")
    return best


def write_inputs(sc: SyntheticScene, out_dir: Path, stem: str, *,
                 with_reveal: bool) -> tuple[Path, Path, np.ndarray, float,
                                             tuple[float, float], str | None]:
    """Кадр и разметка на диск. Точки берутся проецированием истины сцены."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{stem}.png"
    save_image(image_path, sc.render())

    mark: dict = {"class": "window", "mounting": "embedded",
                  "edge_type": "sharp_wall_edge",
                  "corners_px": [[float(u), float(v)]
                                 for u, v in sc.project(sc.openings[0].corners_mm())]}
    side = None
    if with_reveal:
        mark["reveal"], side = reveal_marks(sc)

    marks_path = out_dir / "marks.json"
    marks_path.write_text(json.dumps([mark], ensure_ascii=False, indent=2),
                          encoding="utf-8")

    # Опорная база оператора по нижней стороне фасада. Начало отсчёта — её левый
    # конец.
    base_px, span_mm, origin_mm = operator_base(sc)
    return image_path, marks_path, base_px, span_mm, origin_mm, side


def run_cli(image_path: Path, marks_path: Path, base_px: np.ndarray,
            span_mm: float, out_dir: Path) -> int:
    """Поставляемый CLI с аргументами, посчитанными из сцены."""
    (x0, y0), (x1, y1) = base_px
    # Группами, а не плоским списком: у ключей разная арность (`--origin-px`
    # берёт две величины, `--span-px` — четыре, `--save-rectified` — ни одной), и
    # печать плоского списка парами рвала строки посреди аргумента, выдавая
    # команду, которую нельзя скопировать.
    groups = [[str(image_path)],
              ["--raster-mm-per-px", str(RASTER_MM_PER_PX)],
              ["--origin-px", str(x0), str(y0)],
              ["--span-px", str(x0), str(y0), str(x1), str(y1)],
              ["--span-mm", str(span_mm)],
              ["--marks", str(marks_path)],
              ["--save-rectified"],
              ["--out-dir", str(out_dir)]]
    print("Команда, эквивалентная этому прогону:")
    print("  facade-digitize \\\n    " + " \\\n    ".join(" ".join(g) for g in groups))
    print()
    return cli_main([arg for g in groups for arg in g])


def mask_coverage(path: Path) -> float | None:
    """Доля площади выровненного растра, покрытая фасадом."""
    if not path.exists():
        return None
    mask = load_image(path)[0]
    return float(np.count_nonzero(mask) / mask.size)


def _row(name: str, truth: float, measured: float | None,
         sigma: float | None) -> str:
    if measured is None:
        return f"  {name:<22} {truth:>10.1f}   {'—':>10}   {'—':>8}   не выпущено"
    delta = measured - truth
    verdict = ""
    if sigma is not None and sigma > 0:
        verdict = "в пределах σ" if abs(delta) <= sigma else f"вне σ ({abs(delta) / sigma:.1f}σ)"
    sig = f"{sigma:>8.1f}" if sigma is not None else f"{'—':>8}"
    return f"  {name:<22} {truth:>10.1f}   {measured:>10.1f}   {sig}   {delta:+7.1f}  {verdict}"


def report(sc: SyntheticScene, result_path: Path, out_dir: Path, stem: str,
           origin_mm: tuple[float, float], side: str | None) -> None:
    """Сводка: истина сцены против выхода конвейера.

    Истина приводится к системе координат ВЫХОДНОГО ФАЙЛА: её начало — точка
    оператора (`facade.origin = "operator_reference"`), а не угол стены, поэтому
    все абсолютные величины сцены сдвигаются на `origin_mm`. Габариты от сдвига
    не зависят и сверяются как есть.
    """
    d = json.loads(result_path.read_text(encoding="utf-8"))
    img = d["images"][0]
    op = sc.openings[0]
    cam = sc.camera_on_plane()
    ox, oy = origin_mm

    print(f"Выход: {result_path}")
    print(f"Схема: {d['schema_version']}   режим: {d['mode']}")
    print(f"Начало отсчёта: {d['facade']['origin']} — точка оператора в "
          f"({ox:.0f}, {oy:.0f}) мм от угла стены.")
    print("Истина сцены ниже приведена к этому же началу.")
    print()

    print("ПОЗА КАМЕРЫ, восстановленная из самого кадра (мм)")
    pose = img["pose_to_facade"]
    for name, truth, key in (("C_x", cam.cx - ox, "cx"), ("C_y", cam.cy - oy, "cy"),
                             ("C_z (дистанция)", cam.cz, "cz")):
        got = pose[key]
        rel = f"  ({abs(got - truth) / abs(truth):.2%})" if truth else ""
        print(f"  {name:<22} {truth:>10.1f}   {got:>10.1f}   {got - truth:+7.1f}{rel}")
    print(f"  наклон theta_cam       {'':>10}   {img['theta_cam_deg']:>10.2f}°")
    print("  Расхождение объяснимо тем, что K взята из типового поля зрения")
    print(f"  (camera.calibration = \"{img['camera']['calibration']}\"), а не из истинной")
    print("  матрицы сцены: снимок записан без EXIF.")
    print()

    print("ГАБАРИТ ПРОЁМА, измеренный по ИСХОДНОМУ снимку (мм)")
    print(f"  {'величина':<22} {'истина':>10}   {'измерено':>10}   {'σ':>8}   отклонение")
    if d["elements"]:
        el = d["elements"][0]
        size = el["size_mm"]
        print(_row("ширина", op.width, size["width"], size["sigma_width"]))
        print(_row("высота", op.height, size["height"], size["sigma_height"]))

        # ПОЛОЖЕНИЕ, а не только габарит: цель проекта — координаты в метрической
        # системе, и совпадение габаритов о положении ничего не говорит (разность
        # сторон от сдвига не зависит вовсе). σ положения схема не выпускает —
        # `size_mm` несёт σ только габаритов, — поэтому столбец пуст, а не выдуман.
        contour = np.asarray(el["contour_mm"], dtype=float)
        print(_row("левая кромка X", op.x - ox, float(contour[:, 0].min()), None))
        print(_row("нижняя кромка Y", op.y - oy, float(contour[:, 1].min()), None))

        print(f"  угол визирования на проём: theta = {el['theta']['full_deg']:.2f}° "
              f"(худшая точка контура)")
        print(f"  кромка: edge_reference = \"{el['edge_reference']}\", "
              f"origin = \"{el['origin']}\"")

        print()
        print("ЗАГЛУБЛЕНИЕ по видимой грани откоса (мм)")
        recess = el["recess"]
        if recess is None:
            print("  грань откоса не разметена — глубина не вычислялась")
        elif recess.get("origin") == "measured_from_reveal":
            print(_row(f"глубина (грань {side})", op.depth, recess["value_mm"],
                       recess.get("sigma_mm")))
            if recess.get("theta_perp_deg") is not None:
                print(f"  theta_perp = {recess['theta_perp_deg']:.2f}°, "
                      f"datum = \"{recess.get('datum')}\"")
        else:
            print(f"  origin = \"{recess.get('origin')}\" — глубина НЕ выпущена.")
            print("  Это штатный отказ: конвейер называет причину, а не выдаёт число.")

        if el["meets_tolerance"] is None:
            print()
            print("  meets_tolerance = null: признак соответствия допуску отсутствует.")
            print("  Габарит и σ при этом выпущены — отбраковка помечает, а не")
            print("  уничтожает улику (см. run.GATE_REJECT_WITHHOLDS_TOLERANCE).")
    else:
        print("  элементов в выходе нет")
    print()

    print("ШЛЮЗ КАЧЕСТВА")
    q = img["quality"]
    print(f"  вердикт: {q['verdict']}")
    print(f"  GSD: {q['gsd_mm_px_min']:.2f}…{q['gsd_mm_px_max']:.2f} мм/px, "
          f"theta_p95 = {q['theta_field_deg_p95']:.1f}°")
    for reason in q["reasons"]:
        print(f"  — {reason}")
    print()

    scale = d["facade"]["scale"]
    print(f"МАСШТАБ: источник \"{scale['source']}\", sigma_rel = {scale['sigma_rel']:.4%}, "
          f"meets_tolerance = {scale['meets_tolerance']}")

    cover = mask_coverage(out_dir / f"{stem}_rectified_mask.png")
    if cover is not None:
        print(f"ОХВАТ выровненного растра: {cover:.1%} площади — косой ракурс "
              "оставляет пустые углы,")
        print("  и маска называет их по имени, а не отдаёт серый фон как измеренный фасад.")
    print()
    print("РАСТРЫ для просмотра:")
    print(f"  исходный косой кадр : {out_dir / f'{stem}.png'}")
    print(f"  выровненный         : {out_dir / f'{stem}_rectified.png'}")
    print(f"  маска охвата        : {out_dir / f'{stem}_rectified_mask.png'}")
    print()
    print("ЧЕГО ЭТА ДЕМОНСТРАЦИЯ НЕ ПОКАЗЫВАЕТ: точности на реальных данных.")
    print("Геометрия сцены известна точно, шума матрицы и дисторсии в ней нет, а")
    print("разметка получена проецированием истины — оператор здесь идеален.")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="demo_synthetic",
        description="Косой синтетический кадр через поставляемый CLI, с поверкой "
                    "по истине сцены (задача 18).")
    p.add_argument("--out-dir", type=Path, default=Path("demo_out"),
                   help="куда писать кадр, разметку, выход и растры")
    p.add_argument("--stem", default="synthetic_oblique", help="имя файлов без расширения")
    p.add_argument("--dx", type=float, default=DEFAULT_VIEW["dx"],
                   help="смещение камеры вправо от центра фасада, мм")
    p.add_argument("--dy", type=float, default=DEFAULT_VIEW["dy"],
                   help="смещение камеры вверх от центра фасада, мм")
    p.add_argument("--dist", type=float, default=DEFAULT_VIEW["dist"],
                   help="дистанция до плоскости фасада, мм")
    p.add_argument("--depth", type=float, default=DEFAULT_VIEW["depth"],
                   help="заглубление проёма, мм")
    p.add_argument("--no-reveal", action="store_true",
                   help="не разметать грань откоса: глубина не вычисляется вовсе")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    sc = build_scene(args.dx, args.dy, args.dist, args.depth)

    print(f"Сцена: фасад {FACADE_WIDTH_MM:.0f}×{FACADE_HEIGHT_MM:.0f} мм, проём "
          f"{OPENING['width']:.0f}×{OPENING['height']:.0f} мм, заглубление "
          f"{args.depth:.0f} мм")
    print(f"Камера: dx={args.dx:+.0f}, dy={args.dy:+.0f}, дистанция {args.dist:.0f} мм, "
          f"кадр {SIZE[0]}×{SIZE[1]}")
    print()

    image_path, marks_path, base_px, span_mm, origin_mm, side = write_inputs(
        sc, args.out_dir, args.stem, with_reveal=not args.no_reveal)
    print(f"Опорная база оператора: {span_mm:.0f} мм на высоте {origin_mm[1]:.0f} мм"
          + ("" if span_mm >= FACADE_WIDTH_MM
             else f" (фасад шириной {FACADE_WIDTH_MM:.0f} мм в кадр целиком не влез)"))
    print()

    code = run_cli(image_path, marks_path, base_px, span_mm, args.out_dir)

    result_path = args.out_dir / f"{args.stem}.json"
    if not result_path.exists():
        print(f"Конвейер не выпустил файл {result_path}: снимок обработать не удалось.",
              file=sys.stderr)
        return code or 1

    print("=" * 78)
    report(sc, result_path, args.out_dir, args.stem, origin_mm, side)

    # Ненулевой код CLI здесь ОЖИДАЕМ и не есть неисправность: на ракурсе по
    # умолчанию шлюз качества отбраковывает снимок по разрешению, и это ровно то,
    # что демонстрация показывает. Возвращается 0, если выходной файл выпущен.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
