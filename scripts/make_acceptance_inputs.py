"""Входные файлы ручной приёмки интерфейса оператора. План 3, задача 19.

Каждый сценарий чек-листа (`docs/operator-acceptance-checklist.md`) требует своего
снимка, и искать их по интернету перед каждой приёмкой незачем: скрипт пишет в
`--out-dir` синтетические снимки с ИЗВЕСТНОЙ геометрией и файл `ИСТИНА.txt` с
размерами, против которых сверяется результат.

* `01_фасад.jpg` + `профиль_DEMO-CAM.json` — основной путь; профиль подходит
  к снимку (та же модель в EXIF, то же разрешение, точная K сцены);
* `02_фасад_поворот_EXIF.jpg` — тот же фасад, пиксели записаны повёрнутыми, а тег
  Orientation = 6 велит повернуть их обратно; в интерфейсе кадр обязан стоять прямо;
* `03_фасад_далеко.jpg` — камера в 20 м: разрешение хуже допуска, вердикт `reject`;
* `04_без_линий.jpg` — фактура без линий фасада и светлая табличка 2:1 в перспективе:
  автоматическая плоскость не восстанавливается, нужна ручная по углам таблички;
* `профиль_чужой.json` — профиль другой камеры: интерфейс обязан отказать, назвав её.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import piexif

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_synthetic import DEFAULT_VIEW, SIZE, build_scene

from facade_digitizer.pipeline.calib import CalibrationProfile, save_profile
from facade_digitizer.pipeline.io import save_image

MODEL = "DEMO-CAM"
NEAR_VIEW = {"dx": -1500.0, "dy": -3000.0, "dist": 10000.0, "depth": 150.0}


def _jpeg(path: Path, image: np.ndarray, orientation: int = 1) -> None:
    save_image(path, image)
    exif = {"0th": {piexif.ImageIFD.Model: MODEL.encode(),
                    piexif.ImageIFD.Orientation: orientation}}
    piexif.insert(piexif.dump(exif), str(path))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_acceptance_inputs", description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("acceptance_inputs"))
    args = p.parse_args(argv)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    near = build_scene(**NEAR_VIEW)
    upright = near.render()
    _jpeg(out / "01_фасад.jpg", upright)
    # Orientation 6: показать = повернуть записанное на 90° по часовой. Значит,
    # записать надо повёрнутое на 90° ПРОТИВ часовой (np.rot90, k=1).
    _jpeg(out / "02_фасад_поворот_EXIF.jpg", np.ascontiguousarray(np.rot90(upright, 1)), 6)
    _jpeg(out / "03_фасад_далеко.jpg", build_scene(**DEFAULT_VIEW).render())

    rng = np.random.default_rng(7)
    blank = cv2.GaussianBlur(rng.normal(120, 25, (SIZE[1], SIZE[0])).clip(0, 255)
                             .astype(np.uint8), (0, 0), 3)
    plaque = np.array([[1900, 1500], [3500, 1650], [3450, 2420], [1950, 2250]], np.int32)
    cv2.fillConvexPoly(blank, plaque, 230)
    _jpeg(out / "04_без_линий.jpg", blank)

    for name, model in (("профиль_DEMO-CAM.json", MODEL), ("профиль_чужой.json", "OTHER-CAM")):
        save_profile(CalibrationProfile(model=model, K=near.K.tolist(), dist=[0.0] * 5,
                                        rms_px=0.2, image_size=tuple(SIZE)), out / name)

    op = near.openings[0]
    (out / "ИСТИНА.txt").write_text(
        f"Проём (все снимки фасада): ширина {op.width:.0f} мм, высота {op.height:.0f} мм, "
        f"заглубление {op.depth:.0f} мм.\n"
        f"Опорная база для приёмки: НИЖНЯЯ кромка этого проёма, {op.width:.0f} мм "
        "(короткая база даёт σ масштаба больше — это ожидаемо).\n"
        "Табличка на 04_без_линий.jpg: отношение сторон 2:1 (ширина / высота).\n"
        "Углы таблички в пикселях кадра: "
        + ", ".join(f"({x}, {y})" for x, y in plaque) + "\n",
        encoding="utf-8")
    for path in sorted(out.iterdir()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
