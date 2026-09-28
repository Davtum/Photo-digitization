"""Команда `facade-calibrate`: профиль камеры по снимкам шахматной мишени.

План 3, задача 2. Спецификация, п. 2.2, ставит допуск на габарит под условие
«камера откалибрована», а `calibrate_from_chessboard` до сих пор не вызывалась ни из
одной команды: изготовить профиль было нечем, и условие п. 2.2 оставалось
невыполнимым для любого пользователя, кроме автора тестов.

Снимки читаются тем же `load_image`, что и в конвейере, — с тем же поворотом по
EXIF: профиль, снятый по кадрам в одной ориентации, а применяемый к кадрам в
другой, дал бы перепутанные оси K.
"""
import argparse
import re
import sys
from pathlib import Path

from facade_digitizer.pipeline.calib import calibrate_views, save_profile
from facade_digitizer.pipeline.io import load_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _board(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xх×]\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"«{value}»: ожидается число внутренних углов в виде 9x6 "
            "(столбцы x строки)")
    cols, rows = int(match.group(1)), int(match.group(2))
    if cols < 2 or rows < 2:
        raise argparse.ArgumentTypeError("у мишени не менее двух внутренних углов по стороне")
    return cols, rows


def _positive(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("размер клетки должен быть положительным")
    return number


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="facade-calibrate",
        description="Профиль камеры по снимкам шахматной мишени (план 3, задача 2).")
    p.add_argument("folder", type=Path, help="папка со снимками мишени одной камерой")
    p.add_argument("--board", type=_board, required=True,
                   help="число ВНУТРЕННИХ углов мишени: столбцы x строки, например 9x6")
    p.add_argument("--square-mm", type=_positive, required=True,
                   help="сторона клетки мишени, мм")
    p.add_argument("--model", required=True,
                   help="модель камеры ровно так, как она записана в EXIF снимков фасада: "
                        "профиль сверяется с ней, и несовпадение — отказ")
    p.add_argument("--out", type=Path, required=True, help="куда записать профиль (JSON)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths = sorted(q for q in args.folder.iterdir()
                   if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES) \
        if args.folder.is_dir() else []
    if not paths:
        print(f"в «{args.folder}» нет снимков мишени", file=sys.stderr)
        return 1

    images, sizes = [], set()
    for path in paths:
        image, _meta = load_image(path)
        images.append(image)
        sizes.add((image.shape[1], image.shape[0]))
    if len(sizes) != 1:
        listed = ", ".join(f"{w}×{h}" for w, h in sorted(sizes))
        print(f"снимки мишени разного разрешения ({listed}): профиль верен для одного "
              "разрешения, и снимки фасада должны быть сделаны в нём же", file=sys.stderr)
        return 1
    (image_size,) = sizes

    try:
        profile, accepted = calibrate_views(images, args.board, args.square_mm,
                                            image_size, args.model)
    except ValueError as error:
        print(f"калибровка не выполнена: {error}", file=sys.stderr)
        return 1

    save_profile(profile, args.out)
    fx, fy = profile.K[0][0], profile.K[1][1]
    print(f"профиль: {args.out}")
    print(f"мишень распознана на {accepted} из {len(images)} снимков")
    print(f"остаточная невязка: {profile.rms_px:.3f} px")
    print(f"fx = {fx:.1f}, fy = {fy:.1f} px; разрешение {image_size[0]}×{image_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
