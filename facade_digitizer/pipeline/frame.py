"""Единый кадр: то, по чему считает конвейер, и то, что видит оператор. План 3, задача 3.

Конвейер измеряет по кадру, **повёрнутому по EXIF и исправленному от дисторсии**, и
все пиксельные координаты разметки — в его пикселях. Просмотрщик, показывающий файл
как есть, отправлял бы клики не туда без единой ошибки: рецензия плана 3 измерила
расхождение 13–232 px при k1 = −0.08 и перевёрнутые оси при EXIF Orientation = 6
(`QPixmap` не поворачивает кадр, `cv2.imdecode` поворачивает).

Поэтому кадр читается ОДНОЙ функцией, `load_frame`, и серый (для конвейера) и
цветной (для глаза) варианты получают один поворот, одну K и одну дисторсию. Второго
пути чтения снимка ни у конвейера, ни у интерфейса нет.

Соглашение о пикселях — OpenCV: центры пикселей в целых координатах, допустимая
точка лежит в `[0, w−1] × [0, h−1]`.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calib import intrinsics_from_meta, load_profile, undistort
from .io import AXES_SWAPPED, CameraMeta, _read_exif, decode, read_bytes


@dataclass(frozen=True)
class Frame:
    """Снимок в том виде, в каком его измеряют.

    `gray` — массив, по которому считает конвейер; `color` — тот же кадр в цвете для
    показа оператору (BGR), пиксель в пиксель с `gray`, либо `None`, если цвет не
    запрашивали. `orientation` — тег EXIF, УЖЕ применённый при декодировании.
    """

    gray: np.ndarray
    color: np.ndarray | None
    meta: CameraMeta
    K: np.ndarray
    dist: list[float]
    source: str
    orientation: int

    @property
    def size(self) -> tuple[int, int]:
        """(ширина, высота) кадра после поворота."""
        return self.gray.shape[1], self.gray.shape[0]


def load_frame(path, profile_path=None, *, with_color: bool = True) -> Frame:
    """Снимок → кадр: поворот по EXIF, внутренние параметры, снятие дисторсии.

    Файл читается один раз; серый и цветной варианты декодируются из тех же байтов,
    и каждый — с поворотом по EXIF, который `cv2.imdecode` применяет сам. Серый
    декодируется ровно так, как читал снимок конвейер до этой задачи
    (`IMREAD_GRAYSCALE`), а не переводом цветного: иначе на JPEG менялись бы
    значения пикселей, а с ними — результаты всего прежнего набора.

    `with_color=False` — для конвейера: цвет ему не нужен, а декодирование 20 Мп в
    цвете и снятие с него дисторсии стоят времени без пользы.
    """
    path = Path(path)
    raw = read_bytes(path)
    gray = decode(raw, cv2.IMREAD_GRAYSCALE, path)
    color = decode(raw, cv2.IMREAD_COLOR, path) if with_color else None
    if color is not None and color.shape[:2] != gray.shape:
        raise ValueError(
            f"серый и цветной варианты снимка разошлись по размеру ({gray.shape} против "
            f"{color.shape[:2]}): {path}")

    h, w = gray.shape[:2]
    meta = _read_exif(path, (w, h))
    K, source = intrinsics_from_meta(meta, profile_path)
    dist = list(load_profile(profile_path).dist) if source == "target" else [0.0] * 5

    gray = undistort(gray, K, dist)
    if color is not None:
        color = undistort(color, K, dist)
    return Frame(gray=gray, color=color, meta=meta, K=K, dist=dist, source=source,
                 orientation=meta.orientation)


def _oriented(points: np.ndarray, orientation: int, raw_w: int, raw_h: int) -> np.ndarray:
    """Точки СЫРОГО растра файла → точки кадра после поворота по EXIF.

    Восемь значений тега — все композиции поворота на кратный 90° угол и отражения.
    Формулы сверены не с описанием тега, а с тем, куда кладёт пиксель сам декодер
    (`tests/test_frame.py`, по всем восьми значениям).
    """
    x, y = points[:, 0], points[:, 1]
    xm, ym = raw_w - 1 - x, raw_h - 1 - y
    table = {
        1: (x, y),
        2: (xm, y),
        3: (xm, ym),
        4: (x, ym),
        5: (y, x),
        6: (raw_h - 1 - y, x),
        7: (raw_h - 1 - y, xm),
        8: (y, xm),
    }
    if orientation not in table:
        raise ValueError(f"неизвестное значение EXIF Orientation: {orientation}")
    nx, ny = table[orientation]
    return np.column_stack([nx, ny])


def raw_points_to_frame(points_raw, frame: Frame) -> np.ndarray:
    """Точки, снятые по ИСХОДНОМУ файлу, → точки кадра, в котором считает конвейер.

    Нужна для разметки, сделанной вне приложения (по файлу в стороннем просмотрщике):
    сперва поворот по EXIF, затем снятие дисторсии той же K, в которой её снимает
    `undistort` (`cv2.undistortPoints` с `P = K` — точечный двойник растрового
    `cv2.undistort` без новой матрицы камеры). Порядок существенен: профиль снят по
    кадрам в той ориентации, в какой их отдаёт декодер (`facade-calibrate` читает
    мишень тем же `load_image`), значит и дисторсия задана в повёрнутом кадре.
    """
    pts = np.asarray(points_raw, dtype=float).reshape(-1, 2)
    w, h = frame.size
    raw_w, raw_h = (h, w) if frame.orientation in AXES_SWAPPED else (w, h)
    pts = _oriented(pts, frame.orientation, raw_w, raw_h)
    coeffs = np.asarray(frame.dist, dtype=float)
    if np.any(coeffs):
        pts = cv2.undistortPoints(pts.reshape(-1, 1, 2), frame.K, coeffs,
                                  P=frame.K).reshape(-1, 2)
    return pts
