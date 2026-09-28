"""Внутренние параметры камеры и снятие дисторсии. Спецификация, п. 4.2."""
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .io import CameraMeta

DEFAULT_DIAGONAL_FOV_DEG = 84.0  # типично для DJI Zenmuse L2 и Mavic 3

MIN_CALIBRATION_VIEWS = 5

#: Источник матрицы K по происхождению ширины матрицы. Пути различаются по
#: надёжности: измеренная ширина, выведенная из 35-мм эквивалента и взятая из
#: таблицы моделей — не одно и то же.
#:
#: Уточнение живёт в возвращаемом `intrinsics_from_meta` источнике и доходит до
#: `pipeline.run._camera_record`, где сознательно схлопывается в `"exif"`:
#: `camera.calibration` схемы (раздел 10) знает ровно три происхождения, и поля
#: под уточнение в ней нет. Бюджет п. 6.1 от уточнения не зависит — ε = 2 %
#: одинакова для всех трёх путей EXIF (`elements.FOCAL_ERROR_BY_CALIBRATION`),
#: потому что ни один из них не есть калибровка по мишени.
SOURCE_BY_SENSOR_WIDTH_SOURCE = {
    None: "exif",
    "focal_plane": "exif:focal_plane",
    "crop_factor": "exif:crop_factor",
    "model_table": "exif:model_table",
}


@dataclass(frozen=True)
class CalibrationProfile:
    """Результат разовой калибровки камеры по мишени. Спецификация, п. 4.2."""

    model: str
    K: list[list[float]]
    dist: list[float]
    rms_px: float
    image_size: tuple[int, int]


def save_profile(profile: CalibrationProfile, path) -> None:
    """Запись профиля камеры в JSON."""
    Path(path).write_text(json.dumps(asdict(profile), ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_profile(path) -> CalibrationProfile:
    """Чтение профиля камеры из JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["image_size"] = tuple(data["image_size"])
    return CalibrationProfile(**data)


def calibrate_from_chessboard(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_mm: float,
    image_size: tuple[int, int],
    model: str,
) -> CalibrationProfile:
    """Калибровка по снимкам шахматной мишени; см. `calibrate_views`."""
    profile, _accepted = calibrate_views(images, pattern, square_mm, image_size, model)
    return profile


def calibrate_views(
    images: list[np.ndarray],
    pattern: tuple[int, int],
    square_mm: float,
    image_size: tuple[int, int],
    model: str,
) -> tuple[CalibrationProfile, int]:
    """Калибровка по снимкам шахматной мишени. Выполняется один раз на аппарат.

    Ошибка фокусного расстояния при наклоне искажает отношение сторон как ε·sin²θ:
    при ε = 2 % и θ = 30 % это 7.5 мм на полутораметровом окне. Поэтому камеры
    собственного парка калибруются по мишени, а не по EXIF. Спецификация, п. 4.2.

    Число поданных снимков проверяется до всякой обработки: меньше
    MIN_CALIBRATION_VIEWS видов — отказ с отдельным сообщением, не смешиваемым
    с сообщением о нераспознанной мишени.

    Возвращает профиль и число видов, на которых мишень распознана: команда
    `facade-calibrate` печатает его, потому что профиль по пяти видам из двадцати
    — не то же, что по двадцати, а сам профиль числа видов не хранит.
    """
    if len(images) < MIN_CALIBRATION_VIEWS:
        raise ValueError("для устойчивой калибровки нужно не менее 5 снимков мишени")

    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    obj_points, img_points = [], []
    for img in images:
        found, corners = cv2.findChessboardCorners(img, pattern, None)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            img, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"мишень распознана лишь на {len(obj_points)} снимках из {len(images)}")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    profile = CalibrationProfile(model=model, K=K.tolist(), dist=dist.ravel().tolist(),
                                 rms_px=float(rms), image_size=tuple(image_size))
    return profile, len(obj_points)


def _normalized_model(model) -> str:
    """Имя модели для сверки: регистр и обрамляющие пробелы значения не имеют."""
    return (model or "").strip().upper()


def intrinsics_from_meta(meta: CameraMeta, profile_path=None) -> tuple[np.ndarray, str]:
    """Матрица K и источник её происхождения.

    Приоритет: профиль калибровки по мишени; затем фокусное и ширина матрицы из
    EXIF; затем типичное поле зрения. Источник обязан попасть в выходной файл —
    без него нельзя оценить вклад ошибки фокусного в бюджет (спецификация, п. 6.1).

    Возвращаемый источник называет то, откуда параметры взяты на деле:
    "target" — профиль по мишени; "exif" и уточнённые "exif:*" — фокусное из снимка
    с указанием происхождения ширины матрицы; "database" — типовое поле зрения.

    Переданный профиль сверяется со снимком и по модели камеры, и по разрешению.
    Несовпадение — отказ, а не молчаливый переход к EXIF: профиль передан явным
    действием оператора, и противоречие между этим действием и метаданными снимка
    означает чужой файл, то есть систематически неверные внутренние параметры без
    единого признака неисправности. Кому нужен запасной источник — тот просто не
    передаёт profile_path.

    По той же причине переданный, но ОТСУТСТВУЮЩИЙ файл — отказ, а не переход к
    EXIF. Прежде здесь стояла проверка `Path(profile_path).exists()`, и
    переименованный или удалённый профиль молча превращал калибровку в `"exif"` или
    `"database"` — то есть делал ровно то, что предыдущий абзац запрещает для
    чужого файла (план 3, задача 2).
    """
    w, h = meta.image_size

    if profile_path is not None:
        if not Path(profile_path).is_file():
            raise FileNotFoundError(
                f"профиль калибровки не найден: {profile_path}. Профиль передан явно, "
                "и продолжить без него значило бы молча сменить происхождение "
                "внутренних параметров; чтобы работать без профиля, не передавайте его")
        prof = load_profile(profile_path)
        if _normalized_model(prof.model) != _normalized_model(meta.model):
            raise ValueError(
                f"профиль калиброван на камере «{prof.model}», "
                f"а снимок сделан камерой «{meta.model}»"
            )
        if tuple(prof.image_size) != (w, h):
            prof_w, prof_h = prof.image_size
            raise ValueError(
                f"профиль калиброван при разрешении {prof_w}×{prof_h}, "
                f"а снимок имеет {w}×{h}"
            )
        return np.array(prof.K, dtype=float), "target"

    if meta.focal_mm and meta.sensor_width_mm:
        fx = fy = w * meta.focal_mm / meta.sensor_width_mm
        source = SOURCE_BY_SENSOR_WIDTH_SOURCE[meta.sensor_width_source]
    else:
        diag_px = math.hypot(w, h)
        fx = fy = diag_px / (2.0 * math.tan(math.radians(DEFAULT_DIAGONAL_FOV_DEG) / 2.0))
        source = "database"

    K = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    return K, source


def undistort(image: np.ndarray, K: np.ndarray, dist: list[float]) -> np.ndarray:
    """Снятие дисторсии. При нулевых коэффициентах возвращается исходное изображение.

    Исходный массив возвращается тем же объектом: нулевые коэффициенты не должны
    приводить ни к передискретизации, ни к копированию.
    """
    coeffs = np.asarray(dist, dtype=float)
    if not np.any(coeffs):
        return image
    return cv2.undistort(image, K, coeffs)
