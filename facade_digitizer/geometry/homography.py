"""Гомография приведения к плоскости фасада и восстановление позы камеры."""
import numpy as np

from .camera import CameraOnPlane


def _plane_axes(vh, vv, K):
    Kinv = np.linalg.inv(K)
    d1 = Kinv @ np.asarray(vh, dtype=float)
    d1 = d1 / np.linalg.norm(d1)
    d2 = Kinv @ np.asarray(vv, dtype=float)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / np.linalg.norm(d2)
    return d1, d2


def homography_from_vanishing_points(vh, vv, K, image_size):
    """Изображение -> фронто-параллельный вид, с однозначно заданной ориентацией осей.

    Знак точки схода произволен (v и -v — одна точка), поэтому осям нужно
    доопределение. Существенно, что оси связаны: третья строка гомографии есть
    d1 x d2, поэтому смена знака d1 переворачивает не X, а Y. Физически допустимы
    лишь две комбинации — (d1, d2) и (-d1, -d2), различающиеся поворотом на 180°;
    остальные дают зеркальное отражение, которого при взгляде спереди быть не может.

    Порядок: сначала фиксируется знак нормали (знаменатель гомографии положителен
    для видимых точек), затем из двух оставшихся вариантов выбирается тот, где ось X
    растёт вместе с x изображения.
    """
    d1, d2 = _plane_axes(vh, vv, K)
    Kinv = np.linalg.inv(K)
    w, h = image_size

    centre_ray = Kinv @ np.array([w / 2.0, h / 2.0, 1.0])
    if np.cross(d1, d2) @ centre_ray < 0:
        d1 = -d1  # переворачивает нормаль, сохраняя правую тройку

    def build(a1, a2):
        return np.column_stack([a1, a2, np.cross(a1, a2)]).T @ Kinv

    H = build(d1, d2)
    probe = np.array([[w / 2.0, h / 2.0, 1.0], [w / 2.0 + 50.0, h / 2.0, 1.0]]) @ H.T
    probe = probe[:, :2] / probe[:, 2:3]
    if probe[1, 0] < probe[0, 0]:
        H = build(-d1, -d2)

    return H


def foot_point_in_rectified(H, vh, vv, K):
    """Опорная точка F в ректифицированных координатах.

    Луч из центра камеры вдоль нормали к фасаду пересекает плоскость ровно в F,
    и все точки этого луча имеют один образ — точку схода нормали v3 = K·(d1 x d2).
    Следовательно F = H·v3.
    """
    d1, d2 = _plane_axes(vh, vv, K)
    v3 = K @ np.cross(d1, d2)
    p = H @ v3
    if abs(p[2]) < 1e-12:
        raise ValueError("точка схода нормали вырождена: фасад строго фронтален")
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def rectified_to_facade_mm(points_rect, origin_rect, mm_per_rect_unit):
    """Ректифицированные координаты -> миллиметры в системе фасада.

    Ректифицированный растр — изображение, его ось Y направлена вниз; ось Y фасада
    направлена вверх (глобальное ограничение). Переворот выполняется здесь, в одном
    месте, а не внутри гомографии: гомография остаётся ориентацию сохраняющей, иначе
    получилось бы зеркальное отражение фасада.
    """
    ox, oy = origin_rect
    p = np.atleast_2d(np.asarray(points_rect, dtype=float))
    return np.column_stack([(p[:, 0] - ox) * mm_per_rect_unit,
                            -(p[:, 1] - oy) * mm_per_rect_unit])


def camera_pose(H, vh, vv, K, mm_per_rect_unit, origin_rect):
    """Поза камеры относительно плоскости фасада в миллиметрах.

    В системе координат, задаваемой гомографией из точек схода, опорная точка F
    лежит ровно в начале (H·v3 = (0, 0)), а одна ректифицированная единица
    соответствует расстоянию от камеры до плоскости.
    """
    foot_mm = rectified_to_facade_mm([[0.0, 0.0]], origin_rect, mm_per_rect_unit)[0]
    return CameraOnPlane(cx=float(foot_mm[0]), cy=float(foot_mm[1]),
                         cz=float(mm_per_rect_unit))


def homography_from_four_points(image_pts, aspect_ratio=None, size_mm=None,
                                assume_calibrated=False, K=None, image_size=None):
    """Ручной вариант: четыре точки плюс одно из трёх доопределений."""
    import cv2

    given = sum(x is not None for x in (aspect_ratio, size_mm)) + int(assume_calibrated)
    if given == 0:
        raise ValueError(
            "четырёх точек недостаточно: задайте aspect_ratio, size_mm либо "
            "assume_calibrated=True вместе с K"
        )
    if assume_calibrated:
        if K is None or image_size is None:
            raise ValueError("для assume_calibrated нужны K и image_size")
        src = np.asarray(image_pts, dtype=float)
        l_top = np.cross(np.append(src[0], 1.0), np.append(src[1], 1.0))
        l_bot = np.cross(np.append(src[3], 1.0), np.append(src[2], 1.0))
        l_lft = np.cross(np.append(src[0], 1.0), np.append(src[3], 1.0))
        l_rgt = np.cross(np.append(src[1], 1.0), np.append(src[2], 1.0))
        vh = np.cross(l_top, l_bot)
        vv = np.cross(l_lft, l_rgt)
        return homography_from_vanishing_points(vh, vv, K, image_size)

    w, h = size_mm if size_mm is not None else (float(aspect_ratio), 1.0)
    src = np.asarray(image_pts, dtype=np.float32)
    dst = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst).astype(float)
