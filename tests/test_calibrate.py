"""Профиль калибровки и команда `facade-calibrate`. План 3, задача 2.

Виды шахматной мишени рендерятся здесь же, по известной матрице K и известным
позам: восстановленная K сверяется с заданной, а не с собой.
"""
import math

import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.calib import (
    MIN_CALIBRATION_VIEWS,
    intrinsics_from_meta,
    load_profile,
)
from facade_digitizer.pipeline.calibrate import main as calibrate_main
from facade_digitizer.pipeline.io import CameraMeta, save_image

PATTERN = (9, 6)            # внутренние углы: столбцы, строки
SQUARE_MM = 25.0
SQUARE_PX = 40              # клетка на текстуре мишени
MARGIN_SQUARES = 1          # белое поле вокруг доски — без него детектор не находит края
IMAGE_SIZE = (1280, 960)
K_TRUE = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])


def _board_texture() -> np.ndarray:
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    m = MARGIN_SQUARES
    tex = np.full(((rows + 2 * m) * SQUARE_PX, (cols + 2 * m) * SQUARE_PX), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y0, x0 = (r + m) * SQUARE_PX, (c + m) * SQUARE_PX
                tex[y0:y0 + SQUARE_PX, x0:x0 + SQUARE_PX] = 0
    return tex


def _rotation(ax_deg: float, ay_deg: float) -> np.ndarray:
    ax, ay = math.radians(ax_deg), math.radians(ay_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    return ry @ rx


def _render_view(ax_deg: float, ay_deg: float, distance_mm: float) -> np.ndarray:
    """Вид мишени при заданной позе. Мишень — плоскость Z = 0 в миллиметрах.

    Начало миллиметров — первый внутренний угол, как у `objp` в
    `calibrate_from_chessboard`; текстура переводится в миллиметры матрицей S.
    """
    tex = _board_texture()
    first_corner_px = (MARGIN_SQUARES + 1) * SQUARE_PX
    mm_per_tex = SQUARE_MM / SQUARE_PX
    S = np.array([[mm_per_tex, 0, -first_corner_px * mm_per_tex],
                  [0, mm_per_tex, -first_corner_px * mm_per_tex],
                  [0, 0, 1.0]])
    R = _rotation(ax_deg, ay_deg)
    centre = np.array([(PATTERN[0] - 1) * SQUARE_MM / 2, (PATTERN[1] - 1) * SQUARE_MM / 2, 0])
    t = np.array([0.0, 0.0, distance_mm]) - R @ centre
    H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t])
    return cv2.warpPerspective(tex, H @ S, IMAGE_SIZE, flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


POSES = [(0, 0, 600), (18, 0, 620), (-18, 0, 620), (0, 20, 640), (0, -20, 640),
         (14, 14, 650), (-14, -14, 650), (12, -16, 600)]


@pytest.fixture(scope="module")
def views_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("мишень")      # кириллица в пути — намеренно
    for i, (ax, ay, dist) in enumerate(POSES):
        save_image(d / f"view_{i:02d}.png", _render_view(ax, ay, dist))
    return d


def test_calibrate_command_recovers_the_known_intrinsics(views_dir, tmp_path, capsys):
    out = tmp_path / "профиль.json"
    code = calibrate_main([str(views_dir), "--board", "9x6", "--square-mm", "25",
                           "--model", "Test Camera", "--out", str(out)])
    assert code == 0
    prof = load_profile(out)
    K = np.array(prof.K)
    assert K[0, 0] == pytest.approx(K_TRUE[0, 0], rel=0.01)
    assert K[1, 1] == pytest.approx(K_TRUE[1, 1], rel=0.01)
    assert K[0, 2] == pytest.approx(K_TRUE[0, 2], abs=5.0)
    assert K[1, 2] == pytest.approx(K_TRUE[1, 2], abs=5.0)
    assert tuple(prof.image_size) == IMAGE_SIZE
    assert prof.model == "Test Camera"
    printed = capsys.readouterr().out
    assert f"{len(POSES)} из {len(POSES)}" in printed      # число принятых видов
    assert "px" in printed                                  # остаточная невязка


def test_calibrate_command_refuses_too_few_views(tmp_path, capsys):
    d = tmp_path / "мало"
    d.mkdir()
    for i, (ax, ay, dist) in enumerate(POSES[:MIN_CALIBRATION_VIEWS - 1]):
        save_image(d / f"v{i}.png", _render_view(ax, ay, dist))
    code = calibrate_main([str(d), "--board", "9x6", "--square-mm", "25",
                           "--model", "Test Camera", "--out", str(tmp_path / "p.json")])
    assert code != 0
    assert "не менее 5" in capsys.readouterr().err
    assert not (tmp_path / "p.json").exists()


def test_calibrate_command_refuses_mixed_resolutions(views_dir, tmp_path, capsys):
    d = tmp_path / "смесь"
    d.mkdir()
    for i, (ax, ay, dist) in enumerate(POSES[:MIN_CALIBRATION_VIEWS]):
        save_image(d / f"v{i}.png", _render_view(ax, ay, dist))
    save_image(d / "small.png", cv2.resize(_render_view(0, 0, 600), (640, 480)))
    code = calibrate_main([str(d), "--board", "9x6", "--square-mm", "25",
                           "--model", "Test Camera", "--out", str(tmp_path / "p.json")])
    assert code != 0
    assert "разрешени" in capsys.readouterr().err


def test_calibrate_command_rejects_a_malformed_board(views_dir, tmp_path, capsys):
    with pytest.raises(SystemExit):
        calibrate_main([str(views_dir), "--board", "9на6", "--square-mm", "25",
                        "--model", "X", "--out", str(tmp_path / "p.json")])


def test_missing_profile_file_is_refused_by_name(tmp_path):
    """Профиль передан явным действием оператора: его отсутствие — отказ.

    Прежде `intrinsics_from_meta` проверяла `Path(profile_path).exists()` и при
    отсутствующем файле молча переходила к EXIF или таблице моделей: переименованный
    профиль превращал калибровку в `"database"` без единого признака.
    """
    meta = CameraMeta(model="unknown", focal_mm=None, sensor_width_mm=None,
                      image_size=IMAGE_SIZE, captured_at=None, gnss=None)
    missing = tmp_path / "нет_такого_профиля.json"
    with pytest.raises(FileNotFoundError, match="нет_такого_профиля"):
        intrinsics_from_meta(meta, missing)


def test_no_profile_path_still_falls_back_to_metadata():
    """Запасной путь сохраняется для того, кто профиль НЕ передавал."""
    meta = CameraMeta(model="unknown", focal_mm=None, sensor_width_mm=None,
                      image_size=IMAGE_SIZE, captured_at=None, gnss=None)
    _K, source = intrinsics_from_meta(meta, None)
    assert source == "database"
