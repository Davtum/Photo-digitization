"""`geom.js` против Python: соглашение о пикселях, гомография, σ клика. Задача 8.

Страница переводит экран ↔ кадр сама (иначе каждый сдвиг мыши ходил бы на сервер),
и это единственная геометрия в браузере. Проверяется исполнением модуля в Node:
то же соглашение полпикселя, что у окна Qt (`test_ui_canvas` плана 3), та же
гомография, что у ядра, та же σ клика, что у `ui.zoom`.
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from facade_digitizer.geometry.homography import apply_homography
from facade_digitizer.ui import texts, zoom

GEOM = Path(__file__).resolve().parents[1] / "facade_digitizer" / "web" / "static" / "geom.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="нет Node.js")


def _js(expression: str):
    """Выражение над модулем `g` — через файл, а не `-e`: у обёрток node на Windows
    (Volta, nvm) командная строка проходит через cmd.exe, и `>` стрелочной функции
    становился перенаправлением вывода."""
    script = (f"import * as g from {json.dumps(GEOM.as_uri())};\n"
              f"console.log(JSON.stringify({expression}));\n")
    fd, name = tempfile.mkstemp(suffix=".mjs")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(script)
        out = subprocess.run([NODE, name], capture_output=True, text=True, encoding="utf-8",
                             check=True).stdout
    finally:
        os.unlink(name)
    return json.loads(out)


def _python_screen_to_frame(sx, sy, ox, oy, s, w, h):
    """Эталон — перевод окна Qt: сцена = (экран − сдвиг) / масштаб, кадр = сцена − 0.5."""
    u, v = (sx - ox) / s, (sy - oy) / s
    if not (0.0 <= u < w and 0.0 <= v < h):
        return None
    return [min(max(u - 0.5, 0.0), w - 1.0), min(max(v - 0.5, 0.0), h - 1.0)]


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0, 2.0, 3.0, 8.0])
def test_screen_point_maps_to_frame_pixel_at_several_scales(scale):
    w, h, ox, oy = 640, 480, -37.25, 12.5
    points = [(ox + 5.3 * scale, oy + 7.9 * scale), (100.0, 200.0), (ox, oy),
              (ox + w * scale - 0.01, oy + h * scale - 0.01), (ox - 1, oy), (ox + w * scale, oy)]
    got = _js("[" + ",".join(
        f"g.screenToFrame({x}, {y}, {{ox: {ox}, oy: {oy}, cssScale: {scale}}}, {w}, {h})"
        for x, y in points) + "]")
    for (x, y), result in zip(points, got):
        expected = _python_screen_to_frame(x, y, ox, oy, scale, w, h)
        if expected is None:
            assert result is None, (x, y)
        else:
            assert result == pytest.approx(expected, abs=1e-9), (x, y)


def test_half_pixel_convention_round_trips():
    got = _js("[g.frameToScreen(10, 20, {ox: 3, oy: 4, cssScale: 2}),"
              " g.screenToFrame(...g.frameToScreen(10, 20, {ox: 3, oy: 4, cssScale: 2}),"
              " {ox: 3, oy: 4, cssScale: 2}, 100, 100)]")
    assert got[0] == [3 + 10.5 * 2, 4 + 20.5 * 2]
    assert got[1] == [10, 20]


def test_view_scale_is_physical_pixels():
    assert _js("[g.physicalScale(0.8, 1.25), g.cssScaleFor(2, 1.25)]") == [1.0, 1.6]


def test_apply_h_matches_the_core():
    H = np.array([[1.2, 0.1, 30.0], [-0.05, 0.9, 12.0], [1e-4, 2e-5, 1.0]])
    pts = np.array([[0.0, 0.0], [100.0, 50.0], [640.0, 480.0]])
    got = _js("[" + ",".join(f"g.applyH({json.dumps(H.tolist())}, {x}, {y})"
                             for x, y in pts) + "]")
    assert np.allclose(got, apply_homography(H, pts), atol=1e-9)
    # Мутация, от которой это охраняет: транспонированная H промахивается.
    wrong = apply_homography(H.T, pts)
    assert not np.allclose(got, wrong, atol=1.0)


def test_sigma_matches_ui_zoom():
    scales = [0.5, 1.0, 2.0, 3.0, 4.0]
    sigma = {"screen_px": zoom.SIGMA_SCREEN_PX, "edge_px": zoom.SIGMA_EDGE_PX}
    got = _js("[" + ",".join(f"g.sigmaImagePx({s}, {json.dumps(sigma)})" for s in scales) + "]")
    assert got == pytest.approx([zoom.sigma_image_px(s) for s in scales], rel=1e-12)


@pytest.mark.parametrize("scale", [1.0, 2.0, 4.0, 0.5, 0.25, 1.5])
def test_scale_name_matches_texts(scale):
    assert _js(f"g.scaleName({scale})") == texts.scale_name(scale)
