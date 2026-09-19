import cv2
import numpy as np
import pytest

from facade_digitizer.pipeline.quality import Thresholds, assess, sharpness, usable_fraction
from facade_digitizer.schema import QualityReport
from tests.test_synth import make_scene as scene


def test_blur_lowers_sharpness_by_orders_of_magnitude():
    """Проверяется ПОРЯДОК, а не абсолютное значение: порог ещё не откалиброван."""
    img = scene().render()
    assert sharpness(img) > 20 * sharpness(cv2.GaussianBlur(img, (9, 9), 3.0))
    assert sharpness(img) > 100 * sharpness(cv2.GaussianBlur(img, (31, 31), 12.0))


def test_noise_scores_higher_than_signal():
    """Известное ограничение метрики; зафиксировано тестом, чтобы не забылось."""
    img = scene().render()
    noise = np.random.default_rng(0).integers(0, 255, img.shape, dtype=np.uint8)
    assert sharpness(noise) > sharpness(img)


def test_sharpness_is_invariant_to_brightness_scale():
    """Метрика нормирована на дисперсию яркости, поэтому не зависит от контраста.

    Без нормировки снимок, снятый в полтора раза темнее, получил бы резкость
    вдвое-вчетверо ниже при той же самой геометрии границ, и шлюз отбраковывал бы
    по яркости под видом отбраковки по смазу. Проверяется на float-копиях, чтобы
    квантование uint8 не подмешивалось в сравнение.
    """
    img = scene().render().astype(np.float64)
    assert sharpness(img * 0.5) == pytest.approx(sharpness(img), rel=1e-6)
    assert sharpness(img * 3.0) == pytest.approx(sharpness(img), rel=1e-6)


def test_ok_report_carries_its_inputs():
    """Отчёт — структура схемы (раздел 10), а не кортеж: поля должны быть на местах."""
    report = assess(scene().render(), 3.1, 4.6, 22.0)
    assert isinstance(report, QualityReport)
    assert report.verdict == "ok"
    assert report.reasons == []
    assert report.gsd_mm_px_min == 3.1
    assert report.gsd_mm_px_max == 4.6
    assert report.theta_field_deg_p95 == 22.0
    assert report.sharpness > 0.0


def test_coarse_gsd_is_rejected():
    report = assess(scene().render(), 14.0, 16.0, 22.0)
    assert report.verdict == "reject"
    assert any("разрешение" in r for r in report.reasons)


def test_steep_angle_is_degraded():
    report = assess(scene().render(), 3.1, 4.6, 44.0)
    assert report.verdict == "degraded"
    assert any("угол" in r for r in report.reasons)


def test_small_usable_fraction_is_flagged():
    report = assess(scene().render(), 3.1, 4.6, 22.0, usable=0.3)
    assert report.verdict == "degraded"
    assert any("кадра" in r for r in report.reasons)


def test_thresholds_are_injectable():
    blurred = cv2.GaussianBlur(scene().render(), (31, 31), 12.0)
    strict = Thresholds(sharpness_min=1.0, gsd_max_mm_px=5.0, theta_p95_max_deg=30.0)
    lenient = Thresholds(sharpness_min=0.0, gsd_max_mm_px=5.0, theta_p95_max_deg=30.0)
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=strict).verdict == "degraded"
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=lenient).verdict == "ok"


def test_usable_fraction_counts_angles():
    field = np.array([[10.0, 20.0], [40.0, 50.0]])
    assert usable_fraction(field, 30.0) == pytest.approx(0.5)


@pytest.fixture(scope="module")
def img():
    """Один рендер на все проверки аргументов: сам кадр в них годный и неизменный."""
    return scene().render()


def test_usable_threshold_is_injectable(img):
    """Порог доли кадра — такой же подставляемый порог, как остальные три.

    Вшитое в тело число обошла бы калибровка порогов на реальных данных.
    """
    assert assess(img, 3.1, 4.6, 22.0, usable=0.3).verdict == "degraded"
    lenient = Thresholds(usable_min_fraction=0.2)
    assert assess(img, 3.1, 4.6, 22.0, usable=0.3, thresholds=lenient).verdict == "ok"


def test_non_finite_gsd_is_a_caller_error(img):
    """NaN в разрешении — поломка расчёта, а не свойство снимка.

    Без проверки `nan > порог` ложно, причин нет, и шлюз выдаёт «годен» — отказ
    ровно в том модуле, чьё единственное назначение состоит в отлове негодного входа.
    """
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="gsd_max"):
            assess(img, 3.1, bad, 22.0)
        with pytest.raises(ValueError, match="gsd_min"):
            assess(img, bad, 4.6, 22.0)


def test_non_numeric_gsd_is_a_caller_error(img):
    """Нечисловое значение: math.isfinite на нём поднял бы TypeError вместо внятного отказа."""
    for bad in ("4.6", None, [4.6]):
        with pytest.raises(ValueError, match="gsd_max"):
            assess(img, 3.1, bad, 22.0)


def test_impossible_gsd_is_a_caller_error(img):
    """Разрешение неположительно или границы переставлены — считать по такому нечего."""
    with pytest.raises(ValueError, match="gsd_max"):
        assess(img, 3.1, -4.6, 22.0)
    with pytest.raises(ValueError, match="gsd_min"):
        assess(img, 0.0, 4.6, 22.0)
    with pytest.raises(ValueError, match="переставлены"):
        assess(img, 5.0, 4.6, 22.0)


def test_non_finite_angle_is_a_caller_error(img):
    for bad in (float("nan"), float("inf"), "22.0"):
        with pytest.raises(ValueError, match="theta_p95"):
            assess(img, 3.1, 4.6, bad)


def test_impossible_angle_is_a_caller_error(img):
    """Угол визирования к нормали вне [0°, 90°] не описывает наблюдаемый фасад."""
    with pytest.raises(ValueError, match="theta_p95"):
        assess(img, 3.1, 4.6, -1.0)
    with pytest.raises(ValueError, match="theta_p95"):
        assess(img, 3.1, 4.6, 120.0)


def test_usable_fraction_out_of_range_is_a_caller_error(img):
    """Доля кадра — это доля: вне [0, 1] и нечисловая она бессмысленна."""
    for bad in (float("nan"), float("inf"), 1.5, -0.1, None):
        with pytest.raises(ValueError, match="usable"):
            assess(img, 3.1, 4.6, 22.0, usable=bad)


def test_boundary_arguments_are_accepted(img):
    """Охрана не должна отвергать рабочий вход: крайние допустимые значения проходят."""
    assert assess(img, 3.1, 4.6, 0.0, usable=1.0).verdict == "ok"
    assert assess(img, 4.6, 4.6, 90.0, usable=0.0).verdict == "degraded"
