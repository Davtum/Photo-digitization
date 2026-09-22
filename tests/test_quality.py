import cv2
import numpy as np
import pytest

from facade_digitizer.geometry import angles
from facade_digitizer.pipeline import quality
from facade_digitizer.pipeline.quality import (
    THETA_MAX_DEG,
    Thresholds,
    assess,
    sharpness,
    usable_fraction,
)
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


def test_steep_angle_is_recorded_but_does_not_decide():
    """Крутой P95 углового поля записывается и вердикта не решает. П. 4.1.

    Прежде здесь стоял обратный тест: P95 = 44° давал `degraded`. Порог снят, потому
    что угол визирования в точке кадра равен `arctg(r / f)` и определяется
    объективом: при f = 3600 P95 по кадру равен 38.5° на ЛЮБОЙ сцене, включая
    фронтальную, так что порог отбраковывал бы всю рабочую аппаратуру. Проверка
    перевёрнута и оставлена — она и есть охрана от возвращения порога по недосмотру.
    """
    report = assess(scene().render(), 3.1, 4.6, 44.0)
    assert report.verdict == "ok"
    assert report.reasons == []
    assert report.theta_field_deg_p95 == 44.0


def test_small_usable_fraction_is_flagged():
    report = assess(scene().render(), 3.1, 4.6, 22.0, usable=0.3)
    assert report.verdict == "degraded"
    assert any("кадра" in r for r in report.reasons)


def test_thresholds_are_injectable():
    blurred = cv2.GaussianBlur(scene().render(), (31, 31), 12.0)
    strict = Thresholds(sharpness_min=1.0, gsd_max_mm_px=5.0)
    lenient = Thresholds(sharpness_min=0.0, gsd_max_mm_px=5.0)
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=strict).verdict == "degraded"
    assert assess(blurred, 3.1, 4.6, 22.0, thresholds=lenient).verdict == "ok"


def test_usable_fraction_counts_angles():
    """Доля узлов в пределах углового порога, в том числе при пороге по умолчанию.

    Умолчание проверяется отдельно и намеренно: обе прежние проверки передавали
    порог явно, отчего значение по умолчанию не читал никто, и подмена его на 90°
    проходила молча. Умолчание — не украшение сигнатуры: по нему считает угловое
    условие п. 2.2, и оно обязано совпадать с THETA_MAX_DEG.
    """
    field = np.array([[10.0, 20.0], [40.0, 50.0]])
    assert usable_fraction(field, 30.0) == pytest.approx(0.5)
    assert usable_fraction(field) == pytest.approx(0.5)
    assert THETA_MAX_DEG == 30.0


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


def test_empty_image_is_a_caller_error():
    """Пустой кадр давал nan в резкости и вердикт «с оговоркой» без внятной причины."""
    for bad in (np.zeros((0, 0), np.uint8), np.zeros((0, 16), np.uint8), np.array([])):
        with pytest.raises(ValueError, match="изображение"):
            assess(bad, 3.1, 4.6, 22.0)


def test_non_2d_image_is_a_caller_error():
    """Шлюз меряет резкость полутонового кадра; трёхканальный массив сюда не доходит."""
    with pytest.raises(ValueError, match="изображение"):
        assess(np.zeros((32, 32, 3), np.uint8), 3.1, 4.6, 22.0)
    with pytest.raises(ValueError, match="изображение"):
        assess(np.zeros(32, np.uint8), 3.1, 4.6, 22.0)


def test_non_numeric_image_is_a_caller_error():
    """Список и массив объектов: np.isfinite на них поднял бы TypeError из внутренностей."""
    for bad in ([[1, 2], [3, 4]], None, np.array([[None, 1], [2, 3]], dtype=object)):
        with pytest.raises(ValueError, match="изображение"):
            assess(bad, 3.1, 4.6, 22.0)


def test_non_finite_pixels_are_a_caller_error():
    frame = np.full((32, 32), 100.0)
    frame[5, 5] = np.nan
    with pytest.raises(ValueError, match="изображение"):
        assess(frame, 3.1, 4.6, 22.0)


def test_valid_frames_of_both_kinds_are_accepted(img):
    """Охрана не должна отвергать рабочий кадр: и uint8, и float проходят."""
    assert assess(img, 3.1, 4.6, 22.0).verdict == "ok"
    assert assess(img.astype(np.float64), 3.1, 4.6, 22.0).verdict == "ok"


def test_verdict_is_decided_by_the_condition_not_by_the_wording(img, monkeypatch):
    """`reject` ставится по ЧИСЛОВОМУ условию, а не по слову в тексте причины.

    Прежде вердикт выбирался совпадением по подстроке: `any("разрешение" in r)`.
    Охрана, которая вроде бы это стерегла
    (`test_refusal_messages_do_not_overlap`), относится к СОВСЕМ ДРУГОМУ набору
    строк — к сообщениям отказов `ValueError`, а не к причинам вердикта, — и
    непересечения причин между собой не проверяет вовсе. При этом
    `pipeline.run._needs_operator_model` уже складывает для `QualityReport` того
    же типа причину «разрешение оценено по опорной базе оператора», где то же
    слово есть, а недостаточного разрешения нет.

    Здесь переписывается ТОЛЬКО шаблон чужой причины — резкости, — а разрешение
    остаётся в допуске. Вердикт обязан остаться `degraded`: подстрочный вариант
    дал бы `reject`, то есть «переснимите, слишком мелко» на кадре, разрешение
    которого порога не нарушало.
    """
    monkeypatch.setattr(quality, "SHARPNESS_REASON",
                        "разрешение тут ни при чём: {value:.2e} < {threshold:.2e}")
    report = assess(img, 3.1, 4.6, 22.0,
                    thresholds=Thresholds(sharpness_min=1.0e9))

    assert any("разрешение" in r for r in report.reasons)
    assert report.gsd_mm_px_max < Thresholds().gsd_max_mm_px
    assert report.verdict == "degraded"


def test_reject_follows_the_resolution_threshold_on_both_sides():
    """Тот же признак — и в обратную сторону: у порога вердикт переключается.

    Дополняет проверку выше: та показывает, что чужое слово `reject` не вызывает,
    эта — что своё условие вызывает его всегда, в том числе когда причина не
    единственная.
    """
    t = Thresholds()
    frame = scene().render()
    just_below = assess(frame, 1.0, t.gsd_max_mm_px, 22.0)
    just_above = assess(frame, 1.0, t.gsd_max_mm_px * 1.001, 22.0)

    assert just_below.verdict == "ok"
    assert just_above.verdict == "reject"
    # И при второй, посторонней причине вердикт остаётся `reject`, а не «худшим
    # из двух по алфавиту»: недостаточное разрешение сильнее деградации.
    with_two = assess(frame, 1.0, t.gsd_max_mm_px * 1.001, 22.0, usable=0.1)
    assert with_two.verdict == "reject"
    assert len(with_two.reasons) == 2


def test_the_angle_threshold_has_exactly_one_declaration():
    """`THETA_MAX_DEG` объявлен один раз, и оба потребителя видят ТОТ ЖЕ объект.

    Прежде их было два: константа этого модуля, утверждавшая о себе «величина
    живёт здесь, а не в двух местах порознь», и собственный литерал `30.0` в
    умолчании `geometry.angles.usable_mask`. Совпадение значений проверкой не
    является — два литерала совпадают ровно до первой правки одного из них.
    Поэтому сверяется ТОЖДЕСТВО объектов, а не равенство чисел.
    """
    assert quality.THETA_MAX_DEG is angles.THETA_MAX_DEG


REFUSALS = {
    "gsd_min": lambda img: assess(img, float("nan"), 4.6, 22.0),
    "gsd_max": lambda img: assess(img, 3.1, float("nan"), 22.0),
    "theta_p95": lambda img: assess(img, 3.1, 4.6, float("nan")),
    "usable": lambda img: assess(img, 3.1, 4.6, 22.0, usable=float("nan")),
    "переставлены": lambda img: assess(img, 5.0, 4.6, 22.0),
    "изображение": lambda img: assess(np.zeros((0, 0), np.uint8), 3.1, 4.6, 22.0),
}


def test_refusal_messages_do_not_overlap(img):
    """Подстроки из match= взаимно исключительны: иначе тест зеленеет по чужому отказу.

    Проверяется, что каждая подстрока встречается ровно в одном сообщении из всех, и что
    ни одна не содержит вертикальной черты, которую match= прочитал бы как альтернативу.
    """
    messages = {}
    for key, call in REFUSALS.items():
        assert "|" not in key
        with pytest.raises(ValueError) as exc:
            call(img)
        messages[key] = str(exc.value)
        assert key in messages[key]

    for key in REFUSALS:
        hits = [k for k, message in messages.items() if key in message]
        assert hits == [key], f"подстрока {key!r} подходит и к чужим отказам: {hits}"


class _KindlessDtype:
    """Тип отсчётов, у которого вид пуст, а всё остальное — как у настоящего."""

    kind = ""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)


class FrameWithoutKind(np.ndarray):
    """Кадр, ведущий себя как настоящий везде, кроме вида отсчётов.

    Нужен, чтобы дойти ДО самой проверки вида. Список или массив объектов до неё не
    добирается: испорченная проверка пропускает их, и отказ приходит от последующего
    крушения (AttributeError на .ndim или TypeError в np.isfinite), то есть механизм,
    ради которого проверка написана, остаётся неизолированным.
    """

    @property
    def dtype(self):
        return _KindlessDtype(np.ndarray.dtype.__get__(self))


def test_frame_with_empty_dtype_kind_is_rejected_by_the_kind_check(img):
    """Проверка вида должна сработать сама, а не через крушение ниже по коду.

    У этого кадра вид отсчётов пуст, но cv2 и numpy работают с ним как с обычным
    uint8-кадром: если проверять принадлежность вида по строке "uif", пустая строка
    окажется её подстрокой, кадр пройдёт шлюз и получит вердикт «годен».
    """
    stub = img.view(FrameWithoutKind)
    assert stub.dtype.kind == ""
    assert sharpness(stub) == pytest.approx(sharpness(img), rel=1e-9)   # ведёт себя как кадр
    with pytest.raises(ValueError, match="изображение"):
        assess(stub, 3.1, 4.6, 22.0)
