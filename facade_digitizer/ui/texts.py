"""Тексты интерфейса оператора. Веб-интерфейс, задача 1.

Строки перенесены из окна Qt (`ui/panels.py`, `ui/window.py` плана 3) дословно:
оператор, обученный на окне, и выходной файл говорят одними словами, а причины
качества — ТЕ ЖЕ строки, что `quality.reasons` выходного файла (вторая формулировка
одной причины разошлась бы с первой). Qt здесь не импортируется.
"""
from facade_digitizer.pipeline import run
from facade_digitizer.ui.zoom import (
    RECOMMENDED_MIN_SCALE,
    click_sigma,
    localisation_cost_mm,
    sigma_image_px,
)

#: Почему у элемента нет σ положения: она сводится с общей ошибкой масштаба в одном
#: месте (`pipeline.assemble`, п. 4.4), и у модели, собранной в обход него, её нет.
POSITION_SIGMA_PENDING = "σ не сведена (pipeline.assemble)"
#: Одинокий элемент: взаимного положения без соседа нет.
NO_NEIGHBOUR_TEXT = "соседа нет"

#: Происхождение глубины по схеме → слова для оператора.
RECESS_ORIGIN_TEXT = {
    "unavailable": ("не измерена: грань откоса не видна камере, размечена не та сторона, "
                    "угол визирования на грань ниже порога либо кромка вне плоскости стены"),
    "assumed_class_default": "принята типовой для класса — не измерение",
    "operator": "введена оператором — не измерение",
}

CALIBRATION_TEXT = {
    "target": "калибровка по мишени — условие п. 2.2 выполнено",
    "exif": "K из EXIF — условие п. 2.2 «камера откалибрована» не выполнено",
    "database": "K из таблицы моделей — условие п. 2.2 «камера откалибрована» не выполнено",
}

#: Вердикты шлюза качества — словами ядра (`pipeline.quality`); пояснение — рядом.
VERDICT_TEXT = {
    "ok": "снимок пригоден",
    "degraded": "снимок пригоден с оговорками",
    "reject": "снимок отбракован",
}

#: Значения схемы для разметки проёма — ключ схемы и слова для оператора.
CLASSES = (("window", "окно"), ("door", "дверь"))
MOUNTINGS = (("embedded", "заглублён в проём"), ("flush", "заподлицо"),
             ("protruding", "выступает"))
EDGES = (("sharp_wall_edge", "острая кромка стены"), ("surround", "обрамление"),
         ("cladding_edge", "кромка облицовки"), ("unknown", "не известно"))
SIDES = (("left", "левая"), ("right", "правая"), ("top", "верхняя"),
         ("bottom", "нижняя"))

#: Происхождение длины опорной базы.
SCALE_SOURCES = (("operator_reference", "Длина измерена (рулетка, дальномер)"),
                 ("assumed_floor_height", "Типовая высота этажа (допуск не достигается)"))

RESULT_COLUMNS = ("id", "ширина, мм", "высота, мм", "положение X / Y, мм",
                  "заглубление, мм", "θ", "допуск")


def scale_name(scale: float) -> str:
    """Масштаб просмотра словами окна: «2:1», «1:2»."""
    return f"{scale:g}:1" if scale >= 1 else f"1:{1 / scale:.3g}"


def tolerance_text(element, verdict: str) -> str:
    """Признак соответствия допуску — словами, и при `null` — С ПРИЧИНОЙ.

    `null` значит «утверждать нечем», а пустая ячейка выдала бы отсутствие
    утверждения за его отсутствие в интерфейсе, а не в измерении.
    """
    if element.meets_tolerance is True:
        return "соответствует допуску п. 2.2"
    if element.meets_tolerance is False:
        return "не соответствует: σ габарита больше допуска 10 мм"
    if element.edge_reference != "wall_plane":
        return ("не выпущено: кромка вне плоскости стены или не известна "
                "(п. 5.3, 5.6), вынос не измерен")
    if verdict == "reject":
        return "не выпущено: кадр отбракован по разрешению (п. 2.2)"
    return "не выпущено"


def quality_note(stage: str, verdict: str, *, confidence: str = "") -> str:
    """Пояснение под вердиктом: предварительный (до базы) или окончательный."""
    if stage == "preliminary":
        return (f"{confidence}. Вердикт по разрешению будет после ввода опорной базы: "
                "разрешение считается от её длины.")
    if verdict == "reject":
        return ("Кадр отбракован по разрешению. Разметка разрешена, но признак "
                "соответствия допуску у элементов выпущен не будет. Проверьте длину "
                "базы: опечатка в ней меняет вердикт.")
    return ("Вердикт зависит от длины опорной базы: опечатка в ней меняет его и "
            "признак допуска у всех элементов.")


QUALITY_STAGE_TEXT = {
    "preliminary": "Предварительный: до опорной базы",
    "final": "Окончательный: зависит от введённой длины опорной базы",
}


def end_sigma_lines(ends) -> list[str]:
    """По каждому концу базы — σ клика и цена масштаба грубее 1:1."""
    lines = []
    for i, p in enumerate(ends, 1):
        note = (f" — масштаб {scale_name(p.view_scale)} грубее 1:1; при "
                f"{scale_name(RECOMMENDED_MIN_SCALE)} было бы "
                f"{sigma_image_px(RECOMMENDED_MIN_SCALE):.2f} px"
                if p.view_scale < 1.0 else "")
        lines.append(f"конец {i}: σ клика {click_sigma(p):.2f} px{note}")
    return lines


def base_result_text(ss, reference) -> str:
    """σ масштаба, допуск п. 6.3 и начало отсчёта — после фазы масштаба."""
    origin = ("левый нижний угол фасада (охват полный)"
              if reference.origin_is_facade_corner else "точка оператора (охват частичный)")
    tol = "достигает" if ss.sigma_rel <= run.SIGMA_REL_TOLERANCE else "НЕ достигает"
    return (f"σ масштаба {ss.sigma_rel:.3%} — {tol} допуска п. 6.3; на базе "
            f"{reference.span_mm:.0f} мм это {ss.sigma_rel * reference.span_mm:.1f} мм. "
            f"Начало отсчёта — {origin}. Более длинная база даёт меньшую σ.")


def scale_warning(point, gsd: float | None) -> str:
    """Цена масштаба клика — числом (σ в px и слагаемое бюджета в мм)."""
    sigma = click_sigma(point)
    cost = (f", слагаемое бюджета {localisation_cost_mm(point.view_scale, gsd):.1f} мм "
            f"при разрешении {gsd:.1f} мм/px" if gsd is not None else "")
    if point.view_scale < RECOMMENDED_MIN_SCALE:
        best = sigma_image_px(RECOMMENDED_MIN_SCALE)
        return (f"Масштаб {scale_name(point.view_scale)} грубее рекомендованного "
                f"{RECOMMENDED_MIN_SCALE:g}:1: σ клика {sigma:.2f} px вместо {best:.2f}{cost}.")
    return f"σ клика {sigma:.2f} px{cost}."


def reveal_suggestion(sides) -> str:
    """Какие грани откоса видны камере (п. 5.5); до масштаба — как их узнать."""
    if sides is None:
        return "Видимую грань откоса подскажет поза камеры — задайте опорную базу."
    names = dict(SIDES)
    vertical, horizontal = sides
    return (f"Камере видны грани: {names[vertical]} и {names[horizontal]}. "
            "Ближние закрыты собственной стеной — их не размечают (п. 5.5).")


def result_header(model) -> str:
    image = model.images[0]
    origin = ("от левого нижнего угла фасада" if model.facade.origin == "bottom_left"
              else "от точки оператора (первый конец базы), не от угла здания")
    return (f"Координаты — {origin}. Камера: "
            f"{CALIBRATION_TEXT.get(image.camera.calibration, image.camera.calibration)}. "
            f"Вердикт качества «{image.quality.verdict}» зависит от введённой длины "
            "опорной базы.")


def result_rows(model) -> list[dict]:
    """Строки таблицы результата. Правило: величины без σ не бывает — где σ ещё не
    считается, ячейка называет причину, а не пустует."""
    verdict = model.images[0].quality.verdict
    rows = []
    for e in model.elements:
        s = e.size_mm
        # Положение — нижний левый угол контура: именно для него `assemble` сводит σ
        # абсолютного и взаимного положения.
        x, y = e.contour_mm[0]
        if e.position_sigma_mm is not None:
            neighbour = (f"соседи ± {e.relative_position_sigma_mm:.1f}"
                         if e.relative_position_sigma_mm is not None else NO_NEIGHBOUR_TEXT)
            position = f"{x:.0f} / {y:.0f} ± {e.position_sigma_mm:.1f}; {neighbour}"
        else:
            position = f"{x:.0f} / {y:.0f}; {POSITION_SIGMA_PENDING}"
        if e.recess is None:
            recess = "грань откоса не размечена"
        elif e.recess.value_mm is not None and e.recess.sigma_mm is not None:
            recess = f"{e.recess.value_mm:.1f} ± {e.recess.sigma_mm:.1f}"
        else:
            recess = RECESS_ORIGIN_TEXT.get(e.recess.origin, e.recess.origin)
        rows.append({
            "id": e.id,
            "width": f"{s.width:.1f} ± {s.sigma_width:.1f}",
            "height": f"{s.height:.1f} ± {s.sigma_height:.1f}",
            "position": position,
            "recess": recess,
            "theta": f"{e.theta.full_deg:.1f}°" if e.theta is not None else "—",
            "tolerance": tolerance_text(e, verdict),
            "meets": e.meets_tolerance,
        })
    return rows
