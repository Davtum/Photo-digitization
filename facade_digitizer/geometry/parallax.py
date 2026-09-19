"""Параллакс точек, не лежащих в плоскости фасада. Спецификация, раздел 5.

Точка на глубине d за плоскостью после ректификации оказывается смещённой:

    δ = (C_xy − P_xy) · d / (C_z + d)

Для заглублённых точек (d > 0) смещение направлено К опорной точке F,
для выступающих (d < 0) — ОТ неё.
"""
import math

from .camera import CameraOnPlane

# Опорные допущения п. 6.2, из которых выводится порог применимости по умолчанию.
# Они вынесены в именованные константы, а не вписаны числом в сигнатуру: порог
# обязан прослеживаться до ДОПУЩЕНИЯ, а не до цифры. Прежняя запись `15.0`
# не позволяла увидеть, при каком качестве кромки она верна, — и оказалась
# посчитанной в обход правила п. 6.1 (погрешность одной кромки вместо двух).
REFERENCE_SIGMA_PX = 1.0           # нижняя граница качества режима `assisted`, п. 6.1
REFERENCE_GSD_MM_PER_PX = 5.0      # локальный GSD опорного расчёта, п. 6.1
REFERENCE_DEPTH_MM = 150.0         # типовое заглубление опорной геометрии, п. 6.2
REFERENCE_TOLERANCE_MM = 20.0      # допуск 1σ на глубину заглубления, п. 2.2
REFERENCE_SIGMA_THETA_DEG = 0.344  # P95 направления, разреженная сцена, п. 6.2
REFERENCE_SIGMA_CZ_REL = 0.004     # rms дистанции до плоскости, 0.4 % от C_z, п. 6.2

# Границы и глубина поиска порога. Верхняя граница взята чуть ниже 90°: tg 90°
# не определён, а физического смысла ракурс вдоль плоскости не имеет. Нижняя
# отлична от нуля по той же причине: при θ_⊥ → 0 ширина откоса обращается в ноль.
_THETA_SEARCH_LO_DEG = 1e-9
_THETA_SEARCH_HI_DEG = 89.999
_SEARCH_ITERATIONS = 100


def apparent_position(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Где точка (x, y, −depth) окажется после ректификации."""
    denom = cam.cz + depth
    if denom <= 0:
        raise ValueError("точка оказалась за камерой: cz + depth <= 0")
    k = cam.cz / denom
    return (cam.cx + k * (x - cam.cx), cam.cy + k * (y - cam.cy))


def parallax_offset(
    cam: CameraOnPlane, x: float, y: float, depth: float
) -> tuple[float, float]:
    """Вектор смещения δ = P′ − P_xy."""
    ax, ay = apparent_position(cam, x, y, depth)
    return (ax - x, ay - y)


def correct_for_depth(
    cam: CameraOnPlane, xa: float, ya: float, depth: float
) -> tuple[float, float]:
    """Обратная операция: из наблюдаемого положения в истинное.

    Область определения совпадает с `apparent_position`: функции объявлены взаимно
    обратными, поэтому охрана здесь та же.
    """
    denom = cam.cz + depth
    if denom <= 0:
        raise ValueError("точка оказалась за камерой: cz + depth <= 0")
    k = denom / cam.cz
    return (cam.cx + k * (xa - cam.cx), cam.cy + k * (ya - cam.cy))


def reveal_depth(
    cam: CameraOnPlane,
    edge_x: float,
    edge_y: float,
    reveal_width_mm: float,
    edge_normal: tuple[float, float],
    theta_min_deg: float | None = None,
    sigma_px: float | None = None,
    gsd_mm_per_px: float | None = None,
) -> float:
    """Глубина заглубления по видимой ширине грани откоса.

    Ширина откоса, измеренная ПОПЕРЁК его ребра, равна не полному смещению, а его
    компоненте вдоль нормали к ребру. Замкнутое решение:

        w = d · |u| / (C_z + d)   ⟹   d = w · C_z / (|u| − w)

    где u — проекция вектора от опорной точки к кромке на edge_normal.
    Ориентация нормали безразлична: проекция берётся по модулю, так что
    `(1, 0)` и `(−1, 0)` дают одну и ту же глубину.

    Порог компоненты угла визирования θ_⊥ = arctg(|u| / (C_z + d)) обязателен:
    ниже него знаменатель |u| − w вырождается и глубина уходит в бесконечность —
    при |u| = 1000 и w = 999.9999 замкнутая форма даёт 1.0e11 мм без всякого
    признака неисправности.

    **Величина порога константой не задаётся.** Спецификация отозвала запись
    «θ_⊥ ≥ 15°»: порог есть функция точности локализации кромки и на заявленном
    диапазоне режимов п. 6.1 меняется вчетверо — 10.5° при σ_px = 0.5 и 47.3°
    при σ_px = 3. Прежние 15° получены с σ_w = σ_px·GSD, то есть как погрешность
    ОДНОЙ кромки, тогда как ширина откоса — разность двух кромок и по правилу
    п. 6.1 несёт √2·σ_px·GSD.

    Аргументы порога, в порядке убывания свободы вызывающего кода:

    * `theta_min_deg` — готовое значение. Задаётся, когда порог получен извне
      (например, из измеренной статистики конкретного набора). Вместе с
      `sigma_px` / `gsd_mm_per_px` не задаётся: иначе переданное качество кромки
      молча не использовалось бы.
    * `sigma_px`, `gsd_mm_per_px` — качество локализации кромки. Порог считается
      `reveal_depth_theta_min_deg` по фактическому C_z камеры.
    * ничего — берутся опорные допущения модуля: σ_px = 1 (нижняя граница
      качества режима `assisted`), GSD 5 мм/px, опорное заглубление 150 мм,
      допуск 20 мм, σ_θ = 0.344°, σ_Cz = 0.4 % дистанции. На C_z = 10 м это
      даёт 19.93°, а не прежние 15°.

    Почему опорными взяты именно эти значения. σ_px = 1 и GSD 5 мм/px — строка
    таблицы п. 6.2, названная спецификацией как значение по умолчанию: это
    нижняя граница качества режима `assisted`, то есть худший случай режима,
    в котором допуск ещё заявлен достижимым. Допуск 20 мм — требование п. 2.2
    к `recess_mm`. C_z берётся ФАКТИЧЕСКИЙ, а не опорный: спецификация прямо
    называет его аргументом порога, и он известен из позы до всякой оценки.
    Глубина, напротив, опорная (150 мм): она и есть искомая величина, и
    подставлять сюда уже вычисленную `depth` нельзя — на вырожденной геометрии
    она равна 1e11 мм, порог оказался бы недостижим, и отказ «глубина
    неустойчива» подменился бы отказом «допуск недостижим».

    Ужесточение порога с 15° до ≈19.93° намеренно: прежнее значение принимало
    грани, на которых σ_d достигает 26.8 мм при заявленном допуске 20 мм.
    Спецификация, пп. 5.4, 6.2.
    """
    if theta_min_deg is not None and (sigma_px is not None or gsd_mm_per_px is not None):
        raise ValueError(
            "theta_min_deg и качество кромки (sigma_px, gsd_mm_per_px) задаются не "
            "одновременно: при явном пороге переданное качество кромки осталось бы "
            "молча неиспользованным"
        )

    nx, ny = edge_normal
    norm = math.hypot(nx, ny)
    if norm == 0:
        raise ValueError("нормаль к ребру откоса не может быть нулевой")
    nx, ny = nx / norm, ny / norm

    u = abs((edge_x - cam.cx) * nx + (edge_y - cam.cy) * ny)
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть строго положительной")
    if u <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше расстояния до опорной точки; "
            "угол визирования слишком мал либо ширина измерена неверно"
        )

    depth = reveal_width_mm * cam.cz / (u - reveal_width_mm)

    if theta_min_deg is None:
        theta_min_deg = reveal_depth_theta_min_deg(
            sigma_px=REFERENCE_SIGMA_PX if sigma_px is None else sigma_px,
            gsd_mm_per_px=(
                REFERENCE_GSD_MM_PER_PX if gsd_mm_per_px is None else gsd_mm_per_px
            ),
            cz_mm=cam.cz,
            depth_mm=REFERENCE_DEPTH_MM,
            tolerance_mm=REFERENCE_TOLERANCE_MM,
            sigma_theta_deg=REFERENCE_SIGMA_THETA_DEG,
            sigma_cz_mm=REFERENCE_SIGMA_CZ_REL * cam.cz,
        )

    theta_deg = math.degrees(math.atan(u / (cam.cz + depth)))
    if theta_deg < theta_min_deg:
        raise ValueError(
            f"компонента угла визирования {theta_deg:.2f}° ниже порога "
            f"{theta_min_deg:.2f}°: оценка глубины неустойчива, "
            "грань откоса непригодна для измерения"
        )
    return depth


def reveal_depth_sigma(
    reveal_width_mm: float,
    u_mm: float,
    cz_mm: float,
    sigma_width_mm: float,
    sigma_u_mm: float,
    sigma_cz_mm: float = 0.0,
) -> float:
    """σ глубины: точные частные производные замкнутой формы, сложение по RSS.

    Замкнутая форма дифференцируется по своим ДЕЙСТВИТЕЛЬНО независимым переменным —
    измеренной ширине w, проекции |u| и высоте камеры C_z:

        d = w · C_z / (|u| − w)

        ∂d/∂w   =  C_z · |u| / (|u| − w)²
        ∂d/∂|u| = −w · C_z  / (|u| − w)²
        ∂d/∂C_z =  w        / (|u| − w)

        σ_d = √( (∂d/∂w · σ_w)² + (∂d/∂|u| · σ_u)² + (∂d/∂C_z · σ_Cz)² )

    Вклады складываются квадратично: источники погрешности ПРЕДПОЛАГАЮТСЯ
    независимыми, линейная сумма завышает σ (на типовых числах — до 35 %).
    Это допущение, и оно не консервативно: `|u|` и `C_z` происходят из одной оценки
    позы, а при положительной корреляции квадратичная сумма даёт НИЖНЮЮ оценку σ.
    Строгий бюджет потребует ковариационной матрицы позы.

    Прежняя малоугловая запись через угол оставлена как приближение:

        от ширины:  σ_d = σ_w / tg θ_⊥          занижает РОВНО в 1 + d / C_z
        от позы:    σ_d / d = 2 σ_θ / sin 2θ_⊥  занижает в cos²θ_⊥ / cos²φ,
                                                где φ = arctg(|u| / C_z)

    Обе занижают чувствительность, то есть ошибаются В ОПАСНУЮ СТОРОНУ: заявленная
    точность выходит выше фактической. Причина — θ_⊥ = arctg(|u| / (C_z + d)) сама
    зависит от искомой d и независимой переменной не является. На C_z = 10 м,
    d = 150 мм, θ_⊥ = 30°: ∂d/∂w = 1.75803 против 1.73205, вклад позы 2.32685
    против 2.30940.
    Спецификация, п. 6.2.
    """
    if reveal_width_mm <= 0:
        raise ValueError("ширина откоса должна быть строго положительной")
    if cz_mm <= 0:
        raise ValueError("высота камеры над плоскостью фасада должна быть положительной")
    if u_mm <= reveal_width_mm:
        raise ValueError(
            "вырожденная геометрия: ширина откоса не меньше проекции |u|, "
            "производные не определены"
        )
    if sigma_width_mm < 0 or sigma_u_mm < 0 or sigma_cz_mm < 0:
        raise ValueError("погрешности не могут быть отрицательными")

    denom = (u_mm - reveal_width_mm) ** 2
    dd_dwidth = cz_mm * u_mm / denom
    dd_du = reveal_width_mm * cz_mm / denom
    dd_dcz = reveal_width_mm / (u_mm - reveal_width_mm)

    return math.hypot(
        dd_dwidth * sigma_width_mm,
        dd_du * sigma_u_mm,
        dd_dcz * sigma_cz_mm,
    )


def reveal_depth_theta_min_deg(
    sigma_px: float,
    gsd_mm_per_px: float,
    cz_mm: float,
    depth_mm: float,
    tolerance_mm: float,
    sigma_theta_deg: float,
    sigma_cz_mm: float = 0.0,
) -> float:
    """Наименьший θ_⊥, при котором σ_d не превышает допуска. Спецификация, п. 6.2.

    Порог применимости отозван спецификацией как константа: он зависит от
    точности локализации кромки и на диапазоне режимов п. 6.1 меняется вчетверо.
    Здесь он ВЫЧИСЛЯЕТСЯ из величин, от которых зависит.

    Геометрия по θ_⊥ при заданных C_z и d (п. 6.2):

        w = d · tg θ_⊥        |u| = (C_z + d) · tg θ_⊥

    σ ширины берётся по правилу п. 6.1 для величин, равных разности ДВУХ кромок:

        σ_w = √2 · σ_px · GSD

    Множитель √2 здесь не украшение. Ширина откоса — разность наружной кромки
    проёма и внутренней кромки грани, то есть ровно двухкромочная величина.
    Его отсутствие и породило записанные ранее 15°: при σ_w = σ_px·GSD порог
    выходит 14.49°, по правилу п. 6.1 — 19.93°.

    Угловая погрешность позы переводится в миллиметры по п. 6.2:

        σ_u = C_z · sec²φ · σ_θ,   φ = arctg(|u| / C_z)

    Сама σ_d берётся у `reveal_depth_sigma`, а не пересчитывается здесь. Это
    существенно: порог и σ, выдаваемая потребителю, обязаны происходить из
    одной функции, иначе принятая грань могла бы нести σ выше объявленного
    допуска. На возвращённом пороге `reveal_depth_sigma` даёт ровно `tolerance_mm`.

    σ_d по θ_⊥ не монотонна: вклад кромки убывает как 1/tg θ_⊥, а вклад позы
    растёт как sec²φ, поэтому у σ_d есть минимум (на опорной геометрии — 3.93 мм
    при θ_⊥ ≈ 70°). Возвращается НИЖНИЙ корень, то есть наименьший пригодный
    угол. Если допуск ниже минимума, он недостижим ни при каком ракурсе — это
    отказ с названной причиной, а не None и не бесконечность: вызывающий код
    иначе принял бы недостижимое требование за выполнимое.

    Ноль возвращается только тогда, когда допуск соблюдён при любом угле, — то
    есть когда ограничения на ракурс действительно нет.
    """
    if gsd_mm_per_px <= 0:
        raise ValueError("GSD должен быть строго положительным")
    if cz_mm <= 0:
        raise ValueError("высота камеры над плоскостью фасада должна быть положительной")
    if depth_mm <= 0:
        raise ValueError("опорная глубина заглубления должна быть строго положительной")
    if tolerance_mm <= 0:
        raise ValueError("допуск на глубину должен быть строго положительным")
    if sigma_px < 0 or sigma_theta_deg < 0 or sigma_cz_mm < 0:
        raise ValueError("погрешности не могут быть отрицательными")

    sigma_width_mm = math.sqrt(2.0) * sigma_px * gsd_mm_per_px
    sigma_theta_rad = math.radians(sigma_theta_deg)

    def sigma_at(theta_deg: float) -> float:
        tan = math.tan(math.radians(theta_deg))
        u_mm = (cz_mm + depth_mm) * tan
        phi = math.atan(u_mm / cz_mm)
        sigma_u_mm = cz_mm / math.cos(phi) ** 2 * sigma_theta_rad
        return reveal_depth_sigma(
            depth_mm * tan, u_mm, cz_mm, sigma_width_mm, sigma_u_mm, sigma_cz_mm
        )

    if sigma_at(_THETA_SEARCH_LO_DEG) <= tolerance_mm:
        return 0.0

    # Минимум σ_d: троичный поиск по унимодальной функции. Нужен не сам по себе,
    # а как верхняя граница для корня — без него бисекция могла бы соскользнуть
    # на ВЕРХНЮЮ ветвь, где допуск нарушается снова, и вернуть завышенный порог.
    low, high = _THETA_SEARCH_LO_DEG, _THETA_SEARCH_HI_DEG
    for _ in range(_SEARCH_ITERATIONS):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if sigma_at(left) < sigma_at(right):
            high = right
        else:
            low = left
    theta_best = 0.5 * (low + high)
    sigma_best = sigma_at(theta_best)

    if sigma_best > tolerance_mm:
        raise ValueError(
            f"допуск {tolerance_mm:.2f} мм на глубину недостижим ни при каком угле: "
            f"минимум σ_d = {sigma_best:.2f} мм достигается при θ_⊥ = {theta_best:.2f}°. "
            "Требуется либо более точная локализация кромки, либо более мягкий допуск"
        )

    # Нижний корень: σ_d убывает от бесконечности при θ_⊥ → 0 до минимума.
    low, high = _THETA_SEARCH_LO_DEG, theta_best
    for _ in range(_SEARCH_ITERATIONS):
        middle = 0.5 * (low + high)
        if sigma_at(middle) > tolerance_mm:
            low = middle
        else:
            high = middle
    return high


def visible_reveal_side(
    cam: CameraOnPlane,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> tuple[str, str]:
    """Какие грани откоса видны: дальние от опорной точки.

    Возвращает (вертикальная, горизонтальная). Граница — СЕРЕДИНА проёма, а не его
    край. Если опорная точка попадает внутрь диапазона проёма, соответствующая грань
    всё равно называется по знаку, но её угол визирования мал; отсекает такие грани
    сама `reveal_depth` порогом θ_⊥ (вычисляемым, п. 6.2) — здесь пригодность
    не проверяется.
    Спецификация, п. 5.5.
    """
    vertical = "right" if cam.cx < (x_left + x_right) / 2 else "left"
    horizontal = "top" if cam.cy < (y_bottom + y_top) / 2 else "bottom"
    return (vertical, horizontal)
