"""Модель фасада → DXF. План 3, задача 18; спецификация, п. 10 и п. 13.

«DXF — чтобы результат открывался в CAD с первого дня». Заказчику результат нужен
для построения 3D-моделей, поэтому глубина откоса выносится в третью координату.

**Система координат** — фасада, как в `contour_mm`: миллиметры, X вправо, Y вверх
от начала `facade.origin`, Z — по нормали к стене НАРУЖУ. Внутренний контур проёма
(плоскость, до которой измерено заглубление, `recess.datum`) лежит на Z = −глубина.
Единицы чертежа объявлены миллиметрами (`$INSUNITS = 4`), чтобы CAD не масштабировал
вставку.

**Слои:**

* `FACADE_EXTENT` — охват снимка на фасаде (`facade.bounds_mm`): прямоугольник,
  а не контур здания; контур здания, если размечен, — на `FACADE_BOUNDARY`;
* `WINDOW`, `DOOR`, `FACADE_BOUNDARY` — контуры элементов по классам;
* `REVEAL` — внутренний контур проёма на глубине откоса.

**σ и происхождение — атрибутами (XDATA)** приложения `FACADE_DIGITIZER`, пары
«ключ (1000) — значение (1000 либо 1040)»: в CAD геометрия без погрешности читается
как точная, и именно это п. 2.2 запрещает. Отсутствующая величина (`None`) не
пишется вовсе, а не пишется нулём: нуль — правдоподобное измерение.
"""
from pathlib import Path

import ezdxf

from facade_digitizer.schema import Element, FacadeModel

APPID = "FACADE_DIGITIZER"
LAYERS = {
    "FACADE_EXTENT": 8,       # серый
    "FACADE_BOUNDARY": 7,
    "WINDOW": 5,              # синий
    "DOOR": 3,                # зелёный
    "REVEAL": 4,              # голубой
}
_CLASS_LAYER = {"window": "WINDOW", "door": "DOOR", "facade_boundary": "FACADE_BOUNDARY",
                "reveal": "REVEAL"}


def _pairs(values: dict) -> list[tuple[int, object]]:
    out = []
    for key, value in values.items():
        if value is None:
            continue
        out.append((1000, key))
        if isinstance(value, bool):
            out.append((1000, "true" if value else "false"))
        elif isinstance(value, int | float):
            out.append((1040, float(value)))
        else:
            out.append((1000, str(value)))
    return out


def element_xdata(element: Element, model: FacadeModel) -> dict:
    """Атрибуты элемента, которые обязаны пережить перенос в CAD."""
    size, recess = element.size_mm, element.recess
    return {
        "id": element.id,
        "class": element.class_name,
        "origin": element.origin,
        "mounting": element.mounting,
        "edge_type": element.edge_type,
        "edge_reference": element.edge_reference,
        "width_mm": size.width,
        "height_mm": size.height,
        "sigma_width_mm": size.sigma_width,
        "sigma_height_mm": size.sigma_height,
        "position_sigma_mm": element.position_sigma_mm,
        "relative_position_sigma_mm": element.relative_position_sigma_mm,
        "recess_mm": recess.value_mm if recess else None,
        "sigma_recess_mm": recess.sigma_mm if recess else None,
        "recess_origin": recess.origin if recess else None,
        "meets_tolerance": element.meets_tolerance,
        "scale_sigma_rel": model.facade.scale.sigma_rel,
        "facade_origin": model.facade.origin,
        "schema_version": model.schema_version,
    }


def to_dxf(model: FacadeModel, path) -> Path:
    """Записать модель в DXF (R2013). Возвращает путь."""
    doc = ezdxf.new("R2013", setup=False)
    doc.header["$INSUNITS"] = 4                  # миллиметры
    doc.appids.add(APPID)
    for name, color in LAYERS.items():
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    bounds = model.facade.bounds_mm
    if len(bounds) == 4:
        x0, y0, x1, y1 = bounds
        extent = msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                    close=True, dxfattribs={"layer": "FACADE_EXTENT"})
        extent.set_xdata(APPID, _pairs({"what": "image coverage on facade, not building "
                                                "outline", "coverage": model.coverage}))

    for element in model.elements:
        layer = _CLASS_LAYER[element.class_name]
        points = [(float(x), float(y)) for x, y in element.contour_mm]
        xdata = _pairs(element_xdata(element, model))
        outline = msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})
        outline.set_xdata(APPID, xdata)
        recess = element.recess
        if recess is not None and recess.value_mm is not None:
            inner = msp.add_lwpolyline(points, close=True,
                                       dxfattribs={"layer": "REVEAL",
                                                   "elevation": -float(recess.value_mm)})
            inner.set_xdata(APPID, xdata)
    path = Path(path)
    doc.saveas(path)
    return path


def read_xdata(entity) -> dict:
    """XDATA сущности обратно в словарь — для проверок и для потребителей DXF."""
    tags = list(entity.get_xdata(APPID))
    out = {}
    for (_, key), (code, value) in zip(tags[::2], tags[1::2]):
        out[key] = ({"true": True, "false": False}.get(value, value)
                    if code == 1000 else float(value))
    return out
