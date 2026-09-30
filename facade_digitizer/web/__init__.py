"""Веб-интерфейс оператора: сервер на `localhost` и страница в браузере.

Логика экрана — в `desk` (без HTTP и без Qt), рабочая папка — в `workbench`, HTTP —
тонкий слой `app`. Геометрии здесь нет: всё считает ядро через `ui.session`.
Спецификация — `docs/superpowers/specs/2026-09-30-web-operator-interface-design.md`.
"""
