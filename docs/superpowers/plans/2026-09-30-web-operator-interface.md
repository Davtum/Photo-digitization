# Веб-интерфейс оператора — план реализации

> **Для агентов-исполнителей:** ОБЯЗАТЕЛЬНЫЙ ПОДНАВЫК — superpowers:executing-plans (выбран
> исполнителем: заказчик поручил работу целиком, «работай автономно, принимай решения сам»).
> Шаги размечены чекбоксами (`- [ ]`).

**Цель:** заменить окно PySide6 (показываемое через noVNC) веб-интерфейсом на `localhost`:
пошаговый путь «Снимок → Масштаб → Проёмы → Результат», клик по кадру определяется шагом,
автосохранение, выгрузка в `экспорт/`, — сохранив метрологию плана 3.

**Архитектура:** вся логика экрана — в Qt-free и HTTP-free `web/desk.py` (`Desk`: сессия +
состояние экрана → `state()`), рабочая папка — `web/workbench.py`, HTTP — тонкий `web/app.py`
(FastAPI), страница — статические ES-модули без сборки. Геометрия остаётся в ядре; страница
только переводит экран ↔ кадр и рисует.

**Стек:** Python 3.11+, FastAPI, uvicorn, python-multipart, httpx (тесты), OpenCV (PNG/JPEG),
HTML/CSS/JS (ES2020, без фреймворков), Node.js — только для теста `geom.js` (пропуск без node,
кроме CI).

**Спецификация:** `docs/superpowers/specs/2026-09-30-web-operator-interface-design.md`
(редакция 2). Исполнитель читает её целиком: план ссылается на её пункты (§).

## Глобальные ограничения

- Python ≥ 3.11; `.venv` проекта; `ruff check facade_digitizer tests scripts` чистый, длина строки 100.
- `web/*.py` и `ui/*.py` не импортируют PySide6 (тест).
- Формат файла сессии: `SESSION_FORMAT`/`SESSION_VERSION = 1` — без изменений.
- `view_scale` = физических пикселей экрана на пиксель кадра (CSS-масштаб × `devicePixelRatio`).
- Кадр — PNG без потерь (`cv2.IMWRITE_PNG_COMPRESSION = 1`).
- Сервер слушает только loopback; `Host` ∈ {localhost, 127.0.0.1, [::1]}:порт, иначе 421;
  изменения — только с `X-Facade-Token`, иначе 403.
- Вердикты — словами ядра `ok` / `degraded` / `reject`; тексты причин — строки ядра.
- Экспорт — `<папка снимка>/экспорт/`; сессия — `<папка снимка>/<основа>.session.json`.
- Цвета разметки: плоскость `#fdd835`, область `#1e88e5`, база `#43a047`, проём `#8e24aa`,
  выбранный `#ff7043`, откос `#00acc1`.
- Весь текст интерфейса — по-русски; страница работает без интернета.
- Коммит после каждой задачи; сообщения по-русски, в конце
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Фокус проверки

1. **Зум браузера / масштаб Windows ≠ 100 %** — «1:1» по-прежнему пиксель в пиксель монитора, клик
   записывается с физическим масштабом (задача 8, тест `geom.js`; ручная проверка задачи 11 при
   Ctrl+плюс).
2. **Двойной клик / быстрые клики во время пересчёта** — ни одна точка не теряется и не ложится
   на устаревший кадр: очередь запросов + `409` (задачи 3, 7; тест `test_stale_revision_is_409`).
3. **Снимок в папке только для чтения / OneDrive блокирует файл** — работа продолжается, красная
   плашка и скачивание сессии (задача 6, тест `test_autosave_failure_is_visible`).
4. **Повторное открытие снимка другим запуском сервера** — работа продолжается с того же места с
   той же меткой (задача 6, `test_restore_keeps_file_label_when_server_label_is_random`).
5. **Ввод длины базы с опечаткой и исправлением** — вердикт/таблица не «мигают» на промежуточных
   значениях (задача 4: фиксация на клиенте; тест `test_invalid_span_is_not_a_reject`).

---

## Карта файлов

| Файл | Ответственность |
|---|---|
| `facade_digitizer/ui/texts.py` (новый) | тексты панелей результата/качества/базы (перенос из `panels.py`, `window.py`) |
| `facade_digitizer/ui/session.py` | + `move_reference_end` |
| `facade_digitizer/ui/session_file.py` | атомарная запись, поиск снимка/профиля (`search_dirs`) |
| `facade_digitizer/web/__init__.py` | докстринг пакета |
| `facade_digitizer/web/jobs.py` | фоновые задачи с поколениями; `InlineRunner` для тестов |
| `facade_digitizer/web/images.py` | PNG/JPEG кодирование кадра, маски, растра, миниатюр |
| `facade_digitizer/web/desk.py` | `Desk`: постановки, шаги, операции, пересчёты, `state()` |
| `facade_digitizer/web/workbench.py` | рабочая папка: библиотека, загрузки, профили, открытие/восстановление, автосохранение, метка |
| `facade_digitizer/web/app.py` | FastAPI: маршруты, Host/токен, 409, отдача файлов |
| `facade_digitizer/web/__main__.py` | CLI `facade-digitize-ui`, lock-файл, браузер |
| `facade_digitizer/web/static/*` | `index.html`, `styles.css`, `api.js`, `geom.js`, `viewer.js`, `app.js` |
| удаляются | `ui/window.py`, `ui/panels.py`, `ui/canvas.py`, `ui/rectified.py`, `ui/worker.py`, `ui/__main__.py`, Qt-тесты `test_ui_*` |
| `tests/test_web_*.py` | тесты `Desk`, `Workbench`, API, `geom.js` |
| `scripts/demo_ui.py` | демо через `Desk`; `--show` — сервер + браузер |
| `pyproject.toml`, `Dockerfile`, `docker/start.sh`, `docker-compose.yml`, `.github/workflows/tests.yml` | зависимости, образ без X, CI |
| документы | §5.3 спецификации |

---

### Задача 1: тексты без Qt и дополнения ядра интерфейса

**Файлы:** создать `facade_digitizer/ui/texts.py`; изменить `ui/session.py`, `ui/session_file.py`;
тесты `tests/test_ui_texts.py`, дополнить `tests/test_session_file.py`, `tests/test_ui_session.py`.

**Производит:**
- `ui/texts.py`: `POSITION_SIGMA_PENDING`, `NO_NEIGHBOUR_TEXT`, `RECESS_ORIGIN_TEXT`,
  `CALIBRATION_TEXT`, `tolerance_text(element, verdict) -> str`,
  `result_header(model) -> str`, `result_rows(model) -> list[dict]` (ключи `id, width, height,
  position, recess, theta, tolerance` — строки ячеек `ResultPanel.show_model` без изменений),
  `RESULT_COLUMNS` (заголовки), `scale_name(scale) -> str` («2:1», «1:2»),
  `quality_note(stage, verdict) -> str` (тексты `QualityPanel`), `base_result_text(ss, reference)
  -> str` (текст `_compute_base`), `end_sigma_lines(ends) -> list[str]` (текст `_draw_base`),
  `scale_warning(point, gsd) -> str` (текст `_warn_scale`), `reveal_suggestion(sides) -> str`,
  `MOUNTINGS`, `EDGES`, `SIDES`, `CLASSES` (пары ключ–текст из `MarksPanel`).
- `OperatorSession.move_reference_end(index: int, point: ClickedPoint) -> None` — заменяет конец,
  сбрасывает `scale`/`model`.
- `session_file.save_session(session, path)` — временный файл в той же папке + `os.replace`,
  до 5 попыток при `PermissionError` с паузой 0.1 с; при неудаче исключение выходит наружу.
- `session_file.load_session(path, *, search_dirs=())` — снимок: сохранённый путь, затем
  `Path(path).parent / имя`; профиль: сохранённый путь, затем папка сессии, затем каждая из
  `search_dirs`. Хэш сверяется у найденного файла.

- [ ] Тесты: `test_texts_module_does_not_import_qt`; `test_tolerance_text_names_each_reason`
  (элементы-заглушки `SimpleNamespace` на три ветви + `reject`); `test_scale_name`;
  `test_move_reference_end_replaces_and_invalidates`;
  `test_save_session_is_atomic_and_leaves_no_temp` (после записи в папке ровно один файл);
  `test_load_session_finds_image_next_to_the_session` (сессия с путём несуществующей папки,
  снимок рядом → открывается; другой снимок рядом с тем же именем → отказ по хэшу);
  `test_load_session_finds_profile_in_search_dirs`.
- [ ] Запустить — падают. Реализовать, перенеся строки из `panels.py`/`window.py` дословно.
- [ ] `pytest tests/test_ui_texts.py tests/test_session_file.py tests/test_ui_session.py -q` — зелёные.
- [ ] Коммит «Тексты интерфейса без Qt, перенос конца базы, атомарная сессия».

### Задача 2: фоновые задачи и кодирование изображений

**Файлы:** `web/__init__.py`, `web/jobs.py`, `web/images.py`; тесты `tests/test_web_jobs.py`.

**Производит:**
- `class JobRunner`: `submit(kind: str, fn, on_done, on_error) -> int` — поколение по `kind`
  растёт с каждым `submit`; результат доставляется, только если поколение ещё последнее;
  `cancel(kind)` — поднять поколение; `pending(kind) -> bool`; `wait_idle(timeout)` (для тестов).
  Пул `ThreadPoolExecutor(max_workers=2)`. `on_done`/`on_error` вызываются в потоке пула —
  вызывающий берёт свою блокировку сам.
- `class InlineRunner(JobRunner)`: выполняет `fn` сразу в вызывающем потоке (тесты `Desk`).
- `images.png(array, compression=1) -> bytes`, `images.jpeg(array, quality=90) -> bytes`,
  `images.usable_png(mask) -> bytes` (RGBA: непригодное `(0,0,0,110)`),
  `images.invalid_png(valid_mask) -> bytes` (штриховка окна Qt), `images.thumbnail(path,
  width=360) -> bytes` (`cv2.imread` c `IMREAD_REDUCED_COLOR_4`, при неудаче — полный).

- [ ] Тесты: `test_stale_generation_is_not_delivered` (две задачи одного вида, первая медленнее —
  доставлена только вторая); `test_errors_are_delivered_as_text`; `test_inline_runner_runs_now`;
  `test_png_roundtrip_is_lossless` (случайный BGR 64×48 → `cv2.imdecode` равен побайтно);
  `test_usable_png_marks_unusable`.
- [ ] Реализовать, прогнать, коммит «Фоновые задачи с поколениями и кодирование изображений».

### Задача 3: `Desk` — открытие, кадр, шаг «Снимок»

**Файлы:** `web/desk.py`; тесты `tests/test_web_desk_frame.py`; фикстуры сцен —
`tests/web_scenes.py` (синтетика как в `test_ui_marks`/`test_ui_result`: `near`, `far`,
`no_lines` из `make_acceptance_inputs`).

**Производит (`Desk`):**
```python
Desk(session: OperatorSession, *, runner: JobRunner, save: Callable[[OperatorSession], None],
     lock: threading.RLock | None = None, desk_id: str)
desk.start()                        # фаза кадра в фоне (поколение "frame")
desk.rev: int; desk.busy: str | None; desk.frame_version: int
desk.frame_png: bytes | None; desk.usable_png: bytes | None
desk.state() -> dict                # кэш по rev
desk.act(name: str, **args) -> None # диспетчер по белому списку ACTIONS; ValueError → notice
```
Действия шага ①: `set_step(step)`, `start_manual_plane()`, `set_plane_constraint(kind, aspect,
width_mm, height_mm)`, `apply_manual_plane()`, `start_roi()`, `apply_roi()`, `reset_plane()`,
`stop_placing()`, `set_profile(path|None, discard: bool)`, `click(x, y, view_scale)`.
Постановки: `placing ∈ {None, "plane", "roi", "base", "corners", "reveal"}`; `plane`/`roi`
взаимоисключающие; при `needs_operator` после кадра — `step="frame"`, `placing="plane"`.
После нового кадра (плоскость, область, возврат) — если база готова: масштаб, растр, модель.
`state()["frame"]`: `{version, width, height, info: [строки], camera, plane: {...},
quality: {stage, verdict, reasons, note, usable_fraction}, profile: {name, path}}`.
`state()["hint"]`: `{text, count, total, can_finish}` либо текст навигации.
`state()["overlays"]`: `[{key, points, closed, color, width}]`; `state()["draggable"]`:
`[{key, x, y}]` — пусто при постановке.

- [ ] Тесты: `test_open_runs_frame_phase_and_encodes_lossless_png` (размер PNG = кадр, пиксели =
  `frame.color`); `test_needs_operator_opens_manual_plane_placing`;
  `test_manual_plane_each_constraint_reaches_the_frame`; `test_three_corners_refused_by_name`
  (notice содержит «углов 3 из 4»); `test_plane_and_roi_placings_are_exclusive`;
  `test_roi_is_passed_to_the_frame_stage`; `test_reset_returns_to_the_automatic_plane`;
  `test_new_plane_recomputes_scale_and_model_when_ready`;
  `test_preliminary_verdict_uses_core_words`; `test_profile_change_with_clicks_needs_discard`
  (без `discard` — notice, точки на месте; с `discard` — `frame_version` вырос);
  `test_foreign_profile_is_refused_by_camera_name`; `test_actions_while_busy_raise_busy`
  (`DeskBusy`); `test_state_is_json_serialisable`; `test_web_modules_do_not_import_qt`.
- [ ] Реализовать, зелёные, коммит «Desk: открытие снимка, кадр и плоскость».

### Задача 4: `Desk` — шаг «Масштаб» и выровненный вид

**Файлы:** `web/desk.py`; тесты `tests/test_web_desk_scale.py`.

**Производит:** `restart_base()`, `set_base(span_mm, span_sigma_mm, scale_source,
origin_is_facade_corner)`, `drop(key="base:i", x, y, view_scale, moved)`; постановка `base`
включается при входе на шаг, если концов < 2; растр — фоновая задача вида `"raster"`
(`raster_stage(..., color=True)`), `desk.rect_jpg`, `desk.rect_mask_png`,
`state()["rectified"]`: `{status: "none"|"pending"|"ready"|"error", version, width, height,
H, H_inv, mm_per_px, coverage, text}`. `state()["scale"]`: `{ends: [строки σ], values,
result, tolerance_ok, quality: {...}}`.

- [ ] Тесты (перенос `test_ui_base`, `test_ui_quality`, `test_ui_rectified`):
  `test_base_clicks_keep_view_scale_and_per_end_sigma`; `test_third_click_does_nothing`;
  `test_set_base_computes_scale_and_final_verdict`; `test_invalid_span_is_not_a_reject`
  (0, отрицательное, `None` → «длина не введена», `scale is None`);
  `test_final_verdict_flips_with_the_base_length`; `test_span_uncertainty_enters_sigma_rel`;
  `test_dragging_a_base_end_recomputes_scale`; `test_rectified_H_maps_reference_values`
  (точки кадра через `state H` = `apply_homography(rectified.H)`);
  `test_stale_raster_is_discarded` (два масштаба подряд — в `state` растр второго);
  `test_reject_does_not_block_marking`.
- [ ] Реализовать, зелёные, коммит «Desk: опорная база, масштаб и выровненный вид».

### Задача 5: `Desk` — проёмы, правка, результат, экспорт

**Файлы:** `web/desk.py`; тесты `tests/test_web_desk_marks.py`.

**Производит:** `new_mark(class_)`, `select_mark(id)`, `set_mark_attrs(id, mounting,
edge_type)`, `start_reveal()`, `set_reveal_side(side)`, `undo_point()`, `delete_mark(id)`,
`drop(key="w_000:corner:1", x, y, view_scale, moved)`, `export() -> Exported` (в
`<папка снимка>/экспорт/`), клик вне постановки на шаге ③ — выбор проёма по контуру
(`cv2.pointPolygonTest`). Автопересчёт модели после каждого изменения при `ready_to_measure()`.
`state()["marks"]`: `{items: [{id, class, class_text, mounting, edge_type, corners, reveal_side,
reveal_points, size}], selected, suggestion, warning, reasons}`; `state()["result"]`:
`{header, columns, rows, export: {files: [{name, url}], cli} | None}`;
`state()["sigma"]`: `{screen_px, edge_px, recommended_min_scale}`.

- [ ] Тесты (перенос `test_ui_marks`, `test_ui_edit`, `test_ui_result`, `test_ui_session_file`):
  `test_four_clicks_any_order_make_an_element_with_own_sigma`;
  `test_new_mark_inherits_attributes_of_previous`; `test_reveal_side_is_suggested`;
  `test_changing_reveal_side_clears_points_at_once`; `test_coarse_click_warns_with_mm_cost`
  (и для точки откоса); `test_undo_removes_the_last_point`;
  `test_esc_on_empty_mark_deletes_it`; `test_model_recomputes_itself_when_ready`;
  `test_drop_recomputes_fully_and_logs_one_edit`; `test_drop_without_move_is_not_an_edit`;
  `test_points_are_draggable_only_outside_placing`; `test_click_inside_contour_selects_mark`;
  `test_ids_are_stable_after_delete`; `test_result_rows_have_sigma_or_reason`;
  `test_withheld_tolerance_shows_its_reason` (обрамление; reject);
  `test_export_writes_json_dxf_marks_command_in_export_dir`;
  `test_export_is_reproduced_by_cli_bytewise` (запуск `run.main(args)` в другой каталог).
- [ ] Реализовать, зелёные, коммит «Desk: проёмы, правка, результат и экспорт».

### Задача 6: `Workbench` — рабочая папка, сессии, метка оператора

**Файлы:** `web/workbench.py`; тесты `tests/test_web_workbench.py`.

**Производит:**
```python
Workbench(data_dir: Path, *, operator: str | None, runner: JobRunner | None = None,
          clock=time.monotonic)
wb.library() -> dict       # {folders: [{path, images: [{path, name, status, collision}]}], profiles}
wb.upload_image(name: str, data: bytes) -> str     # относительный путь; sha256-дубликат → существующий
wb.upload_profile(name: str, data: bytes) -> str   # в профили/, проверка calib.load_profile
wb.profiles() -> list[dict]                        # [{name, path}]
wb.thumbnail(rel: str) -> bytes                    # кэш .thumbs/
wb.resolve(rel: str) -> Path                       # внутри data_dir, иначе ValueError
wb.open(rel: str) -> dict                          # state; восстанавливает сессию
wb.answer_operator(choice: "continue"|"restart")
wb.restart() ; wb.save_copy(folder_rel: str) -> str ; wb.close()
wb.desk: Desk | None ; wb.lock: threading.RLock
```
Автосохранение — функция `save`, переданная `Desk`: `session_file.save_session`, статус в
`state()["save"] = {ok, error, path}`. Статусы библиотеки: `new`, `in_progress`, `exported`,
`changed_after_export`; `collision=True` — одинаковая основа в папке, `open` отказывает.

- [ ] Тесты: `test_library_lists_images_with_statuses_and_skips_service_dirs`;
  `test_stem_collision_is_flagged_and_refused`; `test_upload_dedupes_by_sha256`;
  `test_upload_renames_on_name_clash`; `test_resolve_refuses_paths_outside`;
  `test_open_restores_session_next_to_image`; `test_autosave_after_each_change`;
  `test_autosave_failure_is_visible` (путь сессии — каталог → `save.ok is False`, работа идёт);
  `test_restore_keeps_file_label_when_server_label_is_random`;
  `test_explicit_other_label_asks_before_opening`; `test_restart_archives_to_bak`;
  `test_save_copy_names_with_timestamp`; `test_thumbnail_is_cached`;
  `test_upload_profile_checks_format`.
- [ ] Реализовать, зелёные, коммит «Рабочая папка: библиотека, загрузки, автосохранение».

### Задача 7: HTTP-слой

**Файлы:** `web/app.py`; тесты `tests/test_web_api.py`.

**Производит:** `create_app(workbench: Workbench, *, token: str, port: int) -> FastAPI`.
Маршруты: `GET /` (index с токеном), `/static/*`, `GET /api/ping`, `GET /api/library`,
`POST /api/upload` (multipart `file`), `POST /api/profile/upload`, `POST /api/open {path}`,
`POST /api/close`, `GET /api/state`, `POST /api/action {desk, rev, name, args}`,
`GET /api/frame.png`, `/api/usable.png`, `/api/rectified.jpg`, `/api/rectified_mask.png`,
`/api/thumb?path=`, `/api/export/{name}`, `/api/session/download`. Ответы `409` (ревизия, снимок,
занятость) несут `state`. Изменяющие — `POST` с токеном. Кэш: изображения с `?v=` —
`Cache-Control: max-age=31536000, immutable`, прочее — `no-store`.

- [ ] Тесты: `test_foreign_host_is_421`; `test_post_without_token_is_403`;
  `test_upload_open_click_export_download_roundtrip`; `test_frame_png_matches_frame`;
  `test_stale_revision_is_409_with_state`; `test_unknown_action_is_400`;
  `test_path_traversal_is_refused`; `test_index_carries_the_token`.
- [ ] Реализовать, зелёные, коммит «HTTP-слой веб-интерфейса».

### Задача 8: страница

**Файлы:** `web/static/index.html`, `styles.css`, `api.js`, `geom.js`, `viewer.js`, `app.js`;
тест `tests/test_web_geom.py` (через `node --input-type=module`; без node — пропуск, в CI node
есть и охрана `REQUIRE_UI_TESTS` делает пропуск падением).

**`geom.js` производит:** `screenToFrame(sx, sy, view, w, h) -> [x, y] | null`,
`frameToScreen(x, y, view) -> [sx, sy]` (view = `{ox, oy, cssScale}`), `physicalScale(cssScale,
dpr)`, `cssScaleFor(physical, dpr)`, `applyH(H, x, y)`, `sigmaImagePx(scale, sigma)`,
`scaleName(scale)`.

- [ ] Тест `geom.js` против Python: соглашение полпикселя на масштабах 0.25…8 и сдвигах
  (перенос `test_ui_canvas`), последний столбец принят, за ним — `null`; `applyH` = `apply_homography`;
  `sigmaImagePx` = `zoom.sigma_image_px` на 0.5, 1, 2, 3, 4; `scaleName` = `texts.scale_name`.
- [ ] `viewer.js`: холст с backing store × DPR, панорама, колесо (1.25, якорь), Ctrl+колесо —
  зум кадра, фиксированные масштабы в физических пикселях, вписывание только по команде,
  клик ≤ 3 px, захват точки ≤ 8 px вне постановки, перетаскивание локально → `onDrop(key, x, y,
  scale, moved)`, наложения в экранной толщине, маска пригодности, режим «только просмотр» с `H`
  для выровненного вида и `onNavigate(x, y)`.
- [ ] `app.js`: экран «Снимки» (перетаскивание файлов, карточки со статусами), рабочий экран
  (шаги, панели шагов, подсказка, строка состояния, меню ⋯, диалоги), клавиши вне полей ввода,
  очередь действий, опрос при `busy`/растре, сохранение фокуса полей при перерисовке, фиксация
  длины базы по Enter/blur/1 с.
- [ ] Коммит «Страница веб-интерфейса оператора».

### Задача 9: точка входа, удаление Qt, CI, Docker

**Файлы:** `web/__main__.py`; `pyproject.toml` (`ui = ["fastapi>=0.110", "uvicorn>=0.29",
"python-multipart>=0.0.9"]`, `dev += httpx`, `facade-digitize-ui =
"facade_digitizer.web.__main__:main"`, package-data `web/static/*`); удалить Qt-модули и тесты;
`tests/ui_guard.py` — префиксы `test_ui_`, `test_web_`; `tests/conftest.py` без Qt;
`Dockerfile` (`python:3.11-slim`, `libgl1 libglib2.0-0`), `docker/start.sh`,
`docker-compose.yml` (порт `127.0.0.1:8765`), CI (без Qt, с node; docker — `/api/ping`).
Тест `tests/test_web_main.py`: `test_lock_reuses_running_server` (lock с портом живого
тестового сервера → `main` не стартует второй), `test_non_loopback_host_is_refused`.

- [ ] Реализовать; `pytest -q` — зелёный целиком; `ruff` чистый. Коммит «Точка входа веб-интерфейса, удаление Qt, Docker и CI».

### Задача 10: демо и документы

**Файлы:** `scripts/demo_ui.py` (через `Workbench` + `InlineRunner`; `--show` — сервер с
`--open`), `tests/test_ui_demo_script.py` → `tests/test_web_demo_script.py`; документы §5.3.

- [ ] Демо печатает истину и измерение как раньше; тест проверяет габарит в пределах 3σ и файлы в
  `экспорт/`. Документы переписаны. Коммит «Демо и документация под веб-интерфейс».

### Задача 11: ручная проверка в браузере

- [ ] `facade-digitize-ui --data-dir <tmp> --no-browser`, встроенный браузер: путь А на
  `01_фасад.jpg` (А1–А11), Б на `04_без_линий.jpg`, Г с чужим профилем, Ctrl+плюс (DPR 1.1) —
  «1:1» чёткий, клики записаны с физическим масштабом. Найденное — исправить с тестом.
- [ ] Коммит исправлений.

### Задача 12: независимая рецензия ветки

- [ ] Рецензент на Fable по всей ветке против спецификации; находки проверить, исправить,
  прогнать всё, коммит.
