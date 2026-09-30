// Интерфейс оператора: экран «Снимки» и рабочий экран с шагами
// Снимок → Масштаб → Проёмы → Результат.
//
// Страница ничего не считает: она рисует состояние сервера (`state`) и шлёт действия.
// Геометрия страницы — только перевод экран ↔ кадр (geom.js).

import { ApiError, createActor, get, post, upload } from './api.js';
import { applyH, scaleName, sigmaImagePx } from './geom.js';
import { Viewer } from './viewer.js';

const app = document.getElementById('app');
const toasts = document.getElementById('toasts');
const dialog = document.getElementById('dialog');

let S = null;              // состояние рабочего стола
let screen = 'home';
let tab = 'frame';         // 'frame' | 'rect'
let desk = null;           // элементы рабочего экрана
let viewer = null;
let rectViewer = null;
let pollTimer = null;
let lastNotice = null;
let cursorPoint = null;
let highlight = null;

const VERDICT_CLASS = { ok: 'ok', degraded: 'warn', reject: 'bad' };
const STATUS_TEXT = {
  new: ['Новый', ''],
  in_progress: ['В работе', 'accent'],
  exported: ['Выгружен', 'ok'],
  changed_after_export: ['Изменён после выгрузки', 'warn'],
};
const PRESETS = [[0.5, '1:2', '5'], [1, '1:1', '1'], [2, '2:1', '2'], [4, '4:1', '4']];

// --- DOM ---------------------------------------------------------------------------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (key === 'value') el.value = value;
    else if (key === 'checked') el.checked = Boolean(value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// Как replaceChildren, но пустые (null, false) — не узлы: иначе на экране «null».
function fill(el, ...children) {
  el.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
}

function chip(text, kind = '', dot = true) {
  return h('span', { class: `chip ${kind}` }, dot && kind ? h('span', { class: 'dot' }) : null, text);
}

function fmt(value, digits = 1) {
  return value == null ? '—' : Number(value).toFixed(digits).replace('.', ',');
}

function parseNumber(text) {
  const t = String(text ?? '').trim().replace(',', '.');
  if (t === '') return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : NaN;
}

// --- уведомления и диалоги ---------------------------------------------------------

function toast(text, kind = 'info', timeout = kind === 'error' ? 9000 : 5000) {
  const el = h('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    h('div', { class: 'text' }, text),
    h('button', { 'aria-label': 'Закрыть', onclick: () => el.remove() }, '×'));
  toasts.append(el);
  while (toasts.children.length > 3) toasts.firstChild.remove();
  if (timeout) setTimeout(() => el.remove(), timeout);
}

function openDialog(title, body, actions) {
  dialog.replaceChildren(h('form', { class: 'dlg', method: 'dialog' },
    h('h3', {}, title), body,
    h('div', { class: 'actions' }, actions.map(([label, cls, fn]) =>
      h('button', {
        class: `btn ${cls}`, value: label,
        onclick: (e) => { e.preventDefault(); dialog.close(); fn?.(); },
      }, label)))));
  dialog.showModal();
  dialog.querySelector('.btn.primary')?.focus();
}

function confirmDialog(title, text, okLabel, onOk, danger = false) {
  openDialog(title, h('p', {}, text), [
    ['Отмена', 'quiet', null],
    [okLabel, danger ? 'primary danger' : 'primary', onOk],
  ]);
}

function showError(error) {
  toast(error instanceof ApiError || error instanceof Error ? error.message : String(error), 'error');
}

// --- действия ----------------------------------------------------------------------

const act = createActor({
  getState: () => S,
  applyState: (state) => applyState(state),
  onError: (error) => {
    if (error.status === 409 && error.data?.error === 'состояние изменилось') {
      toast('Действие не выполнено: состояние успело измениться — повторите его.', 'error');
    } else if (error.status === 409) {
      toast(`Подождите: ${error.message}`, 'info');
    } else {
      showError(error);
    }
  },
});

function applyState(state) {
  if (!state) return;
  if (S && state.desk && S.desk === state.desk && state.rev < S.rev) return;
  S = state;
  if (!state.desk) {
    if (state.question) showOperatorQuestion(state.question);
    return;
  }
  if (screen !== 'desk') enterDesk();
  renderDesk();
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(pollTimer);
  if (!S?.desk) return;
  if (S.busy || S.rectified?.status === 'pending') {
    pollTimer = setTimeout(async () => {
      try {
        applyState((await get('/api/state')).state);
      } catch (error) {
        showError(error);
      }
    }, 400);
  }
}

async function openImage(path) {
  try {
    const result = await post('/api/open', { path });
    if (!result.state.desk && result.state.question) {
      S = result.state;
      showOperatorQuestion(result.state.question);
      return;
    }
    tab = 'frame';
    applyState(result.state);
  } catch (error) {
    showError(error);
  }
}

function showOperatorQuestion(question) {
  openDialog('Сессия другого оператора', h('p', {}, question.text), [
    ['Отмена', 'quiet', null],
    ['Начать заново', '', () => answerOperator('restart')],
    [`Продолжить как ${question.operator}`, 'primary', () => answerOperator('continue')],
  ]);
}

async function answerOperator(choice) {
  try {
    applyState((await post('/api/operator', { choice })).state);
  } catch (error) {
    showError(error);
  }
}

// --- экран «Снимки» ----------------------------------------------------------------

async function showHome() {
  screen = 'home';
  desk = null;
  viewer = rectViewer = null;
  clearTimeout(pollTimer);
  let library;
  try {
    library = await get('/api/library');
  } catch (error) {
    app.replaceChildren(h('div', { class: 'home' }, h('div', { class: 'empty' }, error.message)));
    return;
  }
  const input = h('input', {
    type: 'file', multiple: true, hidden: true, accept: '.jpg,.jpeg,.png,.tif,.tiff',
    onchange: () => uploadImages([...input.files]),
  });
  const drop = h('label', { class: 'drop' }, input,
    h('div', { class: 'drop-title' }, 'Перетащите снимок фасада сюда'),
    h('div', { class: 'drop-sub' }, 'или ', h('span', { class: 'link' }, 'выберите файл'),
      ' — JPG, PNG или TIFF. Снимок сохранится в папке «снимки».'));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    uploadImages([...e.dataTransfer.files]);
  });

  const folders = library.folders.map((folder) => h('section', { class: 'folder' },
    h('h2', {}, folder.path || 'Рабочая папка'),
    h('div', { class: 'grid' }, folder.images.map(imageCard))));

  app.replaceChildren(h('div', { class: 'home' },
    h('header', { class: 'home-head' },
      h('div', { class: 'brand' },
        h('h1', {}, 'Оцифровка фасада'),
        h('p', {}, 'Откройте снимок, задайте масштаб по известной длине и разметьте проёмы — '
          + 'программа посчитает размеры в миллиметрах с погрешностью и выгрузит JSON и DXF.')),
      h('div', { class: 'home-meta' },
        h('div', {}, 'Рабочая папка'), h('div', { class: 'mono' }, library.data_dir),
        h('div', { style: 'margin-top:6px' }, 'Оператор ', h('span', { class: 'mono' }, library.operator)))),
    drop,
    folders.length ? folders : h('div', { class: 'empty' },
      'В рабочей папке пока нет снимков. Перетащите первый в рамку выше.')));
}

function imageCard(image) {
  const [label, kind] = STATUS_TEXT[image.status] || [image.status, ''];
  return h('button', {
    class: 'card', disabled: image.collision,
    title: image.collision ? 'Рядом лежит снимок с тем же именем и другим расширением' : image.path,
    onclick: () => openImage(image.path),
  },
  h('div', { class: 'thumb' }, h('img', {
    loading: 'lazy', alt: '', src: `/api/thumb?path=${encodeURIComponent(image.path)}`,
  })),
  h('div', { class: 'meta' },
    h('div', { class: 'name' }, image.name),
    image.collision
      ? h('div', { class: 'note' }, 'Одинаковое имя с другим снимком — переименуйте один из них.')
      : h('div', {}, chip(label, kind))));
}

async function uploadImages(files) {
  const images = files.filter((f) => /\.(jpe?g|png|tiff?)$/i.test(f.name));
  if (!images.length) {
    toast('Это не снимок: подходят файлы JPG, PNG и TIFF.', 'error');
    return;
  }
  const drop = app.querySelector('.drop');
  drop?.classList.add('busy');
  let first = null;
  for (const file of images) {
    try {
      const result = await upload('/api/upload', file);
      first ??= result.path;
    } catch (error) {
      showError(error);
    }
  }
  drop?.classList.remove('busy');
  if (first && images.length === 1) openImage(first);
  else showHome();
}

// --- рабочий экран: каркас ------------------------------------------------------------

function enterDesk() {
  screen = 'desk';
  const frameHost = h('div', { class: 'viewport' });
  const rectHost = h('div', { class: 'viewport', hidden: true });
  const tabFrame = h('button', { class: 'tab active', onclick: () => setTab('frame') }, 'Снимок');
  const tabRect = h('button', { class: 'tab', onclick: () => setTab('rect') }, 'Выровненный вид');
  const zoom = h('div', { class: 'zoom' },
    PRESETS.map(([scale, label, key]) => h('button', {
      'data-scale': scale, title: `Масштаб ${label} (клавиша ${key})`,
      onclick: () => viewer?.setPhysicalScale(scale),
    }, label)),
    h('button', { title: 'Вписать снимок (клавиша 0)', onclick: () => viewer?.fit() }, 'Вписать'));
  const menuList = h('div', { class: 'menu-list', hidden: true, role: 'menu' });
  const menuButton = h('button', {
    class: 'btn quiet', 'aria-haspopup': 'menu', title: 'Ещё',
    onclick: (e) => { e.stopPropagation(); toggleMenu(); },
  }, 'Ещё ▾');

  desk = {
    fileName: h('b', {}),
    filePath: h('span', { class: 'muted small' }),
    save: h('span', { class: 'save' }),
    rail: h('nav', { class: 'rail', 'aria-label': 'Шаги работы' }),
    work: h('div', { class: 'work' }),
    stage: h('div', { class: 'stage' }),
    frameHost, rectHost, tabFrame, tabRect, zoom,
    hint: h('div', { class: 'hint' }),
    busy: h('div', { class: 'busy', hidden: true }),
    empty: h('div', { class: 'stage-empty', hidden: true }),
    panel: h('aside', { class: 'panel' }),
    status: h('footer', { class: 'statusbar' }),
    menuList, menuButton,
  };
  desk.stage.append(frameHost, rectHost, h('div', { class: 'tabs' }, tabFrame, tabRect),
    desk.hint, zoom, desk.busy, desk.empty);
  desk.work.append(desk.stage, desk.panel);
  app.replaceChildren(h('div', { class: 'desk' },
    h('header', { class: 'topbar' },
      h('div', { class: 'left' },
        h('button', { class: 'btn quiet', onclick: leaveDesk, title: 'К списку снимков' }, '← Снимки'),
        h('div', { class: 'file' }, desk.fileName, desk.filePath)),
      desk.rail,
      h('div', { class: 'right' }, desk.save, h('div', { class: 'menu' }, menuButton, menuList))),
    desk.work, desk.status));

  viewer = new Viewer(frameHost, {
    onClick: (x, y, scale) => act('click', { x, y, view_scale: scale }),
    onDrop: (key, x, y, scale, moved) => act('drop', { key, x, y, view_scale: scale, moved }),
    onCursor: (p) => { cursorPoint = p; renderStatus(); },
    onView: () => { renderZoom(); renderStatus(); },
    cursor: () => (S?.hint?.placing ? 'crosshair' : 'grab'),
  });
  rectViewer = new Viewer(rectHost, {
    readOnly: true,
    onClick: (x, y) => {
      const rect = S?.rectified;
      if (rect?.status !== 'ready') return;
      const [fx, fy] = applyH(rect.H_inv, x, y);
      setTab('frame');
      viewer.centerOn(fx, fy);
    },
  });
  document.addEventListener('click', closeMenu);
}

function leaveDesk() {
  post('/api/close').catch(() => {});
  S = null;
  document.removeEventListener('click', closeMenu);
  showHome();
}

function setTab(next) {
  if (next === 'rect' && S?.rectified?.status !== 'ready') return;
  tab = next;
  desk.frameHost.hidden = tab !== 'frame';
  desk.rectHost.hidden = tab !== 'rect';
  desk.tabFrame.classList.toggle('active', tab === 'frame');
  desk.tabRect.classList.toggle('active', tab === 'rect');
  desk.zoom.hidden = tab !== 'frame';
  if (tab === 'rect') {
    rectViewer.resize();
    rectViewer.fit();
  } else {
    viewer.resize();
  }
  renderHint();
}

function toggleMenu() {
  const list = desk.menuList;
  if (!list.hidden) {
    list.hidden = true;
    return;
  }
  const item = (label, fn) => h('button', { role: 'menuitem', onclick: () => { list.hidden = true; fn(); } }, label);
  list.replaceChildren(
    item('Сохранить копию сессии…', saveCopy),
    item('Скачать файл сессии', () => { window.location.href = '/api/session/download'; }),
    item('Профиль камеры…', () => act('set_step', { step: 'frame' })),
    h('hr'),
    item('Горячие клавиши', showKeys),
    h('hr'),
    item('Начать разметку заново…', () => confirmDialog('Начать заново?',
      'Все точки, база и проёмы этого снимка будут сброшены. Прежняя сессия сохранится '
      + 'рядом со снимком в файле с расширением .bak.', 'Начать заново', async () => {
        try {
          applyState((await post('/api/restart')).state);
        } catch (error) {
          showError(error);
        }
      }, true)),
  );
  list.hidden = false;
}

function closeMenu() {
  if (desk?.menuList) desk.menuList.hidden = true;
}

function saveCopy() {
  const folder = h('input', { class: 'input', value: `повторы/${S.operator}` });
  openDialog('Сохранить копию сессии', h('div', {},
    h('p', {}, 'Копия ляжет в папку внутри рабочей под именем со временем — так собираются '
      + 'повторные разметки для протокола точности клика.'),
    h('label', { class: 'field' }, h('span', {}, 'Папка'), folder)), [
    ['Отмена', 'quiet', null],
    ['Сохранить копию', 'primary', async () => {
      try {
        const result = await post('/api/session/copy', { folder: folder.value });
        toast(`Копия сохранена: ${result.path}`);
      } catch (error) {
        showError(error);
      }
    }],
  ]);
}

function showKeys() {
  const rows = [
    ['1', 'масштаб 1:1'], ['2', 'масштаб 2:1'], ['4', 'масштаб 4:1'], ['5', 'масштаб 1:2'],
    ['0', 'вписать снимок'], ['колесо', 'приблизить / отдалить под курсором'],
    ['перетаскивание', 'двигать снимок (или точку, если навести на неё)'],
    ['пробел + мышь', 'двигать снимок всегда'], ['Esc', 'закончить постановку точек'],
    ['Ctrl+Z', 'отменить последнюю точку проёма'], ['?', 'эта справка'],
  ];
  openDialog('Горячие клавиши', h('div', { class: 'keys' },
    rows.map(([key, text]) => [h('span', {}, h('kbd', {}, key)), h('span', {}, text)])),
  [['Понятно', 'primary', null]]);
}

// --- рабочий экран: отрисовка ----------------------------------------------------------

function renderDesk() {
  if (!desk || !S) return;
  desk.fileName.textContent = S.image.name || '';
  desk.filePath.textContent = S.image.path || '';
  renderSave();
  renderRail();
  desk.work.classList.toggle('wide', S.step === 'result');
  renderStage();
  renderPanel();
  renderHint();
  renderZoom();
  renderStatus();
  renderNotice();
}

function renderSave() {
  const save = S.save;
  desk.save.className = `save ${save.ok ? '' : 'bad'}`;
  desk.save.title = save.ok ? 'Сессия сохраняется сама после каждого действия' : save.error;
  fill(desk.save, h('span', { class: 'dot' }),
    h('span', { class: 'label' }, save.ok ? 'Сохранено' : 'Сессия не сохраняется'),
    save.ok ? null : h('a', { href: '/api/session/download', style: 'margin-left:6px' }, 'скачать'));
}

function renderRail() {
  const frameReady = Boolean(S.frame?.ready);
  desk.rail.replaceChildren(...S.steps.map((step, i) => {
    const active = step.key === S.step;
    const mark = step.status === 'done' ? '✓' : step.status === 'attention' ? '!' : '';
    return h('button', {
      class: `step ${step.status} ${active ? 'active' : ''}`,
      disabled: step.key !== 'frame' && !frameReady,
      'aria-current': active ? 'step' : null,
      onclick: () => act('set_step', { step: step.key }),
    }, h('span', { class: 'label' }, h('span', { class: 'no' }, `${i + 1}`), step.title,
      mark ? h('span', { class: 'mark' }, mark) : null), active ? h('span', { class: 'bar' }) : null);
  }));
}

function renderStage() {
  const frame = S.frame;
  desk.busy.hidden = !S.busy;
  desk.busy.replaceChildren(h('div', { class: 'box' }, h('div', { class: 'spinner' }), S.busy || ''));
  if (frame?.ready) {
    desk.empty.hidden = true;
    const url = `/api/frame.png?v=${S.desk}-${frame.version}`;
    if (viewer.url !== url) {
      desk.empty.hidden = false;
      desk.empty.textContent = 'Загружаю снимок…';
      viewer.setImage(url, frame.width, frame.height, { fit: true })
        .then(() => { desk.empty.hidden = true; })
        .catch(showError);
    }
    viewer.setLayer('mask', frame.usable && S.step !== 'result'
      ? `/api/usable.png?v=${S.desk}-${frame.usable_version}` : null);
  } else if (!S.busy) {
    viewer.clearImage();
    desk.empty.hidden = false;
    desk.empty.textContent = frame?.error
      ? `Снимок не обработан: ${frame.error}` : 'Снимок ещё не обработан.';
  }
  const grab = S.draggable.length > 0 && !S.placing;
  viewer.setOverlays(S.overlays, S.draggable, grab, highlight, S.rev);

  const rect = S.rectified;
  desk.tabRect.disabled = rect.status !== 'ready';
  desk.tabRect.title = rect.text || '';
  desk.tabRect.textContent = rect.status === 'pending' ? 'Выровненный вид…' : 'Выровненный вид';
  if (rect.status === 'ready') {
    rectViewer.setImage(`/api/rectified.jpg?v=${S.desk}-${rect.version}`, rect.width, rect.height,
      { fit: tab === 'rect' });
    rectViewer.setLayer('hatch', `/api/rectified_mask.png?v=${S.desk}-${rect.version}`);
    const mapped = S.overlays.filter((o) => o.key.startsWith('mark:') || o.key === 'base')
      .map((o) => ({ ...o, keys: null, points: o.points.map(([x, y]) => applyH(rect.H, x, y)) }));
    rectViewer.setOverlays(mapped, [], false, highlight);
  } else if (tab === 'rect') {
    setTab('frame');
  }
}

function renderHint() {
  const hint = S?.hint;
  if (!hint || tab !== 'frame' || S.busy) {
    desk.hint.replaceChildren();
    desk.hint.className = 'hint';
    if (tab === 'rect' && S?.rectified?.text) desk.hint.append(h('span', {}, S.rectified.text));
    return;
  }
  desk.hint.className = `hint ${hint.placing ? 'placing' : ''}`;
  fill(desk.hint,
    hint.placing && hint.total ? h('span', { class: 'count' }, `${hint.count}/${hint.total}`) : null,
    h('span', {}, hint.text),
    hint.placing ? h('button', { class: 'btn', onclick: () => act('stop_placing') }, 'Закончить · Esc') : null);
}

function renderZoom() {
  if (!desk || !viewer) return;
  const scale = viewer.physical;
  for (const button of desk.zoom.querySelectorAll('button[data-scale]')) {
    button.classList.toggle('active', Math.abs(Number(button.dataset.scale) - scale) < 1e-6);
  }
}

function renderStatus() {
  if (!desk || !S || !viewer) return;
  const scale = viewer.physical;
  const sigma = S.sigma;
  const coarse = ['corners', 'reveal'].includes(S.placing) && scale < sigma.recommended_min_scale;
  const parts = [
    h('span', { class: coarse ? 'warn' : '' },
      `Масштаб ${scaleName(scale)} · σ клика ${fmt(sigmaImagePx(scale, sigma), 2)} px`,
      coarse ? ` — у углов увеличьте до ${scaleName(sigma.recommended_min_scale)} и больше` : ''),
    h('span', { class: 'mono' }, cursorPoint
      ? `x ${cursorPoint[0].toFixed(1)}  y ${cursorPoint[1].toFixed(1)}` : ''),
    h('span', { class: 'sep' }),
  ];
  if (viewer.dpr !== 1) {
    parts.push(h('span', { title: 'Масштабы указаны в пикселях монитора' },
      `экран ${Math.round(viewer.dpr * 100)} %`));
  }
  parts.push(h('span', {}, 'Оператор ', h('span', { class: 'mono' }, S.operator || '')));
  desk.status.replaceChildren(...parts);
}

function renderNotice() {
  const notice = S.notice;
  const key = notice ? `${S.desk}|${S.rev}|${JSON.stringify(notice)}` : null;
  if (!notice || key === lastNotice) return;
  lastNotice = key;
  if (notice.kind === 'confirm') {
    confirmDialog('Подтвердите', notice.text, 'Сбросить и продолжить',
      () => act(notice.confirm.name, notice.confirm.args), true);
  } else {
    toast(notice.text, notice.kind === 'error' ? 'error' : 'info');
  }
}

// --- панели шагов --------------------------------------------------------------------

// Перерисовка панели не должна отнимать у оператора поле, в которое он пишет: фокус,
// выделение и недописанный текст переносятся в новое поле с тем же `data-keep`.
function renderPanel() {
  const active = document.activeElement;
  const keep = active?.dataset?.keep && desk.panel.contains(active)
    ? { key: active.dataset.keep, value: active.value, start: active.selectionStart,
        end: active.selectionEnd } : null;
  const scroll = desk.panel.querySelector('.panel-body')?.scrollTop || 0;
  const build = { frame: panelFrame, scale: panelScale, marks: panelMarks, result: panelResult }[S.step];
  const [body, foot] = build();
  desk.panel.replaceChildren(h('div', { class: 'panel-body' }, body), foot ? h('div', { class: 'panel-foot' }, foot) : '');
  desk.panel.querySelector('.panel-body').scrollTop = scroll;
  if (keep) {
    const el = desk.panel.querySelector(`[data-keep="${keep.key}"]`);
    if (el) {
      if ('value' in el && el.type !== 'checkbox' && el.type !== 'radio') el.value = keep.value;
      el.focus({ preventScroll: true });
      try {
        el.setSelectionRange(keep.start, keep.end);
      } catch { /* поле без выделения */ }
    }
  }
}

function nextButton(label, step, enabled) {
  return h('button', { class: 'btn primary wide', disabled: !enabled,
    onclick: () => act('set_step', { step }) }, label);
}

function qualityBlock(q, title) {
  if (!q) return null;
  return h('section', { class: 'block' },
    h('h3', {}, title, q.verdict ? chip(q.verdict, VERDICT_CLASS[q.verdict]) : null),
    h('div', {}, h('b', {}, q.verdict_text), h('span', { class: 'muted' }, ` · ${q.stage_text}`)),
    q.reasons.length ? h('ul', { class: 'reasons' }, q.reasons.map((r) => h('li', {}, r))) : null,
    q.note ? h('p', { class: 'note' }, q.note) : null);
}

function panelFrame() {
  const frame = S.frame || {};
  const plane = frame.plane;
  const placing = S.placing;
  const body = [
    h('h2', {}, 'Снимок'),
    h('p', { class: 'lede' }, 'Проверьте, что снимок распознан и плоскость фасада найдена. '
      + 'Если есть профиль калибровки камеры — подключите его до разметки.'),
  ];
  if (!frame.ready) {
    body.push(h('div', { class: `callout ${frame.error ? 'bad' : ''}` },
      frame.error ? `Снимок не обработан: ${frame.error}` : S.busy || 'Снимок обрабатывается…'));
  } else {
    body.push(h('section', { class: 'block' }, h('h3', {}, 'Сведения'),
      h('ul', { class: 'facts' }, frame.info.map((line) => h('li', {}, line)))));
  }
  body.push(profileBlock(frame));
  if (plane) {
    const status = plane.needs_operator ? chip('не найдена', 'bad')
      : plane.manual ? chip('задана вручную', 'accent') : chip('найдена', 'ok');
    const block = h('section', { class: 'block' }, h('h3', {}, 'Плоскость фасада', status),
      h('p', { class: 'small', style: 'margin:0' }, plane.text));
    if (placing === 'plane') {
      block.append(planeForm(plane));
    } else if (placing === 'roi') {
      block.append(h('div', { class: 'callout accent' },
        'Обведите на снимке участок стены с ровными горизонталями и вертикалями — '
        + 'точки схода будут оценены только по нему. Нужно не менее трёх точек.'),
      h('div', { class: 'row', style: 'margin-top:10px' },
        h('button', { class: 'btn primary', disabled: plane.roi_points < 3,
          onclick: () => act('apply_roi') }, 'Оценить по области'),
        h('button', { class: 'btn quiet', onclick: () => act('stop_placing') }, 'Отмена')));
    } else {
      block.append(h('div', { class: 'row', style: 'margin-top:10px' },
        h('button', { class: 'btn small', onclick: () => act('start_manual_plane') }, 'Задать вручную'),
        h('button', { class: 'btn small', title: 'Для снимка угла здания', onclick: () => act('start_roi') },
          'Обвести область'),
        plane.manual || plane.roi
          ? h('button', { class: 'btn small quiet', onclick: () => act('reset_plane') }, 'Вернуть автоматическую')
          : null));
    }
    body.push(block);
  }
  body.push(qualityBlock(S.quality?.stage === 'preliminary' ? S.quality : null, 'Качество снимка'));
  const ready = frame.ready && plane && !plane.needs_operator;
  return [body, nextButton('Далее: масштаб →', 'scale', ready)];
}

function profileBlock(frame) {
  const current = frame.profile?.path || '';
  const fileInput = h('input', {
    type: 'file', accept: '.json', hidden: true,
    onchange: async () => {
      const file = fileInput.files[0];
      if (!file) return;
      try {
        const result = await upload('/api/profile/upload', file);
        profilesCache = null;
        act('set_profile', { path: result.path, discard: false });
      } catch (error) {
        showError(error);
      }
    },
  });
  const select = h('select', {
    class: 'input', 'aria-label': 'Профиль камеры',
    onchange: () => act('set_profile', { path: select.value || null, discard: false }),
  }, h('option', { value: '' }, 'Без профиля (K из EXIF или таблицы)'));
  loadProfiles(select, current);
  return h('section', { class: 'block' }, h('h3', {}, 'Профиль камеры'),
    h('div', { class: 'row' }, h('div', { class: 'grow' }, select),
      h('button', { class: 'btn', onclick: () => fileInput.click() }, 'Загрузить…'), fileInput),
    h('p', { class: 'note' }, 'Профиль снимает дисторсию и сдвигает кадр: подключайте его '
      + 'до разметки, иначе точки придётся сбросить.'));
}

// Список профилей — один раз на открытый снимок (и после загрузки нового), а не при
// каждой перерисовке панели: библиотека обходит всю рабочую папку.
let profilesCache = null;
async function loadProfiles(select, current) {
  if (profilesCache?.desk !== S.desk) {
    try {
      profilesCache = { desk: S.desk, list: (await get('/api/library')).profiles };
    } catch {
      profilesCache = { desk: null, list: [] };
    }
  }
  const known = new Set();
  for (const profile of profilesCache.list) {
    known.add(profile.path);
    select.append(h('option', { value: profile.path }, profile.name));
  }
  if (current && !known.has(current)) select.append(h('option', { value: current }, current));
  select.value = current;
}

function planeForm(plane) {
  const c = plane.constraint;
  const send = (kind) => {
    const f = desk.panel;
    act('set_plane_constraint', {
      kind: kind || f.querySelector('input[name="constraint"]:checked')?.value || c.kind,
      aspect: parseNumber(f.querySelector('[data-keep="aspect"]').value),
      width_mm: parseNumber(f.querySelector('[data-keep="plane-w"]').value),
      height_mm: parseNumber(f.querySelector('[data-keep="plane-h"]').value),
    });
  };
  const radio = (kind, label, extra) => h('label', { class: 'radio-row' },
    h('input', { type: 'radio', name: 'constraint', value: kind, checked: c.kind === kind,
      onchange: () => send(kind) }),
    h('span', {}, label, extra ? h('div', { style: 'margin-top:4px' }, extra) : null));
  const numberInput = (keep, value, unit) => h('div', { class: 'unit-input', 'data-unit': unit },
    h('input', { class: 'input num', inputmode: 'decimal', 'data-keep': keep, value: value ?? '',
      onchange: () => send() }));
  return h('div', {},
    h('div', { class: 'callout accent' }, `Кликните четыре угла заведомого прямоугольника в `
      + `плоскости стены (проём, панель, край фасада): указано ${plane.points} из 4.`),
    h('p', { class: 'small', style: 'margin:12px 0 4px' }, 'Что известно об этом прямоугольнике:'),
    radio('sizes', 'Ширина и высота', h('div', { class: 'row' },
      h('div', { class: 'grow' }, numberInput('plane-w', c.width_mm, 'мм')), '×',
      h('div', { class: 'grow' }, numberInput('plane-h', c.height_mm, 'мм')))),
    radio('aspect', 'Отношение сторон (ширина / высота)', numberInput('aspect', c.aspect, '')),
    radio('calibrated', 'Камера откалибрована, стороны ортогональны'),
    h('div', { class: 'row', style: 'margin-top:12px' },
      h('button', { class: 'btn primary', disabled: plane.points !== 4,
        onclick: () => act('apply_manual_plane') }, 'Применить плоскость'),
      h('button', { class: 'btn quiet', onclick: () => act('stop_placing') }, 'Отмена')));
}

let baseTimer = null;
function panelScale() {
  const sc = S.scale;
  const v = sc.values;
  const commit = () => {
    clearTimeout(baseTimer);
    const f = desk.panel;
    const next = {
      span_mm: parseNumber(f.querySelector('[data-keep="span"]').value),
      span_sigma_mm: parseNumber(f.querySelector('[data-keep="span-sigma"]').value),
      scale_source: f.querySelector('[data-keep="source"]').value,
      origin_is_facade_corner: f.querySelector('[data-keep="origin"]').checked,
    };
    if (Number.isNaN(next.span_mm) || Number.isNaN(next.span_sigma_mm)) {
      toast('Длина — число в миллиметрах, например 1460 или 1460,5.', 'error');
      return;
    }
    const same = next.span_mm === v.span_mm && next.span_sigma_mm === v.span_sigma_mm
      && next.scale_source === v.scale_source
      && next.origin_is_facade_corner === v.origin_is_facade_corner;
    if (!same) act('set_base', next);
  };
  const later = () => { clearTimeout(baseTimer); baseTimer = setTimeout(commit, 1000); };
  const onKey = (e) => { if (e.key === 'Enter') commit(); };
  const span = h('input', { class: 'input num', inputmode: 'decimal', 'data-keep': 'span',
    value: v.span_mm ?? '', placeholder: 'например, 1460', oninput: later, onchange: commit,
    onkeydown: onKey });
  const spanSigma = h('input', { class: 'input num', inputmode: 'decimal', 'data-keep': 'span-sigma',
    value: v.span_sigma_mm ?? '', placeholder: 'не задана', oninput: later, onchange: commit,
    onkeydown: onKey });

  const ends = h('section', { class: 'block' },
    h('h3', {}, 'Концы базы', chip(`${sc.ends} из 2`, sc.ends === 2 ? 'ok' : 'accent')),
    sc.end_lines.length ? h('ul', { class: 'facts' }, sc.end_lines.map((l) => h('li', {}, l))) : null,
    h('div', { class: 'row', style: 'margin-top:8px' },
      h('button', { class: 'btn small', onclick: () => act('restart_base') },
        sc.ends ? 'Указать заново' : 'Указать на снимке')));

  const fields = h('section', { class: 'block' }, h('h3', {}, 'Длина базы'),
    h('label', { class: 'field' }, h('span', {}, 'Длина между концами'),
      h('div', { class: 'unit-input', 'data-unit': 'мм' }, span)),
    h('label', { class: 'field' }, h('span', {}, 'Погрешность длины — необязательно'),
      h('div', { class: 'unit-input', 'data-unit': 'мм' }, spanSigma)),
    h('label', { class: 'field' }, h('span', {}, 'Откуда длина'),
      h('select', { class: 'input', 'data-keep': 'source', onchange: commit },
        S.choices.scale_sources.map(([key, text]) =>
          h('option', { value: key, selected: key === v.scale_source ? true : null }, text)))),
    h('label', { class: 'check' },
      h('input', { type: 'checkbox', 'data-keep': 'origin', checked: v.origin_is_facade_corner,
        onchange: commit }),
      h('span', {}, 'Первый конец — левый нижний угол фасада',
        h('div', { class: 'note', style: 'margin:2px 0 0' },
          'Тогда координаты — от угла здания и охват полный; иначе — от первого конца базы.'))));

  let result;
  if (sc.result) {
    result = h('section', { class: 'block' },
      h('h3', {}, 'Масштаб', chip(sc.tolerance_ok ? 'в допуске' : 'вне допуска',
        sc.tolerance_ok ? 'ok' : 'warn')),
      h('p', { class: 'small', style: 'margin:0' }, sc.result));
  } else if (sc.ends === 2 && sc.error) {
    result = h('div', { class: 'callout' }, `Масштаб не посчитан: ${sc.error}`);
  }
  const body = [
    h('h2', {}, 'Масштаб'),
    h('p', { class: 'lede' }, 'Кликните на снимке два конца отрезка известной длины в плоскости '
      + 'стены — как можно дальше друг от друга: короткая база даёт большую погрешность. '
      + 'Первый конец станет началом отсчёта.'),
    ends, fields, result,
    S.quality?.stage === 'final' ? qualityBlock(S.quality, 'Качество снимка') : null,
  ];
  return [body, nextButton('Далее: проёмы →', 'marks', Boolean(sc.result))];
}

function panelMarks() {
  const m = S.marks;
  const choices = S.choices;
  const cards = m.items.map((item) => {
    const selected = item.id === m.selected;
    const progress = item.corners < 4 ? `углы ${item.corners} из 4`
      : item.reveal_side ? `откос: ${item.reveal_side_text} ${item.reveal_points}/2`
        : 'откос не размечен';
    const head = h('button', { class: 'mark-head', onclick: () => act('select_mark', { id: item.id }) },
      h('span', { class: 'mark-swatch' }),
      h('span', { class: 'mark-title' }, h('b', {}, `${item.id}`),
        h('span', { class: `progress ${item.corners === 4 ? 'done' : ''}` }, `${item.class_text} · ${progress}`)),
      h('span', { class: 'mark-size' }, !selected && item.size ? item.size : ''));
    const card = h('div', { class: `mark-card ${selected ? 'selected' : ''}` }, head);
    if (selected) card.append(markBody(item, m, choices));
    return card;
  });
  const body = [
    h('h2', {}, 'Проёмы'),
    h('p', { class: 'lede' }, 'Добавьте проём и кликните его четыре угла в любом порядке — '
      + 'у углов приблизьте снимок до 2:1 и больше. Размеры посчитаются сами.'),
    h('div', { class: 'row' },
      h('button', { class: 'btn', onclick: () => act('new_mark', { class_: 'window' }) }, '+ Окно'),
      h('button', { class: 'btn', onclick: () => act('new_mark', { class_: 'door' }) }, '+ Дверь')),
    m.items.length ? h('div', { class: 'marks-list' }, cards)
      : h('div', { class: 'callout', style: 'margin-top:14px' }, 'Проёмов пока нет.'),
    m.warning ? h('p', { class: 'note' }, m.warning) : null,
    m.reasons.length ? h('div', { class: 'callout warn' }, h('b', {}, 'Чтобы посчитать размеры: '),
      h('ul', { class: 'reasons' }, m.reasons.map((r) => h('li', {}, r)))) : null,
  ];
  return [body, nextButton('Далее: результат →', 'result', S.result.rows.length > 0)];
}

function markBody(item, m, choices) {
  const select = (label, key, value, options) => h('label', { class: 'field' }, h('span', {}, label),
    h('select', { class: 'input', onchange: (e) => act('set_mark_attrs', { id: item.id, [key]: e.target.value }) },
      options.map(([k, text]) => h('option', { value: k, selected: k === value ? true : null }, text))));
  const parts = [];
  if (item.size) parts.push(h('div', { class: 'dimline' }, item.size));
  parts.push(select('Установка', 'mounting', item.mounting, choices.mountings),
    select('Тип кромки', 'edge_type', item.edge_type, choices.edges));
  if (item.corners === 4) {
    parts.push(h('label', { class: 'field' }, h('span', {}, 'Грань откоса'),
      h('div', { class: 'row' },
        h('div', { class: 'grow' }, h('select', {
          class: 'input', onchange: (e) => act('set_reveal_side', { side: e.target.value }),
        }, choices.sides.map(([k, text]) =>
          h('option', { value: k, selected: k === m.side_choice ? true : null }, text)))),
        h('button', { class: 'btn', onclick: () => act('start_reveal') },
          item.reveal_points ? 'Разметить заново' : 'Разметить откос'))));
    if (m.suggestion) parts.push(h('p', { class: 'note' }, m.suggestion));
  }
  parts.push(h('div', { class: 'row', style: 'margin-top:10px' },
    h('button', { class: 'btn small', onclick: () => act('undo_point') }, 'Отменить точку'),
    h('button', { class: 'btn small quiet danger', onclick: () => confirmDialog('Удалить проём?',
      `Проём ${item.id} и его точки будут удалены.`, 'Удалить',
      () => act('delete_mark', { id: item.id }), true) }, 'Удалить')));
  return h('div', { class: 'mark-body' }, parts);
}

function panelResult() {
  const r = S.result;
  const body = [h('h2', {}, 'Результат')];
  if (!r.rows.length) {
    body.push(h('div', { class: 'callout' }, 'Размеров пока нет: задайте масштаб и разметьте '
      + 'хотя бы один проём.'));
    return [body, nextButton('← К проёмам', 'marks', true)];
  }
  body.push(h('p', { class: 'lede small' }, r.header));
  const columns = r.columns;
  const keys = ['id', 'width', 'height', 'position', 'recess', 'theta', 'tolerance'];
  body.push(h('div', { style: 'overflow-x:auto' }, h('table', { class: 'result' },
    h('thead', {}, h('tr', {}, columns.map((c) => h('th', {}, c)))),
    h('tbody', {}, r.rows.map((row) => h('tr', {
      onmouseenter: () => { highlight = `mark:${row.id}`; viewer.setHighlight(highlight); rectViewer.setHighlight(highlight); },
      onmouseleave: () => { highlight = null; viewer.setHighlight(null); rectViewer.setHighlight(null); },
    }, keys.map((k) => h('td', {
      class: ['width', 'height', 'theta'].includes(k) ? 'n'
        : k === 'tolerance' ? `tol ${row.meets === true ? 'yes' : ''}` : '',
    }, row[k]))))))));
  const exp = r.export;
  const block = h('section', { class: 'block' }, h('h3', {}, 'Выгрузка',
    exp ? chip(S.steps[3].status === 'done' ? 'актуальна' : 'есть правки после выгрузки',
      S.steps[3].status === 'done' ? 'ok' : 'warn') : null));
  if (exp) {
    block.append(h('p', { class: 'small', style: 'margin:0' }, 'Папка: ', h('span', { class: 'mono' }, exp.dir)),
      h('ul', { class: 'files' }, exp.files.map((f) => h('li', {},
        h('a', { href: `/${f.url}`, download: f.name }, f.name)))),
      h('div', { class: 'cli' }, h('pre', {}, exp.cli),
        h('button', { class: 'btn small', onclick: () => {
          navigator.clipboard?.writeText(exp.cli).then(() => toast('Команда скопирована'));
        } }, 'Копировать')),
      h('p', { class: 'note' }, 'Эта команда воспроизводит тот же JSON без интерфейса.'));
  } else {
    block.append(h('p', { class: 'small', style: 'margin:0' }, 'JSON, DXF, разметка и команда '
      + 'для повтора ляжут в папку «экспорт» рядом со снимком.'));
  }
  body.push(block);
  const label = exp ? 'Выгрузить ещё раз' : 'Выгрузить JSON и DXF';
  return [body, h('button', { class: 'btn primary wide', onclick: () => act('export') }, label)];
}

// --- клавиатура ----------------------------------------------------------------------

function typing(target) {
  return target instanceof HTMLElement
    && (target.isContentEditable || ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName));
}

document.addEventListener('keydown', (e) => {
  if (screen !== 'desk' || dialog.open) return;
  if (e.key === ' ' && !typing(e.target)) {
    viewer.spaceDown = true;
    e.preventDefault();
    return;
  }
  if (e.key === 'Escape') {
    if (desk.menuList && !desk.menuList.hidden) {
      desk.menuList.hidden = true;
    } else if (S?.placing) {
      act('stop_placing');
    }
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !typing(e.target)) {
    if (S?.step === 'marks' && S.marks.selected) {
      e.preventDefault();
      act('undo_point');
    }
    return;
  }
  if (typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
  const preset = PRESETS.find(([, , key]) => key === e.key);
  if (preset) viewer.setPhysicalScale(preset[0]);
  else if (e.key === '0') viewer.fit();
  else if (e.key === '?') showKeys();
});
document.addEventListener('keyup', (e) => {
  if (e.key === ' ' && viewer) viewer.spaceDown = false;
});

// --- запуск --------------------------------------------------------------------------

(async function start() {
  try {
    const { state } = await get('/api/state');
    if (state.desk) {
      applyState(state);
    } else {
      await showHome();
      if (state.question) showOperatorQuestion(state.question);
    }
  } catch (error) {
    app.replaceChildren(h('div', { class: 'home' }, h('div', { class: 'empty' }, error.message)));
  }
})();
