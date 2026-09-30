// Холст кадра: масштаб, панорама, клики, перенос точек, наложения.
//
// Метрология окна Qt сохранена:
// * масштаб просмотра клика — физических пикселей монитора на пиксель кадра
//   (CSS-масштаб × devicePixelRatio), холст рисуется в физических пикселях;
// * при масштабе ≥ 1 пиксели кадра без сглаживания — видно, куда целишься;
// * колесо — шаг 1.25 с якорем под курсором, пределы 1:64 … 32:1; Ctrl+колесо и
//   «щипок» масштабируют кадр, а не страницу;
// * клик — нажатие и отпускание со смещением ≤ 3 px; больше — панорама;
// * перенос точки: захват в радиусе 8 экранных px, только когда разрешён (вне
//   постановки точек); во время переноса точка движется здесь, на сервер уходит
//   одно отпускание;
// * линии 2 px и точки 8 px — экранные при любом масштабе;
// * изменение размера окна масштаб не меняет: вписывание — только по команде.

import { frameToScreen, screenToFrame } from './geom.js';

const MIN_SCALE = 1 / 64;
const MAX_SCALE = 32;
const WHEEL_STEP = 1.25;
const GRAB_RADIUS = 8;
const CLICK_SLOP = 3;

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`изображение не загрузилось: ${url}`));
    img.src = url;
  });
}

export class Viewer {
  constructor(host, options = {}) {
    this.host = host;
    this.options = options;
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'viewer-canvas';
    this.canvas.tabIndex = -1;
    host.append(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.dpr = window.devicePixelRatio || 1;
    this.view = { ox: 0, oy: 0, cssScale: 1 };
    this.cw = 0;
    this.ch = 0;
    this.img = null;
    this.url = null;
    this.w = 0;
    this.h = 0;
    this.layers = {};        // mask, hatch: {url, img}
    this.overlays = [];
    this.draggable = [];
    this.grabEnabled = false;
    this.highlight = null;
    this.pointer = null;
    this.drag = null;        // {key, x, y} — точка в переносе
    this.pending = null;     // отпущена, ждёт ответа сервера
    this.spaceDown = false;
    this.fitted = false;

    new ResizeObserver(() => this.resize()).observe(host);
    // Запасной путь: окно, которое не отрисовывается (свёрнуто, за другими окнами),
    // не получает ResizeObserver, а размер к возвращению уже другой.
    window.addEventListener('resize', () => this.resize());
    this.canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    this.canvas.addEventListener('pointermove', (e) => this.onMove(e));
    this.canvas.addEventListener('pointerup', (e) => this.onUp(e));
    this.canvas.addEventListener('pointercancel', () => { this.pointer = null; this.drag = null; this.draw(); });
    this.canvas.addEventListener('pointerleave', () => this.options.onCursor?.(null));
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.watchDpr();
    this.resize();
  }

  get physical() {
    return this.view.cssScale * this.dpr;
  }

  watchDpr() {
    const query = matchMedia(`(resolution: ${this.dpr}dppx)`);
    query.addEventListener('change', () => {
      this.dpr = window.devicePixelRatio || 1;
      this.resize();
      this.changed();
      this.watchDpr();
    }, { once: true });
  }

  resize() {
    const rect = this.host.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const firstSize = this.cw === 0;
    // Центр вида остаётся на месте: окно растянули — кадр не «уезжает» и масштаб прежний.
    if (!firstSize && this.img) {
      this.view.ox += (rect.width - this.cw) / 2;
      this.view.oy += (rect.height - this.ch) / 2;
    }
    // CSS-размер холста — ровно буфер / DPR: иначе при дробной ширине и DPR ≠ 1
    // браузер растянул бы буфер на доли процента, и «1:1» не было бы пиксель в пиксель.
    this.canvas.width = Math.round(rect.width * this.dpr);
    this.canvas.height = Math.round(rect.height * this.dpr);
    this.cw = this.canvas.width / this.dpr;
    this.ch = this.canvas.height / this.dpr;
    this.canvas.style.width = `${this.cw}px`;
    this.canvas.style.height = `${this.ch}px`;
    if (this.img && !this.fitted) this.fit();
    this.draw();
  }

  async setImage(url, width, height, { fit = false } = {}) {
    if (url === this.url) return;
    this.url = url;
    const img = await loadImage(url);
    if (this.url !== url) return;              // пока грузилось, попросили другое
    const resized = width !== this.w || height !== this.h;
    this.img = img;
    this.w = width;
    this.h = height;
    if (fit || resized || !this.fitted) this.fit();
    this.draw();
    this.options.onLoad?.();
  }

  clearImage() {
    this.url = null;
    this.img = null;
    this.layers = {};
    this.fitted = false;
    this.draw();
  }

  async setLayer(name, url) {
    if (!url) {
      delete this.layers[name];
      this.draw();
      return;
    }
    if (this.layers[name]?.url === url) return;
    this.layers[name] = { url, img: null };
    const img = await loadImage(url).catch(() => null);
    if (this.layers[name]?.url === url) {
      this.layers[name].img = img;
      this.draw();
    }
  }

  // rev — ревизия состояния, из которого наложения: отпущенная точка остаётся на новом
  // месте, пока не придёт состояние новее отпускания (опрос без изменений её не сбросит).
  setOverlays(overlays, draggable, grabEnabled, highlight = null, rev = null) {
    this.overlays = overlays || [];
    this.draggable = draggable || [];
    this.grabEnabled = Boolean(grabEnabled);
    this.highlight = highlight;
    this.rev = rev;
    if (this.pending && (rev == null || this.pending.rev == null || rev > this.pending.rev)) {
      this.pending = null;
    }
    this.draw();
  }

  setHighlight(key) {
    this.highlight = key;
    this.draw();
  }

  // --- вид ------------------------------------------------------------------------

  fit() {
    if (!this.w || !this.cw) return;
    const s = Math.min(this.cw / this.w, this.ch / this.h) * 0.96;
    this.view.cssScale = s;
    this.view.ox = (this.cw - this.w * s) / 2;
    this.view.oy = (this.ch - this.h * s) / 2;
    this.fitted = true;
    this.changed();
  }

  setPhysicalScale(scale, anchor = null) {
    const target = Math.min(Math.max(scale, MIN_SCALE), MAX_SCALE);
    const css = target / this.dpr;
    const [ax, ay] = anchor || [this.cw / 2, this.ch / 2];
    const ix = (ax - this.view.ox) / this.view.cssScale;
    const iy = (ay - this.view.oy) / this.view.cssScale;
    this.view.cssScale = css;
    this.view.ox = ax - ix * css;
    this.view.oy = ay - iy * css;
    this.fitted = true;
    this.changed();
  }

  centerOn(x, y) {
    this.view.ox = this.cw / 2 - (x + 0.5) * this.view.cssScale;
    this.view.oy = this.ch / 2 - (y + 0.5) * this.view.cssScale;
    this.changed();
  }

  changed() {
    this.draw();
    this.options.onView?.(this);
  }

  // --- мышь ---------------------------------------------------------------------

  local(event) {
    const rect = this.canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  hit([sx, sy]) {
    if (!this.grabEnabled) return null;
    let best = null;
    let bestD = GRAB_RADIUS;
    for (const p of this.draggable) {
      const [px, py] = frameToScreen(p.x, p.y, this.view);
      const d = Math.hypot(px - sx, py - sy);
      if (d <= bestD) {
        best = p.key;
        bestD = d;
      }
    }
    return best;
  }

  onDown(event) {
    if (!this.img) return;
    this.canvas.focus({ preventScroll: true });
    const p = this.local(event);
    const pan = event.button === 1 || (event.button === 0 && this.spaceDown);
    if (event.button !== 0 && !pan) return;
    event.preventDefault();
    this.pointer = { x0: p[0], y0: p[1], ox: this.view.ox, oy: this.view.oy, moved: false,
                     mode: pan ? 'pan' : 'click' };
    if (!pan && !this.options.readOnly) {
      const key = this.hit(p);
      if (key) {
        this.pointer.mode = 'drag';
        this.pointer.key = key;
      }
    }
    this.canvas.setPointerCapture(event.pointerId);
    this.updateCursor(p);
  }

  onMove(event) {
    const p = this.local(event);
    this.options.onCursor?.(this.img ? screenToFrame(p[0], p[1], this.view, this.w, this.h) : null);
    const ptr = this.pointer;
    if (!ptr) {
      this.updateCursor(p);
      return;
    }
    const dx = p[0] - ptr.x0;
    const dy = p[1] - ptr.y0;
    if (Math.abs(dx) + Math.abs(dy) > CLICK_SLOP) ptr.moved = true;
    if (!ptr.moved) return;
    if (ptr.mode === 'drag') {
      const f = screenToFrame(p[0], p[1], this.view, this.w, this.h);
      if (f) {
        this.drag = { key: ptr.key, x: f[0], y: f[1] };
        this.draw();
      }
      return;
    }
    ptr.mode = 'pan';
    this.view.ox = ptr.ox + dx;
    this.view.oy = ptr.oy + dy;
    this.changed();
  }

  onUp(event) {
    const ptr = this.pointer;
    this.pointer = null;
    if (!ptr) return;
    const p = this.local(event);
    if (ptr.mode === 'drag') {
      const d = this.drag;
      this.drag = null;
      if (d && ptr.moved) {
        this.pending = { ...d, rev: this.rev };
        this.options.onDrop?.(d.key, d.x, d.y, this.physical, true);
      }
      this.draw();
    } else if (ptr.mode === 'click' && !ptr.moved) {
      const f = screenToFrame(p[0], p[1], this.view, this.w, this.h);
      if (f) this.options.onClick?.(f[0], f[1], this.physical);
    }
    this.updateCursor(p);
  }

  onWheel(event) {
    event.preventDefault();
    if (!this.img) return;
    let dy = event.deltaY;
    if (event.deltaMode === 1) dy *= 33;
    if (event.deltaMode === 2) dy *= 400;
    this.setPhysicalScale(this.physical * WHEEL_STEP ** (-dy / 100), this.local(event));
  }

  updateCursor(p) {
    let cursor = this.options.readOnly ? 'pointer' : (this.options.cursor?.() || 'grab');
    if (this.pointer?.mode === 'pan' && this.pointer.moved) cursor = 'grabbing';
    else if (this.pointer?.mode === 'drag' || (!this.pointer && this.hit(p))) cursor = 'move';
    else if (this.spaceDown) cursor = 'grab';
    this.canvas.style.cursor = cursor;
  }

  // --- рисование ----------------------------------------------------------------

  draw() {
    const c = this.ctx;
    const d = this.dpr;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.img) return;
    const s = this.view.cssScale * d;
    c.setTransform(s, 0, 0, s, this.view.ox * d, this.view.oy * d);
    c.imageSmoothingEnabled = s < 1;
    c.imageSmoothingQuality = 'high';
    c.drawImage(this.img, 0, 0);
    const mask = this.layers.mask?.img;
    if (mask) {
      c.imageSmoothingEnabled = true;            // маска на сетке 64×64 — без ступенек
      c.drawImage(mask, 0, 0, this.w, this.h);
    }
    const hatch = this.layers.hatch?.img;
    if (hatch) {
      c.imageSmoothingEnabled = false;
      c.drawImage(hatch, 0, 0, this.w, this.h);
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
    for (const overlay of this.overlays) this.drawOverlay(overlay);
    if (this.grabEnabled && this.draggable.length) this.drawHandles();
  }

  moved(key, point) {
    const d = this.drag || this.pending;
    return d && key && d.key === key ? [d.x, d.y] : point;
  }

  drawOverlay(o) {
    const c = this.ctx;
    const d = this.dpr;
    const pts = o.points.map((p, i) => {
      const [x, y] = this.moved(o.keys?.[i], p);
      const [sx, sy] = frameToScreen(x, y, this.view);
      return [sx * d, sy * d];
    });
    if (!pts.length) return;
    const lit = this.highlight && this.highlight === o.key;
    c.lineJoin = 'round';
    c.lineCap = 'round';
    if (pts.length >= 2) {
      c.beginPath();
      c.moveTo(pts[0][0], pts[0][1]);
      for (const [x, y] of pts.slice(1)) c.lineTo(x, y);
      if (o.closed && pts.length >= 3) c.closePath();
      if (lit) {
        c.strokeStyle = 'rgba(255,255,255,0.85)';
        c.lineWidth = 6 * d;
        c.stroke();
      }
      c.strokeStyle = o.color;
      c.lineWidth = (lit ? 3 : 2) * d;
      c.stroke();
    }
    c.fillStyle = o.color;
    for (const [x, y] of pts) {
      c.beginPath();
      c.arc(x, y, 4 * d, 0, Math.PI * 2);
      c.fill();
    }
  }

  drawHandles() {
    const c = this.ctx;
    const d = this.dpr;
    c.lineWidth = 1.5 * d;
    c.strokeStyle = 'rgba(255,255,255,0.95)';
    for (const p of this.draggable) {
      const [x, y] = this.moved(p.key, [p.x, p.y]);
      const [sx, sy] = frameToScreen(x, y, this.view);
      c.beginPath();
      c.arc(sx * d, sy * d, 6 * d, 0, Math.PI * 2);
      c.stroke();
    }
  }
}
