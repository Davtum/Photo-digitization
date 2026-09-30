// Единственная геометрия страницы: перевод экран ↔ кадр, гомография, σ клика.
// Проверяется против Python исполнением в Node (tests/test_web_geom.py).
//
// Соглашение о пикселях — как у окна Qt и у конвейера (OpenCV): пиксель кадра i
// на экране занимает [i, i+1), его центр в конвейере — целое i. Поэтому точка кадра
// = точка на изображении − 0.5, прижатая к [0, w−1]; нажатие вне снимка — не точка.
//
// view = {ox, oy, cssScale}: где на экране (CSS-пиксели) лежит левый верхний угол
// кадра и сколько CSS-пикселей приходится на пиксель кадра. Масштаб просмотра,
// который записывается у клика (σ клика), — ФИЗИЧЕСКИХ пикселей монитора на пиксель
// кадра: cssScale × devicePixelRatio. Так «1:1» — пиксель в пиксель монитора при
// любом масштабе Windows и зуме браузера.

export function screenToFrame(sx, sy, view, w, h) {
  const u = (sx - view.ox) / view.cssScale;
  const v = (sy - view.oy) / view.cssScale;
  if (!(u >= 0 && u < w && v >= 0 && v < h)) return null;
  return [Math.min(Math.max(u - 0.5, 0), w - 1), Math.min(Math.max(v - 0.5, 0), h - 1)];
}

export function frameToScreen(x, y, view) {
  return [view.ox + (x + 0.5) * view.cssScale, view.oy + (y + 0.5) * view.cssScale];
}

export function physicalScale(cssScale, dpr) {
  return cssScale * dpr;
}

export function cssScaleFor(physical, dpr) {
  return physical / dpr;
}

// Гомография 3×3 (построчно, как её пишет ядро) применяется к столбцу (x, y, 1).
export function applyH(H, x, y) {
  const w = H[2][0] * x + H[2][1] * y + H[2][2];
  return [(H[0][0] * x + H[0][1] * y + H[0][2]) / w, (H[1][0] * x + H[1][1] * y + H[1][2]) / w];
}

// σ клика в пикселях кадра — формула ui.zoom.sigma_image_px; параметры — с сервера.
export function sigmaImagePx(scale, sigma) {
  return Math.hypot(sigma.screen_px / scale, sigma.edge_px);
}

// Масштаб словами окна Qt и ui.texts.scale_name: «2:1», «1:2».
export function scaleName(scale) {
  return scale >= 1
    ? `${Number(scale.toPrecision(6))}:1`
    : `1:${Number((1 / scale).toPrecision(3))}`;
}
