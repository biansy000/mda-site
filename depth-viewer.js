/* Interactive depth-distribution viewer.
 *
 * Loads the cache assets produced by scripts/convert_depth_viewer_cache.py
 * and reproduces the Gradio app's UX: click an image -> snap to the nearest
 * depth-boundary pixel -> draw the per-pixel mixture pdf on a canvas, plus
 * a one-line stats string.
 *
 * Extension: when the cache also has a points.json + view_<v>.bin produced by
 * scripts/export_depth_viewer_pointclouds.py, render the per-view 3D point
 * cloud below the distribution plot, with the clicked window highlighted in
 * orange so the user can see where the per-pixel distribution lives in 3D.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

(function () {
  const root = document.getElementById('depthViewer');
  if (!root) return;

  // ---------- DOM refs ----------
  const cacheTabs   = root.querySelectorAll('.scene-thumbs[data-role="cache"] .scene-thumb');
  const viewSlider  = root.querySelector('input[data-role="view"]');
  const viewNum     = root.querySelector('.dviewer-view-num');
  const rgbImg      = root.querySelector('.dviewer-rgb');
  const overlayCv   = root.querySelector('.dviewer-overlay');
  const plotCv      = root.querySelector('.dviewer-plot');
  const pcCanvas    = root.querySelector('.dviewer-pc-canvas');
  const pcWrap      = root.querySelector('.dviewer-pc-wrap');
  const pcInfoEl    = root.querySelector('.dviewer-pc-info');

  // ---------- State ----------
  const state = {
    cacheName: cacheTabs[0]?.dataset.cache || 'scan_00183',
    cache: null,
    viewIdx: 0,
    pixel: null, // last snapped pixel: { x, y, idx }
  };

  // Cache-busting query so updated .bin / .png / .json contents are picked
  // up by the browser instead of being served from its HTTP cache. Bump on
  // every regeneration of web/assets/depth_viewer/.
  const ASSET_V = 'v=9';

  // ---------- Loader ----------
  const cacheStore = new Map();
  async function loadCache(name) {
    if (cacheStore.has(name)) return cacheStore.get(name);
    const baseUrl = `assets/depth_viewer/${name}`;
    const [meta, buf] = await Promise.all([
      fetch(`${baseUrl}/meta.json?${ASSET_V}`).then(r => r.json()),
      fetch(`${baseUrl}/data.bin?${ASSET_V}`).then(r => r.arrayBuffer()),
    ]);
    const N = meta.N;
    const views = meta.views.map((v, i) => {
      const K = v.K;
      let off = v.offset_bytes;
      const coords = new Int16Array(buf, off, 2 * K);                off += 2 * K * 2;
      const means  = new Float32Array(buf, off, N * K);              off += 4 * N * K;
      const confs  = new Float32Array(buf, off, N * K);              off += 4 * N * K;
      const logw   = new Float32Array(buf, off, N * K);              off += 4 * N * K;
      const depth  = new Float32Array(buf, off, K);                  off += 4 * K;
      return {
        K, coords, means, confs, logw, depth,
        rgbUrl:  `${baseUrl}/overlay_${i}.png?${ASSET_V}`,
        maskUrl: `${baseUrl}/mask_${i}.png?${ASSET_V}`,
      };
    });
    const cache = { meta, views };
    cacheStore.set(name, cache);
    return cache;
  }

  // ---------- Math ----------
  function softmax(logw) {
    let mx = -Infinity;
    for (let i = 0; i < logw.length; i++) if (logw[i] > mx) mx = logw[i];
    let s = 0;
    const out = new Float32Array(logw.length);
    for (let i = 0; i < logw.length; i++) { out[i] = Math.exp(logw[i] - mx); s += out[i]; }
    const inv = 1 / Math.max(s, 1e-12);
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
  }

  function buildPdf(xs, means, scales, weights, lossType) {
    const eps = 1e-8;
    const y = new Float64Array(xs.length);
    const N = means.length;
    if (lossType === 'l1') {
      for (let k = 0; k < N; k++) {
        const s = Math.max(scales[k], eps);
        const m = means[k];
        const norm = 1 / (2 * s + eps);
        const w = weights[k];
        for (let i = 0; i < xs.length; i++) y[i] += w * norm * Math.exp(-Math.abs(xs[i] - m) / s);
      }
    } else if (lossType === 'l2') {
      for (let k = 0; k < N; k++) {
        const v = Math.max(scales[k], eps);
        const m = means[k];
        const norm = 1 / (Math.sqrt(2 * Math.PI * v) + eps);
        const w = weights[k];
        for (let i = 0; i < xs.length; i++) y[i] += w * norm * Math.exp(-0.5 * ((xs[i] - m) ** 2) / v);
      }
    } else if (lossType === 'logl2') {
      for (let k = 0; k < N; k++) {
        const v = Math.max(scales[k], eps);
        const m = Math.log(Math.max(means[k], 1e-3) + 0.1);
        const norm = 1 / (Math.sqrt(2 * Math.PI * v) + eps);
        const w = weights[k];
        for (let i = 0; i < xs.length; i++) {
          const xl = Math.log(Math.max(xs[i], 1e-3) + 0.1);
          y[i] += w * norm * Math.exp(-0.5 * ((xl - m) ** 2) / v);
        }
      }
    }
    return y;
  }

  function sliceAtPixel(view, idx, alpha) {
    const N = state.cache.meta.N;
    const K = view.K;
    const means  = new Float32Array(N);
    const confs  = new Float32Array(N);
    const logw   = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      means[k] = view.means[k * K + idx];
      confs[k] = view.confs[k * K + idx];
      logw[k]  = view.logw[k * K + idx];
    }
    const scales = new Float32Array(N);
    for (let k = 0; k < N; k++) scales[k] = alpha / Math.max(confs[k], 1e-8);
    const weights = softmax(logw);
    return { means, scales, weights, depth: view.depth[idx] };
  }

  // ---------- Snap to nearest boundary pixel ----------
  function snapNearest(view, x, y) {
    if (view.K === 0) return { x, y, idx: -1 };
    const c = view.coords;
    let bestIdx = 0, bestDist = Infinity;
    for (let i = 0; i < view.K; i++) {
      const dy = c[2 * i] - y;
      const dx = c[2 * i + 1] - x;
      const d  = dy * dy + dx * dx;
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    return { x: c[2 * bestIdx + 1], y: c[2 * bestIdx], idx: bestIdx };
  }

  // ---------- Click overlay (red ring + crosshair + "(x=…, y=…)" label) ----------
  function drawClickMarker(px, py) {
    const W = state.cache.meta.W, H = state.cache.meta.H;
    overlayCv.width  = W;
    overlayCv.height = H;
    const ctx = overlayCv.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    if (px == null) return;
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgb(255, 64, 64)';
    ctx.beginPath();
    ctx.arc(px + 0.5, py + 0.5, 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = '#fff';
    ctx.beginPath();
    ctx.moveTo(px - 7 + 0.5, py + 0.5); ctx.lineTo(px + 7 + 0.5, py + 0.5);
    ctx.moveTo(px + 0.5, py - 7 + 0.5); ctx.lineTo(px + 0.5, py + 7 + 0.5);
    ctx.stroke();
    ctx.font = '14px "Google Sans", sans-serif';
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    ctx.lineWidth = 3;
    const label = `(x=${px}, y=${py})`;
    ctx.strokeText(label, 10, 22);
    ctx.fillText(label, 10, 22);
  }

  // ---------- Plot ----------
  // A single in-flight reveal animation; cancelled whenever a new plot starts.
  let plotAnim = null;
  const prefersReduced = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function drawPlot(means, scales, weights, depth, lossType, modeLabel) {
    // Determine x range as the Python script did.
    const spread = new Float32Array(scales.length);
    if (lossType === 'l2' || lossType === 'logl2') {
      for (let i = 0; i < scales.length; i++) spread[i] = Math.sqrt(Math.max(scales[i], 1e-12));
    } else {
      for (let i = 0; i < scales.length; i++) spread[i] = scales[i];
    }
    let lo =  Infinity, hi = -Infinity;
    for (let i = 0; i < means.length; i++) {
      const a = means[i] - 4 * spread[i], b = means[i] + 4 * spread[i];
      if (a < lo) lo = a;
      if (b > hi) hi = b;
    }
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) {
      lo = Math.min(...means) - 1; hi = Math.max(...means) + 1;
    }
    const M = 512;
    const xs = new Float32Array(M);
    const step = (hi - lo) / (M - 1);
    for (let i = 0; i < M; i++) xs[i] = lo + i * step;
    const ys = buildPdf(xs, means, scales, weights, lossType);

    // Draw onto plotCv. Hidpi-friendly canvas sizing.
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cssW = plotCv.clientWidth || 640;
    const cssH = plotCv.clientHeight || 360;
    plotCv.width  = Math.round(cssW * dpr);
    plotCv.height = Math.round(cssH * dpr);
    const ctx = plotCv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const padL = 56, padR = 14, padT = 32, padB = 38;
    const plotW = cssW - padL - padR;
    const plotH = cssH - padT - padB;
    let yMax = -Infinity;
    for (let i = 0; i < ys.length; i++) if (ys[i] > yMax) yMax = ys[i];
    if (!isFinite(yMax) || yMax <= 0) yMax = 1;
    yMax *= 1.08;

    const xToPx = x => padL + (x - lo) / (hi - lo) * plotW;
    const yToPx = y => padT + plotH - (y / yMax) * plotH;

    // Paint one frame. `prog` (0..1) drives the reveal: the pdf curve rises
    // from the baseline while the mean/decoded verticals fade in.
    function paint(prog) {
      ctx.clearRect(0, 0, cssW, cssH);

      // Background panel
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, cssW, cssH);

      // Grid + axes
      ctx.strokeStyle = '#e6e9ee';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let g = 1; g < 6; g++) {
        const yy = padT + (plotH / 6) * g;
        ctx.moveTo(padL, yy); ctx.lineTo(padL + plotW, yy);
      }
      for (let g = 1; g < 6; g++) {
        const xx = padL + (plotW / 6) * g;
        ctx.moveTo(xx, padT); ctx.lineTo(xx, padT + plotH);
      }
      ctx.stroke();
      ctx.strokeStyle = '#9aa4b2';
      ctx.beginPath();
      ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
      ctx.stroke();

      // Axis ticks (5 evenly spaced)
      ctx.fillStyle = '#555';
      ctx.font = '11px "Google Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      for (let g = 0; g <= 5; g++) {
        const xv = lo + (hi - lo) * (g / 5);
        const xp = padL + (plotW * g / 5);
        ctx.fillText(xv.toFixed(2), xp, padT + plotH + 6);
      }
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let g = 0; g <= 4; g++) {
        const yv = yMax * (1 - g / 4);
        const yp = padT + (plotH * g / 4);
        ctx.fillText(yv.toFixed(2), padL - 8, yp);
      }

      // Component-mean dashed verticals (fade in)
      ctx.save();
      ctx.globalAlpha = prog;
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(120, 120, 130, 0.7)';
      ctx.lineWidth = 1;
      for (let k = 0; k < means.length; k++) {
        const xp = xToPx(means[k]);
        ctx.beginPath();
        ctx.moveTo(xp, padT); ctx.lineTo(xp, padT + plotH);
        ctx.stroke();
      }
      ctx.restore();

      // pdf line — height eases up from the baseline
      ctx.strokeStyle = 'royalblue';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < M; i++) {
        const xp = xToPx(xs[i]);
        const yp = yToPx(ys[i] * prog);
        if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
      }
      ctx.stroke();

      // Decoded depth — crimson solid vertical (fade in)
      if (depth != null && isFinite(depth)) {
        const xp = xToPx(depth);
        ctx.save();
        ctx.globalAlpha = prog;
        ctx.strokeStyle = 'crimson';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xp, padT); ctx.lineTo(xp, padT + plotH);
        ctx.stroke();
        ctx.restore();
      }

      // Title + axis labels
      ctx.fillStyle = '#1f2024';
      ctx.font = '600 13px "Google Sans", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText('Depth distribution at clicked pixel', padL, 8);
      ctx.fillStyle = '#555';
      ctx.font = '11px "Google Sans", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Depth', padL + plotW / 2, cssH - 14);
      ctx.save();
      ctx.translate(14, padT + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText('Density', 0, 0);
      ctx.restore();

      // Legend
      const legY = padT + 4;
      ctx.font = '11px "Google Sans", sans-serif';
      ctx.fillStyle = '#1f2024';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = 'royalblue';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padL + plotW - 158, legY); ctx.lineTo(padL + plotW - 134, legY); ctx.stroke();
      ctx.fillText(`${modeLabel.toUpperCase()} pdf`, padL + plotW - 128, legY);
      ctx.strokeStyle = 'crimson';
      ctx.beginPath();
      ctx.moveTo(padL + plotW - 158, legY + 16); ctx.lineTo(padL + plotW - 134, legY + 16); ctx.stroke();
      ctx.fillText('Decoded depth', padL + plotW - 128, legY + 16);
    }

    // Drive the reveal. Cancel any in-flight animation so rapid clicks restart
    // cleanly. Reduced-motion users get the final frame immediately.
    if (plotAnim != null) { cancelAnimationFrame(plotAnim); plotAnim = null; }
    if (prefersReduced) { paint(1); return; }
    const DURATION_MS = 420;
    let start = null;
    function frame(t) {
      if (start == null) start = t;
      const lin = Math.min(1, (t - start) / DURATION_MS);
      const eased = 1 - Math.pow(1 - lin, 3);   // easeOutCubic
      paint(eased);
      if (lin < 1) { plotAnim = requestAnimationFrame(frame); }
      else { plotAnim = null; }
    }
    plotAnim = requestAnimationFrame(frame);
  }

  // ---------- 3D point-cloud panel ----------
  // Lazy-initialized when a cache with a `points.json` is selected; remains
  // null for caches without 3D data (the panel is hidden in that case).
  let pc = null;
  const pcMetaStore = new Map();   // cacheName -> points.json (or null on 404)
  const pcDataStore = new Map();   // `${cacheName}/${view}` -> { N, positions, colors, pixelIdx }
  // Width of the original image, kept here so we can hand it to the highlight
  // routine (which indexes pixel-pairs by orig_y * W + orig_x).
  const HIGHLIGHT_R_PX = 18;       // pixel radius around the snapped pixel

  function initPC() {
    if (pc || !pcCanvas) return;
    const renderer = new THREE.WebGLRenderer({ canvas: pcCanvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xffffff, 1);
    const scene3 = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 5000);
    const controls = new OrbitControls(camera, pcCanvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.rotateSpeed   = 0.32;
    controls.zoomSpeed     = 0.7;
    controls.panSpeed      = 0.6;

    const baseGeo = new THREE.BufferGeometry();
    const baseMat = new THREE.PointsMaterial({
      size: 2.0, vertexColors: true, sizeAttenuation: false,
    });
    scene3.add(new THREE.Points(baseGeo, baseMat));

    const hiGeo = new THREE.BufferGeometry();
    const hiMat = new THREE.PointsMaterial({
      size: 4.5, color: 0xff7a3a, sizeAttenuation: false,
      transparent: true, opacity: 0.95,
      depthTest: true, depthWrite: false,
    });
    const hiPts = new THREE.Points(hiGeo, hiMat);
    hiPts.visible = false;
    hiPts.renderOrder = 2;
    scene3.add(hiPts);

    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(1, 20, 14),
      new THREE.MeshBasicMaterial({ color: 0xff7a3a, depthTest: false }),
    );
    dot.visible = false;
    dot.renderOrder = 3;
    scene3.add(dot);

    function resize() {
      const w = pcCanvas.clientWidth, h = pcCanvas.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    resize();
    new ResizeObserver(resize).observe(pcCanvas);
    function tick() {
      requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene3, camera);
    }
    tick();

    pc = {
      renderer, scene3, camera, controls,
      baseGeo, baseMat, hiGeo, hiMat, hiPts, dot,
      // Per-view state, refreshed on every view change:
      positions: null,   // Float32Array of (N,3) in world coords
      pixelIdx:  null,   // Uint32Array of (N,) — orig_y * W + orig_x per point
      pixelToPt: null,   // Uint32Array of size W*H — point index for each
                         // original image pixel, or 0xFFFFFFFF if unmapped.
      perViewMeta: null, // points.json scene meta
      W: 0, H: 0,
    };
  }

  async function loadPCMeta(cacheName) {
    if (pcMetaStore.has(cacheName)) return pcMetaStore.get(cacheName);
    try {
      const r = await fetch(`assets/depth_viewer/${cacheName}/points.json?${ASSET_V}`);
      if (!r.ok) throw new Error(r.status);
      const j = await r.json();
      pcMetaStore.set(cacheName, j);
      return j;
    } catch {
      pcMetaStore.set(cacheName, null);
      return null;
    }
  }
  async function loadPCView(cacheName, viewIdx) {
    const key = `${cacheName}/${viewIdx}`;
    if (pcDataStore.has(key)) return pcDataStore.get(key);
    const url = `assets/depth_viewer/${cacheName}/view_${viewIdx}.bin?${ASSET_V}`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(r.status);
      const buf = await r.arrayBuffer();
      const dv = new DataView(buf);
      const N = dv.getUint32(0, true);
      // Layout: u32 N | float32 positions[N*3] | u32 pixel_idx[N] | u8 colors[N*3]
      // u32 arrays come before the u8 block so all u32 byteOffsets stay
      // 4-byte aligned (required by the Uint32Array constructor).
      const positions = new Float32Array(buf, 4,              N * 3);
      const pixelIdx  = new Uint32Array (buf, 4 + N * 12,     N);
      const colors    = new Uint8Array  (buf, 4 + N * 16,     N * 3);
      const data = { N, positions, colors, pixelIdx };
      pcDataStore.set(key, data);
      return data;
    } catch (e) {
      console.warn('[depth-viewer] point cloud missing', cacheName, viewIdx, e);
      pcDataStore.set(key, null);
      return null;
    }
  }

  // Build the (W*H) pixel→point-index lookup so the highlight loop is O(R^2).
  function rebuildPixelLookup(W, H, pixelIdx, N) {
    const tbl = new Uint32Array(W * H).fill(0xFFFFFFFF);
    for (let i = 0; i < N; i++) tbl[pixelIdx[i]] = i;
    return tbl;
  }

  // Set the highlight Points geometry from a window around (x_pix, y_pix).
  // Also drop a sphere marker at the world point corresponding to the boundary
  // pixel's deterministic decoded depth (computed via per-view intrinsics +
  // extrinsics passed in from points.json).
  function updatePCHighlight(x_pix, y_pix, decodedDepth) {
    if (!pc || !pc.positions || !pc.pixelToPt) return;
    const R = HIGHLIGHT_R_PX;
    const W = pc.W, H = pc.H;
    const x0 = Math.max(0, x_pix - R), x1 = Math.min(W - 1, x_pix + R);
    const y0 = Math.max(0, y_pix - R), y1 = Math.min(H - 1, y_pix + R);

    // First pass: count valid points so we can allocate a tight buffer.
    let count = 0;
    for (let y = y0; y <= y1; y++) {
      const row = y * W;
      for (let x = x0; x <= x1; x++) {
        if (pc.pixelToPt[row + x] !== 0xFFFFFFFF) count++;
      }
    }
    if (count === 0) { pc.hiPts.visible = false; pc.dot.visible = false; return; }

    const sub = new Float32Array(count * 3);
    let w = 0;
    for (let y = y0; y <= y1; y++) {
      const row = y * W;
      for (let x = x0; x <= x1; x++) {
        const pti = pc.pixelToPt[row + x];
        if (pti === 0xFFFFFFFF) continue;
        sub[w * 3 + 0] = pc.positions[pti * 3 + 0];
        sub[w * 3 + 1] = pc.positions[pti * 3 + 1];
        sub[w * 3 + 2] = pc.positions[pti * 3 + 2];
        w++;
      }
    }
    pc.hiGeo.setAttribute('position', new THREE.BufferAttribute(sub, 3));
    pc.hiGeo.computeBoundingSphere();
    pc.hiPts.visible = true;

    // Snap-dot at the world coords for (x_pix, y_pix, decodedDepth). Uses the
    // per-view intrinsics + extrinsics to unproject the click.
    if (decodedDepth != null && isFinite(decodedDepth)) {
      const meta = pc.perViewMeta;  // per-view dict from points.json
      const K = meta.intrinsics, RT = meta.extrinsics;
      const fx = K[0][0], fy = K[1][1], cx = K[0][2], cy = K[1][2];
      // Camera-frame point
      const xc = (x_pix - cx) / fx * decodedDepth;
      const yc = (y_pix - cy) / fy * decodedDepth;
      const zc = decodedDepth;
      // World coords: world = (cam - t) @ R, where RT = [R | t] world→cam.
      // Equivalent column-vector form: world = R^T (cam - t).
      const t0 = RT[0][3], t1 = RT[1][3], t2 = RT[2][3];
      const dx = xc - t0, dy = yc - t1, dz = zc - t2;
      // (cam - t) @ R, with R = RT[:,:3]. Multiply row by matrix.
      const wx = dx * RT[0][0] + dy * RT[1][0] + dz * RT[2][0];
      const wy = dx * RT[0][1] + dy * RT[1][1] + dz * RT[2][1];
      const wz = dx * RT[0][2] + dy * RT[1][2] + dz * RT[2][2];
      pc.dot.position.set(wx, wy, wz);
      pc.dot.visible = true;
    } else {
      pc.dot.visible = false;
    }
  }

  async function updatePCForView() {
    if (!pcCanvas || !pcWrap) return;
    const meta = await loadPCMeta(state.cacheName);
    if (!meta) {
      // Hide the panel entirely when no 3D data exists for this cache.
      pcWrap.style.display = 'none';
      return;
    }
    pcWrap.style.display = '';
    if (!pc) initPC();
    const data = await loadPCView(state.cacheName, state.viewIdx);
    if (!data) return;

    pc.positions = data.positions;
    pc.pixelIdx  = data.pixelIdx;
    pc.perViewMeta = meta.views[state.viewIdx];
    pc.W = meta.W; pc.H = meta.H;
    pc.pixelToPt = rebuildPixelLookup(meta.W, meta.H, data.pixelIdx, data.N);

    pc.baseGeo.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
    pc.baseGeo.setAttribute('color',    new THREE.BufferAttribute(data.colors, 3, true));
    pc.baseGeo.computeBoundingSphere();

    const camDesc = pc.perViewMeta.camera;
    pc.camera.position.fromArray(camDesc.position);
    pc.camera.up      .fromArray(camDesc.up);
    pc.controls.target.fromArray(camDesc.target);
    // Pull camera 25% closer to target for a tighter initial framing —
    // mirrors what sky-viewer and unimodal-viewer do (factor 0.75).
    pc.camera.position.lerp(pc.controls.target, 0.25);
    // Slight 5° horizontal tilt around the up axis — subtly oblique start.
    const tiltAxis = new THREE.Vector3().copy(pc.camera.up).normalize();
    const offset = new THREE.Vector3().subVectors(pc.camera.position, pc.controls.target);
    offset.applyAxisAngle(tiltAxis, THREE.MathUtils.degToRad(5));
    pc.camera.position.copy(pc.controls.target).add(offset);
    pc.camera.updateProjectionMatrix();
    pc.controls.update();

    // Scale the snap-dot to the cloud's bounding radius so it reads at the
    // right physical size regardless of scene scale.
    if (pc.baseGeo.boundingSphere) {
      const r = pc.baseGeo.boundingSphere.radius;
      pc.dot.scale.setScalar(Math.max(r * 0.012, 0.003));
    }
    pc.hiPts.visible = false;
    pc.dot.visible = false;
    if (pcInfoEl) pcInfoEl.textContent = `${data.N.toLocaleString()} pts`;
  }

  // ---------- Render orchestration ----------
  function setActiveTab(group, predicate) {
    group.forEach(btn => btn.classList.toggle('is-active', predicate(btn)));
  }

  function renderCurrent(forceImageReload = false) {
    const view = state.cache.views[state.viewIdx];
    if (forceImageReload) rgbImg.src = view.rgbUrl;
    if (!state.pixel || state.pixel.idx >= view.K) {
      // Default to the first boundary coord for this view.
      if (view.K > 0) {
        state.pixel = { x: view.coords[1], y: view.coords[0], idx: 0 };
      } else {
        state.pixel = { x: state.cache.meta.W >> 1, y: state.cache.meta.H >> 1, idx: -1 };
      }
    } else {
      state.pixel = snapNearest(view, state.pixel.x, state.pixel.y);
    }
    drawClickMarker(state.pixel.x, state.pixel.y);
    if (state.pixel.idx >= 0) {
      const sliced = sliceAtPixel(view, state.pixel.idx, state.cache.meta.alpha);
      drawPlot(sliced.means, sliced.scales, sliced.weights, sliced.depth,
               state.cache.meta.loss_type, 'gmm');
      // Light up the same pixel window in 3D using the decoded depth as the
      // snap-dot location. No-op when the 3D panel is hidden / not loaded.
      if (pc && pc.positions) updatePCHighlight(state.pixel.x, state.pixel.y, sliced.depth);
    } else {
      const ctx = plotCv.getContext('2d');
      ctx.clearRect(0, 0, plotCv.width, plotCv.height);
      if (pc) { pc.hiPts.visible = false; pc.dot.visible = false; }
    }
  }

  async function selectCache(name) {
    state.cacheName = name;
    state.cache = await loadCache(name);
    state.viewIdx = 0;
    state.pixel = null;
    viewSlider.max = state.cache.meta.L - 1;
    viewSlider.value = 0;
    viewNum.textContent = '0';
    setActiveTab(cacheTabs, btn => btn.dataset.cache === name);
    // Kick off the 3D-panel update in parallel; renderCurrent will re-trigger
    // the per-pixel highlight once it has access to the loaded point cloud.
    const pcReady = updatePCForView();
    renderCurrent(true);
    await pcReady;
    // After the cloud is in place, re-apply the highlight so the user lands
    // on the pre-selected pixel with the 3D context visible.
    if (pc && pc.positions && state.pixel && state.pixel.idx >= 0) {
      const v = state.cache.views[state.viewIdx];
      const sliced = sliceAtPixel(v, state.pixel.idx, state.cache.meta.alpha);
      updatePCHighlight(state.pixel.x, state.pixel.y, sliced.depth);
    }
  }

  // ---------- Click handling ----------
  function pixelFromEvent(e) {
    const r = rgbImg.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return null;
    const W = state.cache.meta.W, H = state.cache.meta.H;
    const x = Math.round(((e.clientX - r.left) / r.width)  * W);
    const y = Math.round(((e.clientY - r.top)  / r.height) * H);
    return {
      x: Math.max(0, Math.min(W - 1, x)),
      y: Math.max(0, Math.min(H - 1, y)),
    };
  }

  function handleClick(e) {
    const p = pixelFromEvent(e);
    if (!p) return;
    const view = state.cache.views[state.viewIdx];
    state.pixel = snapNearest(view, p.x, p.y);
    renderCurrent(false);
  }

  // ---------- Wire DOM ----------
  rgbImg.addEventListener('click',     handleClick);
  overlayCv.addEventListener('click',  handleClick);

  cacheTabs.forEach(btn => btn.addEventListener('click', () => selectCache(btn.dataset.cache)));
  viewSlider.addEventListener('input', async () => {
    state.viewIdx = +viewSlider.value;
    viewNum.textContent = String(state.viewIdx);
    state.pixel = null;
    // The point cloud is per-view, so swap it whenever the view changes.
    await updatePCForView();
    renderCurrent(true);
  });

  // Redraw plot when the panel resizes (mobile/desktop transitions).
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => {
      if (state.cache) renderCurrent(false);
    }).observe(plotCv);
  }

  selectCache(state.cacheName).catch(err => {
    console.error('[depth-viewer] failed to load cache', err);
  });
})();
