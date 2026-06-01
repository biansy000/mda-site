// Interactive baseline (unimodal / DA3-GIANT) 3D point-cloud viewer for the
// "Why Flying Points Happen" section. Click any red depth edge in the input
// image; the corresponding window of pixels lights up in orange in the 3D
// viewer so the flying points stranded between surfaces become visible.
//
// Data and the snap algorithm come from web/unimodal_webpage_interactive/.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BASE = 'unimodal_webpage_interactive/';
const POINT_SIZE = 2.6;
const HIGHLIGHT_RADIUS = 18;  // window half-size, in image pixels (3D highlight)
const ZOOM_RADIUS      = 8;   // window half-size, in image pixels (2D zoom).
                              // Smaller = stronger zoom-in. 8 → 17×17 px.

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function fetchBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.arrayBuffer();
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = () => reject(new Error(`image load failed: ${url}`));
    img.src = url;
  });
}

async function init(root) {
  const inputImg = root.querySelector('.uv-image');
  const marker   = root.querySelector('.uv-marker');
  const canvas   = root.querySelector('.uv-canvas');
  const snapEl   = root.querySelector('.uv-snap');
  const zoomRgb   = root.querySelector('.uv-zoom-rgb');
  const zoomDepth = root.querySelector('.uv-zoom-depth');
  if (!inputImg || !canvas) return;
  const zoomRgbCtx   = zoomRgb   ? zoomRgb.getContext('2d')   : null;
  const zoomDepthCtx = zoomDepth ? zoomDepth.getContext('2d') : null;

  let manifest;
  try { manifest = await fetchJSON(BASE + 'manifest.json'); }
  catch (e) { console.error('unimodal-viewer manifest:', e); return; }
  const s = manifest.scene;
  const { H, W } = s;

  // ------ Three.js scene ----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xffffff, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 5000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed   = 0.32;   // tuned down to match sky viewer
  controls.zoomSpeed     = 0.7;
  controls.panSpeed      = 0.6;

  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({
    size: POINT_SIZE, vertexColors: true, sizeAttenuation: false,
  });
  scene.add(new THREE.Points(geometry, material));

  // Orange overlay points (clicked window) drawn on top of the base cloud.
  const hiGeo = new THREE.BufferGeometry();
  const hiMat = new THREE.PointsMaterial({
    size: POINT_SIZE * 1.8 + 1.8,
    color: 0xff7a3a,
    sizeAttenuation: false,
    transparent: true, opacity: 0.95,
    depthTest: true, depthWrite: false,
  });
  const hiObj = new THREE.Points(hiGeo, hiMat);
  hiObj.visible = false;
  hiObj.renderOrder = 2;
  scene.add(hiObj);

  // Snap marker — a small sphere at the clicked-and-snapped 3D location.
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(1, 20, 14),
    new THREE.MeshBasicMaterial({ color: 0xff7a3a, depthTest: false }),
  );
  dot.visible = false;
  dot.renderOrder = 3;
  scene.add(dot);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(canvas);

  function tick() {
    requestAnimationFrame(tick);
    controls.update();
    renderer.render(scene, camera);
  }
  tick();

  // ------ Load point cloud + nearest-edge lookup + zoom images -------------
  const [posBuf, nearBuf, rgbImg, depthImg] = await Promise.all([
    fetchBuffer(BASE + s.bin),
    fetchBuffer(BASE + s.nearest_map),
    zoomRgbCtx   ? loadImage(BASE + s.input_image).catch(() => null) : null,
    zoomDepthCtx ? loadImage(BASE + s.depth_image).catch(() => null) : null,
  ]);

  // Resize the 2D zoom canvases to their CSS box (HiDPI not needed — they
  // already render at logical resolution; the crop is just upscaled pixels).
  // The zoom panels are square (aspect-ratio: 1/1) so width == height.
  function syncZoomCanvases() {
    for (const c of [zoomRgb, zoomDepth]) {
      if (!c) continue;
      const r = c.getBoundingClientRect();
      const px = Math.max(1, Math.floor(r.width));
      if (c.width !== px) { c.width = px; c.height = px; }
    }
  }
  if (zoomRgb)   { syncZoomCanvases(); new ResizeObserver(syncZoomCanvases).observe(zoomRgb); }
  if (zoomDepth) { new ResizeObserver(syncZoomCanvases).observe(zoomDepth); }

  // Redraw the RGB + depth zoom crops centered on the snapped pixel.
  function updateZooms(x_pix, y_pix) {
    const R = ZOOM_RADIUS;
    const srcX = Math.max(0, x_pix - R);
    const srcY = Math.max(0, y_pix - R);
    const srcW = Math.min(W - srcX, 2 * R + 1);
    const srcH = Math.min(H - srcY, 2 * R + 1);
    const pairs = [[zoomRgb, zoomRgbCtx, rgbImg], [zoomDepth, zoomDepthCtx, depthImg]];
    for (const [c, ctx, img] of pairs) {
      if (!c || !ctx) continue;
      const cw = c.width, ch = c.height;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, cw, ch);
      if (!img) continue;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, srcX, srcY, srcW, srcH, 0, 0, cw, ch);
      // Crosshair at the snapped pixel's screen position within the crop.
      const cx = ((x_pix - srcX) / srcW) * cw;
      const cy = ((y_pix - srcY) / srcH) * ch;
      ctx.strokeStyle = 'rgba(255, 122, 58, 0.95)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 9, cy); ctx.lineTo(cx + 9, cy);
      ctx.moveTo(cx, cy - 9); ctx.lineTo(cx, cy + 9);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(255, 122, 58, 0.55)';
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // Point bin layout: [u32 N][float32 positions * N*3][u8 colors * N*3]
  const dv = new DataView(posBuf);
  const N = dv.getUint32(0, true);
  const positions = new Float32Array(posBuf, 4, N * 3);
  const colors    = new Uint8Array(posBuf, 4 + N * 3 * 4, N * 3);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(colors, 3, true));
  geometry.computeBoundingSphere();

  // Scale the snap-marker sphere to the cloud's bounding radius so it reads
  // at the right physical size regardless of scene scale.
  if (geometry.boundingSphere) {
    const r = geometry.boundingSphere.radius;
    dot.scale.setScalar(Math.max(r * 0.012, 0.003));
  }

  // Nearest-edge map: channel-first uint16 — first H*W are nY, then H*W nX.
  const planeBytes = H * W * 2;
  const nearestY = new Uint16Array(nearBuf, 0, H * W);
  const nearestX = new Uint16Array(nearBuf, planeBytes, H * W);

  // ------ Default camera (manifest-provided) -------------------------------
  // Pull the camera 25% closer to the orbit target for a tighter initial
  // framing (otherwise the cloud sits in only the center of the canvas).
  const INITIAL_ZOOM_FACTOR = 0.75;

  function applyCamera() {
    camera.position.fromArray(s.camera.position);
    camera.up      .fromArray(s.camera.up);
    controls.target.fromArray(s.camera.target);
    // If the source camera coincides with the cloud center, push back along
    // the (camera → target) direction so the user sees the scene framed.
    if (geometry.boundingSphere) {
      const c = geometry.boundingSphere.center;
      const r = geometry.boundingSphere.radius;
      const dir = new THREE.Vector3().subVectors(camera.position, c);
      if (dir.length() < r * 0.05) {
        dir.copy(camera.position).sub(controls.target);
        if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
        dir.normalize().multiplyScalar(r * 0.8);
        camera.position.copy(controls.target).add(dir);
      }
    }
    camera.position.lerp(controls.target, 1 - INITIAL_ZOOM_FACTOR);
    // Slight 5° horizontal tilt around the up axis — subtly oblique start.
    const tiltAxis = new THREE.Vector3().copy(camera.up).normalize();
    const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
    offset.applyAxisAngle(tiltAxis, THREE.MathUtils.degToRad(5));
    camera.position.copy(controls.target).add(offset);
    camera.updateProjectionMatrix();
    controls.update();
  }
  applyCamera();

  // ------ Image + click handling -------------------------------------------
  // The "display" image has the red edge overlay baked in so the user can
  // see exactly which pixels are pickable.
  inputImg.src = BASE + (s.display_image || s.input_image);

  // Compute the viewport rect of the *visible* image content inside the
  // <img>, accounting for `object-fit: contain` letterboxing.
  function imageRect() {
    const r = inputImg.getBoundingClientRect();
    const nw = inputImg.naturalWidth || W;
    const nh = inputImg.naturalHeight || H;
    if (!nw || !nh || !r.width || !r.height) {
      return { left: r.left, top: r.top, width: r.width, height: r.height };
    }
    const elemAspect = r.width / r.height;
    const imgAspect = nw / nh;
    let cw, ch;
    if (elemAspect > imgAspect) { ch = r.height; cw = ch * imgAspect; }
    else                        { cw = r.width;  ch = cw / imgAspect; }
    return {
      left: r.left + (r.width - cw) / 2,
      top:  r.top  + (r.height - ch) / 2,
      width: cw, height: ch,
    };
  }
  function pointerToPixel(clientX, clientY) {
    const r = imageRect();
    const dx = clientX - r.left, dy = clientY - r.top;
    if (dx < 0 || dy < 0 || dx > r.width || dy > r.height) return null;
    return { x: (dx / r.width) * W, y: (dy / r.height) * H, rect: r };
  }
  function snapToEdge(xi, yi) {
    const x = Math.max(0, Math.min(W - 1, Math.round(xi)));
    const y = Math.max(0, Math.min(H - 1, Math.round(yi)));
    const idx = y * W + x;
    return { x: nearestX[idx], y: nearestY[idx] };
  }
  function placeMarker(x_pix, y_pix, rect) {
    const host = inputImg.parentElement;
    const hr = host.getBoundingClientRect();
    marker.style.left = (rect.left + (x_pix / W) * rect.width  - hr.left) + 'px';
    marker.style.top  = (rect.top  + (y_pix / H) * rect.height - hr.top)  + 'px';
    marker.style.display = 'block';
    // Remember image-pixel coords so resize handlers can re-anchor.
    marker.dataset.x = x_pix;
    marker.dataset.y = y_pix;
  }

  // Re-render the chosen pixel window in solid orange on top of the cloud,
  // plus the central snap-dot sphere at the snapped pixel's 3D location.
  function updateHighlight(x_pix, y_pix) {
    const R = HIGHLIGHT_RADIUS;
    const x0 = Math.max(0, x_pix - R), x1 = Math.min(W - 1, x_pix + R);
    const y0 = Math.max(0, y_pix - R), y1 = Math.min(H - 1, y_pix + R);
    const winW = x1 - x0 + 1, winH = y1 - y0 + 1;
    const subPos = new Float32Array(winW * winH * 3);
    let w = 0;
    for (let y = y0; y <= y1; y++) {
      const row = y * W;
      for (let x = x0; x <= x1; x++) {
        const idx = (row + x) * 3;
        subPos[w * 3 + 0] = positions[idx + 0];
        subPos[w * 3 + 1] = positions[idx + 1];
        subPos[w * 3 + 2] = positions[idx + 2];
        w++;
      }
    }
    hiGeo.setAttribute('position', new THREE.BufferAttribute(subPos, 3));
    hiGeo.computeBoundingSphere();
    hiObj.visible = true;

    const i = (y_pix * W + x_pix) * 3;
    dot.position.set(positions[i + 0], positions[i + 1], positions[i + 2]);
    dot.visible = true;
  }

  function snapAt(clientX, clientY) {
    const p = pointerToPixel(clientX, clientY);
    if (!p) return;
    const snap = snapToEdge(p.x, p.y);
    placeMarker(snap.x, snap.y, p.rect);
    updateHighlight(snap.x, snap.y);
    updateZooms(snap.x, snap.y);
    if (snapEl) snapEl.textContent = `(${snap.x}, ${snap.y}) — flying points lit in orange`;
  }
  const imgHost = inputImg.parentElement;
  imgHost.addEventListener('click', e => snapAt(e.clientX, e.clientY));
  imgHost.addEventListener('touchend', e => {
    if (!e.changedTouches.length) return;
    const t = e.changedTouches[0];
    snapAt(t.clientX, t.clientY);
    e.preventDefault();
  }, { passive: false });

  // On first load, snap to a likely-interesting edge near the center so the
  // viewer arrives with the artifact already visible instead of empty.
  function snapDefault() {
    const r = imageRect();
    if (!r.width || !r.height) return;
    const snap = snapToEdge(W * 0.5, H * 0.45);
    placeMarker(snap.x, snap.y, r);
    updateHighlight(snap.x, snap.y);
    updateZooms(snap.x, snap.y);
    if (snapEl) snapEl.textContent = `(${snap.x}, ${snap.y}) — flying points lit in orange`;
  }
  if (inputImg.complete && inputImg.naturalWidth) snapDefault();
  else inputImg.addEventListener('load', snapDefault, { once: true });

  // After layout reflows (viewport resize, font load), re-anchor the marker
  // to its last-known image-pixel coordinates stored on `marker.dataset`.
  window.addEventListener('resize', () => {
    if (marker.style.display !== 'block') return;
    const xPix = +marker.dataset.x, yPix = +marker.dataset.y;
    if (!isNaN(xPix) && !isNaN(yPix)) placeMarker(xPix, yPix, imageRect());
  });
}

(async () => {
  const root = document.getElementById('unimodalViewer');
  if (!root) return;

  // Lazy-init when the section scrolls near the viewport — saves the Three.js
  // boot cost for visitors who never reach "Why Flying Points Happen".
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries, obs) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          obs.disconnect();
          init(root).catch(e => console.error('unimodal-viewer init:', e));
          break;
        }
      }
    }, { rootMargin: '400px 0px' });
    io.observe(root);
  } else {
    init(root).catch(e => console.error('unimodal-viewer init:', e));
  }
})();
