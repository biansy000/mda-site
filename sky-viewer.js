// Sky section 3D point-cloud viewer (in-page, per-cell).
//
// Each `.sky-row` carries `data-scene="sky_N"`; inside it, two `.sky-cell`
// elements with `data-variant="baseline" | "wsky"` host independent Three.js
// canvases sharing one camera state per row — so the user can drag either
// view to rotate both for a fair side-by-side comparison.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MANIFEST_URL = 'sky_point_rendering/manifest.json';
const ASSET_BASE   = 'sky_point_rendering/';
const POINT_SIZE   = 1.4;

const sceneDataCache = new Map();

async function loadManifest() {
  const r = await fetch(MANIFEST_URL);
  if (!r.ok) throw new Error('manifest fetch failed: ' + r.status);
  return r.json();
}

async function loadSceneBin(sceneEntry) {
  if (sceneDataCache.has(sceneEntry.id)) return sceneDataCache.get(sceneEntry.id);
  const r = await fetch(ASSET_BASE + sceneEntry.bin);
  if (!r.ok) throw new Error('bin fetch failed: ' + r.status);
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  const N = dv.getUint32(0, true);
  const posBytes = N * 3 * 4;
  // Layout: [u32 N][float32 posBase * N*3][float32 posWsky * N*3][u8 colors * N*3]
  const posBase  = new Float32Array(buf, 4, N * 3);
  const posWsky  = new Float32Array(buf, 4 + posBytes, N * 3);
  const colorsU8 = new Uint8Array (buf, 4 + posBytes * 2, N * 3);
  const data = { N, posBase, posWsky, colorsU8 };
  sceneDataCache.set(sceneEntry.id, data);
  return data;
}

function buildView(cellEl) {
  const canvas = cellEl.querySelector('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xffffff, 1);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 5000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.rotateSpeed   = 0.32;   // tuned down — orbit felt too whippy at 0.85
  controls.zoomSpeed     = 0.7;
  controls.panSpeed      = 0.6;

  const geometry = new THREE.BufferGeometry();
  const material = new THREE.PointsMaterial({
    size: POINT_SIZE,
    vertexColors: true,
    sizeAttenuation: false,
  });
  scene.add(new THREE.Points(geometry, material));

  function resize() {
    const w = cellEl.clientWidth, h = cellEl.clientHeight;
    if (w === 0 || h === 0) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(cellEl);

  return { renderer, scene, camera, controls, geometry, material };
}

function loadPositions(view, posArr, colorsU8) {
  view.geometry.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  view.geometry.setAttribute('color',    new THREE.BufferAttribute(colorsU8, 3, true));
  view.geometry.computeBoundingSphere();
}

// Initial framing tightness — fraction of the source camera→target distance
// the viewer's camera should sit at. <1 pulls the camera closer to the
// target, giving a tighter starting view that fills more of the canvas.
const INITIAL_ZOOM_FACTOR = 0.75;

// Apply the per-scene default camera. If the source camera coincides with the
// cloud center (which happens because unprojection is anchored at the camera),
// back off along the view direction so the user sees the scene with stand-off.
function applyCamera(view, camDesc) {
  view.camera.position.fromArray(camDesc.position);
  view.camera.up      .fromArray(camDesc.up);
  view.controls.target.fromArray(camDesc.target);

  const bs = view.geometry.boundingSphere;
  if (bs) {
    const dir = new THREE.Vector3().subVectors(view.camera.position, bs.center);
    if (dir.length() < bs.radius * 0.05) {
      dir.copy(view.camera.position).sub(view.controls.target);
      if (dir.lengthSq() < 1e-6) dir.set(0, 0, 1);
      dir.normalize().multiplyScalar(bs.radius * 0.8);
      view.camera.position.copy(view.controls.target).add(dir);
    }
  }
  // Pull camera toward target for a tighter initial framing.
  view.camera.position.lerp(view.controls.target, 1 - INITIAL_ZOOM_FACTOR);
  // Slight 5° horizontal tilt around the up axis — gives a subtly oblique
  // starting view instead of staring at the scene head-on.
  const tiltAxis = new THREE.Vector3().copy(view.camera.up).normalize();
  const offset = new THREE.Vector3().subVectors(view.camera.position, view.controls.target);
  offset.applyAxisAngle(tiltAxis, THREE.MathUtils.degToRad(5));
  view.camera.position.copy(view.controls.target).add(offset);
  view.camera.updateProjectionMatrix();
  view.controls.update();
}

// Mirror camera + target from source view to target view, guarded against the
// feedback loop that would otherwise occur when both views listen to each
// other's `change` events (which fire continuously during damping).
function syncCamera(source, target) {
  if (source._syncing || target._syncing) return;
  source._syncing = true;
  target._syncing = true;
  target.camera.position.copy(source.camera.position);
  target.camera.up      .copy(source.camera.up);
  target.controls.target.copy(source.controls.target);
  target.controls.update();
  source._syncing = false;
  target._syncing = false;
}

async function initRow(rowEl, manifest) {
  const sceneId = rowEl.dataset.scene;
  const sceneEntry = manifest.scenes.find(s => s.id === sceneId);
  if (!sceneEntry) { console.warn('sky-viewer: scene not found:', sceneId); return; }

  const baseCell = rowEl.querySelector('.sky-cell[data-variant="baseline"]');
  const oursCell = rowEl.querySelector('.sky-cell[data-variant="wsky"]');
  if (!baseCell || !oursCell) return;

  const baseView = buildView(baseCell);
  const oursView = buildView(oursCell);

  // Two-way sync: drag either canvas to rotate/zoom/pan both.
  baseView.controls.addEventListener('change', () => syncCamera(baseView, oursView));
  oursView.controls.addEventListener('change', () => syncCamera(oursView, baseView));

  const data = await loadSceneBin(sceneEntry);
  loadPositions(baseView, data.posBase, data.colorsU8);
  loadPositions(oursView, data.posWsky, data.colorsU8);
  applyCamera(baseView, sceneEntry.camera);
  applyCamera(oursView, sceneEntry.camera);

  rowEl.classList.add('sky-row-ready');

  function tick() {
    requestAnimationFrame(tick);
    baseView.controls.update();
    oursView.controls.update();
    baseView.renderer.render(baseView.scene, baseView.camera);
    oursView.renderer.render(oursView.scene, oursView.camera);
  }
  tick();
}

(async () => {
  let manifest;
  try { manifest = await loadManifest(); }
  catch (e) { console.error('sky-viewer manifest load:', e); return; }

  const rows = Array.from(document.querySelectorAll('.sky-row'));
  if (rows.length === 0) return;

  // Lazy-init per row: only build Three.js scenes when the row scrolls near
  // the viewport. Keeps page load lighter when the visitor never reaches Sky.
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          io.unobserve(entry.target);
          initRow(entry.target, manifest).catch(e => console.error('sky-viewer initRow:', e));
        }
      }
    }, { rootMargin: '400px 0px' });
    rows.forEach(row => io.observe(row));
  } else {
    rows.forEach(row => initRow(row, manifest).catch(e => console.error('sky-viewer initRow:', e)));
  }
})();
