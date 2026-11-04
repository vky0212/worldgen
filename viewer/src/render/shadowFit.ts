/**
 * Fits a directional light's shadow camera to what the camera looks at
 * (instead of the whole world), so shadow detail does not depend on the
 * world's size. Used by the standard and the photo-realistic rendering.
 *
 * The frustum half-size grows with the view distance in 25 % steps and its
 * centre snaps to shadow-map texels, so zooming and panning do not shimmer
 * and the map only needs re-rendering when the fit actually changed.
 */
import { Vector3 } from 'three';
import type { DirectionalLight } from 'three';

export interface ShadowFitParams {
  /** Largest half-size (m) of the shadow frustum. */
  maxHalf: number;
  /** Shadow map resolution (texels per side). */
  mapSize: number;
  /** Vertical extent (m) of everything that may cast into the frustum. */
  zSpan: number;
  /** Smallest half-size (m). */
  minHalf?: number;
  /** Fraction of the view distance covered by the half-size. */
  coverage?: number;
}

const UP = new Vector3(0, 1, 0);
const X = new Vector3(1, 0, 0);
const STEP = 1.25;

/** Half-size for a view distance: quantised to 25 % steps above `minHalf`. */
export function shadowHalfSize(viewDistance: number, p: ShadowFitParams): number {
  const minHalf = p.minHalf ?? 40;
  const want = Math.min(p.maxHalf, Math.max(minHalf, viewDistance * (p.coverage ?? 0.9)));
  return minHalf * STEP ** Math.ceil(Math.log(want / minHalf) / Math.log(STEP));
}

export class ShadowFitter {
  /** Number of fits that required a shadow-map update. */
  updates = 0;
  private key = '';
  private readonly right = new Vector3();
  private readonly upL = new Vector3();
  private readonly center = new Vector3();

  /** Forces the next fit to update (e.g. shadow casters changed). */
  invalidate(): void {
    this.key = '';
  }

  /**
   * Points `light` (towards `sunDir`, normalised, pointing at the sun) at
   * `focus`. Returns true when the shadow camera moved and the map must be
   * re-rendered.
   */
  fit(light: DirectionalLight, sunDir: Vector3, focus: Vector3, viewDistance: number, p: ShadowFitParams): boolean {
    const half = shadowHalfSize(viewDistance, p);
    const texel = (2 * half) / p.mapSize;
    const L = sunDir;
    const right = this.right.crossVectors(L, Math.abs(L.y) > 0.99 ? X : UP).normalize();
    const upL = this.upL.crossVectors(right, L).normalize();
    // Snap the centre to shadow-map texels (no shimmering while panning).
    const snap = (v: number): number => Math.round(v / (texel * 8)) * texel * 8;
    const cr = snap(focus.dot(right));
    const cu = snap(focus.dot(upL));
    const cl = focus.dot(L);
    const key = `${cr.toFixed(2)}|${cu.toFixed(2)}|${half}|${L.x.toFixed(4)},${L.y.toFixed(4)},${L.z.toFixed(4)}|${p.mapSize}|${p.zSpan.toFixed(0)}`;
    if (key === this.key) return false;
    this.key = key;
    const center = this.center.set(0, 0, 0).addScaledVector(right, cr).addScaledVector(upL, cu).addScaledVector(L, cl);
    const zspan = Math.max(200, p.zSpan + 200);
    const dist = half * 1.5 + zspan;
    light.target.position.copy(center);
    light.target.updateMatrixWorld();
    light.position.copy(center).addScaledVector(L, dist);
    light.updateMatrixWorld();
    const cam = light.shadow.camera;
    cam.left = -half;
    cam.right = half;
    cam.top = half;
    cam.bottom = -half;
    cam.near = 1;
    cam.far = dist + half * 1.5 + zspan;
    cam.updateProjectionMatrix();
    light.shadow.bias = -0.0003;
    light.shadow.normalBias = texel * 1.6;
    light.shadow.needsUpdate = true;
    this.updates++;
    return true;
  }
}
