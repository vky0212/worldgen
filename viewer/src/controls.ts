/**
 * Camera handling: map/orbit mode (three's MapControls) and a fly mode
 * (WASD + Q/E, Shift = faster, mouse-drag look). Also framing, animated
 * fly-to and dynamic near/far planes. The far plane follows the view range
 * (altitude above the ground), not the world's size, so the streamed area
 * stays bounded on large worlds.
 */
import { Euler, MathUtils, PerspectiveCamera, Vector3 } from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';
import { enuToThree } from './geo';
import type { LocalBounds } from './types';

export type CameraMode = 'orbit' | 'fly';

/** Returns the world-space point hit by a ray (for re-targeting orbit mode). */
export type GroundProbe = (origin: Vector3, direction: Vector3) => Vector3 | null;

/** Approximate ground elevation (ENU z) at local (x, y), or null when unknown. */
export type GroundLevel = (x: number, y: number) => number | null;

/** View range (m) at ground level and its upper limit. */
export const MIN_VIEW_RANGE = 6000;
export const MAX_VIEW_RANGE = 60000;
/** Extra view range per metre of altitude above the ground. */
const RANGE_PER_ALTITUDE = 8;

/** How far the camera sees from `altitude` metres above the ground. */
export function viewRangeAt(altitude: number): number {
  return MathUtils.clamp(MIN_VIEW_RANGE + Math.max(0, altitude) * RANGE_PER_ALTITUDE, MIN_VIEW_RANGE, MAX_VIEW_RANGE);
}

interface Tween {
  fromPos: Vector3;
  toPos: Vector3;
  fromTarget: Vector3;
  toTarget: Vector3;
  t: number;
  duration: number;
}

const MOVE_KEYS = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight']);

export class CameraRig {
  readonly camera: PerspectiveCamera;
  readonly map: MapControls;
  mode: CameraMode = 'orbit';
  onModeChange: ((mode: CameraMode) => void) | null = null;
  probe: GroundProbe | null = null;
  /** Ground under the camera (fly-mode view distance, view range). */
  groundLevel: GroundLevel | null = null;
  /** When true the camera is driven from outside (benchmark): input and tweens are ignored. */
  external = false;

  private readonly dom: HTMLElement;
  private bounds: LocalBounds = { xmin: -500, ymin: -500, xmax: 500, ymax: 500, zmin: 0, zmax: 50 };
  private extent = 1000;
  private readonly keys = new Set<string>();
  private yaw = 0;
  private pitch = 0;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private flySpeed = 40;
  private tween: Tween | null = null;
  private readonly flyTarget = new Vector3();

  constructor(camera: PerspectiveCamera, dom: HTMLElement) {
    this.camera = camera;
    this.dom = dom;
    this.map = new MapControls(camera, dom);
    this.map.enableDamping = true;
    this.map.dampingFactor = 0.12;
    this.map.screenSpacePanning = false;
    this.map.maxPolarAngle = Math.PI * 0.495;
    this.map.minDistance = 2;
    this.map.zoomToCursor = true;

    dom.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    dom.addEventListener('wheel', this.onWheel, { passive: false });
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', () => this.keys.clear());
  }

  setBounds(b: LocalBounds): void {
    this.bounds = b;
    this.extent = Math.max(100, b.xmax - b.xmin, b.ymax - b.ymin);
    this.map.maxDistance = this.extent * 4;
    this.flySpeed = MathUtils.clamp(this.extent / 40, 10, 200);
  }

  /** Frames the whole world from the south, looking north and down. */
  frame(b: LocalBounds = this.bounds): void {
    const cx = (b.xmin + b.xmax) / 2;
    const cy = (b.ymin + b.ymax) / 2;
    const cz = ((b.zmin ?? 0) + (b.zmax ?? b.zmin ?? 0)) / 2;
    const extent = Math.max(100, b.xmax - b.xmin, b.ymax - b.ymin);
    const target = enuToThree(cx, cy, b.zmin ?? cz);
    // Fit the extent into the narrower of the two field-of-view angles.
    const vfov = MathUtils.degToRad(this.camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const d = ((extent / 2) / Math.tan(Math.min(vfov, hfov) / 2)) * 1.1;
    const pos = target.clone().add(new Vector3(0, d * 0.76, d * 0.65));
    this.animateTo(pos, target, 0);
  }

  /**
   * Moves the camera to look at `target` (three coords) from `distance`,
   * keeping the current heading and using a ~50° downward pitch.
   */
  flyTo(target: Vector3, distance = 150, duration = 0.9): void {
    const dir = new Vector3();
    this.camera.getWorldDirection(dir);
    dir.y = 0;
    if (dir.lengthSq() < 1e-6) dir.set(0, 0, -1);
    dir.normalize();
    const pitch = MathUtils.degToRad(50);
    const offset = dir.multiplyScalar(-Math.cos(pitch) * distance).add(new Vector3(0, Math.sin(pitch) * distance, 0));
    this.animateTo(target.clone().add(offset), target, duration);
  }

  setMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    this.tween = null;
    if (mode === 'fly') {
      this.map.enabled = false;
      const e = new Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
      this.yaw = e.y;
      this.pitch = e.x;
    } else {
      // Re-target the orbit controls on what the camera is looking at.
      const dir = new Vector3();
      this.camera.getWorldDirection(dir);
      const hit = this.probe?.(this.camera.position.clone(), dir.clone()) ?? null;
      const target = hit ?? this.planeHit(dir) ?? this.camera.position.clone().addScaledVector(dir, 200);
      this.map.target.copy(target);
      this.map.enabled = true;
      this.map.update();
    }
    this.mode = mode;
    this.keys.clear();
    this.onModeChange?.(mode);
  }

  toggleMode(): void {
    this.setMode(this.mode === 'orbit' ? 'fly' : 'orbit');
  }

  /** Point the camera orbits around (orbit) or a point ahead of it (fly). */
  get focus(): Vector3 {
    if (this.mode === 'orbit') return this.map.target;
    const dir = new Vector3();
    this.camera.getWorldDirection(dir);
    return this.flyTarget.copy(this.camera.position).addScaledVector(dir, 100);
  }

  /** Approximate distance from the camera to what it looks at. */
  viewDistance(): number {
    if (this.mode === 'orbit' && !this.external) return this.camera.position.distanceTo(this.map.target);
    return Math.max(5, this.altitude());
  }

  /** Height (m) of the camera above the ground below it. */
  altitude(): number {
    const p = this.camera.position;
    const ground = this.groundLevel?.(p.x, -p.z) ?? this.bounds.zmin ?? 0;
    return p.y - ground;
  }

  /** How far the camera sees (m): the far plane, fog and streamed area follow it. */
  viewRange(): number {
    return viewRangeAt(this.altitude());
  }

  update(dt: number): void {
    if (this.external) {
      this.updateClipping();
      return;
    }
    if (this.tween) {
      const tw = this.tween;
      tw.t = tw.duration > 0 ? Math.min(1, tw.t + dt / tw.duration) : 1;
      const k = tw.t < 0.5 ? 4 * tw.t ** 3 : 1 - (-2 * tw.t + 2) ** 3 / 2;
      this.camera.position.lerpVectors(tw.fromPos, tw.toPos, k);
      const target = new Vector3().lerpVectors(tw.fromTarget, tw.toTarget, k);
      if (this.mode === 'orbit') {
        this.map.target.copy(target);
      } else {
        this.camera.lookAt(target);
        const e = new Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
        this.yaw = e.y;
        this.pitch = e.x;
      }
      if (tw.t >= 1) this.tween = null;
    }

    if (this.mode === 'orbit') {
      this.map.update(dt);
    } else {
      this.updateFly(dt);
    }
    this.updateClipping();
  }

  dispose(): void {
    this.map.dispose();
    this.dom.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    this.dom.removeEventListener('wheel', this.onWheel);
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
  }

  private animateTo(pos: Vector3, target: Vector3, duration: number): void {
    const fromTarget = this.mode === 'orbit' ? this.map.target.clone() : this.focus.clone();
    this.tween = { fromPos: this.camera.position.clone(), toPos: pos, fromTarget, toTarget: target.clone(), t: 0, duration };
    if (duration <= 0) this.update(0);
  }

  private planeHit(dir: Vector3): Vector3 | null {
    const groundY = this.bounds.zmin ?? 0;
    if (dir.y >= -1e-3) return null;
    const t = (groundY - this.camera.position.y) / dir.y;
    if (t <= 0 || t > this.extent * 4) return null;
    return this.camera.position.clone().addScaledVector(dir, t);
  }

  private updateFly(dt: number): void {
    this.camera.quaternion.setFromEuler(new Euler(this.pitch, this.yaw, 0, 'YXZ'));
    const k = this.keys;
    const forward = new Vector3();
    this.camera.getWorldDirection(forward);
    const right = new Vector3().crossVectors(forward, this.camera.up).normalize();
    const move = new Vector3();
    if (k.has('KeyW') || k.has('ArrowUp')) move.add(forward);
    if (k.has('KeyS') || k.has('ArrowDown')) move.sub(forward);
    if (k.has('KeyD') || k.has('ArrowRight')) move.add(right);
    if (k.has('KeyA') || k.has('ArrowLeft')) move.sub(right);
    if (k.has('KeyE')) move.y += 1;
    if (k.has('KeyQ')) move.y -= 1;
    if (move.lengthSq() === 0) return;
    const fast = k.has('ShiftLeft') || k.has('ShiftRight');
    move.normalize().multiplyScalar(this.flySpeed * (fast ? 5 : 1) * dt);
    this.camera.position.add(move);
  }

  /** Keeps depth precision usable from street level to the full overview. */
  private updateClipping(): void {
    const d = this.viewDistance();
    const near = MathUtils.clamp(d * 0.002, 0.1, 20);
    // What the camera looks at always stays in front of the far plane.
    const far = Math.max(this.viewRange(), d * 1.5);
    if (Math.abs(near - this.camera.near) / this.camera.near > 0.1 || Math.abs(far - this.camera.far) / this.camera.far > 0.1) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  private readonly onPointerDown = (ev: PointerEvent): void => {
    if (this.mode !== 'fly') return;
    if (ev.button !== 0 && ev.button !== 2) return;
    this.dragging = true;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
  };

  private readonly onPointerMove = (ev: PointerEvent): void => {
    if (!this.dragging || this.mode !== 'fly') return;
    const dx = ev.clientX - this.lastX;
    const dy = ev.clientY - this.lastY;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
    this.yaw -= dx * 0.0035;
    this.pitch = MathUtils.clamp(this.pitch - dy * 0.0035, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
    this.tween = null;
  };

  private readonly onPointerUp = (): void => {
    this.dragging = false;
  };

  private readonly onWheel = (ev: WheelEvent): void => {
    if (this.mode !== 'fly') return;
    ev.preventDefault();
    this.flySpeed = MathUtils.clamp(this.flySpeed * (ev.deltaY < 0 ? 1.2 : 1 / 1.2), 1, 2000);
  };

  private readonly onKeyDown = (ev: KeyboardEvent): void => {
    if (isTyping(ev)) return;
    if (this.mode === 'fly' && MOVE_KEYS.has(ev.code)) {
      this.keys.add(ev.code);
      this.tween = null;
      if (ev.code.startsWith('Arrow')) ev.preventDefault();
    }
  };

  private readonly onKeyUp = (ev: KeyboardEvent): void => {
    this.keys.delete(ev.code);
  };

  get speed(): number {
    return this.flySpeed;
  }
}

/** True when a key event targets a text input (shortcuts must not fire). */
export function isTyping(ev: KeyboardEvent): boolean {
  const t = ev.target as HTMLElement | null;
  if (!t) return false;
  const tag = t.tagName;
  if (tag === 'INPUT') {
    const type = (t as HTMLInputElement).type;
    return !['checkbox', 'radio', 'button', 'submit', 'range', 'color'].includes(type);
  }
  return tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
}
