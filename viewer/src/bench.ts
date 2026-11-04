/**
 * Fly-through benchmark (`?bench=<path.json>` or `?bench=auto`).
 *
 * Phases: warm-up (camera at the start of the path until the view has
 * loaded), flight (deterministic spline, wall-time driven), settle (camera
 * still at the end until nothing more loads). Per frame it records CPU and
 * GPU frame time, renderer statistics, tiles per state, GPU bytes, JS heap
 * and holes. Results are published on `window.worldgenBench` (live status
 * while running) and logged to the console when done.
 */
import { Vector3 } from 'three';
import type { PerspectiveCamera, WebGLRenderer } from 'three';
import { BENCH_DEFAULTS, SplinePath, summarize } from './benchPath';
import type { BenchPath, Summary } from './benchPath';
import type { CameraRig } from './controls';
import { enuToThree } from './geo';
import type { GpuTimer } from './render/gpuTimer';
import type { StreamStats, TileStreamer } from './stream/TileStreamer';

export type BenchPhase = 'warmup' | 'flying' | 'settling' | 'done';

export interface BenchSample {
  t: number;
  frameMs: number;
  gpuBytes: number;
  heapBytes: number | null;
  rendered: number;
  loaded: number;
  loading: number;
  holes: number;
}

export interface BenchResult {
  status: BenchPhase;
  path: string;
  durationS: number;
  budgetBytes: number;
  viewport: { width: number; height: number; pixelRatio: number };
  gpuTimer: boolean;
  warmup: { ms: number; settled: boolean; loaded: number } | null;
  flight: {
    frames: number;
    frameMs: Summary | null;
    cpuMs: Summary | null;
    gpuMs: Summary | null;
    traversalMs: Summary | null;
    holes: { max: number; mean: number; framesWithHoles: number };
    drawCalls: Summary | null;
    triangles: Summary | null;
  } | null;
  peak: { gpuBytes: number; heapBytes: number | null; rendered: number; loaded: number; geometries: number; textures: number };
  settle: { ms: number; settled: boolean; holes: number; holesAfter2s: number | null } | null;
  end: (StreamStats & { geometries: number; textures: number; heapBytes: number | null }) | null;
  resources: Record<string, number> | null;
  /** Tiles that could not be loaded (content id → error), at most 20. */
  failures: Record<string, string>;
  /** Timeline, one sample per 500 ms. */
  timeline: BenchSample[];
}

export interface BenchDeps {
  camera: PerspectiveCamera;
  rig: CameraRig;
  renderer: WebGLRenderer;
  streamer: TileStreamer;
  gpu: GpuTimer;
  /** Per-tile resources still alive (leak check at the end). */
  resources: () => Record<string, number>;
  onDone: (r: BenchResult) => void;
}

declare global {
  interface Window {
    worldgenBench?: BenchResult;
  }
}

function heapBytes(): number | null {
  const mem = (performance as unknown as { memory?: { usedJSHeapSize?: number } }).memory;
  return typeof mem?.usedJSHeapSize === 'number' ? mem.usedJSHeapSize : null;
}

export class FlyBench {
  phase: BenchPhase = 'warmup';
  readonly result: BenchResult;
  private readonly deps: BenchDeps;
  private readonly path: BenchPath;
  private readonly spline: SplinePath;
  private readonly durationMs: number;
  private phaseStart = -1;
  private lastNow = -1;
  private lastSampleAt = -Infinity;
  private t0 = 0;
  private readonly frameMs: number[] = [];
  private readonly cpuMs: number[] = [];
  private readonly gpuMs: number[] = [];
  private readonly travMs: number[] = [];
  private readonly calls: number[] = [];
  private readonly tris: number[] = [];
  private holesMax = 0;
  private holesSum = 0;
  private holeFrames = 0;
  private lastTraversals = 0;
  private readonly pos = new Vector3();
  private readonly look = new Vector3();

  constructor(path: BenchPath, deps: BenchDeps) {
    this.path = path;
    this.deps = deps;
    this.spline = new SplinePath(path.points, path.targets ?? null);
    this.durationMs = (path.duration_s ?? BENCH_DEFAULTS.duration_s) * 1000;
    const r = deps.renderer;
    const canvas = r.domElement;
    this.result = {
      status: 'warmup',
      path: path.name ?? 'custom',
      durationS: this.durationMs / 1000,
      budgetBytes: deps.streamer.budgetBytes,
      viewport: { width: canvas.clientWidth, height: canvas.clientHeight, pixelRatio: r.getPixelRatio() },
      gpuTimer: deps.gpu.available,
      warmup: null,
      flight: null,
      peak: { gpuBytes: 0, heapBytes: heapBytes(), rendered: 0, loaded: 0, geometries: 0, textures: 0 },
      settle: null,
      end: null,
      resources: null,
      failures: {},
      timeline: [],
    };
    window.worldgenBench = this.result;
    deps.rig.external = true;
    deps.rig.map.enabled = false;
    // Count every pass of a frame (photo mode renders several).
    r.info.autoReset = false;
  }

  /** Positions the camera for this frame (before the streamer and the render). */
  beforeFrame(now: number): void {
    if (this.phase === 'done') return;
    if (this.phaseStart < 0) {
      this.phaseStart = now;
      this.t0 = now;
    }
    let u = 0;
    if (this.phase === 'flying') u = Math.min(1, (now - this.phaseStart) / this.durationMs);
    else if (this.phase === 'settling') u = 1;
    const p = this.spline.position(u);
    const t = this.spline.target(u);
    this.pos.copy(enuToThree(p[0], p[1], p[2]));
    this.look.copy(enuToThree(t[0], t[1], t[2]));
    const cam = this.deps.camera;
    cam.position.copy(this.pos);
    cam.lookAt(this.look);
    cam.updateMatrixWorld();
    this.deps.renderer.info.reset();
  }

  /** Records the frame that was just drawn. `cpuMs` = main-thread time of the frame. */
  afterFrame(now: number, cpuMs: number): void {
    if (this.phase === 'done') return;
    const st = this.deps.streamer.stats();
    const info = this.deps.renderer.info;
    const heap = heapBytes();
    const peak = this.result.peak;
    peak.gpuBytes = Math.max(peak.gpuBytes, st.gpuBytes);
    peak.rendered = Math.max(peak.rendered, st.rendered);
    peak.loaded = Math.max(peak.loaded, st.loaded);
    peak.geometries = Math.max(peak.geometries, info.memory.geometries);
    peak.textures = Math.max(peak.textures, info.memory.textures);
    if (heap !== null) peak.heapBytes = Math.max(peak.heapBytes ?? 0, heap);
    const frameMs = this.lastNow >= 0 ? now - this.lastNow : 0;
    this.lastNow = now;
    if (now - this.lastSampleAt >= 500) {
      this.lastSampleAt = now;
      this.result.timeline.push({
        t: Math.round(now - this.t0),
        frameMs: Math.round(frameMs * 10) / 10,
        gpuBytes: st.gpuBytes,
        heapBytes: heap,
        rendered: st.rendered,
        loaded: st.loaded,
        loading: st.loading + st.queued + st.uploading,
        holes: st.holes,
      });
    }
    const elapsed = now - this.phaseStart;

    if (this.phase === 'warmup') {
      const limit = (this.path.warmup_s ?? BENCH_DEFAULTS.warmup_s) * 1000;
      if ((st.settled && st.traversals > 2) || elapsed >= limit) {
        this.result.warmup = { ms: Math.round(elapsed), settled: st.settled, loaded: st.loaded };
        this.enter('flying', now);
      }
      return;
    }

    if (this.phase === 'flying') {
      if (frameMs > 0) this.frameMs.push(frameMs);
      this.cpuMs.push(cpuMs);
      const gpu = this.deps.gpu.lastMs;
      if (gpu !== null) this.gpuMs.push(gpu);
      if (st.traversals !== this.lastTraversals) {
        this.lastTraversals = st.traversals;
        this.travMs.push(st.traversalMs);
      }
      this.calls.push(info.render.calls);
      this.tris.push(info.render.triangles);
      this.holesMax = Math.max(this.holesMax, st.holes);
      this.holesSum += st.holes;
      if (st.holes > 0) this.holeFrames++;
      if (elapsed >= this.durationMs) {
        this.result.flight = {
          frames: this.cpuMs.length,
          frameMs: summarize(this.frameMs),
          cpuMs: summarize(this.cpuMs),
          gpuMs: summarize(this.gpuMs),
          traversalMs: summarize(this.travMs),
          holes: { max: this.holesMax, mean: Math.round((this.holesSum / Math.max(1, this.cpuMs.length)) * 100) / 100, framesWithHoles: this.holeFrames },
          drawCalls: summarize(this.calls),
          triangles: summarize(this.tris),
        };
        this.enter('settling', now);
      }
      return;
    }

    // settling
    const s = this.result.settle ?? { ms: -1, settled: false, holes: -1, holesAfter2s: null };
    this.result.settle = s;
    if (s.holesAfter2s === null && elapsed >= 2000) s.holesAfter2s = st.holes;
    if (s.ms < 0 && st.settled) {
      s.ms = Math.round(elapsed);
      s.settled = true;
      s.holes = st.holes;
    }
    const limit = (this.path.settle_timeout_s ?? BENCH_DEFAULTS.settle_timeout_s) * 1000;
    if ((s.ms >= 0 && s.holesAfter2s !== null) || elapsed >= limit) {
      if (s.ms < 0) {
        s.ms = Math.round(elapsed);
        s.holes = st.holes;
      }
      if (s.holesAfter2s === null) s.holesAfter2s = st.holes;
      this.result.end = { ...st, geometries: info.memory.geometries, textures: info.memory.textures, heapBytes: heap };
      this.result.resources = this.deps.resources();
      this.result.failures = Object.fromEntries([...this.deps.streamer.failures()].slice(0, 20));
      this.enter('done', now);
      const rig = this.deps.rig;
      rig.external = false;
      rig.map.target.copy(this.look);
      rig.map.enabled = rig.mode === 'orbit';
      if (rig.mode === 'orbit') rig.map.update();
      this.deps.renderer.info.autoReset = true;
      console.info('[worldgen bench]', JSON.stringify(this.result));
      this.deps.onDone(this.result);
    }
  }

  private enter(phase: BenchPhase, now: number): void {
    this.phase = phase;
    this.result.status = phase;
    this.phaseStart = now;
  }
}
