/**
 * Graphics levels for the photo-realistic mode and the frame-rate governor
 * that keeps the frame rate above the target (45 fps) by lowering the render
 * resolution first and shedding expensive features second.
 *
 * Pure logic (no three.js, no DOM) so it can be unit-tested under node.
 */

export type GraphicsLevel = 'low' | 'medium' | 'high' | 'ultra';
export type LevelChoice = GraphicsLevel | 'auto';

export const LEVEL_ORDER: readonly GraphicsLevel[] = ['low', 'medium', 'high', 'ultra'];

export const LEVEL_LABELS: Record<LevelChoice, string> = {
  auto: 'Auto',
  low: 'Low',
  medium: 'Medium',
  high: 'High',
  ultra: 'Ultra',
};

export interface QualitySettings {
  /** Multiplier of min(devicePixelRatio, 2). */
  renderScale: number;
  /** MSAA samples of the HDR scene target (0 = none). */
  msaa: number;
  shadows: boolean;
  shadowMapSize: number;
  /** Largest half-size (m) of the camera-fitted shadow frustum. */
  shadowMaxHalf: number;
  terrainCastsShadow: boolean;
  /** Trees closer than this (m) cast shadows. */
  treeShadowDist: number;
  /** Full trees (trunk + foliage) closer than this (m). */
  treeNear: number;
  /** Foliage only (no trunks) closer than this (m). */
  treeMid: number;
  /** Low-poly tree impostors closer than this (m); hidden beyond. */
  treeFar: number;
  ao: boolean;
  aoFullRes: boolean;
  bloom: boolean;
  /** Pixel size of the procedural texture arrays. */
  textureSize: number;
  anisotropy: number;
  /** 0 albedo only · 1 + normal maps · 2 + glass reflections · 3 + window parallax / night lights. */
  facadeDetail: 0 | 1 | 2 | 3;
  /** 0 flat · 1 + animated normals · 2 + sun glint · 3 + fine ripples. */
  waterDetail: 0 | 1 | 2 | 3;
  wind: boolean;
}

export const LEVELS: Record<GraphicsLevel, QualitySettings> = {
  low: {
    renderScale: 0.5, msaa: 0, shadows: true, shadowMapSize: 1024, shadowMaxHalf: 500, terrainCastsShadow: false,
    treeShadowDist: 0, treeNear: 150, treeMid: 350, treeFar: 1600,
    ao: false, aoFullRes: false, bloom: false, textureSize: 128, anisotropy: 1, facadeDetail: 0, waterDetail: 1, wind: false,
  },
  medium: {
    renderScale: 0.65, msaa: 2, shadows: true, shadowMapSize: 2048, shadowMaxHalf: 900, terrainCastsShadow: false,
    treeShadowDist: 300, treeNear: 250, treeMid: 550, treeFar: 2600,
    ao: false, aoFullRes: false, bloom: false, textureSize: 256, anisotropy: 4, facadeDetail: 1, waterDetail: 1, wind: false,
  },
  high: {
    renderScale: 0.8, msaa: 4, shadows: true, shadowMapSize: 4096, shadowMaxHalf: 1400, terrainCastsShadow: false,
    treeShadowDist: 500, treeNear: 350, treeMid: 800, treeFar: 4200,
    ao: true, aoFullRes: false, bloom: true, textureSize: 256, anisotropy: 8, facadeDetail: 2, waterDetail: 2, wind: true,
  },
  ultra: {
    renderScale: 1, msaa: 4, shadows: true, shadowMapSize: 4096, shadowMaxHalf: 2000, terrainCastsShadow: true,
    treeShadowDist: 800, treeNear: 500, treeMid: 1200, treeFar: 1e9,
    ao: true, aoFullRes: true, bloom: true, textureSize: 512, anisotropy: 16, facadeDetail: 3, waterDetail: 3, wind: true,
  },
};

/** Features the governor may switch off, cheapest loss of quality first. */
export type Feature = 'ao' | 'bloom' | 'shadowDetail' | 'treeShadows' | 'treeDistance' | 'msaa' | 'shadows';

export const SHED_ORDER: readonly Feature[] = ['ao', 'bloom', 'shadowDetail', 'treeShadows', 'treeDistance', 'msaa', 'shadows'];

export const FEATURE_LABELS: Record<Feature, string> = {
  ao: 'ambient occlusion',
  bloom: 'bloom',
  shadowDetail: 'shadow resolution',
  treeShadows: 'tree shadows',
  treeDistance: 'tree draw distance',
  msaa: 'anti-aliasing',
  shadows: 'shadows',
};

/** Settings of a level with the given features shed. */
export function applyShed(base: QualitySettings, shed: readonly Feature[]): QualitySettings {
  const s = { ...base };
  for (const f of shed) {
    switch (f) {
      case 'ao':
        s.ao = false;
        break;
      case 'bloom':
        s.bloom = false;
        break;
      case 'shadowDetail':
        s.shadowMapSize = Math.max(1024, s.shadowMapSize / 2);
        s.shadowMaxHalf = Math.min(s.shadowMaxHalf, 700);
        s.terrainCastsShadow = false;
        break;
      case 'treeShadows':
        s.treeShadowDist = 0;
        break;
      case 'treeDistance':
        s.treeNear *= 0.6;
        s.treeMid *= 0.6;
        s.treeFar = Math.min(s.treeFar, 2500) * 0.6;
        break;
      case 'msaa':
        s.msaa = 0;
        break;
      case 'shadows':
        s.shadows = false;
        break;
    }
  }
  return s;
}

/** Features of `base` that are actually on (shedding an off feature is pointless). */
export function sheddable(base: QualitySettings): Feature[] {
  return SHED_ORDER.filter((f) => {
    switch (f) {
      case 'ao':
        return base.ao;
      case 'bloom':
        return base.bloom;
      case 'shadowDetail':
        return base.shadows && (base.shadowMapSize > 1024 || base.shadowMaxHalf > 700 || base.terrainCastsShadow);
      case 'treeShadows':
        return base.shadows && base.treeShadowDist > 0;
      case 'treeDistance':
        return true;
      case 'msaa':
        return base.msaa > 0;
      case 'shadows':
        return base.shadows;
    }
  });
}

export function lowerLevel(level: GraphicsLevel): GraphicsLevel | null {
  const i = LEVEL_ORDER.indexOf(level);
  return i > 0 ? LEVEL_ORDER[i - 1] : null;
}

export function higherLevel(level: GraphicsLevel): GraphicsLevel | null {
  const i = LEVEL_ORDER.indexOf(level);
  return i >= 0 && i < LEVEL_ORDER.length - 1 ? LEVEL_ORDER[i + 1] : null;
}

// ---------------------------------------------------------------- governor

export interface GovernorOptions {
  /** Frame rate that must be maintained. */
  targetFps: number;
  /** Below this the render scale goes down. */
  degradeFps: number;
  /** At or above this (or with GPU headroom) quality comes back. */
  upgradeFps: number;
  minScale: number;
  stepDown: number;
  stepUp: number;
  /** GPU frame time (ms) under which there is headroom to add quality. */
  gpuHeadroomMs: number;
}

export const DEFAULT_GOVERNOR: GovernorOptions = {
  targetFps: 45,
  degradeFps: 48,
  upgradeFps: 58,
  minScale: 0.5,
  stepDown: 0.1,
  stepUp: 0.05,
  gpuHeadroomMs: 11,
};

export type GovernorEvent =
  | { kind: 'scale'; scale: number }
  | { kind: 'shed'; feature: Feature }
  | { kind: 'restore'; feature: Feature }
  /** Nothing left to lower and still too slow (Auto may drop a level). */
  | { kind: 'exhausted' }
  /** Everything at maximum and lots of headroom (Auto may raise a level). */
  | { kind: 'surplus' };

/**
 * Keeps an exponential moving average of the frame time and, from it,
 * adjusts the render scale (fast path) and the set of shed features (slow
 * path) with hysteresis. Feed it one sample per frame.
 */
export class FpsGovernor {
  readonly opts: GovernorOptions;
  scale = 1;
  maxScale = 1;
  /** Shed features, in shedding order. */
  shed: Feature[] = [];
  private available: Feature[] = [];
  private ema = 1000 / 60;
  private gpuEma: number | null = null;
  private samples = 0;
  private pausedUntil = 0;
  private lowSince: number | null = null;
  private highSince: number | null = null;
  private lastChange = -Infinity;
  /** Per feature: earliest time it may be restored again (grows when it keeps failing). */
  private readonly restoreAfter = new Map<Feature, number>();
  private readonly backoff = new Map<Feature, number>();
  private readonly restoredAt = new Map<Feature, number>();
  private lastSignal = -Infinity;

  constructor(opts: Partial<GovernorOptions> = {}) {
    this.opts = { ...DEFAULT_GOVERNOR, ...opts };
  }

  /** Starts over for a (new) level: full scale, nothing shed. */
  reset(maxScale: number, available: readonly Feature[], now = 0): void {
    this.maxScale = maxScale;
    this.scale = maxScale;
    this.available = [...available];
    this.shed = [];
    this.samples = 0;
    this.lowSince = null;
    this.highSince = null;
    this.lastChange = now;
    this.lastSignal = now;
    this.restoreAfter.clear();
    this.backoff.clear();
    this.restoredAt.clear();
  }

  /** Ignores frames until `until` (tile loading, shader compiles, hidden tab). */
  pause(until: number): void {
    this.pausedUntil = Math.max(this.pausedUntil, until);
    this.lowSince = null;
    this.highSince = null;
  }

  get fps(): number {
    return 1000 / this.ema;
  }

  get gpuMs(): number | null {
    return this.gpuEma;
  }

  /** True while the image is below what the level asks for. */
  get degraded(): boolean {
    return this.shed.length > 0 || this.scale < this.maxScale - 1e-6;
  }

  sample(now: number, frameMs: number, gpuMs: number | null = null): GovernorEvent | null {
    if (now < this.pausedUntil) return null;
    // Frames longer than 250 ms are hitches (GC, compile, tab switch), not load.
    if (!(frameMs > 0) || frameMs > 250) return null;
    const a = this.samples < 10 ? 0.3 : 0.08;
    this.ema = this.samples === 0 ? frameMs : this.ema + (frameMs - this.ema) * a;
    if (gpuMs !== null && Number.isFinite(gpuMs)) {
      this.gpuEma = this.gpuEma === null ? gpuMs : this.gpuEma + (gpuMs - this.gpuEma) * 0.1;
    }
    this.samples++;
    if (this.samples < 15) return null;

    const o = this.opts;
    const fps = this.fps;
    const slow = fps < o.degradeFps;
    const headroom = fps >= o.upgradeFps && (this.gpuEma === null || this.gpuEma < o.gpuHeadroomMs);
    this.lowSince = slow ? (this.lowSince ?? now) : null;
    this.highSince = headroom ? (this.highSince ?? now) : null;

    if (slow && this.lowSince !== null) {
      const lowFor = now - this.lowSince;
      if (this.scale > o.minScale + 1e-6 && lowFor >= 500 && now - this.lastChange >= 400) {
        this.scale = Math.max(o.minScale, round2(this.scale - o.stepDown));
        return this.changed(now, { kind: 'scale', scale: this.scale });
      }
      if (this.scale <= o.minScale + 1e-6 && fps < o.targetFps && lowFor >= 2000 && now - this.lastChange >= 1000) {
        const next = this.available.find((f) => !this.shed.includes(f));
        if (next) {
          this.shed.push(next);
          // A feature restored less than 10 s ago that has to go again waits longer next time.
          const restored = this.restoredAt.get(next);
          if (restored !== undefined && now - restored < 10000) {
            this.backoff.set(next, Math.min(120000, (this.backoff.get(next) ?? 5000) * 2));
          }
          this.restoreAfter.set(next, now + (this.backoff.get(next) ?? 5000));
          return this.changed(now, { kind: 'shed', feature: next });
        }
        if (now - this.lastSignal >= 3000) {
          this.lastSignal = now;
          return { kind: 'exhausted' };
        }
      }
      return null;
    }

    if (headroom && this.highSince !== null) {
      const highFor = now - this.highSince;
      if (this.scale < this.maxScale - 1e-6 && highFor >= 3000 && now - this.lastChange >= 1500) {
        this.scale = Math.min(this.maxScale, round2(this.scale + o.stepUp));
        return this.changed(now, { kind: 'scale', scale: this.scale });
      }
      if (this.scale >= this.maxScale - 1e-6 && this.shed.length && highFor >= 5000 && now - this.lastChange >= 3000) {
        const last = this.shed[this.shed.length - 1];
        if (now >= (this.restoreAfter.get(last) ?? 0)) {
          this.shed.pop();
          this.restoredAt.set(last, now);
          return this.changed(now, { kind: 'restore', feature: last });
        }
      }
      if (!this.degraded && highFor >= 10000 && now - this.lastChange >= 10000 && now - this.lastSignal >= 10000) {
        this.lastSignal = now;
        return { kind: 'surplus' };
      }
    }
    return null;
  }

  private changed(now: number, e: GovernorEvent): GovernorEvent {
    this.lastChange = now;
    this.lowSince = null;
    this.highSince = null;
    return e;
  }
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
