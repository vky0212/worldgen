/**
 * Colour modes: recolour pickable meshes per feature from tile metadata.
 *
 * Non-"materials" modes write a `color` vertex attribute (or per-instance
 * colours for instanced trees) and swap in a vertex-coloured material; the
 * original material and original `color` attribute (terrain COLOR_0) are kept
 * in WeakMaps and restored when switching back to "materials".
 */
import {
  BufferAttribute,
  Color,
  InstancedBufferAttribute,
  InstancedMesh,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshLambertMaterial,
} from 'three';
import type { BufferGeometry, Material, Side } from 'three';
import { FEATURE_ATTR } from './picking';
import { markShared } from './stream/content';
import type { EntityRecord } from './types';
import type { LoadedTile, Pickable, WorldModel } from './world';

export type ColorMode = 'materials' | 'observed' | 'confidence' | 'source';

export const COLOR_MODES: { id: ColorMode; label: string }[] = [
  { id: 'materials', label: 'Materials' },
  { id: 'observed', label: 'Observed vs inferred' },
  { id: 'confidence', label: 'Confidence' },
  { id: 'source', label: 'Source' },
];

export interface LegendEntry {
  color: string;
  label: string;
  gradient?: string;
}

const OBSERVED = '#2fb36d';
const INFERRED = '#f59e0b';
const UNCERTAIN = '#e5484d';
const UNKNOWN = '#8a8f98';
/** Terrain is blended towards this grey so features on top stay readable. */
const TERRAIN_MUTE = new Color('#b4b8bd');
const TERRAIN_MUTE_AMOUNT = 0.6;
const SOURCE_PALETTE = [
  '#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#17becf',
  '#bcbd22', '#7f7f7f',
];

/** Red (0) → yellow → green (1). */
export function confidenceColor(c: number, target = new Color()): Color {
  const t = Math.min(1, Math.max(0, c));
  return target.setHSL((t * 120) / 360, 0.75, 0.48);
}

export function hasInferred(r: EntityRecord): boolean {
  return !!r.inferred && Object.keys(r.inferred).length > 0;
}

interface SavedState {
  material: Material | Material[];
  instanceColor: InstancedBufferAttribute | null;
}

interface SavedGeometry {
  colorAttr: BufferAttribute | null;
}

export class ColorModeManager {
  mode: ColorMode = 'materials';
  private readonly saved = new WeakMap<Pickable['object'], SavedState>();
  /** Per geometry (geometries may be shared between objects). */
  private readonly savedGeometry = new WeakMap<BufferGeometry, SavedGeometry>();
  /**
   * Colour buffers written by the analytic modes, kept across mode switches
   * (reused instead of re-allocated) until their tile is unloaded.
   */
  private readonly generatedColor = new WeakMap<BufferGeometry, BufferAttribute>();
  private readonly generatedInstanceColor = new WeakMap<InstancedMesh, InstancedBufferAttribute>();
  private readonly materials = new Map<string, Material>();
  private readonly sourceColors = new Map<string, string>();

  constructor(knownSources: readonly string[] = []) {
    for (const s of knownSources) this.sourceColorHex(s);
  }

  /** Applies `mode` to every pickable in the world. */
  setMode(mode: ColorMode, world: WorldModel): void {
    this.mode = mode;
    for (const p of world.pickables) this.applyTo(p, world);
  }

  /** Re-applies the current mode to one tile (after its GLB or meta loads). */
  refreshTile(tileId: string, world: WorldModel): void {
    const tile = world.tiles.get(tileId);
    if (!tile) return;
    for (const p of tile.pickables) this.applyTo(p, world);
  }

  /**
   * A tile is being unloaded: puts its own materials back and leaves the
   * colour buffers this manager generated attached to the tile's geometries
   * (under a private name) so the tile's disposal frees them on the GPU too.
   */
  forgetTile(tile: LoadedTile): void {
    for (const p of tile.pickables) {
      this.restore(p);
      const obj = p.object;
      const geometry = obj.geometry as BufferGeometry;
      const gen = this.generatedColor.get(geometry);
      if (gen) {
        geometry.setAttribute('_wg_release_color', gen);
        this.generatedColor.delete(geometry);
      }
      if (obj instanceof InstancedMesh) {
        const ic = this.generatedInstanceColor.get(obj);
        if (ic) {
          geometry.setAttribute('_wg_release_instance_color', ic);
          this.generatedInstanceColor.delete(obj);
        }
      }
    }
  }

  legend(): LegendEntry[] {
    switch (this.mode) {
      case 'observed':
        return [
          { color: OBSERVED, label: 'Fully observed' },
          { color: INFERRED, label: 'Has inferred properties' },
          { color: UNCERTAIN, label: 'Uncertain geometry' },
          { color: UNKNOWN, label: 'No metadata' },
        ];
      case 'confidence': {
        const stops = [0, 0.25, 0.5, 0.75, 1].map((c) => `#${confidenceColor(c).getHexString()}`);
        return [
          { color: stops[0], label: '0 → 1 confidence', gradient: `linear-gradient(90deg, ${stops.join(', ')})` },
          { color: UNKNOWN, label: 'No confidence value' },
        ];
      }
      case 'source':
        return [
          ...[...this.sourceColors].map(([label, color]) => ({ color, label })),
          { color: UNKNOWN, label: 'Unknown source' },
        ];
      default:
        return [];
    }
  }

  /** Colour of a record in the current mode (hex string). */
  colorFor(record: EntityRecord | undefined): string {
    if (!record) return UNKNOWN;
    switch (this.mode) {
      case 'observed':
        if (record.uncertain) return UNCERTAIN;
        return hasInferred(record) ? INFERRED : OBSERVED;
      case 'confidence':
        return typeof record.confidence === 'number' && Number.isFinite(record.confidence)
          ? `#${confidenceColor(record.confidence).getHexString()}`
          : UNKNOWN;
      case 'source':
        return record.source ? this.sourceColorHex(record.source) : UNKNOWN;
      default:
        return UNKNOWN;
    }
  }

  private sourceColorHex(source: string): string {
    let c = this.sourceColors.get(source);
    if (!c) {
      c = SOURCE_PALETTE[this.sourceColors.size % SOURCE_PALETTE.length];
      this.sourceColors.set(source, c);
    }
    return c;
  }

  private applyTo(p: Pickable, world: WorldModel): void {
    const obj = p.object;
    if (this.mode === 'materials') {
      this.restore(p);
      return;
    }
    if (obj instanceof InstancedMesh && p.instanceIds) {
      this.applyInstanced(obj, p, world);
    } else if (p.featureIds && (obj instanceof Mesh || obj instanceof Line)) {
      this.applyVertex(obj, p, world);
    }
  }

  private save(obj: Pickable['object']): SavedState {
    let s = this.saved.get(obj);
    if (!s) {
      s = { material: obj.material, instanceColor: obj instanceof InstancedMesh ? obj.instanceColor : null };
      this.saved.set(obj, s);
    }
    return s;
  }

  private saveGeometry(geometry: BufferGeometry): SavedGeometry {
    let g = this.savedGeometry.get(geometry);
    if (!g) {
      g = { colorAttr: (geometry.getAttribute('color') as BufferAttribute | undefined) ?? null };
      this.savedGeometry.set(geometry, g);
    }
    return g;
  }

  private restore(p: Pickable): void {
    const obj = p.object;
    const s = this.saved.get(obj);
    if (!s) return;
    obj.material = s.material;
    if (obj instanceof InstancedMesh) {
      obj.instanceColor = s.instanceColor;
    } else {
      const geometry = obj.geometry as BufferGeometry;
      const g = this.savedGeometry.get(geometry);
      if (g) {
        // The generated buffer stays in generatedColor for reuse and release.
        if (g.colorAttr) geometry.setAttribute('color', g.colorAttr);
        else geometry.deleteAttribute('color');
        this.savedGeometry.delete(geometry);
      }
    }
    this.saved.delete(obj);
  }

  private applyVertex(obj: Mesh | Line, p: Pickable, world: WorldModel): void {
    const geometry = obj.geometry as BufferGeometry;
    const fid = geometry.getAttribute(FEATURE_ATTR);
    if (!fid || !p.featureIds) return;
    const saved = this.save(obj);
    this.saveGeometry(geometry);

    // Colour table per feature index.
    const tmp = new Color();
    const table = new Float32Array(p.featureIds.length * 3);
    const mute = p.layer === 'terrain';
    p.featureIds.forEach((id, k) => {
      tmp.set(this.colorFor(world.recordFor(id, p.tileId)));
      if (mute) tmp.lerp(TERRAIN_MUTE, TERRAIN_MUTE_AMOUNT);
      table[k * 3] = tmp.r;
      table[k * 3 + 1] = tmp.g;
      table[k * 3 + 2] = tmp.b;
    });
    const fallback = new Color(UNKNOWN);

    const n = fid.count;
    let attr = this.generatedColor.get(geometry);
    if (!attr || attr.count !== n) {
      attr = new BufferAttribute(new Float32Array(n * 3), 3);
      this.generatedColor.set(geometry, attr);
    }
    geometry.setAttribute('color', attr);
    const arr = attr.array as Float32Array;
    for (let v = 0; v < n; v++) {
      const k = Math.round(fid.getX(v));
      if (k >= 0 && k * 3 + 2 < table.length) {
        arr[v * 3] = table[k * 3];
        arr[v * 3 + 1] = table[k * 3 + 1];
        arr[v * 3 + 2] = table[k * 3 + 2];
      } else {
        arr[v * 3] = fallback.r;
        arr[v * 3 + 1] = fallback.g;
        arr[v * 3 + 2] = fallback.b;
      }
    }
    attr.needsUpdate = true;

    if (obj instanceof Line) {
      obj.material = this.material('line', 0);
    } else {
      if (!geometry.getAttribute('normal')) geometry.computeVertexNormals();
      obj.material = this.material('mesh', sideOf(saved.material));
    }
  }

  private applyInstanced(obj: InstancedMesh, p: Pickable, world: WorldModel): void {
    const ids = p.instanceIds!;
    this.save(obj);
    const n = obj.count;
    const saved = this.saved.get(obj)!;
    let ic = this.generatedInstanceColor.get(obj);
    if (!ic || ic.count !== n) {
      ic = new InstancedBufferAttribute(new Float32Array(n * 3), 3);
      this.generatedInstanceColor.set(obj, ic);
    }
    obj.instanceColor = ic;
    const tmp = new Color();
    const cache = new Map<string, string>();
    for (let i = 0; i < n; i++) {
      const id = ids[i];
      let hex = cache.get(id);
      if (!hex) {
        hex = this.colorFor(typeof id === 'string' ? world.recordFor(id, p.tileId) : undefined);
        cache.set(id, hex);
      }
      obj.setColorAt(i, tmp.set(hex));
    }
    ic.needsUpdate = true;
    obj.material = this.material('instance', sideOf(saved.material));
  }

  private material(kind: 'mesh' | 'line' | 'instance', side: Side): Material {
    const key = `${kind}:${side}`;
    let m = this.materials.get(key);
    if (!m) {
      // Instanced meshes get their colour from instanceColor (no vertex colours).
      m = markShared(
        kind === 'line'
          ? new LineBasicMaterial({ vertexColors: true })
          : new MeshLambertMaterial({ vertexColors: kind === 'mesh', side }),
      );
      this.materials.set(key, m);
    }
    return m;
  }
}

function sideOf(m: Material | Material[]): Side {
  const first = Array.isArray(m) ? m[0] : m;
  return first?.side ?? 0;
}
