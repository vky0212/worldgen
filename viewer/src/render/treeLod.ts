/**
 * Distance-based level of detail for the instanced trees, per tile:
 * full trees near the camera, canopies without trunks further out, low-poly
 * impostors (sharing the instance matrices) beyond, nothing past the draw
 * distance. Trees dominate the triangle count of a world (tens of millions
 * at full detail), so this is what keeps large views above the frame target.
 */
import { Box3, BufferAttribute, BufferGeometry, InstancedMesh, Vector3 } from 'three';
import type { Object3D } from 'three';
import type { QualitySettings } from './quality';
import type { TileEntry } from '../types';

interface TreeGroup {
  trunks: InstancedMesh[];
  foliage: InstancedMesh[];
  /** Foliage meshes that only show close up (grass). */
  ground: InstancedMesh[];
  lod: InstancedMesh | null;
}

interface TileTrees {
  box: Box3;
  groups: TreeGroup[];
  /** 0 hidden · 1 impostors · 2 canopies · 3 full; -1 = not applied yet. */
  state: number;
  shadows: boolean;
}

const State = { Hidden: 0, Impostor: 1, Canopy: 2, Full: 3 } as const;
type State = (typeof State)[keyof typeof State];

const lodGeometries = new Map<string, BufferGeometry>();

export class TreeLod {
  private readonly tiles = new Map<string, TileTrees>();
  private readonly cam = new Vector3();
  private lastUpdate = -Infinity;
  private readonly lastCam = new Vector3(Infinity, Infinity, Infinity);
  /** Instances drawn at each detail, for stats. */
  counts = { full: 0, canopy: 0, impostor: 0, hidden: 0 };

  /** Registers the trees layer of a tile. `trees` are the layer nodes. */
  addTile(tileId: string, entry: TileEntry, trees: readonly Object3D[]): void {
    const groups: TreeGroup[] = [];
    for (const node of trees) {
      const byParent = new Map<Object3D, TreeGroup>();
      node.traverse((o) => {
        if (!(o instanceof InstancedMesh) || !o.parent) return;
        let g = byParent.get(o.parent);
        if (!g) {
          g = { trunks: [], foliage: [], ground: [], lod: null };
          byParent.set(o.parent, g);
          groups.push(g);
        }
        const name = (Array.isArray(o.material) ? o.material[0] : o.material)?.name ?? '';
        if (name === 'tree_trunk') g.trunks.push(o);
        else if (name.includes('grass')) g.ground.push(o);
        else g.foliage.push(o);
      });
    }
    if (!groups.length) return;
    const b = entry.bounds ?? [0, 0, 0, 0];
    const z = entry.zrange ?? [0, 100];
    const box = new Box3(new Vector3(b[0], z[0], -b[3]), new Vector3(b[2], z[1] + 40, -b[1]));
    this.tiles.set(tileId, { box, groups, state: -1, shadows: false });
    this.lastUpdate = -Infinity;
  }

  /**
   * Applies the LOD for the camera. Throttled to 4 Hz unless the camera
   * moved a lot. Returns true when shadow casters changed.
   */
  update(camera: Vector3, s: QualitySettings, now: number, force = false): boolean {
    if (!force && now - this.lastUpdate < 250 && camera.distanceToSquared(this.lastCam) < 400) return false;
    this.lastUpdate = now;
    this.lastCam.copy(camera);
    this.cam.copy(camera);
    let shadowChanged = false;
    const counts = { full: 0, canopy: 0, impostor: 0, hidden: 0 };
    for (const t of this.tiles.values()) {
      const d = t.box.distanceToPoint(this.cam);
      const state = d < s.treeNear ? State.Full : d < s.treeMid ? State.Canopy : d < s.treeFar ? State.Impostor : State.Hidden;
      const shadows = s.shadows && d < s.treeShadowDist;
      if (state !== t.state || force) {
        this.apply(t, state);
        t.state = state;
      }
      if (shadows !== t.shadows || force) {
        for (const g of t.groups) for (const m of [...g.trunks, ...g.foliage]) m.castShadow = shadows;
        t.shadows = shadows;
        shadowChanged = true;
      }
      const n = t.groups.reduce((sum, g) => sum + (g.foliage[0]?.count ?? 0), 0);
      if (state === State.Full) counts.full += n;
      else if (state === State.Canopy) counts.canopy += n;
      else if (state === State.Impostor) counts.impostor += n;
      else counts.hidden += n;
      // Colour modes swap materials: impostors follow their canopy.
      for (const g of t.groups) {
        if (g.lod && g.foliage[0]) {
          g.lod.material = g.foliage[0].material;
          g.lod.instanceColor = g.foliage[0].instanceColor;
        }
      }
    }
    this.counts = counts;
    return shadowChanged || force;
  }

  /** Everything visible again (standard mode). */
  reset(castShadow: boolean): void {
    for (const t of this.tiles.values()) {
      for (const g of t.groups) {
        for (const m of [...g.trunks, ...g.foliage, ...g.ground]) {
          m.visible = true;
          m.castShadow = castShadow;
        }
        if (g.lod) g.lod.visible = false;
      }
      t.state = -1;
      t.shadows = castShadow;
    }
    this.lastUpdate = -Infinity;
  }

  private apply(t: TileTrees, state: State): void {
    for (const g of t.groups) {
      for (const m of g.trunks) m.visible = state === State.Full;
      for (const m of g.ground) m.visible = state >= State.Canopy;
      for (const m of g.foliage) m.visible = state >= State.Canopy;
      if (state === State.Impostor && !g.lod) g.lod = makeImpostor(g.foliage[0]);
      if (g.lod) g.lod.visible = state === State.Impostor;
    }
  }
}

/** Low-poly stand-in sharing the canopy's instance matrices. */
function makeImpostor(src: InstancedMesh | undefined): InstancedMesh | null {
  if (!src || !src.parent) return null;
  const name = (Array.isArray(src.material) ? src.material[0] : src.material)?.name ?? '';
  src.geometry.computeBoundingBox();
  const bb = src.geometry.boundingBox!;
  const key = `${name}:${bb.min.toArray().map((v) => v.toFixed(2))}:${bb.max.toArray().map((v) => v.toFixed(2))}`;
  let geo = lodGeometries.get(key);
  if (!geo) {
    geo = name.includes('conifer') ? pyramid(bb) : octahedron(bb);
    lodGeometries.set(key, geo);
  }
  const lod = new InstancedMesh(geo, src.material, src.count);
  lod.instanceMatrix = src.instanceMatrix;
  lod.instanceColor = src.instanceColor;
  lod.count = src.count;
  lod.name = `${src.name}_lod`;
  lod.castShadow = false;
  lod.receiveShadow = false;
  lod.matrixAutoUpdate = false;
  lod.matrix.copy(src.matrix);
  src.parent.add(lod);
  lod.updateMatrixWorld(true);
  lod.computeBoundingSphere();
  lod.userData.lodOf = src.uuid;
  return lod;
}

function build(positions: number[], normals: number[]): BufferGeometry {
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(new Float32Array(positions), 3));
  g.setAttribute('normal', new BufferAttribute(new Float32Array(normals), 3));
  g.computeBoundingSphere();
  return g;
}

/** Four-sided pyramid (conifers): 4 triangles. */
function pyramid(bb: Box3): BufferGeometry {
  const c = bb.getCenter(new Vector3());
  const r = Math.max(bb.max.x - bb.min.x, bb.max.z - bb.min.z) / 2;
  const apex = [c.x, bb.max.y, c.z];
  const y0 = bb.min.y + (bb.max.y - bb.min.y) * 0.05;
  const base = [
    [c.x + r, y0, c.z],
    [c.x, y0, c.z - r],
    [c.x - r, y0, c.z],
    [c.x, y0, c.z + r],
  ];
  const pos: number[] = [];
  const nor: number[] = [];
  for (let i = 0; i < 4; i++) {
    const a = base[i];
    const b = base[(i + 1) % 4];
    pos.push(...a, ...b, ...apex);
    for (const v of [a, b, apex]) {
      const n = new Vector3(v[0] - c.x, 0, v[2] - c.z).normalize().multiplyScalar(0.8).add(new Vector3(0, 0.6, 0)).normalize();
      nor.push(n.x, n.y, n.z);
    }
  }
  return build(pos, nor);
}

/** Octahedron (broadleaf crowns): 8 triangles with rounded normals. */
function octahedron(bb: Box3): BufferGeometry {
  const c = bb.getCenter(new Vector3());
  const h = bb.getSize(new Vector3()).multiplyScalar(0.5);
  const v = {
    px: [c.x + h.x, c.y, c.z], nx: [c.x - h.x, c.y, c.z],
    py: [c.x, c.y + h.y, c.z], ny: [c.x, c.y - h.y, c.z],
    pz: [c.x, c.y, c.z + h.z], nz: [c.x, c.y, c.z - h.z],
  };
  const faces: (keyof typeof v)[][] = [
    ['px', 'py', 'pz'], ['pz', 'py', 'nx'], ['nx', 'py', 'nz'], ['nz', 'py', 'px'],
    ['pz', 'ny', 'px'], ['nx', 'ny', 'pz'], ['nz', 'ny', 'nx'], ['px', 'ny', 'nz'],
  ];
  const pos: number[] = [];
  const nor: number[] = [];
  for (const f of faces) {
    for (const k of f) {
      const p = v[k];
      pos.push(...p);
      const n = new Vector3((p[0] - c.x) / Math.max(h.x, 1e-3), (p[1] - c.y) / Math.max(h.y, 1e-3) + 0.3, (p[2] - c.z) / Math.max(h.z, 1e-3)).normalize();
      nor.push(n.x, n.y, n.z);
    }
  }
  return build(pos, nor);
}
