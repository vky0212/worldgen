/**
 * In-memory model of the loaded world: tile scenes, layer nodes, pickable
 * meshes and the entity registry built from the tile meta.json files.
 *
 * Tiles come and go as the camera moves (see stream/TileStreamer): addTile
 * registers a tile's scene, removeTile unregisters it again (layer nodes,
 * pickables, entity occurrences) so it can be disposed. meta.json documents
 * are loaded lazily and independently of the GLBs (setMeta / dropMeta).
 */
import { Group, InstancedMesh, Line, Mesh, Points } from 'three';
import type { BufferGeometry, Object3D } from 'three';
import { FEATURE_ATTR, featureIdsOf, instanceIdsOf } from './picking';
import { disposeTree, estimateGpuBytes } from './stream/content';
import type { TileContent } from './stream/content';
import { LAYER_NAMES } from './types';
import type { EntityRecord, TileEntry, TileMeta } from './types';

/** A drawable object (mesh, line, instanced mesh) that can be picked/coloured. */
export interface Pickable {
  object: Mesh | Line | Points;
  tileId: string;
  layer: string;
  /** Feature id list of a mesh carrying `_feature_id_0`, else null. */
  featureIds: string[] | null;
  /** Entity ids per instance for instanced meshes, else null. */
  instanceIds: string[] | null;
}

export interface LoadedTile extends TileContent {
  entry: TileEntry;
  layers: Map<string, Object3D[]>;
  pickables: Pickable[];
  /** Entity ids drawn in this tile (keys into the occurrence index). */
  entityIds: Set<string>;
}

export interface EntityEntry {
  id: string;
  record: EntityRecord;
  tiles: string[];
}

/** A place where an entity is drawn: an object + feature index or instance index. */
export interface EntityOccurrence {
  pickable: Pickable;
  featureIndex?: number;
  instanceIndex?: number;
}

interface MetaState {
  meta: TileMeta | null;
  error: string | null;
}

export class WorldModel {
  readonly root = new Group();
  readonly tiles = new Map<string, LoadedTile>();
  /** Layer nodes of the loaded tiles, per layer. */
  readonly layerNodes = new Map<string, Set<Object3D>>();
  private readonly layerVisible = new Map<string, boolean>();
  private readonly metas = new Map<string, MetaState>();
  private readonly entities = new Map<string, EntityEntry>();
  private readonly occurrences = new Map<string, EntityOccurrence[]>();
  private readonly tileOfRoot = new WeakMap<Object3D, string>();
  private readonly pickableOf = new WeakMap<Object3D, Pickable>();
  private pickableCache: Pickable[] | null = null;

  constructor() {
    this.root.name = 'worldgen_tiles';
  }

  /** Every pickable of every loaded tile. */
  get pickables(): Pickable[] {
    if (!this.pickableCache) {
      const all: Pickable[] = [];
      for (const t of this.tiles.values()) for (const p of t.pickables) all.push(p);
      this.pickableCache = all;
    }
    return this.pickableCache;
  }

  /** Registers a freshly loaded tile scene and returns the tile record. */
  addTile(entry: TileEntry, scene: Group): LoadedTile {
    if (this.tiles.has(entry.id)) this.removeTile(entry.id)?.dispose();
    const tile: LoadedTile = {
      entry,
      root: scene,
      layers: new Map(),
      pickables: [],
      entityIds: new Set(),
      gpuBytes: 0,
      dispose: () => disposeTree(scene),
    };
    this.tileOfRoot.set(scene, entry.id);
    collectLayerNodes(scene, tile.layers);

    for (const [layer, nodes] of tile.layers) {
      let set = this.layerNodes.get(layer);
      if (!set) {
        set = new Set();
        this.layerNodes.set(layer, set);
      }
      for (const node of nodes) {
        set.add(node);
        node.visible = this.isLayerVisible(layer);
        node.traverse((obj) => {
          const p = makePickable(obj, entry.id, layer);
          if (!p) return;
          tile.pickables.push(p);
          this.pickableOf.set(obj, p);
          this.indexOccurrences(p, tile.entityIds);
        });
      }
    }

    // Static content: compute matrices once.
    scene.updateMatrixWorld(true);
    scene.traverse((o) => {
      o.matrixAutoUpdate = false;
    });
    tile.gpuBytes = estimateGpuBytes(scene);

    this.tiles.set(entry.id, tile);
    this.pickableCache = null;
    this.root.add(scene);
    return tile;
  }

  /**
   * Unregisters a tile (scene detached, layer nodes, pickables and entity
   * occurrences removed) and returns it; the caller disposes it.
   */
  removeTile(id: string): LoadedTile | undefined {
    const tile = this.tiles.get(id);
    if (!tile) return undefined;
    this.tiles.delete(id);
    this.pickableCache = null;
    this.root.remove(tile.root);
    for (const [layer, nodes] of tile.layers) {
      const set = this.layerNodes.get(layer);
      if (!set) continue;
      for (const n of nodes) set.delete(n);
    }
    for (const eid of tile.entityIds) {
      const list = this.occurrences.get(eid);
      if (!list) continue;
      const kept = list.filter((o) => o.pickable.tileId !== id);
      if (kept.length) this.occurrences.set(eid, kept);
      else this.occurrences.delete(eid);
    }
    return tile;
  }

  /** Attaches a tile's meta.json (independent of whether its GLB is loaded). */
  setMeta(tileId: string, meta: TileMeta | null, error: string | null): void {
    if (this.metas.has(tileId)) this.dropMeta(tileId);
    this.metas.set(tileId, { meta, error });
    if (!meta) return;
    for (const [id, record] of Object.entries(meta.entities)) {
      if (!record || typeof record !== 'object') continue;
      const e = this.entities.get(id);
      if (e) {
        if (!e.tiles.includes(tileId)) e.tiles.push(tileId);
      } else {
        this.entities.set(id, { id, record, tiles: [tileId] });
      }
    }
  }

  /** Forgets a tile's meta.json (evicted from the meta cache). */
  dropMeta(tileId: string): void {
    const st = this.metas.get(tileId);
    if (!st) return;
    this.metas.delete(tileId);
    if (!st.meta) return;
    for (const id of Object.keys(st.meta.entities)) {
      const e = this.entities.get(id);
      if (!e) continue;
      e.tiles = e.tiles.filter((t) => t !== tileId);
      if (!e.tiles.length) {
        this.entities.delete(id);
      } else if (e.record === st.meta.entities[id]) {
        const other = this.metas.get(e.tiles[0])?.meta?.entities[id];
        if (other) e.record = other;
      }
    }
  }

  hasMeta(tileId: string): boolean {
    return this.metas.has(tileId);
  }

  metaOf(tileId: string): TileMeta | null {
    return this.metas.get(tileId)?.meta ?? null;
  }

  get entityCount(): number {
    return this.entities.size;
  }

  getEntity(id: string): EntityEntry | undefined {
    return this.entities.get(id);
  }

  /** Record for an entity, preferring the given tile's own meta. */
  recordFor(id: string, tileId?: string): EntityRecord | undefined {
    if (tileId) {
      const r = this.metas.get(tileId)?.meta?.entities[id];
      if (r) return r;
    }
    return this.entities.get(id)?.record;
  }

  occurrencesOf(id: string): EntityOccurrence[] {
    return this.occurrences.get(id) ?? [];
  }

  pickableFor(obj: Object3D): Pickable | undefined {
    return this.pickableOf.get(obj);
  }

  /** Tile id owning a scene-graph object (walks up to the tile root). */
  tileIdOf(obj: Object3D): string | null {
    for (let o: Object3D | null = obj; o; o = o.parent) {
      const id = this.tileOfRoot.get(o);
      if (id !== undefined) return id;
    }
    return null;
  }

  /** Loaded tiles whose bounds contain local (x, y). */
  tilesAt(x: number, y: number): LoadedTile[] {
    const out: LoadedTile[] = [];
    for (const t of this.tiles.values()) {
      const b = t.entry.bounds;
      if (b && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) out.push(t);
    }
    return out;
  }

  /** All layer names: known contract layers first, then any extra ones found. */
  layerNames(manifestLayers: readonly string[] = []): string[] {
    const names: string[] = [...LAYER_NAMES];
    for (const n of [...manifestLayers, ...this.layerNodes.keys()]) if (!names.includes(n)) names.push(n);
    return names;
  }

  isLayerVisible(layer: string): boolean {
    return this.layerVisible.get(layer) ?? true;
  }

  setLayerVisible(layer: string, visible: boolean): void {
    this.layerVisible.set(layer, visible);
    for (const node of this.layerNodes.get(layer) ?? []) node.visible = visible;
  }

  /** Pickables whose layer is on and that are visible in the scene graph. */
  visiblePickables(exclude: readonly string[] = []): Pickable[] {
    return this.pickables.filter((p) => !exclude.includes(p.layer) && isShown(p.object));
  }

  private indexOccurrences(p: Pickable, ids: Set<string>): void {
    if (p.featureIds) {
      p.featureIds.forEach((id, featureIndex) => this.pushOccurrence(id, { pickable: p, featureIndex }, ids));
    }
    if (p.instanceIds) {
      p.instanceIds.forEach((id, instanceIndex) => this.pushOccurrence(id, { pickable: p, instanceIndex }, ids));
    }
  }

  private pushOccurrence(id: string, occ: EntityOccurrence, ids: Set<string>): void {
    if (typeof id !== 'string') return;
    ids.add(id);
    const list = this.occurrences.get(id);
    if (list) list.push(occ);
    else this.occurrences.set(id, [occ]);
  }
}

/** True when the object and all its ancestors are visible. */
export function isShown(obj: Object3D): boolean {
  for (let o: Object3D | null = obj; o; o = o.parent) if (!o.visible) return false;
  return true;
}

function layerNameOf(obj: Object3D): string | null {
  const ud = obj.userData as Record<string, unknown>;
  if (typeof ud.layer === 'string' && ud.layer) return ud.layer;
  return null;
}

/**
 * Finds layer nodes under a tile scene. Per contract they are the children of
 * the `tile_<id>` root and carry `extras.layer`; we also accept a known layer
 * name (original glTF name is kept in `userData.name` by GLTFLoader, while
 * `object.name` may be sanitised or suffixed).
 */
function collectLayerNodes(scene: Group, out: Map<string, Object3D[]>): void {
  const known = new Set<string>(LAYER_NAMES);
  const visit = (obj: Object3D, depth: number): void => {
    for (const child of obj.children) {
      const byExtras = layerNameOf(child);
      const originalName = typeof child.userData.name === 'string' ? (child.userData.name as string) : child.name;
      const layer = byExtras ?? (known.has(originalName) ? originalName : null);
      if (layer) {
        const list = out.get(layer);
        if (list) list.push(child);
        else out.set(layer, [child]);
      } else if (depth < 3) {
        visit(child, depth + 1);
      }
    }
  };
  // Normally: scene -> tile_<id> root -> layer nodes.
  visit(scene, 0);
  if (out.size === 0 && scene.children.length > 0) {
    // Non-conforming tile: keep its content visible and pickable as "other".
    out.set('other', [...scene.children]);
  }
}

function makePickable(obj: Object3D, tileId: string, layer: string): Pickable | null {
  if (!(obj instanceof Mesh || obj instanceof Line || obj instanceof Points)) return null;
  const geometry = obj.geometry as BufferGeometry;
  const instanceIds = obj instanceof InstancedMesh ? instanceIdsOf(obj) : null;
  const featureIds = geometry.getAttribute(FEATURE_ATTR) ? featureIdsOf(obj) : null;
  return { object: obj, tileId, layer, featureIds, instanceIds };
}
