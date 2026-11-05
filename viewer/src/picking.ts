/**
 * Picking (raycasting) and selection highlighting.
 *
 * GLTFLoader behaviour this relies on (three r187):
 *  - custom attribute `_FEATURE_ID_0` -> `geometry.attributes._feature_id_0`
 *    (name lower-cased);
 *  - primitive extras -> `geometry.userData`, mesh extras -> `mesh.userData`,
 *    node extras -> the node object's `userData` (when a node holds a single
 *    primitive the node *is* the Mesh, so node and mesh extras are merged);
 *  - EXT_mesh_gpu_instancing -> InstancedMesh (mesh userData is copied onto it;
 *    node extras such as `instance_ids` land on the InstancedMesh, or on its
 *    parent Group when the mesh has several primitives).
 */
import {
  Box3,
  Box3Helper,
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  InstancedMesh,
  Line,
  LineBasicMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  Raycaster,
  Vector2,
  Vector3,
} from 'three';
import type { Camera, Intersection, Object3D, Points } from 'three';
import { enuToThree } from './geo';
import type { EntityOccurrence, WorldModel } from './world';

export const FEATURE_ATTR = '_feature_id_0';

type Drawable = Mesh | Line | Points;

/** Looks up `feature_ids` on geometry, object, then ancestors' userData. */
export function featureIdsOf(obj: Object3D): string[] | null {
  const geometry = (obj as Partial<Drawable>).geometry as BufferGeometry | undefined;
  const fromGeom = geometry?.userData?.feature_ids;
  if (Array.isArray(fromGeom)) return fromGeom as string[];
  for (let o: Object3D | null = obj; o; o = o.parent) {
    const ids = (o.userData as Record<string, unknown>).feature_ids;
    if (Array.isArray(ids)) return ids as string[];
  }
  return null;
}

/** Looks up `instance_ids` on the object then its ancestors. */
export function instanceIdsOf(obj: Object3D): string[] | null {
  for (let o: Object3D | null = obj; o; o = o.parent) {
    const ids = (o.userData as Record<string, unknown>).instance_ids;
    if (Array.isArray(ids)) return ids as string[];
  }
  return null;
}

export interface EntityHit {
  entityId: string;
  tileId: string | null;
  object: Object3D;
  point: Vector3;
  featureIndex?: number;
  instanceIndex?: number;
}

/** Maps a raycast intersection on a tile object to an entity id. */
export function resolveEntityHit(hit: Intersection, world: WorldModel): EntityHit | null {
  const obj = hit.object;
  const tileId = world.tileIdOf(obj);
  const pickable = world.pickableFor(obj);
  // Tiles of a coarser level of detail stand for many tiles and hold no entities.
  if (tileId && (world.tiles.get(tileId)?.entry.level ?? 0) > 0) return null;

  if (obj instanceof InstancedMesh && hit.instanceId !== undefined) {
    const ids = pickable?.instanceIds ?? instanceIdsOf(obj);
    const id = ids?.[hit.instanceId];
    if (typeof id === 'string') {
      return { entityId: id, tileId, object: obj, point: hit.point.clone(), instanceIndex: hit.instanceId };
    }
  }

  const geometry = (obj as Partial<Drawable>).geometry as BufferGeometry | undefined;
  const attr = geometry?.getAttribute(FEATURE_ATTR);
  if (geometry && attr) {
    let vertex: number | undefined;
    if (hit.face) vertex = hit.face.a;
    else if (hit.index !== undefined && hit.index !== null) {
      // Lines/points: `index` is the index-buffer position (indexed) or vertex.
      vertex = geometry.index ? geometry.index.getX(hit.index) : hit.index;
    }
    if (vertex !== undefined) {
      const featureIndex = Math.round(attr.getX(vertex));
      const ids = pickable?.featureIds ?? featureIdsOf(obj);
      const id = ids?.[featureIndex];
      if (typeof id === 'string') return { entityId: id, tileId, object: obj, point: hit.point.clone(), featureIndex };
    }
  }

  // Fallback: an object that is a single entity (e.g. terrain without ids).
  if (tileId) {
    const layer = pickable?.layer;
    if (layer === 'terrain') {
      return { entityId: `terrain:${tileId}`, tileId, object: obj, point: hit.point.clone() };
    }
  }
  return null;
}

/** Raycasting helper with pixel-based thresholds for lines and points. */
export class Picker {
  readonly raycaster = new Raycaster();
  private readonly ndc = new Vector2();
  private readonly camera: Camera & { fov?: number };
  private readonly dom: HTMLElement;

  constructor(camera: Camera, dom: HTMLElement) {
    this.camera = camera;
    this.dom = dom;
  }

  /** Sets the ray from a client (mouse) position. */
  setFromClient(clientX: number, clientY: number): Raycaster {
    const rect = this.dom.getBoundingClientRect();
    this.ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    return this.raycaster;
  }

  /** World-space size of `px` screen pixels at `distance` from the camera. */
  worldSizeOfPixels(px: number, distance: number): number {
    const fov = (this.camera.fov ?? 50) * (Math.PI / 180);
    const h = Math.max(1, this.dom.clientHeight);
    return (2 * Math.max(distance, 1) * Math.tan(fov / 2) * px) / h;
  }

  setThresholds(px: number, distance: number): void {
    const t = this.worldSizeOfPixels(px, distance);
    this.raycaster.params.Line.threshold = t;
    this.raycaster.params.Points.threshold = t;
    // Fat lines (LineSegments2) take an extra pick width in pixels.
    (this.raycaster.params as unknown as { Line2: { threshold: number } }).Line2 = { threshold: px };
  }

  intersect(objects: Object3D[]): Intersection[] {
    return this.raycaster.intersectObjects(objects, false);
  }
}

const HIGHLIGHT_COLOR = new Color(0xffe14d);
const MAX_OVERLAY_TRIANGLES = 600_000;

/**
 * Selection highlight: a translucent overlay of the selected feature's own
 * triangles/segments plus an always-on-top bounding box.
 */
export class Highlighter {
  readonly group = new Group();
  private readonly meshMaterial = new MeshBasicMaterial({
    color: HIGHLIGHT_COLOR,
    transparent: true,
    opacity: 0.45,
    side: DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -4,
    polygonOffsetUnits: -4,
  });
  private readonly lineMaterial = new LineBasicMaterial({ color: HIGHLIGHT_COLOR, depthTest: false, transparent: true });
  private readonly instanceMaterial = new MeshBasicMaterial({
    color: HIGHLIGHT_COLOR,
    wireframe: true,
    depthTest: false,
    transparent: true,
  });

  constructor() {
    this.group.name = 'selection_highlight';
    this.group.renderOrder = 10;
  }

  clear(): void {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      if (child instanceof Mesh || child instanceof LineSegments) {
        // Instance overlays share the source geometry: only dispose our own.
        if (child.userData.ownsGeometry) child.geometry.dispose();
      }
      if (child instanceof Box3Helper) {
        child.geometry.dispose();
        (child.material as LineBasicMaterial).dispose();
      }
    }
  }

  /**
   * Highlights every occurrence of an entity. Falls back to a small box around
   * `centroid` (ENU) when the entity is not drawn in any loaded tile.
   * Returns the world-space bounding box of the highlight (or null).
   */
  show(occurrences: EntityOccurrence[], centroid?: [number, number, number] | null): Box3 | null {
    this.clear();
    const box = new Box3();
    for (const occ of occurrences) this.addOccurrence(occ, box);
    if (box.isEmpty() && centroid && centroid.every(Number.isFinite)) {
      const c = enuToThree(centroid[0], centroid[1], centroid[2]);
      box.setFromCenterAndSize(c, new Vector3(6, 6, 6));
    }
    if (box.isEmpty()) return null;
    const padded = box.clone().expandByScalar(0.5);
    const helper = new Box3Helper(padded, HIGHLIGHT_COLOR);
    const mat = helper.material as LineBasicMaterial;
    mat.depthTest = false;
    mat.transparent = true;
    helper.renderOrder = 11;
    this.group.add(helper);
    return box;
  }

  /** Highlights a world-space box only (e.g. for an issue location). */
  showBox(box: Box3): void {
    this.clear();
    const helper = new Box3Helper(box.clone(), HIGHLIGHT_COLOR);
    (helper.material as LineBasicMaterial).depthTest = false;
    this.group.add(helper);
  }

  private addOccurrence(occ: EntityOccurrence, box: Box3): void {
    const obj = occ.pickable.object;
    obj.updateWorldMatrix(true, false);
    const geometry = obj.geometry as BufferGeometry;

    if (occ.instanceIndex !== undefined && obj instanceof InstancedMesh) {
      const m = new Matrix4();
      obj.getMatrixAt(occ.instanceIndex, m);
      m.premultiply(obj.matrixWorld);
      if (!geometry.boundingBox) geometry.computeBoundingBox();
      box.union(geometry.boundingBox!.clone().applyMatrix4(m));
      const overlay = new Mesh(geometry, this.instanceMaterial);
      overlay.matrixAutoUpdate = false;
      overlay.matrix.copy(m);
      overlay.matrixWorldNeedsUpdate = true;
      overlay.renderOrder = 10;
      this.group.add(overlay);
      return;
    }

    if (occ.featureIndex === undefined) return;
    const attr = geometry.getAttribute(FEATURE_ATTR);
    const pos = geometry.getAttribute('position');
    if (!attr || !pos) return;
    if (!(obj instanceof Mesh) && !(obj instanceof Line)) return; // points: box only below
    const k = occ.featureIndex;
    const index = geometry.index;
    const count = index ? index.count : pos.count;
    const isLine = obj instanceof Line;
    // Vertices per primitive and step between primitives: triangles (3, 3),
    // LINES segments (2, 2), line strips (2, 1).
    const verts = isLine ? 2 : 3;
    const step = obj instanceof LineSegments ? 2 : isLine ? 1 : 3;
    const out: number[] = [];
    const v = new Vector3();
    const mw = obj.matrixWorld;
    let prims = 0;
    for (let i = 0; i + verts - 1 < count; i += step) {
      const a = index ? index.getX(i) : i;
      if (Math.round(attr.getX(a)) !== k) continue;
      prims++;
      for (let j = 0; j < verts; j++) {
        const vi = index ? index.getX(i + j) : i + j;
        v.fromBufferAttribute(pos, vi).applyMatrix4(mw);
        box.expandByPoint(v);
        if (prims <= MAX_OVERLAY_TRIANGLES) out.push(v.x, v.y, v.z);
      }
    }
    if (out.length === 0) return;
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(out, 3));
    const overlay = isLine ? new LineSegments(g, this.lineMaterial) : new Mesh(g, this.meshMaterial);
    overlay.userData.ownsGeometry = true;
    overlay.renderOrder = 10;
    this.group.add(overlay);
  }
}
