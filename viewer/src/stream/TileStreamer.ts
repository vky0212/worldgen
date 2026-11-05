/**
 * Streams tile content for the camera: traversal → prioritised requests →
 * parse → time-sliced upload → LRU eviction under a GPU memory budget.
 *
 *  - The traversal (stream/traversal.ts) runs every frame while the camera
 *    moves and at 10 Hz otherwise; it decides what is drawn (other loaded
 *    tiles are hidden) and what to load.
 *  - Up to `maxRequests` GLB fetches run at once, nearest first; requests the
 *    camera no longer needs are aborted.
 *  - Parsed tiles are added to the scene within ~4 ms per frame.
 *  - Loaded tiles stay cached (hidden) until the budget is needed: the least
 *    recently used ones that are not drawn, not ancestors of drawn tiles and
 *    not waiting for their siblings are evicted and disposed.
 */
import { Frustum, Matrix4 } from 'three';
import type { Group, PerspectiveCamera } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildWgtScene } from '../content/wgtScene';
import type { WgtWorld } from '../content/wgtScene';
import { HttpError, errorText, worldUrl } from '../loader';
import { DEFAULT_GPU_BUDGET, LruCache } from './cache';
import { disposeTree } from './content';
import type { TileContent } from './content';
import { RequestQueue } from './requestQueue';
import { DEFAULT_TRAVERSAL, traverse } from './traversal';
import type { ContentStatus, Plane, TraversalOptions, TraversalResult, ViewState, Want } from './traversal';
import type { TileNode, Tileset } from './tileset';

export interface TileStreamerOptions {
  base: URL;
  tileset: Tileset;
  camera: PerspectiveCamera;
  /** Viewport height in CSS pixels (for the screen-space error). */
  viewportHeight: () => number;
  /** GPU memory budget (bytes) for tile content. */
  budgetBytes?: number;
  /** Concurrent GLB fetches. */
  maxRequests?: number;
  /** Main-thread time (ms) per frame for adding parsed tiles to the scene. */
  uploadBudgetMs?: number;
  traversal?: Partial<TraversalOptions>;
  /** Registers a parsed tile (world, renderers) and returns its content. */
  addContent(node: TileNode, scene: Group): TileContent;
  /** Unregisters and disposes evicted content. */
  removeContent(node: TileNode, content: TileContent): void;
  onError?(node: TileNode, message: string): void;
  /** After each traversal (meta prefetch, overlays). */
  onTraversal?(result: TraversalResult): void;
  /** World-level inputs of data-first (.wgt) tiles: tree archetypes and parameters. */
  dataWorld?: Omit<WgtWorld, 'tileSize'>;
}

export interface StreamStats {
  /** Content tiles in the tileset. */
  total: number;
  /** Drawn this frame. */
  rendered: number;
  /** Loaded (drawn or cached). */
  loaded: number;
  /** Fetching. */
  loading: number;
  /** Waiting for a fetch slot (or for budget). */
  queued: number;
  /** Parsed, waiting to be added to the scene. */
  uploading: number;
  failed: number;
  /** Visible selected tiles with nothing drawn over them. */
  holes: number;
  /** Visible selected tiles. */
  selected: number;
  gpuBytes: number;
  budgetBytes: number;
  /** Wall time of the last traversal (ms). */
  traversalMs: number;
  traversals: number;
  requested: number;
  aborted: number;
  evicted: number;
  /**
   * Nothing left to load for this view: no fetch or upload in progress and
   * nothing selected missing that the budget would allow loading. Holes can
   * remain only when the budget is exhausted.
   */
  settled: boolean;
}

interface Parsed {
  node: TileNode;
  scene: Group;
}

interface Failure {
  message: string;
  attempts: number;
  /** performance.now() after which the tile may be requested again. */
  retryAt: number;
}

const DEFAULT_AVG_BYTES = 1.5 * 1024 * 1024;
/** A failed tile is retried (2 s, then 4 s later) before it is given up. */
const MAX_ATTEMPTS = 3;
const RETRY_BASE_MS = 2000;

export class TileStreamer {
  readonly tileset: Tileset;
  private readonly opts: TileStreamerOptions;
  private readonly gltf = new GLTFLoader();
  private readonly cache: LruCache<TileContent>;
  private readonly queue: RequestQueue;
  private readonly parsed: Parsed[] = [];
  private readonly parsedIds = new Set<string>();
  private readonly failed = new Map<string, Failure>();
  private readonly uploadBudgetMs: number;
  private readonly traversalOpts: TraversalOptions;
  private prevRefined: ReadonlySet<string> = new Set();
  private rendered = new Set<TileNode>();
  private pinned = new Set<string>();
  private lastResult: TraversalResult | null = null;
  private lastTraversal = -Infinity;
  private dirty = true;
  private readonly lastView = new Float64Array(32);
  private bytesSum = 0;
  private bytesCount = 0;
  private traversalMs = 0;
  private traversals = 0;
  private readonly frustum = new Frustum();
  private readonly projView = new Matrix4();

  constructor(opts: TileStreamerOptions) {
    this.opts = opts;
    this.tileset = opts.tileset;
    this.uploadBudgetMs = opts.uploadBudgetMs ?? 4;
    this.traversalOpts = { ...DEFAULT_TRAVERSAL, ...opts.traversal };
    this.cache = new LruCache<TileContent>(opts.budgetBytes ?? DEFAULT_GPU_BUDGET, (id, content) => this.evict(id, content));
    this.queue = new RequestQueue({
      maxActive: opts.maxRequests ?? 6,
      run: (id, signal) => this.fetchTile(id, signal),
      canStart: () => this.hasRoomForOneMore(),
      onSettled: (id, outcome, error) => {
        if (outcome !== 'error') return;
        const node = this.tileset.nodes.get(id);
        const message = describeLoadError(error);
        // Missing files are final; other errors (network, server busy) are retried.
        const final = error instanceof HttpError && (error.status === 404 || error.status === 410);
        const attempts = final ? MAX_ATTEMPTS : (this.failed.get(id)?.attempts ?? 0) + 1;
        this.failed.set(id, { message, attempts, retryAt: performance.now() + RETRY_BASE_MS * 2 ** (attempts - 1) });
        this.dirty = true;
        const label = `Tile ${node?.content?.id ?? id}: ${message}`;
        if (attempts >= MAX_ATTEMPTS) {
          if (node) this.opts.onError?.(node, label);
        } else {
          console.warn(`[worldgen] ${label} (attempt ${attempts} of ${MAX_ATTEMPTS}, retrying)`);
        }
      },
    });
  }

  get budgetBytes(): number {
    return this.cache.budget;
  }

  /** Loaded content of a node, if any. */
  contentOf(node: TileNode): TileContent | undefined {
    return this.cache.get(node.id);
  }

  /** Nodes drawn in the last traversal. */
  get renderedNodes(): ReadonlySet<TileNode> {
    return this.rendered;
  }

  get lastTraversalResult(): TraversalResult | null {
    return this.lastResult;
  }

  /** GPU memory of a tile changed outside its content (e.g. an orthophoto arrived). */
  addBytes(contentId: string, bytes: number): void {
    const node = this.tileset.byContentId.get(contentId);
    if (!node || !this.cache.has(node.id)) return;
    const content = this.cache.get(node.id)!;
    content.gpuBytes += bytes;
    this.cache.setCost(node.id, content.gpuBytes);
  }

  /** Forces a traversal on the next update (e.g. after the camera jumped). */
  invalidate(): void {
    this.dirty = true;
  }

  /** Per frame: uploads parsed tiles, then re-selects when needed. */
  update(now: number): void {
    if (this.upload()) this.dirty = true;
    this.opts.camera.updateMatrixWorld();
    const moved = this.cameraMoved();
    if (moved || this.dirty || now - this.lastTraversal >= 100) {
      this.lastTraversal = now;
      this.dirty = false;
      this.runTraversal();
    }
  }

  stats(): StreamStats {
    const r = this.lastResult;
    return {
      total: this.tileset.contentNodes.length,
      rendered: this.rendered.size,
      loaded: this.cache.size,
      loading: this.queue.activeCount,
      queued: this.queue.queuedCount,
      uploading: this.parsed.length,
      failed: this.permanentFailures(),
      holes: r?.holes ?? 0,
      selected: r?.selected ?? 0,
      gpuBytes: this.cache.total,
      budgetBytes: this.cache.budget,
      traversalMs: this.traversalMs,
      traversals: this.traversals,
      requested: this.queue.startedCount,
      aborted: this.queue.abortedCount,
      evicted: this.cache.evictedCount,
      settled: !!r && this.queue.activeCount === 0 && this.parsed.length === 0 && !this.wantsMore(r),
    };
  }

  /** Stops all loading and frees every loaded tile. */
  dispose(): void {
    this.queue.clear();
    for (const p of this.parsed) disposeTree(p.scene);
    this.parsed.length = 0;
    this.parsedIds.clear();
    for (const id of this.cache.keys()) {
      const content = this.cache.delete(id);
      const node = this.tileset.nodes.get(id);
      if (content && node) this.opts.removeContent(node, content);
    }
    this.rendered.clear();
  }

  // ------------------------------------------------------------ internals

  private status = (node: TileNode): ContentStatus => {
    if (this.cache.has(node.id)) return 'ready';
    if (this.isFailed(node.id)) return 'failed';
    return 'pending';
  };

  /** Failed and not (yet) due for a retry. */
  private isFailed(id: string): boolean {
    const f = this.failed.get(id);
    return !!f && (f.attempts >= MAX_ATTEMPTS || performance.now() < f.retryAt);
  }

  private permanentFailures(): number {
    let n = 0;
    for (const f of this.failed.values()) if (f.attempts >= MAX_ATTEMPTS) n++;
    return n;
  }

  /** Error messages of the tiles that were given up (content id → message). */
  failures(): Map<string, string> {
    const out = new Map<string, string>();
    for (const [id, f] of this.failed) {
      if (f.attempts >= MAX_ATTEMPTS) out.set(this.tileset.nodes.get(id)?.content?.id ?? id, f.message);
    }
    return out;
  }

  private viewState(): ViewState {
    const cam = this.opts.camera;
    this.projView.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.projView);
    // three (x, y, z) = ENU (x, z, -y): n_t·p_t = n.x·x + n.y·z − n.z·y.
    const planes: Plane[] = this.frustum.planes.map((p) => [p.normal.x, -p.normal.z, p.normal.y, p.constant]);
    const fov = (cam.fov * Math.PI) / 180;
    const h = Math.max(1, this.opts.viewportHeight());
    return {
      position: [cam.position.x, -cam.position.z, cam.position.y],
      planes,
      sseFactor: h / (2 * Math.tan(fov / 2)),
    };
  }

  private cameraMoved(): boolean {
    const cam = this.opts.camera;
    const w = cam.matrixWorld.elements;
    const p = cam.projectionMatrix.elements;
    const last = this.lastView;
    let moved = false;
    for (let k = 0; k < 16; k++) {
      // Translation in metres; rotation/projection terms are unit-scale.
      const tol = k >= 12 && k <= 14 ? 0.05 : 1e-5;
      if (Math.abs(w[k] - last[k]) > tol) moved = true;
      if (Math.abs(p[k] - last[16 + k]) > 1e-6 * Math.max(1, Math.abs(p[k]))) moved = true;
    }
    if (moved) {
      for (let k = 0; k < 16; k++) {
        last[k] = w[k];
        last[16 + k] = p[k];
      }
    }
    return moved;
  }

  private runTraversal(): void {
    const t0 = performance.now();
    const res = traverse(this.tileset.roots, this.viewState(), this.status, this.prevRefined, this.traversalOpts);
    this.prevRefined = res.refined;
    this.lastResult = res;
    this.traversals++;

    // Show exactly the selection.
    const next = new Set(res.render);
    for (const n of this.rendered) {
      if (!next.has(n)) {
        const c = this.cache.get(n.id);
        if (c) c.root.visible = false;
      }
    }
    for (const n of next) {
      const c = this.cache.get(n.id);
      if (c) c.root.visible = true;
    }
    this.rendered = next;

    // Pins: drawn tiles, their ancestors, and visible tiles waiting for siblings.
    const pinned = new Set<string>();
    for (const n of next) {
      pinned.add(n.id);
      for (let p = n.parent; p; p = p.parent) if (p.content) pinned.add(p.id);
    }
    for (const w of res.wanted) {
      if (w.ready && w.tier <= 1) pinned.add(w.node.id);
    }
    this.pinned = pinned;

    // Recency: everything the selection uses, least urgent first so the most urgent ends up newest.
    for (let k = res.wanted.length - 1; k >= 0; k--) if (res.wanted[k].ready) this.cache.touch(res.wanted[k].node.id);
    for (const n of next) this.cache.touch(n.id);

    // Make room for what is about to load, then request it.
    const missing = res.wanted.filter((w) => !w.ready && !this.parsedIds.has(w.node.id) && !this.isFailed(w.node.id));
    if (missing.length) {
      const headroom = Math.min(missing.length, (this.opts.maxRequests ?? 6) + this.parsed.length) * this.avgBytes();
      this.cache.trim((id) => this.pinned.has(id), Math.max(0, this.cache.budget - headroom));
    } else {
      this.cache.trim((id) => this.pinned.has(id));
    }
    this.queue.update(missing.map((w: Want) => ({ id: w.node.id, priority: w.priority })));

    this.traversalMs = performance.now() - t0;
    this.opts.onTraversal?.(res);
  }

  private wantsMore(r: TraversalResult): boolean {
    for (const w of r.wanted) {
      if (w.ready || w.tier > 1) continue;
      const f = this.failed.get(w.node.id);
      if (f) {
        if (f.attempts < MAX_ATTEMPTS) return true; // a retry is due
        continue;
      }
      // Waiting for budget that cannot be freed is "settled" too: nothing will change.
      if (this.hasRoomForOneMore()) return true;
    }
    return false;
  }

  private avgBytes(): number {
    return this.bytesCount ? this.bytesSum / this.bytesCount : DEFAULT_AVG_BYTES;
  }

  private hasRoomForOneMore(): boolean {
    const inFlight = this.queue.activeCount + this.parsed.length + 1;
    return this.cache.total + inFlight * this.avgBytes() <= this.cache.budget;
  }

  private async fetchTile(id: string, signal: AbortSignal): Promise<void> {
    const node = this.tileset.nodes.get(id);
    const entry = node?.content?.entry;
    if (!node || !entry) throw new Error(`unknown tile ${id}`);
    if (!entry.glb) throw new Error('tile index entry has no "glb" path');
    const url = worldUrl(this.opts.base, entry.glb);
    let res: Response;
    try {
      res = await fetch(url, { signal });
    } catch (e) {
      if (signal.aborted) throw e;
      throw new HttpError(0, url, `Network error while fetching ${url}: ${errorText(e)}`);
    }
    if (!res.ok) throw new HttpError(res.status, url, res.status === 404 ? 'GLB not found (404)' : `HTTP ${res.status} for ${url}`);
    const data = await res.arrayBuffer();
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');
    const scene = entry.format === 'wgt'
      ? await buildWgtScene(data, this.gltf, new URL('.', url).href, { tileSize: this.tileset.tileSize, ...this.opts.dataWorld },
        { skirts: node.skirts, covers: node.partial ? this.tileset.leafAt : null })
      : ((await this.gltf.parseAsync(data, new URL('.', url).href)).scene as Group);
    if (signal.aborted) {
      disposeTree(scene);
      throw new DOMException('aborted', 'AbortError');
    }
    this.parsed.push({ node, scene });
    this.parsedIds.add(id);
  }

  /** Adds parsed tiles to the scene within the per-frame budget; true when any was added. */
  private upload(): boolean {
    if (!this.parsed.length) return false;
    const t0 = performance.now();
    let added = 0;
    while (this.parsed.length) {
      if (added > 0 && performance.now() - t0 >= this.uploadBudgetMs) break;
      const { node, scene } = this.parsed.shift()!;
      this.parsedIds.delete(node.id);
      this.failed.delete(node.id);
      if (this.cache.has(node.id)) {
        disposeTree(scene);
        continue;
      }
      const content = this.opts.addContent(node, scene);
      content.root.visible = false;
      this.cache.set(node.id, content, content.gpuBytes);
      this.bytesSum += content.gpuBytes;
      this.bytesCount++;
      added++;
    }
    return added > 0;
  }

  /** LRU eviction (never a drawn tile: those are pinned). */
  private evict(id: string, content: TileContent): void {
    const node = this.tileset.nodes.get(id);
    if (node) this.opts.removeContent(node, content);
  }
}

function describeLoadError(e: unknown): string {
  const msg = errorText(e);
  if (/responded with 404/.test(msg)) return 'GLB not found (404)';
  return msg;
}
