/**
 * WorldGen viewer entry point: wires loading, scene, camera, overlays and UI.
 */
import {
  Box3,
  Color,
  DirectionalLight,
  Fog,
  HemisphereLight,
  Mesh,
  PerspectiveCamera,
  Plane,
  Scene,
  Vector3,
  WebGLRenderer,
} from 'three';
import type { Group, Material, Object3D } from 'three';
import { FlyBench } from './bench';
import { autoPath, validatePath } from './benchPath';
import type { BenchPath } from './benchPath';
import { COLOR_MODES, ColorModeManager } from './colorModes';
import { PhotorealRenderer } from './render/photoreal';
import type { GroundMode, PhotorealStats } from './render/photoreal';
import { LEVEL_LABELS } from './render/quality';
import type { LevelChoice } from './render/quality';
import type { ColorMode } from './colorModes';
import { CameraRig, isTyping } from './controls';
import { GeoFrame, enuToThree, formatLatLon, formatMetres, parseLatLon, threeToEnu } from './geo';
import { HttpError, checkSchema, errorText, fetchJson, fetchOptionalJson, resolveWorldBase, worldUrl } from './loader';
import { Highlighter, Picker, resolveEntityHit } from './picking';
import type { EntityHit } from './picking';
import { ShadowFitter } from './render/shadowFit';
import { RouteView } from './routeView';
import { Router } from './routing';
import type { TravelMode } from './routing';
import { DEFAULT_GPU_BUDGET, parseBudgetMb } from './stream/cache';
import { MetaCache } from './stream/metaCache';
import type { NearTile } from './stream/metaCache';
import { keyAt, keyId } from './stream/quadtree';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadArchetypes } from './content/archetypes';
import { decodeGraphCore } from './content/graphCore';
import type { GraphStrings } from './content/graphCore';
import { Locator, derivedTile } from './stream/locate';
import type { LocateDoc } from './stream/locate';
import { TileStreamer } from './stream/TileStreamer';
import { fillUri, indexFromTileset, tilesetFromIndex } from './stream/tileset';
import type { TileNode, Tileset } from './stream/tileset';
import type { TraversalResult } from './stream/traversal';
import { TopologyOverlay } from './topology';
import type { GraphPick } from './topology';
import type {
  Diagnostics,
  GraphEdge,
  GraphNode,
  Issue,
  LocalBounds,
  Manifest,
  RoadGraph,
  SubtreeDoc,
  TileEntry,
  TileIndex,
  TileMeta,
  TilesetDoc,
  TurnRestriction,
} from './types';
import {
  InfoPanel,
  ProgressBar,
  StatusBar,
  Toasts,
  appendChildren,
  checkbox,
  clear,
  createSection,
  entityLink,
  h,
  renderLegend,
} from './ui';
import { hasWorldParam, studioInfo, worldPicker, worldSwitcher } from './studio';
import { ValidationOverlay, hasLocation, renderIssueList } from './validation';
import type { IssueFilter } from './validation';
import { WorldModel } from './world';
import type { LoadedTile } from './world';

const SKY = '#b9d3ea';
const PREFS = { render: 'worldgen.viewer.render', graphics: 'worldgen.viewer.graphics', tod: 'worldgen.viewer.timeOfDay', ground: 'worldgen.viewer.ground' };
type RenderMode = 'standard' | 'photo';

function readPref(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writePref(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private mode or blocked storage: preferences are just not remembered.
  }
}
const DEFAULT_BOUNDS: LocalBounds = { xmin: -500, ymin: -500, xmax: 500, ymax: 500, zmin: 0, zmax: 50 };
/** Standard rendering: fixed sun from the south-west, high (three.js axes). */
const STANDARD_SUN_DIR = new Vector3(-0.5, 1.1, 0.6).normalize();
/** Standard rendering: largest half-size (m) of the camera-fitted shadow frustum. */
const STANDARD_SHADOW_HALF = 1500;
/** meta.json is prefetched for tiles this close to the camera (m). */
const META_RADIUS = 300;
/** At most this many meta.json documents are kept. */
const META_CAPACITY = 64;

interface RouteEndpoint {
  x: number;
  y: number;
  nodeId: string | null;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing from index.html`);
  return el as T;
}

class App {
  // DOM
  private readonly viewport = byId<HTMLElement>('viewport');
  private readonly sections = byId<HTMLElement>('sidebar-sections');
  private readonly toasts = new Toasts(byId('toasts'));
  private readonly progress = new ProgressBar(byId('progress'));
  private readonly status = new StatusBar(byId('statusbar'));
  private readonly info = new InfoPanel(byId('info'));
  private readonly overlayMessage = byId<HTMLElement>('overlay-message');
  private readonly toolHint = byId<HTMLElement>('tool-hint');

  // three.js
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly camera = new PerspectiveCamera(55, 1, 0.5, 20000);
  private readonly rig: CameraRig;
  private readonly picker: Picker;
  private readonly hemi = new HemisphereLight(0xe3efff, 0x50473a, 1.7);
  private readonly sun = new DirectionalLight(0xffffff, 2.2);
  private readonly fog = new Fog(SKY, 1000, 8000);

  // world
  private readonly base = resolveWorldBase();
  private manifest: Manifest | null = null;
  private geo = new GeoFrame(null);
  private bounds: LocalBounds = DEFAULT_BOUNDS;
  private readonly world = new WorldModel();
  private readonly photoreal: PhotorealRenderer;
  private readonly highlighter = new Highlighter();
  private colors = new ColorModeManager();
  private topology: TopologyOverlay | null = null;
  private router: Router | null = null;
  private readonly routeView = new RouteView();
  private validation: ValidationOverlay | null = null;
  private readonly incident = new Map<string, GraphEdge[]>();
  private readonly restrictionsByEdge = new Map<string, TurnRestriction[]>();
  private readonly restrictionsByNode = new Map<string, TurnRestriction[]>();
  private readonly loadMessages: string[] = [];
  private tileset: Tileset | null = null;
  private streamer: TileStreamer | null = null;
  /** tiles/tileset.json of a data-first (2.x) world. */
  private tilesetDoc: TilesetDoc | null = null;
  /** Data-first worlds: entity id -> tiles recording it (tiles/locate). */
  private locator: Locator | null = null;
  private metaCache: MetaCache | null = null;
  private readonly standardShadow = new ShadowFitter();
  private bench: FlyBench | null = null;
  private tileErrorCount = 0;
  private tileErrorToastAt = 0;
  private metaFailureShown = false;
  private layerListDirty = false;
  private layerListAt = 0;
  private validationDirtyAt = 0;
  private highlightDirty = false;
  private metaNearAt = 0;
  private streamStatusAt = 0;
  private streamProgressShown = false;

  // state
  private selectedEntity: string | null = null;
  private shadows = false;
  private routeActive = false;
  private routePicking: 'start' | 'end' = 'start';
  private routeStart: RouteEndpoint | null = null;
  private routeEnd: RouteEndpoint | null = null;
  private travelMode: TravelMode = 'motor_vehicle';
  private ignoreOneway = false;
  private issueFilter: IssueFilter = { severities: new Set(['error', 'warning', 'info']), text: '', locatedOnly: false };
  private pointerDown: { x: number; y: number; t: number; button: number } | null = null;
  private hoverPending = false;
  private lastPointer: { x: number; y: number } | null = null;
  private lastFrame = performance.now();
  private fpsEma = 60;
  private fpsShownAt = 0;
  private fogOn = false;

  // UI refs filled by buildSidebar()
  private worldInfoBody!: HTMLElement;
  private messagesBody!: HTMLElement;
  private layerList!: HTMLElement;
  private modeButton!: HTMLButtonElement;
  private topoCheck!: HTMLInputElement;
  private topoNote!: HTMLElement;
  private topoLegend!: HTMLElement;
  private validationCheck!: HTMLInputElement;
  private validationNote!: HTMLElement;
  private routeCheck!: HTMLInputElement;
  private colorLegend!: HTMLElement;
  private routeToolButton!: HTMLButtonElement;
  private routeStartInput!: HTMLInputElement;
  private routeEndInput!: HTMLInputElement;
  private routeResult!: HTMLElement;
  private routeStats!: HTMLElement;
  private validationSummary!: HTMLElement;
  private issueListBody!: HTMLElement;
  private renderSelect!: HTMLSelectElement;
  private graphicsSelect!: HTMLSelectElement;
  private todInput!: HTMLInputElement;
  private groundSelect!: HTMLSelectElement;
  private groundNote!: HTMLElement;
  private todLabel!: HTMLElement;
  private photoNote!: HTMLElement;
  private fogCheck!: HTMLInputElement;
  private shadowCheck!: HTMLInputElement;

  constructor() {
    this.renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = false;
    this.viewport.prepend(this.renderer.domElement);
    this.renderer.domElement.classList.add('gl-canvas');
    this.renderer.domElement.tabIndex = 0;

    this.scene.background = new Color(SKY);
    this.scene.add(this.hemi, this.sun, this.sun.target);
    this.scene.add(this.world.root, this.highlighter.group, this.routeView.group);

    this.photoreal = new PhotorealRenderer(this.renderer, this.scene, this.camera, this.world, this.sun, this.hemi);
    this.photoreal.onStats = (st) => this.showPhotoStats(st);

    this.rig = new CameraRig(this.camera, this.renderer.domElement);
    this.rig.probe = (origin, dir) => this.probeRay(origin, dir);
    this.rig.groundLevel = (x, y) => this.approxGroundZ(x, y);
    this.rig.onModeChange = (mode) => this.onCameraMode(mode);
    this.picker = new Picker(this.camera, this.renderer.domElement);

    this.buildSidebar();
    this.status.setHint('Click to inspect · F toggles fly mode');
    this.info.onClose = () => this.clearSelection();
    this.toasts.onLog = (e) => this.addLoadMessage(`${e.level}: ${e.text}`);

    this.bindEvents();
    this.applyRenderPrefs();
    void this.initStudio();
    new ResizeObserver(() => this.resize()).observe(this.viewport);
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  // ------------------------------------------------------------ loading

  async start(): Promise<void> {
    this.progress.set(0, 'Loading manifest…');
    const manifestUrl = worldUrl(this.base, 'manifest.json');
    try {
      const manifest = await fetchJson<Manifest>(manifestUrl);
      for (const w of checkSchema(manifest, 'worldgen.manifest', 'manifest.json')) this.toasts.show(w, 'warn');
      this.manifest = manifest;
    } catch (e) {
      // Schema-version rejections are not a serving problem; everything else is.
      const servingProblem = e instanceof HttpError || /not valid JSON/.test(errorText(e));
      this.showFatal('Could not load the world manifest', errorText(e), servingProblem);
      if (servingProblem && !hasWorldParam()) void this.offerWorldPicker();
    }
    const m = this.manifest;
    this.geo = new GeoFrame(m);
    if (m && this.geo.warning) this.toasts.show(this.geo.warning, 'warn');
    this.colors = new ColorModeManager((m?.sources ?? []).map((s) => s.name).filter(Boolean));
    const where = m?.region?.center ?? m?.origin ?? null;
    this.photoreal.setLocation(where?.lat ?? null, where?.lon ?? null);
    this.photoNote.textContent = `Building styles: ${this.photoreal.regionLabel}`;
    this.renderWorldInfo();

    // Tile index (attempted even without a manifest): a 2.x tileset with its subtrees, or a 1.x flat index.
    let index: TileIndex | null = null;
    const tilesetPath = m?.tiling?.tileset;
    const indexPath = tilesetPath ?? m?.tiling?.index ?? 'tiles/index.json';
    try {
      if (tilesetPath) {
        const ts = await fetchJson<TilesetDoc>(worldUrl(this.base, tilesetPath));
        for (const w of checkSchema(ts, 'worldgen.tileset', tilesetPath)) this.toasts.show(w, 'warn');
        const subtrees = await Promise.all((ts.subtrees ?? []).map(async (st) => {
          const path = fillUri(ts.uri.subtree, { si: st.si, sj: st.sj });
          const sub = await fetchJson<SubtreeDoc>(worldUrl(this.base, path));
          checkSchema(sub, 'worldgen.tileset_subtree', path);
          return sub;
        }));
        index = indexFromTileset(ts, subtrees);
        this.tilesetDoc = ts;
        if (ts.locate) this.locator = new Locator(ts.locate, (p) => fetchOptionalJson<LocateDoc>(worldUrl(this.base, p)));
      } else {
        index = await fetchJson<TileIndex>(worldUrl(this.base, indexPath));
        for (const w of checkSchema(index, 'worldgen.tile_index', indexPath)) this.toasts.show(w, 'warn');
      }
      if (!Array.isArray(index.tiles)) throw new Error(`${indexPath} has no "tiles" array`);
    } catch (e) {
      index = null;
      if (m) this.toasts.show(`Tile index unavailable: ${errorText(e)}`, 'error');
    }

    this.bounds = m?.local_bounds ?? boundsFromIndex(index) ?? DEFAULT_BOUNDS;
    this.setupFraming();
    this.applyCameraParam();

    if (m || index) {
      void this.loadGraph();
      void this.loadDiagnostics();
    }

    if (!index || index.tiles.length === 0) {
      this.progress.hide();
      if (index) this.toasts.show('The tile index lists no tiles.', 'warn');
      return;
    }
    this.startStreaming(index);
  }

  private setupFraming(): void {
    const b = this.bounds;
    this.rig.setBounds(b);
    this.rig.frame(b);
    this.configureStandardSun();
    this.photoreal.setBounds(b);
  }

  /**
   * `?cam=x,y,z,distance,yaw,pitch`: look at local ENU (x, y, z) from `distance`
   * metres, heading `yaw` degrees (0 = from the south, 90 = from the east) and
   * `pitch` degrees above the horizon. Reproducible views for reviews and bug reports.
   */
  private applyCameraParam(): void {
    const raw = new URLSearchParams(window.location.search).get('cam');
    if (!raw) return;
    const v = raw.split(',').map(Number);
    if (v.length < 3 || v.slice(0, 3).some((n) => !Number.isFinite(n))) {
      this.toasts.show('Ignoring ?cam: expected x,y,z[,distance,yaw,pitch].', 'warn');
      return;
    }
    const [x, y, z, dist = 150, yaw = 0, pitch = 35] = v;
    const target = enuToThree(x, y, z);
    const yr = (yaw * Math.PI) / 180;
    const pr = (pitch * Math.PI) / 180;
    const offset = new Vector3(Math.sin(yr) * Math.cos(pr), Math.sin(pr), Math.cos(yr) * Math.cos(pr)).multiplyScalar(dist);
    this.rig.map.target.copy(target);
    this.camera.position.copy(target).add(offset);
    this.camera.lookAt(target);
    this.rig.map.update();
  }

  /**
   * Fixed sun of the standard rendering. Its shadow frustum follows what the
   * camera looks at (fitStandardShadow), whatever the size of the world.
   */
  private configureStandardSun(): void {
    const focus = this.rig.focus;
    this.sun.target.position.copy(focus);
    this.sun.target.updateMatrixWorld();
    this.sun.position.copy(focus).addScaledVector(STANDARD_SUN_DIR, 2000);
    this.sun.updateMatrixWorld();
    if (this.sun.shadow.mapSize.x !== 4096) {
      this.sun.shadow.mapSize.set(4096, 4096);
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
    }
    this.sun.shadow.radius = 1;
    this.standardShadow.invalidate();
  }

  private fitStandardShadow(): void {
    const zSpan = (this.bounds.zmax ?? 0) - (this.bounds.zmin ?? 0);
    this.standardShadow.fit(this.sun, STANDARD_SUN_DIR, this.rig.focus, this.rig.viewDistance(), {
      maxHalf: STANDARD_SHADOW_HALF,
      mapSize: this.sun.shadow.mapSize.x,
      zSpan,
    });
  }

  /** Starts streaming the world's tiles for the camera (see stream/TileStreamer). */
  private startStreaming(index: TileIndex): void {
    this.photoreal.setOrthoSource(this.base, index.ortho);
    if (!index.ortho) {
      // Worlds built without imagery: landcover colours only.
      this.groundSelect.value = 'procedural';
      this.groundSelect.disabled = true;
      this.groundNote.textContent = 'This world has no satellite imagery (build with sources.imagery enabled).';
      this.groundNote.hidden = false;
      this.photoreal.setGround('procedural');
    }
    const b = this.bounds;
    const tileset = tilesetFromIndex(index, [b.zmin ?? 0, b.zmax ?? 50]);
    this.tileset = tileset;
    if (tileset.skipped) this.addLoadMessage(`${tileset.skipped} tile index entries were skipped (duplicate or invalid).`);
    const params = new URLSearchParams(window.location.search);
    const rawBudget = params.get('gpuBudget');
    const budget = parseBudgetMb(rawBudget) ?? DEFAULT_GPU_BUDGET;
    if (rawBudget && parseBudgetMb(rawBudget) === null) this.toasts.show(`Ignoring ?gpuBudget=${rawBudget}: expected megabytes, e.g. 768 or 1.5GB.`, 'warn');

    this.metaCache = new MetaCache({
      capacity: META_CAPACITY,
      load: (entry, signal) => this.fetchMeta(entry, signal),
      onLoaded: (id, meta, error) => this.onMetaLoaded(id, meta, error),
      onEvicted: (id) => this.onMetaEvicted(id),
    });
    const ts = this.tilesetDoc;
    this.streamer = new TileStreamer({
      base: this.base,
      tileset,
      dataWorld: ts ? {
        archetypes: ts.archetypes
          ? loadArchetypes(new GLTFLoader(), worldUrl(this.base, ts.archetypes)).catch((e: unknown) => {
            this.addLoadMessage(`Tree archetypes unavailable: ${errorText(e)}`);
            return null;
          })
          : undefined,
        trees: ts.trees
          ? { cellSize: ts.trees.cell_size, seed: ts.trees.seed, archetypes: ts.trees.archetypes, categories: ts.trees.categories }
          : undefined,
        buildingMaterials: ts.buildings?.materials,
      } : undefined,
      camera: this.camera,
      viewportHeight: () => this.renderer.domElement.clientHeight,
      budgetBytes: budget,
      addContent: (node, scene) => this.addTile(node, scene),
      removeContent: (_node, content) => this.removeTile(content as LoadedTile),
      onError: (_node, message) => this.onTileError(message),
      onTraversal: (res) => this.afterTraversal(res),
    });
    this.photoreal.onTileBytes = (id, bytes) => this.streamer?.addBytes(id, bytes);
    this.progress.set(0, `Streaming ${tileset.contentNodes.length} tiles…`);
    this.streamProgressShown = true;
    void this.startBenchIfRequested(params);
  }

  /** A parsed tile GLB joins the scene. */
  private addTile(node: TileNode, scene: Group): LoadedTile {
    const entry = node.content!.entry;
    const tile = this.world.addTile(entry, scene);
    if (this.shadows && !this.photoreal.enabled) applyShadows(scene, true);
    this.photoreal.onTileAdded(tile);
    if (this.colors.mode !== 'materials') this.colors.refreshTile(entry.id, this.world);
    this.layerListDirty = true;
    if (this.validation && !this.validationDirtyAt && this.validation.needsGroundIn(entry.bounds)) {
      // Markers there can sit on the real terrain now.
      this.validationDirtyAt = performance.now() + 400;
    }
    if (this.selectedEntity && tile.entityIds.has(this.selectedEntity)) this.highlightDirty = true;
    return tile;
  }

  /** An evicted tile leaves the scene and frees its GPU memory. */
  private removeTile(tile: LoadedTile): void {
    const selectedHere = !!this.selectedEntity && tile.entityIds.has(this.selectedEntity);
    // Highlight overlays of instanced features share the tile's geometry.
    if (selectedHere) this.highlighter.clear();
    this.colors.forgetTile(tile);
    this.photoreal.onTileRemoved(tile);
    this.world.removeTile(tile.entry.id);
    tile.dispose();
    this.layerListDirty = true;
    if (selectedHere) this.highlightDirty = true;
  }

  private onTileError(message: string): void {
    this.tileErrorCount++;
    this.addLoadMessage(message);
    const now = performance.now();
    if (now - this.tileErrorToastAt > 5000) {
      this.tileErrorToastAt = now;
      this.toasts.show(`${this.tileErrorCount} tile(s) failed to load (see World → Messages).`, 'warn');
    }
  }

  private async fetchMeta(entry: TileEntry, signal: AbortSignal): Promise<TileMeta | null> {
    const path = entry.meta!;
    const meta = await fetchOptionalJson<TileMeta>(worldUrl(this.base, path), signal);
    if (!meta) return null;
    checkSchema(meta, 'worldgen.tile_meta', path);
    if (!meta.entities || typeof meta.entities !== 'object') meta.entities = {};
    return meta;
  }

  private onMetaLoaded(tileId: string, meta: TileMeta | null, error: string | null): void {
    this.world.setMeta(tileId, meta, error);
    if (error) {
      this.addLoadMessage(`meta ${tileId}: ${error}`);
      if (!this.metaFailureShown) {
        this.metaFailureShown = true;
        this.toasts.show('Some tile meta.json files are missing or invalid; inspection is limited there (see World → Messages).', 'warn');
      }
    }
    if (!meta) return;
    if (this.world.tiles.has(tileId)) {
      this.photoreal.onMetaLoaded(tileId);
      if (this.colors.mode !== 'materials') this.colors.refreshTile(tileId, this.world);
    }
    if (this.selectedEntity && meta.entities[this.selectedEntity]) this.refreshSelectedEntity();
  }

  private onMetaEvicted(tileId: string): void {
    this.world.dropMeta(tileId);
    if (this.colors.mode !== 'materials' && this.world.tiles.has(tileId)) this.colors.refreshTile(tileId, this.world);
  }

  /** Loads the meta.json of these tiles (if not loaded yet); resolves when all are there. */
  private ensureMeta(tileIds: readonly string[]): Promise<void> {
    const cache = this.metaCache;
    const ts = this.tileset;
    if (!cache || !ts) return Promise.resolve();
    const pending: Promise<unknown>[] = [];
    for (const id of new Set(tileIds)) {
      const entry = ts.byContentId.get(id)?.content?.entry;
      if (entry && !this.world.hasMeta(id)) pending.push(cache.require(entry));
    }
    return Promise.all(pending).then(() => undefined);
  }

  /** After each tile selection: meta.json for the tiles near the camera (or every drawn tile in colour modes). */
  private afterTraversal(res: TraversalResult): void {
    const now = performance.now();
    if (!this.metaCache || now - this.metaNearAt < 250) return;
    this.metaNearAt = now;
    const colourMode = this.colors.mode !== 'materials';
    const near: NearTile[] = [];
    for (const w of res.wanted) {
      const c = w.node.content;
      if (!c) continue;
      if (w.distance <= META_RADIUS || (colourMode && w.ready && w.tier <= 1)) near.push({ entry: c.entry, distance: w.distance });
    }
    near.sort((a, b) => a.distance - b.distance);
    this.metaCache.setNear(near.slice(0, Math.max(8, this.metaCache.capacity - 8)));
  }

  /** Index tiles with the same key as local (x, y) and its neighbours (for entity lookups). */
  private tileIdsAround(x: number, y: number, ring = 0): string[] {
    const ts = this.tileset;
    if (!ts) return [];
    const k = keyAt(x, y, ts.tileSize);
    const out: string[] = [];
    for (let di = -ring; di <= ring; di++) {
      for (let dj = -ring; dj <= ring; dj++) {
        const id = ts.nodes.get(keyId({ level: 0, i: k.i + di, j: k.j + dj }))?.content?.id;
        if (id) out.push(id);
      }
    }
    return out;
  }

  /** Tiles around the given tiles (3 × 3 each). */
  private neighbourTiles(tileIds: readonly string[]): string[] {
    const ts = this.tileset;
    if (!ts) return [...tileIds];
    const out = new Set<string>();
    for (const id of tileIds) {
      const b = ts.byContentId.get(id)?.bounds;
      if (!b) continue;
      for (const t of this.tileIdsAround((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1)) out.add(t);
    }
    return [...out];
  }

  private async startBenchIfRequested(params: URLSearchParams): Promise<void> {
    const raw = params.get('bench');
    const streamer = this.streamer;
    const ts = this.tileset;
    if (!raw || !streamer || !ts) return;
    let path: BenchPath;
    try {
      if (raw === 'auto' || raw === '1') {
        const b = this.bounds;
        const duration = Number(params.get('benchDuration'));
        path = autoPath(
          {
            bounds: [b.xmin, b.ymin, b.xmax, b.ymax],
            topAt: (x, y) => this.zRangeNear(x, y, 300)[1],
            groundAt: (x, y) => this.zRangeNear(x, y, 0)[0],
          },
          Number.isFinite(duration) && duration > 0 ? duration : undefined,
        );
      } else {
        path = validatePath(await fetchJson<unknown>(new URL(raw, window.location.href).href));
      }
    } catch (e) {
      this.toasts.show(`Benchmark path unavailable: ${errorText(e)}`, 'error');
      return;
    }
    this.toasts.show(`Benchmark "${path.name ?? raw}" running: results on window.worldgenBench.`, 'info', 4000);
    this.bench = new FlyBench(path, {
      camera: this.camera,
      rig: this.rig,
      renderer: this.renderer,
      streamer,
      gpu: this.photoreal.gpu,
      resources: () => ({
        streamerLoaded: streamer.stats().loaded,
        worldTiles: this.world.tiles.size,
        worldPickables: this.world.pickables.length,
        metaDocs: this.metaCache?.size ?? 0,
        ...this.photoreal.resourceCounts(),
      }),
      onDone: (r) => {
        const s = r.flight?.frameMs;
        this.toasts.show(`Benchmark done${s ? `: p95 frame ${s.p95} ms` : ''} · peak GPU ${formatMb(r.peak.gpuBytes)} (window.worldgenBench).`, 'info', 8000);
      },
    });
  }

  /** z range [min, max] of the index tiles within `radius` m of local (x, y). */
  private zRangeNear(x: number, y: number, radius: number): [number, number] {
    const ts = this.tileset;
    let lo = Infinity;
    let hi = -Infinity;
    if (ts) {
      const ring = Math.ceil(radius / ts.tileSize);
      for (const id of this.tileIdsAround(x, y, ring)) {
        const n = ts.byContentId.get(id)!;
        lo = Math.min(lo, n.zmin);
        hi = Math.max(hi, n.zmax);
      }
    }
    if (!Number.isFinite(lo)) return [this.bounds.zmin ?? 0, this.bounds.zmax ?? 50];
    return [lo, hi];
  }

  /** Ground elevation from the tile index (lowest point of the tile at x, y), null outside the world. */
  private approxGroundZ(x: number, y: number): number | null {
    const ts = this.tileset;
    if (!ts) return null;
    const n = ts.nodes.get(keyId(keyAt(x, y, ts.tileSize)));
    return n?.content ? n.zmin : null;
  }

  /** Deferred work after tiles came or went (at most a few times per second). */
  private flushDeferred(now: number): void {
    if (this.layerListDirty && now - this.layerListAt > 500) {
      this.layerListDirty = false;
      this.layerListAt = now;
      this.renderLayerList();
    }
    if (this.validationDirtyAt && now >= this.validationDirtyAt) {
      this.validationDirtyAt = 0;
      this.validation?.applyFilter(this.issueFilter);
    }
    if (this.highlightDirty) {
      this.highlightDirty = false;
      this.refreshHighlight();
    }
  }

  private updateStreamStatus(now: number): void {
    const streamer = this.streamer;
    if (!streamer || now - this.streamStatusAt < 250) return;
    this.streamStatusAt = now;
    const st = streamer.stats();
    const busy = !st.settled;
    const el = this.status.tiles;
    el.textContent = `${st.rendered} tiles · ${formatMb(st.gpuBytes)}`;
    el.classList.toggle('over-budget', st.gpuBytes > st.budgetBytes);
    el.title = [
      `Tiles drawn: ${st.rendered} (${st.loaded} loaded of ${st.total})`,
      `Loading: ${st.loading} · queued: ${st.queued} · uploading: ${st.uploading}${st.failed ? ` · failed: ${st.failed}` : ''}`,
      `GPU memory: ${formatMb(st.gpuBytes)} of ${formatMb(st.budgetBytes)} budget (?gpuBudget=<MB>)`,
      `Holes: ${st.holes} · evicted: ${st.evicted} · aborted: ${st.aborted}`,
      `meta.json documents: ${this.metaCache?.size ?? 0} of ${this.metaCache?.capacity ?? 0}`,
    ].join('\n');
    if (busy) {
      const want = Math.max(1, st.selected);
      this.progress.set(Math.min(1, st.rendered / want), `Streaming tiles · ${st.rendered}/${st.selected} in view · ${st.loading + st.queued} loading · ${formatMb(st.gpuBytes)}`);
      this.streamProgressShown = true;
    } else if (this.streamProgressShown) {
      this.streamProgressShown = false;
      this.progress.set(1, `${st.rendered} tiles in view · ${formatMb(st.gpuBytes)} of ${formatMb(st.budgetBytes)}`);
      this.progress.hide(1200);
    }
  }

  /** The binary graph of a data-first world (graph/core.wgt + strings.json), null when absent. */
  private async fetchGraphCore(corePath: string, stringsPath: string): Promise<RoadGraph | null> {
    const url = worldUrl(this.base, corePath);
    const res = await fetch(url, { cache: 'no-cache' });
    if (res.status === 404 || res.status === 410) return null;
    if (!res.ok) throw new HttpError(res.status, url, `HTTP ${res.status} for ${url}`);
    const [core, strings] = await Promise.all([res.arrayBuffer(), fetchJson<GraphStrings>(worldUrl(this.base, stringsPath))]);
    return decodeGraphCore(core, strings);
  }

  private async loadGraph(): Promise<void> {
    const files = this.manifest?.files;
    let path = files?.graph ?? 'graph/road_graph.json';
    try {
      let graph: RoadGraph | null = null;
      if (files?.graph_core && files.graph_strings) {
        graph = await this.fetchGraphCore(files.graph_core, files.graph_strings);
        if (graph) path = files.graph_core;
      }
      graph ??= await fetchOptionalJson<RoadGraph>(worldUrl(this.base, path));
      if (!graph) {
        this.setOverlayNote(this.topoCheck, this.topoNote, `${path} not found — topology and routing unavailable.`);
        this.routeStats.textContent = 'No road graph: routing unavailable.';
        return;
      }
      for (const w of checkSchema(graph, 'worldgen.road_graph', path)) this.toasts.show(w, 'warn');
      if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) throw new Error(`${path} lacks nodes/edges arrays`);
      this.topology = new TopologyOverlay(graph);
      this.topology.visible = this.topoCheck.checked;
      this.scene.add(this.topology.group);
      this.router = new Router(graph);
      this.indexGraph(graph);
      this.topoCheck.disabled = false;
      this.topoNote.textContent = `${graph.nodes.length} nodes · ${graph.edges.length} edges · ${this.topology.onewayCount} one-way`;
      const legend = this.topology.legend();
      clear(this.topoLegend);
      const edgesBox = h('div');
      const nodesBox = h('div');
      renderLegend(edgesBox, legend.edges);
      renderLegend(nodesBox, legend.nodes);
      this.topoLegend.append(h('div', { class: 'legend-title' }, 'Edges'), edgesBox, h('div', { class: 'legend-title' }, 'Nodes'), nodesBox);
      const s = this.router.stats;
      this.routeStats.textContent =
        `Graph: ${s.nodes} nodes, ${s.edges} edges. Turn restrictions: ${s.restrictionsEnforced} enforced` +
        (s.restrictionsNotEnforced ? `, ${s.restrictionsNotEnforced} marked not enforceable` : '') +
        (s.restrictionsSkipped ? `, ${s.restrictionsSkipped} skipped` : '') +
        '.';
      this.routeToolButton.disabled = false;
      for (const r of s.skippedReasons.slice(0, 50)) this.addLoadMessage(`routing: ${r}`);
    } catch (e) {
      this.setOverlayNote(this.topoCheck, this.topoNote, `Road graph unavailable: ${errorText(e)}`);
      this.routeStats.textContent = 'No road graph: routing unavailable.';
      this.toasts.show(`Road graph could not be loaded: ${errorText(e)}`, 'warn');
    }
  }

  private indexGraph(graph: RoadGraph): void {
    for (const e of graph.edges) {
      for (const n of [e.from, e.to]) {
        const list = this.incident.get(n);
        if (list) {
          if (!list.includes(e)) list.push(e);
        } else this.incident.set(n, [e]);
      }
    }
    for (const r of graph.restrictions ?? []) {
      for (const id of [r.from_edge, r.to_edge, ...(r.via_edges ?? [])]) {
        const list = this.restrictionsByEdge.get(id) ?? [];
        list.push(r);
        this.restrictionsByEdge.set(id, list);
      }
      const list = this.restrictionsByNode.get(r.via_node) ?? [];
      list.push(r);
      this.restrictionsByNode.set(r.via_node, list);
    }
  }

  private async loadDiagnostics(): Promise<void> {
    const path = this.manifest?.files?.diagnostics ?? this.manifest?.validation?.report ?? 'diagnostics.json';
    try {
      const diag = await fetchOptionalJson<Diagnostics>(worldUrl(this.base, path));
      if (!diag) {
        this.setOverlayNote(this.validationCheck, this.validationNote, `${path} not found — no validation report.`);
        this.validationSummary.textContent = 'No validation report.';
        return;
      }
      for (const w of checkSchema(diag, 'worldgen.diagnostics', path)) this.toasts.show(w, 'warn');
      this.validation = new ValidationOverlay(
        diag,
        (x, y) => this.groundZAt(x, y),
        (x, y) => this.approxGroundZ(x, y),
      );
      this.validation.visible = this.validationCheck.checked;
      this.scene.add(this.validation.group);
      this.validationCheck.disabled = false;
      const c = this.validation.counts();
      const located = this.validation.issues.filter(hasLocation).length;
      this.validationNote.textContent = `${located} of ${this.validation.issues.length} issues have a location`;
      clear(this.validationSummary);
      this.validationSummary.append(
        h('span', { class: 'sev sev-error' }, `${c.error} errors`),
        h('span', { class: 'sev sev-warning' }, `${c.warning} warnings`),
        h('span', { class: 'sev sev-info' }, `${c.info} info`),
      );
      const checks = diag.summary?.checks;
      if (checks && Object.keys(checks).length) {
        const t = h('table', { class: 'kv small' });
        for (const [name, v] of Object.entries(checks)) {
          t.append(h('tr', null, h('th', null, name), h('td', null, `${v?.status ?? '?'}${v?.issues ? ` (${v.issues})` : ''}`)));
        }
        this.validationSummary.append(h('details', { class: 'checks' }, h('summary', null, 'Checks'), t));
      }
      this.applyIssueFilter();
      if (c.error > 0) this.toasts.show(`Validation report: ${c.error} errors, ${c.warning} warnings.`, 'warn');
    } catch (e) {
      this.setOverlayNote(this.validationCheck, this.validationNote, `Validation report unavailable: ${errorText(e)}`);
      this.validationSummary.textContent = 'Validation report could not be loaded.';
      this.toasts.show(`diagnostics could not be loaded: ${errorText(e)}`, 'warn');
    }
  }

  // ------------------------------------------------------------ sidebar

  private buildSidebar(): void {
    const s = this.sections;

    // World info + attribution
    this.worldInfoBody = createSection(s, 'World', true);
    this.worldInfoBody.append(h('p', { class: 'muted' }, `Loading ${this.base.href} …`));

    // View / camera
    const view = createSection(s, 'View', true);
    this.modeButton = h('button', { class: 'btn', title: 'Toggle orbit / fly camera (F)' }, 'Switch to fly mode (F)');
    this.modeButton.addEventListener('click', () => this.rig.toggleMode());
    const frameBtn = h('button', { class: 'btn', title: 'Frame the whole world' }, 'Frame world');
    frameBtn.addEventListener('click', () => this.rig.frame(this.bounds));
    this.renderSelect = h('select', { class: 'select', title: 'Rendering style' });
    this.renderSelect.append(h('option', { value: 'standard' }, 'Standard'), h('option', { value: 'photo' }, 'Photo-realistic'));
    this.renderSelect.addEventListener('change', () => this.setRenderMode(this.renderSelect.value as RenderMode));
    this.graphicsSelect = h('select', { class: 'select', title: 'Graphics level. The frame rate is held above 45 fps at every level by lowering the resolution, then expensive effects.' });
    for (const c of ['auto', 'low', 'medium', 'high', 'ultra'] as const) this.graphicsSelect.append(h('option', { value: c }, LEVEL_LABELS[c]));
    this.graphicsSelect.addEventListener('change', () => {
      const v = this.graphicsSelect.value as LevelChoice;
      writePref(PREFS.graphics, v);
      this.photoreal.setLevel(v);
    });
    this.todInput = h('input', { type: 'range', min: '4.5', max: '20.5', step: '0.25', value: '10.5', class: 'range', title: 'Local solar time' });
    this.todLabel = h('span', { class: 'range-value mono' }, '10:30');
    this.todInput.addEventListener('input', () => {
      const v = Number(this.todInput.value);
      this.todLabel.textContent = formatHour(v);
      this.photoreal.setTimeOfDay(v);
      writePref(PREFS.tod, String(v));
    });
    this.groundSelect = h('select', { class: 'select', title: 'What the ground colour comes from, under the procedural detail' });
    this.groundSelect.append(h('option', { value: 'satellite' }, 'Satellite imagery'), h('option', { value: 'procedural' }, 'Landcover colours'));
    this.groundSelect.addEventListener('change', () => {
      const v = this.groundSelect.value as GroundMode;
      writePref(PREFS.ground, v);
      this.photoreal.setGround(v);
    });
    this.groundNote = h('div', { class: 'note small photo-only' });
    this.groundNote.hidden = true;
    this.photoNote = h('div', { class: 'note small' }, '');
    const fogLabel = checkbox('Fog', false, (v) => {
      this.fogOn = v;
      if (!this.photoreal.enabled) this.scene.fog = v ? this.fog : null;
    });
    this.fogCheck = fogLabel.querySelector('input')!;
    const shadowLabel = checkbox('Shadows', false, (v) => this.setShadows(v));
    this.shadowCheck = shadowLabel.querySelector('input')!;
    view.append(
      h('div', { class: 'btn-row' }, this.modeButton, frameBtn),
      h('label', { class: 'field' }, h('span', null, 'Render'), this.renderSelect),
      h('label', { class: 'field photo-only' }, h('span', null, 'Graphics'), this.graphicsSelect),
      h('label', { class: 'field photo-only' }, h('span', null, 'Time'), this.todInput, this.todLabel),
      h('label', { class: 'field photo-only' }, h('span', null, 'Ground'), this.groundSelect),
      this.groundNote,
      this.photoNote,
      fogLabel,
      shadowLabel,
      h(
        'div',
        { class: 'help small' },
        h('div', null, h('b', null, 'Orbit: '), 'left-drag pan · right-drag rotate · wheel zoom'),
        h('div', null, h('b', null, 'Fly: '), 'WASD move · Q/E down/up · Shift faster · drag to look · wheel = speed'),
        h('div', null, h('b', null, 'Keys: '), 'F fly/orbit · Esc clear selection / stop tool'),
      ),
    );

    // Go to
    const goto = createSection(s, 'Go to', true);
    const gotoInput = h('input', { type: 'text', placeholder: 'lat, lon  e.g. 25.0330, 121.5654', class: 'text' });
    const gotoBtn = h('button', { class: 'btn' }, 'Go');
    const doGoto = (): void => this.goTo(gotoInput.value);
    gotoBtn.addEventListener('click', doGoto);
    gotoInput.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') doGoto();
    });
    goto.append(h('div', { class: 'input-row' }, gotoInput, gotoBtn));

    // Layers
    const layers = createSection(s, 'Layers', true);
    this.layerList = h('div', { class: 'layer-list' });
    layers.append(this.layerList);
    this.renderLayerList();

    // Overlays
    const overlays = createSection(s, 'Overlays', true);
    this.topoNote = h('div', { class: 'note small' }, 'Loading road graph…');
    this.topoLegend = h('div', { class: 'legend' });
    const topoLabel = checkbox('Road topology graph', false, (v) => {
      if (this.topology) this.topology.visible = v;
      this.topoLegend.hidden = !v;
    });
    this.topoCheck = topoLabel.querySelector('input')!;
    this.topoCheck.disabled = true;
    this.topoLegend.hidden = true;
    const nodesLabel = checkbox('Graph nodes', true, (v) => this.topology?.setNodesVisible(v));
    nodesLabel.classList.add('indent');

    this.validationNote = h('div', { class: 'note small' }, 'Loading validation report…');
    const valLabel = checkbox('Validation markers', true, (v) => {
      if (this.validation) this.validation.visible = v;
    });
    this.validationCheck = valLabel.querySelector('input')!;
    this.validationCheck.disabled = true;

    const routeLabel = checkbox('Route', true, (v) => {
      this.routeView.group.visible = v;
    });
    this.routeCheck = routeLabel.querySelector('input')!;
    overlays.append(topoLabel, nodesLabel, this.topoNote, this.topoLegend, valLabel, this.validationNote, routeLabel);

    // Colour mode
    const color = createSection(s, 'Colour mode', true);
    const select = h('select', { class: 'select' });
    for (const m of COLOR_MODES) select.append(h('option', { value: m.id }, m.label));
    select.addEventListener('change', () => this.setColorMode(select.value as ColorMode));
    this.colorLegend = h('div', { class: 'legend' });
    color.append(select, this.colorLegend);

    // Route
    const route = createSection(s, 'Route', false);
    const modeSelect = h('select', { class: 'select' });
    for (const [id, label] of [
      ['motor_vehicle', 'Motor vehicle'],
      ['bicycle', 'Bicycle'],
      ['foot', 'Foot'],
    ] as const) {
      modeSelect.append(h('option', { value: id }, label));
    }
    modeSelect.addEventListener('change', () => {
      this.travelMode = modeSelect.value as TravelMode;
      this.recomputeRoute();
    });
    this.routeToolButton = h('button', { class: 'btn', title: 'Click start then end point in the 3D view' }, 'Pick on map');
    this.routeToolButton.disabled = true;
    this.routeToolButton.addEventListener('click', () => this.setRouteTool(!this.routeActive));
    const clearBtn = h('button', { class: 'btn' }, 'Clear');
    clearBtn.addEventListener('click', () => this.clearRoute());
    this.routeStartInput = h('input', { type: 'text', class: 'text', placeholder: 'start lat, lon' });
    this.routeEndInput = h('input', { type: 'text', class: 'text', placeholder: 'end lat, lon' });
    const goBtn = h('button', { class: 'btn' }, 'Route');
    goBtn.addEventListener('click', () => this.routeFromInputs());
    for (const inp of [this.routeStartInput, this.routeEndInput]) {
      inp.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') this.routeFromInputs();
      });
    }
    this.routeResult = h('div', { class: 'route-result' });
    this.routeStats = h('div', { class: 'note small' }, 'Loading road graph…');
    route.append(
      h('label', { class: 'field' }, h('span', null, 'Mode'), modeSelect),
      checkbox('Ignore one-way restrictions', false, (v) => {
        this.ignoreOneway = v;
        this.recomputeRoute();
      }),
      h('div', { class: 'btn-row' }, this.routeToolButton, clearBtn),
      this.routeStartInput,
      this.routeEndInput,
      h('div', { class: 'btn-row' }, goBtn),
      this.routeResult,
      this.routeStats,
    );

    // Validation
    const val = createSection(s, 'Validation', true);
    this.validationSummary = h('div', { class: 'val-summary' }, 'Loading…');
    const sevRow = h('div', { class: 'sev-filters' });
    for (const sev of ['error', 'warning', 'info']) {
      const cb = checkbox(sev, true, (v) => {
        if (v) this.issueFilter.severities.add(sev);
        else this.issueFilter.severities.delete(sev);
        this.applyIssueFilter();
      });
      cb.classList.add(`sev-${sev}`);
      sevRow.append(cb);
    }
    const search = h('input', { type: 'search', class: 'text', placeholder: 'Filter issues (check, message, id)…' });
    let t = 0;
    search.addEventListener('input', () => {
      window.clearTimeout(t);
      t = window.setTimeout(() => {
        this.issueFilter.text = search.value.trim();
        this.applyIssueFilter();
      }, 150);
    });
    const located = checkbox('Only issues with a location', false, (v) => {
      this.issueFilter.locatedOnly = v;
      this.applyIssueFilter();
    });
    this.issueListBody = h('div', { class: 'issue-list-wrap' });
    val.append(this.validationSummary, sevRow, located, search, this.issueListBody);

    // Messages (load problems), collapsed by default
    this.messagesBody = createSection(s, 'Messages', false);
    this.messagesBody.append(h('p', { class: 'muted small' }, 'No problems so far.'));

    this.renderColorLegend();
  }

  private renderWorldInfo(): void {
    const body = this.worldInfoBody;
    clear(body);
    const m = this.manifest;
    if (!m) {
      body.append(
        h('p', { class: 'warn-text' }, 'No manifest loaded.'),
        h('p', { class: 'small mono wrap' }, this.base.href),
      );
      this.status.attribution.textContent = 'Attribution unavailable (no manifest)';
      return;
    }
    const rows: [string, string][] = [];
    const r = m.region;
    if (r?.name) rows.push(['Region', String(r.name)]);
    if (r?.center) rows.push(['Centre', `${formatLatLon(r.center, 5)}${r.radius_m ? ` · r ${formatMetres(r.radius_m)}` : ''}`]);
    if (r?.bbox_wgs84) rows.push(['BBox (W,S,E,N)', r.bbox_wgs84.map((v) => v.toFixed(5)).join(', ')]);
    if (m.origin) rows.push(['Origin', `${formatLatLon(m.origin, 6)}${m.origin.height ? ` · h ${m.origin.height} m` : ''}`]);
    if (m.crs) {
      rows.push(['CRS', `${m.crs.geographic ?? 'EPSG:4326'} → local ${m.crs.local?.method ?? '?'}`]);
      if (m.crs.vertical_datum) rows.push(['Vertical datum', m.crs.vertical_datum]);
    }
    const lb = m.local_bounds;
    if (lb) rows.push(['Local bounds', `x ${lb.xmin}…${lb.xmax}, y ${lb.ymin}…${lb.ymax}${lb.zmin !== undefined ? `, z ${lb.zmin}…${lb.zmax}` : ''}`]);
    if (m.tiling) rows.push(['Tiles', `${m.tiling.tile_count ?? '?'} × ${m.tiling.tile_size ?? '?'} m`]);
    if (m.compiler) rows.push(['Compiler', `${m.compiler.name ?? ''} ${m.compiler.version ?? ''}`.trim()]);
    if (m.build?.created_utc) rows.push(['Built', m.build.created_utc]);
    if (m.validation) rows.push(['Validation', `${m.validation.errors ?? 0} errors, ${m.validation.warnings ?? 0} warnings`]);
    rows.push(['Schema', `${m.schema ?? '?'} ${m.schema_version ?? ''}`]);

    const table = h('table', { class: 'kv small' });
    for (const [k, v] of rows) table.append(h('tr', null, h('th', null, k), h('td', null, v)));
    body.append(table);
    if (m.crs?.local?.proj) {
      body.append(h('details', { class: 'small' }, h('summary', null, 'PROJ string'), h('code', { class: 'mono wrap block' }, m.crs.local.proj)));
    }

    const sources = Array.isArray(m.sources) ? m.sources : [];
    body.append(h('h4', null, `Sources & attribution (${sources.length})`));
    const list = h('ul', { class: 'sources' });
    for (const src of sources) {
      list.append(
        h(
          'li',
          null,
          h('div', null, h('b', null, src.name), ' ', src.status ? h('span', { class: `status status-${src.status}` }, src.status) : null, src.license ? h('span', { class: 'muted' }, ` ${src.license}`) : null),
          src.attribution ? h('div', { class: 'attribution' }, src.attribution) : h('div', { class: 'muted' }, 'no attribution text'),
          src.dataset_version ? h('div', { class: 'muted small' }, `version ${src.dataset_version}`) : null,
        ),
      );
    }
    if (!sources.length) list.append(h('li', { class: 'muted' }, 'No sources listed in the manifest.'));
    body.append(list, h('div', { class: 'small muted wrap' }, `World: ${this.base.href}`));

    const attributions = [...new Set(sources.map((s) => s.attribution).filter((a): a is string => !!a))];
    this.status.attribution.textContent = attributions.length ? attributions.join(' · ') : 'Sources: see World panel';
    this.status.attribution.title = attributions.join('\n');
    document.title = `WorldGen · ${r?.name ?? (r?.center ? formatLatLon(r.center, 3) : 'viewer')}`;
  }

  private renderLayerList(): void {
    const list = this.layerList;
    clear(list);
    const names = this.world.layerNames(this.manifest?.layers ?? []);
    for (const name of names) {
      const count = this.world.layerNodes.get(name)?.size ?? 0;
      const label = checkbox(name, this.world.isLayerVisible(name), (v) => this.world.setLayerVisible(name, v), h('span', { class: 'count' }, count ? `${count} tiles` : '—'));
      if (!count) label.classList.add('empty');
      list.append(label);
    }
  }

  private renderColorLegend(): void {
    renderLegend(this.colorLegend, this.colors.legend());
    if (this.colors.mode !== 'materials') {
      this.colorLegend.append(
        h(
          'div',
          { class: 'muted small' },
          `Colours come from tile meta.json (no metadata = grey), loaded for the ${META_CAPACITY - 8} nearest tiles in view. Terrain is drawn muted.`,
        ),
      );
    }
  }

  private addLoadMessage(text: string): void {
    if (this.loadMessages.length === 0) clear(this.messagesBody);
    this.loadMessages.push(text);
    if (this.loadMessages.length <= 300) this.messagesBody.append(h('div', { class: 'message small' }, text));
  }

  private setOverlayNote(input: HTMLInputElement, note: HTMLElement, text: string): void {
    input.disabled = true;
    input.checked = false;
    note.textContent = text;
    note.classList.add('warn-text');
  }

  private showFatal(title: string, detail: string, showServeHint: boolean): void {
    clear(this.overlayMessage);
    appendChildren(this.overlayMessage, [
      h('h2', null, title),
      h('p', { class: 'mono wrap' }, detail),
      showServeHint
        ? h(
            'div',
            null,
            h('p', null, 'Serve a compiled world, for example:'),
            h('pre', null, 'WORLDGEN_WORLD=../generated/taipei npm run dev'),
            h('p', null, 'or open this page with ', h('code', null, '?world=<url of the world directory>'), '.'),
          )
        : null,
      h('button', { class: 'btn', onclick: () => (this.overlayMessage.hidden = true) }, 'Dismiss'),
    ]);
    this.overlayMessage.hidden = false;
  }

  // ------------------------------------------------------------ worldgen serve integration

  /** With a reachable WorldGen API: a "World" selector above the sidebar sections. */
  private async initStudio(): Promise<void> {
    const info = await studioInfo();
    if (!info) return;
    this.sections.before(worldSwitcher(info, this.base));
  }

  /** The default world failed to load: offer the API's worlds (if any) instead. */
  private async offerWorldPicker(): Promise<void> {
    const info = await studioInfo();
    if (!info || this.overlayMessage.hidden) return;
    clear(this.overlayMessage);
    appendChildren(this.overlayMessage, worldPicker(info, () => (this.overlayMessage.hidden = true)));
    this.overlayMessage.hidden = false;
    this.progress.hide();
  }

  // ------------------------------------------------------------ events

  private bindEvents(): void {
    const canvas = this.renderer.domElement;
    canvas.addEventListener('pointerdown', (ev) => {
      this.pointerDown = { x: ev.clientX, y: ev.clientY, t: performance.now(), button: ev.button };
    });
    canvas.addEventListener('pointerup', (ev) => {
      const d = this.pointerDown;
      this.pointerDown = null;
      if (!d || d.button !== 0 || ev.button !== 0) return;
      if (Math.hypot(ev.clientX - d.x, ev.clientY - d.y) > 5 || performance.now() - d.t > 700) return;
      this.handleClick(ev.clientX, ev.clientY);
    });
    canvas.addEventListener('pointermove', (ev) => {
      this.lastPointer = { x: ev.clientX, y: ev.clientY };
      if (this.hoverPending) return;
      this.hoverPending = true;
      window.setTimeout(() => {
        this.hoverPending = false;
        this.probeHover();
      }, 80);
    });
    canvas.addEventListener('pointerleave', () => {
      this.lastPointer = null;
    });
    canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());
    window.addEventListener('keydown', (ev) => {
      if (isTyping(ev) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (ev.code === 'KeyF') {
        this.rig.toggleMode();
        ev.preventDefault();
      } else if (ev.key === 'Escape') {
        if (this.routeActive) this.setRouteTool(false);
        else this.clearSelection();
      }
    });
  }

  private onCameraMode(mode: 'orbit' | 'fly'): void {
    this.modeButton.textContent = mode === 'fly' ? 'Switch to orbit mode (F)' : 'Switch to fly mode (F)';
    this.status.mode.textContent = mode === 'fly' ? 'Fly' : 'Orbit';
    this.status.mode.classList.toggle('fly', mode === 'fly');
  }

  private resize(): void {
    const w = Math.max(1, this.viewport.clientWidth);
    const hgt = Math.max(1, this.viewport.clientHeight);
    this.renderer.setSize(w, hgt);
    this.camera.aspect = w / hgt;
    this.camera.updateProjectionMatrix();
  }

  private frame(): void {
    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.bench?.beforeFrame(now);
    this.rig.update(dt);
    const range = this.rig.viewRange();
    this.streamer?.update(now);
    this.flushDeferred(now);
    this.routeView.updateMarkerScale(this.rig.viewDistance());
    if (this.photoreal.enabled) {
      this.photoreal.render(this.rig.focus, this.rig.viewDistance(), range);
    } else {
      if (this.fogOn) {
        this.fog.near = range * 0.35;
        this.fog.far = range * 0.95;
      }
      if (this.shadows) this.fitStandardShadow();
      const gpu = this.photoreal.gpu;
      gpu.begin();
      this.renderer.render(this.scene, this.camera);
      gpu.end();
      gpu.poll();
      if (dt > 0) this.fpsEma += (1 / dt - this.fpsEma) * 0.05;
      if (now - this.fpsShownAt > 500) {
        this.fpsShownAt = now;
        this.status.fps.textContent = `${Math.round(this.fpsEma)} fps`;
        this.status.fps.className = 'sb-item sb-fps';
        this.status.fps.title = `Frame rate (standard rendering)${gpu.lastMs !== null ? ` · GPU ${gpu.lastMs.toFixed(1)} ms per frame` : ''}`;
      }
    }
    this.bench?.afterFrame(now, performance.now() - now);
    this.updateStreamStatus(now);
  }

  // ------------------------------------------------------------ render mode

  private applyRenderPrefs(): void {
    const params = new URLSearchParams(window.location.search);
    const mode = (params.get('render') ?? readPref(PREFS.render)) === 'photo' ? 'photo' : 'standard';
    const g = params.get('graphics') ?? readPref(PREFS.graphics) ?? 'auto';
    const choice: LevelChoice = (['auto', 'low', 'medium', 'high', 'ultra'] as const).includes(g as LevelChoice) ? (g as LevelChoice) : 'auto';
    const tod = Number(params.get('tod') ?? readPref(PREFS.tod) ?? 10.5);
    const hour = Number.isFinite(tod) ? Math.min(20.5, Math.max(4.5, tod)) : 10.5;
    this.graphicsSelect.value = choice;
    this.todInput.value = String(hour);
    this.todLabel.textContent = formatHour(hour);
    this.photoreal.setTimeOfDay(hour);
    this.photoreal.setLevel(choice);
    // Satellite imagery is the default ground wherever the world has it.
    const ground: GroundMode = (params.get('ground') ?? readPref(PREFS.ground)) === 'procedural' ? 'procedural' : 'satellite';
    this.groundSelect.value = ground;
    this.photoreal.setGround(ground);
    this.setRenderMode(mode, false);
  }

  private setRenderMode(mode: RenderMode, remember = true): void {
    if (remember) writePref(PREFS.render, mode);
    this.renderSelect.value = mode;
    const photo = mode === 'photo';
    this.sections.classList.toggle('photo-mode', photo);
    this.photoreal.setEnabled(photo);
    this.fogCheck.disabled = photo;
    this.shadowCheck.disabled = photo;
    if (!photo) {
      // Back to the standard look: the checkboxes' state applies again.
      this.scene.fog = this.fogOn ? this.fog : null;
      this.configureStandardSun();
      this.setShadows(this.shadows);
      this.photoNote.hidden = true;
    } else {
      this.photoNote.hidden = false;
    }
  }

  private showPhotoStats(st: PhotorealStats): void {
    const el = this.status.fps;
    el.textContent = this.photoreal.statusText();
    el.className = `sb-item sb-fps${st.degraded ? ' degraded' : ''}${st.fps < 45 ? ' slow' : ''}`;
    const lines = [
      `Photo-realistic · ${LEVEL_LABELS[st.choice]}${st.choice === 'auto' ? ` (${LEVEL_LABELS[st.level]})` : ''}`,
      `Render scale ${st.scale.toFixed(2)}× of the level's resolution`,
      st.gpuMs !== null ? `GPU ${st.gpuMs.toFixed(1)} ms per frame` : 'GPU timing unavailable in this browser',
      st.shed.length ? `Lowered to hold 45 fps: ${st.shed.join(', ')}` : 'All effects of this level are on',
      `Trees: ${st.trees.full} full · ${st.trees.canopy} canopy · ${st.trees.impostor} impostor · ${st.trees.hidden} beyond draw distance`,
    ];
    el.title = lines.join('\n');
  }

  // ------------------------------------------------------------ picking

  private meshPickables(excludeLayers: string[] = []): Object3D[] {
    return this.world
      .visiblePickables(excludeLayers)
      .filter((p) => p.object instanceof Mesh)
      .map((p) => p.object);
  }

  /** First surface hit along a ray (three coords), ignoring trees/vegetation outlines. */
  private probeRay(origin: Vector3, dir: Vector3): Vector3 | null {
    this.picker.raycaster.set(origin, dir.clone().normalize());
    const hit = this.picker.intersect(this.meshPickables(['trees', 'vegetation_regions']))[0];
    return hit ? hit.point.clone() : null;
  }

  /**
   * Terrain (or any surface) elevation at local x, y from the loaded tiles
   * there; null when no tile is loaded at that point.
   */
  private groundZAt(x: number, y: number): number | null {
    const tiles = this.world.tilesAt(x, y);
    if (!tiles.length) return null;
    const top = (this.bounds.zmax ?? 1000) + 1000;
    const origin = enuToThree(x, y, top);
    const pick = (accept: (layer: string) => boolean): Object3D[] =>
      tiles.flatMap((t) => t.pickables.filter((p) => accept(p.layer) && p.object instanceof Mesh).map((p) => p.object));
    let objects = pick((l) => l === 'terrain');
    if (!objects.length) objects = pick((l) => l !== 'trees' && l !== 'vegetation_regions');
    this.picker.raycaster.set(origin, new Vector3(0, -1, 0));
    const hit = this.picker.intersect(objects)[0];
    return hit ? hit.point.y : null;
  }

  /** Point under the cursor: surface hit or the ground plane. */
  private pointUnderCursor(clientX: number, clientY: number): { point: Vector3; onSurface: boolean } | null {
    const ray = this.picker.setFromClient(clientX, clientY);
    const hit = this.picker.intersect(this.meshPickables(['trees', 'vegetation_regions']))[0];
    if (hit) return { point: hit.point.clone(), onSurface: true };
    const plane = new Plane(new Vector3(0, 1, 0), -(this.bounds.zmin ?? 0));
    const p = ray.ray.intersectPlane(plane, new Vector3());
    return p ? { point: p, onSurface: false } : null;
  }

  private probeHover(): void {
    if (!this.lastPointer) return;
    const res = this.pointUnderCursor(this.lastPointer.x, this.lastPointer.y);
    if (!res) {
      this.status.setCursor(null, null, null, null);
      return;
    }
    const enu = threeToEnu(res.point);
    this.status.setCursor(this.geo.toLatLon(enu.x, enu.y), enu.x, enu.y, res.onSurface ? enu.z : null, this.manifest?.crs?.vertical_datum ?? '');
  }

  private handleClick(clientX: number, clientY: number): void {
    if (this.routeActive) {
      this.handleRouteClick(clientX, clientY);
      return;
    }
    this.picker.setFromClient(clientX, clientY);
    const viewDist = this.rig.viewDistance();
    this.picker.setThresholds(4, viewDist);
    const worldHits = this.picker.intersect(this.world.visiblePickables().map((p) => p.object));
    const first = worldHits[0];
    this.picker.setThresholds(8, first?.distance ?? viewDist);

    const issueHit = this.validation?.pick(this.picker.raycaster);
    if (issueHit) {
      this.selectIssue(issueHit.issue, false);
      return;
    }
    const graphHit = this.topology?.pick(this.picker.raycaster) ?? null;
    if (graphHit && (!first || graphHit.distance <= first.distance + Math.max(3, first.distance * 0.03))) {
      this.selectGraph(graphHit, false);
      return;
    }
    if (first) {
      const res = resolveEntityHit(first, this.world);
      if (res) {
        this.selectEntity(res.entityId, { hit: res, fly: false });
        return;
      }
      this.highlighter.clear();
      this.info.show('Unidentified object', h('p', { class: 'muted' }, `This ${first.object.type} carries no feature id (no _FEATURE_ID_0 / feature_ids).`));
      return;
    }
    this.clearSelection();
  }

  // ------------------------------------------------------------ selection

  private clearSelection(): void {
    this.selectedEntity = null;
    this.highlighter.clear();
    this.topology?.clearHighlight();
    this.info.hide();
  }

  private selectEntity(id: string, opts: { hit?: EntityHit; fly?: boolean } = {}): void {
    const hit = opts.hit;
    const entry = this.world.getEntity(id);
    const record = this.world.recordFor(id, hit?.tileId ?? undefined);
    let occurrences = this.world.occurrencesOf(id);
    if (hit?.instanceIndex !== undefined) {
      occurrences = occurrences.filter((o) => o.pickable.object === hit.object && o.instanceIndex === hit.instanceIndex);
    }
    this.topology?.clearHighlight();
    const box = this.highlighter.show(occurrences, record?.centroid ?? null);
    this.selectedEntity = id;
    const centroid = record?.centroid;
    const latlon = centroid ? this.geo.toLatLon(centroid[0], centroid[1]) : hit ? this.latLonOfThree(hit.point) : null;
    const drawnIn = [...new Set(occurrences.map((o) => o.pickable.tileId))];
    const tiles = entry?.tiles ?? (hit?.tileId ? [hit.tileId] : drawnIn);
    this.info.showEntity({
      id,
      record,
      tiles,
      layer: hit ? this.world.pickableFor(hit.object)?.layer : occurrences[0]?.pickable.layer,
      latlon,
      instanceIndex: hit?.instanceIndex,
      onEntityLink: (other) => this.selectById(other, this.neighbourTiles(tiles)),
    });
    if (opts.fly && box) this.flyToBox(box);
    // meta.json is loaded lazily: fetch it for the tiles that draw this feature.
    const metaTiles = [...new Set([...(hit?.tileId ? [hit.tileId] : []), ...drawnIn, ...tiles])];
    this.metaCache?.setPinned(metaTiles);
    if (!record && metaTiles.some((t) => !this.world.hasMeta(t))) void this.ensureMeta(metaTiles);
  }

  private refreshSelectedEntity(): void {
    const id = this.selectedEntity;
    if (id) this.selectEntity(id);
  }

  /** Re-draws the selection highlight after tiles drawing it came or went. */
  private refreshHighlight(): void {
    const id = this.selectedEntity;
    if (!id) return;
    this.highlighter.show(this.world.occurrencesOf(id), this.world.recordFor(id)?.centroid ?? null);
  }

  /**
   * Selects anything by id: tile entity, graph edge or node. `hintTiles` are
   * tiles whose meta.json may describe the entity when it is not loaded yet.
   */
  private selectById(id: string, hintTiles: readonly string[] = [], located = false): void {
    if (this.world.getEntity(id) || this.world.occurrencesOf(id).length) {
      this.selectEntity(id, { fly: true });
      return;
    }
    const edge = this.topology?.edgeById.get(id);
    if (edge) {
      this.selectGraph({ type: 'edge', edge, distance: 0, point: new Vector3() }, true);
      return;
    }
    const node = this.topology?.nodeById.get(id);
    if (node) {
      this.selectGraph({ type: 'node', node, distance: 0, point: new Vector3() }, true);
      return;
    }
    const roadEdges = this.topology ? this.topology.graph.edges.filter((e) => e.road_id === id) : [];
    if (roadEdges.length && this.topology) {
      this.highlighter.clear();
      this.topology.highlightEdges(roadEdges);
      if (!this.topology.visible) this.toasts.show('Enable the road topology overlay to see the highlighted edges.', 'info');
      this.info.show(
        'Road (graph edges)',
        h('div', { class: 'entity-id mono' }, id),
        h('p', { class: 'muted' }, 'Not present in any loaded tile; showing its graph edges.'),
        h('ul', { class: 'link-list' }, ...roadEdges.map((e) => h('li', null, entityLink(e.id, (x) => this.selectById(x))))),
      );
      return;
    }
    const unloaded = hintTiles.filter((t) => this.tileset?.byContentId.has(t) && !this.world.hasMeta(t));
    if (unloaded.length) {
      void this.ensureMeta(unloaded).then(() => this.selectById(id, [], located));
      return;
    }
    if (!located && this.locator && this.tileset) {
      // Not in any loaded tile: ask the locator which tiles record it, load their meta.json.
      const derived = derivedTile(id, this.tileset.tileSize, this.tilesetDoc?.trees?.cell_size);
      void this.locator.tilesOf(id).then((tiles) => this.selectById(id, derived ? [...tiles, derived] : tiles, true));
      return;
    }
    this.toasts.show(`"${id}" is not in any loaded tile or in the road graph.`, 'warn');
  }

  private selectGraph(pick: GraphPick, fly: boolean): void {
    if (!this.topology) return;
    this.highlighter.clear();
    this.selectedEntity = null;
    if (pick.type === 'edge') {
      const e = pick.edge;
      const onLink = (id: string): void => this.selectById(id, e.tiles ?? []);
      this.topology.highlightEdges([e]);
      this.info.showEdge(e, this.restrictionsByEdge.get(e.id) ?? [], onLink);
      if (fly) {
        const pts = (e.geometry && e.geometry.length ? e.geometry : [e.from, e.to].map((n) => this.topology!.nodeById.get(n)).filter((n): n is GraphNode => !!n).map((n) => [n.x, n.y, n.z ?? 0] as [number, number, number]));
        const box = new Box3();
        for (const p of pts) box.expandByPoint(enuToThree(p[0], p[1], p[2] ?? 0));
        if (!box.isEmpty()) this.flyToBox(box);
      }
    } else {
      const n = pick.node;
      const onLink = (id: string): void => this.selectById(id, this.tileIdsAround(n.x, n.y, 1));
      this.topology.highlightNode(n);
      this.info.showNode(n, this.incident.get(n.id) ?? [], this.restrictionsByNode.get(n.id) ?? [], this.geo.toLatLon(n.x, n.y), onLink);
      if (fly) this.rig.flyTo(enuToThree(n.x, n.y, n.z ?? 0), 120);
    }
  }

  private selectIssue(issue: Issue, fly: boolean): void {
    this.topology?.clearHighlight();
    this.selectedEntity = null;
    const loc = hasLocation(issue) ? issue.location! : null;
    let target: Vector3 | null = null;
    if (loc) {
      const z = this.validation?.markerZ(issue) ?? loc.z ?? 0;
      target = enuToThree(loc.x, loc.y, z);
      this.highlighter.showBox(new Box3().setFromCenterAndSize(target, new Vector3(8, 8, 8)));
    } else {
      this.highlighter.clear();
    }
    const latlon = loc ? (typeof loc.lat === 'number' && typeof loc.lon === 'number' ? { lat: loc.lat, lon: loc.lon } : this.geo.toLatLon(loc.x, loc.y)) : null;
    const flyFn = target ? () => this.rig.flyTo(target!, 120) : null;
    const hints = [...(issue.tile ? [issue.tile] : []), ...(loc ? this.tileIdsAround(loc.x, loc.y, 1) : [])];
    this.info.showIssue(issue, latlon, (id) => this.selectById(id, hints), flyFn);
    if (fly && flyFn) flyFn();
    if (fly && !loc) this.toasts.show('This issue has no location.', 'info', 3000);
  }

  private flyToBox(box: Box3): void {
    const size = box.getSize(new Vector3()).length();
    this.rig.flyTo(box.getCenter(new Vector3()), Math.max(40, size * 1.6));
  }

  private latLonOfThree(p: Vector3) {
    const e = threeToEnu(p);
    return this.geo.toLatLon(e.x, e.y);
  }

  // ------------------------------------------------------------ colours, layers, view

  private setColorMode(mode: ColorMode): void {
    this.colors.setMode(mode, this.world);
    this.renderColorLegend();
    if (mode !== 'materials' && this.world.entityCount === 0) {
      this.toasts.show('No tile metadata loaded yet: features will be grey.', 'warn');
    }
  }

  private setShadows(on: boolean): void {
    this.shadows = on;
    if (this.photoreal.enabled) return;
    this.renderer.shadowMap.enabled = on;
    this.sun.castShadow = on;
    applyShadows(this.world.root, on);
    this.scene.traverse((o) => {
      const mat = (o as Mesh).material as Material | Material[] | undefined;
      if (!mat) return;
      for (const m of Array.isArray(mat) ? mat : [mat]) m.needsUpdate = true;
    });
  }

  private goTo(text: string): void {
    const ll = parseLatLon(text);
    if (!ll) {
      this.toasts.show('Enter coordinates as "lat, lon" in decimal degrees.', 'warn');
      return;
    }
    const p = this.geo.toLocal(ll.lat, ll.lon);
    if (!p) {
      this.toasts.show('Geographic conversion unavailable (manifest has no CRS).', 'error');
      return;
    }
    const b = this.bounds;
    if (p.x < b.xmin || p.x > b.xmax || p.y < b.ymin || p.y > b.ymax) {
      this.toasts.show(`${formatLatLon(ll)} is outside the compiled area.`, 'warn');
    }
    const z = this.groundZAt(p.x, p.y) ?? this.approxGroundZ(p.x, p.y) ?? b.zmin ?? 0;
    this.rig.flyTo(enuToThree(p.x, p.y, z), 250);
  }

  // ------------------------------------------------------------ validation list

  private applyIssueFilter(): void {
    if (!this.validation) return;
    this.validation.applyFilter(this.issueFilter);
    renderIssueList(this.issueListBody, this.validation.issues, this.issueFilter, (issue) => this.selectIssue(issue, true));
  }

  // ------------------------------------------------------------ routing

  private setRouteTool(active: boolean): void {
    if (active && !this.router) {
      this.toasts.show('No road graph loaded: routing is unavailable.', 'warn');
      return;
    }
    this.routeActive = active;
    this.routePicking = 'start';
    this.routeToolButton.classList.toggle('active', active);
    this.routeToolButton.textContent = active ? 'Stop picking' : 'Pick on map';
    this.renderer.domElement.classList.toggle('picking', active);
    this.updateToolHint();
  }

  private updateToolHint(): void {
    if (!this.routeActive) {
      this.toolHint.hidden = true;
      this.status.setHint('Click to inspect · F toggles fly mode');
      return;
    }
    const text = this.routePicking === 'start' ? 'Route: click the START point (Esc to stop)' : 'Route: click the END point (Esc to stop)';
    this.toolHint.textContent = text;
    this.toolHint.hidden = false;
    this.status.setHint(text);
  }

  private handleRouteClick(clientX: number, clientY: number): void {
    const res = this.pointUnderCursor(clientX, clientY);
    if (!res) return;
    const enu = threeToEnu(res.point);
    const which = this.routePicking;
    if (which === 'start') {
      this.routeEnd = null;
      this.routeView.setMarker('end', null);
      this.routeView.clearLine();
      this.routeResult.textContent = '';
      this.routeEndInput.value = '';
    }
    if (!this.setEndpoint(which, enu.x, enu.y)) return;
    this.routePicking = which === 'start' ? 'end' : 'start';
    this.updateToolHint();
    if (which === 'end') this.recomputeRoute();
  }

  /** Snaps a local point to the nearest graph node usable by the current mode. */
  private setEndpoint(which: 'start' | 'end', x: number, y: number): boolean {
    if (!this.router || !this.topology) return false;
    const nodeId = this.router.nearestNode(x, y, this.travelMode);
    if (!nodeId) {
      this.routeResult.textContent = `No graph node is usable by ${this.travelMode.replace('_', ' ')}.`;
      return false;
    }
    const endpoint = { x, y, nodeId };
    if (which === 'start') this.routeStart = endpoint;
    else this.routeEnd = endpoint;
    const node = this.topology.nodeById.get(nodeId) ?? null;
    this.routeView.setMarker(which, node);
    if (node) {
      const ll = this.geo.toLatLon(node.x, node.y);
      const input = which === 'start' ? this.routeStartInput : this.routeEndInput;
      if (ll) input.value = formatLatLon(ll);
    }
    return true;
  }

  private routeFromInputs(): void {
    if (!this.router) {
      this.toasts.show('No road graph loaded: routing is unavailable.', 'warn');
      return;
    }
    const a = parseLatLon(this.routeStartInput.value);
    const b = parseLatLon(this.routeEndInput.value);
    if (!a || !b) {
      this.routeResult.textContent = 'Enter start and end as "lat, lon".';
      return;
    }
    const pa = this.geo.toLocal(a.lat, a.lon);
    const pb = this.geo.toLocal(b.lat, b.lon);
    if (!pa || !pb) {
      this.routeResult.textContent = 'Geographic conversion unavailable (manifest has no CRS).';
      return;
    }
    if (this.setEndpoint('start', pa.x, pa.y) && this.setEndpoint('end', pb.x, pb.y)) this.recomputeRoute();
  }

  private recomputeRoute(): void {
    const s = this.routeStart;
    const e = this.routeEnd;
    if (!this.router || !this.topology || !s || !e) return;
    // Re-snap: the usable node set depends on the travel mode.
    if (!this.setEndpoint('start', s.x, s.y) || !this.setEndpoint('end', e.x, e.y)) {
      this.routeView.clearLine();
      return;
    }
    const start = this.routeStart!.nodeId!;
    const end = this.routeEnd!.nodeId!;
    const t0 = performance.now();
    const result = this.router.route(start, end, { mode: this.travelMode, ignoreOneway: this.ignoreOneway });
    const ms = performance.now() - t0;
    clear(this.routeResult);
    if (!result.found) {
      this.routeView.clearLine();
      this.routeResult.append(h('div', { class: 'no-route' }, 'No route'), h('div', { class: 'small muted' }, result.reason ?? ''));
      return;
    }
    this.routeView.showRoute(result, this.topology.edgeById, this.topology.nodeById);
    this.routeView.group.visible = true;
    this.routeCheck.checked = true;
    this.routeResult.append(
      h('div', { class: 'route-length' }, formatMetres(result.length)),
      h('div', { class: 'small' }, `${result.steps.length} edges · ${result.nodeIds.length} nodes · ${ms.toFixed(1)} ms`),
      h('div', { class: 'small muted' }, `${start} → ${end}`),
    );
  }

  private clearRoute(): void {
    this.routeStart = null;
    this.routeEnd = null;
    this.routeView.clear();
    this.routeStartInput.value = '';
    this.routeEndInput.value = '';
    clear(this.routeResult);
    if (this.routeActive) {
      this.routePicking = 'start';
      this.updateToolHint();
    }
  }
}

function formatMb(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(2)} GB` : `${Math.round(mb)} MB`;
}

function formatHour(h: number): string {
  const hh = Math.floor(h);
  const mm = Math.round((h - hh) * 60);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}

function applyShadows(root: Object3D, on: boolean): void {
  root.traverse((o) => {
    if (!(o instanceof Mesh)) return;
    const isTerrain = /terrain|water/.test(o.parent?.userData?.layer ?? o.userData?.layer ?? '');
    o.castShadow = on && !isTerrain;
    o.receiveShadow = on;
  });
}

function boundsFromIndex(index: TileIndex | null): LocalBounds | null {
  if (!index || !Array.isArray(index.tiles) || index.tiles.length === 0) return null;
  const b: LocalBounds = { xmin: Infinity, ymin: Infinity, xmax: -Infinity, ymax: -Infinity, zmin: Infinity, zmax: -Infinity };
  for (const t of index.tiles) {
    if (!t.bounds) continue;
    b.xmin = Math.min(b.xmin, t.bounds[0]);
    b.ymin = Math.min(b.ymin, t.bounds[1]);
    b.xmax = Math.max(b.xmax, t.bounds[2]);
    b.ymax = Math.max(b.ymax, t.bounds[3]);
    if (t.zrange) {
      b.zmin = Math.min(b.zmin!, t.zrange[0]);
      b.zmax = Math.max(b.zmax!, t.zrange[1]);
    }
  }
  if (!Number.isFinite(b.xmin)) return null;
  if (!Number.isFinite(b.zmin!)) {
    b.zmin = 0;
    b.zmax = 50;
  }
  return b;
}

// Expose for debugging in the console (no behaviour depends on it).
declare global {
  interface Window {
    worldgenViewer?: unknown;
  }
}

try {
  const app = new App();
  window.worldgenViewer = app;
  app.start().catch((e) => {
    console.error(e);
    const box = document.getElementById('overlay-message');
    if (box) {
      box.textContent = `Unexpected error: ${errorText(e)}`;
      box.hidden = false;
    }
  });
} catch (e) {
  console.error(e);
  document.body.append(h('div', { class: 'fatal' }, `The viewer could not start: ${errorText(e)}. WebGL may be unavailable.`));
}
