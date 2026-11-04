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
import type { Material, Object3D } from 'three';
import { COLOR_MODES, ColorModeManager } from './colorModes';
import type { ColorMode } from './colorModes';
import { CameraRig, isTyping } from './controls';
import { GeoFrame, enuToThree, formatLatLon, formatMetres, parseLatLon, threeToEnu } from './geo';
import { HttpError, TileLoader, checkSchema, errorText, fetchJson, fetchOptionalJson, resolveWorldBase, worldUrl } from './loader';
import { Highlighter, Picker, resolveEntityHit } from './picking';
import type { EntityHit } from './picking';
import { RouteView } from './routeView';
import { Router } from './routing';
import type { TravelMode } from './routing';
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
  TileIndex,
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
import { ValidationOverlay, hasLocation, renderIssueList } from './validation';
import type { IssueFilter } from './validation';
import { WorldModel } from './world';

const SKY = '#b9d3ea';
const DEFAULT_BOUNDS: LocalBounds = { xmin: -500, ymin: -500, xmax: 500, ymax: 500, zmin: 0, zmax: 50 };

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

    this.rig = new CameraRig(this.camera, this.renderer.domElement);
    this.rig.probe = (origin, dir) => this.probeRay(origin, dir);
    this.rig.onModeChange = (mode) => this.onCameraMode(mode);
    this.picker = new Picker(this.camera, this.renderer.domElement);

    this.buildSidebar();
    this.status.setHint('Click to inspect · F toggles fly mode');
    this.info.onClose = () => this.clearSelection();
    this.toasts.onLog = (e) => this.addLoadMessage(`${e.level}: ${e.text}`);

    this.bindEvents();
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
    }
    const m = this.manifest;
    this.geo = new GeoFrame(m);
    if (m && this.geo.warning) this.toasts.show(this.geo.warning, 'warn');
    this.colors = new ColorModeManager((m?.sources ?? []).map((s) => s.name).filter(Boolean));
    this.renderWorldInfo();

    // Tile index (attempted even without a manifest).
    let index: TileIndex | null = null;
    const indexPath = m?.tiling?.index ?? 'tiles/index.json';
    try {
      index = await fetchJson<TileIndex>(worldUrl(this.base, indexPath));
      for (const w of checkSchema(index, 'worldgen.tile_index', indexPath)) this.toasts.show(w, 'warn');
      if (!Array.isArray(index.tiles)) throw new Error(`${indexPath} has no "tiles" array`);
    } catch (e) {
      index = null;
      if (m) this.toasts.show(`Tile index unavailable: ${errorText(e)}`, 'error');
    }

    this.bounds = m?.local_bounds ?? boundsFromIndex(index) ?? DEFAULT_BOUNDS;
    this.setupFraming();

    if (m || index) {
      void this.loadGraph();
      void this.loadDiagnostics();
    }

    if (!index || index.tiles.length === 0) {
      this.progress.hide();
      if (index) this.toasts.show('The tile index lists no tiles.', 'warn');
      return;
    }
    await this.loadTiles(index);
  }

  private setupFraming(): void {
    const b = this.bounds;
    this.rig.setBounds(b);
    this.rig.frame(b);
    const extent = Math.max(100, b.xmax - b.xmin, b.ymax - b.ymin);
    const center = enuToThree((b.xmin + b.xmax) / 2, (b.ymin + b.ymax) / 2, b.zmin ?? 0);
    this.sun.position.copy(center).add(new Vector3(-0.5 * extent, 1.1 * extent, 0.6 * extent));
    this.sun.target.position.copy(center);
    const cam = this.sun.shadow.camera;
    cam.left = cam.bottom = -extent * 0.75;
    cam.right = cam.top = extent * 0.75;
    cam.near = 1;
    cam.far = extent * 4;
    cam.updateProjectionMatrix();
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0005;
    this.fog.near = extent * 1.2;
    this.fog.far = extent * 6;
  }

  private async loadTiles(index: TileIndex): Promise<void> {
    const loader = new TileLoader(this.base, 4);
    const b = this.bounds;
    const focus = { x: (b.xmin + b.xmax) / 2, y: (b.ymin + b.ymax) / 2 };
    let loadedTiles = 0;
    let metaFailures = 0;
    const tileErrors: string[] = [];
    await loader.loadAll(index.tiles, focus, {
      onTile: (entry, scene) => {
        this.world.addTile(entry, scene);
        if (this.shadows) applyShadows(scene, true);
        if (this.colors.mode !== 'materials') this.colors.refreshTile(entry.id, this.world);
        if (++loadedTiles % 4 === 1) this.renderLayerList();
      },
      onMeta: (entry, meta, error) => {
        this.world.setMeta(entry, meta, error);
        if (error) {
          metaFailures++;
          this.addLoadMessage(`meta ${entry.id}: ${error}`);
        }
        if (meta && this.colors.mode !== 'materials') this.colors.refreshTile(entry.id, this.world);
        if (meta && this.selectedEntity && meta.entities[this.selectedEntity]) this.refreshSelectedEntity();
      },
      onProgress: (p) => {
        const doneCount = p.loaded + p.failed;
        const failed = p.failed ? ` · ${p.failed} failed` : '';
        this.progress.set(p.total ? doneCount / p.total : 1, `Loading tiles ${doneCount}/${p.total}${failed}`);
      },
      onError: (_entry, message) => {
        tileErrors.push(message);
        this.addLoadMessage(message);
      },
    });
    this.progress.set(1, `Loaded ${loadedTiles}/${index.tiles.length} tiles`);
    this.progress.hide(1500);
    this.renderLayerList();
    if (tileErrors.length) {
      this.toasts.show(
        `${tileErrors.length} of ${index.tiles.length} tiles failed to load (see World → Messages).`,
        tileErrors.length === index.tiles.length ? 'error' : 'warn',
      );
    }
    if (metaFailures) this.toasts.show(`${metaFailures} tile meta.json files are missing or invalid; inspection is limited there.`, 'warn');
    // Markers without z can now sit on the terrain.
    if (this.validation) this.applyIssueFilter();
  }

  private async loadGraph(): Promise<void> {
    const path = this.manifest?.files?.graph ?? 'graph/road_graph.json';
    try {
      const graph = await fetchOptionalJson<RoadGraph>(worldUrl(this.base, path));
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
      this.validation = new ValidationOverlay(diag, (x, y) => this.groundZAt(x, y));
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
    view.append(
      h('div', { class: 'btn-row' }, this.modeButton, frameBtn),
      checkbox('Fog', false, (v) => {
        this.scene.fog = v ? this.fog : null;
      }),
      checkbox('Shadows', false, (v) => this.setShadows(v)),
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
      const count = this.world.layerNodes.get(name)?.length ?? 0;
      const label = checkbox(name, this.world.isLayerVisible(name), (v) => this.world.setLayerVisible(name, v), h('span', { class: 'count' }, count ? `${count} tiles` : '—'));
      if (!count) label.classList.add('empty');
      list.append(label);
    }
  }

  private renderColorLegend(): void {
    renderLegend(this.colorLegend, this.colors.legend());
    if (this.colors.mode !== 'materials') {
      this.colorLegend.append(
        h('div', { class: 'muted small' }, 'Colours come from tile meta.json (no metadata = grey). Terrain is drawn muted.'),
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
    this.rig.update(dt);
    this.routeView.updateMarkerScale(this.rig.viewDistance());
    this.renderer.render(this.scene, this.camera);
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

  /** Terrain (or any surface) elevation at local x, y; null when nothing is there. */
  private groundZAt(x: number, y: number): number | null {
    const top = (this.bounds.zmax ?? 1000) + 1000;
    const origin = enuToThree(x, y, top);
    let objects: Object3D[] = this.world.pickables.filter((p) => p.layer === 'terrain' && p.object instanceof Mesh).map((p) => p.object);
    if (!objects.length) objects = this.meshPickables(['trees', 'vegetation_regions']);
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
    const tiles = entry?.tiles ?? (hit?.tileId ? [hit.tileId] : []);
    this.info.showEntity({
      id,
      record,
      tiles,
      layer: hit ? this.world.pickableFor(hit.object)?.layer : occurrences[0]?.pickable.layer,
      latlon,
      instanceIndex: hit?.instanceIndex,
      onEntityLink: (other) => this.selectById(other),
    });
    if (opts.fly && box) this.flyToBox(box);
  }

  private refreshSelectedEntity(): void {
    const id = this.selectedEntity;
    if (id) this.selectEntity(id);
  }

  /** Selects anything by id: tile entity, graph edge or node. */
  private selectById(id: string): void {
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
    this.toasts.show(`"${id}" is not in any loaded tile or in the road graph.`, 'warn');
  }

  private selectGraph(pick: GraphPick, fly: boolean): void {
    if (!this.topology) return;
    this.highlighter.clear();
    this.selectedEntity = null;
    const onLink = (id: string): void => this.selectById(id);
    if (pick.type === 'edge') {
      const e = pick.edge;
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
    this.info.showIssue(issue, latlon, (id) => this.selectById(id), flyFn);
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
    const z = this.groundZAt(p.x, p.y) ?? b.zmin ?? 0;
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
