/**
 * Generate page: pick a region on a map, preview its tile plan, start a
 * WorldGen build and follow its progress live (docs/studio-api.md).
 */
import 'leaflet/dist/leaflet.css';
import '../style.css';
import './generate.css';
import * as L from 'leaflet';
import { ApiError, api, errorMessage, isUnreachable, viewerHref } from '../api';
import type { Health, Job, Plan, Profile, RegionInfo, RoadSurface, WorldInfo } from '../apiTypes';
import { checkbox, clear, createSection, h } from '../dom';
import { JobTileLayer, PlanPreviewLayer, TileLegend, WorldsLayer, createMap } from './mapLayers';
import { isTerminal, tileRings } from './progress';
import type { ProgressState } from './progress';
import { ProgressView } from './progressView';
import {
  DEFAULT_BUDGET,
  DEFAULT_RADIUS_M,
  FALLBACK_MAX_RADIUS_M,
  LARGE_TILE_COUNT,
  MIN_RADIUS_M,
  PRESETS,
  jobBody,
  nameProblem,
  normalizeBbox,
  planParams,
  regionCenter,
  regionFromRequest,
  regionFromWorld,
  regionKey,
  regionProblem,
  suggestName,
} from './region';
import type { Region } from './region';
import { RegionEditor } from './regionEditor';
import { JobSession } from './session';
import type { ConnectionStatus } from './session';

const LS_JOB = 'worldgen.generate.activeJob';
const LS_REGION = 'worldgen.generate.region';
const LS_DARK = 'worldgen.generate.darkMap';
const POLL_MS = 5000;

// ---------------------------------------------------------------- storage (may throw: private mode, blocked)

let storageOk = true;
function storageGet(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    storageOk = false;
    return null;
  }
}
function storageSet(key: string, value: string | null): void {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    storageOk = false;
  }
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing from generate.html`);
  return el as T;
}

type PlanStatus =
  | { kind: 'idle' }
  | { kind: 'loading'; key: string; previous: Plan | null }
  | { kind: 'ok'; key: string; plan: Plan }
  | { kind: 'error'; key: string; status: number; message: string };

interface Place {
  label: string;
  lat: number;
  lon: number;
}

class GeneratePage {
  private readonly sections = byId<HTMLElement>('gen-sections');
  private readonly banner = byId<HTMLElement>('api-banner');
  private readonly mapHint = byId<HTMLElement>('map-hint');
  private readonly toasts = byId<HTMLElement>('toasts');

  private readonly map: L.Map;
  private readonly editor: RegionEditor;
  private readonly preview = new PlanPreviewLayer();
  private readonly jobTiles = new JobTileLayer();
  private readonly jobOutline = L.layerGroup();
  private readonly worldsLayer = new WorldsLayer();
  private readonly legend = new TileLegend();
  private readonly progress: ProgressView;

  private health: Health | null = null;
  private apiUp = false;
  private apiChecked = false;
  private profiles: Profile[] = [];
  private worlds: WorldInfo[] = [];
  private jobs: Job[] = [];
  private plan: PlanStatus = { kind: 'idle' };
  private planTimer = 0;
  private planAbort: AbortController | null = null;
  private maxRadius = FALLBACK_MAX_RADIUS_M;
  private place: Place | null = null;
  private nameTouched = false;
  private session: JobSession | null = null;
  private submitting = false;
  private submitError = '';
  private jobGridShown = false;
  private fittedJob = '';
  private pollTimer = 0;
  private healthTimer = 0;
  private saveTimer = 0;

  // form refs
  private modeButtons!: Record<Region['kind'], HTMLButtonElement>;
  private circleFields!: HTMLElement;
  private rectFields!: HTMLElement;
  private latInput!: HTMLInputElement;
  private lonInput!: HTMLInputElement;
  private radiusRange!: HTMLInputElement;
  private radiusNumber!: HTMLInputElement;
  private bboxInputs!: Record<'south' | 'west' | 'north' | 'east', HTMLInputElement>;
  private drawBtn!: HTMLButtonElement;
  private presetSelect!: HTMLSelectElement;
  private planBody!: HTMLElement;
  private nameInput!: HTMLInputElement;
  private nameHint!: HTMLElement;
  private profileSelect!: HTMLSelectElement;
  private profileDesc!: HTMLElement;
  private offlineCheck!: HTMLInputElement;
  private forceCheck!: HTMLInputElement;
  private budgetCheck!: HTMLInputElement;
  private budgetBuildings!: HTMLInputElement;
  private budgetTrees!: HTMLInputElement;
  private roadSurfaceSelect!: HTMLSelectElement;
  private generateBtn!: HTMLButtonElement;
  private generateReason!: HTMLElement;
  private jobsList!: HTMLElement;
  private worldsList!: HTMLElement;
  private showWorlds!: HTMLInputElement;

  constructor() {
    this.map = createMap(byId('map'));
    const initial = this.loadSavedRegion();
    this.editor = new RegionEditor(this.map, initial.region);
    this.place = initial.place;
    this.map.fitBounds(this.editor.bounds(), { padding: [40, 40], maxZoom: 17 });

    this.preview.group.addTo(this.map);
    this.worldsLayer.group.addTo(this.map);
    this.legend.control.addTo(this.map);
    this.addDarkToggle();

    this.jobTiles.onTileClick = (id, status, latlng) => {
      L.popup({ className: 'dark-popup' })
        .setLatLng(latlng)
        .setContent(h('div', { class: 'tile-popup' }, h('div', { class: 'mono' }, `Tile ${id}`), h('span', { class: `tile-state ts-${status}` }, status)))
        .openOn(this.map);
    };

    this.progress = new ProgressView(byId('job-panel'), {
      onCancel: () => void this.cancelJob(),
      onClose: () => this.closePanel(),
    });

    this.buildSidebar();
    this.editor.onChange = () => this.onRegionChanged();
    this.editor.onArmedChange = (armed) => {
      this.drawBtn.classList.toggle('active', armed);
      this.drawBtn.textContent = armed ? 'Drawing… (Esc)' : 'Draw rectangle';
      this.mapHint.textContent = 'Drag on the map to draw the rectangle · Esc keeps the current one';
      this.mapHint.hidden = !armed;
    };
    this.syncInputs(true);
    this.updateName();
    this.renderPlan();
    this.updateGenerate();

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.apiUp) void this.refreshJobs();
    });
    void this.checkHealth();
  }

  // ------------------------------------------------------------ sidebar

  private buildSidebar(): void {
    const s = this.sections;

    // Region
    const region = createSection(s, 'Region', true);
    const circleBtn = h('button', { class: 'btn seg-btn', title: 'Centre + radius' }, '◯ Circle');
    const rectBtn = h('button', { class: 'btn seg-btn', title: 'Bounding box' }, '▭ Rectangle');
    circleBtn.addEventListener('click', () => this.editor.setMode('circle'));
    rectBtn.addEventListener('click', () => this.editor.setMode('bbox'));
    this.modeButtons = { circle: circleBtn, bbox: rectBtn };

    const num = (placeholder: string, step: string): HTMLInputElement => h('input', { type: 'number', class: 'text', step, placeholder });
    this.latInput = num('latitude', '0.0001');
    this.lonInput = num('longitude', '0.0001');
    this.radiusRange = h('input', { type: 'range', class: 'range', min: String(MIN_RADIUS_M), max: String(this.maxRadius), step: '50' });
    this.radiusNumber = h('input', { type: 'number', class: 'text radius-number', min: String(MIN_RADIUS_M), max: String(this.maxRadius), step: '50' });
    for (const inp of [this.latInput, this.lonInput]) inp.addEventListener('input', () => this.onCircleInput());
    this.radiusRange.addEventListener('input', () => this.onRadiusInput(this.radiusRange.value, true));
    this.radiusNumber.addEventListener('input', () => this.onRadiusInput(this.radiusNumber.value, false));
    this.radiusNumber.addEventListener('change', () => this.syncInputs(true));
    this.circleFields = h(
      'div',
      { class: 'fields' },
      h('div', { class: 'grid2' }, h('label', { class: 'lbl' }, h('span', null, 'Latitude'), this.latInput), h('label', { class: 'lbl' }, h('span', null, 'Longitude'), this.lonInput)),
      h('label', { class: 'lbl' }, h('span', null, 'Radius (m)'), h('div', { class: 'radius-row' }, this.radiusRange, this.radiusNumber)),
      h('div', { class: 'small muted' }, 'Click the map to set the centre; drag the handles to move or resize.'),
    );

    const bb = {
      south: num('south', '0.0001'),
      west: num('west', '0.0001'),
      north: num('north', '0.0001'),
      east: num('east', '0.0001'),
    };
    this.bboxInputs = bb;
    for (const inp of Object.values(bb)) inp.addEventListener('input', () => this.onBboxInput());
    this.drawBtn = h('button', { class: 'btn' }, 'Draw rectangle');
    this.drawBtn.addEventListener('click', () => this.editor.setArmed(!this.editor.isArmed));
    this.rectFields = h(
      'div',
      { class: 'fields' },
      h(
        'div',
        { class: 'grid2' },
        h('label', { class: 'lbl' }, h('span', null, 'North'), bb.north),
        h('label', { class: 'lbl' }, h('span', null, 'East'), bb.east),
        h('label', { class: 'lbl' }, h('span', null, 'South'), bb.south),
        h('label', { class: 'lbl' }, h('span', null, 'West'), bb.west),
      ),
      h('div', { class: 'btn-row' }, this.drawBtn),
      h('div', { class: 'small muted' }, 'Drag on the map while drawing; drag the corner handles to adjust.'),
    );

    this.presetSelect = h('select', { class: 'select' }, h('option', { value: '' }, 'Jump to a preset…'));
    for (const p of PRESETS) this.presetSelect.append(h('option', { value: p.id }, `${p.label} (${p.lat.toFixed(4)}, ${p.lon.toFixed(4)})`));
    this.presetSelect.addEventListener('change', () => {
      const p = PRESETS.find((x) => x.id === this.presetSelect.value);
      this.presetSelect.value = '';
      if (!p) return;
      const cur = this.editor.getRegion();
      const radius = cur.kind === 'circle' ? cur.radius_m : DEFAULT_RADIUS_M;
      this.place = { label: p.id, lat: p.lat, lon: p.lon };
      this.editor.setRegion({ kind: 'circle', lat: p.lat, lon: p.lon, radius_m: radius });
      this.editor.fit();
    });
    const fitBtn = h('button', { class: 'btn', title: 'Zoom the map to the selected region' }, 'Fit to selection');
    fitBtn.addEventListener('click', () => this.editor.fit());

    region.append(h('div', { class: 'seg-group' }, circleBtn, rectBtn), this.circleFields, this.rectFields, this.presetSelect, h('div', { class: 'btn-row' }, fitBtn));

    // Plan preview
    const planSec = createSection(s, 'Plan preview', true);
    this.planBody = h('div', { class: 'plan-body' });
    planSec.append(this.planBody);

    // Build form
    const build = createSection(s, 'Build', true);
    this.nameInput = h('input', { type: 'text', class: 'text mono', maxlength: '48', spellcheck: 'false', autocomplete: 'off' });
    this.nameInput.addEventListener('input', () => {
      this.nameTouched = true;
      this.updateName();
    });
    // An emptied field goes back to the suggestion once the user leaves it.
    this.nameInput.addEventListener('blur', () => {
      if (this.nameInput.value) return;
      this.nameTouched = false;
      this.updateName();
    });
    const resetName = h('button', { class: 'btn', title: 'Use the suggested name' }, '↺');
    resetName.addEventListener('click', () => {
      this.nameTouched = false;
      this.updateName();
    });
    this.nameHint = h('div', { class: 'small muted' });
    this.profileSelect = h('select', { class: 'select' }, h('option', { value: 'basic' }, 'basic'));
    this.profileSelect.addEventListener('change', () => {
      this.renderProfileDesc();
      this.schedulePlan(0);
    });
    this.profileDesc = h('div', { class: 'small muted' });
    const offline = checkbox('Offline (cached sources only)', false, () => this.updateGenerate());
    const force = checkbox('Force (rebuild unchanged tiles)', false, () => this.updateGenerate());
    this.offlineCheck = offline.querySelector('input')!;
    this.forceCheck = force.querySelector('input')!;
    const numberInput = (value: number, title: string): HTMLInputElement =>
      h('input', { type: 'number', class: 'text', min: '1', step: '1', value: String(value), title });
    this.budgetBuildings = numberInput(DEFAULT_BUDGET.buildings_per_tile, 'Buildings rendered per 256 m tile');
    this.budgetTrees = numberInput(DEFAULT_BUDGET.trees_per_tile, 'Trees, shrubs and grass clusters per tile');
    const budget = checkbox('Drop smaller buildings & trees over budget', DEFAULT_BUDGET.enabled, (v) => {
      this.budgetBuildings.disabled = !v;
      this.budgetTrees.disabled = !v;
    });
    this.budgetCheck = budget.querySelector('input')!;
    const budgetRow = h('div', { class: 'budget-row' },
      h('label', { class: 'lbl' }, h('span', null, 'Buildings / tile'), this.budgetBuildings),
      h('label', { class: 'lbl' }, h('span', null, 'Trees / tile'), this.budgetTrees));
    const budgetHint = h('div', { class: 'small muted' },
      'Above the budget, smaller items are dropped first, at random; they stay in the GeoParquet data.');
    this.roadSurfaceSelect = h('select', { class: 'select' },
      h('option', { value: 'drape' }, 'Follow terrain (draped)'),
      h('option', { value: 'ribbon' }, 'Flat ribbons on the road profile'));
    this.generateBtn = h('button', { class: 'btn btn-primary btn-big' }, 'Generate world');
    this.generateBtn.addEventListener('click', () => void this.generate());
    this.generateReason = h('div', { class: 'small generate-reason' });
    build.append(
      h('label', { class: 'lbl' }, h('span', null, 'Name'), h('div', { class: 'input-row' }, this.nameInput, resetName)),
      this.nameHint,
      h('label', { class: 'lbl' }, h('span', null, 'Profile'), this.profileSelect),
      this.profileDesc,
      h('label', { class: 'lbl' }, h('span', null, 'Road surface'), this.roadSurfaceSelect),
      budget,
      budgetRow,
      budgetHint,
      offline,
      force,
      this.generateBtn,
      this.generateReason,
    );

    // Jobs
    const jobs = createSection(s, 'Jobs', true);
    this.jobsList = h('ul', { class: 'item-list' });
    jobs.append(this.jobsList);

    // Worlds
    const worlds = createSection(s, 'Worlds', true);
    const show = checkbox('Show on map', true, (v) => {
      if (v) this.worldsLayer.group.addTo(this.map);
      else this.worldsLayer.group.remove();
    });
    this.showWorlds = show.querySelector('input')!;
    this.worldsList = h('ul', { class: 'item-list' });
    worlds.append(show, this.worldsList);
    this.renderJobs();
    this.renderWorlds();
  }

  private addDarkToggle(): void {
    const ctl = new L.Control({ position: 'topright' });
    const container = this.map.getContainer();
    const apply = (on: boolean): void => {
      container.classList.toggle('map-dark', on);
      btn.classList.toggle('active', on);
    };
    const btn = h('button', { class: 'map-btn', title: 'Toggle dark map tiles', 'aria-label': 'Toggle dark map tiles' }, '◐');
    btn.addEventListener('click', () => {
      const on = !container.classList.contains('map-dark');
      apply(on);
      storageSet(LS_DARK, on ? '1' : null);
    });
    ctl.onAdd = () => {
      const wrap = h('div', { class: 'leaflet-bar' }, btn);
      L.DomEvent.disableClickPropagation(wrap);
      return wrap;
    };
    ctl.addTo(this.map);
    apply(storageGet(LS_DARK) === '1');
  }

  // ------------------------------------------------------------ region

  private loadSavedRegion(): { region: Region; place: Place | null } {
    const p = PRESETS[0];
    const fallback = { region: { kind: 'circle', lat: p.lat, lon: p.lon, radius_m: DEFAULT_RADIUS_M } as Region, place: { label: p.id, lat: p.lat, lon: p.lon } };
    const raw = storageGet(LS_REGION);
    if (!raw) return fallback;
    try {
      const saved = JSON.parse(raw) as { region?: Region; place?: Place | null };
      if (saved.region && (saved.region.kind === 'circle' || saved.region.kind === 'bbox') && !regionProblem(saved.region)) {
        return { region: saved.region, place: saved.place ?? null };
      }
    } catch {
      // ignore a corrupt entry
    }
    return fallback;
  }

  private onRegionChanged(): void {
    const r = this.editor.getRegion();
    // A place name only sticks while the centre stays near it.
    if (this.place) {
      const c = regionCenter(r);
      if (this.map.distance([c.lat, c.lon], [this.place.lat, this.place.lon]) > 750) this.place = null;
    }
    this.syncInputs(false);
    this.updateName();
    window.clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(() => storageSet(LS_REGION, JSON.stringify({ region: this.editor.getRegion(), place: this.place })), 400);
    // Editing a new region replaces a finished job's tiles with the preview.
    if (this.jobGridShown && (!this.session || isTerminal(this.session.state.phase))) this.showJobGrid(false);
    this.submitError = '';
    this.schedulePlan();
    this.updateGenerate();
  }

  /** Copies the region into the inputs; skips the focused one unless `force`. */
  private syncInputs(force: boolean): void {
    const r = this.editor.getRegion();
    const set = (inp: HTMLInputElement, v: number, digits: number): void => {
      if (!force && document.activeElement === inp) return;
      inp.value = Number.isFinite(v) ? String(Number(v.toFixed(digits))) : '';
    };
    this.modeButtons.circle.classList.toggle('active', r.kind === 'circle');
    this.modeButtons.bbox.classList.toggle('active', r.kind === 'bbox');
    this.circleFields.hidden = r.kind !== 'circle';
    this.rectFields.hidden = r.kind !== 'bbox';
    if (r.kind === 'circle') {
      set(this.latInput, r.lat, 6);
      set(this.lonInput, r.lon, 6);
      set(this.radiusNumber, r.radius_m, 0);
      this.radiusRange.value = String(Math.round(r.radius_m));
    } else {
      set(this.bboxInputs.south, r.south, 6);
      set(this.bboxInputs.west, r.west, 6);
      set(this.bboxInputs.north, r.north, 6);
      set(this.bboxInputs.east, r.east, 6);
    }
  }

  private onCircleInput(): void {
    const r = this.editor.getRegion();
    if (r.kind !== 'circle') return;
    const lat = Number.parseFloat(this.latInput.value);
    const lon = Number.parseFloat(this.lonInput.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 85 || Math.abs(lon) > 180) return;
    this.editor.setRegion({ ...r, lat, lon });
    if (!this.map.getBounds().contains([lat, lon])) this.map.panTo([lat, lon]);
  }

  private onRadiusInput(value: string, fromRange: boolean): void {
    const r = this.editor.getRegion();
    const v = Number.parseFloat(value);
    if (r.kind !== 'circle' || !Number.isFinite(v)) return;
    if (!fromRange && v < MIN_RADIUS_M) return; // still typing
    this.editor.setRegion({ ...r, radius_m: Math.round(v) });
    if (fromRange) this.radiusNumber.value = String(Math.round(v));
    else this.radiusRange.value = String(Math.round(v));
  }

  private onBboxInput(): void {
    if (this.editor.getRegion().kind !== 'bbox') return;
    const v = (k: keyof typeof this.bboxInputs): number => Number.parseFloat(this.bboxInputs[k].value);
    const b = { south: v('south'), west: v('west'), north: v('north'), east: v('east') };
    if (!Object.values(b).every(Number.isFinite) || b.south >= b.north || b.west >= b.east) return;
    this.editor.setRegion(normalizeBbox(b));
  }

  // ------------------------------------------------------------ name / profile

  private suggestion(): string {
    return suggestName(this.editor.getRegion(), this.place?.label ?? null);
  }

  private updateName(): void {
    if (!this.nameTouched) this.nameInput.value = this.suggestion();
    const name = this.nameInput.value;
    const problem = nameProblem(name);
    this.nameInput.classList.toggle('invalid', !!problem);
    clear(this.nameHint);
    if (problem) {
      this.nameHint.className = 'small error-text';
      this.nameHint.textContent = problem;
    } else if (this.worlds.some((w) => w.name === name)) {
      this.nameHint.className = 'small warn-text';
      this.nameHint.textContent = `A world named “${name}” exists: it will be rebuilt incrementally in place.`;
    } else {
      this.nameHint.className = 'small muted';
      this.nameHint.textContent = this.nameTouched ? `Output: worlds/${name}/` : `Suggested name · output: worlds/${name}/`;
    }
    this.updateGenerate();
  }

  private renderProfiles(): void {
    const current = this.profileSelect.value;
    clear(this.profileSelect);
    const list = this.profiles.length ? this.profiles : [{ name: 'basic' }];
    for (const p of list) this.profileSelect.append(h('option', { value: p.name }, p.name));
    const wanted = list.some((p) => p.name === current) ? current : list.some((p) => p.name === 'basic') ? 'basic' : list[0].name;
    this.profileSelect.value = wanted;
    this.renderProfileDesc();
  }

  private renderProfileDesc(): void {
    const p = this.profiles.find((x) => x.name === this.profileSelect.value);
    this.profileDesc.textContent = p?.description ?? '';
    this.profileDesc.hidden = !p?.description;
  }

  // ------------------------------------------------------------ plan preview

  private schedulePlan(delay = 350): void {
    window.clearTimeout(this.planTimer);
    const previous = this.plan.kind === 'ok' ? this.plan.plan : this.plan.kind === 'loading' ? this.plan.previous : null;
    if (this.plan.kind !== 'idle' || this.apiUp) {
      this.plan = { kind: 'loading', key: this.planKey(), previous };
      this.preview.setStale(true);
      this.renderPlan();
    }
    this.planTimer = window.setTimeout(() => void this.fetchPlan(), delay);
  }

  private planKey(): string {
    return `${regionKey(this.editor.getRegion())}|${this.profileSelect.value}`;
  }

  private async fetchPlan(): Promise<void> {
    if (!this.apiUp) {
      this.plan = { kind: 'idle' };
      this.renderPlan();
      return;
    }
    const region = this.editor.getRegion();
    const key = this.planKey();
    const problem = regionProblem(region);
    if (problem) {
      this.plan = { kind: 'error', key, status: -1, message: problem };
      this.preview.clear();
      this.renderPlan();
      this.updateGenerate();
      return;
    }
    this.planAbort?.abort();
    const ctrl = new AbortController();
    this.planAbort = ctrl;
    try {
      const plan = await api.plan(planParams(region, this.profileSelect.value), ctrl.signal);
      if (ctrl !== this.planAbort) return;
      this.plan = { kind: 'ok', key, plan };
      const max = plan.limits?.max_radius_m;
      if (typeof max === 'number' && max >= MIN_RADIUS_M && max !== this.maxRadius) this.setMaxRadius(max);
      this.preview.set(plan.tiles ?? [], plan.footprint);
    } catch (e) {
      if (ctrl !== this.planAbort) return; // superseded by a newer request
      const status = e instanceof ApiError ? e.status : 0;
      this.plan = { kind: 'error', key, status, message: errorMessage(e) };
      this.preview.clear();
      if (isUnreachable(e)) this.onApiDown(e);
    }
    this.renderPlan();
    this.updateGenerate();
  }

  private setMaxRadius(max: number): void {
    this.maxRadius = max;
    this.editor.setMaxRadius(max);
    for (const inp of [this.radiusRange, this.radiusNumber]) inp.max = String(max);
    this.syncInputs(false);
  }

  private renderPlan(): void {
    const body = this.planBody;
    clear(body);
    const p = this.plan;
    if (!this.apiUp) {
      body.append(h('div', { class: 'muted small' }, this.apiChecked ? 'The plan preview needs the WorldGen API.' : 'Connecting to the WorldGen API…'));
      return;
    }
    const plan = p.kind === 'ok' ? p.plan : p.kind === 'loading' ? p.previous : null;
    if (plan) {
      const count = plan.tile_count ?? plan.tiles?.length ?? 0;
      const tile = plan.tile_size ? `${plan.tile_size} m` : '—';
      const area = typeof plan.area_km2 === 'number' ? `${plan.area_km2 < 10 ? plan.area_km2.toFixed(2) : plan.area_km2.toFixed(1)} km²` : '—';
      body.append(
        h(
          'div',
          { class: `plan-stats${p.kind === 'loading' ? ' stale' : ''}` },
          h('div', { class: 'stat' }, h('div', { class: 'stat-value' }, String(count)), h('div', { class: 'stat-label' }, 'tiles')),
          h('div', { class: 'stat' }, h('div', { class: 'stat-value' }, area), h('div', { class: 'stat-label' }, 'area')),
          h('div', { class: 'stat' }, h('div', { class: 'stat-value' }, tile), h('div', { class: 'stat-label' }, 'tile size')),
        ),
      );
      const maxTiles = plan.limits?.max_tiles;
      if (p.kind === 'ok' && count > LARGE_TILE_COUNT) {
        body.append(
          h(
            'div',
            { class: 'card card-warn small' },
            `Large region: ${count} tiles${maxTiles ? ` (limit ${maxTiles})` : ''}. Downloads and the build can take a long time.`,
          ),
        );
      }
    }
    if (p.kind === 'loading') body.append(h('div', { class: 'small muted loading-line' }, h('span', { class: 'spinner' }), 'Computing the plan…'));
    if (p.kind === 'error') {
      const rejected = p.status === 400 || p.status === -1;
      body.append(
        h('div', { class: `card ${rejected ? 'card-error' : 'card-warn'} small` }, h('b', null, rejected ? 'Region not accepted: ' : 'No plan: '), p.message),
      );
    }
  }

  // ------------------------------------------------------------ generate

  private generateBlocker(): string | null {
    if (!this.apiUp) return 'The WorldGen API is not reachable.';
    if (this.submitting) return 'Starting the build…';
    const attached = this.session && !isTerminal(this.session.state.phase) && !this.session.disposedOrEnded;
    if (attached) return 'A build is in progress. Close its panel (it keeps running) to queue another one.';
    const region = this.editor.getRegion();
    const rp = regionProblem(region);
    if (rp) return rp;
    const np = nameProblem(this.nameInput.value);
    if (np) return `Name: ${np}`;
    if (this.plan.kind === 'error' && (this.plan.status === 400 || this.plan.status === -1) && this.plan.key === this.planKey()) {
      return `The server rejected this region: ${this.plan.message}`;
    }
    return null;
  }

  private updateGenerate(): void {
    if (!this.generateBtn) return;
    const blocker = this.generateBlocker();
    this.generateBtn.disabled = !!blocker;
    const name = this.nameInput.value;
    const exists = this.worlds.some((w) => w.name === name);
    this.generateBtn.textContent = this.submitting ? 'Starting…' : exists ? `Rebuild “${name}”` : 'Generate world';
    const busy = this.jobs.some((j) => j.state === 'running' && j.id !== this.session?.id);
    this.generateReason.className = `small generate-reason ${this.submitError ? 'error-text' : 'muted'}`;
    this.generateReason.textContent = this.submitError || blocker || (busy ? 'Another build is running: this one will be queued.' : '');
  }

  private async generate(): Promise<void> {
    if (this.generateBlocker()) return;
    const region = this.editor.getRegion();
    const body = jobBody(region, {
      name: this.nameInput.value,
      profile: this.profileSelect.value || null,
      offline: this.offlineCheck.checked,
      force: this.forceCheck.checked,
      budget: {
        enabled: this.budgetCheck.checked,
        buildings_per_tile: Number(this.budgetBuildings.value) || undefined,
        trees_per_tile: Number(this.budgetTrees.value) || undefined,
      },
      roadSurface: this.roadSurfaceSelect.value as RoadSurface,
    });
    this.submitting = true;
    this.submitError = '';
    this.updateGenerate();
    try {
      const { job } = await api.createJob(body);
      this.submitting = false;
      this.attach(job.id, job);
      void this.refreshJobs();
    } catch (e) {
      this.submitting = false;
      this.submitError = `Could not start the build: ${errorMessage(e)}`;
      if (isUnreachable(e)) this.onApiDown(e);
    }
    this.updateGenerate();
  }

  // ------------------------------------------------------------ job following

  private attach(jobId: string, job: Job | null = null): void {
    this.detach();
    storageSet(LS_JOB, jobId);
    this.fittedJob = '';
    this.jobTiles.clear();
    this.jobOutline.clearLayers();
    this.progress.open(jobId, job);
    this.showJobGrid(true);
    const session = new JobSession(
      jobId,
      {
        onState: (state, prev) => this.onJobState(session, state, prev),
        onJob: (j) => {
          if (session !== this.session) return;
          this.progress.setJob(j);
        },
        onConnection: (status, detail) => this.onConnection(session, status, detail),
      },
      job,
    );
    this.session = session;
    this.progress.update(session.state);
    this.editor.setVisible(false);
    this.renderJobs();
    this.updateGenerate();
    void session.start();
  }

  /** Stops following the current job (the job itself keeps running). */
  private detach(): void {
    if (!this.session) return;
    this.session.dispose();
    this.session = null;
  }

  private onJobState(session: JobSession, s: ProgressState, prev: ProgressState): void {
    if (session !== this.session) return;
    this.progress.update(s);
    if (s.planTiles !== prev.planTiles || s.region !== prev.region) {
      this.jobTiles.setTiles(tileRings(s));
      this.drawJobOutline(s.region);
      if (this.jobGridShown && this.fittedJob !== session.id) {
        const b = this.jobTiles.bounds();
        if (b) {
          this.map.fitBounds(b, { padding: [40, 40], maxZoom: 17 });
          this.fittedJob = session.id;
        }
      }
    }
    for (const [id, st] of Object.entries(s.tiles)) {
      if (prev.tiles[id] !== st || s.planTiles !== prev.planTiles) this.jobTiles.setStatus(id, st);
    }
    this.legend.setCounts(s.tileCounts);
    if (isTerminal(s.phase) !== isTerminal(prev.phase) || s.ended !== prev.ended) {
      this.editor.setVisible(isTerminal(s.phase));
      void this.refreshJobs();
      if (s.phase === 'succeeded') void this.refreshWorlds();
    }
    this.updateGenerate();
  }

  private onConnection(session: JobSession, status: ConnectionStatus, detail?: string): void {
    if (session !== this.session) return;
    this.progress.setConnection(status, detail);
    if (status === 'lost') {
      storageSet(LS_JOB, null);
      this.editor.setVisible(true);
      if (session.state.lastSeq === 0) {
        // Never saw an event (e.g. a stale id after a server restart): just close.
        this.closePanel();
        this.toast(detail ?? 'The build is no longer known to the server.', 'warn');
      }
      this.updateGenerate();
    }
  }

  private drawJobOutline(region: RegionInfo | null): void {
    this.jobOutline.clearLayers();
    if (!region) return;
    const style: L.PathOptions = { className: 'job-footprint', interactive: false, fill: false, weight: 2 };
    if (region.center && typeof region.radius_m === 'number') {
      L.circle([region.center.lat, region.center.lon], { ...style, radius: region.radius_m }).addTo(this.jobOutline);
    } else if (Array.isArray(region.bbox_wgs84) && region.bbox_wgs84.length === 4) {
      const [w, s, e, n] = region.bbox_wgs84;
      L.rectangle([[s, w], [n, e]], style).addTo(this.jobOutline);
    }
  }

  private showJobGrid(on: boolean): void {
    this.jobGridShown = on;
    if (on) {
      this.preview.group.remove();
      this.jobTiles.group.addTo(this.map);
      this.jobOutline.addTo(this.map);
    } else {
      this.jobTiles.group.remove();
      this.jobOutline.remove();
      this.preview.group.addTo(this.map);
    }
    this.legend.setVisible(on);
  }

  private closePanel(): void {
    const s = this.session;
    if (s && !isTerminal(s.state.phase) && !s.disposedOrEnded) {
      this.toast(`Build “${s.job?.name ?? s.id}” keeps running; reopen it from Jobs.`, 'info');
    }
    this.detach();
    storageSet(LS_JOB, null);
    this.progress.close();
    this.showJobGrid(false);
    this.editor.setVisible(true);
    this.map.closePopup();
    this.renderJobs();
    this.updateGenerate();
  }

  private async cancelJob(): Promise<void> {
    const s = this.session;
    if (!s) return;
    this.progress.setCancelling(true);
    try {
      const { job } = await api.cancelJob(s.id);
      if (s === this.session) this.progress.setJob(job);
    } catch (e) {
      if (s === this.session) this.progress.setCancelling(false);
      this.toast(`Cancel failed: ${errorMessage(e)}`, 'error');
    }
  }

  // ------------------------------------------------------------ API availability

  private async checkHealth(): Promise<void> {
    window.clearTimeout(this.healthTimer);
    try {
      this.health = await api.health();
      if (!this.health || this.health.ok === false) throw new ApiError(0, 'The API reported it is not ready.');
      this.onApiUp();
    } catch (e) {
      this.onApiDown(e);
    }
  }

  private onApiUp(): void {
    const first = !this.apiChecked;
    this.apiChecked = true;
    if (this.apiUp) return;
    this.apiUp = true;
    this.banner.hidden = true;
    void this.loadProfiles();
    void this.refreshWorlds();
    void this.refreshJobs().then(() => {
      if (first) this.reattach();
    });
    this.schedulePlan(0);
    this.startPolling();
    this.updateGenerate();
  }

  private onApiDown(e: unknown): void {
    this.apiChecked = true;
    const wasUp = this.apiUp;
    this.apiUp = false;
    window.clearInterval(this.pollTimer);
    this.showBanner(errorMessage(e));
    if (wasUp || this.plan.kind !== 'idle') {
      this.plan = { kind: 'idle' };
      this.preview.clear();
    }
    this.renderPlan();
    this.updateGenerate();
    window.clearTimeout(this.healthTimer);
    this.healthTimer = window.setTimeout(() => void this.checkHealth(), POLL_MS);
  }

  private showBanner(detail: string): void {
    clear(this.banner);
    const retry = h('button', { class: 'btn' }, 'Retry now');
    retry.addEventListener('click', () => void this.checkHealth());
    this.banner.append(
      h('div', { class: 'banner-title' }, 'The WorldGen API is not reachable'),
      h(
        'div',
        null,
        'Start it from the repository root, then reload this page (or wait: it retries every 5 s):',
      ),
      h('pre', null, 'worldgen serve --output-root ./generated'),
      h(
        'div',
        { class: 'small muted' },
        'It serves this page at ',
        h('code', null, 'http://127.0.0.1:8765/generate.html'),
        '. With ',
        h('code', null, 'npm run dev'),
        ', ',
        h('code', null, '/api'),
        ' is proxied to ',
        h('code', null, 'WORLDGEN_API'),
        ' (default ',
        h('code', null, 'http://127.0.0.1:8765'),
        ').',
      ),
      h('div', { class: 'small muted mono wrap' }, detail),
      h('div', { class: 'btn-row' }, retry),
    );
    this.banner.hidden = false;
  }

  private startPolling(): void {
    window.clearInterval(this.pollTimer);
    this.pollTimer = window.setInterval(() => {
      if (document.visibilityState === 'visible') void this.refreshJobs();
    }, POLL_MS);
  }

  private reattach(): void {
    const stored = storageGet(LS_JOB);
    if (stored) {
      this.attach(stored, this.jobs.find((j) => j.id === stored) ?? null);
      return;
    }
    // Without localStorage we cannot remember the job: follow a running one.
    if (!storageOk) {
      const running = this.jobs.find((j) => j.state === 'running') ?? this.jobs.find((j) => j.state === 'queued');
      if (running) this.attach(running.id, running);
    }
  }

  private async loadProfiles(): Promise<void> {
    try {
      const { profiles } = await api.profiles();
      this.profiles = Array.isArray(profiles) ? profiles : [];
    } catch (e) {
      this.toast(`Could not load profiles: ${errorMessage(e)}`, 'warn');
    }
    this.renderProfiles();
  }

  // ------------------------------------------------------------ jobs list

  private async refreshJobs(): Promise<void> {
    try {
      const { jobs } = await api.jobs();
      this.jobs = Array.isArray(jobs) ? jobs : [];
      this.renderJobs();
      this.updateGenerate();
    } catch (e) {
      if (isUnreachable(e)) this.onApiDown(e);
    }
  }

  private renderJobs(): void {
    const list = this.jobsList;
    if (!list) return;
    clear(list);
    if (!this.jobs.length) {
      list.append(h('li', { class: 'muted small empty' }, this.apiUp ? 'No builds yet.' : '—'));
      return;
    }
    for (const job of this.jobs.slice(0, 12)) {
      const active = this.session?.id === job.id;
      const pct = typeof job.progress === 'number' ? ` ${Math.round(job.progress * 100)}%` : '';
      const stateText = job.state === 'running' ? `running${pct}` : job.state;
      const li = h(
        'li',
        { class: `item job-item${active ? ' active' : ''}`, title: active ? 'Shown in the progress panel' : 'Show this build’s progress' },
        h('div', { class: 'item-main' }, h('span', { class: 'item-name mono' }, job.name), h('span', { class: `state-badge state-${job.state}` }, stateText)),
        h('div', { class: 'item-sub small muted' }, [job.id, ago(job.created_utc), job.stage && job.state === 'running' ? job.stage : ''].filter(Boolean).join(' · ')),
      );
      if ((job.state === 'failed' || job.state === 'cancelled') && regionFromRequest(job.request)) {
        const retry = h('button', { class: 'btn btn-small', title: 'Fill the form with this build’s region and settings' }, 'Retry');
        retry.addEventListener('click', (ev) => {
          ev.stopPropagation();
          this.retry(job);
        });
        li.querySelector('.item-main')?.append(retry);
      }
      li.addEventListener('click', () => {
        if (this.session?.id !== job.id) this.attach(job.id, job);
      });
      list.append(li);
    }
  }

  // ------------------------------------------------------------ worlds

  private async refreshWorlds(): Promise<void> {
    try {
      const { worlds } = await api.worlds();
      this.worlds = Array.isArray(worlds) ? worlds : [];
    } catch (e) {
      if (isUnreachable(e)) return this.onApiDown(e);
      this.toast(`Could not list worlds: ${errorMessage(e)}`, 'warn');
    }
    this.worldsLayer.set(this.worlds, { popup: (w) => this.worldPopup(w) });
    if (!this.showWorlds.checked) this.worldsLayer.group.remove();
    this.renderWorlds();
    this.updateName();
  }

  private renderWorlds(): void {
    const list = this.worldsList;
    clear(list);
    if (!this.worlds.length) {
      list.append(h('li', { class: 'muted small empty' }, this.apiUp ? 'No compiled worlds yet.' : '—'));
      return;
    }
    for (const w of this.worlds) {
      const rebuild = h('button', { class: 'btn btn-small', title: 'Fill the form with this world’s region and name' }, 'Rebuild');
      rebuild.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.rebuild(w);
      });
      const open = h('a', { class: 'btn btn-small', href: viewerHref(w.url), title: 'Open in the 3D viewer' }, 'Open');
      open.addEventListener('click', (ev) => ev.stopPropagation());
      const li = h(
        'li',
        { class: 'item world-item', title: 'Show on the map' },
        h('div', { class: 'item-main' }, h('span', { class: 'item-name mono' }, w.name), h('span', { class: 'item-actions' }, open, rebuild)),
        h('div', { class: 'item-sub small muted' }, [w.tile_count ? `${w.tile_count} tiles` : '', w.profile ?? '', ago(w.created_utc)].filter(Boolean).join(' · ')),
      );
      li.addEventListener('click', () => {
        if (!this.showWorlds.checked) {
          this.showWorlds.checked = true;
          this.worldsLayer.group.addTo(this.map);
        }
        this.worldsLayer.open(w.name, this.map);
      });
      list.append(li);
    }
  }

  private worldPopup(w: WorldInfo): HTMLElement {
    const st = w.statistics ?? {};
    const v = w.validation ?? {};
    const rows: [string, string][] = [];
    if (w.created_utc) rows.push(['Built', `${w.created_utc.replace('T', ' ').replace('Z', ' UTC')}`]);
    if (w.profile) rows.push(['Profile', w.profile]);
    if (w.tile_count) rows.push(['Tiles', String(w.tile_count)]);
    if (w.center) rows.push(['Centre', `${w.center.lat.toFixed(5)}, ${w.center.lon.toFixed(5)}${w.radius_m ? ` · r ${Math.round(w.radius_m)} m` : ''}`]);
    const statText = ['roads', 'buildings', 'trees', 'bridges', 'tunnels']
      .filter((k) => typeof st[k] === 'number')
      .map((k) => `${st[k].toLocaleString('en-US')} ${k}`)
      .join(' · ');
    if (statText) rows.push(['Content', statText]);
    if (v.errors !== undefined || v.warnings !== undefined) rows.push(['Validation', `${v.errors ?? 0} errors · ${v.warnings ?? 0} warnings`]);
    const table = h('table', { class: 'kv small' });
    for (const [k, val] of rows) table.append(h('tr', null, h('th', null, k), h('td', null, val)));
    const rebuild = h('button', { class: 'btn' }, 'Rebuild');
    rebuild.addEventListener('click', () => {
      this.map.closePopup();
      this.rebuild(w);
    });
    return h(
      'div',
      { class: 'world-popup' },
      h('div', { class: 'wp-title mono' }, w.name),
      table,
      h('div', { class: 'btn-row' }, h('a', { class: 'btn btn-primary', href: viewerHref(w.url) }, 'Open in viewer'), rebuild),
    );
  }

  /** Fills the form from a previous job's request. */
  private retry(job: Job): void {
    const req = job.request;
    const region = regionFromRequest(req);
    if (!req || !region) return;
    if (this.session?.id === job.id || (this.session && isTerminal(this.session.state.phase))) this.closePanel();
    const c = regionCenter(region);
    this.place = { label: job.name, lat: c.lat, lon: c.lon };
    this.nameTouched = true;
    this.nameInput.value = req.name ?? job.name;
    if (req.profile && [...this.profileSelect.options].some((o) => o.value === req.profile)) {
      this.profileSelect.value = req.profile;
      this.renderProfileDesc();
    }
    this.offlineCheck.checked = !!req.offline;
    this.forceCheck.checked = !!req.force;
    const b = req.budget ?? DEFAULT_BUDGET;
    this.budgetCheck.checked = b.enabled;
    this.budgetBuildings.value = String(b.buildings_per_tile ?? DEFAULT_BUDGET.buildings_per_tile);
    this.budgetTrees.value = String(b.trees_per_tile ?? DEFAULT_BUDGET.trees_per_tile);
    this.budgetBuildings.disabled = this.budgetTrees.disabled = !b.enabled;
    this.roadSurfaceSelect.value = req.road_surface ?? 'drape';
    this.editor.setRegion(region);
    this.editor.fit();
    this.updateName();
  }

  private rebuild(w: WorldInfo): void {
    const region = regionFromWorld(w);
    if (!region) {
      this.toast(`World “${w.name}” does not report its region.`, 'warn');
      return;
    }
    const c = regionCenter(region);
    this.place = { label: w.name, lat: c.lat, lon: c.lon };
    this.nameTouched = true;
    this.nameInput.value = w.name;
    if (w.profile && [...this.profileSelect.options].some((o) => o.value === w.profile)) {
      this.profileSelect.value = w.profile;
      this.renderProfileDesc();
    }
    if (this.session && isTerminal(this.session.state.phase)) this.closePanel();
    this.editor.setRegion(region);
    this.editor.fit();
    this.updateName();
  }

  // ------------------------------------------------------------ misc

  private toast(text: string, level: 'info' | 'warn' | 'error' = 'info'): void {
    const el = h('div', { class: `toast toast-${level}`, role: level === 'error' ? 'alert' : 'status', title: 'Click to dismiss' }, text);
    el.addEventListener('click', () => el.remove());
    this.toasts.append(el);
    while (this.toasts.children.length > 4) this.toasts.firstElementChild?.remove();
    window.setTimeout(() => el.remove(), level === 'error' ? 10000 : 6000);
  }
}

/** "3 min ago" from an ISO timestamp. */
function ago(iso: string | null | undefined): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(t).toISOString().slice(0, 10);
}

try {
  const page = new GeneratePage();
  (window as unknown as { worldgenGenerate?: unknown }).worldgenGenerate = page;
} catch (e) {
  console.error(e);
  document.body.append(h('div', { class: 'fatal' }, `The page could not start: ${errorMessage(e)}`));
}
