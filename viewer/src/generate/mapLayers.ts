/**
 * Leaflet map setup and the generate page's map layers: the tile grid
 * (plan preview and live job state) and the outlines of existing worlds.
 */
import * as L from 'leaflet';
import type { LatLonPair, WorldInfo } from '../apiTypes';
import { h } from '../dom';
import type { TileStatus } from './progress';

const OSM_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors';

/**
 * Creates the map with the OSM standard base layer. `?tiles=<url template>`
 * overrides the tile URL (optionally with `?tiles_attribution=<text>`).
 */
export function createMap(container: HTMLElement): L.Map {
  const map = L.map(container, {
    zoomControl: true,
    boxZoom: false,
    worldCopyJump: true,
    minZoom: 2,
    maxZoom: 19,
  });
  const params = new URLSearchParams(window.location.search);
  const custom = params.get('tiles')?.trim();
  const attribution = custom ? escapeHtml(params.get('tiles_attribution')?.trim() || '© OpenStreetMap contributors') : OSM_ATTRIBUTION;
  L.tileLayer(custom || OSM_TILES, { maxZoom: 19, attribution }).addTo(map);
  L.control.scale({ imperial: false, position: 'bottomright' }).addTo(map);
  return map;
}

export function ringToLatLngs(ring: readonly LatLonPair[]): L.LatLngExpression[] {
  return ring.map(([lat, lon]) => [lat, lon] as L.LatLngTuple);
}

// ---------------------------------------------------------------- tile grid

/**
 * Plan preview: the region footprint plus every tile outline drawn as one
 * multi-polygon (cheap even for a thousand tiles). Not interactive.
 */
export class PlanPreviewLayer {
  readonly group = L.layerGroup();
  private grid: L.Polygon | null = null;
  private footprint: L.Polygon | null = null;

  set(tiles: readonly { ring?: LatLonPair[] }[], footprint: readonly LatLonPair[] | null | undefined): void {
    this.clear();
    const rings = tiles.filter((t) => Array.isArray(t.ring) && t.ring.length >= 3).map((t) => [ringToLatLngs(t.ring!)]);
    if (rings.length) {
      this.grid = L.polygon(rings as L.LatLngExpression[][][], { className: 'plan-grid', interactive: false, weight: 1 });
      this.group.addLayer(this.grid);
    }
    if (footprint && footprint.length >= 3) {
      this.footprint = L.polygon(ringToLatLngs(footprint), { className: 'plan-footprint', interactive: false, weight: 2, fill: false });
      this.group.addLayer(this.footprint);
    }
  }

  setStale(stale: boolean): void {
    for (const layer of [this.grid, this.footprint]) {
      const el = layer?.getElement();
      if (el) el.classList.toggle('stale', stale);
    }
  }

  clear(): void {
    this.group.clearLayers();
    this.grid = null;
    this.footprint = null;
  }
}

const TILE_STATUSES: readonly TileStatus[] = ['pending', 'building', 'built', 'unchanged'];

/** Live job tiles: one SVG polygon per tile, styled by CSS class (`tile-<status>`). */
export class JobTileLayer {
  readonly group = L.featureGroup();
  private readonly polys = new Map<string, L.Polygon>();
  private readonly status = new Map<string, TileStatus>();
  private footprint: L.Polygon | null = null;
  private readonly renderer = L.svg({ padding: 0.3 });
  onTileClick: ((id: string, status: TileStatus, latlng: L.LatLng) => void) | null = null;

  setTiles(tiles: readonly { id: string; ring: LatLonPair[] }[], footprint?: readonly LatLonPair[] | null): void {
    this.clear();
    if (footprint && footprint.length >= 3) {
      this.footprint = L.polygon(ringToLatLngs(footprint), {
        className: 'job-footprint',
        interactive: false,
        fill: false,
        weight: 2,
        renderer: this.renderer,
      });
      this.group.addLayer(this.footprint);
    }
    for (const t of tiles) {
      const poly = L.polygon(ringToLatLngs(t.ring), {
        className: 'tile tile-pending',
        weight: 1,
        renderer: this.renderer,
        bubblingMouseEvents: false,
      });
      poly.on('click', (e: L.LeafletMouseEvent) => this.onTileClick?.(t.id, this.status.get(t.id) ?? 'pending', e.latlng));
      this.polys.set(t.id, poly);
      this.status.set(t.id, 'pending');
      this.group.addLayer(poly);
    }
  }

  get size(): number {
    return this.polys.size;
  }

  setStatus(id: string, status: TileStatus): void {
    const poly = this.polys.get(id);
    if (!poly || this.status.get(id) === status) return;
    this.status.set(id, status);
    const cls = `tile tile-${status}`;
    poly.options.className = cls; // used if the path is (re)created later
    const el = poly.getElement();
    if (el) {
      for (const s of TILE_STATUSES) el.classList.remove(`tile-${s}`);
      el.classList.add(`tile-${status}`);
    }
    if (status === 'building') poly.bringToFront();
  }

  bounds(): L.LatLngBounds | null {
    const b = this.group.getBounds();
    return b.isValid() ? b : null;
  }

  clear(): void {
    this.group.clearLayers();
    this.polys.clear();
    this.status.clear();
    this.footprint = null;
  }
}

/** Map legend for the tile states, with counts. */
export class TileLegend {
  readonly control: L.Control;
  private readonly root: HTMLElement;
  private readonly counts = new Map<TileStatus, HTMLElement>();

  constructor() {
    const labels: Record<TileStatus, string> = { pending: 'Pending', building: 'Building', built: 'Built', unchanged: 'Unchanged' };
    this.root = h('div', { class: 'tile-legend' });
    for (const s of TILE_STATUSES) {
      const count = h('span', { class: 'tl-count' }, '0');
      this.counts.set(s, count);
      this.root.append(h('div', { class: 'tl-row' }, h('span', { class: `tl-swatch tl-${s}` }), h('span', null, labels[s]), count));
    }
    L.DomEvent.disableClickPropagation(this.root);
    this.control = new L.Control({ position: 'bottomleft' });
    this.control.onAdd = () => this.root;
    this.root.hidden = true;
  }

  setCounts(counts: Record<TileStatus, number>): void {
    for (const s of TILE_STATUSES) {
      const el = this.counts.get(s);
      if (el) el.textContent = String(counts[s] ?? 0);
    }
  }

  setVisible(visible: boolean): void {
    this.root.hidden = !visible;
  }
}

// ---------------------------------------------------------------- worlds

export interface WorldActions {
  popup(world: WorldInfo): HTMLElement;
}

/** Existing worlds: dashed footprint outline + a clickable centre dot with a popup. */
export class WorldsLayer {
  readonly group = L.featureGroup();
  private readonly byName = new Map<string, L.Layer[]>();

  set(worlds: readonly WorldInfo[], actions: WorldActions): void {
    this.group.clearLayers();
    this.byName.clear();
    for (const w of worlds) {
      const layers: L.Layer[] = [];
      const popup = (): HTMLElement => actions.popup(w);
      if (Array.isArray(w.footprint) && w.footprint.length >= 3) {
        const outline = L.polygon(ringToLatLngs(w.footprint), {
          className: 'world-footprint',
          weight: 2,
          fill: false,
          bubblingMouseEvents: false,
        });
        outline.bindPopup(popup, { className: 'dark-popup', maxWidth: 300 });
        outline.bindTooltip(w.name, { sticky: true, className: 'dark-tooltip' });
        layers.push(outline);
      }
      const c = w.center ?? centerOfBbox(w.bbox_wgs84);
      if (c) {
        const dot = L.circleMarker([c.lat, c.lon], {
          radius: 6,
          className: 'world-dot',
          weight: 2,
          bubblingMouseEvents: false,
        });
        dot.bindPopup(popup, { className: 'dark-popup', maxWidth: 300 });
        dot.bindTooltip(w.name, { direction: 'top', offset: [0, -6], className: 'dark-tooltip' });
        layers.push(dot);
      }
      for (const l of layers) this.group.addLayer(l);
      this.byName.set(w.name, layers);
    }
  }

  open(name: string, map: L.Map): void {
    const layers = this.byName.get(name);
    if (!layers?.length) return;
    const fg = L.featureGroup(layers);
    const b = fg.getBounds();
    if (b.isValid()) map.fitBounds(b, { padding: [40, 40], maxZoom: 16 });
    const last = layers[layers.length - 1];
    (last as L.CircleMarker).openPopup?.();
  }
}

function centerOfBbox(b: [number, number, number, number] | null | undefined): { lat: number; lon: number } | null {
  if (!Array.isArray(b) || b.length !== 4) return null;
  return { lat: (b[1] + b[3]) / 2, lon: (b[0] + b[2]) / 2 };
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);
}
