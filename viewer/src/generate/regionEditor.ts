/**
 * Interactive region selection on the Leaflet map.
 *
 *  - Circle: a click on the map sets the centre; the centre handle drags the
 *    circle, the edge handle changes the radius (clamped to [min, max]).
 *  - Rectangle: while the draw tool is armed, dragging on the map draws a
 *    bbox (map panning is paused meanwhile); corner handles resize it and the
 *    centre handle moves it.
 *
 * Every change goes through `onChange(region)`.
 */
import * as L from 'leaflet';
import { FALLBACK_MAX_RADIUS_M, MIN_RADIUS_M, convertRegion, normalizeBbox } from './region';
import type { BboxRegion, CircleRegion, Region } from './region';

const DEG = Math.PI / 180;
const EARTH_RADIUS = 6371008.8;

function handleIcon(cls: string, title = ''): L.DivIcon {
  return L.divIcon({ className: `region-handle ${cls}`, iconSize: [14, 14], iconAnchor: [7, 7], html: title ? `<span class="sr-only">${title}</span>` : '' });
}

/** Point at `dist` metres from `c` in the direction of `toward` (local equirectangular). */
function towards(c: L.LatLng, toward: L.LatLng, dist: number): L.LatLng {
  const cos = Math.max(0.01, Math.cos(c.lat * DEG));
  const dx = (toward.lng - c.lng) * DEG * cos;
  const dy = (toward.lat - c.lat) * DEG;
  const len = Math.hypot(dx, dy);
  if (len === 0) return L.latLng(c.lat, c.lng + dist / (EARTH_RADIUS * cos) / DEG);
  const k = dist / EARTH_RADIUS / len;
  return L.latLng(c.lat + (dy * k) / DEG, c.lng + (dx * k) / (DEG * cos));
}

export class RegionEditor {
  onChange: ((region: Region) => void) | null = null;
  onArmedChange: ((armed: boolean) => void) | null = null;
  readonly group = L.layerGroup();

  private readonly map: L.Map;
  private region: Region;
  private maxRadius = FALLBACK_MAX_RADIUS_M;
  private visible = true;
  private armed = false;
  private draw: { start: L.LatLng; startPx: L.Point; pointerId: number; original: Region } | null = null;
  private dragOrigin: { region: BboxRegion; start: L.LatLng } | null = null;

  private readonly circle: L.Circle;
  private readonly centerHandle: L.Marker;
  private readonly edgeHandle: L.Marker;
  private readonly rect: L.Rectangle;
  private readonly corners: L.Marker[];
  private readonly moveHandle: L.Marker;

  constructor(map: L.Map, initial: Region) {
    this.map = map;
    this.region = initial;

    this.circle = L.circle([0, 0], { radius: 100, className: 'region-shape', interactive: false, weight: 2 });
    this.centerHandle = L.marker([0, 0], { icon: handleIcon('handle-center', 'centre'), draggable: true, keyboard: false, title: 'Drag to move the centre', zIndexOffset: 1000 });
    this.edgeHandle = L.marker([0, 0], { icon: handleIcon('handle-edge', 'radius'), draggable: true, keyboard: false, title: 'Drag to change the radius', zIndexOffset: 1000 });

    this.rect = L.rectangle([[0, 0], [0, 0]], { className: 'region-shape', interactive: false, weight: 2 });
    this.corners = [0, 1, 2, 3].map((i) =>
      L.marker([0, 0], { icon: handleIcon(`handle-corner handle-corner-${i}`), draggable: true, keyboard: false, title: 'Drag to resize', zIndexOffset: 1000 }),
    );
    this.moveHandle = L.marker([0, 0], { icon: handleIcon('handle-center handle-move'), draggable: true, keyboard: false, title: 'Drag to move the rectangle', zIndexOffset: 1000 });

    this.bindCircleHandles();
    this.bindRectHandles();
    this.bindDrawing();
    map.on('click', (e: L.LeafletMouseEvent) => {
      if (!this.visible || this.region.kind !== 'circle') return;
      this.update({ ...this.region, lat: e.latlng.lat, lon: e.latlng.lng });
    });

    this.group.addTo(map);
    this.render();
  }

  getRegion(): Region {
    return this.region;
  }

  get isArmed(): boolean {
    return this.armed;
  }

  /** Sets the region (switching mode if needed) and notifies `onChange`. */
  setRegion(region: Region, notify = true): void {
    if (region.kind === 'circle') region = { ...region, radius_m: Math.max(MIN_RADIUS_M, region.radius_m) };
    this.region = region;
    if (region.kind !== 'bbox') this.setArmed(false);
    this.render();
    if (notify) this.onChange?.(region);
  }

  /** Switches between circle and rectangle, converting the current region. */
  setMode(kind: Region['kind']): void {
    if (this.region.kind === kind) return;
    this.setRegion(convertRegion(this.region, kind, this.maxRadius));
    if (kind === 'bbox') this.setArmed(true);
  }

  setMaxRadius(m: number): void {
    if (Number.isFinite(m) && m >= MIN_RADIUS_M) this.maxRadius = m;
  }

  setVisible(visible: boolean): void {
    if (this.visible === visible) return;
    this.visible = visible;
    if (visible) this.group.addTo(this.map);
    else {
      this.group.remove();
      this.setArmed(false);
    }
  }

  /** Arms / disarms the rectangle draw tool. */
  setArmed(armed: boolean): void {
    armed = armed && this.visible && this.region.kind === 'bbox';
    if (this.armed === armed) return;
    this.armed = armed;
    this.draw = null;
    this.map.getContainer().classList.toggle('drawing', armed);
    if (armed) this.map.dragging.disable();
    else this.map.dragging.enable();
    this.onArmedChange?.(armed);
  }

  bounds(): L.LatLngBounds {
    const r = this.region;
    if (r.kind === 'circle') return L.latLng(r.lat, r.lon).toBounds(r.radius_m * 2);
    return L.latLngBounds([r.south, r.west], [r.north, r.east]);
  }

  fit(): void {
    this.map.fitBounds(this.bounds(), { padding: [40, 40], maxZoom: 17 });
  }

  // ------------------------------------------------------------ rendering

  private update(region: Region): void {
    this.region = region;
    this.render();
    this.onChange?.(region);
  }

  private render(): void {
    const r = this.region;
    const want: L.Layer[] =
      r.kind === 'circle' ? [this.circle, this.centerHandle, this.edgeHandle] : [this.rect, ...this.corners, this.moveHandle];
    for (const l of [this.circle, this.centerHandle, this.edgeHandle, this.rect, ...this.corners, this.moveHandle]) {
      if (want.includes(l)) {
        if (!this.group.hasLayer(l)) this.group.addLayer(l);
      } else if (this.group.hasLayer(l)) this.group.removeLayer(l);
    }
    if (r.kind === 'circle') this.renderCircle(r);
    else this.renderRect(r, null);
  }

  private renderCircle(r: CircleRegion, skipEdge = false): void {
    const c = L.latLng(r.lat, r.lon);
    this.circle.setLatLng(c);
    this.circle.setRadius(r.radius_m);
    this.centerHandle.setLatLng(c);
    if (!skipEdge) this.edgeHandle.setLatLng([r.lat, c.toBounds(r.radius_m * 2).getEast()]);
  }

  private renderRect(r: BboxRegion, skip: L.Marker | null): void {
    this.rect.setBounds(L.latLngBounds([r.south, r.west], [r.north, r.east]));
    const pts: L.LatLngTuple[] = [
      [r.south, r.west],
      [r.north, r.west],
      [r.north, r.east],
      [r.south, r.east],
    ];
    this.corners.forEach((m, i) => {
      if (m !== skip) m.setLatLng(pts[i]);
    });
    if (this.moveHandle !== skip) this.moveHandle.setLatLng([(r.south + r.north) / 2, (r.west + r.east) / 2]);
  }

  // ------------------------------------------------------------ handles

  private bindCircleHandles(): void {
    this.centerHandle.on('drag', () => {
      if (this.region.kind !== 'circle') return;
      const ll = this.centerHandle.getLatLng();
      this.region = { ...this.region, lat: ll.lat, lon: ll.lng };
      this.renderCircle(this.region);
      this.onChange?.(this.region);
    });
    this.edgeHandle.on('drag', () => {
      if (this.region.kind !== 'circle') return;
      const c = L.latLng(this.region.lat, this.region.lon);
      const ll = this.edgeHandle.getLatLng();
      const raw = this.map.distance(c, ll);
      const radius = Math.round(Math.min(this.maxRadius, Math.max(MIN_RADIUS_M, raw)));
      if (radius !== Math.round(raw)) this.edgeHandle.setLatLng(towards(c, ll, radius));
      this.region = { ...this.region, radius_m: radius };
      this.renderCircle(this.region, true);
      this.onChange?.(this.region);
    });
    this.edgeHandle.on('dragend', () => this.render());
  }

  private bindRectHandles(): void {
    this.corners.forEach((marker, i) => {
      marker.on('dragstart', () => {
        if (this.region.kind === 'bbox') this.dragOrigin = { region: this.region, start: marker.getLatLng() };
      });
      marker.on('drag', () => {
        const o = this.dragOrigin;
        if (!o || this.region.kind !== 'bbox') return;
        const ll = marker.getLatLng();
        const r = o.region;
        // The opposite corner stays put.
        const opp: [number, number][] = [
          [r.north, r.east],
          [r.south, r.east],
          [r.south, r.west],
          [r.north, r.west],
        ];
        const [olat, olon] = opp[i];
        this.region = normalizeBbox({ south: ll.lat, north: olat, west: ll.lng, east: olon });
        this.renderRect(this.region, marker);
        this.onChange?.(this.region);
      });
      marker.on('dragend', () => {
        this.dragOrigin = null;
        this.render();
      });
    });
    this.moveHandle.on('dragstart', () => {
      if (this.region.kind === 'bbox') this.dragOrigin = { region: this.region, start: this.moveHandle.getLatLng() };
    });
    this.moveHandle.on('drag', () => {
      const o = this.dragOrigin;
      if (!o) return;
      const ll = this.moveHandle.getLatLng();
      const dLat = ll.lat - o.start.lat;
      const dLon = ll.lng - o.start.lng;
      const r = o.region;
      this.region = normalizeBbox({ south: r.south + dLat, north: r.north + dLat, west: r.west + dLon, east: r.east + dLon });
      this.renderRect(this.region, this.moveHandle);
      this.onChange?.(this.region);
    });
    this.moveHandle.on('dragend', () => {
      this.dragOrigin = null;
      this.render();
    });
  }

  // ------------------------------------------------------------ rectangle drawing

  private bindDrawing(): void {
    const el = this.map.getContainer();
    const ignore = (ev: PointerEvent): boolean =>
      !(ev.target instanceof Element) || !!ev.target.closest('.leaflet-control, .leaflet-marker-icon, .leaflet-popup');

    // Capture phase: runs before Leaflet's own handlers on the map panes.
    el.addEventListener(
      'pointerdown',
      (ev) => {
        if (!this.armed || ev.button !== 0 || ignore(ev)) return;
        ev.preventDefault();
        ev.stopPropagation();
        const start = this.map.mouseEventToLatLng(ev);
        this.draw = { start, startPx: this.map.mouseEventToContainerPoint(ev), pointerId: ev.pointerId, original: this.region };
        try {
          el.setPointerCapture(ev.pointerId);
        } catch {
          // Synthetic events cannot be captured; drawing still works.
        }
      },
      true,
    );
    el.addEventListener(
      'pointermove',
      (ev) => {
        const d = this.draw;
        if (!d || ev.pointerId !== d.pointerId) return;
        ev.preventDefault();
        const ll = this.map.mouseEventToLatLng(ev);
        if (this.region.kind !== 'bbox') return;
        if (this.map.mouseEventToContainerPoint(ev).distanceTo(d.startPx) < 4) return;
        this.region = normalizeBbox({ south: d.start.lat, north: ll.lat, west: d.start.lng, east: ll.lng });
        this.renderRect(this.region, null);
      },
      true,
    );
    const finish = (ev: PointerEvent, cancelled: boolean): void => {
      const d = this.draw;
      if (!d || ev.pointerId !== d.pointerId) return;
      this.draw = null;
      try {
        el.releasePointerCapture(ev.pointerId);
      } catch {
        // ignore
      }
      const moved = this.map.mouseEventToContainerPoint(ev).distanceTo(d.startPx) >= 8;
      if (cancelled || !moved || this.region.kind !== 'bbox') {
        this.region = d.original;
        this.render();
        return;
      }
      const ll = this.map.mouseEventToLatLng(ev);
      this.setArmed(false);
      this.update(normalizeBbox({ south: d.start.lat, north: ll.lat, west: d.start.lng, east: ll.lng }));
    };
    el.addEventListener('pointerup', (ev) => finish(ev, false), true);
    el.addEventListener('pointercancel', (ev) => finish(ev, true), true);
    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && this.armed) {
        if (this.draw) this.region = this.draw.original;
        this.draw = null;
        this.setArmed(false);
        this.render();
      }
    });
  }
}
