/**
 * Plain-DOM UI helpers: element builder, collapsible sections, toasts,
 * progress bar, status bar and the info (inspection) panel.
 */
import { appendChildren, clear, h } from './dom';
import type { Child } from './dom';
import { formatLatLon, formatMetres } from './geo';
import type { LatLon } from './geo';
import type { LegendEntry } from './colorModes';
import type { EntityRecord, GraphEdge, GraphNode, Issue, TurnRestriction } from './types';

export { appendChildren, checkbox, clear, createSection, h } from './dom';

/** Collapsible sidebar section (a <details> element). Returns its body. */
export function swatch(color: string, gradient?: string): HTMLSpanElement {
  return h('span', { class: gradient ? 'swatch swatch-wide' : 'swatch', style: `background:${gradient ?? color}` });
}

export function renderLegend(container: HTMLElement, entries: LegendEntry[]): void {
  clear(container);
  for (const e of entries) {
    container.append(h('div', { class: 'legend-row' }, swatch(e.color, e.gradient), h('span', null, e.label)));
  }
}

export function formatValue(v: unknown): string {
  if (v === null) return 'null';
  if (v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)));
  if (typeof v === 'string') return v;
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

// ---------------------------------------------------------------- toasts

export type ToastLevel = 'info' | 'warn' | 'error';

export class Toasts {
  private readonly root: HTMLElement;
  private readonly log: { level: ToastLevel; text: string }[] = [];
  onLog: ((entry: { level: ToastLevel; text: string }) => void) | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  show(text: string, level: ToastLevel = 'info', timeoutMs = level === 'error' ? 12000 : 6000): void {
    const entry = { level, text };
    this.log.push(entry);
    this.onLog?.(entry);
    const el = h('div', { class: `toast toast-${level}`, role: level === 'error' ? 'alert' : 'status', title: 'Click to dismiss' }, text);
    el.addEventListener('click', () => el.remove());
    this.root.append(el);
    while (this.root.children.length > 5) this.root.firstElementChild?.remove();
    if (timeoutMs > 0) window.setTimeout(() => el.remove(), timeoutMs);
  }

  entries(): readonly { level: ToastLevel; text: string }[] {
    return this.log;
  }
}

// ---------------------------------------------------------------- progress

export class ProgressBar {
  private readonly root: HTMLElement;
  private readonly bar: HTMLElement;
  private readonly label: HTMLElement;

  constructor(root: HTMLElement) {
    this.root = root;
    this.bar = h('div', { class: 'progress-fill' });
    this.label = h('div', { class: 'progress-label' });
    root.append(h('div', { class: 'progress-track' }, this.bar), this.label);
    root.hidden = true;
  }

  set(fraction: number, text: string): void {
    this.root.hidden = false;
    this.bar.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
    this.label.textContent = text;
  }

  hide(delayMs = 0): void {
    if (delayMs > 0) window.setTimeout(() => (this.root.hidden = true), delayMs);
    else this.root.hidden = true;
  }
}

// ---------------------------------------------------------------- status bar

export class StatusBar {
  readonly mode: HTMLElement;
  readonly latlon: HTMLElement;
  readonly local: HTMLElement;
  readonly elevation: HTMLElement;
  readonly hint: HTMLElement;
  readonly attribution: HTMLElement;

  constructor(root: HTMLElement) {
    this.mode = h('span', { class: 'sb-item sb-mode' }, 'Orbit');
    this.latlon = h('span', { class: 'sb-item', title: 'Cursor latitude, longitude (WGS84)' }, 'lat, lon —');
    this.local = h('span', { class: 'sb-item', title: 'Cursor local ENU x (east), y (north) in metres' }, 'x, y —');
    this.elevation = h('span', { class: 'sb-item', title: 'Elevation of the surface under the cursor' }, 'z —');
    this.hint = h('span', { class: 'sb-item sb-hint' });
    this.attribution = h('span', { class: 'sb-attribution' });
    root.append(this.mode, this.latlon, this.local, this.elevation, this.hint, this.attribution);
  }

  setCursor(latlon: LatLon | null, x: number | null, y: number | null, z: number | null, datum = ''): void {
    this.latlon.textContent = latlon ? formatLatLon(latlon) : 'lat, lon —';
    this.local.textContent = x !== null && y !== null ? `x ${x.toFixed(1)}  y ${y.toFixed(1)}` : 'x, y —';
    this.elevation.textContent = z !== null ? `z ${z.toFixed(1)} m${datum ? ` (${datum})` : ''}` : 'z —';
  }

  setHint(text: string): void {
    this.hint.textContent = text;
  }
}

// ---------------------------------------------------------------- info panel

export interface EntityView {
  id: string;
  record: EntityRecord | undefined;
  tiles: string[];
  layer?: string;
  latlon: LatLon | null;
  instanceIndex?: number;
  onEntityLink?: (id: string) => void;
}

export class InfoPanel {
  private readonly root: HTMLElement;
  private readonly title: HTMLElement;
  private readonly body: HTMLElement;
  onClose: (() => void) | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.title = h('div', { class: 'info-title' });
    const close = h('button', { class: 'icon-btn', title: 'Close (Esc)', 'aria-label': 'Close' }, '×');
    close.addEventListener('click', () => {
      this.hide();
      this.onClose?.();
    });
    this.body = h('div', { class: 'info-body' });
    root.append(h('div', { class: 'info-header' }, this.title, close), this.body);
    root.hidden = true;
  }

  hide(): void {
    this.root.hidden = true;
    clear(this.body);
  }

  show(title: string, ...content: Child[]): void {
    this.title.textContent = title;
    clear(this.body);
    appendChildren(this.body, content);
    this.root.hidden = false;
    this.body.scrollTop = 0;
  }

  showEntity(v: EntityView): void {
    const r = v.record;
    const badges = h('div', { class: 'badges' });
    badges.append(h('span', { class: 'badge badge-kind' }, r?.kind ?? 'unknown kind'));
    if (r?.uncertain) badges.append(h('span', { class: 'badge badge-uncertain', title: 'Geometry is an uncertain estimate' }, 'uncertain'));
    if (r && r.inferred && Object.keys(r.inferred).length) badges.append(h('span', { class: 'badge badge-inferred' }, 'has inferred'));

    const content: Child[] = [badges, h('div', { class: 'entity-id mono' }, v.id)];
    if (!r) {
      content.push(
        h('p', { class: 'muted' }, 'No metadata record for this entity (the tile meta.json may be missing or still loading).'),
      );
    } else {
      const rows: [string, Child][] = [
        ['Source', r.source ?? '—'],
        ['Source id', sourceLink(r.source, r.source_id)],
        ['Confidence', confidenceBar(r.confidence)],
        ['Layer', r.layer ?? v.layer ?? '—'],
        ['Tile(s)', v.tiles.join(', ') || '—'],
      ];
      if (v.instanceIndex !== undefined) rows.push(['Instance', String(v.instanceIndex)]);
      if (r.centroid && r.centroid.length >= 2) {
        rows.push(['Centroid (ENU)', r.centroid.map((c) => (Number.isFinite(c) ? c.toFixed(1) : '?')).join(', ')]);
      }
      if (v.latlon) rows.push(['Lat, lon', formatLatLon(v.latlon)]);
      content.push(kvTable(rows));

      const props = r.properties ?? {};
      const propKeys = Object.keys(props).sort();
      content.push(h('h4', null, `Observed properties (${propKeys.length})`));
      content.push(
        propKeys.length
          ? kvTable(propKeys.map((k) => [k, linkify(props[k], v.onEntityLink)]), 'props')
          : h('p', { class: 'muted' }, 'none'),
      );

      const inferred = r.inferred ?? {};
      const infKeys = Object.keys(inferred).sort();
      content.push(h('h4', { class: 'inferred-heading' }, `Inferred properties (${infKeys.length})`));
      if (infKeys.length) {
        const table = h('table', { class: 'kv inferred' });
        table.append(h('tr', null, h('th', null, 'key'), h('th', null, 'value'), h('th', null, 'method'), h('th', null, 'conf.')));
        for (const k of infKeys) {
          const iv = inferred[k];
          const value = iv && typeof iv === 'object' && 'value' in iv ? iv.value : iv;
          table.append(
            h(
              'tr',
              null,
              h('td', null, k),
              h('td', null, formatValue(value)),
              h('td', null, iv?.method ?? '—'),
              h('td', null, typeof iv?.confidence === 'number' ? iv.confidence.toFixed(2) : '—'),
            ),
          );
        }
        content.push(table);
      } else {
        content.push(h('p', { class: 'muted' }, 'none'));
      }

      const missing = Array.isArray(r.missing) ? r.missing : [];
      content.push(h('h4', null, `Missing (${missing.length})`));
      content.push(
        missing.length
          ? h('div', { class: 'chips' }, ...missing.map((m) => h('span', { class: 'chip chip-missing' }, String(m))))
          : h('p', { class: 'muted' }, 'none'),
      );

      const known = new Set(['kind', 'layer', 'source', 'source_id', 'confidence', 'properties', 'inferred', 'missing', 'uncertain', 'centroid']);
      const extra = Object.keys(r).filter((k) => !known.has(k));
      if (extra.length) {
        content.push(h('h4', null, 'Other fields'));
        content.push(kvTable(extra.map((k) => [k, linkify(r[k], v.onEntityLink)])));
      }
    }
    this.show(r?.kind ? `${capitalize(r.kind)}` : 'Entity', ...content);
  }

  showEdge(
    edge: GraphEdge,
    restrictions: TurnRestriction[],
    onLink: (id: string) => void,
  ): void {
    const modes = edge.modes ?? {};
    const modeChips = h(
      'div',
      { class: 'chips' },
      ...(['motor_vehicle', 'bicycle', 'foot'] as const).map((m) =>
        h('span', { class: `chip ${modes[m] === false ? 'chip-off' : 'chip-on'}` }, `${m.replace('_', ' ')}: ${modes[m] === false ? 'no' : 'yes'}`),
      ),
    );
    const rows: [string, Child][] = [
      ['Id', h('span', { class: 'mono' }, edge.id)],
      ['Road', edge.road_id ? entityLink(edge.road_id, onLink) : '—'],
      ['Name', edge.name ?? '—'],
      ['Class', edge.class ?? '—'],
      ['Direction', directionText(edge.direction)],
      ['From → to', h('span', null, entityLink(edge.from, onLink), ' → ', entityLink(edge.to, onLink))],
      ['Bridge', edge.bridge ? 'yes' : 'no'],
      ['Tunnel', edge.tunnel ? 'yes' : 'no'],
      ['Layer', formatValue(edge.layer ?? 0)],
      ['Structure', edge.structure_id ? entityLink(edge.structure_id, onLink) : '—'],
      ['Length', formatMetres(edge.length_m)],
      ['Max speed', edge.maxspeed_kmh ? `${edge.maxspeed_kmh} km/h` : '—'],
      ['Tiles', (edge.tiles ?? []).join(', ') || '—'],
    ];
    const content: Child[] = [kvTable(rows), h('h4', null, 'Modes'), modeChips];
    if (restrictions.length) {
      content.push(h('h4', null, `Turn restrictions (${restrictions.length})`));
      content.push(restrictionTable(restrictions, onLink));
    }
    this.show('Graph edge', ...content);
  }

  showNode(node: GraphNode, edges: GraphEdge[], restrictions: TurnRestriction[], latlon: LatLon | null, onLink: (id: string) => void): void {
    const rows: [string, Child][] = [
      ['Id', h('span', { class: 'mono' }, node.id)],
      ['Kind', node.kind],
      ['OSM id', node.osm_id ? sourceLink('osm', `node/${node.osm_id}`) : '—'],
      ['Degree', formatValue(node.degree ?? edges.length)],
      ['Local x, y, z', [node.x, node.y, node.z ?? 0].map((c) => c.toFixed(1)).join(', ')],
      ['Lat, lon', latlon ? formatLatLon(latlon) : node.lat !== undefined && node.lon !== undefined ? formatLatLon({ lat: node.lat, lon: node.lon }) : '—'],
    ];
    const content: Child[] = [kvTable(rows)];
    if (edges.length) {
      content.push(h('h4', null, `Edges (${edges.length})`));
      content.push(h('ul', { class: 'link-list' }, ...edges.map((e) => h('li', null, entityLink(e.id, onLink), ` ${e.class ?? ''} ${directionText(e.direction)}`))));
    }
    if (restrictions.length) {
      content.push(h('h4', null, `Turn restrictions via this node (${restrictions.length})`));
      content.push(restrictionTable(restrictions, onLink));
    }
    this.show('Graph node', ...content);
  }

  showIssue(issue: Issue, latlon: LatLon | null, onEntity: (id: string) => void, onFly: (() => void) | null): void {
    const rows: [string, Child][] = [
      ['Severity', h('span', { class: `sev sev-${issue.severity}` }, issue.severity)],
      ['Check', issue.check],
      ['Id', h('span', { class: 'mono' }, issue.id)],
      ['Tile', issue.tile ?? '—'],
    ];
    if (issue.location) {
      const l = issue.location;
      rows.push(['Location (ENU)', `${l.x.toFixed(1)}, ${l.y.toFixed(1)}${typeof l.z === 'number' ? `, ${l.z.toFixed(1)}` : ''}`]);
      if (latlon) rows.push(['Lat, lon', formatLatLon(latlon)]);
    } else {
      rows.push(['Location', 'none']);
    }
    const ids = issue.entity_ids ?? [];
    const content: Child[] = [h('p', { class: 'issue-message' }, issue.message), kvTable(rows)];
    if (onFly) content.push(h('button', { class: 'btn', onclick: onFly }, 'Fly to location'));
    content.push(h('h4', null, `Entities (${ids.length})`));
    content.push(
      ids.length
        ? h('ul', { class: 'link-list' }, ...ids.map((id) => h('li', null, entityLink(id, onEntity))))
        : h('p', { class: 'muted' }, 'none'),
    );
    this.show('Validation issue', ...content);
  }
}

function kvTable(rows: [string, Child][], cls = ''): HTMLTableElement {
  const t = h('table', { class: `kv ${cls}`.trim() });
  for (const [k, v] of rows) t.append(h('tr', null, h('th', null, k), h('td', null, v)));
  return t;
}

function restrictionTable(rs: TurnRestriction[], onLink: (id: string) => void): HTMLElement {
  return h(
    'ul',
    { class: 'link-list' },
    ...rs.map((r) =>
      h(
        'li',
        { class: r.enforced === false ? 'muted' : '' },
        h('strong', null, r.restriction ?? r.kind ?? 'restriction'),
        ` (${r.kind ?? '?'}${r.enforced === false ? ', not enforced' : ''}): `,
        entityLink(r.from_edge, onLink),
        ' → ',
        entityLink(r.via_node, onLink),
        ' → ',
        entityLink(r.to_edge, onLink),
      ),
    ),
  );
}

export function entityLink(id: string, onClick: (id: string) => void): HTMLAnchorElement {
  const a = h('a', { href: '#', class: 'entity-link mono', title: `Select ${id}` }, id);
  a.addEventListener('click', (ev) => {
    ev.preventDefault();
    onClick(id);
  });
  return a;
}

/** Strings that look like entity / graph ids become clickable. */
function linkify(v: unknown, onLink?: (id: string) => void): Child {
  if (onLink && typeof v === 'string' && /^(?:[a-z_]+:[^\s]+)$/.test(v) && !/^https?:/.test(v)) return entityLink(v, onLink);
  return formatValue(v);
}

function sourceLink(source?: string, sourceId?: string): Child {
  if (!sourceId) return '—';
  if (source === 'osm' && /^(node|way|relation)\/\d+$/.test(sourceId)) {
    return h('a', { href: `https://www.openstreetmap.org/${sourceId}`, target: '_blank', rel: 'noopener noreferrer' }, sourceId);
  }
  return sourceId;
}

function confidenceBar(c: number | undefined): Child {
  if (typeof c !== 'number' || !Number.isFinite(c)) return '—';
  const pct = Math.round(Math.min(1, Math.max(0, c)) * 100);
  const hue = Math.round(Math.min(1, Math.max(0, c)) * 120);
  return h(
    'span',
    { class: 'conf' },
    h('span', { class: 'conf-track' }, h('span', { class: 'conf-fill', style: `width:${pct}%;background:hsl(${hue},70%,45%)` })),
    ` ${c.toFixed(2)}`,
  );
}

function directionText(d: string): string {
  switch (d) {
    case 'both':
      return 'both ways';
    case 'forward':
      return 'one-way (from → to)';
    case 'backward':
      return 'one-way (to → from)';
    default:
      return d ?? '—';
  }
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ');
}
