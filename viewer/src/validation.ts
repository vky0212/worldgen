/**
 * Validation overlay: diagnostics.json issues as severity-coloured pins in the
 * 3D view plus a filterable issue list for the sidebar.
 */
import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Points,
  PointsMaterial,
} from 'three';
import type { Raycaster } from 'three';
import { getCircleTexture } from './topology';
import { clear, h } from './ui';
import type { Diagnostics, Issue } from './types';

export const SEVERITY_COLORS: Record<string, string> = {
  error: '#ef4444',
  warning: '#f59e0b',
  info: '#3b82f6',
};
const SEVERITY_ORDER: Record<string, number> = { error: 0, warning: 1, info: 2 };
const PIN_HEIGHT = 12;
const MAX_LIST = 400;

export interface IssueFilter {
  severities: Set<string>;
  text: string;
  /** Only issues with a location. */
  locatedOnly: boolean;
}

export function issueMatches(issue: Issue, f: IssueFilter): boolean {
  if (!f.severities.has(normSeverity(issue.severity))) return false;
  if (f.locatedOnly && !hasLocation(issue)) return false;
  if (!f.text) return true;
  const q = f.text.toLowerCase();
  return (
    issue.message?.toLowerCase().includes(q) ||
    issue.check?.toLowerCase().includes(q) ||
    issue.id?.toLowerCase().includes(q) ||
    (issue.entity_ids ?? []).some((id) => id.toLowerCase().includes(q))
  );
}

export function hasLocation(issue: Issue): boolean {
  const l = issue.location;
  return !!l && Number.isFinite(l.x) && Number.isFinite(l.y);
}

function normSeverity(s: string): string {
  return s in SEVERITY_ORDER ? s : 'info';
}

/** z for a marker: issue z, or the supplied ground lookup, or 0. */
export type GroundLookup = (x: number, y: number) => number | null;

export class ValidationOverlay {
  readonly group = new Group();
  readonly issues: Issue[];
  readonly diagnostics: Diagnostics;
  private shown: Issue[] = [];
  private points: Points | null = null;
  private pins: LineSegments | null = null;
  private readonly ground: GroundLookup;

  constructor(diagnostics: Diagnostics, ground: GroundLookup) {
    this.diagnostics = diagnostics;
    this.ground = ground;
    this.group.name = 'validation_overlay';
    this.issues = (Array.isArray(diagnostics.issues) ? diagnostics.issues : [])
      .filter((i) => i && typeof i === 'object')
      .sort(
        (a, b) =>
          (SEVERITY_ORDER[normSeverity(a.severity)] ?? 3) - (SEVERITY_ORDER[normSeverity(b.severity)] ?? 3) ||
          String(a.check).localeCompare(String(b.check)),
      );
  }

  counts(): Record<string, number> {
    const c: Record<string, number> = { error: 0, warning: 0, info: 0 };
    for (const i of this.issues) c[normSeverity(i.severity)]++;
    return c;
  }

  get visible(): boolean {
    return this.group.visible;
  }

  set visible(v: boolean) {
    this.group.visible = v;
  }

  /** Elevation used for an issue marker. */
  markerZ(issue: Issue): number {
    const l = issue.location!;
    if (typeof l.z === 'number' && Number.isFinite(l.z)) return l.z;
    return this.ground(l.x, l.y) ?? 0;
  }

  /** Rebuilds the markers for the issues matching the filter. */
  applyFilter(f: IssueFilter): void {
    this.disposeMarkers();
    this.shown = this.issues.filter((i) => hasLocation(i) && issueMatches(i, f));
    if (!this.shown.length) return;
    const pos: number[] = [];
    const col: number[] = [];
    const pinPos: number[] = [];
    const pinCol: number[] = [];
    const c = new Color();
    for (const issue of this.shown) {
      const l = issue.location!;
      const z = this.markerZ(issue);
      c.set(SEVERITY_COLORS[normSeverity(issue.severity)]);
      pos.push(l.x, z + PIN_HEIGHT, -l.y);
      col.push(c.r, c.g, c.b);
      pinPos.push(l.x, z, -l.y, l.x, z + PIN_HEIGHT, -l.y);
      pinCol.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new Float32BufferAttribute(col, 3));
    this.points = new Points(
      g,
      new PointsMaterial({
        size: 14,
        sizeAttenuation: false,
        vertexColors: true,
        map: getCircleTexture(),
        alphaTest: 0.5,
        depthTest: false,
        transparent: true,
      }),
    );
    this.points.renderOrder = 20;
    const pg = new BufferGeometry();
    pg.setAttribute('position', new Float32BufferAttribute(pinPos, 3));
    pg.setAttribute('color', new Float32BufferAttribute(pinCol, 3));
    this.pins = new LineSegments(pg, new LineBasicMaterial({ vertexColors: true }));
    this.pins.renderOrder = 19;
    this.group.add(this.pins, this.points);
  }

  pick(raycaster: Raycaster): { issue: Issue; distance: number } | null {
    if (!this.group.visible || !this.points) return null;
    const hit = raycaster.intersectObject(this.points, false)[0];
    if (!hit || hit.index === undefined || hit.index === null) return null;
    const issue = this.shown[hit.index];
    return issue ? { issue, distance: hit.distance } : null;
  }

  private disposeMarkers(): void {
    for (const o of [this.points, this.pins]) {
      if (!o) continue;
      this.group.remove(o);
      o.geometry.dispose();
      (o.material as PointsMaterial | LineBasicMaterial).dispose();
    }
    this.points = null;
    this.pins = null;
  }
}

/** Renders the filterable issue list into `container`. */
export function renderIssueList(
  container: HTMLElement,
  issues: Issue[],
  filter: IssueFilter,
  onSelect: (issue: Issue) => void,
): void {
  clear(container);
  const matching = issues.filter((i) => issueMatches(i, filter));
  container.append(h('div', { class: 'muted small' }, `${matching.length} of ${issues.length} issues`));
  const list = h('ul', { class: 'issue-list' });
  for (const issue of matching.slice(0, MAX_LIST)) {
    const sev = normSeverity(issue.severity);
    const li = h(
      'li',
      { class: `issue sev-${sev}`, tabindex: 0, title: issue.message },
      h('span', { class: 'sev-dot', style: `background:${SEVERITY_COLORS[sev]}` }),
      h('span', { class: 'issue-check' }, issue.check),
      hasLocation(issue) ? null : h('span', { class: 'issue-noloc', title: 'No location' }, '∅'),
      h('div', { class: 'issue-msg' }, issue.message),
    );
    li.addEventListener('click', () => onSelect(issue));
    li.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        onSelect(issue);
      }
    });
    list.append(li);
  }
  container.append(list);
  if (matching.length > MAX_LIST) {
    container.append(h('div', { class: 'muted small' }, `… ${matching.length - MAX_LIST} more (refine the filter)`));
  }
}
