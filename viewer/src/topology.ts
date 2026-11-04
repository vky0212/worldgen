/**
 * Road topology overlay: graph edges as line segments (coloured by type,
 * one-way chevrons, tunnels dashed and drawn through terrain) and nodes as
 * points coloured by kind. Also provides picking of edges/nodes.
 */
import {
  BufferGeometry,
  CanvasTexture,
  Color,
  Float32BufferAttribute,
  Group,
  Points,
  PointsMaterial,
  Vector3,
} from 'three';
import type { Intersection, Raycaster, Texture } from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import type { GraphEdge, GraphNode, RoadGraph, Vec3 } from './types';
import type { LegendEntry } from './colorModes';

/** Graph lines are drawn this far above the stored z (metres). */
export const GRAPH_LIFT = 0.5;

export const EDGE_COLORS = {
  road: '#3b82f6',
  oneway: '#22d3ee',
  bridge: '#f97316',
  tunnel: '#a855f7',
  nonMotor: '#94a3b8',
} as const;

export const NODE_COLORS: Record<string, string> = {
  intersection: '#fde047',
  dead_end: '#ef4444',
  connector: '#a3a3a3',
  boundary: '#ec4899',
};
const NODE_OTHER = '#ffffff';

export type GraphPick =
  | { type: 'edge'; edge: GraphEdge; distance: number; point: Vector3 }
  | { type: 'node'; node: GraphNode; distance: number; point: Vector3 };

export function edgeIsOneway(e: GraphEdge): boolean {
  return e.direction === 'forward' || e.direction === 'backward';
}

/** ENU polyline of an edge, from-node to to-node. */
export function edgePolyline(e: GraphEdge, nodes: Map<string, GraphNode>): Vec3[] {
  if (Array.isArray(e.geometry) && e.geometry.length >= 2) {
    return e.geometry.filter((p) => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
  }
  const a = nodes.get(e.from);
  const b = nodes.get(e.to);
  if (!a || !b) return [];
  return [
    [a.x, a.y, a.z ?? 0],
    [b.x, b.y, b.z ?? 0],
  ];
}

/** ENU -> three with the graph lift applied. */
function toThree(p: Vec3, lift: number, out: number[]): void {
  out.push(p[0], (p[2] ?? 0) + lift, -p[1]);
}

let circleTexture: Texture | null = null;
/** Round point sprite shared by graph nodes and validation markers. */
export function getCircleTexture(): Texture {
  if (circleTexture) return circleTexture;
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = 6;
    ctx.strokeStyle = 'rgba(0,0,0,0.85)';
    ctx.stroke();
  }
  circleTexture = new CanvasTexture(canvas);
  return circleTexture;
}

interface SegmentLayer {
  object: LineSegments2;
  /** Edge index per segment. */
  segEdge: number[];
}

export class TopologyOverlay {
  readonly group = new Group();
  readonly graph: RoadGraph;
  readonly nodeById = new Map<string, GraphNode>();
  readonly edgeById = new Map<string, GraphEdge>();
  readonly onewayCount: number;
  private readonly layers: SegmentLayer[] = [];
  private readonly nodePoints: Points | null = null;
  private readonly nodeList: GraphNode[] = [];
  private readonly highlight = new Group();

  constructor(graph: RoadGraph) {
    this.graph = graph;
    this.group.name = 'topology_overlay';
    for (const n of graph.nodes ?? []) {
      if (n && typeof n.id === 'string' && Number.isFinite(n.x) && Number.isFinite(n.y)) {
        this.nodeById.set(n.id, n);
        this.nodeList.push(n);
      }
    }
    const edges = (graph.edges ?? []).filter((e) => e && typeof e.id === 'string');
    edges.forEach((e) => this.edgeById.set(e.id, e));

    const ground = { pos: [] as number[], col: [] as number[], seg: [] as number[] };
    const tunnel = { pos: [] as number[], col: [] as number[], seg: [] as number[] };
    const arrows = { pos: [] as number[], col: [] as number[], seg: [] as number[] };
    const c = new Color();
    let oneway = 0;

    edges.forEach((e, ei) => {
      const line = edgePolyline(e, this.nodeById);
      if (line.length < 2) return;
      const target = e.tunnel ? tunnel : ground;
      const motor = e.modes?.motor_vehicle !== false;
      const isOneway = edgeIsOneway(e);
      if (isOneway) oneway++;
      const hex = e.tunnel
        ? EDGE_COLORS.tunnel
        : e.bridge
          ? EDGE_COLORS.bridge
          : !motor
            ? EDGE_COLORS.nonMotor
            : isOneway
              ? EDGE_COLORS.oneway
              : EDGE_COLORS.road;
      c.set(hex);
      for (let k = 0; k + 1 < line.length; k++) {
        toThree(line[k], GRAPH_LIFT, target.pos);
        toThree(line[k + 1], GRAPH_LIFT, target.pos);
        target.col.push(c.r, c.g, c.b, c.r, c.g, c.b);
        target.seg.push(ei);
      }
      if (isOneway) this.addChevron(line, e.direction === 'backward', c, ei, arrows);
    });
    this.onewayCount = oneway;

    // Screen-space fat lines (LineSegments2): readable at any zoom level.
    if (ground.seg.length) this.addLayer(ground, { linewidth: 3 }, 3);
    if (arrows.seg.length) this.addLayer(arrows, { linewidth: 2.5 }, 3);
    if (tunnel.seg.length) {
      // Tunnels are dashed and drawn on top of everything so they can be seen
      // through the terrain.
      const layer = this.addLayer(tunnel, { linewidth: 3, dashed: true, dashSize: 4, gapSize: 3, depthTest: false }, 4);
      layer.object.computeLineDistances();
    }

    if (this.nodeList.length) {
      const pos: number[] = [];
      const col: number[] = [];
      for (const n of this.nodeList) {
        pos.push(n.x, (n.z ?? 0) + GRAPH_LIFT, -n.y);
        c.set(NODE_COLORS[n.kind] ?? NODE_OTHER);
        col.push(c.r, c.g, c.b);
      }
      const g = new BufferGeometry();
      g.setAttribute('position', new Float32BufferAttribute(pos, 3));
      g.setAttribute('color', new Float32BufferAttribute(col, 3));
      const mat = new PointsMaterial({
        size: 9,
        sizeAttenuation: false,
        vertexColors: true,
        map: getCircleTexture(),
        alphaTest: 0.5,
        transparent: true,
        depthWrite: false,
      });
      const points = new Points(g, mat);
      points.renderOrder = 5;
      points.name = 'graph_nodes';
      this.nodePoints = points;
      this.group.add(points);
    }
    this.highlight.renderOrder = 12;
    this.group.add(this.highlight);
  }

  get visible(): boolean {
    return this.group.visible;
  }

  set visible(v: boolean) {
    this.group.visible = v;
  }

  setNodesVisible(v: boolean): void {
    if (this.nodePoints) this.nodePoints.visible = v;
  }

  legend(): { edges: LegendEntry[]; nodes: LegendEntry[] } {
    return {
      edges: [
        { color: EDGE_COLORS.road, label: 'Road (two-way)' },
        { color: EDGE_COLORS.oneway, label: 'One-way (chevron = direction)' },
        { color: EDGE_COLORS.bridge, label: 'Bridge' },
        { color: EDGE_COLORS.tunnel, label: 'Tunnel (dashed, through terrain)' },
        { color: EDGE_COLORS.nonMotor, label: 'No motor vehicles' },
      ],
      nodes: [
        ...Object.entries(NODE_COLORS).map(([label, color]) => ({ color, label: label.replace('_', ' ') })),
        { color: NODE_OTHER, label: 'other' },
      ],
    };
  }

  /**
   * Picks the closest node (preferred) or edge under the ray. The raycaster
   * must have been set from the camera (fat-line picking is screen-space);
   * `params.Line2.threshold` (pixels) widens the edge hit area.
   */
  pick(raycaster: Raycaster): GraphPick | null {
    if (!this.group.visible) return null;
    if (this.nodePoints && this.nodePoints.visible) {
      const hits = raycaster.intersectObject(this.nodePoints, false);
      const h = hits[0];
      if (h && h.index !== undefined && h.index !== null) {
        const node = this.nodeList[h.index];
        if (node) return { type: 'node', node, distance: h.distance, point: h.point.clone() };
      }
    }
    let best: { hit: Intersection; layer: SegmentLayer } | null = null;
    for (const layer of this.layers) {
      const hits = raycaster.intersectObject(layer.object, false);
      if (hits[0] && (!best || hits[0].distance < best.hit.distance)) best = { hit: hits[0], layer };
    }
    const seg = best?.hit.faceIndex;
    if (!best || seg === undefined || seg === null) return null;
    // LineSegments2 reports the segment index as `faceIndex`.
    const edgeIndex = best.layer.segEdge[seg];
    const edge = edgeIndex !== undefined ? (this.graph.edges ?? [])[edgeIndex] : undefined;
    if (!edge) return null;
    return { type: 'edge', edge, distance: best.hit.distance, point: best.hit.point.clone() };
  }

  /** Thick highlight of edges (selection) – screen-space width lines. */
  highlightEdges(edges: GraphEdge[], color = '#ffe14d'): void {
    this.clearHighlight();
    for (const e of edges) {
      const line = edgePolyline(e, this.nodeById);
      if (line.length < 2) continue;
      this.highlight.add(makeFatLine(line, color, 6, GRAPH_LIFT + 0.2));
    }
  }

  highlightNode(node: GraphNode): void {
    this.clearHighlight();
    const g = new BufferGeometry();
    g.setAttribute('position', new Float32BufferAttribute([node.x, (node.z ?? 0) + GRAPH_LIFT, -node.y], 3));
    const mat = new PointsMaterial({
      size: 18,
      sizeAttenuation: false,
      color: '#ffe14d',
      map: getCircleTexture(),
      alphaTest: 0.5,
      depthTest: false,
      transparent: true,
    });
    const p = new Points(g, mat);
    p.renderOrder = 13;
    this.highlight.add(p);
  }

  clearHighlight(): void {
    for (const child of [...this.highlight.children]) {
      this.highlight.remove(child);
      disposeObject(child);
    }
  }

  private addLayer(
    data: { pos: number[]; col: number[]; seg: number[] },
    params: { linewidth: number; dashed?: boolean; dashSize?: number; gapSize?: number; depthTest?: boolean },
    renderOrder: number,
  ): SegmentLayer {
    const g = new LineSegmentsGeometry();
    g.setPositions(data.pos);
    g.setColors(data.col);
    const material = new LineMaterial({
      vertexColors: true,
      worldUnits: false,
      transparent: true,
      depthWrite: false,
      ...params,
    });
    const obj = new LineSegments2(g, material);
    obj.renderOrder = renderOrder;
    const layer = { object: obj, segEdge: data.seg };
    this.layers.push(layer);
    this.group.add(obj);
    return layer;
  }

  /** Adds a direction chevron at the middle of a one-way edge. */
  private addChevron(
    line: Vec3[],
    reversed: boolean,
    color: Color,
    edgeIndex: number,
    out: { pos: number[]; col: number[]; seg: number[] },
  ): void {
    let total = 0;
    const lens: number[] = [];
    for (let k = 0; k + 1 < line.length; k++) {
      const l = Math.hypot(line[k + 1][0] - line[k][0], line[k + 1][1] - line[k][1]);
      lens.push(l);
      total += l;
    }
    if (total <= 0.5) return;
    let target = total / 2;
    let k = 0;
    while (k < lens.length - 1 && target > lens[k]) {
      target -= lens[k];
      k++;
    }
    const a = line[k];
    const b = line[k + 1];
    const t = lens[k] > 0 ? target / lens[k] : 0;
    const mx = a[0] + (b[0] - a[0]) * t;
    const my = a[1] + (b[1] - a[1]) * t;
    const mz = (a[2] ?? 0) + ((b[2] ?? 0) - (a[2] ?? 0)) * t;
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    const dl = Math.hypot(dx, dy) || 1;
    dx /= dl;
    dy /= dl;
    if (reversed) {
      dx = -dx;
      dy = -dy;
    }
    const size = Math.min(6, Math.max(1.5, total / 4));
    const tip: Vec3 = [mx + dx * size * 0.5, my + dy * size * 0.5, mz];
    for (const sgn of [1, -1]) {
      // Wings at ±150° from the travel direction.
      const cos = Math.cos((150 * Math.PI) / 180);
      const sin = Math.sin((150 * Math.PI) / 180) * sgn;
      const wx = dx * cos - dy * sin;
      const wy = dx * sin + dy * cos;
      toThree(tip, GRAPH_LIFT + 0.05, out.pos);
      toThree([tip[0] + wx * size, tip[1] + wy * size, mz], GRAPH_LIFT + 0.05, out.pos);
      out.col.push(color.r, color.g, color.b, color.r, color.g, color.b);
      out.seg.push(edgeIndex);
    }
  }
}

/** Screen-space thick polyline (ENU input) drawn on top of the scene. */
export function makeFatLine(line: Vec3[], color: string, widthPx: number, lift: number): Line2 {
  const positions: number[] = [];
  for (const p of line) toThree(p, lift, positions);
  const g = new LineGeometry();
  g.setPositions(positions);
  const mat = new LineMaterial({
    color: new Color(color).getHex(),
    linewidth: widthPx,
    worldUnits: false,
    depthTest: false,
    transparent: true,
    opacity: 0.95,
  });
  const l = new Line2(g, mat);
  l.computeLineDistances();
  l.renderOrder = 12;
  return l;
}

export function disposeObject(obj: { traverse?: (cb: (o: unknown) => void) => void }): void {
  obj.traverse?.((o) => {
    const m = o as { geometry?: { dispose(): void }; material?: { dispose(): void } | { dispose(): void }[] };
    m.geometry?.dispose();
    if (Array.isArray(m.material)) m.material.forEach((x) => x.dispose());
    else m.material?.dispose();
  });
}
