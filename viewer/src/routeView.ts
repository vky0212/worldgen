/**
 * Three.js side of routing: start/end markers and the route polyline.
 * The routing algorithm itself lives in routing.ts (pure, unit-tested).
 */
import { Group, Mesh, MeshBasicMaterial, SphereGeometry } from 'three';
import { enuToThree } from './geo';
import type { RouteResult } from './routing';
import { disposeObject, edgePolyline, makeFatLine } from './topology';
import type { GraphEdge, GraphNode, Vec3 } from './types';

const ROUTE_LIFT = 2.0;

export class RouteView {
  readonly group = new Group();
  private readonly startMarker: Mesh;
  private readonly endMarker: Mesh;
  private line: Group | null = null;

  constructor() {
    this.group.name = 'route_overlay';
    const geo = new SphereGeometry(2.5, 20, 12);
    this.startMarker = new Mesh(geo, new MeshBasicMaterial({ color: '#22c55e', depthTest: false, transparent: true }));
    this.endMarker = new Mesh(geo, new MeshBasicMaterial({ color: '#ef4444', depthTest: false, transparent: true }));
    for (const m of [this.startMarker, this.endMarker]) {
      m.visible = false;
      m.renderOrder = 14;
      this.group.add(m);
    }
  }

  setMarker(which: 'start' | 'end', node: GraphNode | null): void {
    const m = which === 'start' ? this.startMarker : this.endMarker;
    if (!node) {
      m.visible = false;
      return;
    }
    enuToThree(node.x, node.y, (node.z ?? 0) + ROUTE_LIFT + 1.5, m.position);
    m.visible = true;
  }

  /** Scales markers so they stay visible from far away. */
  updateMarkerScale(cameraDistance: number): void {
    const s = Math.max(1, cameraDistance / 250);
    this.startMarker.scale.setScalar(s);
    this.endMarker.scale.setScalar(s);
  }

  showRoute(result: RouteResult, edgeById: Map<string, GraphEdge>, nodeById: Map<string, GraphNode>): void {
    this.clearLine();
    if (!result.found || result.steps.length === 0) return;
    const pts: Vec3[] = [];
    for (const step of result.steps) {
      const edge = edgeById.get(step.edgeId);
      if (!edge) continue;
      let poly = edgePolyline(edge, nodeById);
      if (step.reversed) poly = [...poly].reverse();
      for (const p of poly) {
        const last = pts[pts.length - 1];
        if (last && last[0] === p[0] && last[1] === p[1] && (last[2] ?? 0) === (p[2] ?? 0)) continue;
        pts.push([p[0], p[1], p[2] ?? 0]);
      }
    }
    if (pts.length < 2) return;
    const g = new Group();
    g.add(makeFatLine(pts, '#ff3df2', 7, ROUTE_LIFT));
    this.line = g;
    this.group.add(g);
  }

  clearLine(): void {
    if (this.line) {
      this.group.remove(this.line);
      disposeObject(this.line);
      this.line = null;
    }
  }

  clear(): void {
    this.clearLine();
    this.setMarker('start', null);
    this.setMarker('end', null);
  }
}
