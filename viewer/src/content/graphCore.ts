/**
 * Binary road graph of data-first worlds (`graph/core.wgt` + `graph/strings.json`,
 * docs/output-format.md, src/worldgen/export/graphcore.py), decoded into the
 * `road_graph.json` document shape the rest of the viewer uses.
 */
import { readWgt } from './wgt.ts';
import type { GraphEdge, GraphNode, RoadGraph, TurnRestriction, Vec3 } from '../types';

export interface GraphStrings {
  schema?: string;
  schema_version?: string;
  nodes: string[];
  edges: string[];
  roads: string[];
  names: string[];
  structures: string[];
  /** Per restriction; `from_edge`, `via_node`, `to_edge`, `via_edges`: ids the graph does not contain. */
  restrictions: {
    id: string; restriction: string; from_way: string; to_way: string; source_id: string;
    from_edge?: string; via_node?: string; to_edge?: string; via_edges?: string[];
  }[];
}

interface GraphTables {
  graph_schema_version?: string;
  origin?: { lat: number; lon: number; height?: number } | null;
  crs?: { local?: string };
  classes: string[];
  kinds: string[];
  directions: string[];
  restriction_kinds: string[];
  modes: string[];
}

const NONE = 0xffffffff;
type Kind = 'f64' | 'u32' | 'i32' | 'u16' | 'u8' | 'i8';
const SIZE: Record<Kind, number> = { f64: 8, u32: 4, i32: 4, u16: 2, u8: 1, i8: 1 };

/** Columnar section reader: a u32 count, then arrays each starting 8-byte aligned. */
class Section {
  readonly count: number;
  private readonly view: DataView;
  private off = 4;
  constructor(data: Uint8Array) {
    this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    this.count = this.view.getUint32(0, true);
  }

  read(kind: Kind, length: number): number[] {
    this.off += (8 - (this.off % 8)) % 8;
    if (this.off + length * SIZE[kind] > this.view.byteLength) throw new Error('graph section is truncated');
    const out = new Array<number>(length);
    const v = this.view;
    const o = this.off;
    for (let k = 0; k < length; k++) {
      switch (kind) {
        case 'f64': out[k] = v.getFloat64(o + 8 * k, true); break;
        case 'u32': out[k] = v.getUint32(o + 4 * k, true); break;
        case 'i32': out[k] = v.getInt32(o + 4 * k, true); break;
        case 'u16': out[k] = v.getUint16(o + 2 * k, true); break;
        case 'u8': out[k] = v.getUint8(o + k); break;
        default: out[k] = v.getInt8(o + k);
      }
    }
    this.off += length * SIZE[kind];
    return out;
  }
}

function section(sections: Map<string, Uint8Array>, tag: string): Uint8Array {
  const s = sections.get(tag);
  if (!s) throw new Error(`graph core has no ${tag} section`);
  return s;
}

export async function decodeGraphCore(buffer: ArrayBuffer, strings: GraphStrings): Promise<RoadGraph> {
  const c = await readWgt(buffer);
  const tab = JSON.parse(new TextDecoder().decode(section(c.sections, 'TAB'))) as GraphTables;

  const nod = new Section(section(c.sections, 'NOD'));
  const N = nod.count;
  const [nx, ny, nz] = [nod.read('i32', N), nod.read('i32', N), nod.read('i32', N)];
  const degree = nod.read('u16', N);
  const kind = nod.read('u8', N);
  const nodeIds = strings.nodes;
  if (nodeIds.length !== N) throw new Error(`graph strings list ${nodeIds.length} nodes, core ${N}`);
  const nodes: GraphNode[] = new Array(N);
  for (let k = 0; k < N; k++) {
    const id = nodeIds[k];
    const osm = /^n:osm:-?\d+$/.test(id) ? Number(id.slice(6)) : null;
    nodes[k] = { id, osm_id: osm, kind: tab.kinds[kind[k]], x: nx[k] / 1000, y: ny[k] / 1000, z: nz[k] / 1000, degree: degree[k] };
  }

  const geo = new Section(section(c.sections, 'GEO'));
  const xyz = geo.read('i32', 3 * geo.count);
  const til = new Section(section(c.sections, 'TIL'));
  const ij = til.read('i32', 2 * til.count);

  const edg = new Section(section(c.sections, 'EDG'));
  const E = edg.count;
  const maxspeed = edg.read('f64', E);
  const from = edg.read('u32', E);
  const to = edg.read('u32', E);
  const length = edg.read('u32', E);
  const road = edg.read('u32', E);
  const name = edg.read('u32', E);
  const structure = edg.read('u32', E);
  const gStart = edg.read('u32', E + 1);
  const tStart = edg.read('u32', E + 1);
  const width = edg.read('u16', E);
  const direction = edg.read('u8', E);
  const cls = edg.read('u8', E);
  const flags = edg.read('u8', E);
  const layer = edg.read('i8', E);
  const edgeIds = strings.edges;
  if (edgeIds.length !== E) throw new Error(`graph strings list ${edgeIds.length} edges, core ${E}`);
  const opt = (table: string[], k: number): string | null => (k === NONE ? null : table[k]);
  const edges: GraphEdge[] = new Array(E);
  for (let k = 0; k < E; k++) {
    const geometry: Vec3[] = [];
    for (let v = gStart[k]; v < gStart[k + 1]; v++) geometry.push([xyz[3 * v] / 1000, xyz[3 * v + 1] / 1000, xyz[3 * v + 2] / 1000]);
    const tiles: string[] = [];
    for (let t = tStart[k]; t < tStart[k + 1]; t++) tiles.push(`${ij[2 * t]}_${ij[2 * t + 1]}`);
    const f = flags[k];
    const modes: Record<string, boolean> = {};
    tab.modes.forEach((m, b) => { modes[m] = (f & (8 << b)) !== 0; });
    edges[k] = {
      id: edgeIds[k], road_id: strings.roads[road[k]], from: nodeIds[from[k]], to: nodeIds[to[k]],
      direction: tab.directions[direction[k]], length_m: length[k] / 1000, class: tab.classes[cls[k]],
      name: opt(strings.names, name[k]), bridge: (f & 1) !== 0, tunnel: (f & 2) !== 0, structure_implied: (f & 4) !== 0,
      layer: layer[k], structure_id: opt(strings.structures, structure[k]), modes,
      maxspeed_kmh: Number.isNaN(maxspeed[k]) ? null : maxspeed[k], width_m: width[k] / 100, geometry, tiles,
    };
  }

  const rst = new Section(section(c.sections, 'RST'));
  const R = rst.count;
  const fromEdge = rst.read('i32', R);
  const viaNode = rst.read('i32', R);
  const toEdge = rst.read('i32', R);
  const viaStart = rst.read('u32', R + 1);
  const via = rst.read('u32', viaStart[R] ?? 0);
  const rkind = rst.read('u8', R);
  const enforced = rst.read('u8', R);
  const restrictions: TurnRestriction[] = new Array(R);
  for (let k = 0; k < R; k++) {
    const s = strings.restrictions[k];
    restrictions[k] = {
      id: s.id, restriction: s.restriction, kind: tab.restriction_kinds[rkind[k]],
      from_edge: (fromEdge[k] >= 0 ? edgeIds[fromEdge[k]] : s.from_edge ?? null) as string,
      via_node: (viaNode[k] >= 0 ? nodeIds[viaNode[k]] : s.via_node ?? null) as string,
      to_edge: (toEdge[k] >= 0 ? edgeIds[toEdge[k]] : s.to_edge ?? null) as string,
      via_edges: s.via_edges ?? via.slice(viaStart[k], viaStart[k + 1]).map((e) => edgeIds[e]),
      from_way: s.from_way, to_way: s.to_way, source_id: s.source_id, enforced: enforced[k] !== 0,
    };
  }
  return {
    schema: 'worldgen.road_graph', schema_version: tab.graph_schema_version, origin: tab.origin ?? undefined,
    crs: tab.crs as RoadGraph['crs'], nodes, edges, restrictions,
  };
}
