/**
 * Types mirroring docs/output-format.md (schema_version 1.x).
 *
 * Everything that is not strictly required by the viewer is optional so that a
 * partially written or older world never crashes the app.
 */

/** Local ENU coordinate triple: x = east, y = north, z = up (metres). */
export type Vec3 = [number, number, number];

/** Layer node names used in tile GLBs, in display order. */
export const LAYER_NAMES = [
  'terrain',
  'roads',
  'bridges',
  'tunnels',
  'buildings',
  'water',
  'vegetation_regions',
  'trees',
] as const;

export type KnownLayer = (typeof LAYER_NAMES)[number];

export interface SchemaDoc {
  schema?: string;
  schema_version?: string;
}

export interface SourceInfo {
  name: string;
  dataset_version?: string;
  license?: string;
  attribution?: string;
  status?: 'ok' | 'partial' | 'missing' | 'disabled' | string;
  [key: string]: unknown;
}

export interface LocalBounds {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  zmin?: number;
  zmax?: number;
}

export interface Manifest extends SchemaDoc {
  compiler?: { name?: string; version?: string };
  build?: { created_utc?: string; duration_s?: number };
  region?: {
    name?: string;
    center?: { lat: number; lon: number };
    radius_m?: number;
    bbox_wgs84?: [number, number, number, number];
    [key: string]: unknown;
  };
  crs?: {
    geographic?: string;
    local?: { method?: string; proj?: string; wkt?: string };
    vertical_datum?: string;
    units?: string;
    axes?: string;
  };
  origin?: { lat: number; lon: number; height?: number };
  gltf?: { up?: string; mapping?: string };
  local_bounds?: LocalBounds;
  tiling?: { tile_size?: number; tile_count?: number; index?: string };
  layers?: string[];
  config?: Record<string, unknown>;
  config_hash?: string;
  sources?: SourceInfo[];
  outputs?: Record<string, { sha256?: string; bytes?: number }>;
  statistics?: Record<string, unknown>;
  validation?: { errors?: number; warnings?: number; report?: string };
  files?: {
    graph?: string;
    diagnostics?: string;
    combined_glb?: string;
    heightfield?: string;
    [key: string]: string | undefined;
  };
}

export interface TileEntry {
  id: string;
  i: number;
  j: number;
  /** [xmin, ymin, xmax, ymax] in local ENU metres. */
  bounds: [number, number, number, number];
  zrange?: [number, number];
  glb: string;
  collision?: string | null;
  meta?: string | null;
  hash?: string;
  counts?: Record<string, number>;
}

export interface TileIndex extends SchemaDoc {
  tile_size?: number;
  tiles: TileEntry[];
}

export interface InferredValue {
  value: unknown;
  method?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface EntityRecord {
  kind: string;
  layer?: string;
  source?: string;
  source_id?: string;
  confidence?: number;
  properties?: Record<string, unknown>;
  inferred?: Record<string, InferredValue>;
  missing?: string[];
  uncertain?: boolean;
  centroid?: Vec3;
  [key: string]: unknown;
}

export interface TileMeta extends SchemaDoc {
  tile?: string;
  entities: Record<string, EntityRecord>;
}

export type NodeKind = 'intersection' | 'dead_end' | 'connector' | 'boundary' | string;
export type EdgeDirection = 'both' | 'forward' | 'backward';
export type TravelMode = 'motor_vehicle' | 'bicycle' | 'foot';

export interface GraphNode {
  id: string;
  osm_id?: number | null;
  kind: NodeKind;
  x: number;
  y: number;
  z?: number;
  lon?: number;
  lat?: number;
  degree?: number;
}

export interface GraphEdge {
  id: string;
  road_id?: string;
  from: string;
  to: string;
  direction: EdgeDirection | string;
  length_m: number;
  class?: string;
  name?: string | null;
  bridge?: boolean;
  tunnel?: boolean;
  layer?: number;
  structure_id?: string | null;
  modes?: Partial<Record<TravelMode, boolean>> | null;
  maxspeed_kmh?: number | null;
  geometry?: Vec3[];
  tiles?: string[];
}

export interface TurnRestriction {
  id: string;
  restriction?: string;
  kind?: 'prohibitory' | 'mandatory' | string;
  from_edge: string;
  via_node: string;
  to_edge: string;
  via_edges?: string[] | null;
  enforced?: boolean;
}

export interface RoadGraph extends SchemaDoc {
  origin?: { lat: number; lon: number; height?: number };
  crs?: { local?: string };
  nodes: GraphNode[];
  edges: GraphEdge[];
  restrictions?: TurnRestriction[];
}

export type Severity = 'error' | 'warning' | 'info';

export interface IssueLocation {
  x: number;
  y: number;
  z?: number | null;
  lon?: number | null;
  lat?: number | null;
}

export interface Issue {
  id: string;
  check: string;
  severity: Severity | string;
  message: string;
  entity_ids?: string[];
  location?: IssueLocation | null;
  tile?: string | null;
}

export interface Diagnostics extends SchemaDoc {
  summary?: {
    errors?: number;
    warnings?: number;
    info?: number;
    checks?: Record<string, { status?: string; issues?: number }>;
  };
  issues: Issue[];
}
