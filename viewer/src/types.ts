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
  /** 1.x: `index` (flat tiles/index.json); 2.x data-first: `mode: "data"` and `tileset`. */
  tiling?: { tile_size?: number; tile_count?: number; index?: string; tileset?: string; mode?: string };
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
  /** Optional satellite orthophoto JPEG (north up, covering `bounds`). */
  ortho?: string | null;
  hash?: string;
  counts?: Record<string, number>;
  /** Content format: a baked GLB (default) or a data-first `.wgt` container (format 2.x). */
  format?: 'glb' | 'wgt';
  /**
   * Level of detail (format 2.4): 0 = the full-detail tiles, L = a coarser tile standing for
   * 2^L x 2^L of them (terrain only, no meta.json). Absent = 0.
   */
  level?: number;
  /** Coarser levels: metres the tile may differ from the full-detail world (refine above τ px). */
  geometricError?: number;
  /** Coarser levels: largest difference between its terrain and the level-0 terrain (m). */
  terrainError?: number;
  /** Ground size of a pixel of the tile's orthophoto when it differs from the world's (coarser levels). */
  orthoMpp?: number;
}

/** tiles/tileset.json of a data-first world (format 2.x). */
export interface TilesetDoc extends SchemaDoc {
  tile_size: number;
  subtree_tiles?: number;
  levels?: { level: number; size: number; spacing?: number; geometric_error?: number; tiles?: number; ortho?: { pixels?: number; m_per_px?: number } }[];
  uri: { content: string; meta?: string; collision?: string; ortho?: string; subtree: string };
  terrain?: { sections?: string[]; quantum?: number; apron?: number; triangulation?: string };
  layers?: string[];
  ortho?: OrthoIndexInfo | null;
  bounds?: [number, number, number, number] | null;
  subtrees: { si: number; sj: number; bounds?: [number, number, number, number]; zrange?: [number, number] | null; tiles?: number }[];
  /** Unit tree archetype meshes instanced by every tile (TRE sections). */
  archetypes?: string;
  trees?: { seed: number; cell_size: number; archetypes: string[]; categories: string[] };
  /** Building footprint tables (BLD sections, format 2.2) and their material palette. */
  buildings?: { sections?: string[]; materials: BuildingMaterialDoc[]; roof_shapes?: string[]; uv_scale?: number };
  /** Entity locator (format 2.3): bucket files mapping entity ids to tiles. */
  locate?: { uri: string; buckets: number; hash?: string; ids?: number };
}

/** A building material of a data-first world: glTF PBR factors (linear base colour). */
export interface BuildingMaterialDoc {
  name: string;
  color: [number, number, number, number];
  roughness: number;
  metallic: number;
}

/** tiles/sub/<si>_<sj>.json: the subtree's tiles per level, columnar. */
export interface SubtreeDoc extends SchemaDoc {
  si: number;
  sj: number;
  levels: {
    level: number;
    i: number[];
    j: number[];
    zmin: (number | null)[];
    zmax: (number | null)[];
    hash?: string[];
    /** bit 0: meta.json, bit 1: ortho.jpg, bit 2: collision.glb */
    flags?: number[];
    counts?: Record<string, number[]>;
    /** Levels above 0 (format 2.4): geometric error and terrain error per tile (m). */
    err?: number[];
    terr?: number[];
  }[];
}

/** tiles/index.json → ortho: settings of the baked satellite orthophotos. */
export interface OrthoIndexInfo {
  format?: string;
  pixels?: number;
  /** Ground size of one image pixel (tile_size / pixels). */
  m_per_px?: number;
  /** Resolution of the imagery itself (the image may be upsampled). */
  effective_source_m_per_px?: number;
  source_m_per_px?: number;
  [key: string]: unknown;
}

export interface TileIndex extends SchemaDoc {
  tile_size?: number;
  ortho?: OrthoIndexInfo | null;
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
  structure_implied?: boolean;
  width_m?: number;
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
  from_way?: string;
  to_way?: string;
  source_id?: string;
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
