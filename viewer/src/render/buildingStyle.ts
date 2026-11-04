/**
 * Geo data → building appearance. Picks a facade archetype, ground-floor
 * archetype, roof material, tints, storey height and night lighting for every
 * building from its tile metadata (class, amenity, levels, height, material,
 * colours, roof shape) and a regional style profile derived from the world's
 * location. Deterministic per entity id.
 *
 * Also defines the layer layout of the procedural texture arrays, shared with
 * the GPU texture engine and the building shader.
 *
 * Pure (type-only imports): unit-tested under node.
 */
import type { EntityRecord } from '../types';

// ---------------------------------------------------------------- texture layers

/** Layer indices of the procedural texture arrays. */
export const LAYER = {
  // facades: one tile = one storey × one bay
  PLASTER: 0,
  CERAMIC: 1,
  BRICK: 2,
  CONCRETE: 3,
  CURTAIN: 4,
  METAL: 5,
  TIMBER: 6,
  STONE: 7,
  SHOPFRONT: 8,
  BLANK: 9,
  // roofs: planar tiles
  ROOF_CLAY: 10,
  ROOF_SLATE: 11,
  ROOF_METAL: 12,
  ROOF_FLAT: 13,
  ROOF_SHINGLE: 14,
  // ground detail
  GRASS: 15,
  ROCK: 16,
  SOIL: 17,
  ASPHALT: 18,
  GRAVEL: 19,
  FOREST: 20,
  CONCRETE_PLAIN: 21,
} as const;

export type LayerId = (typeof LAYER)[keyof typeof LAYER];
export const LAYER_COUNT = 22;
export const FACADE_LAYERS = 10;

/** Bay width (m) of each facade layer (index < FACADE_LAYERS). */
export const BAY_WIDTH: readonly number[] = [3.0, 3.2, 2.6, 3.6, 1.6, 1.0, 2.2, 2.8, 4.0, 3.0];

/** Size (m) of one roof texture tile, indexed by layer - ROOF_CLAY. */
export const ROOF_TILE_M: readonly number[] = [2.0, 2.0, 2.5, 8.0, 2.0];

// ---------------------------------------------------------------- region profiles

export type RegionId =
  | 'east_asia_subtropical'
  | 'east_asia_temperate'
  | 'tropical'
  | 'arid'
  | 'mediterranean'
  | 'temperate_europe'
  | 'nordic'
  | 'north_america'
  | 'generic';

type Weighted<T> = readonly (readonly [T, number])[];
type RGB = readonly [number, number, number];

export type ClassGroup = 'residential' | 'apartments' | 'commercial' | 'industrial' | 'civic' | 'religious' | 'small' | 'unknown';

export interface RegionProfile {
  id: RegionId;
  label: string;
  /** Facade archetype weights per building class group. */
  walls: Record<ClassGroup, Weighted<LayerId>>;
  /** Pitched roof material weights. */
  pitchedRoofs: Weighted<LayerId>;
  /** Probability that a building without a roof_shape gets a flat roof. */
  flatRoofShare: number;
  /** Wall tints (sRGB 0..1) per facade layer, picked by seed. */
  wallTints: Partial<Record<LayerId, readonly RGB[]>>;
  roofTints: Partial<Record<LayerId, readonly RGB[]>>;
  /** Share of multi-storey residential buildings with a shop ground floor. */
  shopGroundShare: number;
  defaultFloorHeight: number;
}

const L = LAYER;

const WHITES: readonly RGB[] = [[0.92, 0.91, 0.87], [0.88, 0.86, 0.8], [0.95, 0.93, 0.88], [0.82, 0.82, 0.8]];
const CREAMS: readonly RGB[] = [[0.93, 0.87, 0.74], [0.9, 0.82, 0.68], [0.86, 0.8, 0.7], [0.94, 0.9, 0.8]];
const PASTELS: readonly RGB[] = [[0.95, 0.85, 0.7], [0.88, 0.72, 0.6], [0.93, 0.9, 0.72], [0.8, 0.86, 0.84], [0.95, 0.8, 0.72]];
const OCHRES: readonly RGB[] = [[0.9, 0.75, 0.52], [0.85, 0.64, 0.45], [0.93, 0.83, 0.62], [0.8, 0.6, 0.45]];
const SAND: readonly RGB[] = [[0.87, 0.78, 0.62], [0.82, 0.72, 0.56], [0.9, 0.84, 0.72]];
const GREYS: readonly RGB[] = [[0.72, 0.72, 0.7], [0.64, 0.64, 0.62], [0.78, 0.77, 0.74], [0.58, 0.58, 0.57]];
const CERAMICS: readonly RGB[] = [[0.86, 0.83, 0.78], [0.74, 0.74, 0.72], [0.8, 0.72, 0.64], [0.66, 0.42, 0.34], [0.9, 0.88, 0.84], [0.7, 0.66, 0.6]];
const BRICKS: readonly RGB[] = [[0.62, 0.3, 0.22], [0.55, 0.28, 0.2], [0.68, 0.38, 0.28], [0.5, 0.33, 0.27], [0.74, 0.55, 0.42]];
const METALS: readonly RGB[] = [[0.86, 0.87, 0.86], [0.55, 0.66, 0.78], [0.62, 0.72, 0.66], [0.75, 0.76, 0.78], [0.8, 0.5, 0.4]];
const WOODS: readonly RGB[] = [[0.62, 0.44, 0.3], [0.52, 0.36, 0.24], [0.7, 0.55, 0.38], [0.45, 0.3, 0.22]];
const NORDIC_WOODS: readonly RGB[] = [[0.55, 0.16, 0.12], [0.85, 0.75, 0.5], [0.9, 0.9, 0.86], [0.35, 0.4, 0.45], [0.55, 0.16, 0.12]];
const STONES: readonly RGB[] = [[0.7, 0.68, 0.63], [0.62, 0.6, 0.56], [0.76, 0.7, 0.6], [0.55, 0.55, 0.53]];
const GLASS: readonly RGB[] = [[0.6, 0.72, 0.78], [0.5, 0.6, 0.62], [0.7, 0.75, 0.8], [0.45, 0.55, 0.65]];

const CLAY_ROOFS: readonly RGB[] = [[0.66, 0.3, 0.2], [0.58, 0.26, 0.18], [0.72, 0.4, 0.26], [0.55, 0.32, 0.24]];
const DARK_ROOFS: readonly RGB[] = [[0.3, 0.31, 0.33], [0.36, 0.36, 0.38], [0.26, 0.27, 0.3], [0.4, 0.38, 0.36]];
const METAL_ROOFS: readonly RGB[] = [[0.68, 0.7, 0.72], [0.4, 0.52, 0.62], [0.55, 0.3, 0.25], [0.45, 0.55, 0.45], [0.6, 0.62, 0.62]];
const FLAT_ROOFS: readonly RGB[] = [[0.7, 0.69, 0.66], [0.62, 0.6, 0.57], [0.75, 0.73, 0.7], [0.55, 0.56, 0.55]];
const SHINGLE_ROOFS: readonly RGB[] = [[0.32, 0.3, 0.3], [0.42, 0.36, 0.32], [0.28, 0.3, 0.33], [0.48, 0.42, 0.36]];

const COMMON_TINTS: Partial<Record<LayerId, readonly RGB[]>> = {
  [L.PLASTER]: WHITES,
  [L.CERAMIC]: CERAMICS,
  [L.BRICK]: BRICKS,
  [L.CONCRETE]: GREYS,
  [L.CURTAIN]: GLASS,
  [L.METAL]: METALS,
  [L.TIMBER]: WOODS,
  [L.STONE]: STONES,
  [L.SHOPFRONT]: GREYS,
  [L.BLANK]: WHITES,
};

const COMMON_ROOF_TINTS: Partial<Record<LayerId, readonly RGB[]>> = {
  [L.ROOF_CLAY]: CLAY_ROOFS,
  [L.ROOF_SLATE]: DARK_ROOFS,
  [L.ROOF_METAL]: METAL_ROOFS,
  [L.ROOF_FLAT]: FLAT_ROOFS,
  [L.ROOF_SHINGLE]: SHINGLE_ROOFS,
};

function profile(
  id: RegionId,
  label: string,
  walls: Record<ClassGroup, Weighted<LayerId>>,
  rest: Partial<Omit<RegionProfile, 'id' | 'label' | 'walls'>> = {},
): RegionProfile {
  return {
    id,
    label,
    walls,
    pitchedRoofs: rest.pitchedRoofs ?? [[L.ROOF_CLAY, 2], [L.ROOF_SLATE, 1]],
    flatRoofShare: rest.flatRoofShare ?? 0.4,
    wallTints: { ...COMMON_TINTS, ...rest.wallTints },
    roofTints: { ...COMMON_ROOF_TINTS, ...rest.roofTints },
    shopGroundShare: rest.shopGroundShare ?? 0.1,
    defaultFloorHeight: rest.defaultFloorHeight ?? 3.0,
  };
}

export const PROFILES: Record<RegionId, RegionProfile> = {
  east_asia_subtropical: profile(
    'east_asia_subtropical',
    'Subtropical East Asia',
    {
      residential: [[L.CERAMIC, 4], [L.PLASTER, 2], [L.CONCRETE, 2], [L.BRICK, 0.5]],
      apartments: [[L.CERAMIC, 5], [L.CONCRETE, 3], [L.PLASTER, 1]],
      commercial: [[L.CERAMIC, 2], [L.CURTAIN, 2], [L.CONCRETE, 2]],
      industrial: [[L.METAL, 5], [L.CONCRETE, 2]],
      civic: [[L.CONCRETE, 3], [L.CERAMIC, 2], [L.PLASTER, 1]],
      religious: [[L.PLASTER, 2], [L.STONE, 1], [L.TIMBER, 1]],
      small: [[L.CONCRETE, 2], [L.TIMBER, 2], [L.PLASTER, 1], [L.METAL, 1]],
      unknown: [[L.CERAMIC, 3], [L.CONCRETE, 3], [L.PLASTER, 2], [L.METAL, 1]],
    },
    {
      pitchedRoofs: [[L.ROOF_METAL, 3], [L.ROOF_CLAY, 2], [L.ROOF_SLATE, 1]],
      flatRoofShare: 0.75,
      shopGroundShare: 0.45,
      wallTints: { [L.PLASTER]: [...WHITES, ...CREAMS], [L.CONCRETE]: GREYS },
      roofTints: { [L.ROOF_METAL]: [[0.42, 0.55, 0.68], [0.68, 0.7, 0.72], [0.62, 0.35, 0.28], [0.45, 0.58, 0.48]] },
      defaultFloorHeight: 3.2,
    },
  ),
  east_asia_temperate: profile(
    'east_asia_temperate',
    'Temperate East Asia',
    {
      residential: [[L.PLASTER, 3], [L.TIMBER, 1], [L.CERAMIC, 1], [L.METAL, 1]],
      apartments: [[L.CONCRETE, 4], [L.CERAMIC, 2], [L.PLASTER, 2]],
      commercial: [[L.CURTAIN, 3], [L.CONCRETE, 2], [L.CERAMIC, 1]],
      industrial: [[L.METAL, 4], [L.CONCRETE, 2]],
      civic: [[L.CONCRETE, 3], [L.PLASTER, 1], [L.BRICK, 1]],
      religious: [[L.TIMBER, 3], [L.PLASTER, 1]],
      small: [[L.TIMBER, 2], [L.PLASTER, 1], [L.METAL, 1]],
      unknown: [[L.PLASTER, 3], [L.CONCRETE, 2], [L.CERAMIC, 1]],
    },
    { pitchedRoofs: [[L.ROOF_SLATE, 3], [L.ROOF_METAL, 2], [L.ROOF_CLAY, 1]], flatRoofShare: 0.45, shopGroundShare: 0.25 },
  ),
  tropical: profile(
    'tropical',
    'Tropical',
    {
      residential: [[L.PLASTER, 4], [L.CONCRETE, 2], [L.TIMBER, 1]],
      apartments: [[L.PLASTER, 3], [L.CONCRETE, 3], [L.CERAMIC, 1]],
      commercial: [[L.PLASTER, 2], [L.CURTAIN, 2], [L.CONCRETE, 2]],
      industrial: [[L.METAL, 4], [L.CONCRETE, 2]],
      civic: [[L.PLASTER, 3], [L.CONCRETE, 2]],
      religious: [[L.PLASTER, 3], [L.STONE, 1]],
      small: [[L.TIMBER, 2], [L.PLASTER, 2], [L.METAL, 2]],
      unknown: [[L.PLASTER, 4], [L.CONCRETE, 2], [L.METAL, 1]],
    },
    {
      pitchedRoofs: [[L.ROOF_METAL, 3], [L.ROOF_CLAY, 3]],
      flatRoofShare: 0.45,
      shopGroundShare: 0.3,
      wallTints: { [L.PLASTER]: [...PASTELS, ...WHITES] },
    },
  ),
  arid: profile(
    'arid',
    'Arid',
    {
      residential: [[L.PLASTER, 5], [L.STONE, 1]],
      apartments: [[L.PLASTER, 3], [L.CONCRETE, 2], [L.STONE, 1]],
      commercial: [[L.PLASTER, 2], [L.CURTAIN, 1], [L.STONE, 1]],
      industrial: [[L.METAL, 3], [L.CONCRETE, 2]],
      civic: [[L.STONE, 2], [L.PLASTER, 2]],
      religious: [[L.STONE, 2], [L.PLASTER, 2]],
      small: [[L.PLASTER, 3], [L.STONE, 1]],
      unknown: [[L.PLASTER, 4], [L.STONE, 1], [L.CONCRETE, 1]],
    },
    {
      pitchedRoofs: [[L.ROOF_CLAY, 2], [L.ROOF_METAL, 1]],
      flatRoofShare: 0.9,
      wallTints: { [L.PLASTER]: [...SAND, ...CREAMS], [L.STONE]: SAND },
    },
  ),
  mediterranean: profile(
    'mediterranean',
    'Mediterranean',
    {
      residential: [[L.PLASTER, 5], [L.STONE, 1]],
      apartments: [[L.PLASTER, 4], [L.CONCRETE, 1]],
      commercial: [[L.PLASTER, 3], [L.STONE, 1], [L.CURTAIN, 1]],
      industrial: [[L.METAL, 3], [L.CONCRETE, 2]],
      civic: [[L.STONE, 2], [L.PLASTER, 2]],
      religious: [[L.STONE, 3], [L.PLASTER, 1]],
      small: [[L.PLASTER, 2], [L.STONE, 2]],
      unknown: [[L.PLASTER, 4], [L.STONE, 1]],
    },
    {
      pitchedRoofs: [[L.ROOF_CLAY, 5], [L.ROOF_SLATE, 0.5]],
      flatRoofShare: 0.3,
      shopGroundShare: 0.3,
      wallTints: { [L.PLASTER]: [...OCHRES, ...WHITES, ...PASTELS] },
    },
  ),
  temperate_europe: profile(
    'temperate_europe',
    'Temperate Europe',
    {
      residential: [[L.BRICK, 3], [L.PLASTER, 3], [L.STONE, 0.5]],
      apartments: [[L.PLASTER, 3], [L.BRICK, 2], [L.CONCRETE, 2]],
      commercial: [[L.CURTAIN, 2], [L.PLASTER, 2], [L.STONE, 1], [L.BRICK, 1]],
      industrial: [[L.METAL, 3], [L.BRICK, 1], [L.CONCRETE, 2]],
      civic: [[L.STONE, 2], [L.BRICK, 2], [L.CONCRETE, 1]],
      religious: [[L.STONE, 4], [L.BRICK, 1]],
      small: [[L.TIMBER, 1], [L.BRICK, 1], [L.PLASTER, 1]],
      unknown: [[L.PLASTER, 3], [L.BRICK, 3]],
    },
    {
      pitchedRoofs: [[L.ROOF_CLAY, 3], [L.ROOF_SLATE, 3]],
      flatRoofShare: 0.15,
      shopGroundShare: 0.25,
      wallTints: { [L.PLASTER]: [...WHITES, ...CREAMS, ...PASTELS] },
    },
  ),
  nordic: profile(
    'nordic',
    'Nordic',
    {
      residential: [[L.TIMBER, 4], [L.PLASTER, 2], [L.BRICK, 1]],
      apartments: [[L.PLASTER, 3], [L.BRICK, 2], [L.CONCRETE, 2]],
      commercial: [[L.CURTAIN, 2], [L.PLASTER, 2], [L.BRICK, 1]],
      industrial: [[L.METAL, 3], [L.CONCRETE, 1]],
      civic: [[L.BRICK, 2], [L.STONE, 1], [L.PLASTER, 1]],
      religious: [[L.STONE, 2], [L.TIMBER, 2], [L.PLASTER, 1]],
      small: [[L.TIMBER, 4]],
      unknown: [[L.TIMBER, 3], [L.PLASTER, 2]],
    },
    {
      pitchedRoofs: [[L.ROOF_SLATE, 3], [L.ROOF_METAL, 2], [L.ROOF_CLAY, 1]],
      flatRoofShare: 0.1,
      wallTints: { [L.TIMBER]: NORDIC_WOODS },
    },
  ),
  north_america: profile(
    'north_america',
    'North America',
    {
      residential: [[L.TIMBER, 4], [L.BRICK, 2], [L.PLASTER, 1]],
      apartments: [[L.BRICK, 3], [L.CONCRETE, 2], [L.PLASTER, 1]],
      commercial: [[L.CURTAIN, 3], [L.CONCRETE, 2], [L.BRICK, 1]],
      industrial: [[L.METAL, 4], [L.CONCRETE, 2]],
      civic: [[L.BRICK, 2], [L.CONCRETE, 2], [L.STONE, 1]],
      religious: [[L.BRICK, 2], [L.STONE, 2], [L.TIMBER, 1]],
      small: [[L.TIMBER, 3], [L.METAL, 1]],
      unknown: [[L.TIMBER, 2], [L.BRICK, 2], [L.CONCRETE, 1]],
    },
    {
      pitchedRoofs: [[L.ROOF_SHINGLE, 5], [L.ROOF_METAL, 1]],
      flatRoofShare: 0.3,
      shopGroundShare: 0.2,
      wallTints: { [L.TIMBER]: [[0.88, 0.88, 0.85], [0.7, 0.75, 0.78], [0.8, 0.76, 0.66], [0.6, 0.65, 0.62]] },
    },
  ),
  generic: profile('generic', 'Generic', {
    residential: [[L.PLASTER, 3], [L.BRICK, 1], [L.CONCRETE, 1]],
    apartments: [[L.CONCRETE, 3], [L.PLASTER, 2]],
    commercial: [[L.CURTAIN, 2], [L.CONCRETE, 2], [L.PLASTER, 1]],
    industrial: [[L.METAL, 3], [L.CONCRETE, 2]],
    civic: [[L.CONCRETE, 2], [L.PLASTER, 1], [L.STONE, 1]],
    religious: [[L.STONE, 2], [L.PLASTER, 1]],
    small: [[L.PLASTER, 2], [L.TIMBER, 1], [L.METAL, 1]],
    unknown: [[L.PLASTER, 3], [L.CONCRETE, 2]],
  }),
};

/** Coarse climate/culture region of a location (lat/lon in degrees). */
export function regionIdAt(lat: number, lon: number): RegionId {
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return 'generic';
  const alat = Math.abs(lat);
  if (lon >= 95 && lon <= 150 && lat >= 0 && lat <= 55) {
    if (lat < 27) return lat < 12 ? 'tropical' : 'east_asia_subtropical';
    return 'east_asia_temperate';
  }
  if (lon >= -170 && lon <= -50 && lat >= 24 && lat <= 72) return 'north_america';
  if (lon >= -30 && lon <= 45 && lat >= 57) return 'nordic';
  if (lon >= -10 && lon <= 40 && lat >= 35 && lat < 45) return 'mediterranean';
  if (lon >= -12 && lon <= 40 && lat >= 45 && lat < 57) return 'temperate_europe';
  if (lon >= -20 && lon <= 65 && lat >= 12 && lat < 38) return 'arid';
  if (lon >= 112 && lon <= 155 && lat <= -18 && lat >= -32) return 'arid';
  if (alat < 23.5) return 'tropical';
  return 'generic';
}

export function regionProfileAt(lat: number, lon: number): RegionProfile {
  return PROFILES[regionIdAt(lat, lon)];
}

// ---------------------------------------------------------------- per-building style

export interface BuildingStyle {
  wall: LayerId;
  ground: LayerId;
  roof: LayerId;
  floorHeight: number;
  /** Linear RGB multipliers. */
  wallTint: [number, number, number];
  roofTint: [number, number, number];
  /** 0..1 roof glossiness boost (glazed tiles, metal). */
  roofGloss: number;
  /** Elevation (m, local z) the storeys start from. */
  groundZ: number;
  /** Fraction of windows lit at night. */
  litFraction: number;
  /** 0..1: how much façade detail the data supports (confidence). */
  detail: number;
  seed: number;
  group: ClassGroup;
}

const RESIDENTIAL = new Set(['house', 'detached', 'semidetached_house', 'terrace', 'residential', 'bungalow', 'farm', 'cabin', 'static_caravan']);
const APARTMENTS = new Set(['apartments', 'dormitory', 'flats']);
const COMMERCIAL = new Set(['commercial', 'retail', 'office', 'hotel', 'supermarket', 'mall', 'bank']);
const INDUSTRIAL = new Set(['industrial', 'warehouse', 'factory', 'manufacture', 'garage', 'garages', 'shed', 'service', 'greenhouse', 'hangar', 'roof', 'carport', 'storage_tank', 'barn', 'farm_auxiliary', 'transformer_tower']);
const CIVIC = new Set(['school', 'university', 'hospital', 'public', 'civic', 'government', 'train_station', 'transportation', 'college', 'kindergarten', 'fire_station', 'police', 'library', 'museum', 'stadium', 'sports_hall']);
const RELIGIOUS = new Set(['temple', 'church', 'mosque', 'shrine', 'cathedral', 'chapel', 'monastery', 'synagogue', 'religious']);
const SMALL = new Set(['toilets', 'guardhouse', 'shelter', 'hut', 'gazebo', 'pavilion', 'kiosk', 'bunker', 'gatehouse', 'ruins']);
const SHOP_AMENITIES = new Set(['restaurant', 'cafe', 'fast_food', 'bank', 'pharmacy', 'bar', 'pub', 'marketplace', 'shop', 'convenience']);
const PITCHED = new Set(['gabled', 'hipped', 'pyramidal', 'skillion', 'gambrel', 'mansard', 'half-hipped', 'saltbox', 'round', 'dome', 'onion', 'cone']);

/** FNV-1a hash of a string → [0, 1). */
export function hash01(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0) / 4294967296;
}

/** Further independent [0, 1) values from a seed. */
function rng(seed: number, k: number): number {
  const x = Math.sin(seed * 12.9898 + k * 78.233) * 43758.5453;
  return x - Math.floor(x);
}

function pick<T>(items: Weighted<T>, r: number): T {
  const total = items.reduce((s, [, w]) => s + w, 0);
  let t = r * total;
  for (const [v, w] of items) {
    if ((t -= w) < 0) return v;
  }
  return items[items.length - 1][0];
}

function num(v: unknown): number | null {
  const n = typeof v === 'number' ? v : typeof v === 'string' ? Number.parseFloat(v) : NaN;
  return Number.isFinite(n) ? n : null;
}

function value(record: EntityRecord, key: string): unknown {
  const p = record.properties?.[key];
  if (p !== undefined && p !== null) return p;
  return record.inferred?.[key]?.value;
}

export function classGroup(cls: string, amenity: string | null): ClassGroup {
  if (amenity === 'place_of_worship') return 'religious';
  if (amenity && ['shelter', 'toilets'].includes(amenity)) return 'small';
  if (RELIGIOUS.has(cls)) return 'religious';
  if (APARTMENTS.has(cls)) return 'apartments';
  if (RESIDENTIAL.has(cls)) return 'residential';
  if (COMMERCIAL.has(cls)) return 'commercial';
  if (INDUSTRIAL.has(cls)) return 'industrial';
  if (CIVIC.has(cls)) return 'civic';
  if (SMALL.has(cls)) return 'small';
  if (amenity && SHOP_AMENITIES.has(amenity)) return 'commercial';
  if (amenity && ['school', 'police', 'hospital', 'townhall', 'library', 'community_centre'].includes(amenity)) return 'civic';
  return 'unknown';
}

const MATERIAL_LAYERS: Record<string, LayerId> = {
  stone: L.STONE,
  sandstone: L.STONE,
  limestone: L.STONE,
  granite: L.STONE,
  brick: L.BRICK,
  glass: L.CURTAIN,
  mirror: L.CURTAIN,
  wood: L.TIMBER,
  timber_framing: L.TIMBER,
  timber: L.TIMBER,
  concrete: L.CONCRETE,
  reinforced_concrete: L.CONCRETE,
  metal: L.METAL,
  steel: L.METAL,
  metal_plates: L.METAL,
  plaster: L.PLASTER,
  cement_block: L.CONCRETE,
  tiles: L.CERAMIC,
  ceramic: L.CERAMIC,
};

/** Named OSM colours as realistic material colours (sRGB 0..1). */
const NAMED: Record<string, RGB> = {
  white: [0.92, 0.91, 0.88],
  black: [0.12, 0.12, 0.13],
  grey: [0.55, 0.55, 0.55],
  gray: [0.55, 0.55, 0.55],
  darkgrey: [0.32, 0.32, 0.33],
  darkgray: [0.32, 0.32, 0.33],
  lightgrey: [0.75, 0.75, 0.74],
  lightgray: [0.75, 0.75, 0.74],
  silver: [0.74, 0.75, 0.77],
  red: [0.64, 0.22, 0.16],
  darkred: [0.45, 0.14, 0.11],
  maroon: [0.45, 0.16, 0.14],
  brown: [0.45, 0.3, 0.2],
  orange: [0.85, 0.48, 0.22],
  yellow: [0.88, 0.72, 0.28],
  gold: [0.85, 0.68, 0.3],
  beige: [0.88, 0.82, 0.68],
  cream: [0.93, 0.89, 0.76],
  tan: [0.8, 0.68, 0.52],
  green: [0.24, 0.48, 0.32],
  darkgreen: [0.14, 0.3, 0.2],
  lightgreen: [0.55, 0.72, 0.5],
  olive: [0.45, 0.45, 0.25],
  blue: [0.24, 0.38, 0.62],
  darkblue: [0.14, 0.2, 0.36],
  lightblue: [0.6, 0.74, 0.86],
  navy: [0.14, 0.18, 0.32],
  teal: [0.18, 0.46, 0.46],
  pink: [0.9, 0.66, 0.66],
  purple: [0.46, 0.3, 0.5],
  terracotta: [0.7, 0.36, 0.24],
};

/** Parses an OSM colour (name or #rgb / #rrggbb) to sRGB 0..1, or null. */
export function parseColour(text: unknown): [number, number, number] | null {
  if (typeof text !== 'string') return null;
  const t = text.trim().toLowerCase().replace(/[\s_-]/g, '');
  if (!t) return null;
  if (NAMED[t]) return [...NAMED[t]] as [number, number, number];
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/.exec(t);
  if (!m) return null;
  const hex = m[1].length === 3 ? m[1].split('').map((c) => c + c).join('') : m[1];
  const rgb: [number, number, number] = [0, 1, 2].map((i) => Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16) / 255) as [number, number, number];
  // Pure display colours are not material colours: keep them out of the extremes.
  return rgb.map((c) => 0.08 + c * 0.84) as [number, number, number];
}

export function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function linear(rgb: RGB): [number, number, number] {
  return [srgbToLinear(rgb[0]), srgbToLinear(rgb[1]), srgbToLinear(rgb[2])];
}

function jitter(rgb: RGB, r: number): RGB {
  const k = 0.94 + r * 0.12;
  return [Math.min(1, rgb[0] * k), Math.min(1, rgb[1] * k), Math.min(1, rgb[2] * k)];
}

/** glTF building material name → class group (fallback before meta.json arrives). */
export function groupOfMaterial(name: string): ClassGroup {
  if (name.includes('residential')) return 'residential';
  if (name.includes('commercial')) return 'commercial';
  if (name.includes('industrial')) return 'industrial';
  if (name.includes('civic')) return 'civic';
  return 'unknown';
}

/**
 * Appearance of one building. `record` may be undefined (meta not loaded):
 * the style then follows the glTF material's class group.
 */
export function styleFor(
  id: string,
  record: EntityRecord | undefined,
  region: RegionProfile,
  fallbackGroup: ClassGroup = 'unknown',
): BuildingStyle {
  const seed = hash01(id);
  const r = (k: number): number => rng(seed, k);
  const props = record?.properties ?? {};
  const cls = String(props.class ?? 'yes').toLowerCase();
  const amenity = typeof props.amenity === 'string' ? props.amenity.toLowerCase() : null;
  const group = record ? classGroup(cls, amenity) : fallbackGroup;

  const height = record ? num(value(record, 'height')) : null;
  let levels = record ? num(value(record, 'levels')) : null;
  if (levels !== null) levels = Math.max(1, Math.round(levels));
  let floorHeight = region.defaultFloorHeight;
  if (height !== null && levels !== null) floorHeight = height / levels;
  else if (height !== null) levels = Math.max(1, Math.round(height / floorHeight));
  floorHeight = Math.min(5, Math.max(2.6, floorHeight));
  const nLevels = levels ?? 1;

  // Facade.
  let wall: LayerId;
  const material = typeof props.material === 'string' ? props.material.toLowerCase() : null;
  if (material && MATERIAL_LAYERS[material] !== undefined) {
    wall = MATERIAL_LAYERS[material];
  } else if (nLevels >= 8 && (group === 'commercial' || group === 'unknown')) {
    wall = r(1) < 0.6 ? L.CURTAIN : L.CONCRETE;
  } else if (nLevels >= 8) {
    wall = pick(region.walls.apartments, r(1));
  } else if (group === 'unknown' && nLevels >= 4) {
    wall = pick(region.walls.apartments, r(1));
  } else {
    // Low-rise buildings are rarely glass towers.
    const low = region.walls[group].filter(([l]) => l !== L.CURTAIN);
    wall = pick(nLevels < 3 && low.length ? low : region.walls[group], r(1));
  }
  if (cls === 'greenhouse') wall = L.CURTAIN;
  // Tiny utility structures rarely have real windows.
  if ((cls === 'toilets' || cls === 'shed' || cls === 'garage' || cls === 'roof') && wall === L.PLASTER) wall = L.BLANK;

  // Ground floor.
  let ground: LayerId = wall;
  const shopAmenity = amenity !== null && SHOP_AMENITIES.has(amenity);
  if (wall !== L.CURTAIN && wall !== L.METAL && wall !== L.TIMBER) {
    if (group === 'commercial' || shopAmenity) ground = L.SHOPFRONT;
    else if ((group === 'apartments' || group === 'unknown' || group === 'residential') && nLevels >= 3 && r(2) < region.shopGroundShare) {
      ground = L.SHOPFRONT;
    }
  }

  // Roof.
  const roofShape = record ? String(value(record, 'roof_shape') ?? '').toLowerCase() : '';
  let roof: LayerId;
  let roofGloss = 0;
  if (roofShape && PITCHED.has(roofShape)) roof = pick(region.pitchedRoofs, r(3));
  else if (roofShape === 'flat') roof = L.ROOF_FLAT;
  else roof = r(3) < region.flatRoofShare || nLevels >= 4 ? L.ROOF_FLAT : pick(region.pitchedRoofs, r(4));
  if (group === 'industrial' && roof !== L.ROOF_FLAT) roof = L.ROOF_METAL;
  if (roof === L.ROOF_METAL) roofGloss = 0.4;

  // Colours: tagged colours win, else the regional palette.
  const wallTags = parseColour(props.colour) ?? parseColour(props['building:colour']);
  const roofTags = parseColour(props.roof_colour) ?? parseColour(props['roof:colour']);
  const wallPalette = region.wallTints[wall] ?? WHITES;
  const roofPalette = region.roofTints[roof] ?? FLAT_ROOFS;
  let wallTint = linear(wallTags ?? jitter(wallPalette[Math.floor(r(5) * wallPalette.length)], r(6)));
  const roofTint = linear(roofTags ?? jitter(roofPalette[Math.floor(r(7) * roofPalette.length)], r(8)));
  if (wall === L.CURTAIN && !wallTags) wallTint = linear(GLASS[Math.floor(r(5) * GLASS.length)]);
  if (group === 'religious' && region.id.startsWith('east_asia') && roof !== L.ROOF_FLAT) {
    roof = L.ROOF_CLAY;
    roofGloss = 0.7; // glazed temple tiles
  }
  if (roofTags && roof === L.ROOF_CLAY && (props.roof_colour === 'green' || props.roof_colour === 'yellow')) roofGloss = 0.7;

  const litFraction = { residential: 0.35, apartments: 0.45, commercial: 0.6, industrial: 0.1, civic: 0.25, religious: 0.15, small: 0.05, unknown: 0.3 }[group];
  const confidence = typeof record?.confidence === 'number' ? record.confidence : 0.5;
  const inferredCount = Object.keys(record?.inferred ?? {}).length;
  const detail = Math.max(0.2, Math.min(1, confidence - inferredCount * 0.05));

  const groundZ =
    num(record?.inferred?.ground_elevation?.value) ?? num(props.ground_elevation) ?? (record?.centroid ? record.centroid[2] - 1 : 0);

  return { wall, ground, roof, floorHeight, wallTint, roofTint, roofGloss, groundZ, litFraction, detail, seed, group };
}

/** Texels per building in the style texture (RGBA float each). */
export const STYLE_TEXELS = 4;

/**
 * Packs styles into a Float32Array laid out as STYLE_TEXELS columns × N rows:
 * t0 (wall, ground, roof, floorHeight) · t1 (wallTint, seed) ·
 * t2 (roofTint, roofGloss) · t3 (groundZ, litFraction, detail, 0).
 */
export function packStyles(styles: readonly BuildingStyle[], out: Float32Array = new Float32Array(Math.max(1, styles.length) * STYLE_TEXELS * 4)): Float32Array {
  styles.forEach((s, i) => {
    const o = i * STYLE_TEXELS * 4;
    out.set([s.wall, s.ground, s.roof, s.floorHeight], o);
    out.set([...s.wallTint, s.seed], o + 4);
    out.set([...s.roofTint, s.roofGloss], o + 8);
    out.set([s.groundZ, s.litFraction, s.detail, 0], o + 12);
  });
  return out;
}
