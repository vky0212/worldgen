/**
 * Entity locator of data-first worlds (`tiles/locate/<bucket>.json`, docs/output-format.md):
 * which tiles record an entity, for ids not in any loaded tile. Pure apart from the injected
 * loader; buckets are fetched once and cached.
 */

/** tileset.json `locate`. */
export interface LocateEntry {
  uri: string;
  buckets: number;
  hash?: string;
  ids?: number;
}

export interface LocateDoc {
  bucket?: number;
  entities: Record<string, string[]>;
}

/** FNV-1a 32 of the UTF-8 bytes of `text`. */
export function fnv1a32(text: string): number {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(text)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h >>> 0;
}

export function bucketOf(id: string, buckets: number): number {
  return fnv1a32(id) & (buckets - 1);
}

/**
 * Tile of ids the locator does not list because the id gives it: `terrain:<i>_<j>`, and
 * generated trees `tree:gen:<cx>_<cy>_...` when tiles are whole multiples of the sampling cell.
 */
export function derivedTile(id: string, tileSize: number, cellSize?: number): string | null {
  const t = /^terrain:(-?\d+)_(-?\d+)$/.exec(id);
  if (t) return `${Number(t[1])}_${Number(t[2])}`;
  const g = /^tree:gen:(-?\d+)_(-?\d+)_g?\d+$/.exec(id);
  if (g && cellSize && tileSize % cellSize === 0) {
    const per = tileSize / cellSize;
    return `${Math.floor(Number(g[1]) / per)}_${Math.floor(Number(g[2]) / per)}`;
  }
  return null;
}

export class Locator {
  private readonly entry: LocateEntry;
  private readonly load: (path: string) => Promise<LocateDoc | null>;
  private readonly docs = new Map<number, Promise<LocateDoc | null>>();

  constructor(entry: LocateEntry, load: (path: string) => Promise<LocateDoc | null>) {
    this.entry = entry;
    this.load = load;
  }

  /** Tiles recording `id` (empty when it is unknown or its bucket is unavailable). */
  async tilesOf(id: string): Promise<string[]> {
    const b = bucketOf(id, Math.max(1, this.entry.buckets));
    let doc = this.docs.get(b);
    if (!doc) {
      doc = this.load(this.entry.uri.replace('{bucket}', String(b))).catch(() => null);
      this.docs.set(b, doc);
    }
    return (await doc)?.entities[id] ?? [];
  }
}
