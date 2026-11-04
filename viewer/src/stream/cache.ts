/**
 * Least-recently-used cache with a cost budget (bytes of GPU memory for tile
 * content, or a document count for meta.json).
 *
 * Nothing is evicted implicitly: `trim()` evicts least recently used entries
 * that are not pinned until the total fits the budget. The streamer pins what
 * it draws, the ancestors of what it draws and visible tiles still waiting for
 * their siblings, so eviction never opens a hole in the current view.
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */

export const DEFAULT_GPU_BUDGET = 1024 * 1024 * 1024;

interface Entry<V> {
  value: V;
  cost: number;
}

export class LruCache<V> {
  budget: number;
  private readonly map = new Map<string, Entry<V>>();
  private totalCost = 0;
  private readonly onEvict: ((id: string, value: V) => void) | null;
  evictedCount = 0;

  constructor(budget: number, onEvict?: (id: string, value: V) => void) {
    this.budget = budget;
    this.onEvict = onEvict ?? null;
  }

  get total(): number {
    return this.totalCost;
  }

  get size(): number {
    return this.map.size;
  }

  has(id: string): boolean {
    return this.map.has(id);
  }

  /** Value without changing the recency order. */
  get(id: string): V | undefined {
    return this.map.get(id)?.value;
  }

  costOf(id: string): number {
    return this.map.get(id)?.cost ?? 0;
  }

  /** Adds or replaces an entry as the most recently used. */
  set(id: string, value: V, cost: number): void {
    const prev = this.map.get(id);
    if (prev) {
      this.totalCost -= prev.cost;
      this.map.delete(id);
    }
    const c = Math.max(0, cost);
    this.map.set(id, { value, cost: c });
    this.totalCost += c;
  }

  /** Updates an entry's cost (e.g. a texture was added to a tile). */
  setCost(id: string, cost: number): void {
    const e = this.map.get(id);
    if (!e) return;
    const c = Math.max(0, cost);
    this.totalCost += c - e.cost;
    e.cost = c;
  }

  /** Marks an entry as the most recently used. */
  touch(id: string): void {
    const e = this.map.get(id);
    if (!e) return;
    this.map.delete(id);
    this.map.set(id, e);
  }

  /** Removes an entry without calling onEvict; returns its value. */
  delete(id: string): V | undefined {
    const e = this.map.get(id);
    if (!e) return undefined;
    this.map.delete(id);
    this.totalCost -= e.cost;
    return e.value;
  }

  /** Ids from least to most recently used. */
  keys(): string[] {
    return [...this.map.keys()];
  }

  /**
   * Evicts unpinned entries, least recently used first, until the total
   * fits `budget`. Pinned entries are never evicted, so the total can stay
   * above the budget when everything left is pinned. Returns the evicted ids.
   */
  trim(isPinned: (id: string) => boolean = () => false, budget = this.budget): string[] {
    const evicted: string[] = [];
    if (this.totalCost <= budget) return evicted;
    for (const [id, e] of [...this.map]) {
      if (this.totalCost <= budget) break;
      if (isPinned(id)) continue;
      this.map.delete(id);
      this.totalCost -= e.cost;
      this.evictedCount++;
      evicted.push(id);
      this.onEvict?.(id, e.value);
    }
    return evicted;
  }
}

/**
 * Parses a budget in megabytes ("512", "1536MB", "1.5GB"); null when absent
 * or invalid.
 */
export function parseBudgetMb(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const m = /^\s*(\d+(?:\.\d+)?)\s*(mb|m|gb|g)?\s*$/i.exec(raw);
  if (!m) return null;
  const v = Number(m[1]);
  if (!Number.isFinite(v) || v <= 0) return null;
  const unit = (m[2] ?? 'mb').toLowerCase();
  const mb = unit.startsWith('g') ? v * 1024 : v;
  return Math.round(mb * 1024 * 1024);
}
