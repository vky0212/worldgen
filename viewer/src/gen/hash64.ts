/**
 * The compiler's portable counter-based hash (docs/vegetation.md, "Random numbers";
 * src/worldgen/hashing.py hash_u64 / hash_unit), with BigInt: exact, for the few values
 * a viewer derives one at a time (per-tree seeds of inspected trees). Pure module.
 */

const MASK = (1n << 64n) - 1n;
const GOLDEN = 0x9e3779b97f4a7c15n;
const MIX1 = 0xbf58476d1ce4e5b9n;
const MIX2 = 0x94d049bb133111ebn;

/** hash_u64(w0, w1, ...): words are integers (two's complement 64-bit). */
export function hashU64(...words: (bigint | number)[]): bigint {
  let h = 0n;
  for (const w of words) {
    h = ((h ^ (BigInt(w) & MASK)) + GOLDEN) & MASK;
    h ^= h >> 30n;
    h = (h * MIX1) & MASK;
    h ^= h >> 27n;
    h = (h * MIX2) & MASK;
    h ^= h >> 31n;
  }
  return h;
}

/** hash_unit: the top 53 bits of hash_u64 divided by 2^53, in [0, 1). */
export function hashUnit(...words: (bigint | number)[]): number {
  return Number(hashU64(...words) >> 11n) / 2 ** 53;
}
