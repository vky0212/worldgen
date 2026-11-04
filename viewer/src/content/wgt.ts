/**
 * Data-first tile container (`.wgt`, format 2.x) and its terrain sections.
 * Contract: docs/output-format.md, "Data-first tiles"; reference implementation:
 * src/worldgen/export/container.py. Pure module (no three.js, no DOM).
 */

export interface WgtContainer {
  level: number;
  i: number;
  j: number;
  /** Section payloads by tag (trailing spaces removed), already inflated. */
  sections: Map<string, Uint8Array>;
}

const MAGIC = 'WGT1';
const HEADER_BYTES = 24;
const SECTION_BYTES = 16;
const FLAG_DEFLATE = 1;

async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  // slice(): a copy backed by a plain ArrayBuffer, which every Blob typing accepts.
  const stream = new Blob([data.slice()]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function ascii(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += String.fromCharCode(b);
  return s;
}

/** Parses a container and inflates its compressed sections. */
export async function readWgt(buffer: ArrayBuffer): Promise<WgtContainer> {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  if (buffer.byteLength < HEADER_BYTES || ascii(bytes.subarray(0, 4)) !== MAGIC) {
    throw new Error('not a WorldGen tile container (.wgt)');
  }
  const version = view.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported .wgt version ${version}`);
  const count = view.getUint16(6, true);
  const level = view.getUint8(8);
  const i = view.getInt32(12, true);
  const j = view.getInt32(16, true);
  const sections = new Map<string, Uint8Array>();
  for (let k = 0; k < count; k++) {
    const at = HEADER_BYTES + k * SECTION_BYTES;
    const tag = ascii(bytes.subarray(at, at + 4)).replace(/ +$/, '');
    const offset = view.getUint32(at + 4, true);
    const length = view.getUint32(at + 8, true);
    const flags = view.getUint16(at + 12, true);
    if (offset + length > buffer.byteLength) throw new Error(`.wgt section ${tag} runs past the end of the file`);
    const raw = bytes.subarray(offset, offset + length);
    sections.set(tag, flags & FLAG_DEFLATE ? await inflateRaw(raw) : raw);
  }
  return { level, i, j, sections };
}

/** Decoded `HGT `: heights (m) row 0 south, with a one-node apron on every side. */
export interface HeightGrid {
  nx: number;
  ny: number;
  apron: number;
  spacing: number;
  quantum: number;
  /** Local ENU of node (0, 0): the tile corner minus the apron. */
  x0: number;
  y0: number;
  z: Float64Array;
}

const HGT_HEADER = 48;

export function decodeHeights(data: Uint8Array): HeightGrid {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const nx = view.getUint16(0, true);
  const ny = view.getUint16(2, true);
  const dtype = view.getUint8(4);
  const apron = view.getUint8(5);
  const spacing = view.getFloat64(8, true);
  const quantum = view.getFloat64(16, true);
  const offset = Number(view.getBigInt64(24, true));
  const x0 = view.getFloat64(32, true);
  const y0 = view.getFloat64(40, true);
  const n = nx * ny;
  const width = dtype === 1 ? 2 : dtype === 2 ? 4 : 0;
  if (!width) throw new Error(`unknown HGT dtype ${dtype}`);
  if (HGT_HEADER + n * width > data.byteLength) throw new Error('HGT section is truncated');
  const z = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const v = width === 2 ? view.getUint16(HGT_HEADER + 2 * k, true) : view.getUint32(HGT_HEADER + 4 * k, true);
    // (offset + value) * quantum in float64: bit for bit the compiler's rounded surface.
    z[k] = (offset + v) * quantum;
  }
  return { nx, ny, apron, spacing, quantum, x0, y0, z };
}

/** `RGB ` (channels 3) or `LCC ` (channels 1): u16 nx, u16 ny, then u8 per node. */
export function decodeGrid(data: Uint8Array, channels: number): { nx: number; ny: number; data: Uint8Array } {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const nx = view.getUint16(0, true);
  const ny = view.getUint16(2, true);
  const n = nx * ny * channels;
  if (4 + n > data.byteLength) throw new Error('grid section is truncated');
  return { nx, ny, data: data.subarray(4, 4 + n) };
}
