# Performance gates

The scale-up roadmap allows native code (numba first, Rust/PyO3 only where numba gives less
than 3×) only where a measurement shows a hot kernel that vectorised Python cannot fix. This
page records the measurements behind those decisions and how to repeat them.

Target: a ~40 × 40 km world (~1,600 km², ~24,600 tiles of 256 m) built in ≤ 15 min on a
16-core machine.

## Tools

* `benchmarks/scale_bench.py`: time per stage and per validation check, peak memory and
  output size against area (`--fractions`), any config override (`--set
  parallel.workers=1`), A/B against saved results (`--baseline`). It times compile,
  validation and export, not the manifest.
* `benchmarks/coordinator_profile.py`: a full `build()` under cProfile on the main process
  only. The time that process spends waiting for its workers
  (`WaitForMultipleObjects`, `poll`, `wait`) is the parallel part. Everything else it reports
  is serial and bounds the wall time however many workers run. With `--workers 1` every chunk
  runs inline, so the profile also shows the workers' own hot spots.

## Gate 1: vegetation kernels (after phase 3)

Tree placement runs in the export's worker processes, chunk by chunk. Profiled in one
process, it costs the same per tree on every world measured: 16 µs per tree on
`n25_179_smooth` (535k trees), 19 µs on `n25_160_e121_425` (2.96M trees), profiler overhead
included. Of that, about 7 µs goes to Matérn thinning (`_matern_ii`) and about 7 µs to the
exclusion test (GEOS `contains_xy`, already native). With the default config that is about
10 % of the build's CPU time, below the gate's 25 %.

Decision: no native kernel. In a forest-dense 40 km world, placement would be ~20 % of CPU
time, spread over the workers.

## Gate 2: the serial coordinator (after phase 5)

Measured on 2026-11-05: data mode, 8 workers (the default maximum) on 16 cores. The worlds
are `n25_179_smooth` (31.7 km², 484 tiles) and `n25_160_e121_425` (71.3 km², 1,088 tiles,
built with a world-wide tree budget, so its trees are placed twice).

**Workers barely help at this size.** `n25_179_smooth` takes 56.6 s with 1 worker, 36.6 s
with 8 and 37.8 s with 15. The windowed compile stages gain only 1.8× from 8 workers, and
terrain is slower in the pool (2.4 s) than in one process (1.05 s).

**Where the coordinator spent its time** (71.3 km², before the fixes below; 128 s wall):

| serial work | s | kind |
|---|---|---|
| manifest: re-reading every output file to hash it | 11.7 | file I/O: ~3.5 ms per first read with on-access scanning |
| vegetation stage: exclusions (`union_all`), regions (`rasterize`) | 12.2 | GEOS, rasterio |
| export coordinator: merging chunks, GeoParquet, GeoTIFF, fields, snapshot | ~14 | pyarrow, I/O, pickling |
| validation: building meshes, terrain seams | 7.5 | trimesh, numpy |
| road elevations, junctions | 8.6 | Python and shapely per edge / junction |
| normalize, topology, acquire | 5.5 | OSM parsing, graph |

In the workers, tile input hashes re-normalised (`shapely.set_precision`) every large polygon
for every tile it touches. That made them 12.8× slower for 2.25× the area: quadratic.

**Decision: no native port.** None of the serial work is a numeric loop that numba or Rust
would speed up. It is file I/O, GEOS and rasterio calls, pyarrow writes and orchestration.
The fixes are structural:

1. **Manifest hashes taken as files are written** (`hashing.write_hashed`; tile and coarse
   tile files are written by the workers). The hashes are recorded in `build_state.json` with
   each file's size and modification time, and the manifest reads back only files that
   changed since or that no writer recorded.
2. **One geometry hash per entity per process** (`export.tiles.EntityIndex`), instead of
   one per entity per tile.

Both keep every output byte-identical, including the manifest's hashes. Two alternating
rounds against main:

| world | main | with the fixes | manifest |
|---|---|---|---|
| 31.7 km² | 29.6 / 28.1 s | 24.1 / 24.2 s | 5.9 → 0.4 s |
| 71.3 km² | 95.3 / 90.7 s | 77.0 / 74.0 s | 10.3 → 0.7 s |

**What remains.** At 71.3 km² the coordinator still spends ~41 s of a 78 s build on serial
work. Scaled to 1,600 km² (×22) that is ~15 min before any parallel work, so it is still the
limit. In order of size, what to move into the workers or restructure next:

1. vegetation exclusions and regions (8.4 s): per window, like the other grid stages;
2. the export's GeoParquet rows and chunk merging (~6 s): written per chunk by the workers;
3. validation's building meshes and terrain seams (5 s): per chunk;
4. road elevations and junctions (5.2 s): per connected component or tile block;
5. pool snapshot pickling (2.2 s) and OSM parsing (phase 5c, `.osm.pbf`).

Some worker-side costs still grow faster than area: vegetation region outlines are clipped
per tile (`region_outline_pieces`, `insert_tile_crossings`). Coarse tiles re-read their
level-0 tiles' files, about one first read per level-0 file per level.
