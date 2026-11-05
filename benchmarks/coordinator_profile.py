"""Where a parallel build's coordinator spends its time (docs/performance.md, gate 2).

Builds a world - the region and config of an already built world's manifest, sources read
offline from the cache - with cProfile on the main process only, and reports the main
process's wall time per stage and its hot spots. Time the coordinator spends waiting for its
worker processes shows as ``WaitForMultipleObjects`` / ``poll`` / ``wait``; everything else
it reports is serial work, which bounds the build's wall time however many workers it has.
With ``--workers 1`` the whole build runs in this process, workers' work included.

    uv run python benchmarks/coordinator_profile.py --world generated/n25_179_smooth --workers 8 \\
        --set export.mode=data --out build-out --stats coordinator.prof

Config keys that the current compiler no longer knows (a world built by an older version)
are dropped with a note.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import shutil
import time
from pathlib import Path
from typing import Any

from worldgen.build import BuildOptions, build
from worldgen.config import WorldConfig, deep_merge, from_dict, parse_override
from worldgen.errors import ConfigError

# Functions (path suffix, name) whose cumulative time is the serial work worth watching.
WATCH = [
    ("build.py", "write_manifest"), ("pipeline.py", "acquire"), ("osm_normalize.py", "normalize_osm"),
    ("graph.py", "build"), ("elevation.py", "compute_elevations"), ("junctions.py", "compute_junctions"),
    ("pipeline.py", "vegetation_exclusions"), ("regions.py", "extract_regions"),
    ("checks.py", "validate_world"), ("checks.py", "check_building_meshes"), ("checks.py", "check_terrain_seams"),
    ("chunked.py", "export"), ("chunked.py", "keys"), ("chunked.py", "coarse"), ("chunked.py", "merge"),
    ("chunked.py", "finish"), ("vectors.py", "write_geoparquet"), ("build.py", "export_fields"),
    ("parallel.py", "write_snapshot"),
]


def _config(world: Path, overrides: list[str]) -> dict[str, Any]:
    base = json.loads((world / "manifest.json").read_text(encoding="utf-8"))["config"]
    for expr in overrides:
        base = deep_merge(base, parse_override(expr))
    for _ in range(50):
        try:
            from_dict(WorldConfig, base)
            return base
        except ConfigError as e:
            msg = str(e)
            if "unknown keys" not in msg:
                raise
            path = msg.split(":")[0].split(".")[1:]
            node = base
            for p in path:
                node = node[p]
            for key in msg.split("unknown keys ")[1].strip("[]").replace("'", "").split(", "):
                node.pop(key, None)
                print(f"note: dropped config.{'.'.join([*path, key])} (unknown to this compiler)")
    raise SystemExit("could not clean the manifest's config")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, help="built world directory; its manifest gives region and config")
    ap.add_argument("--workers", type=int, default=8, help="parallel.workers (default 8)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, repeatable")
    ap.add_argument("--cache-dir", default=".worldgen-cache", help="source cache (read offline)")
    ap.add_argument("--out", required=True, help="output directory (replaced)")
    ap.add_argument("--stats", help="also save the cProfile statistics here")
    ap.add_argument("--top", type=int, default=30, help="functions to list by own time (default 30)")
    args = ap.parse_args()

    cfg: WorldConfig = from_dict(WorldConfig, _config(Path(args.world), args.set))
    cfg.parallel.workers = args.workers
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    res = build(cfg, BuildOptions(out, cache_dir=Path(args.cache_dir).resolve(), offline=True))
    prof.disable()
    wall = time.perf_counter() - t0
    stats = pstats.Stats(prof)
    if args.stats:
        stats.dump_stats(args.stats)
    print(f"wall {wall:.1f} s, {len(res.tiles_written)} tiles, {args.workers} worker(s)")
    print("stages (s): " + ", ".join(f"{k} {v:.2f}" for k, v in res.world.timings.items()))
    raw: dict[tuple[str, int, str], Any] = stats.stats  # type: ignore[attr-defined]
    cum: dict[str, float] = {}
    for (fn, _, name), (_, _, _, ct, _) in raw.items():
        f = fn.replace("\\", "/")
        for suffix, want in WATCH:
            if name == want and f.endswith("/" + suffix):
                cum[f"{suffix}:{name}"] = cum.get(f"{suffix}:{name}", 0.0) + ct
    print("\nwatched coordinator work (cumulative s):")
    for k, v in sorted(cum.items(), key=lambda kv: -kv[1]):
        print(f"  {v:8.2f}  {k}")
    print(f"\ntop {args.top} by own time:")
    stats.sort_stats("tottime").print_stats(args.top)


if __name__ == "__main__":
    main()
