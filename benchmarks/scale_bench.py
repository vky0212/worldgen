"""Scale benchmark: build time, peak memory and output size against region area.

Builds centred sub-regions of one region from a single set of source data, each in its own
process (so peak memory is per run), and reports time per stage and per validation check,
peak memory and output size, normalised per km². Sources are read from the cache only
(``--world``) or from a synthetic fixture (``--fixture``), so a run never touches the network
and repeated runs measure the same input.

    # sub-regions of an already built world (its manifest gives region and config)
    uv run python benchmarks/scale_bench.py --world generated/taipei --fractions 0.25,0.5,1

    # a cheaper level of detail for comparison
    uv run python benchmarks/scale_bench.py --world generated/taipei --fractions 1 \
        --set terrain.resolution=16 --set vegetation.enabled=false --label far-lod

    # save results and compare a later run (another branch, a native kernel) against them
    uv run python benchmarks/scale_bench.py --world generated/taipei --json before.json
    uv run python benchmarks/scale_bench.py --world generated/taipei --baseline before.json

``--fractions`` are side-length fractions of the region's bounding box (0.5 = a quarter of
the area). Compile and validation run in memory; export writes to a temporary directory that
is removed afterwards unless ``--keep`` is given.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "SCALE_BENCH_RESULT "


# --------------------------------------------------------------------------------------
# peak memory of this process


def _win_counters() -> Any:
    import ctypes
    import ctypes.wintypes as wt

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wt.HANDLE
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(Counters), wt.DWORD]
    psapi.GetProcessMemoryInfo.restype = wt.BOOL
    c = Counters()
    c.cb = ctypes.sizeof(c)
    return c if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) else None


def peak_rss_mb() -> float | None:
    if sys.platform == "win32":
        c = _win_counters()
        return None if c is None else c.PeakWorkingSetSize / 2**20
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10  # bytes on macOS, KiB on Linux


def current_rss_mb() -> float | None:
    if sys.platform == "win32":
        c = _win_counters()
        return None if c is None else c.WorkingSetSize / 2**20
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        return None  # no cheap current-RSS reading on this platform


class StagePeaks:
    """Largest working set seen while each stage runs, sampled by a background thread.

    Short spikes between samples can be missed; the process-wide ``peak_mb`` is exact.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.stage = "setup"
        self.peaks: dict[str, float] = {}
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample()

    def sample(self) -> None:
        m = current_rss_mb()
        if m is not None and m > self.peaks.get(self.stage, 0.0):
            self.peaks[self.stage] = m

    def enter(self, name: str) -> None:
        self.sample()
        self.stage = name
        self.sample()

    def start(self) -> StagePeaks:
        self._thread.start()
        return self

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()
        self.sample()
        return {k: round(v, 1) for k, v in self.peaks.items()}

    def track(self, stage_cm: Any) -> Any:
        """Wrap a ``worldgen.log.stage``-style context manager so samples go to its stage."""
        peaks = self

        @contextlib.contextmanager
        def tracked(name: str, timings: dict[str, float]) -> Any:
            prev = peaks.stage
            peaks.enter(name)
            try:
                with stage_cm(name, timings):
                    yield
            finally:
                peaks.enter(prev)

        return tracked


# --------------------------------------------------------------------------------------
# one measurement (child process)


def _output_sizes(out: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for p in out.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(out).as_posix()
        if rel.startswith("tiles/"):
            name = p.name
            if name.endswith(".collision.glb"):
                key = "tiles_collision"
            elif name.endswith(".glb"):
                key = "tiles_glb"
            elif name.endswith(".meta.json"):
                key = "tiles_meta"
            elif name.endswith(".ortho.jpg"):
                key = "tiles_ortho"
            else:
                key = "tiles_other"
        elif rel == "world.glb":
            key = "world_glb"
        else:
            key = rel.split("/")[0] if "/" in rel else "other"
        sizes[key] = sizes.get(key, 0) + p.stat().st_size
    return dict(sorted(sizes.items()))


def measure(args: argparse.Namespace) -> dict[str, Any]:
    from worldgen import pipeline
    from worldgen.build import BuildOptions, export_world, fixture_region
    from worldgen.config import WorldConfig, deep_merge, from_dict, load_config, parse_override
    from worldgen.pipeline import acquire, bundle_from_fixture, compile_world
    from worldgen.sources.synthetic import load_fixture
    from worldgen.spatial.region import Region
    from worldgen.validation import checks

    peaks = StagePeaks().start()
    pipeline.stage = peaks.track(pipeline.stage)  # compile_world's stages report their memory
    t_start = time.perf_counter()
    if args.world:
        manifest = json.loads(Path(args.world, "manifest.json").read_text(encoding="utf-8"))
        base = manifest["config"]
    else:
        base = load_config("basic", region=fixture_region(args.fixture)).to_dict()
    for expr in args.set:
        base = deep_merge(base, parse_override(expr))
    cfg_full: WorldConfig = from_dict(WorldConfig, base)
    region_full = Region.from_config(cfg_full)

    t = time.perf_counter()
    if args.world:
        bundle = acquire(region_full, cfg_full, args.cache_dir, offline=True)
    else:
        bundle = bundle_from_fixture(load_fixture(args.fixture, region_full.center, cfg_full.projection.method))
    t_acquire = time.perf_counter() - t
    sources = {r.name: r.status for r in bundle.records}

    w, s, e, n = region_full.describe()["bbox_wgs84"]
    cy, cx = (s + n) / 2.0, (w + e) / 2.0
    hw, hh = (e - w) / 2.0 * args.fraction, (n - s) / 2.0 * args.fraction
    cfg_dict = deep_merge(base, {"region": {"bbox": [cy - hh, cx - hw, cy + hh, cx + hw], "center_lat": None,
                                            "center_lon": None, "radius_m": None}})
    cfg: WorldConfig = from_dict(WorldConfig, cfg_dict)
    cfg.validate()
    region = Region.from_config(cfg)
    x0, y0, x1, y1 = region.extent
    km2 = (x1 - x0) * (y1 - y0) / 1e6

    t = time.perf_counter()
    world = compile_world(region, cfg, bundle)
    t_compile = time.perf_counter() - t

    check_times: dict[str, float] = {}
    run = checks.Report.run

    def timed_run(self, name, fn, details=None):  # noqa: ANN001, ANN202
        t0 = time.perf_counter()
        run(self, name, fn, details)
        check_times[name] = round(time.perf_counter() - t0, 3)

    checks.Report.run = timed_run
    peaks.enter("validate")
    try:
        t = time.perf_counter()
        report = checks.validate_world(world)
        t_validate = time.perf_counter() - t
    finally:
        checks.Report.run = run

    peaks.enter("export")
    t_export = None
    sizes: dict[str, int] = {}
    if not args.no_export:
        if args.keep:
            out = Path(args.keep)
            if out.exists() and any(out.iterdir()):
                raise SystemExit(f"--keep directory {out} is not empty; choose a new or empty directory")
        else:
            out = Path(tempfile.mkdtemp(prefix="scale_bench_"))
        try:
            t = time.perf_counter()
            export_world(world, report, BuildOptions(output=out, force=True))
            t_export = time.perf_counter() - t
            sizes = _output_sizes(out)
        finally:
            if not args.keep:
                shutil.rmtree(out, ignore_errors=True)

    stage_peaks = peaks.stop()
    tr = world.trees
    return {
        "fraction": args.fraction,
        "tiles": len(region.tiles()),
        "km2": round(km2, 3),
        "acquire_s": round(t_acquire, 3),
        "compile_s": round(t_compile, 3),
        "validate_s": round(t_validate, 3),
        "export_s": None if t_export is None else round(t_export, 3),
        "total_s": round(time.perf_counter() - t_start, 3),
        "stages_s": {k: round(v, 3) for k, v in world.timings.items()},
        "checks_s": check_times,
        "peak_mb": None if (p := peak_rss_mb()) is None else round(p, 1),
        # Working set sampled during each stage ("setup" = loading sources and config).
        "stage_peak_mb": stage_peaks,
        "output_bytes": sizes,
        "counts": {"trees": len(tr), "trees_rendered": int(tr.rendered_mask().sum()) if len(tr) else 0,
                   "buildings": len(world.buildings), "graph_edges": len(world.graph.edges),
                   "heightfield": list(world.terrain.z.shape)},
        "sources": sources,
    }


# --------------------------------------------------------------------------------------
# driver (parent process)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _work_s(r: dict[str, Any]) -> float:
    """Compile + validate + export: the part that scales with the region (acquire is a cache read)."""
    return r["compile_s"] + r["validate_s"] + (r["export_s"] or 0.0)


def _row(r: dict[str, Any], base: dict[str, Any] | None) -> str:
    work = _work_s(r)
    out_mb = sum(r["output_bytes"].values()) / 1e6
    top = sorted(r["stages_s"].items(), key=lambda kv: -kv[1])[:2]
    top += sorted(r["checks_s"].items(), key=lambda kv: -kv[1])[:1]
    cols = [f"{r['fraction']:>5g}", f"{r['tiles']:>6d}", f"{r['km2']:>7.2f}", f"{r['compile_s']:>8.1f}",
            f"{r['validate_s']:>8.1f}", f"{r['export_s'] or 0:>7.1f}", f"{work:>7.1f}",
            f"{work / r['km2']:>6.2f}", f"{(r['peak_mb'] or 0):>7.0f}", f"{out_mb / r['km2']:>7.1f}"]
    line = "  ".join(cols) + "   " + ", ".join(f"{k} {v:.1f}" for k, v in top)
    if r.get("stage_peak_mb"):
        stage, mb = max(r["stage_peak_mb"].items(), key=lambda kv: kv[1])
        line += f"   peak in {stage}"
    if base is not None:
        line += f"   [x{_work_s(base) / max(work, 1e-9):.2f} vs baseline]"
    return line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--world", help="built world directory; its manifest gives region and config")
    src.add_argument("--fixture", help="synthetic fixture name (worldgen fixtures)")
    ap.add_argument("--fractions", default="1", help="comma-separated side-length fractions (default 1)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override, repeatable")
    ap.add_argument("--cache-dir", default=".worldgen-cache", help="source cache (read offline)")
    ap.add_argument("--no-export", action="store_true", help="skip the export stage")
    ap.add_argument("--keep", metavar="DIR", help="keep the exported files in DIR (child runs only use it)")
    ap.add_argument("--label", default="", help="free text stored with the results")
    ap.add_argument("--json", metavar="PATH", help="write all results to this JSON file")
    ap.add_argument("--baseline", metavar="PATH", help="earlier --json results to compare against")
    ap.add_argument("--fraction", type=float, help=argparse.SUPPRESS)  # child process
    args = ap.parse_args()

    if args.fraction is not None:
        print(RESULT_PREFIX + json.dumps(measure(args)), flush=True)
        return

    fractions = [float(f) for f in args.fractions.split(",") if f.strip()]
    baseline: dict[float, dict[str, Any]] = {}
    if args.baseline:
        for r in json.loads(Path(args.baseline).read_text(encoding="utf-8"))["runs"]:
            baseline[float(r["fraction"])] = r
    root = Path(__file__).resolve().parents[1]
    results = []
    print("frac   tiles      km2   compile  validate   export     work   s/km2  peak MB  MB/km2   slowest")
    for f in fractions:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--fraction", repr(f), "--cache-dir", args.cache_dir]
        cmd += ["--world", args.world] if args.world else ["--fixture", args.fixture]
        for expr in args.set:
            cmd += ["--set", expr]
        if args.no_export:
            cmd.append("--no-export")
        if args.keep:
            cmd += ["--keep", str(Path(args.keep) / f"f{f:g}")]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)), None)
        if proc.returncode != 0 or line is None:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            raise SystemExit(f"benchmark run for fraction {f} failed (exit code {proc.returncode})")
        r = json.loads(line[len(RESULT_PREFIX):])
        missing = sorted(k for k, v in r["sources"].items() if v in ("failed", "missing"))
        if missing:
            print(f"  warning: sources {', '.join(missing)} not in the cache; results lack that data")
        results.append(r)
        print(_row(r, baseline.get(f)), flush=True)

    doc = {
        "schema": "worldgen.scale_bench", "label": args.label,
        "input": {"world": args.world, "fixture": args.fixture, "overrides": args.set},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "cpus": os.cpu_count(), **_git_state(root)},
        "runs": results,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"results written to {args.json}")


if __name__ == "__main__":
    main()
