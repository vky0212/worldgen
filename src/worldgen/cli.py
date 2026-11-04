"""Command line interface.

    worldgen build --center 25.0330,121.5654 --radius 1000 --profile basic --output ./generated/taipei
    worldgen build --fixture demo_town --output ./generated/demo
    worldgen route --world ./generated/taipei --from 25.0340,121.5600 --to 25.0300,121.5700
    worldgen validate --world ./generated/taipei
    worldgen transform --world ./generated/taipei 25.0330,121.5654
    worldgen compare ./generated/a ./generated/b
    worldgen serve --world ./generated/taipei
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldgen import __version__
from worldgen.errors import WorldGenError
from worldgen.log import configure_logging, get_logger

log = get_logger("cli")


def _region_args(args: argparse.Namespace) -> dict[str, Any] | None:
    from worldgen.spatial.frame import GeoPoint

    if args.fixture:
        from worldgen.build import fixture_region

        return fixture_region(args.fixture)
    if args.bbox:
        parts = [float(v) for v in args.bbox.split(",")]
        if len(parts) != 4:
            raise WorldGenError("--bbox expects south,west,north,east")
        return {"bbox": parts, "name": args.name}
    if args.center:
        c = GeoPoint.parse(args.center)
        if args.radius is None or args.radius <= 0:
            raise WorldGenError("--radius (metres, > 0) is required with --center")
        return {"center_lat": c.lat, "center_lon": c.lon, "radius_m": float(args.radius), "name": args.name}
    return None


def cmd_build(args: argparse.Namespace) -> int:
    from worldgen.build import BuildOptions, build
    from worldgen.config import load_config

    region = _region_args(args)
    overrides = list(args.set or [])
    if args.tile_size:
        overrides.append(f"tiling.tile_size={args.tile_size}")
    cfg = load_config(args.profile, args.config, overrides, region)
    only = set(args.tiles.split(",")) if args.tiles else None
    opts = BuildOptions(Path(args.output), Path(args.cache_dir), args.offline, args.fixture, only, args.force)
    if args.progress == "jsonl":
        return _build_jsonl(cfg, opts)
    res = build(cfg, opts)
    m = res.manifest
    s = m["statistics"]
    v = m["validation"]
    print(f"World written to {opts.output.resolve()}")
    print(f"  region     : {m['region']['center']} r={m['region']['radius_m']} m, {m['tiling']['tile_count']} tiles "
          f"({len(res.tiles_written)} built, {len(res.tiles_skipped)} unchanged)")
    print(f"  roads      : {s['roads']} ways, graph {s['graph_nodes']} nodes / {s['graph_edges']} edges, "
          f"{s['bridges']} bridges, {s['tunnels']} tunnels, {s['grade_separated_crossings']} grade separations")
    print(f"  buildings  : {s['buildings']} ({s['buildings_with_observed_height']} with observed height)")
    print(f"  vegetation : {s['vegetation_regions']} regions, {s['trees']} trees/plants ({s['trees_observed']} observed)")
    print(f"  validation : {v['errors']} errors, {v['warnings']} warnings, {v['info']} info -> diagnostics.json")
    for src in m["sources"]:
        print(f"  source     : {src['name']:<14} {src.get('status', 'ok'):<9} {src.get('dataset_version', '')}")
    return 0


def _build_jsonl(cfg, opts) -> int:  # noqa: ANN001
    """Run a build emitting progress events as JSON lines on stdout (docs/studio-api.md)."""
    import logging

    from worldgen.build import build
    from worldgen.progress import Progress, jsonl_sink, use

    logger = logging.getLogger("worldgen")
    logger.setLevel(logging.INFO)
    for h in logger.handlers:
        h.setLevel(logging.CRITICAL + 1)   # every log record travels as a ``log`` event instead
    progress = Progress(jsonl_sink(sys.stdout))
    with use(progress):
        try:
            res = build(cfg, opts)
        except Exception as exc:  # report every failure as an event, then a non-zero exit
            progress.emit("error", message=f"{type(exc).__name__}: {exc}")
            return 1
        m = res.manifest
        stats = {k: v for k, v in m["statistics"].items() if isinstance(v, (int, float))}
        progress.emit("done", output=str(opts.output.resolve()), summary={
            "statistics": stats, "validation": m["validation"], "tiles_built": len(res.tiles_written),
            "tiles_unchanged": len(res.tiles_skipped), "duration_s": m["build"]["duration_s"],
        }, overall=1.0)
    return 0


def _load_graph(world: Path):  # noqa: ANN202
    from worldgen.topology.graph import RoadGraph

    doc = json.loads((world / "graph" / "road_graph.json").read_text(encoding="utf-8"))
    return RoadGraph.from_dict(doc)


def _frame_of(world: Path):  # noqa: ANN202
    from worldgen.spatial.frame import GeoPoint, LocalFrame

    m = json.loads((world / "manifest.json").read_text(encoding="utf-8"))
    o = m["origin"]
    return LocalFrame(GeoPoint(o["lat"], o["lon"]), m["crs"]["local"]["method"]), m


def cmd_route(args: argparse.Namespace) -> int:
    from worldgen.spatial.frame import GeoPoint

    world = Path(args.world)
    g = _load_graph(world)
    frame, _ = _frame_of(world)
    mode = None if args.mode == "any" else args.mode
    a = frame.point_to_local(GeoPoint.parse(getattr(args, "from")))
    b = frame.point_to_local(GeoPoint.parse(args.to))
    na = g.nearest_node(*a, mode=mode)
    nb = g.nearest_node(*b, mode=mode)
    if na is None or nb is None:
        print("no road usable by this mode in the world", file=sys.stderr)
        return 2
    res = g.shortest_path(na, nb, mode=mode, restrictions=not args.ignore_restrictions)
    if res is None:
        out = {"found": False, "from_node": na, "to_node": nb, "mode": args.mode}
        print(json.dumps(out, indent=2) if args.json else f"no route from {na} to {nb} for mode {args.mode}")
        return 1
    nodes, edges, length = res
    roads = []
    for e in edges:
        rid = g.edges[e].road_id
        if not roads or roads[-1] != rid:
            roads.append(rid)
    structures = sorted({g.edges[e].structure_id for e in edges if g.edges[e].structure_id})
    if args.json:
        coords = []
        for e, n0 in zip(edges, nodes[:-1]):
            ed = g.edges[e]
            c = ed.coords if ed.u == n0 else ed.coords[::-1]
            coords += [[float(x), float(y)] for x, y in c]
        lon, lat = frame.to_wgs84([c[0] for c in coords], [c[1] for c in coords]) if coords else ([], [])
        print(json.dumps({"found": True, "length_m": round(length, 2), "nodes": nodes, "edges": edges,
                          "roads": roads, "structures": structures,
                          "geometry_wgs84": [[round(float(x), 7), round(float(y), 7)] for x, y in zip(lon, lat)]},
                         indent=2))
    else:
        print(f"route {na} -> {nb}: {length:.1f} m over {len(edges)} edges ({args.mode})")
        names = []
        for rid in roads:
            e = next(x for x in g.edges.values() if x.road_id == rid)
            names.append(f"{e.name or rid} [{e.road_class}{', bridge' if e.bridge else ''}{', tunnel' if e.tunnel else ''}]")
        for n in names:
            print(f"  via {n}")
        if structures:
            print(f"  structures: {', '.join(structures)}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from worldgen.validation.assets import validate_outputs

    res = validate_outputs(Path(args.world))
    for line in res["lines"]:
        print(line)
    return 0 if res["errors"] == 0 else 1


def cmd_compare(args: argparse.Namespace) -> int:
    from worldgen.validation.assets import compare_worlds

    res = compare_worlds(Path(args.a), Path(args.b))
    for line in res["lines"]:
        print(line)
    return 0 if res["identical"] else 1


def cmd_transform(args: argparse.Namespace) -> int:
    from worldgen.spatial.frame import GeoPoint

    frame, m = _frame_of(Path(args.world))
    for p in args.points:
        if args.inverse:
            x, y = (float(v) for v in p.split(","))
            g = frame.point_to_wgs84(x, y)
            print(f"{x:.3f},{y:.3f} -> {g.lat:.8f},{g.lon:.8f}")
        else:
            g = GeoPoint.parse(p)
            x, y = frame.point_to_local(g)
            back = frame.point_to_wgs84(x, y)
            err = abs(back.lat - g.lat) + abs(back.lon - g.lon)
            print(f"{g.lat},{g.lon} -> x={x:.3f} y={y:.3f} (round-trip error {err:.1e} deg)")
    return 0


def cmd_fixtures(args: argparse.Namespace) -> int:
    from worldgen.sources.synthetic import FIXTURES, load_fixture

    for name in sorted(FIXTURES):
        fx = load_fixture(name)
        print(f"{name:<24} r={fx.radius_m:>5.0f} m  {fx.description}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from worldgen.serve import serve

    return serve(Path(args.world) if args.world else None, args.port, Path(args.viewer) if args.viewer else None,
                 args.open, Path(args.output_root), Path(args.cache_dir))


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="worldgen", description="Geospatial world compiler")
    p.add_argument("--version", action="version", version=f"worldgen {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="compile a geographic region into a world")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--center", help="lat,lon of the region centre (WGS84)")
    g.add_argument("--bbox", help="south,west,north,east (WGS84)")
    g.add_argument("--fixture", help="build a synthetic test scenario instead of real data")
    b.add_argument("--radius", type=float, help="region radius in metres (with --center)")
    b.add_argument("--name", help="optional region name")
    b.add_argument("--profile", default="basic", help="profile name (configs/<name>.yaml) or YAML path")
    b.add_argument("--config", help="extra YAML configuration merged over the profile")
    b.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    b.add_argument("--tile-size", type=float, help="tile size in metres")
    b.add_argument("--output", "-o", required=True, help="output directory")
    b.add_argument("--cache-dir", default=".worldgen-cache", help="source cache directory")
    b.add_argument("--offline", action="store_true", help="use cached source data only (no network)")
    b.add_argument("--tiles", help="comma-separated tile ids to (re)build, e.g. 0_0,1_0")
    b.add_argument("--force", action="store_true", help="rebuild tiles even when their inputs are unchanged")
    b.add_argument("--progress", choices=["none", "jsonl"], default="none",
                   help="jsonl: write progress events (docs/studio-api.md) to stdout instead of a summary")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("route", help="shortest path on a compiled world's road graph")
    r.add_argument("--world", required=True)
    r.add_argument("--from", required=True, help="lat,lon")
    r.add_argument("--to", required=True, help="lat,lon")
    r.add_argument("--mode", default="motor_vehicle", choices=["motor_vehicle", "bicycle", "foot", "any"])
    r.add_argument("--ignore-restrictions", action="store_true")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_route)

    v = sub.add_parser("validate", help="re-validate the files of a compiled world")
    v.add_argument("--world", required=True)
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("compare", help="compare two compiled worlds (sources, config, outputs)")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)

    t = sub.add_parser("transform", help="convert lat,lon to local metres (or back with --inverse)")
    t.add_argument("--world", required=True)
    t.add_argument("--inverse", action="store_true", help="input is x,y local metres")
    t.add_argument("points", nargs="+")
    t.set_defaults(func=cmd_transform)

    f = sub.add_parser("fixtures", help="list the synthetic test scenarios")
    f.set_defaults(func=cmd_fixtures)

    s = sub.add_parser("serve", help="serve the viewer, compiled worlds and the generation API")
    s.add_argument("--world", help="compiled world served at /world/ (optional)")
    s.add_argument("--output-root", default="generated", help="directory of worlds built from the browser")
    s.add_argument("--cache-dir", default=".worldgen-cache", help="source cache used by browser builds")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--viewer", help="path to the built viewer (default: viewer/dist)")
    s.add_argument("--open", action="store_true", help="open a browser")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    # Street names are often non-Latin; never crash on a legacy console encoding.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    parser = make_parser()
    args = parser.parse_args(argv)
    configure_logging(-1 if args.quiet else args.verbose)
    try:
        return int(args.func(args))
    except WorldGenError as exc:
        log.error("%s", exc)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
