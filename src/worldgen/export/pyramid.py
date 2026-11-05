"""Coarser levels of detail of a data-first world (format 2.4): the terrain pyramid.

A tile of level ``L`` (1 to 4) stands for the ``2**L x 2**L`` level-0 tiles of its square,
``tile_size * 2**L`` metres on a side, and exists where any of them does. It has as many
heightmap nodes per side as a level-0 tile, ``2**L`` times further apart:

* heights: the level-0 heights at every ``2**L``-th node. The levels are nested, so a coarse
  vertex is a level-0 vertex and every level samples one surface;
* colours: ``heightmap.level_colors``, a binomial pyramid of the colours level 0 shows from
  above (the terrain colour, with the water surface blended over water);
* land-cover codes: the level-0 codes at the coarse nodes;
* orthophoto: the imagery baked for the tile's square at the level's pixel size, with the
  level-0 colour levels.

A coarse tile covers only its level-0 tiles: consumers leave out the cells over missing ones.

Geometric error (``err``): how far drawing the tile may be from the full-detail world it
stands for, in metres. It is the largest of

* ``terr``, the terrain error: the largest height difference between the level-0 heightmap and
  the coarse mesh at the level-0 nodes of the tile's level-0 tiles;
* the height of what the tile leaves out: the ``lift`` of its level-0 tiles, the largest height
  of their buildings, trees, bridges and other layers above the terrain;
* half the tile's node spacing;

and never less than a child's. Viewers refine a tile when its error projects to more pixels
than they accept.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from worldgen.export.container import write_container
from worldgen.export.heightmap import level_node_range, terrain_sections, tile_node_range
from worldgen.export.tileset import DATA_SCHEMA_VERSION, SUBTREE_LEVEL, level_paths
from worldgen.hashing import sha256_json
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import TileGrid, parse_tile_id, tile_id

# Part of every coarse tile's input hash: bump when what a coarse tile holds changes.
PYRAMID_VERSION = 1

Key = tuple[int, int]


def top_level(max_levels: int, tile_size: float, resolution: float) -> int:
    """Highest level to build: at most ``max_levels`` and the subtree level, and only levels
    whose lattice (every ``2**L``-th node) has a node on every tile corner."""
    nodes = int(round(tile_size / resolution))
    if nodes <= 0 or abs(nodes * resolution - tile_size) > 1e-6 * tile_size:
        return 0
    top = 0
    while top < min(max_levels, SUBTREE_LEVEL) and nodes % (2 << top) == 0:
        top += 1
    return top


def plan_levels(tiles: list[str], top: int) -> dict[int, dict[Key, list[Key]]]:
    """Level -> coarse tile -> its level-0 tiles, all in (j, i) order. Levels stop at the first
    one that has as many tiles as the level below: its tiles would stand for the same level-0
    tiles as their only children, at a lower resolution."""
    leaves = [parse_tile_id(t) for t in tiles]
    out: dict[int, dict[Key, list[Key]]] = {}
    count = len(leaves)
    for level in range(1, top + 1):
        groups: dict[Key, list[Key]] = {}
        for i, j in leaves:
            groups.setdefault((i >> level, j >> level), []).append((i, j))
        if len(groups) >= count:
            break
        out[level] = {k: sorted(groups[k], key=lambda t: (t[1], t[0]))
                      for k in sorted(groups, key=lambda t: (t[1], t[0]))}
        count = len(groups)
    return out


def level_bounds(tile_size: float, level: int, i: int, j: int) -> tuple[float, float, float, float]:
    s = tile_size * (1 << level)
    return (i * s, j * s, (i + 1) * s, (j + 1) * s)


def coarse_hash(level: int, i: int, j: int, leaf_hashes: dict[Key, str], extra: dict[str, Any]) -> str:
    """Input hash of a coarse tile: the hashes of the level-0 tiles of its square and of the
    ring of level-0 tiles around it (its colour filter and apron reach into them), the format
    and pyramid versions, and ``extra`` (quantum, imagery)."""
    k = 1 << level
    near = []
    for b in range(j * k - 1, (j + 1) * k + 1):
        for a in range(i * k - 1, (i + 1) * k + 1):
            h = leaf_hashes.get((a, b))
            if h is not None:
                near.append([a, b, h])
    return sha256_json({"pyramid": PYRAMID_VERSION, "format": DATA_SCHEMA_VERSION, "level": level,
                        "tile": [i, j], "leaves": near, **extra})


def terrain_error(hf: Heightfield, bounds: tuple[float, float, float, float], level: int,
                  leaf_bounds: list[tuple[float, float, float, float]]) -> float:
    """Largest |level-0 height - coarse mesh| at the level-0 nodes of the given level-0 tiles
    (the coarse mesh splits its cells along the same diagonal), rounded up to the millimetre."""
    k = 1 << level
    r = hf.resolution
    r0, r1, c0, c1 = level_node_range(hf, bounds, level)
    coarse = Heightfield(hf.xmin + c0 * r, hf.ymin + r0 * r, r * k, hf.z[r0:r1 + 1:k, c0:c1 + 1:k])
    worst = 0.0
    for b in leaf_bounds:
        a0, a1, b0, b1 = tile_node_range(hf, b)
        xs = hf.xmin + np.arange(b0, b1 + 1) * r
        ys = hf.ymin + np.arange(a0, a1 + 1) * r
        gx, gy = np.meshgrid(xs, ys)
        d = np.abs(hf.z[a0:a1 + 1, b0:b1 + 1] - coarse.sample(gx, gy))
        worst = max(worst, float(d.max()))
    return math.ceil(worst * 1000.0 - 1e-6) / 1000.0


def export_coarse_tile(world: Any, rgba: np.ndarray, baker: Any, out: Path, level: int, i: int, j: int,
                       leaves: list[Key], h: str) -> dict[str, Any]:
    """Writes a coarse tile's ``.wgt`` (and orthophoto with a ``baker`` for its level); returns
    its ``build_state.json`` entry: hash, terrain error, terrain height range, ortho."""
    cfg = world.config
    size = cfg.tiling.tile_size
    hf = world.terrain
    bounds = level_bounds(size, level, i, j)
    paths = level_paths(level, i, j)
    sections = terrain_sections(hf, bounds, cfg.height_quantum(), rgba, world.terrain_codes, level)
    content = out / paths["content"]
    content.parent.mkdir(parents=True, exist_ok=True)
    content.write_bytes(write_container(level, i, j, sections))
    ortho = out / paths["ortho"]
    if baker is not None:
        ortho.write_bytes(baker.bake(bounds))
    elif ortho.exists():
        ortho.unlink()
    grid = TileGrid(size)
    terr = terrain_error(hf, bounds, level, [grid.bounds(tile_id(a, b)) for a, b in leaves])
    r0, r1, c0, c1 = level_node_range(hf, bounds, level)
    k = 1 << level
    z = hf.z[r0:r1 + 1:k, c0:c1 + 1:k]
    return {"hash": h, "terr": terr, "zrange": [round(float(z.min()), 3), round(float(z.max()), 3)],
            "ortho": baker is not None}


def level_work(plan: dict[int, dict[Key, list[Key]]], leaf_hashes: dict[Key, str], prev: dict[str, Any],
               out: Path, force: bool, extra: dict[str, Any]
               ) -> tuple[list[tuple[Any, ...]], dict[int, dict[Key, dict[str, Any]]]]:
    """Coarse tiles to (re)write - ``(level, i, j, leaves, hash)`` items - and the build states
    of those whose inputs and files are unchanged (``prev``: build_state.json ``levels``)."""
    items: list[tuple[Any, ...]] = []
    kept: dict[int, dict[Key, dict[str, Any]]] = {level: {} for level in plan}
    for level, tiles in plan.items():
        for (i, j), leaves in tiles.items():
            h = coarse_hash(level, i, j, leaf_hashes, extra)
            old = prev.get(state_key(level, i, j))
            paths = level_paths(level, i, j)
            ok = (out / paths["content"]).is_file() and (extra.get("imagery") is None or (out / paths["ortho"]).is_file())
            if not force and ok and old is not None and old.get("hash") == h:
                kept[level][(i, j)] = old
            else:
                items.append((level, i, j, leaves, h))
    return items, kept


def state_key(level: int, i: int, j: int) -> str:
    """Key of a coarse tile in build_state.json ``levels``."""
    return f"{level}/{i}_{j}"


def remove_stale_levels(out: Path, plan: dict[int, dict[Key, list[Key]]]) -> None:
    """Deletes coarse tile files that the current levels do not have (an earlier build's)."""
    keep = {level_paths(level, i, j)[k] for level, tiles in plan.items() for (i, j) in tiles for k in ("content", "ortho")}
    root = out / "tiles"
    if not root.is_dir():
        return
    for d in root.iterdir():
        if not (d.is_dir() and d.name.isdigit() and int(d.name) >= 1):
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.relative_to(out).as_posix() not in keep:
                f.unlink()
        for sub in sorted((x for x in d.rglob("*") if x.is_dir()), key=lambda x: len(x.parts), reverse=True):
            if not any(sub.iterdir()):
                sub.rmdir()
        if not any(d.iterdir()):
            d.rmdir()


def finish_levels(plan: dict[int, dict[Key, list[Key]]], tile_state: dict[str, dict[str, Any]],
                  level_state: dict[int, dict[Key, dict[str, Any]]], resolution: float
                  ) -> dict[int, dict[Key, dict[str, Any]]]:
    """Index columns of the coarse tiles, bottom up: ``err`` (module docstring), ``terr``,
    ``zrange`` covering the tile's terrain and everything its children hold, ``ortho``, ``hash``."""
    lift: dict[Key, float] = {}
    err: dict[Key, float] = {}
    zr: dict[Key, list[float] | None] = {}
    for t, st in tile_state.items():
        key = parse_tile_id(t)
        lift[key] = float(st.get("lift") or 0.0)
        err[key] = 0.0
        zr[key] = st.get("zrange")
    out: dict[int, dict[Key, dict[str, Any]]] = {}
    for level in sorted(plan):
        k_lift: dict[Key, float] = {}
        k_err: dict[Key, float] = {}
        k_zr: dict[Key, list[float] | None] = {}
        children: dict[Key, list[Key]] = {}
        for c in lift:  # the tiles of the level below
            children.setdefault((c[0] >> 1, c[1] >> 1), []).append(c)
        rows = {}
        for key in plan[level]:
            st = level_state[level][key]
            kids = children.get(key, [])
            lo = [st["zrange"][0]] + [zr[c][0] for c in kids if zr.get(c)]
            hi = [st["zrange"][1]] + [zr[c][1] for c in kids if zr.get(c)]
            k_lift[key] = max([lift[c] for c in kids], default=0.0)
            floor = resolution * (1 << level) / 2.0
            e = max(st["terr"], k_lift[key], floor, max([err[c] for c in kids], default=0.0))
            k_err[key] = round(e, 3)
            k_zr[key] = [min(lo), max(hi)]
            rows[key] = {"err": k_err[key], "terr": st["terr"], "zrange": k_zr[key], "ortho": st["ortho"],
                         "hash": st["hash"]}
        out[level] = rows
        lift, err, zr = k_lift, k_err, k_zr
    return out
