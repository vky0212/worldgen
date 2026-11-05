"""Tile file layout and the tile index documents of both export modes.

``baked`` (format 1.x): ``tiles/<i>_<j>.glb|.meta.json|.collision.glb|.ortho.jpg`` and one flat
``tiles/index.json``. ``data`` (format 2.x): ``tiles/<level>/<si>_<sj>/<i>_<j>.wgt`` (plus the
same side files) grouped by 4096 m subtree squares, ``tiles/tileset.json`` and one columnar
``tiles/sub/<si>_<sj>.json`` per subtree, so an index never has to list every tile at once.
Levels 1 to 4 (``pyramid``) are coarser tiles of the same squares; a subtree square is one
level-4 tile.
"""

from __future__ import annotations

from typing import Any

from worldgen.tiling.grid import TileGrid, parse_tile_id

DATA_SCHEMA_VERSION = "2.5.0"
# Tiles per subtree side: a subtree covers 16 x 16 level-0 tiles (4096 m with 256 m tiles),
# i.e. one tile of level 4.
SUBTREE_TILES = 16
SUBTREE_LEVEL = 4

DATA_URI = {
    "content": "tiles/{level}/{si}_{sj}/{i}_{j}.wgt",
    "meta": "tiles/0/{si}_{sj}/{i}_{j}.meta.json",
    "collision": "tiles/0/{si}_{sj}/{i}_{j}.collision.glb",
    "ortho": "tiles/{level}/{si}_{sj}/{i}_{j}.ortho.jpg",
    "subtree": "tiles/sub/{si}_{sj}.json",
}


def subtree_of(i: int, j: int, level: int = 0) -> tuple[int, int]:
    """Subtree square of a tile of ``level`` (floor division, so negative indices work)."""
    n = SUBTREE_LEVEL - level
    return i >> n, j >> n


def level_paths(level: int, i: int, j: int) -> dict[str, str]:
    """Output paths (relative to the world) of a coarse tile: its container and orthophoto."""
    si, sj = subtree_of(i, j, level)
    return {k: DATA_URI[k].format(level=level, si=si, sj=sj, i=i, j=j) for k in ("content", "ortho")}


def tile_paths(mode: str, tid: str) -> dict[str, str]:
    """Output paths (relative to the world) of a level-0 tile: content, meta, collision, ortho."""
    if mode == "baked":
        return {"content": f"tiles/{tid}.glb", "meta": f"tiles/{tid}.meta.json",
                "collision": f"tiles/{tid}.collision.glb", "ortho": f"tiles/{tid}.ortho.jpg"}
    i, j = parse_tile_id(tid)
    si, sj = subtree_of(i, j)
    return {k: DATA_URI[k].format(level=0, si=si, sj=sj, i=i, j=j) for k in ("content", "meta", "collision", "ortho")}


def data_index(tiles: list[str], tile_state: dict[str, dict[str, Any]], tile_size: float, resolution: float,
               quantum: float, layers: list[str], ortho: dict[str, Any] | None, landcover_codes: dict[str, str],
               levels: dict[int, dict[tuple[int, int], dict[str, Any]]] | None = None,
               level_ortho: dict[int, dict[str, Any]] | None = None
               ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``tiles/tileset.json`` and the subtree documents (path -> document) of a data-first world.

    ``levels``: the coarse tiles (``pyramid.finish_levels``) by level; ``level_ortho``: the pixel
    size of each level's orthophotos."""
    grid = TileGrid(tile_size)
    levels = levels or {}
    groups: dict[tuple[int, int], list[str]] = {}
    for t in tiles:
        groups.setdefault(subtree_of(*parse_tile_id(t)), []).append(t)
    coarse: dict[tuple[int, int], dict[int, list[tuple[int, int]]]] = {}
    for level in sorted(levels):
        for key in levels[level]:
            coarse.setdefault(subtree_of(*key, level), {}).setdefault(level, []).append(key)
    subtrees, docs = [], {}
    span = SUBTREE_TILES * tile_size
    for (si, sj) in sorted(groups, key=lambda k: (k[1], k[0])):
        members = groups[(si, sj)]
        states = [tile_state.get(t, {}) for t in members]
        ij = [parse_tile_id(t) for t in members]
        zr = [st.get("zrange") or [None, None] for st in states]
        cols: dict[str, list[Any]] = {
            "i": [a for a, _ in ij], "j": [b for _, b in ij],
            "zmin": [z[0] for z in zr], "zmax": [z[1] for z in zr],
            "hash": [(st.get("hash") or "")[:16] for st in states],
            # bit 0: meta.json, bit 1: ortho.jpg, bit 2: collision.glb
            "flags": [1 | (2 if st.get("ortho") else 0) | (4 if st.get("collision") else 0) for st in states],
        }
        keys = sorted({k for st in states for k in (st.get("counts") or {})})
        counts = {k: [int((st.get("counts") or {}).get(k, 0)) for st in states] for k in keys}
        zmins = [z for z in cols["zmin"] if z is not None]
        zmaxs = [z for z in cols["zmax"] if z is not None]
        doc_levels: list[dict[str, Any]] = [{"level": 0, **cols, "counts": counts}]
        for level, members_l in sorted(coarse.get((si, sj), {}).items()):
            rows = [levels[level][k] for k in members_l]
            doc_levels.append({
                "level": level, "i": [a for a, _ in members_l], "j": [b for _, b in members_l],
                "zmin": [r["zrange"][0] for r in rows], "zmax": [r["zrange"][1] for r in rows],
                "err": [r["err"] for r in rows], "terr": [r["terr"] for r in rows],
                "hash": [r["hash"][:16] for r in rows], "flags": [2 if r["ortho"] else 0 for r in rows],
            })
            zmins += [r["zrange"][0] for r in rows]
            zmaxs += [r["zrange"][1] for r in rows]
        path = DATA_URI["subtree"].format(si=si, sj=sj)
        docs[path] = {"schema": "worldgen.tileset_subtree", "schema_version": DATA_SCHEMA_VERSION,
                      "si": si, "sj": sj, "levels": doc_levels}
        subtrees.append({"si": si, "sj": sj, "bounds": [si * span, sj * span, (si + 1) * span, (sj + 1) * span],
                         "zrange": [min(zmins), max(zmaxs)] if zmins else None, "tiles": len(members)})
    xs = [grid.bounds(t) for t in tiles]
    level_docs: list[dict[str, Any]] = [{"level": 0, "size": tile_size, "spacing": resolution, "geometric_error": 0.0}]
    for level in sorted(levels):
        k = 1 << level
        doc: dict[str, Any] = {"level": level, "size": tile_size * k, "spacing": resolution * k,
                               "geometric_error": max(r["err"] for r in levels[level].values()),
                               "tiles": len(levels[level])}
        if level_ortho and level in level_ortho:
            doc["ortho"] = level_ortho[level]
        level_docs.append(doc)
    tileset = {
        "schema": "worldgen.tileset", "schema_version": DATA_SCHEMA_VERSION,
        "tile_size": tile_size, "subtree_tiles": SUBTREE_TILES,
        "levels": level_docs,
        "uri": DATA_URI,
        "terrain": {"sections": ["HGT", "RGB", "LCC"], "quantum": quantum,
                    "triangulation": "quad (i,j)-(i+1,j+1) diagonal", "apron": 1},
        "layers": layers, "landcover_codes": landcover_codes, "ortho": ortho,
        "bounds": [min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs)]
        if xs else None,
        "subtrees": subtrees,
    }
    return tileset, docs
