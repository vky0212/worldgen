"""Binary road graph of data-first worlds: ``graph/core.wgt`` and ``graph/strings.json``.

The same graph as ``graph/road_graph.json`` (which is still written), split so a runtime can
load the topology, attributes and geometry as typed arrays and fetch the strings (ids, names)
separately. ``core.wgt`` is a ``.wgt`` container (``export/container.py``; level, i and j are
0) with these sections, each holding a ``u32`` count and then columnar little-endian arrays,
every array starting at an 8-byte aligned offset of the section::

    TAB   JSON: "graph_schema_version" (of the road_graph.json document), origin, crs, and
          the code tables "classes", "kinds", "directions", "restriction_kinds", "modes"
    NOD   u32 N | i32 x[N], i32 y[N], i32 z[N] (mm, local frame) | u16 degree[N] | u8 kind[N]
    EDG   u32 E | f64 maxspeed_kmh[E] (NaN: none) | u32 from[E], u32 to[E] (node rows)
          | u32 length[E] (mm) | u32 road[E], u32 name[E], u32 structure[E] (rows of the
          strings tables, 0xFFFFFFFF: none) | u32 geometry[E + 1], u32 tiles[E + 1] (first
          GEO vertex / TIL entry of each edge) | u16 width[E] (cm) | u8 direction[E]
          | u8 class[E] | u8 flags[E] | i8 layer[E]
    GEO   u32 V | i32 xyz[3V] (mm): edge polylines (bridge and tunnel decks: their profile)
    TIL   u32 T | i32 ij[2T]: the tiles of each edge
    RST   u32 R | i32 from_edge[R], i32 via_node[R], i32 to_edge[R] (-1: none)
          | u32 via[R + 1] (first VIA entry) | u32 via_edges[...] | u8 kind[R] | u8 enforced[R]

Edge ``flags``: bit 0 bridge, 1 tunnel, 2 structure implied, 3 / 4 / 5 open to
motor_vehicle / bicycle / foot. Rows follow ``road_graph.json`` (nodes, edges and
restrictions sorted by id). ``strings.json`` holds ``nodes`` and ``edges`` (ids per row),
``roads``, ``names`` and ``structures`` (tables) and per restriction its ``id``,
``restriction``, ``from_way``, ``to_way`` and ``source_id``. ``decode_graph`` rebuilds the
``road_graph.json`` document from both (without node lon/lat).
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

import numpy as np

from worldgen.export.container import read_container, write_container

DIRECTIONS = ["both", "forward", "backward"]
RESTRICTION_KINDS = ["prohibitory", "mandatory"]
MODES = ["motor_vehicle", "bicycle", "foot"]
NONE = 0xFFFFFFFF
FLAG_BRIDGE, FLAG_TUNNEL, FLAG_IMPLIED = 1, 2, 4
_MODE_FLAGS = {m: 8 << k for k, m in enumerate(MODES)}

NODE_COLUMNS = [("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("degree", "<u2"), ("kind", "u1")]
EDGE_COLUMNS = [("maxspeed", "<f8"), ("from", "<u4"), ("to", "<u4"), ("length", "<u4"), ("road", "<u4"),
                ("name", "<u4"), ("structure", "<u4"), ("geometry", "<u4", 1), ("tiles", "<u4", 1),
                ("width", "<u2"), ("direction", "u1"), ("class", "u1"), ("flags", "u1"), ("layer", "i1")]
RESTRICTION_COLUMNS = [("from_edge", "<i4"), ("via_node", "<i4"), ("to_edge", "<i4"), ("via", "<u4", 1),
                       ("via_edges", "<u4", None), ("kind", "u1"), ("enforced", "u1")]


def _mm(values: Any) -> np.ndarray:
    """Millimetres of values as road_graph.json writes them (``round(v, 3)``)."""
    return np.array([round(round(float(v), 3) * 1000.0) for v in values], dtype=np.int64)


def _pack(count: int, cols: list[np.ndarray]) -> bytes:
    out = bytearray(struct.pack("<I", count))
    for a in cols:
        out += b"\0" * (-len(out) % 8)
        out += np.ascontiguousarray(a).tobytes()
    return bytes(out)


def _unpack(data: bytes, spec: list[tuple], extra: dict[str, int] | None = None) -> tuple[int, dict[str, np.ndarray]]:
    """Columns of a section; a spec entry (name, dtype, plus) has ``count + plus`` items
    (plus None: ``extra[name]`` items)."""
    (n,) = struct.unpack_from("<I", data, 0)
    off = 4
    cols: dict[str, np.ndarray] = {}
    for item in spec:
        name, dtype = item[0], item[1]
        plus = item[2] if len(item) > 2 else 0
        length = (extra or {})[name] if plus is None else n + plus
        off += -off % 8
        cols[name] = np.frombuffer(data, dtype=dtype, count=length, offset=off)
        off += length * np.dtype(dtype).itemsize
    if off > len(data):
        raise ValueError("truncated graph section")
    return n, cols


def graph_files(graph: Any, origin: dict[str, Any] | None, crs_local: str | None, schema_version: str
                ) -> tuple[bytes, dict[str, Any]]:
    """``core.wgt`` bytes and the ``strings.json`` document of a ``RoadGraph``."""
    node_ids = sorted(graph.nodes)
    edge_ids = sorted(graph.edges)
    restrictions = sorted(graph.restrictions, key=lambda r: r.id)
    nrow = {nid: k for k, nid in enumerate(node_ids)}
    erow = {eid: k for k, eid in enumerate(edge_ids)}
    nodes = [graph.nodes[i] for i in node_ids]
    edges = [graph.edges[i] for i in edge_ids]
    kinds = sorted({n.kind for n in nodes})
    classes = sorted({e.road_class for e in edges})
    if len(kinds) > 255 or len(classes) > 255:
        raise ValueError("more than 255 node kinds or road classes")
    tables: dict[str, list[str]] = {"roads": [], "names": [], "structures": []}
    index: dict[str, dict[str, int]] = {k: {} for k in tables}

    def ref(table: str, value: str | None) -> int:
        if value is None:
            return NONE
        k = index[table].get(value)
        if k is None:
            k = index[table][value] = len(tables[table])
            tables[table].append(value)
        return k

    nod = _pack(len(nodes), [
        _mm([n.x for n in nodes]).astype("<i4"), _mm([n.y for n in nodes]).astype("<i4"),
        _mm([n.z for n in nodes]).astype("<i4"), np.array([min(n.degree, 0xFFFF) for n in nodes], dtype="<u2"),
        np.array([kinds.index(n.kind) for n in nodes], dtype="u1")])

    geo: list[np.ndarray] = []
    til: list[tuple[int, int]] = []
    gstart, tstart = [0], [0]
    cols: dict[str, list[Any]] = {name: [] for name, *_ in EDGE_COLUMNS}
    for e in edges:
        z = e.z if e.z is not None else np.zeros(len(e.coords))
        pts = e.profile if e.profile is not None else np.column_stack([e.coords, z])
        geo.append(_mm(np.asarray(pts, dtype=float).ravel()).reshape(-1, 3))
        gstart.append(gstart[-1] + len(pts))
        for t in e.tiles:
            i, j = t.split("_")
            til.append((int(i), int(j)))
        tstart.append(len(til))
        unknown = set(e.modes) - set(MODES)
        if unknown:
            raise ValueError(f"{e.id}: travel modes {sorted(unknown)} cannot be stored")
        flags = (FLAG_BRIDGE if e.bridge else 0) | (FLAG_TUNNEL if e.tunnel else 0) | (
            FLAG_IMPLIED if e.structure_implied else 0)
        for m, bit in _MODE_FLAGS.items():
            if e.modes.get(m):
                flags |= bit
        cols["maxspeed"].append(math.nan if e.maxspeed_kmh is None else float(e.maxspeed_kmh))
        cols["from"].append(nrow[e.u])
        cols["to"].append(nrow[e.v])
        cols["length"].append(round(round(e.length, 3) * 1000.0))
        cols["road"].append(ref("roads", e.road_id))
        cols["name"].append(ref("names", e.name))
        cols["structure"].append(ref("structures", e.structure_id))
        cols["width"].append(round(round(e.width, 2) * 100.0))
        cols["direction"].append(DIRECTIONS.index(e.direction))
        cols["class"].append(classes.index(e.road_class))
        cols["flags"].append(flags)
        cols["layer"].append(int(e.layer))
    cols["geometry"], cols["tiles"] = gstart, tstart
    edg = _pack(len(edges), [np.asarray(cols[name], dtype=dt) for name, dt, *_ in EDGE_COLUMNS])
    xyz = np.concatenate(geo) if geo else np.zeros((0, 3), dtype=np.int64)
    geo_b = _pack(len(xyz), [xyz.astype("<i4").ravel()])
    til_b = _pack(len(til), [np.asarray(til, dtype="<i4").reshape(-1)])

    via: list[int] = []
    vstart = [0]
    rcols: dict[str, list[int]] = {"from_edge": [], "via_node": [], "to_edge": [], "kind": [], "enforced": []}
    for r in restrictions:
        rcols["from_edge"].append(erow[r.from_edge] if r.from_edge is not None else -1)
        rcols["via_node"].append(nrow[r.via_node] if r.via_node is not None else -1)
        rcols["to_edge"].append(erow[r.to_edge] if r.to_edge is not None else -1)
        via += [erow[v] for v in r.via_edges]
        vstart.append(len(via))
        rcols["kind"].append(1 if r.mandatory else 0)
        rcols["enforced"].append(1 if r.enforced else 0)
    rst = _pack(len(restrictions), [
        np.asarray(rcols["from_edge"], dtype="<i4"), np.asarray(rcols["via_node"], dtype="<i4"),
        np.asarray(rcols["to_edge"], dtype="<i4"), np.asarray(vstart, dtype="<u4"), np.asarray(via, dtype="<u4"),
        np.asarray(rcols["kind"], dtype="u1"), np.asarray(rcols["enforced"], dtype="u1")])

    from worldgen import SCHEMA_VERSION

    tab = {"graph_schema_version": SCHEMA_VERSION, "origin": origin, "crs": {"local": crs_local},
           "classes": classes, "kinds": kinds,
           "directions": DIRECTIONS, "restriction_kinds": RESTRICTION_KINDS, "modes": MODES}
    core = write_container(0, 0, 0, [
        ("TAB", json.dumps(tab, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), True),
        ("NOD", nod, True), ("EDG", edg, True), ("GEO", geo_b, True), ("TIL", til_b, True), ("RST", rst, True)])
    strings = {"schema": "worldgen.graph_strings", "schema_version": schema_version, "nodes": node_ids,
               "edges": edge_ids, **tables,
               "restrictions": [{"id": r.id, "restriction": r.restriction, "from_way": r.from_way,
                                 "to_way": r.to_way, "source_id": r.source_id} for r in restrictions]}
    return core, strings


def decode_graph(core: bytes, strings: dict[str, Any]) -> dict[str, Any]:
    """The ``road_graph.json`` document (node lon/lat omitted) from ``core.wgt`` + ``strings.json``."""
    c = read_container(core)
    tab = json.loads(c.sections["TAB"])
    n, nc = _unpack(c.sections["NOD"], NODE_COLUMNS)
    node_ids = strings["nodes"]
    nodes = []
    for k in range(n):
        nid = node_ids[k]
        osm = int(nid[6:]) if nid.startswith("n:osm:") and nid[6:].lstrip("-").isdigit() else None
        nodes.append({"id": nid, "osm_id": osm, "kind": tab["kinds"][nc["kind"][k]], "x": int(nc["x"][k]) / 1000,
                      "y": int(nc["y"][k]) / 1000, "z": int(nc["z"][k]) / 1000, "degree": int(nc["degree"][k])})
    e_n, ec = _unpack(c.sections["EDG"], EDGE_COLUMNS)
    _, gc = _unpack(c.sections["GEO"], [("xyz", "<i4", None)], {"xyz": 3 * struct.unpack_from("<I", c.sections["GEO"])[0]})
    xyz = gc["xyz"].reshape(-1, 3)
    _, tc = _unpack(c.sections["TIL"], [("ij", "<i4", None)], {"ij": 2 * struct.unpack_from("<I", c.sections["TIL"])[0]})
    ij = tc["ij"].reshape(-1, 2)

    def opt(table: str, k: int) -> str | None:
        return None if k == NONE else strings[table][k]

    edges = []
    for k in range(e_n):
        f = int(ec["flags"][k])
        g0, g1 = int(ec["geometry"][k]), int(ec["geometry"][k + 1])
        t0, t1 = int(ec["tiles"][k]), int(ec["tiles"][k + 1])
        sp = float(ec["maxspeed"][k])
        edges.append({
            "id": strings["edges"][k], "road_id": strings["roads"][int(ec["road"][k])],
            "from": node_ids[int(ec["from"][k])], "to": node_ids[int(ec["to"][k])],
            "direction": tab["directions"][ec["direction"][k]], "length_m": int(ec["length"][k]) / 1000,
            "class": tab["classes"][ec["class"][k]], "name": opt("names", int(ec["name"][k])),
            "bridge": bool(f & FLAG_BRIDGE), "tunnel": bool(f & FLAG_TUNNEL),
            "structure_implied": bool(f & FLAG_IMPLIED), "layer": int(ec["layer"][k]),
            "structure_id": opt("structures", int(ec["structure"][k])),
            "modes": {m: bool(f & bit) for m, bit in _MODE_FLAGS.items()},
            "maxspeed_kmh": None if math.isnan(sp) else sp, "width_m": int(ec["width"][k]) / 100,
            "geometry": [[int(a) / 1000, int(b) / 1000, int(cz) / 1000] for a, b, cz in xyz[g0:g1]],
            "tiles": [f"{int(a)}_{int(b)}" for a, b in ij[t0:t1]],
        })
    r_n, rc = _unpack(c.sections["RST"], RESTRICTION_COLUMNS,
                      {"via_edges": int(_unpack(c.sections["RST"], RESTRICTION_COLUMNS[:4])[1]["via"][-1])})
    restrictions = []
    for k in range(r_n):
        s = strings["restrictions"][k]
        fe, vn, te = int(rc["from_edge"][k]), int(rc["via_node"][k]), int(rc["to_edge"][k])
        restrictions.append({
            "id": s["id"], "restriction": s["restriction"], "kind": tab["restriction_kinds"][rc["kind"][k]],
            "from_edge": strings["edges"][fe] if fe >= 0 else None,
            "via_node": node_ids[vn] if vn >= 0 else None,
            "to_edge": strings["edges"][te] if te >= 0 else None,
            "via_edges": [strings["edges"][int(v)] for v in rc["via_edges"][int(rc["via"][k]):int(rc["via"][k + 1])]],
            "from_way": s["from_way"], "to_way": s["to_way"], "source_id": s["source_id"],
            "enforced": bool(rc["enforced"][k]),
        })
    return {"schema": "worldgen.road_graph", "schema_version": tab["graph_schema_version"], "origin": tab["origin"],
            "crs": tab["crs"], "nodes": nodes, "edges": edges, "restrictions": restrictions}
