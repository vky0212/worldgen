"""Entity locator of data-first worlds: ``tiles/locate/<bucket>.json``.

Maps every entity id with a record in some tile — the ``meta.json`` records and the mapped
trees of ``TOB `` sections — to the tiles holding it, so a runtime can resolve a link or an
issue to an entity in a tile it has not loaded. Ids are spread over ``buckets`` files (a power
of two, about 4096 ids per file) by FNV-1a 32 of their UTF-8 bytes. Not listed, because the
tile follows from the id: terrain records (``terrain:<i>_<j>``) and generated trees
(``tree:gen:<cx>_<cy>_...``: the tile holding sampling cell ``(cx, cy)``).
"""

from __future__ import annotations

from typing import Any

LOCATE_URI = "tiles/locate/{bucket}.json"
IDS_PER_BUCKET = 4096
MAX_BUCKETS = 4096


def fnv1a32(text: str) -> int:
    h = 0x811C9DC5
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def bucket_count(n_ids: int) -> int:
    b = 1
    while b < MAX_BUCKETS and n_ids > b * IDS_PER_BUCKET:
        b *= 2
    return b


def located(entity_id: str) -> bool:
    """Whether the locator lists the id (its tile cannot be derived from the id itself)."""
    return not (entity_id.startswith("terrain:") or entity_id.startswith("tree:gen:"))


def locate_files(tile_entities: dict[str, list[str]], schema_version: str
                 ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``tileset.json`` ``locate`` entry and the bucket documents (path -> document) for
    ``tile_entities`` (tile id -> entity ids recorded in it)."""
    where: dict[str, list[str]] = {}
    for tid in sorted(tile_entities):
        for eid in tile_entities[tid]:
            if located(eid):
                where.setdefault(eid, []).append(tid)
    buckets = bucket_count(len(where))
    docs: dict[int, dict[str, list[str]]] = {}
    for eid in sorted(where):
        docs.setdefault(fnv1a32(eid) & (buckets - 1), {})[eid] = where[eid]
    files = {LOCATE_URI.format(bucket=b): {"schema": "worldgen.locate", "schema_version": schema_version,
                                           "bucket": b, "entities": ents}
             for b, ents in sorted(docs.items())}
    entry = {"uri": LOCATE_URI, "buckets": buckets, "hash": "fnv1a32", "ids": len(where)}
    return entry, files
