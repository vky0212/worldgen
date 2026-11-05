"""Hashes the build takes once: entity geometry hashes per process, output file hashes as written."""

from __future__ import annotations

import os

import shapely

from worldgen.export.tiles import entity_strtree, tile_input_hash, world_tiles
from worldgen.hashing import sha256_bytes, sha256_file, write_hashed


def test_entity_geometry_hashes_are_computed_once_and_unchanged(compiled):
    world = compiled("demo_town")
    index = entity_strtree(world)
    assert index.ents
    for k, e in enumerate(index.ents[:200]):
        h = index.geometry_hash(k)
        assert h == sha256_bytes(shapely.to_wkb(shapely.set_precision(e.geometry, 0.001)))
        assert index.geometry_hash(k) is h                  # cached
    # Tile hashes with one shared index equal those computed with a fresh index per tile.
    shared = entity_strtree(world)
    for tid in world_tiles(world):
        assert tile_input_hash(world, tid, shared) == tile_input_hash(world, tid)


def test_write_hashed_records_the_file_as_written(tmp_path):
    data = os.urandom(5000)
    path = tmp_path / "a.bin"
    sha, size, mtime = write_hashed(path, data)
    assert path.read_bytes() == data
    assert (sha, size, mtime) == (sha256_file(path), len(data), path.stat().st_mtime_ns)
