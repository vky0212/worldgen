"""Canonical geographic data model.

Semantic entities are independent of any generated mesh. Every entity keeps:

* ``id``                 stable identifier, unchanged across compiler stages
* ``geometry``           shapely geometry in the local metric frame (x east, y north)
* ``properties``         *observed* values, normalised (units: metres, km/h)
* ``source``/``source_id`` provenance of the observation
* ``confidence``         0..1 confidence in the entity's existence and geometry
* ``inferred_properties`` estimates, each with method and confidence
* ``missing``            property names known to be unknown (neither observed nor estimated)

An observed value and an estimate are never stored in the same map, so missing
values remain distinguishable from estimated ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from shapely.geometry.base import BaseGeometry


@dataclass
class Inferred:
    value: Any
    method: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        v = self.value
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float):
            v = round(v, 3)
        return {"value": v, "method": self.method, "confidence": round(float(self.confidence), 3)}


@dataclass
class Entity:
    kind: ClassVar[str] = "entity"
    layer: ClassVar[str] = ""

    id: str
    geometry: BaseGeometry
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    source_id: str = ""
    confidence: float = 1.0
    inferred_properties: dict[str, Inferred] = field(default_factory=dict)
    missing: set[str] = field(default_factory=set)
    uncertain: bool = False
    budget_dropped: bool = False     # kept in semantic data, not rendered (instance budget)

    def get(self, name: str, default: Any = None) -> Any:
        """Observed value, else inferred value, else ``default``."""
        v = self.properties.get(name)
        if v is not None:
            return v
        inf = self.inferred_properties.get(name)
        if inf is not None:
            return inf.value
        return default

    def is_observed(self, name: str) -> bool:
        return self.properties.get(name) is not None

    def infer(self, name: str, value: Any, method: str, confidence: float) -> None:
        if self.properties.get(name) is not None:
            raise ValueError(f"{self.id}: refusing to infer observed property {name!r}")
        self.inferred_properties[name] = Inferred(value, method, confidence)
        self.missing.discard(name)

    def mark_missing(self, name: str) -> None:
        if self.properties.get(name) is None and name not in self.inferred_properties:
            self.missing.add(name)

    def record(self) -> dict[str, Any]:
        """Serializable metadata record (no geometry)."""
        rec = {
            "kind": self.kind,
            "layer": self.layer,
            "source": self.source,
            "source_id": self.source_id,
            "confidence": round(float(self.confidence), 3),
            "properties": {k: _jsonable(v) for k, v in sorted(self.properties.items()) if v is not None},
            "inferred": {k: v.to_dict() for k, v in sorted(self.inferred_properties.items())},
            "missing": sorted(self.missing),
            "uncertain": bool(self.uncertain),
        }
        if self.budget_dropped:
            rec["budget_dropped"] = True
        return rec


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float):
        return round(v, 3)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


@dataclass
class Road(Entity):
    """A road centreline (one OSM way, or one in-region piece of it).

    ``node_ids`` holds the topology node id for every vertex of ``geometry``
    (vertices that are not graph nodes still carry their source node id).
    ``elevation`` is the computed elevation for every vertex (filled by the
    transportation stage).
    """

    kind: ClassVar[str] = "road"
    layer: ClassVar[str] = "roads"
    node_ids: list[str] = field(default_factory=list)
    elevation: np.ndarray | None = None

    @property
    def road_class(self) -> str:
        return str(self.properties.get("class", "road"))

    @property
    def is_bridge(self) -> bool:
        """Tagged bridge, or a structure implied by a positive layer (see inferred_properties)."""
        return bool(self.get("bridge"))

    @property
    def is_tunnel(self) -> bool:
        return bool(self.get("tunnel"))

    @property
    def structure_implied(self) -> bool:
        """True when bridge/tunnel status is an estimate rather than a source tag."""
        return (self.is_bridge and not self.is_observed("bridge")) or (self.is_tunnel and not self.is_observed("tunnel"))

    @property
    def level(self) -> int:
        """Effective OSM layer (relative ordering only, never an absolute height)."""
        return int(self.get("layer", 0) or 0)

    @property
    def width(self) -> float:
        return float(self.get("width", 6.0))


@dataclass
class Structure(Entity):
    """Common fields of bridges and tunnels."""

    road_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    crossings: list[str] = field(default_factory=list)
    anchor_node_ids: list[str] = field(default_factory=list)
    level: int = 0


@dataclass
class Bridge(Structure):
    kind: ClassVar[str] = "bridge"
    layer: ClassVar[str] = "bridges"


@dataclass
class Tunnel(Structure):
    kind: ClassVar[str] = "tunnel"
    layer: ClassVar[str] = "tunnels"


@dataclass
class Building(Entity):
    kind: ClassVar[str] = "building"
    layer: ClassVar[str] = "buildings"
    is_part: bool = False
    parent_id: str | None = None
    rendered_by_parts: bool = False
    base_elevation: float | None = None


@dataclass
class Waterbody(Entity):
    """Water area (Polygon/MultiPolygon) or waterway centreline (LineString)."""

    kind: ClassVar[str] = "water"
    layer: ClassVar[str] = "water"
    crossings: list[str] = field(default_factory=list)

    @property
    def is_area(self) -> bool:
        return self.geometry.geom_type in ("Polygon", "MultiPolygon")


@dataclass
class LandcoverArea(Entity):
    kind: ClassVar[str] = "landcover"
    layer: ClassVar[str] = "landcover"


@dataclass
class Intersection(Entity):
    kind: ClassVar[str] = "intersection"
    layer: ClassVar[str] = "roads"
    node_id: str = ""
    elevation: float = 0.0


@dataclass
class TurnRestriction:
    id: str
    restriction: str
    from_way: str
    to_way: str
    via_node: str | None
    via_ways: list[str]
    source_id: str
    # Resolved against the graph by the topology stage.
    from_edge: str | None = None
    to_edge: str | None = None
    via_edges: list[str] = field(default_factory=list)
    enforced: bool = False

    @property
    def mandatory(self) -> bool:
        return self.restriction.startswith("only_")


@dataclass
class VegetationRegion(Entity):
    kind: ClassVar[str] = "vegetation_region"
    layer: ClassVar[str] = "vegetation_regions"


@dataclass
class ObservedTree:
    """A tree explicitly mapped in a source (OSM natural=tree), before placement."""

    id: str
    x: float
    y: float
    properties: dict[str, Any]
    source: str
    source_id: str
    from_row: bool = False


@dataclass
class TreeTable:
    """Columnar TreeInstance storage (observed and generated trees).

    Column semantics follow the TreeInstance definition of the vegetation
    extension: stable id, local position, ground elevation, tree height, crown
    and trunk radius, archetype, placement confidence, observed flag and the
    deterministic seed used for generated attributes.
    """

    ids: list[str] = field(default_factory=list)
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ground_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    height: np.ndarray = field(default_factory=lambda: np.zeros(0))
    crown_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    trunk_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    archetype: list[str] = field(default_factory=list)
    species: list[str | None] = field(default_factory=list)
    category: list[str] = field(default_factory=list)
    confidence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    observed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    seed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.uint64))
    source: list[str] = field(default_factory=list)
    source_id: list[str] = field(default_factory=list)
    # Per-tree names of properties that were estimated (e.g. "height") for observed trees.
    inferred: list[list[str]] = field(default_factory=list)
    # False where the instance budget dropped the tree from rendering (None: all rendered).
    rendered: np.ndarray | None = None

    def rendered_mask(self) -> np.ndarray:
        return np.ones(len(self), dtype=bool) if self.rendered is None else self.rendered

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, mask: np.ndarray) -> TreeTable:
        idx = np.nonzero(mask)[0]
        pick = lambda seq: [seq[i] for i in idx]  # noqa: E731
        return TreeTable(
            ids=pick(self.ids), x=self.x[idx], y=self.y[idx], ground_z=self.ground_z[idx],
            height=self.height[idx], crown_radius=self.crown_radius[idx],
            trunk_radius=self.trunk_radius[idx], yaw=self.yaw[idx], archetype=pick(self.archetype),
            species=pick(self.species), category=pick(self.category), confidence=self.confidence[idx],
            observed=self.observed[idx], seed=self.seed[idx], source=pick(self.source),
            source_id=pick(self.source_id), inferred=pick(self.inferred),
            rendered=None if self.rendered is None else self.rendered[idx],
        )

    def record(self, i: int) -> dict[str, Any]:
        observed = bool(self.observed[i])
        props: dict[str, Any] = {"archetype": self.archetype[i], "category": self.category[i]}
        inferred: dict[str, Any] = {}
        values = {
            "height": float(self.height[i]),
            "crown_radius": float(self.crown_radius[i]),
            "trunk_radius": float(self.trunk_radius[i]),
        }
        for k, v in values.items():
            if observed and k not in self.inferred[i]:
                props[k] = round(v, 2)
            else:
                inferred[k] = {"value": round(v, 2), "method": "procedural" if not observed else "allometric",
                               "confidence": 0.3 if not observed else 0.5}
        if self.species[i]:
            props["species"] = self.species[i]
        props["ground_elevation"] = round(float(self.ground_z[i]), 2)
        props["observed"] = observed
        props["seed"] = int(self.seed[i])
        return {
            "kind": "tree",
            "layer": "trees",
            "source": self.source[i],
            "source_id": self.source_id[i],
            "confidence": round(float(self.confidence[i]), 3),
            "properties": props,
            "inferred": inferred,
            "missing": [] if self.species[i] else ["species"],
            "uncertain": not observed,
        }
