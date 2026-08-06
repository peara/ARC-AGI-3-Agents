"""Contract tests for the OptiTrack adapter output schema.

These tests lock the (LogicalRegistry, EntityCatalog) interface that the
OptiTrack adapter (Task 6) must produce.  They test only the existing type
signatures and construction behavior of the modules the adapter depends on;
they do not exercise any optimization or tracking logic.
"""

from __future__ import annotations

import pytest

from perception.entities import Entity, EntityCatalog, LifecycleState
from entity.logical_registry import LogicalRegistry
from perception.registry import ObjectRegistry, Observation, Track


class TestEntitySchema:
    """Entity field types, construction, and defaults."""

    def test_entity_fields_present(self):
        ent = Entity(
            id=1,
            members=frozenset({1}),
            composition="singleton",
            role="mover",
            centroid=(5.0, 7.0),
            size=4,
            cells=frozenset({(5, 6), (5, 7), (6, 6), (6, 7)}),
            bbox=(5, 6, 6, 7),
            affordances={"solid": True},
            meta={"orientation": "up"},
            lifecycle=LifecycleState.ACTIVE,
        )
        assert ent.id == 1
        assert ent.members == frozenset({1})
        assert ent.composition == "singleton"
        assert ent.role == "mover"
        assert ent.centroid == (5.0, 7.0)
        assert ent.size == 4
        assert ent.cells == frozenset({(5, 6), (5, 7), (6, 6), (6, 7)})
        assert ent.bbox == (5, 6, 6, 7)
        assert ent.affordances == {"solid": True}
        assert ent.meta == {"orientation": "up"}
        assert ent.lifecycle == LifecycleState.ACTIVE

    def test_entity_field_types(self):
        ent = Entity(id=7, members=frozenset({7, 8}), composition="compound")
        assert isinstance(ent.id, int)
        assert isinstance(ent.members, frozenset)
        assert all(isinstance(m, int) for m in ent.members)
        assert isinstance(ent.composition, str)
        assert ent.role is None or isinstance(ent.role, str)
        assert ent.centroid is None or isinstance(ent.centroid, tuple)
        assert ent.size is None or isinstance(ent.size, int)
        assert ent.cells is None or isinstance(ent.cells, frozenset)
        assert ent.bbox is None or isinstance(ent.bbox, tuple)
        assert isinstance(ent.affordances, dict)
        assert isinstance(ent.meta, dict)
        assert isinstance(ent.lifecycle, LifecycleState)

    def test_entity_defaults(self):
        ent = Entity(id=0, members=frozenset({0}), composition="singleton")
        assert ent.role is None
        assert ent.centroid is None
        assert ent.size is None
        assert ent.cells is None
        assert ent.bbox is None
        assert ent.affordances == {"solid": None, "interactable": None}
        assert ent.meta == {}
        assert ent.lifecycle == LifecycleState.ACTIVE


class TestLifecycleStateSchema:
    """LifecycleState enum values."""

    def test_lifecycle_values(self):
        assert LifecycleState.ACTIVE == "active"
        assert LifecycleState.MERGED == "merged"
        assert LifecycleState.DORMANT == "dormant"
        assert LifecycleState.DEAD == "dead"


class TestEntityCatalogSchema:
    """EntityCatalog structure and track_to_entity contract."""

    def test_catalog_entities_type(self):
        ent_a = Entity(id=0, members=frozenset({0}), composition="singleton")
        ent_b = Entity(id=1, members=frozenset({1, 2}), composition="compound")
        catalog = EntityCatalog(entities={0: ent_a, 1: ent_b})
        assert isinstance(catalog.entities, dict)
        assert all(isinstance(k, int) and isinstance(v, Entity)
                   for k, v in catalog.entities.items())

    def test_catalog_track_to_entity_property(self):
        ent_a = Entity(id=0, members=frozenset({0}), composition="singleton")
        ent_b = Entity(id=1, members=frozenset({1, 2}), composition="compound")
        catalog = EntityCatalog(entities={0: ent_a, 1: ent_b})
        assert isinstance(catalog.track_to_entity, dict)
        assert catalog.track_to_entity == {0: 0, 1: 1, 2: 1}
        assert all(isinstance(k, int) and isinstance(v, int)
                   for k, v in catalog.track_to_entity.items())


class TestLogicalRegistrySchema:
    """LogicalRegistry structure and logical_map contract."""

    def _make_track(self, tid: int, frame_idx: int = 0) -> Track:
        obs = Observation(
            frame_idx=frame_idx,
            color=1,
            size=4,
            centroid=(2.0, 3.0),
            bbox=(1, 2, 3, 4),
            shape_key=frozenset({(0, 0)}),
            cells=frozenset({(1, 2), (1, 3), (2, 2), (2, 3)}),
            match_rule="new",
            displacement=None,
            structural=False,
        )
        return Track(id=tid, color=1, observations=[obs], alive=True)

    def test_logical_registry_structure(self):
        raw = ObjectRegistry()
        raw.frame_idx = 0
        raw.events = []
        raw.tracks[0] = self._make_track(0)
        raw.tracks[1] = self._make_track(1)

        lreg = LogicalRegistry(real_registry=raw, logical_map={0: 0, 1: 1})

        assert isinstance(lreg.tracks, dict)
        assert all(isinstance(k, int) and isinstance(v, Track)
                   for k, v in lreg.tracks.items())
        assert isinstance(lreg.frame_idx, int)
        assert isinstance(lreg.events, list)

    def test_logical_map_property(self):
        raw = ObjectRegistry()
        raw.frame_idx = 0
        raw.events = []
        raw.tracks[0] = self._make_track(0)
        raw.tracks[1] = self._make_track(1)

        lreg = LogicalRegistry(real_registry=raw, logical_map={0: 0, 1: 0})
        assert isinstance(lreg.logical_map, dict)
        assert lreg.logical_map == {0: 0, 1: 0}
        assert all(isinstance(k, int) and isinstance(v, int)
                   for k, v in lreg.logical_map.items())


if __name__ == "__main__":
    _ = pytest.main([__file__, "-v"])
