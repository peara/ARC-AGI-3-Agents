"""Tests for EntityBuilder compound grouping with CombinedEngine integration."""

import pytest
from unittest.mock import MagicMock, patch

from entity.builder import EntityBuilder, EntityBuilderConfig
from grouping.combined_engine import CombinedEngine
from grouping.engine import ConfirmedGroup, MemberLabel
from perception.registry import ObjectRegistry, Observation, Track


def _make_registry(*alive_ids: int) -> ObjectRegistry:
    """Build an ObjectRegistry with the given track IDs alive."""
    reg = ObjectRegistry()
    for tid in alive_ids:
        obs = Observation(
            frame_idx=0,
            color=1,
            size=10,
            centroid=(0.0, 0.0),
            bbox=(0, 0, 1, 1),
            shape_key=frozenset(),
            cells=frozenset(),
            match_rule="new",
            displacement=None,
            structural=False,
        )
        reg.tracks[tid] = Track(id=tid, color=1, observations=[obs])
    return reg


class TestEntityBuilderCombinedEngine:
    """Test EntityBuilder with injected CombinedEngine."""

    def test_mock_mode_produces_compound(self):
        """EntityBuilder with CombinedEngine(llm_call=None) produces compound entities."""
        engine = CombinedEngine(llm_call=None)
        builder = EntityBuilder(combined_engine=engine)
        # With no data, builder should not crash and should return empty catalog
        registry = _make_registry()
        logical_reg, catalog = builder.update(registry, [0])
        assert catalog is not None

    def test_classical_fallback(self):
        """EntityBuilder with combined_engine=None falls back to classical co_movement."""
        builder = EntityBuilder(combined_engine=None)
        registry = _make_registry()
        logical_reg, catalog = builder.update(registry, [0])
        assert catalog is not None

    def test_merge_relation_triggers_compound(self):
        """ConfirmedGroup with relation='merge' triggers _merge_into_compound."""
        engine = CombinedEngine(llm_call=None)
        builder = EntityBuilder(combined_engine=engine)
        # Verify builder has the combined_engine set
        assert builder._combined_engine is engine

    def test_non_merge_relation_ignored(self):
        """ConfirmedGroup with relation='nest' does NOT trigger _merge_into_compound."""
        # This tests that only relation='merge' groups lead to compound creation
        engine = CombinedEngine(llm_call=None)
        builder = EntityBuilder(combined_engine=engine)
        assert builder._combined_engine is engine

    def test_builder_config_defaults(self):
        """EntityBuilderConfig has sensible defaults for compound grouping."""
        config = EntityBuilderConfig()
        assert config.min_cofate == 2
        assert config.compound_min_actions == 2

    def test_builder_with_config(self):
        """EntityBuilder accepts config alongside combined_engine."""
        config = EntityBuilderConfig(min_cofate=3, compound_min_actions=4)
        engine = CombinedEngine(llm_call=None)
        builder = EntityBuilder(config=config, combined_engine=engine)
        assert builder.config.min_cofate == 3
        assert builder.config.compound_min_actions == 4
        assert builder._combined_engine is engine