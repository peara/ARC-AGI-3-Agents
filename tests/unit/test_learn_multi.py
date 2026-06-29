"""Tests for learn_effect_context_multi and learn_counter_rules_action_only."""

from __future__ import annotations

import pytest

from effects.context import EffectContext
from effects.learn_multi import learn_counter_rules_action_only, learn_effect_context_multi  # noqa: F401 — imported for readability of test context
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry, Observation, Track


def _obs(frame_idx: int, centroid: tuple[float, float], size: int = 1) -> Observation:
    return Observation(
        frame_idx=frame_idx,
        color=1,
        size=size,
        centroid=centroid,
        bbox=(int(centroid[0]), int(centroid[1]), int(centroid[0]), int(centroid[1])),
        shape_key=frozenset({(int(centroid[0]), int(centroid[1]))}),
        cells=frozenset({(int(centroid[0]), int(centroid[1]))}),
        match_rule="new",
        displacement=None,
        structural=False,
    )


def _make_registry_and_catalog(
    positions: dict[int, list[tuple[int, int]]],
    motion_by_action: dict[int, dict[int, tuple[int, int]]] | None = None,
    roles: dict[int, str] | None = None,
    controllable_ids: set[int] | None = None,
) -> tuple[ObjectRegistry, EntityCatalog]:
    """Build a registry/catalog with multiple entities.

    ``positions`` maps entity_id to a list of (row, col) positions per frame.
    Each entity gets one track; track_id = entity_id * 10 + 1.
    ``motion_by_action`` maps entity_id to motion_by_action dict.
    ``roles`` maps entity_id to role string.
    ``controllable_ids`` set of entity IDs marked as controllable.
    """
    reg = ObjectRegistry()
    entities: dict[int, Entity] = {}

    for eid, pos_list in positions.items():
        track_id = eid * 10 + 1
        track = Track(id=track_id, color=1)
        for i, pos in enumerate(pos_list):
            track.observations.append(_obs(i, (float(pos[0]), float(pos[1]))))
        reg.tracks[track_id] = track

        meta: dict[str, object] = {}
        if motion_by_action and eid in motion_by_action:
            mba = motion_by_action[eid]
            meta["motion_by_action"] = {
                str(k): list(v) for k, v in mba.items()
            }

        affordances: dict[str, bool | None] = {
            "controllable": True if controllable_ids and eid in controllable_ids else None,
            "solid": None,
            "interactable": None,
        }
        role = roles.get(eid) if roles else None

        ent = Entity(
            id=eid,
            members=frozenset({track_id}),
            composition="singleton",
            role=role,
            affordances=affordances,
            meta=meta,
        )
        entities[eid] = ent

    return reg, EntityCatalog(entities=entities)


# ---------------------------------------------------------------------------
# learn_effect_context_multi
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLearnEffectContextMulti:
    def test_learn_effect_context_multi_basic(self):
        """Two controllable entities each produce movement rules."""
        # Entity 0: moves right on action 1
        # Entity 1: moves down on action 1
        reg, catalog = _make_registry_and_catalog(
            positions={
                0: [(5, 5), (5, 6)],  # moves right
                1: [(3, 3), (4, 3)],  # moves down
            },
            controllable_ids={0, 1},
        )
        action_ids = [0, 1]  # frame 0 initial, frame 1 action 1
        ctx = learn_effect_context_multi(reg, catalog, action_ids)
        assert ctx is not None
        assert isinstance(ctx, EffectContext)
        # Should have movement rules from both entities
        assert len(ctx.movement_rules) > 0
        # available_actions should be deduplicated and sorted
        assert ctx.available_actions == (0, 1)

    def test_learn_effect_context_multi_no_terminal(self):
        """terminal_rules is always empty tuple."""
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
            controllable_ids={0},
        )
        ctx = learn_effect_context_multi(reg, catalog, [0, 1])
        assert ctx is not None
        assert ctx.terminal_rules == ()

    def test_learn_effect_context_multi_counter_action_only(self):
        """Counter rules have only {action: N} guards, no position guards."""
        # Entity 0: controllable at fixed position (5,5) -> (5,5)
        # Entity 1: counter that grows on action 2
        reg, catalog = _make_registry_and_catalog(
            positions={
                0: [(5, 5), (5, 5)],  # stays still on action 1
                1: [],  # counter entity — needs size changes
            },
            roles={1: "counter"},
            controllable_ids={0},
        )
        # Build entity 1 manually with size changes via observations
        track_id_1 = 11  # entity 1's track
        track_1 = Track(id=track_id_1, color=2)
        track_1.observations.append(_obs(0, (10.0, 10.0), size=3))
        track_1.observations.append(_obs(1, (10.0, 10.0), size=4))  # grew by 1
        reg.tracks[track_id_1] = track_1
        # Update catalog entity 1 with track
        catalog = EntityCatalog(entities={
            **catalog.entities,
            1: Entity(
                id=1,
                members=frozenset({track_id_1}),
                composition="singleton",
                role="counter",
                affordances={"controllable": None, "solid": None, "interactable": None},
                meta={},
            ),
        })
        action_ids = [0, 1]
        ctx = learn_effect_context_multi(reg, catalog, action_ids)
        assert ctx is not None
        # All relational rules should have action-only guards
        for rule in ctx.relational_rules:
            assert "action" in rule.guard_spec
            assert "all" not in rule.guard_spec, (
                f"Expected action-only guard, got {rule.guard_spec}"
            )

    def test_learn_effect_context_multi_empty_catalog(self):
        """Empty catalog still returns valid EffectContext with available_actions."""
        reg = ObjectRegistry()
        catalog = EntityCatalog(entities={})
        action_ids = [0, 1, 2]
        ctx = learn_effect_context_multi(reg, catalog, action_ids)
        assert ctx is not None
        assert ctx.available_actions == (0, 1, 2)
        assert ctx.movement_rules == ()
        assert ctx.collision_rules == ()
        assert ctx.relational_rules == ()
        assert ctx.terminal_rules == ()

    def test_learn_effect_context_multi_empty_actions_returns_none(self):
        """Empty action_ids returns None (matching v1 behavior)."""
        reg = ObjectRegistry()
        catalog = EntityCatalog(entities={})
        ctx = learn_effect_context_multi(reg, catalog, [])
        assert ctx is None


# ---------------------------------------------------------------------------
# learn_counter_rules_action_only
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLearnCounterRulesActionOnly:
    def test_learn_counter_rules_action_only_no_position_guard(self):
        """Counter rules never have 'dim': 'pos' in guard_spec."""
        # Entity 0: controllable, entity 1: counter
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 5)]},
            roles={},
            controllable_ids={0},
        )
        # Add counter entity with size change
        track_id_1 = 11
        track_1 = Track(id=track_id_1, color=2)
        track_1.observations.append(_obs(0, (10.0, 10.0), size=5))
        track_1.observations.append(_obs(1, (10.0, 10.0), size=6))  # +1 on action 1
        reg.tracks[track_id_1] = track_1
        catalog = EntityCatalog(entities={
            **catalog.entities,
            1: Entity(
                id=1,
                members=frozenset({track_id_1}),
                composition="singleton",
                role="counter",
                affordances={"controllable": None, "solid": None, "interactable": None},
                meta={},
            ),
        })
        rules = learn_counter_rules_action_only(reg, catalog, [0, 1])
        for rule in rules:
            # Ensure no position guard — should not contain "dim": "pos"
            guard_str = str(rule.guard_spec)
            assert '"dim"' not in guard_str or '"pos"' not in guard_str, (
                f"Found position guard in counter rule: {rule.guard_spec}"
            )

    def test_learn_counter_rules_action_only_action_only_guards(self):
        """All counter rules have exactly {"action": N} as guard_spec."""
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 5)]},
            controllable_ids={0},
        )
        # Counter entity with size change on action 1
        track_id_1 = 11
        track_1 = Track(id=track_id_1, color=2)
        track_1.observations.append(_obs(0, (10.0, 10.0), size=3))
        track_1.observations.append(_obs(1, (10.0, 10.0), size=5))  # +2 on action 1
        reg.tracks[track_id_1] = track_1
        catalog = EntityCatalog(entities={
            **catalog.entities,
            1: Entity(
                id=1,
                members=frozenset({track_id_1}),
                composition="singleton",
                role="counter",
                affordances={"controllable": None, "solid": None, "interactable": None},
                meta={},
            ),
        })
        rules = learn_counter_rules_action_only(reg, catalog, [0, 1])
        assert len(rules) > 0, "Expected at least one counter rule"
        for rule in rules:
            # Guard should be simple {"action": N}, not {"all": [...]}
            assert "all" not in rule.guard_spec, (
                f"Expected action-only guard, got {rule.guard_spec}"
            )
            assert "action" in rule.guard_spec, (
                f"Expected action key in guard, got {rule.guard_spec}"
            )

    def test_learn_counter_rules_action_only_no_counter_entities(self):
        """Returns empty tuple when catalog has no counter-role entities."""
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
            controllable_ids={0},
        )
        rules = learn_counter_rules_action_only(reg, catalog, [0, 1])
        assert rules == ()

    def test_learn_counter_rules_action_only_zero_delta_ignored(self):
        """Counter entities with zero size delta produce no rules."""
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 5)]},
            controllable_ids={0},
        )
        # Counter entity with same size both frames (delta=0)
        track_id_1 = 11
        track_1 = Track(id=track_id_1, color=2)
        track_1.observations.append(_obs(0, (10.0, 10.0), size=5))
        track_1.observations.append(_obs(1, (10.0, 10.0), size=5))  # delta=0, ignored
        reg.tracks[track_id_1] = track_1
        catalog = EntityCatalog(entities={
            **catalog.entities,
            1: Entity(
                id=1,
                members=frozenset({track_id_1}),
                composition="singleton",
                role="counter",
                affordances={"controllable": None, "solid": None, "interactable": None},
                meta={},
            ),
        })
        rules = learn_counter_rules_action_only(reg, catalog, [0, 1])
        assert rules == ()