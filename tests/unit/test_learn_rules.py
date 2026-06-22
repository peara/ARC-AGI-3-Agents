"""Tests for learn_movement_rules and learn_collision_rules."""

from __future__ import annotations

import pytest

from effects.learn import learn_collision_rules, learn_movement_rules
from effects.rules import Effect, Rule
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry, Observation


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
    positions: list[tuple[int, int]],
    entity_id: int = 0,
    controllable: bool = True,
    motion_by_action: dict[int, tuple[int, int]] | None = None,
) -> tuple[ObjectRegistry, EntityCatalog]:
    """Build a minimal registry/catalog with one entity moving through *positions*.

    Frame i has the entity at positions[i].
    """
    from perception.registry import Track

    track_id = 10
    track_obj = Track(id=track_id, color=1)
    for i, pos in enumerate(positions):
        track_obj.observations.append(
            _obs(i, (float(pos[0]), float(pos[1])))
        )

    reg = ObjectRegistry()
    reg.tracks = {track_id: track_obj}

    meta: dict[str, object] = {}
    if motion_by_action is not None:
        meta["motion_by_action"] = {
            str(k): list(v) for k, v in motion_by_action.items()
        }

    ent = Entity(
        id=entity_id,
        members=frozenset({track_id}),
        composition="singleton",
        affordances={"controllable": controllable if controllable else None},
        meta=meta,
    )
    catalog = EntityCatalog(entities={entity_id: ent})
    return reg, catalog


@pytest.mark.unit
class TestLearnMovementRules:
    def test_movement_and_collision_rules(self):
        """Entity moves right on action 1, stays put on action 2 (collision)."""
        # positions: (5,5) -> (5,6) [action 1], (5,6) -> (5,6) [action 2]
        # action_ids[0]=0 (initial, unused), action_ids[1]=1, action_ids[2]=2
        reg, catalog = _make_registry_and_catalog(
            positions=[(5, 5), (5, 6), (5, 6)]
        )
        movement_rules, collision_rules, available_actions = learn_movement_rules(
            reg, catalog, [0, 1, 2], entity_id=0
        )
        # One positional movement rule: action=1, pos=(5,5) -> (5,6)
        pos_move = [
            r for r in movement_rules
            if r.kind == "movement" and "all" in r.guard_spec
        ]
        assert len(pos_move) >= 1
        move_rule = pos_move[0]
        assert move_rule.effects[0].op == "set"
        assert move_rule.effects[0].value == (5, 6)

        # One collision rule: action=2, pos=(5,6) stays at (5,6)
        assert len(collision_rules) == 1
        cr = collision_rules[0]
        assert cr.kind == "collision"
        assert cr.effects[0].op == "revert"

    def test_positional_movement_rule_with_set_op(self):
        """pos_before != pos_after produces a positional movement rule with op='set'."""
        reg, catalog = _make_registry_and_catalog(
            positions=[(0, 0), (1, 1)]
        )
        movement_rules, collision_rules, _ = learn_movement_rules(
            reg, catalog, [0, 1], entity_id=0
        )
        pos_rules = [
            r for r in movement_rules
            if "all" in r.guard_spec
        ]
        assert len(pos_rules) == 1
        assert pos_rules[0].effects[0].op == "set"
        assert pos_rules[0].effects[0].value == (1, 1)

    def test_available_actions_deduplicates_and_sorts(self):
        """available_actions includes all unique action_ids, sorted."""
        reg, catalog = _make_registry_and_catalog(
            positions=[(0, 0), (0, 1)]
        )
        _, _, available_actions = learn_movement_rules(
            reg, catalog, [3, 1, 3, 1], entity_id=0
        )
        assert available_actions == (1, 3)

    def test_none_entity_returns_empty_rules(self):
        """If entity_id not in catalog, return empty rules but available_actions."""
        reg = ObjectRegistry()
        catalog = EntityCatalog(entities={})
        movement_rules, collision_rules, available_actions = learn_movement_rules(
            reg, catalog, [0, 1, 2], entity_id=99
        )
        assert movement_rules == ()
        assert collision_rules == ()
        assert available_actions == (0, 1, 2)

    def test_generic_movement_rule_emitted(self):
        """Generic delta rules are emitted for each action with observed motion."""
        reg, catalog = _make_registry_and_catalog(
            positions=[(5, 5), (5, 6)]
        )
        movement_rules, _, _ = learn_movement_rules(
            reg, catalog, [0, 1], entity_id=0
        )
        # Should have a generic rule for action 1 with delta
        generic_rules = [
            r for r in movement_rules
            if "all" not in r.guard_spec and "action" in r.guard_spec
        ]
        assert len(generic_rules) >= 1
        gen = generic_rules[0]
        assert gen.effects[0].op == "delta"


@pytest.mark.unit
class TestLearnCollisionRules:
    def test_returns_only_collision_bucket(self):
        """learn_collision_rules returns just the collision tuple."""
        reg, catalog = _make_registry_and_catalog(
            positions=[(5, 5), (5, 6), (5, 6)]
        )
        collision_rules = learn_collision_rules(
            reg, catalog, [0, 1, 2], entity_id=0
        )
        assert isinstance(collision_rules, tuple)
        for r in collision_rules:
            assert isinstance(r, Rule)
            assert r.kind == "collision"