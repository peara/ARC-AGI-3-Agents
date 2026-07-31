"""Tests for compute_transition_diff with movement_rules producing expected_motions and blocked."""

from __future__ import annotations

from effects.rules import Effect, Rule
from planning.transition_diff import compute_transition_diff


def _movement_rule(entity_id: int, action: int, dr: int, dc: int) -> Rule:
    return Rule(
        guard_spec={"action": action},
        effects=(Effect(dim="pos", of=entity_id, op="delta", value=(dr, dc)),),
        support=3,
        kind="delta",
    )


def _obs(entity_id: int, dim: str, before, after) -> dict:
    return {"entity": entity_id, "dim": dim, "before": before, "after": after}


class TestTransitionDiffMovementRules:
    def test_expected_motions_from_movement_rule(self):
        obs = {
            "action": 1,
            "before": [[[12, "pos", [40, 20]]]],
            "after": [[[12, "pos", [36, 20]]]],
        }
        rule = _movement_rule(entity_id=12, action=1, dr=-4, dc=0)
        diff = compute_transition_diff(obs, movement_rules=(rule,))
        assert len(diff["expected_motions"]) == 1
        em = diff["expected_motions"][0]
        assert em["entity"] == 12
        assert em["delta"] == [-4.0, 0.0]

    def test_blocked_when_entity_pos_does_not_change(self):
        obs = {
            "action": 1,
            "before": [[[12, "pos", [40, 20]]]],
            "after": [[[12, "pos", [40, 20]]]],
        }
        rule = _movement_rule(entity_id=12, action=1, dr=-4, dc=0)
        diff = compute_transition_diff(obs, movement_rules=(rule,))
        assert len(diff["expected_motions"]) == 1
        pos_entry = next(e for e in diff["changed"] if e["entity"] == 12 and e["dim"] == "pos")
        assert pos_entry["blocked"] is True

    def test_blocked_entity_not_in_changed(self):
        obs = {
            "action": 1,
            "before": [[[5, "pos", [10, 10]]]],
            "after": [[[5, "pos", [10, 10]]]],
        }
        rule = _movement_rule(entity_id=12, action=1, dr=-4, dc=0)
        diff = compute_transition_diff(obs, movement_rules=(rule,))
        assert len(diff["expected_motions"]) == 1
        pos_entry = next(
            (e for e in diff["changed"] if e["entity"] == 12 and e["dim"] == "pos"),
            None,
        )
        assert pos_entry is not None
        assert pos_entry["blocked"] is True

    def test_no_expected_motions_without_rules(self):
        obs = {
            "action": 1,
            "before": [[[12, "pos", [40, 20]]]],
            "after": [[[12, "pos", [36, 20]]]],
        }
        diff = compute_transition_diff(obs, movement_rules=())
        assert diff["expected_motions"] == []

    def test_controllable_id_not_in_result(self):
        obs = {
            "action": 1,
            "before": [[[12, "pos", [40, 20]]]],
            "after": [[[12, "pos", [36, 20]]]],
        }
        rule = _movement_rule(entity_id=12, action=1, dr=-4, dc=0)
        diff = compute_transition_diff(obs, movement_rules=(rule,))
        assert "controllable_id" not in diff