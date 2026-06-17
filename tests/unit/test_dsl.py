"""Tests for effects/dsl.py rule serialization round-trips."""

from __future__ import annotations

import json

import pytest

from effects.dsl import dsl_to_rule, rule_to_dsl
from effects.rules import Effect, Rule


@pytest.mark.unit
class TestRuleDSL:
    """Round-trip and validation tests for rule_to_dsl / dsl_to_rule."""

    def test_counter_roundtrip_no_pos_guard(self):
        rule = Rule(
            guard_spec={"action": 3},
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_counter_roundtrip_with_pos_guard(self):
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [10, 15]},
                ]
            },
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_terminal_roundtrip(self):
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [10, 15]},
                ]
            },
            effects=(Effect("terminal", 0, "set", "game_over"),),
            support=2,
        )
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_two_hop_idempotency(self):
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [10, 15]},
                ]
            },
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        d1 = rule_to_dsl(rule)
        rule2 = dsl_to_rule(d1)
        d2 = rule_to_dsl(rule2)
        assert d1 == d2

    def test_dsl_json_serializable(self):
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [10, 15]},
                ]
            },
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        dsl = rule_to_dsl(rule)
        serialized = json.dumps(dsl)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized == dsl

    def test_validation_delta_size_zero(self):
        with pytest.raises(ValueError, match="delta effect must have non-zero value"):
            Rule(
                guard_spec={"action": 3},
                effects=(Effect("size", 17, "delta", 0),),
                support=2,
            )

    def test_validation_unknown_kind(self):
        dsl = {
            "kind": "unknown",
            "entity_id": 17,
            "action": 3,
            "effect": {},
            "guard": {},
            "support": 2,
        }
        with pytest.raises(ValueError, match="unknown kind"):
            dsl_to_rule(dsl)

    def test_counter_action_only_guard(self):
        rule = Rule(
            guard_spec={"action": 5},
            effects=(Effect("size", 17, "delta", 1),),
            support=3,
        )
        dsl = rule_to_dsl(rule)
        assert "action" in dsl["guard"]
        assert "all" not in dsl["guard"]
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_multiple_counter_rules_differing_guard_pos_preserve_keys(self):
        r1 = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [10, 15]},
                ]
            },
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        r2 = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 0, "eq": [5, 5]},
                ]
            },
            effects=(Effect("size", 17, "delta", -2),),
            support=2,
        )
        assert r1 != r2
        d1 = rule_to_dsl(r1)
        d2 = rule_to_dsl(r2)
        rr1 = dsl_to_rule(d1)
        rr2 = dsl_to_rule(d2)
        assert rr1 != rr2
        # Verify guard_specs differ
        assert rr1.guard_spec != rr2.guard_spec