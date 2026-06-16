"""Tests for effects/dsl.py rule serialization round-trips."""

from __future__ import annotations

import json

import pytest

from effects.dsl import dsl_to_rule, rule_to_dsl
from effects.rules import CounterRule, TerminalRule


@pytest.mark.unit
class TestRuleDSL:
    """Round-trip and validation tests for rule_to_dsl / dsl_to_rule."""

    def test_counter_roundtrip_no_pos_guard(self):
        rule = CounterRule(entity_id=17, action=3, delta_size=-2, support=2)
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_counter_roundtrip_with_pos_guard(self):
        rule = CounterRule(
            entity_id=17,
            action=3,
            delta_size=-2,
            support=2,
            controllable_id=0,
            guard_pos=(10, 15),
        )
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_terminal_roundtrip(self):
        rule = TerminalRule(
            entity_id=0,
            guard_key=((10, 15), 3),
            terminal="game_over",
            support=2,
        )
        dsl = rule_to_dsl(rule)
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_two_hop_idempotency(self):
        rule = CounterRule(
            entity_id=17,
            action=3,
            delta_size=-2,
            support=2,
            controllable_id=0,
            guard_pos=(10, 15),
        )
        d1 = rule_to_dsl(rule)
        rule2 = dsl_to_rule(d1)
        d2 = rule_to_dsl(rule2)
        assert d1 == d2

    def test_dsl_json_serializable(self):
        rule = CounterRule(
            entity_id=17,
            action=3,
            delta_size=-2,
            support=2,
            controllable_id=0,
            guard_pos=(10, 15),
        )
        dsl = rule_to_dsl(rule)
        serialized = json.dumps(dsl)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized == dsl

    def test_validation_delta_size_zero(self):
        dsl = {
            "kind": "delta",
            "entity_id": 17,
            "action": 3,
            "effect": {"dim": "size", "of": 17, "delta": 0},
            "guard": {"action": 3},
            "support": 2,
        }
        with pytest.raises(ValueError, match="delta_size must not be 0"):
            dsl_to_rule(dsl)

    def test_validation_guard_pos_without_controllable_id(self):
        dsl = {
            "kind": "delta",
            "entity_id": 17,
            "action": 3,
            "effect": {"dim": "size", "of": 17, "delta": -2},
            "guard": {
                "all": [
                    {"action": 3},
                    {"dim": "pos", "eq": [10, 15]},
                ]
            },
            "support": 2,
        }
        with pytest.raises(ValueError, match="guard_pos requires controllable_id"):
            dsl_to_rule(dsl)

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
        rule = CounterRule(entity_id=17, action=5, delta_size=1, support=3)
        dsl = rule_to_dsl(rule)
        assert "action" in dsl["guard"]
        assert "all" not in dsl["guard"]
        rule2 = dsl_to_rule(dsl)
        assert rule2 == rule

    def test_multiple_counter_rules_differing_guard_pos_preserve_keys(self):
        r1 = CounterRule(
            entity_id=17,
            action=3,
            delta_size=-2,
            support=2,
            controllable_id=0,
            guard_pos=(10, 15),
        )
        r2 = CounterRule(
            entity_id=17,
            action=3,
            delta_size=-2,
            support=2,
            controllable_id=0,
            guard_pos=(5, 5),
        )
        assert r1 != r2
        d1 = rule_to_dsl(r1)
        d2 = rule_to_dsl(r2)
        rr1 = dsl_to_rule(d1)
        rr2 = dsl_to_rule(d2)
        assert rr1 != rr2
        assert rr1.guard_pos == (10, 15)
        assert rr2.guard_pos == (5, 5)