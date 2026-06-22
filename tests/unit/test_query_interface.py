"""Tests for planning.query.QueryInterface bundle assembly."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from effects import Effect, EffectContext, Rule
from effects.residual import ResidualEntry
from planning.query import QueryInterface


def _make_scene(
    summary_dict: dict[str, object] | None = None,
    step_observations: tuple[object, ...] = (),
) -> MagicMock:
    """Build a mock SceneSnapshot with configurable summary and step_observations."""
    scene = MagicMock()
    scene.summary.return_value = summary_dict if summary_dict is not None else {
        "frame_idx": 0,
        "n_observed": 1,
    }
    scene.step_observations = step_observations
    return scene


def _step(
    frame_idx: int = 0,
    action_id: int = 1,
    state_name: str = "NOT_FINISHED",
    levels_completed: int = 0,
    delta: dict[str, int] | None = None,
) -> MagicMock:
    """Build a mock StepObservation."""
    s = MagicMock()
    s.frame_idx = frame_idx
    s.action_id = action_id
    s.state_name = state_name
    s.levels_completed = levels_completed
    s.delta = delta
    return s


def _make_ctx(
    terminal_rules: tuple[Rule, ...] = (),
    relational_rules: tuple[Rule, ...] = (),
    proposed_rules: tuple[Rule, ...] = (),
    confirm_threshold: int = 2,
) -> EffectContext:
    """Build a minimal EffectContext."""
    return EffectContext(
        terminal_rules=terminal_rules,
        relational_rules=relational_rules,
        proposed_rules=proposed_rules,
        confirm_threshold=confirm_threshold,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQueryInterface:
    def test_bundle_json_round_trip(self):
        scene = _make_scene(summary_dict={"frame_idx": 3, "n_observed": 5})
        qi = QueryInterface(scene)
        bundle = qi.bundle()
        encoded = json.dumps(bundle, sort_keys=True)
        roundtrip = json.loads(encoded)
        assert roundtrip["scene"]["frame_idx"] == 3
        assert roundtrip["scene"]["n_observed"] == 5
        assert roundtrip["context_note"] == bundle["context_note"]

    def test_bundle_top_level_keys(self):
        scene = _make_scene()
        qi = QueryInterface(scene)
        bundle = qi.bundle()
        expected_keys = {"scene", "action_legend", "engine_rules", "recent_actions", "unknowns", "context_note", "residual", "pruned_rules"}
        assert set(bundle.keys()) == expected_keys

    def test_bundle_with_effect_context(self):
        tr = Rule(
            guard_spec={
                "all": [
                    {"action": 3},
                    {"dim": "pos", "of": 7, "eq": [1, 2]},
                ]
            },
            effects=(Effect("terminal", 7, "set", "game_over"),),
            support=4,
        )
        cr = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=3,
        )
        ctx = _make_ctx(
            terminal_rules=(tr,),
            relational_rules=(cr,),
            proposed_rules=(cr,),
            confirm_threshold=5,
        )
        scene = _make_scene()
        qi = QueryInterface(scene, ctx=ctx)
        bundle = qi.bundle()
        rules = bundle["engine_rules"]
        assert isinstance(rules, dict)
        assert rules["confirm_threshold"] == 5
        assert isinstance(rules["confirmed"], list)
        assert isinstance(rules["proposed"], list)
        # terminal_rules + relational_rules → confirmed
        assert len(rules["confirmed"]) == 2
        # proposed_rules → proposed
        assert len(rules["proposed"]) == 1

    def test_bundle_without_effect_context(self):
        scene = _make_scene()
        qi = QueryInterface(scene, ctx=None)
        bundle = qi.bundle()
        rules = bundle["engine_rules"]
        assert rules == {"confirm_threshold": 2, "confirmed": [], "proposed": []}

    def test_fields_filter(self):
        scene = _make_scene()
        qi = QueryInterface(scene)
        bundle = qi.bundle(fields=("scene", "engine_rules"))
        assert "scene" in bundle
        assert "engine_rules" in bundle
        # context_note always present
        assert "context_note" in bundle
        # excluded fields absent
        assert "action_legend" not in bundle
        assert "recent_actions" not in bundle

    def test_max_recent_limits_recent_actions(self):
        steps = tuple(
            _step(frame_idx=i, action_id=i + 1) for i in range(10)
        )
        scene = _make_scene(step_observations=steps)
        qi = QueryInterface(scene)
        bundle = qi.bundle(max_recent=3)
        recent = bundle["recent_actions"]
        assert len(recent) == 3
        # Should be the last 3 steps (indices 7, 8, 9)
        assert recent[0]["frame_idx"] == 7
        assert recent[2]["frame_idx"] == 9

    def test_action_legend_present_or_empty(self):
        scene = _make_scene()
        qi_with = QueryInterface(scene, action_legend={1: "up", 2: "down"})
        assert qi_with.bundle()["action_legend"] == {1: "up", 2: "down"}

        qi_none = QueryInterface(scene, action_legend=None)
        assert qi_none.bundle()["action_legend"] == {}

    def test_available_actions_presence(self):
        scene = _make_scene()
        qi_with = QueryInterface(scene, available_actions=[1, 2, 3])
        bundle_with = qi_with.bundle()
        assert "available_actions" in bundle_with
        assert bundle_with["available_actions"] == [1, 2, 3]

        qi_none = QueryInterface(scene, available_actions=None)
        bundle_none = qi_none.bundle()
        assert "available_actions" not in bundle_none

    def test_recent_actions_omit_none_delta(self):
        steps = (
            _step(frame_idx=0, action_id=1, delta={"17": -2}),
            _step(frame_idx=1, action_id=2, delta=None),
        )
        scene = _make_scene(step_observations=steps)
        qi = QueryInterface(scene)
        bundle = qi.bundle()
        recent = bundle["recent_actions"]
        # Entry with delta should include it
        assert "delta" in recent[0]
        assert recent[0]["delta"] == {"17": -2}
        # Entry with delta=None should omit the key
        assert "delta" not in recent[1]

    def test_bundle_residual_default(self):
        scene = _make_scene()
        qi = QueryInterface(scene)
        bundle = qi.bundle()
        assert "residual" in bundle
        assert bundle["residual"] == []

    def test_bundle_pruned_rules_default(self):
        scene = _make_scene()
        qi = QueryInterface(scene)
        bundle = qi.bundle()
        assert "pruned_rules" in bundle
        assert bundle["pruned_rules"] == []

    def test_bundle_with_residual(self):
        scene = _make_scene()
        residual = (
            ResidualEntry(entity_id=5, dim="size", predicted=3, observed=1),
            ResidualEntry(entity_id=None, dim="terminal", predicted="NOT_FINISHED", observed="GAME_OVER"),
        )
        qi = QueryInterface(scene, residual=residual)
        bundle = qi.bundle()
        assert len(bundle["residual"]) == 2
        assert bundle["residual"][0] == {
            "dim": "size",
            "entity_id": 5,
            "predicted": 3,
            "observed": 1,
        }
        assert bundle["residual"][1] == {
            "dim": "terminal",
            "entity_id": None,
            "predicted": "NOT_FINISHED",
            "observed": "GAME_OVER",
        }

    def test_bundle_with_pruned_rules(self):
        pruned = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 10, "delta", -1),),
            support=1,
        )
        scene = _make_scene()
        qi = QueryInterface(scene, pruned_rules=(pruned,))
        bundle = qi.bundle()
        assert len(bundle["pruned_rules"]) == 1
        dsl = bundle["pruned_rules"][0]
        assert dsl["kind"] == "delta"
        assert dsl["guard"] == {"action": 2}
        assert dsl["effect"]["of"] == 10