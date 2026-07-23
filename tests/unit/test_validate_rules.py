"""Tests for validate_rules_against_history()."""

from __future__ import annotations

import pytest

from effects import (
    Effect,
    EffectContext,
    Rule,
    SceneState,
    TransitionHistory,
    validate_rules_against_history,
)
from effects.engine import _ProjectionSpec
from effects.transition_history import Transition


def _make_transition(frame_idx: int, action: int, state_before: SceneState, state_after: SceneState) -> Transition:
    return Transition(
        frame_idx=frame_idx,
        action=action,
        state_before=state_before,
        state_after=state_after,
    )


def _empty_history() -> TransitionHistory:
    return TransitionHistory()


def _history_with(*transitions: Transition) -> TransitionHistory:
    h = TransitionHistory()
    for t in transitions:
        h.append(
            state_before=t.state_before,
            action=t.action,
            state_after=t.state_after,
            frame_idx=t.frame_idx,
        )
    return h


@pytest.mark.unit
class TestValidateRulesAgainstHistory:
    def test_empty_history_returns_empty(self):
        proposed = (
            Rule(
                guard_spec={"action": 1},
                effects=(Effect("pos", 0, "delta", (1, 0)),),
                support=0,
                kind="movement",
            ),
        )
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history(proposed, ctx, _empty_history(), spec)
        assert result == []

    def test_empty_proposed_rules_returns_empty(self):
        history = _history_with(
            _make_transition(
                0,
                1,
                SceneState(relevant=((0, ("pos", (1, 1))),)),
                SceneState(relevant=((0, ("pos", (2, 1))),)),
            )
        )
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((), ctx, history, spec)
        assert result == []

    def test_all_rules_pass_no_counter_evidence(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert result == []

    def test_rule_mispredicts_produces_counter_evidence(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (5, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert len(result) == 1
        ce = result[0]
        assert ce.frame_idx == 0
        assert ce.action == 1
        assert ce.state_before_summary == {0: (1, 1)}
        assert ce.predicted_values[0]["pos"] == (6, 1)
        assert ce.observed_values[0]["pos"] == (2, 1)
        assert len(ce.fired_rules) >= 1

    def test_unknown_prediction_skipped(self):
        rule = Rule(
            guard_spec={"action": 99},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert result == []

    def test_multiple_transitions_mixed(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        before_matching = SceneState(relevant=((0, ("pos", (1, 1))),))
        after_matching = SceneState(relevant=((0, ("pos", (2, 1))),))
        after_mismatch = SceneState(relevant=((0, ("pos", (6, 5))),))
        history = _history_with(
            _make_transition(0, 1, before_matching, after_matching),
            _make_transition(1, 1, before_matching, after_mismatch),
        )
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert len(result) == 1
        assert result[0].frame_idx == 1

    def test_does_not_mutate_original_ctx(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (5, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        original_len = len(ctx.proposed_rules)
        _ = validate_rules_against_history((rule,), ctx, history, spec)
        assert len(ctx.proposed_rules) == original_len

    def test_terminal_dim_in_counter_evidence(self):
        from effects.state import TERMINAL_GAME_OVER

        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=0,
        )
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),), terminal="alive")
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",), include_terminal=True)
        result = validate_rules_against_history(
            (rule, movement_rule), ctx, history, spec
        )
        assert len(result) == 1
        ce = result[0]
        assert "terminal" in ce.predicted_values.get(0, {})
        assert ce.predicted_values[0]["terminal"] == TERMINAL_GAME_OVER
        assert ce.observed_values[0]["terminal"] == "alive"

    def test_fired_rules_serialized_as_dicts(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (5, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        after = SceneState(relevant=((0, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert len(result) == 1
        for r_dict in result[0].fired_rules:
            assert isinstance(r_dict, dict)
            assert "guard_spec" in r_dict
            assert "effects" in r_dict

    def test_fired_rules_excluded_when_no_effect_overlap(self):
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 13, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        size_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 5, "delta", 2),),
            support=0,
        )
        before = SceneState(
            relevant=(
                (5, ("size", 3)),
                (13, ("pos", (1, 1))),
            )
        )
        after = SceneState(
            relevant=(
                (5, ("size", 3)),
                (13, ("pos", (2, 1))),
            )
        )
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(5, 13), dims=("pos", "size"))
        result = validate_rules_against_history(
            (movement_rule, size_rule), ctx, history, spec
        )
        assert len(result) == 1
        ce = result[0]
        assert ce.predicted_values[5]["size"] == 5
        assert ce.observed_values[5]["size"] == 3
        pos_fired = [r for r in ce.fired_rules if "pos" in {e["dim"] for e in r["effects"]}]
        size_fired = [r for r in ce.fired_rules if "size" in {e["dim"] for e in r["effects"]}]
        assert len(pos_fired) == 0
        assert len(size_fired) == 1

    def test_fired_rules_included_when_effect_overlaps(self):
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 13, "delta", (5, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(relevant=((13, ("pos", (1, 1))),))
        after = SceneState(relevant=((13, ("pos", (2, 1))),))
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(13,), dims=("pos",))
        result = validate_rules_against_history((movement_rule,), ctx, history, spec)
        assert len(result) == 1
        ce = result[0]
        assert len(ce.fired_rules) == 1
        assert ce.fired_rules[0]["guard_spec"] == {"action": 1}

    def test_multiple_entity_dims(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 5, "delta", 2),),
            support=0,
        )
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=0,
            kind="movement",
        )
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (5, ("size", 3)),
            )
        )
        after = SceneState(
            relevant=(
                (0, ("pos", (2, 1))),
                (5, ("size", 3)),
            )
        )
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0, 5), dims=("pos", "size"))
        result = validate_rules_against_history(
            (rule, movement_rule), ctx, history, spec
        )
        assert len(result) == 1
        ce = result[0]
        assert 0 in ce.predicted_values
        assert 5 in ce.predicted_values
        assert "pos" in ce.predicted_values[0]
        assert "size" in ce.predicted_values[5]
        assert ce.predicted_values[5]["size"] == 5
        assert ce.observed_values[5]["size"] == 3

    def test_existing_ctx_rules_included(self):
        existing_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        proposed_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 5, "set", 99),),
            support=0,
        )
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (5, ("size", 3)),
            )
        )
        after = SceneState(
            relevant=(
                (0, ("pos", (2, 1))),
                (5, ("size", 3)),
            )
        )
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext(movement_rules=(existing_rule,))
        spec = _ProjectionSpec(entities=(0, 5), dims=("pos", "size"))
        result = validate_rules_against_history(
            (proposed_rule,), ctx, history, spec
        )
        assert len(result) == 1
        assert result[0].predicted_values[5]["size"] == 99
        assert result[0].observed_values[5]["size"] == 3

    def test_entity_not_in_historical_frame_skipped(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 1, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        before0 = SceneState(relevant=((0, ("pos", (1, 1))),))
        after0 = SceneState(relevant=((0, ("pos", (1, 1))),))
        before1 = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (1, ("pos", (5, 5))),
            )
        )
        after1 = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (1, ("pos", (6, 5))),
            )
        )
        history = _history_with(
            _make_transition(0, 1, before0, after0),
            _make_transition(1, 1, before1, after1),
        )
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0, 1), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert result == []

    def test_entity_existing_at_all_frames_still_validated(self):
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (5, 0)),),
            support=0,
            kind="movement",
        )
        before0 = SceneState(relevant=((0, ("pos", (1, 1))),))
        after0 = SceneState(relevant=((0, ("pos", (2, 1))),))
        before1 = SceneState(relevant=((0, ("pos", (2, 1))),))
        after1 = SceneState(relevant=((0, ("pos", (3, 1))),))
        history = _history_with(
            _make_transition(0, 1, before0, after0),
            _make_transition(1, 1, before1, after1),
        )
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(0,), dims=("pos",))
        result = validate_rules_against_history((rule,), ctx, history, spec)
        assert len(result) == 2
        assert result[0].frame_idx == 0
        assert result[1].frame_idx == 1

    def test_no_counter_evidence_when_no_fired_rules_overlap_residual(self):
        """Residual caused by entities changing without any fired rule
        should not produce counter-evidence — the proposed rules didn't
        cause the mismatch."""
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 13, "delta", (4, 0)),),
            support=0,
            kind="movement",
        )
        # Entity 5 changes size but no rule targets it — residual exists
        # but no fired rule overlaps, so no counter-evidence.
        before = SceneState(
            relevant=(
                (5, ("size", 3)),
                (13, ("pos", (1, 1))),
            )
        )
        after = SceneState(
            relevant=(
                (5, ("size", 2)),
                (13, ("pos", (5, 1))),
            )
        )
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext()
        spec = _ProjectionSpec(entities=(5, 13), dims=("pos", "size"))
        result = validate_rules_against_history((movement_rule,), ctx, history, spec)
        assert result == []

    def test_confirmed_rule_not_in_counter_evidence(self):
        """Confirmed rules that fire and produce residuals should NOT appear
        in counter-evidence — only proposed rules being validated should.

        Scenario: confirmed delta rule predicts e5.size=63 but observed is 61
        (wrong), while proposed movement rule for e15.pos is correct.
        The confirmed rule's residual on e5.size should NOT generate
        counter-evidence because the proposed rule doesn't overlap e5.size.
        """
        confirmed_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 5, "delta", -1),),
            support=5,
        )
        proposed_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 15, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        # confirmed_rule predicts e5.size = 64 - 1 = 63, but observed = 61
        # proposed_rule predicts e15.pos = (3,3) + (0,1) = (3,4), observed (3,4) ✓
        # Residual on e5.size (63 vs 61), but proposed rule affects e15.pos
        # → no proposed rule overlaps residual → no counter-evidence
        before = SceneState(
            relevant=(
                (5, ("size", 64)),
                (15, ("pos", (3, 3))),
            )
        )
        after = SceneState(
            relevant=(
                (5, ("size", 61)),
                (15, ("pos", (3, 4))),
            )
        )
        history = _history_with(_make_transition(0, 1, before, after))
        ctx = EffectContext(relational_rules=(confirmed_rule,))
        spec = _ProjectionSpec(entities=(5, 15), dims=("pos", "size"))
        result = validate_rules_against_history(
            (proposed_rule,), ctx, history, spec
        )
        assert result == []