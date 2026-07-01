"""Tests for retroactive rule testing against transition history."""

from __future__ import annotations

from effects import (
    Effect,
    EffectContext,
    Rule,
    SceneState,
    TransitionHistory,
    engine_step,
    retroactive_test,
)


def _ctx_with_proposed(rule: Rule, *, confirm_threshold: int = 2) -> EffectContext:
    """Minimal EffectContext with one proposed rule."""
    return EffectContext(
        movement_rules=(),
        collision_rules=(),
        proposed_rules=(rule,),
        available_actions=(1, 2),
        confirm_threshold=confirm_threshold,
    )


def _state(entity_id: int, pos: tuple[int, int]) -> SceneState:
    return SceneState(relevant=((entity_id, ("pos", pos)),))


# ---------------------------------------------------------------------------
# retroactive_test
# ---------------------------------------------------------------------------


class TestRetroactiveTest:
    def test_empty_history_returns_zero(self) -> None:
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        assert retroactive_test(rule, h) == 0

    def test_matching_transitions_counted(self) -> None:
        """Rule 'action 1 → pos delta (0, 1)' matches 3 historical transitions."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        for i in range(3):
            h.append(
                state_before=_state(0, (5, i)),
                action=1,
                state_after=_state(0, (5, i + 1)),
                frame_idx=i,
            )
        assert retroactive_test(rule, h) == 3

    def test_non_matching_guard_not_counted(self) -> None:
        """Rule for action 1 doesn't match transitions with action 2."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        h.append(
            state_before=_state(0, (5, 0)),
            action=2,  # different action
            state_after=_state(0, (5, 1)),
            frame_idx=0,
        )
        assert retroactive_test(rule, h) == 0

    def test_non_matching_effect_not_counted(self) -> None:
        """Rule predicts delta (0,1) but observed delta is (0,2) — no match."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        h.append(
            state_before=_state(0, (5, 0)),
            action=1,
            state_after=_state(0, (5, 2)),  # delta (0,2), not (0,1)
            frame_idx=0,
        )
        assert retroactive_test(rule, h) == 0

    def test_partial_match_not_counted(self) -> None:
        """Guard fires but effect doesn't match — no count."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        h.append(
            state_before=_state(0, (5, 0)),
            action=1,  # guard fires
            state_after=_state(0, (6, 0)),  # delta (1,0), not (0,1)
            frame_idx=0,
        )
        assert retroactive_test(rule, h) == 0

    def test_mixed_history_counts_only_matches(self) -> None:
        """3 matching + 2 non-matching → count = 3."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        h = TransitionHistory()
        # 3 matching
        for i in range(3):
            h.append(
                state_before=_state(0, (5, i)),
                action=1,
                state_after=_state(0, (5, i + 1)),
                frame_idx=i,
            )
        # 2 non-matching (different action)
        for i in range(2):
            h.append(
                state_before=_state(0, (5, i)),
                action=2,
                state_after=_state(0, (5, i + 1)),
                frame_idx=i + 3,
            )
        assert retroactive_test(rule, h) == 3


# ---------------------------------------------------------------------------
# engine_step integration
# ---------------------------------------------------------------------------


class TestEngineStepRetroactive:
    def test_retroactive_bumps_support_immediately(self) -> None:
        """A proposed rule with 3 historical matches gets support=3 in one engine_step."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        ctx = _ctx_with_proposed(rule, confirm_threshold=2)

        # Build history with 3 matching transitions
        h = TransitionHistory()
        for i in range(3):
            h.append(
                state_before=_state(0, (5, i)),
                action=1,
                state_after=_state(0, (5, i + 1)),
                frame_idx=i,
            )

        # New transition that triggers the rule
        state_before = _state(0, (5, 3))
        observed = _state(0, (5, 4))

        result = engine_step(
            ctx,
            state_before,
            action=1,
            observed=observed,
            entity_ids=(0,),
            dims=("pos",),
            history=h,
        )
        # Rule should be promoted (support >= threshold=2)
        assert len(result.movement_rules) == 1
        assert result.movement_rules[0].support >= 2
        assert len(result.proposed_rules) == 0

    def test_no_history_no_bump(self) -> None:
        """Without history, proposed rule stays at support=0 + 1 confirm = 1."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        ctx = _ctx_with_proposed(rule, confirm_threshold=2)

        state_before = _state(0, (5, 0))
        observed = _state(0, (5, 1))

        result = engine_step(
            ctx,
            state_before,
            action=1,
            observed=observed,
            entity_ids=(0,),
            dims=("pos",),
        )
        # No history → only confirm_rules adds 1 → support=1, below threshold
        assert len(result.proposed_rules) == 1
        assert result.proposed_rules[0].support == 1

    def test_current_transition_not_double_counted(self) -> None:
        """The current transition is NOT in history during engine_step.

        If it were, confirm_rules would bump support AND retroactive would
        count it, leading to double-counting. This test verifies the
        expected usage: history is appended AFTER engine_step.
        """
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        ctx = _ctx_with_proposed(rule, confirm_threshold=10)

        # History with 2 matching transitions
        h = TransitionHistory()
        for i in range(2):
            h.append(
                state_before=_state(0, (5, i)),
                action=1,
                state_after=_state(0, (5, i + 1)),
                frame_idx=i,
            )

        # Current transition also matches
        state_before = _state(0, (5, 2))
        observed = _state(0, (5, 3))

        result = engine_step(
            ctx,
            state_before,
            action=1,
            observed=observed,
            entity_ids=(0,),
            dims=("pos",),
            history=h,
        )
        # Retroactive: 2 matches from history → support=2
        # Confirm: +1 from current transition → support=3
        # Total: 3 (NOT 4)
        proposed = result.proposed_rules
        assert len(proposed) == 1
        assert proposed[0].support == 3

    def test_empty_history_no_effect(self) -> None:
        """Empty history is same as no history."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        ctx = _ctx_with_proposed(rule, confirm_threshold=2)
        h = TransitionHistory()

        state_before = _state(0, (5, 0))
        observed = _state(0, (5, 1))

        result = engine_step(
            ctx,
            state_before,
            action=1,
            observed=observed,
            entity_ids=(0,),
            dims=("pos",),
            history=h,
        )
        # Empty history → only confirm adds 1
        assert len(result.proposed_rules) == 1
        assert result.proposed_rules[0].support == 1