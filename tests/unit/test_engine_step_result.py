"""Tests for EngineStepResult and run_engine_step."""
from __future__ import annotations

import pytest

from effects import EffectContext, SceneState
from effects.engine_step_result import EngineStepResult, run_engine_step
from planning.search import PlanSpec


class TestEngineStepResult:
    """Test the EngineStepResult dataclass."""

    def test_frozen(self) -> None:
        """EngineStepResult should be frozen (immutable)."""
        ctx = EffectContext()
        result = EngineStepResult(
            ctx=ctx,
            residual=(),
            observed_transition=None,
        )
        with pytest.raises(AttributeError):
            result.ctx = ctx  # type: ignore[misc]

    def test_fields(self) -> None:
        """EngineStepResult should have the expected fields."""
        ctx = EffectContext()
        result = EngineStepResult(
            ctx=ctx,
            residual=(),
            observed_transition=None,
        )
        assert result.ctx is ctx
        assert result.residual == ()
        assert result.observed_transition is None


class TestRunEngineStep:
    """Test the run_engine_step function."""

    def test_unknown_prediction_returns_early(self) -> None:
        """When prediction is unknown, return observed_transition and no residual."""
        ctx = EffectContext()
        state_before = SceneState({})
        action = 1
        observed = SceneState({})
        spec = PlanSpec(entities=[0], dims=("pos",), goal=lambda s: False)

        result = run_engine_step(
            ctx=ctx,
            state_before=state_before,
            action=action,
            observed=observed,
            spec=spec,
            controllable_id=None,
            history=None,
        )

        assert result.ctx is ctx  # ctx unchanged when prediction unknown
        assert result.residual == ()
        assert result.observed_transition == (state_before, action, observed)

    def test_known_prediction_computes_residual(self) -> None:
        """When prediction is known, compute residual and run engine step."""
        # Create a context with a movement rule so predict returns a known result
        from effects import Effect, Rule

        # Add a rule so predict is not unknown
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect(dim="pos", of=0, op="set", value=(5, 10)),),
            support=1,
        )
        object.__setattr__(rule, "kind", "movement")
        
        ctx = EffectContext(proposed_rules=(rule,))

        # SceneState.relevant MUST be a tuple of (entity_id, (dim, value))
        state_before = SceneState(relevant=((0, ("pos", (10, 10))),))
        action = 1
        observed = SceneState(relevant=((0, ("pos", (5, 10))),))
        spec = PlanSpec(entities=[0], dims=("pos",), goal=lambda s: False)

        result = run_engine_step(
            ctx=ctx,
            state_before=state_before,
            action=action,
            observed=observed,
            spec=spec,
            controllable_id=0,
            history=None,
        )

        # With a known prediction, residual should be computed and observed_transition should be None
        assert result.observed_transition is None
        assert isinstance(result.ctx, EffectContext)
        # ctx should be updated (not the same object)
        assert result.ctx is not ctx
