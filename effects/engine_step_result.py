from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .context import EffectContext
from .engine import engine_step
from .predict import predict
from .residual import ResidualEntry, compute_residual
from .state import SceneState
from .transition_history import TransitionHistory

_UNKNOWN_ACTION_CLS: type | None = None


def _unknown_action_cls() -> type:
    global _UNKNOWN_ACTION_CLS
    if _UNKNOWN_ACTION_CLS is None:
        from planning.query import UnknownAction
        _UNKNOWN_ACTION_CLS = UnknownAction
    return _UNKNOWN_ACTION_CLS


class _ProjectionSpec(Protocol):
    """Minimal protocol for entity projection specs.

    Satisfied by ``PlanSpec`` and ``SnapshotProjection`` — any object
    with ``entities``, ``dims``, and ``include_terminal`` attributes.
    """

    entities: list[int]
    dims: tuple[str, ...]
    include_terminal: bool


@dataclass(frozen=True)
class EngineStepResult:
    ctx: EffectContext
    residual: tuple[ResidualEntry, ...]
    observed_transition: tuple[SceneState, int, SceneState] | None
    unknowns: tuple[object, ...]  # tuple[UnknownAction, ...] — lazy to avoid circular import


def run_engine_step(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    observed: SceneState,
    spec: _ProjectionSpec,
    controllable_id: int | None = None,
    history: TransitionHistory | None = None,
) -> EngineStepResult:
    """Execute a single rule engine step: predict -> residual -> engine_step.

    If the prediction is unknown, returns early with the observed transition
    and an ``unknowns`` tuple describing the unknown action.
    Otherwise, computes the residual and runs the engine step to update the context.
    """
    pred = predict(state_before, action, ctx)
    UA = _unknown_action_cls()

    if pred.unknown:
        return EngineStepResult(
            ctx=ctx,
            residual=(),
            observed_transition=(state_before, action, observed),
            unknowns=(UA(action=action, state=state_before),),
        )

    residual = compute_residual(
        pred.state,
        observed,
        entity_ids=tuple(spec.entities),
        dims=spec.dims,
        include_terminal=spec.include_terminal,
    )

    updated_ctx = engine_step(
        ctx,
        state_before,
        action,
        observed,
        entity_ids=tuple(spec.entities),
        dims=spec.dims,
        include_terminal=spec.include_terminal,
        controllable_id=controllable_id,
        history=history,
    )

    return EngineStepResult(
        ctx=updated_ctx,
        residual=residual,
        observed_transition=None,
        unknowns=(),
    )