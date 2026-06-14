"""Top-level forward predictor: kinematics + relational/terminal rules."""

from __future__ import annotations

from .context import EffectContext
from .kinematics import predict_move
from .state import TERMINAL_GAME_OVER, SceneState


def predict(
    state: SceneState,
    action: int,
    ctx: EffectContext,
) -> SceneState | None:
    """Predict the next symbolic state after ``action``."""
    if ctx.non_markovian and not ctx.has_confirmed(state, action):
        return None

    nxt = predict_move(state, action, ctx.movement)
    if nxt is None:
        return None

    for rule in ctx.terminal_rules:
        if rule.guard(state, action):
            nxt = rule.apply(nxt, action)

    for rule in ctx.relational_rules:
        if rule.guard(state, action):
            nxt = rule.apply(nxt, action)

    return nxt


def replay_predicted(
    start: SceneState, actions: list[int], ctx: EffectContext
) -> SceneState | None:
    """Step ``predict`` along ``actions``; None if any step fails."""
    state = start
    for action in actions:
        nxt = predict(state, action, ctx)
        if nxt is None:
            return None
        state = nxt
    return state


def is_terminal_dead_end(state: SceneState) -> bool:
    return state.terminal == TERMINAL_GAME_OVER
