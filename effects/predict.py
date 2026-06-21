"""Top-level forward predictor: kinematics + relational/terminal rules."""

from __future__ import annotations

from .context import EffectContext
from .kinematics import predict_move
from .state import TERMINAL_GAME_OVER, SceneState


def predict(
    state: SceneState,
    action: int,
    ctx: EffectContext,
    *,
    entity_cells: dict[int, frozenset[tuple[int, int]]] | None = None,
) -> SceneState | None:
    """Predict the next symbolic state after ``action``.

    Dual-path logic:
    - If ``ctx.movement_rules`` is non-empty, try movement rules first.
      If no movement rule's guard matches, fall back to ``predict_move``.
    - If ``ctx.movement_rules`` is empty, use ``predict_move`` directly
      (current/fallback behaviour).
    After movement resolution, terminal and relational rules are applied
    in order, with ``state_before`` and ``entity_cells`` forwarded to
    ``Rule.apply()`` when available.
    """
    if ctx.non_markovian and not ctx.has_confirmed(state, action):
        return None

    if ctx.movement_rules:
        # Movement rules path: try movement rules first
        movement_applied = False
        nxt: SceneState = state
        for rule in ctx.movement_rules:
            if rule.guard(state, action):
                nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
                movement_applied = True
        if not movement_applied:
            # No movement rule matched; fall back to kinematic model
            moved = predict_move(state, action, ctx.movement)
            if moved is None:
                return None
            nxt = moved
        # Apply effect rules (terminal + relational)
        for rule in ctx.terminal_rules:
            if rule.guard(state, action):
                nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
        for rule in ctx.relational_rules:
            if rule.guard(state, action):
                nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
        return nxt
    else:
        # Fallback path: current behaviour using predict_move
        moved = predict_move(state, action, ctx.movement)
        if moved is None:
            return None
        nxt = moved
        for rule in ctx.terminal_rules:
            if rule.guard(state, action):
                nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
        for rule in ctx.relational_rules:
            if rule.guard(state, action):
                nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
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
