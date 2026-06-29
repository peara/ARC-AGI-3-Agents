"""Forward predictor: rules-only path, returns Prediction."""

from __future__ import annotations

from dataclasses import dataclass

from .context import EffectContext
from .state import TERMINAL_GAME_OVER, SceneState


@dataclass(frozen=True)
class Prediction:
    """Result of predict(): known outcome or unknown action-effect pair.

    When unknown=True, no rule covers the (state, action) pair — the agent
    should try it (curiosity signal). When unknown=False, the rule engine
    produced a known next state.
    """

    state: SceneState
    unknown: bool = False


def predict(
    state: SceneState,
    action: int,
    ctx: EffectContext,
    *,
    entity_cells: dict[int, frozenset[tuple[int, int]]] | None = None,
) -> Prediction:
    """Predict the next symbolic state after ``action``.

    Rules-only path: confirmed and proposed movement rules propose candidate
    positions, collision rules revert positions that collide, terminal and
    relational rules apply effects. If no movement rule guard matches
    (confirmed or proposed), returns ``unknown=True``.

    Proposed rules (support=0) are treated the same as confirmed — they make
    the action "known" so the engine can confirm or prune them via observation.

    Collision rules evaluate guards against the pre-action state. This matches
    how collision rules are learned — the guard captures the entity's position
    before the movement, and ``op="revert"`` restores that position.
    Effects with ``op="revert"`` restore values from ``state_before``.
    """

    nxt: SceneState = state
    any_fired = False
    for rule in ctx.movement_rules:
        if rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
            any_fired = True
    for rule in ctx.proposed_rules:
        if rule.kind == "movement" and rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
            any_fired = True
    if not any_fired:
        return Prediction(state, unknown=True)

    for rule in ctx.collision_rules:
        if rule.guard(state, action, entity_cells=entity_cells):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.proposed_rules:
        if rule.kind == "collision" and rule.guard(state, action, entity_cells=entity_cells):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.terminal_rules:
        if rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.proposed_rules:
        if rule.kind == "terminal" and rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.relational_rules:
        if rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.proposed_rules:
        if rule.kind == "delta" and rule.guard(state, action):
            nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    return Prediction(nxt, unknown=False)


def replay_predicted(
    start: SceneState, actions: list[int], ctx: EffectContext
) -> SceneState | None:
    """Step ``predict`` along ``actions``; None if any step is unknown."""
    state = start
    for action in actions:
        pred = predict(state, action, ctx)
        if pred.unknown:
            return None
        state = pred.state
    return state


def is_terminal_dead_end(state: SceneState) -> bool:
    return state.terminal == TERMINAL_GAME_OVER