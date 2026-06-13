"""Offline evaluation of plans against a fixed recording (any game).

A recording is one observed trajectory. We can validate that each plan step
either matches an observed (pos, action) -> next_pos transition from that
episode, or is an extrapolation with no contradicting observation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .entities import EntityCatalog
from .planning import (
    MovementModel,
    PlanSpec,
    Pos,
    SceneState,
    entity_pos_at,
    goal_pos,
    learn_movement_model,
    plan_bfs,
    predict_move,
    replay_predicted,
    snapshot,
)
from .registry import ObjectRegistry

ObservedStep = tuple[Pos, int, Pos]


@dataclass(frozen=True)
class StepCheck:
    step_index: int
    action: int
    pos_before: Pos
    predicted_pos: Pos | None
    observed_next: Pos | None
    status: str  # matched | extrapolated | predict_failed | diverged


@dataclass(frozen=True)
class PlanEvalResult:
    plan: list[int]
    goal_pos: Pos
    predict_reached_goal: bool
    steps: tuple[StepCheck, ...]
    matched_steps: int
    extrapolated_steps: int
    diverged_steps: int


def collect_observed_steps(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
) -> list[ObservedStep]:
    out: list[ObservedStep] = []
    for fidx in range(1, len(action_ids)):
        pos_before = entity_pos_at(reg, catalog, entity_id, fidx - 1)
        pos_after = entity_pos_at(reg, catalog, entity_id, fidx)
        if pos_before is None or pos_after is None:
            continue
        out.append((pos_before, int(action_ids[fidx]), pos_after))
    return out


def _observed_next(
    observed: list[ObservedStep], pos: Pos, action: int
) -> Pos | None:
    hits = [nxt for p, a, nxt in observed if p == pos and a == action]
    if not hits:
        return None
    return hits[0]


def verify_plan_on_recording(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    start_frame: int,
    plan: list[int],
    model: MovementModel,
    goal: Pos,
) -> PlanEvalResult:
    start = snapshot(
        reg,
        catalog,
        PlanSpec(entities=[entity_id], goal=lambda s: False),
        start_frame,
    )
    if start is None:
        raise ValueError(f"cannot snapshot entity {entity_id} at frame {start_frame}")

    observed = collect_observed_steps(reg, catalog, action_ids, entity_id)
    checks: list[StepCheck] = []
    state: SceneState = start

    for i, action in enumerate(plan):
        pos_before = state.pos(entity_id)
        if pos_before is None:
            checks.append(
                StepCheck(i, action, (0, 0), None, None, "predict_failed")
            )
            break

        nxt = predict_move(state, action, model)
        if nxt is None:
            checks.append(
                StepCheck(i, action, pos_before, None, None, "predict_failed")
            )
            break

        pred_pos = nxt.pos(entity_id)
        obs_next = _observed_next(observed, pos_before, action)
        if obs_next is not None and pred_pos != obs_next:
            status = "diverged"
        elif obs_next is not None:
            status = "matched"
        else:
            status = "extrapolated"

        checks.append(
            StepCheck(i, action, pos_before, pred_pos, obs_next, status)
        )
        state = nxt

    end = replay_predicted(start, plan, model)
    end_pos = end.pos(entity_id) if end else None

    return PlanEvalResult(
        plan=plan,
        goal_pos=goal,
        predict_reached_goal=end_pos == goal,
        steps=tuple(checks),
        matched_steps=sum(1 for c in checks if c.status == "matched"),
        extrapolated_steps=sum(1 for c in checks if c.status == "extrapolated"),
        diverged_steps=sum(1 for c in checks if c.status == "diverged"),
    )


def plan_and_evaluate(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    start_frame: int,
    goal_frame: int,
    *,
    max_nodes: int = 10_000,
) -> PlanEvalResult | None:
    """Plan between two frames and evaluate against the recording."""
    start = snapshot(
        reg,
        catalog,
        PlanSpec(entities=[entity_id], goal=lambda s: False),
        start_frame,
    )
    goal_snap = snapshot(
        reg,
        catalog,
        PlanSpec(entities=[entity_id], goal=lambda s: False),
        goal_frame,
    )
    if start is None or goal_snap is None:
        return None

    target = goal_snap.pos(entity_id)
    if target is None:
        return None

    model = learn_movement_model(reg, catalog, action_ids, entity_id)
    if model is None or not model.motion_by_action:
        return None

    plan = plan_bfs(
        start,
        goal_pos(entity_id, target),
        sorted(model.motion_by_action),
        model,
        max_nodes=max_nodes,
    )
    if plan is None:
        return None

    return verify_plan_on_recording(
        reg,
        catalog,
        action_ids,
        entity_id,
        start_frame,
        plan,
        model,
        target,
    )
