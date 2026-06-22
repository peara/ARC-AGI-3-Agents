"""Offline evaluation of plans against a fixed recording (any game)."""

from __future__ import annotations

from dataclasses import dataclass

from effects import (
    EffectContext,
    Pos,
    SceneState,
    entity_pos_at,
    frame_meta_from_steps,
    learn_effect_context,
    predict,
    replay_predicted,
)
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry
from perception.session import PerceptionSession

from .search import PlanSpec, goal_pos, plan_bfs, snapshot

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


def build_effect_context(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    *,
    frame_meta=None,
    step_observations=(),
    non_markovian: bool = False,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> EffectContext | None:
    meta = frame_meta
    if meta is None:
        meta = frame_meta_from_steps(step_observations)
    return learn_effect_context(
        reg,
        catalog,
        action_ids,
        meta,
        entity_id,
        non_markovian=non_markovian,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )


def verify_plan_on_recording(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    start_frame: int,
    plan: list[int],
    ctx: EffectContext,
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

        nxt = predict(state, action, ctx)
        if nxt.unknown:
            checks.append(
                StepCheck(i, action, pos_before, None, None, "predict_failed")
            )
            break

        nxt_state = nxt.state
        pred_pos = nxt_state.pos(entity_id)
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
        state = nxt_state

    end = replay_predicted(start, plan, ctx)
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
    non_markovian: bool = False,
    step_observations=(),
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

    ctx = build_effect_context(
        reg,
        catalog,
        action_ids,
        entity_id,
        step_observations=step_observations,
        non_markovian=non_markovian,
    )
    if ctx is None or not ctx.available_actions:
        return None

    plan = plan_bfs(
        start,
        goal_pos(entity_id, target),
        sorted(ctx.available_actions),
        ctx,
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
        ctx,
        target,
    )


def plan_and_evaluate_session(
    session: PerceptionSession,
    entity_id: int,
    start_frame: int,
    goal_frame: int,
    *,
    max_nodes: int = 10_000,
) -> PlanEvalResult | None:
    scene = session.snapshot()
    non_markov = len(scene.determinism_violations) > 0
    return plan_and_evaluate(
        session.registry,
        scene.catalog,
        list(session.action_ids),
        entity_id,
        start_frame,
        goal_frame,
        max_nodes=max_nodes,
        non_markovian=non_markov,
        step_observations=scene.step_observations,
    )
