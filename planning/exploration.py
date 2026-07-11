"""Curiosity exploration planner: random cold start, BFS, verify/replan."""

from __future__ import annotations

import random

from effects import (
    EffectContext,
    Pos,
    entity_size_at,
    frame_meta_from_steps,
    inject_llm_proposals,
    learn_effect_context,
    merge_effect_context,
    predict,
    replay_predicted,
)
from effects.rules import Rule
from perception.session import RESET_ACTION, SceneSnapshot

from .adapters import snapshot_from_scene
from .heuristics import (
    ExplorationConfig,
    curiosity_entity_target,
    reach_radius,
    within,
)
from .protocol import PlannerStatus
from .search import PlanSpec, plan_bfs


class ExplorationPolicy:
    """Curiosity-driven planner with an online BFS verify/replan loop.

    Reads ``SceneSnapshot`` from a ``PerceptionSession``; does not ingest frames.
    """

    def __init__(
        self,
        action_space: list[int],
        *,
        config: ExplorationConfig | None = None,
        grid_rows: int = 64,
        grid_cols: int = 64,
    ) -> None:
        self.action_space = [a for a in action_space if a != RESET_ACTION]
        self.cfg = config or ExplorationConfig()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.visited: set[Pos] = set()
        self.reached_targets: set[Pos] = set()
        self.plan: list[int] = []
        self.target: Pos | None = None
        self._ctx: EffectContext | None = None
        self._engine_ctx: EffectContext | None = None

        self._expect: tuple[Pos, int, Pos] | None = None
        self._last_scene: SceneSnapshot | None = None
        self._last_phase = "init"
        self._last_diverged = False
        self.rng = random.Random(self.cfg.seed)

    def on_observed(self, scene: SceneSnapshot) -> None:
        """Verify last prediction and update exploration bookkeeping."""
        self._last_scene = scene
        self._verify_expectation(scene)
        pos = scene.controllable_pos()
        if pos is not None:
            self.visited.add(pos)

    def _verify_expectation(self, scene: SceneSnapshot) -> None:
        self._last_diverged = False
        if self._expect is not None:
            _pos_before, _action, predicted = self._expect
            actual = scene.controllable_pos()
            if actual is None or actual != predicted:
                self._last_diverged = True
                self.plan = []
            self._expect = None

    def _engine_plan_spec(self, scene: SceneSnapshot) -> PlanSpec:
        """Projection for residual-driven rule learning (pos + tracked sizes)."""
        ctrl = scene.controllable_id()
        entities: list[int] = []
        dims: list[str] = []
        if ctrl is not None:
            entities.append(ctrl)
            dims.append("pos")
        for eid in sorted(scene.catalog.entities):
            if ctrl is not None and eid == ctrl:
                continue
            if (
                entity_size_at(scene.registry, scene.catalog, eid, scene.frame_idx)
                is None
            ):
                continue
            if eid not in entities:
                entities.append(eid)
            if "size" not in dims:
                dims.append("size")
        return PlanSpec(
            entities=entities,
            dims=tuple(dims) if dims else ("pos",),
            goal=lambda s: False,
        )

    def decide(
        self,
        scene: SceneSnapshot,
        available_actions: list[int] | None = None,
    ) -> int:
        self._last_scene = scene
        actions = self._legal_actions(available_actions)
        if not actions:
            self._last_phase = "no_actions"
            self._expect = None
            return RESET_ACTION

        controllable_id = scene.controllable_id()
        if controllable_id is None or scene.n_observed < self.cfg.min_random_steps:
            action = self._random_action(actions, phase="explore_random")
            return self.record_step(scene, controllable_id, action)

        base = learn_effect_context(
            scene.registry,
            scene.catalog,
            list(scene.action_ids),
            frame_meta_from_steps(scene.step_observations),
            controllable_id,
            grid_rows=scene.grid_rows,
            grid_cols=scene.grid_cols,
        )
        if base is None or not base.available_actions:
            action = self._random_action(actions, phase="explore_random")
            return self.record_step(scene, controllable_id, action)

        if self._engine_ctx is None:
            self._engine_ctx = base
        else:
            self._engine_ctx = merge_effect_context(base, self._engine_ctx)
        self._ctx = self._engine_ctx

        if not self.plan:
            self._plan_toward_unknown(scene, actions, controllable_id)

        if not self.plan:
            action = self._random_action(actions, phase="frontier_exhausted")
            return self.record_step(scene, controllable_id, action)

        action = self.plan.pop(0)
        return self.record_step(scene, controllable_id, action)

    def record_step(
        self,
        scene: SceneSnapshot,
        controllable_id: int | None,
        action: int,
    ) -> int:
        """Track divergence expectation for the next observe."""
        self._expect = None
        if controllable_id is None or self._ctx is None:
            return action

        verify_state = self._snapshot_state(scene, controllable_id)
        if verify_state is None:
            return action
        nxt = predict(verify_state, action, self._ctx)
        before = verify_state.pos(controllable_id)
        after = nxt.state.pos(controllable_id) if not nxt.unknown else None
        if before is not None and after is not None:
            self._expect = (before, action, after)
        return action

    def _plan_toward_unknown(
        self,
        scene: SceneSnapshot,
        actions: list[int],
        controllable_id: int,
    ) -> None:
        self.target = None
        start = self._snapshot_state(scene, controllable_id)
        ctx = self._ctx
        if start is None or ctx is None:
            return

        legal = sorted(set(actions) & set(ctx.available_actions))
        if not legal:
            legal = list(actions)

        current = scene.controllable_pos()
        if current is not None:
            entity_target = curiosity_entity_target(
                scene,
                controllable_id=controllable_id,
                current=current,
                reached_targets=self.reached_targets,
                cfg=self.cfg,
                movement_rules=ctx.movement_rules,
            )
            if entity_target is not None:
                radius = reach_radius(self.cfg, ctx.movement_rules)
                plan, _unknowns = plan_bfs(
                    start,
                    lambda s: within(s.pos(controllable_id), entity_target, radius),
                    legal,
                    ctx,
                    max_nodes=self.cfg.max_nodes,
                )
                if plan:
                    self.plan = plan
                    self.target = entity_target
                    self.reached_targets.add(entity_target)
                    self._last_phase = "seek_entity"
                    return

        visited = self.visited
        plan, _unknowns = plan_bfs(
            start,
            lambda s: s.pos(controllable_id) not in visited,
            legal,
            ctx,
            max_nodes=self.cfg.max_nodes,
        )
        if plan:
            self.plan = plan
            end = replay_predicted(start, plan, ctx)
            self.target = end.pos(controllable_id) if end else None
            self._last_phase = "frontier"
            return

        self.plan = []

    def _random_action(self, actions: list[int], *, phase: str) -> int:
        self._last_phase = phase
        return self.rng.choice(actions)

    def _snapshot_state(
        self, scene: SceneSnapshot, controllable_id: int
    ) -> object | None:
        """Narrow projection for divergence detection (controllable-only)."""
        return snapshot_from_scene(
            scene,
            PlanSpec(entities=[controllable_id], goal=lambda s: False),
        )

    def _legal_actions(self, available: list[int] | None) -> list[int]:
        if available:
            pool = [int(a) for a in available if int(a) != RESET_ACTION]
            if pool:
                return pool
        return list(self.action_space)

    @property
    def context(self) -> EffectContext | None:
        return self._ctx

    def update_context(self, ctx: EffectContext) -> None:
        """Feed an updated EffectContext back from the agent's engine step."""
        self._ctx = ctx
        self._engine_ctx = ctx

    def inject_llm_proposals(self, proposals: tuple[Rule, ...]) -> None:
        """Inject LLM-proposed rules into the context immediately.

        Rules enter ``proposed_rules`` with support=0 right away so that
        ``predict`` and BFS see them on the same frame — no 1-step delay.
        """
        if not proposals:
            return
        if self._engine_ctx is not None:
            self._engine_ctx = inject_llm_proposals(self._engine_ctx, proposals)
            self._ctx = self._engine_ctx

    @property
    def controllable_id(self) -> int | None:
        if self._last_scene is None:
            return None
        return self._last_scene.controllable_id()

    def status(self) -> PlannerStatus:
        n_obs = self._last_scene.n_observed if self._last_scene else 0
        ctrl = self._last_scene.controllable_id() if self._last_scene else None
        return PlannerStatus(
            phase=self._last_phase,
            controllable_id=ctrl,
            target=self.target,
            plan_len=len(self.plan),
            n_observed=n_obs,
            n_visited=len(self.visited),
            diverged=self._last_diverged,
        )