"""Curiosity exploration planner: random cold start, BFS, verify/replan."""

from __future__ import annotations

import random

from effects import (
    MovementModel,
    Pos,
    SceneState,
    learn_movement_model,
    predict_move,
    replay_predicted,
)
from perception.session import RESET_ACTION, SceneSnapshot

from .heuristics import (
    ExplorationConfig,
    curiosity_entity_target,
    reach_radius,
    within,
)
from .protocol import PlannerStatus
from .search import PlanSpec, plan_bfs, snapshot


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
        self._model: MovementModel | None = None

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
        if self._expect is None:
            return
        _pos_before, _action, predicted = self._expect
        self._expect = None
        actual = scene.controllable_pos()
        if actual is None or actual != predicted:
            self._last_diverged = True
            self.plan = []

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
            return self._random_action(actions, phase="explore_random")

        self._model = learn_movement_model(
            scene.registry,
            scene.catalog,
            list(scene.action_ids),
            controllable_id,
            grid_rows=scene.grid_rows,
            grid_cols=scene.grid_cols,
        )
        if self._model is None or not self._model.motion_by_action:
            return self._random_action(actions, phase="explore_random")

        if not self.plan:
            self._plan_toward_unknown(scene, actions, controllable_id)

        if not self.plan:
            return self._random_action(actions, phase="frontier_exhausted")

        action = self.plan.pop(0)
        self._set_expectation(scene, controllable_id, action)
        return action

    def _random_action(self, actions: list[int], *, phase: str) -> int:
        self._last_phase = phase
        self._expect = None
        return self.rng.choice(actions)

    def _set_expectation(
        self, scene: SceneSnapshot, controllable_id: int, action: int
    ) -> None:
        state = self._snapshot_state(scene, controllable_id)
        if state is None or self._model is None:
            self._expect = None
            return
        before = state.pos(controllable_id)
        nxt = predict_move(state, action, self._model)
        after = nxt.pos(controllable_id) if nxt else None
        if before is not None and after is not None:
            self._expect = (before, action, after)
        else:
            self._expect = None

    def _plan_toward_unknown(
        self,
        scene: SceneSnapshot,
        actions: list[int],
        controllable_id: int,
    ) -> None:
        self.target = None
        start = self._snapshot_state(scene, controllable_id)
        model = self._model
        if start is None or model is None:
            return

        model_actions = sorted(set(actions) & set(model.motion_by_action))
        if not model_actions:
            model_actions = sorted(model.motion_by_action)

        current = scene.controllable_pos()
        if current is not None:
            entity_target = curiosity_entity_target(
                scene,
                controllable_id=controllable_id,
                current=current,
                reached_targets=self.reached_targets,
                cfg=self.cfg,
                model=model,
            )
            if entity_target is not None:
                radius = reach_radius(self.cfg, model)
                plan = plan_bfs(
                    start,
                    lambda s: within(s.pos(controllable_id), entity_target, radius),
                    model_actions,
                    model,
                    max_nodes=self.cfg.max_nodes,
                )
                if plan:
                    self.plan = plan
                    self.target = entity_target
                    self.reached_targets.add(entity_target)
                    self._last_phase = "seek_entity"
                    return

        visited = self.visited
        plan = plan_bfs(
            start,
            lambda s: s.pos(controllable_id) not in visited,
            model_actions,
            model,
            max_nodes=self.cfg.max_nodes,
        )
        if plan:
            self.plan = plan
            end = replay_predicted(start, plan, model)
            self.target = end.pos(controllable_id) if end else None
            self._last_phase = "frontier"
            return

        self.plan = []

    def _snapshot_state(
        self, scene: SceneSnapshot, controllable_id: int
    ) -> SceneState | None:
        return snapshot(
            scene.registry,
            scene.catalog,
            PlanSpec(entities=[controllable_id], goal=lambda s: False),
            scene.frame_idx,
        )

    def _legal_actions(self, available: list[int] | None) -> list[int]:
        if available:
            pool = [int(a) for a in available if int(a) != RESET_ACTION]
            if pool:
                return pool
        return list(self.action_space)

    @property
    def model(self) -> MovementModel | None:
        return self._model

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
