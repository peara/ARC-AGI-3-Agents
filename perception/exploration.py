"""Curiosity-driven exploration with an online BFS verify/replan loop.

This is the live counterpart to Rung 5 (partial-state planning). It closes the
loop the recording-based evaluator could only check offline:

    plan -> execute one step -> re-observe -> detect divergence -> replan.

Two phases, driven by *confidence*, not by hardcoded game knowledge:

  Phase 1 (cold start). Nothing is confirmed yet — we don't know which blob we
  control. So the agent acts randomly to generate action→effect evidence. The
  registry/roles pipeline watches passively (it is action-agnostic by design).

  Phase 2 (controllable known). Once ``detect_controllable`` fires, we learn a
  movement model and use BFS to steer the controllable entity toward the
  *unknown*: first an unconfirmed entity (likely interactive), else the nearest
  unvisited frontier cell. Each planned step is verified against the next
  observation; any surprise (a block we extrapolated through, lost tracking,
  unexpected jump) drops the stale plan and forces a replan against the now
  richer movement model.

Pure-Python / numpy. The planner speaks in *action ids* and grids, never in
``GameAction``, so the same object runs online (inside an Agent) and offline
(replayed against a recording or a simulated grid world) for tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .entities import EntityCatalog, build_entities
from .objects import to_grid
from .planning import (
    MovementModel,
    PlanSpec,
    Pos,
    SceneState,
    entity_pos_at,
    learn_movement_model,
    plan_bfs,
    predict_move,
    replay_predicted,
    snapshot,
)
from .registry import ObjectRegistry
from .roles import assign_roles

RESET_ACTION = 0


@dataclass
class ExplorationConfig:
    """Knobs for the curiosity loop. Defaults tuned for 5-cell-step movers."""

    min_random_steps: int = 6      # cold-start probes before trusting detection
    min_samples: int = 3           # controllable detector: moving samples needed
    agree: float = 0.8             # controllable detector: action→disp agreement
    max_nodes: int = 10_000        # BFS node budget per plan
    reach_radius: int | None = None  # "arrived at entity" tolerance; None = step
    seed: int | None = None


@dataclass
class PlannerStatus:
    """A snapshot of what the planner just decided (for logging / tests)."""

    phase: str                     # explore_random | seek_entity | frontier | ...
    controllable_id: int | None
    target: Pos | None
    plan_len: int
    n_observed: int
    n_visited: int
    diverged: bool = False


class ExplorationPlanner:
    """Stateful curiosity planner. Feed observations, ask for actions.

    Usage (online or offline):
        planner = ExplorationPlanner(action_space=[1, 2, 3, 4])
        planner.observe(frame, produced_by=RESET_ACTION)   # first real frame
        action = planner.decide(available_actions=[1, 2, 3, 4])
        # ... apply action in the env, get next frame ...
        planner.observe(next_frame, produced_by=action)
        action = planner.decide(...)
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

        self.reg = ObjectRegistry()
        self.action_ids: list[int] = []
        self.catalog: EntityCatalog | None = None
        self.controllable_id: int | None = None
        self.model: MovementModel | None = None

        self.visited: set[Pos] = set()
        self.reached_targets: set[Pos] = set()
        self.plan: list[int] = []
        self.target: Pos | None = None

        # Verify-loop expectation: (pos_before, action, predicted_after).
        self._expect: tuple[Pos, int, Pos] | None = None
        self._n_obs = 0
        self._last_phase = "init"
        self._last_diverged = False
        self.rng = random.Random(self.cfg.seed)

    # ----------------------------------------------------------------- observe

    def observe(self, frame: object, produced_by: int) -> None:
        """Ingest the frame produced by ``produced_by`` and refresh perception."""
        grid = to_grid(frame)
        self.action_ids.append(int(produced_by))
        self.reg.update(grid)  # action-agnostic on purpose
        self._n_obs += 1

        # Rebuild the derived layers from the (now longer) trajectory history.
        self.catalog = assign_roles(
            build_entities(self.reg),
            self.reg,
            self.action_ids,
        )
        ctrl = self.catalog.controllable()
        self.controllable_id = ctrl.id if ctrl else None

        self._verify_expectation()

        pos = self._current_pos()
        if pos is not None:
            self.visited.add(pos)

    def _verify_expectation(self) -> None:
        """Compare the last prediction with what actually happened."""
        self._last_diverged = False
        if self._expect is None:
            return
        _pos_before, _action, predicted = self._expect
        self._expect = None
        actual = self._current_pos()
        if actual is None or actual != predicted:
            # Surprise: the live transition/block is now in the registry, so the
            # next learn_movement_model() will absorb it. Just drop the plan.
            self._last_diverged = True
            self.plan = []

    # ------------------------------------------------------------------ decide

    def decide(self, available_actions: list[int] | None = None) -> int:
        """Return the next action id to send to the environment."""
        actions = self._legal_actions(available_actions)
        if not actions:
            self._last_phase = "no_actions"
            self._expect = None
            return RESET_ACTION

        # Phase 1: cold start — explore randomly until a controllable emerges.
        if self.controllable_id is None or self._n_obs < self.cfg.min_random_steps:
            return self._random_action(actions, phase="explore_random")

        self._refresh_model()
        if self.model is None or not self.model.motion_by_action:
            return self._random_action(actions, phase="explore_random")

        if not self.plan:
            self._plan_toward_unknown(actions)

        if not self.plan:
            return self._random_action(actions, phase="frontier_exhausted")

        action = self.plan.pop(0)
        self._set_expectation(action)
        return action

    def _random_action(self, actions: list[int], *, phase: str) -> int:
        self._last_phase = phase
        self._expect = None
        return self.rng.choice(actions)

    def _set_expectation(self, action: int) -> None:
        state = self._snapshot_state()
        if state is None:
            self._expect = None
            return
        before = state.pos(self.controllable_id)  # type: ignore[arg-type]
        nxt = predict_move(state, action, self.model)  # type: ignore[arg-type]
        after = nxt.pos(self.controllable_id) if nxt else None  # type: ignore[arg-type]
        if before is not None and after is not None:
            self._expect = (before, action, after)
        else:
            self._expect = None

    # ------------------------------------------------------------- planning

    def _plan_toward_unknown(self, actions: list[int]) -> None:
        """Pick a curiosity target and BFS to it; leave plan empty on failure."""
        self.target = None
        start = self._snapshot_state()
        if start is None or self.model is None:
            return
        eid = self.controllable_id
        assert eid is not None

        model_actions = sorted(set(actions) & set(self.model.motion_by_action))
        if not model_actions:
            model_actions = sorted(self.model.motion_by_action)

        # Tier 1: head for an unconfirmed (likely interactive) entity.
        entity_target = self._curiosity_entity_target()
        if entity_target is not None:
            radius = self._reach_radius()
            plan = plan_bfs(
                start,
                lambda s: _within(s.pos(eid), entity_target, radius),
                model_actions,
                self.model,
                max_nodes=self.cfg.max_nodes,
            )
            if plan:
                self.plan = plan
                self.target = entity_target
                self.reached_targets.add(entity_target)
                self._last_phase = "seek_entity"
                return

        # Tier 2: head for the nearest unvisited frontier cell.
        visited = self.visited
        plan = plan_bfs(
            start,
            lambda s: s.pos(eid) not in visited,
            model_actions,
            self.model,
            max_nodes=self.cfg.max_nodes,
        )
        if plan:
            self.plan = plan
            end = replay_predicted(start, plan, self.model)
            self.target = end.pos(eid) if end else None
            self._last_phase = "frontier"
            return

        self.plan = []

    def _curiosity_entity_target(self) -> Pos | None:
        """Nearest unconfirmed, non-structural entity we have not yet reached."""
        if self.catalog is None or self.controllable_id is None:
            return None
        cur = self._current_pos()
        if cur is None:
            return None

        best: Pos | None = None
        best_d = None
        for eid, ent in self.catalog.entities.items():
            if eid == self.controllable_id:
                continue
            if ent.affordances.get("controllable") is True:
                continue
            if self._is_structural_entity(ent):
                continue
            pos = entity_pos_at(self.reg, self.catalog, eid, self.reg.frame_idx)
            if pos is None:
                continue
            radius = self._reach_radius()
            if any(_within(pos, t, radius) for t in self.reached_targets):
                continue
            if _within(cur, pos, radius):  # already standing on it
                continue
            d = abs(pos[0] - cur[0]) + abs(pos[1] - cur[1])
            if best_d is None or d < best_d:
                best_d, best = d, pos
        return best

    def _is_structural_entity(self, ent) -> bool:
        for tid in ent.members:
            track = self.reg.tracks.get(tid)
            if track and track.observations:
                if sum(o.structural for o in track.observations) > track.n_obs / 2:
                    return True
        return False

    # ------------------------------------------------------------- internals

    def _refresh_model(self) -> None:
        if self.catalog is None or self.controllable_id is None:
            self.model = None
            return
        self.model = learn_movement_model(
            self.reg,
            self.catalog,
            self.action_ids,
            self.controllable_id,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
        )

    def _snapshot_state(self) -> SceneState | None:
        if self.catalog is None or self.controllable_id is None:
            return None
        return snapshot(
            self.reg,
            self.catalog,
            PlanSpec(entities=[self.controllable_id], goal=lambda s: False),
            self.reg.frame_idx,
        )

    def _current_pos(self) -> Pos | None:
        if self.catalog is None or self.controllable_id is None:
            return None
        return entity_pos_at(
            self.reg, self.catalog, self.controllable_id, self.reg.frame_idx
        )

    def _reach_radius(self) -> int:
        if self.cfg.reach_radius is not None:
            return self.cfg.reach_radius
        if self.model and self.model.motion_by_action:
            mags = [
                max(abs(dr), abs(dc))
                for dr, dc in self.model.motion_by_action.values()
            ]
            if mags:
                return max(mags)
        return 1

    def _legal_actions(self, available: list[int] | None) -> list[int]:
        if available:
            pool = [int(a) for a in available if int(a) != RESET_ACTION]
            if pool:
                return pool
        return list(self.action_space)

    def status(self) -> PlannerStatus:
        return PlannerStatus(
            phase=self._last_phase,
            controllable_id=self.controllable_id,
            target=self.target,
            plan_len=len(self.plan),
            n_observed=self._n_obs,
            n_visited=len(self.visited),
            diverged=self._last_diverged,
        )


def _within(pos: Pos | None, target: Pos | None, radius: int) -> bool:
    if pos is None or target is None:
        return False
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= radius
