"""Rule-first exploration policy: directed by rules, state-fingerprint novelty."""

from __future__ import annotations

import random

from effects import (
    EffectContext,
    Pos,
    entity_size_at,
    inject_llm_proposals,
    inject_validated_proposals,
)
from effects.predict import predict
from effects.rules import Rule
from effects.state import SceneState
from perception.session import RESET_ACTION, SceneSnapshot

from .adapters import snapshot_from_scene
from .heuristics import ExplorationConfig
from .protocol import PlannerStatus
from .search import PlanSpec, plan_bfs


class RuleFirstPolicy:
    """Rule-driven planner using state-fingerprint novelty.

    Phase gate: transitions from random to directed when EffectContext
    has movement rules.
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

        self.visited: set[tuple[object, ...]] = set()
        self.plan: list[int] = []
        self.target: Pos | None = None
        self._ctx: EffectContext | None = None
        self._engine_ctx: EffectContext | None = None

        self._last_scene: SceneSnapshot | None = None
        self._last_phase = "init"
        self._last_diverged = False
        self._expect: tuple[object, ...] | None = None
        self.rng = random.Random(self.cfg.seed)

    # ------------------------------------------------------------------
    # Observing
    # ------------------------------------------------------------------

    def on_observed(self, scene: SceneSnapshot) -> None:
        """Verify last prediction and update exploration bookkeeping."""
        self._last_scene = scene
        self._verify_expectation(scene)
        state = self._snapshot_state(scene)
        if state is not None:
            spec = self._engine_plan_spec(scene)
            self.visited.add(state.fingerprint(internal_dims=spec.internal_dims))

    # ------------------------------------------------------------------
    # Engine step helpers
    # ------------------------------------------------------------------

    def _rule_entity_ids(self) -> list[int]:
        """Collect entity IDs referenced in movement and collision rules."""
        ids: set[int] = set()
        for rule in self._ctx.movement_rules if self._ctx else ():
            for eff in rule.effects:
                ids.add(eff.of)
        for rule in self._ctx.collision_rules if self._ctx else ():
            for eff in rule.effects:
                ids.add(eff.of)
        return sorted(ids)

    def _engine_plan_spec(self, scene: SceneSnapshot) -> PlanSpec:
        """Projection for residual-driven rule learning.

        All tracked entities get ``pos`` and ``cells``. ``cells`` is a
        tracked dim used by overlap guards in the adapter path.
        Compound entities also get ``orientation`` so the engine can
        detect rotation that isn't captured by centroid movement alone.
        """

        rule_ids = self._rule_entity_ids()
        entities: list[int] = [
            eid for eid in rule_ids
            if eid in scene.catalog.entities and scene.catalog.entities[eid].lifecycle.value == "active"
        ]
        dims: list[str] = ["pos", "cells"]
        for eid in sorted(scene.catalog.entities):
            ent = scene.catalog.entities[eid]
            if ent.lifecycle.value not in ("active",):
                continue
            if eid in entities:
                if ent.meta.get("orientation") is not None:
                    dims.append("orientation")
                continue
            if (
                entity_size_at(scene.registry, scene.catalog, eid, scene.frame_idx)
                is None
            ):
                continue
            entities.append(eid)
            dims.append("size")
        seen: set[str] = set()
        unique_dims: list[str] = []
        for d in dims:
            if d not in seen:
                seen.add(d)
                unique_dims.append(d)
        return PlanSpec(
            entities=entities,
            dims=tuple(unique_dims) if unique_dims else ("pos",),
            internal_dims=("cells",),
            goal=lambda s: False,
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        scene: SceneSnapshot,
        available_actions: list[int] | None = None,
    ) -> int:
        self._last_scene = scene
        actions = self._legal_actions(available_actions)
        if not actions:
            self._last_phase = "no_actions"
            return RESET_ACTION

        if self._engine_ctx is None:
            self._engine_ctx = EffectContext(
                available_actions=tuple(sorted(set(scene.action_ids) - {RESET_ACTION}))
            )
            self._ctx = self._engine_ctx

        # Phase gate: random until we have movement rules
        if (
            self._ctx is None
            or len(self._ctx.movement_rules) == 0
            or scene.n_observed < self.cfg.min_random_steps
        ):
            action = self._random_action(actions, phase="explore_random")
            return self.record_step(scene, action)

        if not self._ctx.available_actions:
            action = self._random_action(actions, phase="explore_random")
            return self.record_step(scene, action)

        if not self.plan:
            self._plan_toward_unknown(scene, actions)

        if not self.plan:
            action = self._random_action(actions, phase="frontier_exhausted")
            return self.record_step(scene, action)

        action = self.plan.pop(0)
        return self.record_step(scene, action)

    def record_step(
        self,
        scene: SceneSnapshot,
        action: int,
    ) -> int:
        """Predict next state and store fingerprint for divergence verification."""
        self._expect = None
        if self._ctx is None:
            return action
        verify_state = self._snapshot_state(scene)
        if verify_state is None:
            return action
        pred = predict(verify_state, action, self._ctx)
        if pred.unknown:
            return action
        spec = self._engine_plan_spec(scene)
        self._expect = pred.state.fingerprint(internal_dims=spec.internal_dims)
        return action

    def _verify_expectation(self, scene: SceneSnapshot) -> None:
        """Compare predicted state fingerprint against observed state."""
        self._last_diverged = False
        if self._expect is None:
            return
        observed = self._snapshot_state(scene)
        if observed is None:
            self._last_diverged = True
            self.plan = []
            self._expect = None
            return
        spec = self._engine_plan_spec(scene)
        if observed.fingerprint(internal_dims=spec.internal_dims) != self._expect:
            self._last_diverged = True
            self.plan = []
        self._expect = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_toward_unknown(
        self,
        scene: SceneSnapshot,
        actions: list[int],
    ) -> None:
        """BFS toward novel state fingerprints."""
        self.target = None
        start = self._snapshot_state(scene)
        ctx = self._ctx
        if start is None or ctx is None:
            return

        legal = sorted(set(actions) & set(ctx.available_actions))
        if not legal:
            legal = list(actions)

        visited = self.visited
        spec = self._engine_plan_spec(scene)
        plan, _unknowns = plan_bfs(
            start,
            lambda s: s.fingerprint(internal_dims=spec.internal_dims) not in visited,
            legal,
            ctx,
            max_nodes=self.cfg.max_nodes,
            internal_dims=spec.internal_dims,
        )
        if plan:
            self.plan = plan
            self._last_phase = "frontier"
            return

        self.plan = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_action(self, actions: list[int], *, phase: str) -> int:
        self._last_phase = phase
        return self.rng.choice(actions)

    def _snapshot_state(self, scene: SceneSnapshot) -> SceneState | None:
        """Build SceneState covering all rule-tracked entities."""
        spec = self._engine_plan_spec(scene)
        return snapshot_from_scene(scene, spec)

    def _legal_actions(self, available: list[int] | None) -> list[int]:
        if available:
            pool = [int(a) for a in available if int(a) != RESET_ACTION]
            if pool:
                return pool
        return list(self.action_space)

    # ------------------------------------------------------------------
    # Properties / status
    # ------------------------------------------------------------------

    @property
    def context(self) -> EffectContext | None:
        return self._ctx

    def update_context(self, ctx: EffectContext) -> None:
        """Feed an updated EffectContext back from the agent's engine step."""
        self._ctx = ctx
        self._engine_ctx = ctx

    def inject_llm_proposals(self, proposals: tuple[Rule, ...]) -> None:
        """Inject LLM-proposed rules into the context immediately."""
        if not proposals:
            return
        if self._engine_ctx is not None:
            self._engine_ctx = inject_llm_proposals(self._engine_ctx, proposals)
            self._ctx = self._engine_ctx

    def inject_validated_proposals(self, proposals: tuple[Rule, ...]) -> None:
        """Inject history-validated rules directly into confirmed buckets."""
        if not proposals:
            return
        if self._engine_ctx is not None:
            self._engine_ctx = inject_validated_proposals(self._engine_ctx, proposals)
            self._ctx = self._engine_ctx

    def status(self) -> PlannerStatus:
        n_obs = self._last_scene.n_observed if self._last_scene else 0
        return PlannerStatus(
            phase=self._last_phase,
            target=self.target,
            plan_len=len(self.plan),
            n_observed=n_obs,
            n_visited=len(self.visited),
            diverged=self._last_diverged,
        )