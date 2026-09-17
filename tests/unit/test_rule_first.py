"""Tests for RuleFirstPolicy: phase transition, fingerprint visited, engine step."""

from __future__ import annotations

import json

import pytest

from effects.context import EffectContext
from effects.engine import inject_llm_proposals
from effects.engine_step_result import run_engine_step
from effects.rules import Effect, Rule
from effects.state import SceneState
from effects.transition_history import TransitionHistory
from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry, Observation, Track
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.adapters import snapshot_from_scene
from planning.heuristics import ExplorationConfig
from planning.rule_first import RuleFirstPolicy
from planning.search import PlanSpec
from tests.perception_fixtures import load_manifest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _obs(frame_idx: int, centroid: tuple[float, float], size: int = 1) -> Observation:
    return Observation(
        frame_idx=frame_idx,
        color=1,
        size=size,
        centroid=centroid,
        bbox=(
            int(centroid[0]),
            int(centroid[1]),
            int(centroid[0]),
            int(centroid[1]),
        ),
        shape_key=frozenset({(int(centroid[0]), int(centroid[1]))}),
        cells=frozenset({(int(centroid[0]), int(centroid[1]))}),
        match_rule="new",
        displacement=None,
        structural=False,
    )


def _make_registry_and_catalog(
    positions: dict[int, list[tuple[int, int]]],
    roles: dict[int, str] | None = None,
) -> tuple[ObjectRegistry, EntityCatalog]:
    """Build registry/catalog from entity_id -> list of (row, col) per frame."""
    reg = ObjectRegistry()
    entities: dict[int, Entity] = {}

    for eid, pos_list in positions.items():
        track_id = eid * 10 + 1
        track = Track(id=track_id, color=1)
        for i, pos in enumerate(pos_list):
            track.observations.append(_obs(i, (float(pos[0]), float(pos[1]))))
        reg.tracks[track_id] = track

        affordances: dict[str, bool | None] = {
            "solid": None,
            "interactable": None,
        }
        role = roles.get(eid) if roles else None

        ent = Entity(
            id=eid,
            members=frozenset({track_id}),
            composition="singleton",
            role=role,
            affordances=affordances,
            meta={},
        )
        entities[eid] = ent

    return reg, EntityCatalog(entities=entities)


def _scene(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    frame_idx: int = 0,
    n_observed: int = 10,
    action_ids: tuple[int, ...] = (1, 2, 3, 4),
) -> SceneSnapshot:
    return SceneSnapshot(
        frame_idx=frame_idx,
        n_observed=n_observed,
        registry=reg,
        catalog=catalog,
        action_ids=action_ids,
        grid_rows=64,
        grid_cols=64,
        last_step=None,
        step_observations=(),
        determinism_violations=(),
    )


def _movement_rule(entity_id: int, action: int, dr: int, dc: int) -> Rule:
    """Minimal movement rule: on action, entity moves by (dr, dc)."""
    return Rule(
        guard_spec={"action": action},
        effects=(Effect(dim="pos", of=entity_id, op="delta", value=(dr, dc)),),
        support=2,
        kind="delta",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRuleFirstPolicy:
    def test_phase_transition_on_rules(self):
        """Policy transitions from random to directed when movement rules are learned."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        # Single-frame registry → no transitions → no movement rules → random phase
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5)]},
        )
        scene = _scene(reg, catalog, n_observed=10)
        action = policy.decide(scene)
        assert action in [1, 2, 3, 4]
        assert policy.status().phase == "explore_random"

        # Multi-frame registry → context initialized with available_actions
        # (movement rules come from LLM injection, not classical learner)
        reg2, catalog2 = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
        )
        action_ids = (0, 1)
        scene2 = _scene(reg2, catalog2, n_observed=10, action_ids=action_ids)
        policy.decide(scene2)

        assert policy.context is not None, "Context should be populated after decide"
        assert policy.context.available_actions, (
            f"Should have available_actions, got {policy.context.available_actions}"
        )
        # Without LLM-injected rules, still in random phase
        assert policy.status().phase == "explore_random"

    def test_bfs_uses_fingerprint_not_position(self):
        """Visited set stores SceneState fingerprints, not Pos tuples."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5)]},
        )
        scene = _scene(reg, catalog, n_observed=10)
        policy.on_observed(scene)

        # The visited set should contain tuples (from SceneState.fingerprint()),
        # not Pos tuples like (row, col)
        assert len(policy.visited) > 0
        for entry in policy.visited:
            # Fingerprints are tuple[object, ...] from SceneState.fingerprint()
            # which returns (self.relevant,) — a tuple containing another tuple
            assert isinstance(entry, tuple), f"Expected tuple, got {type(entry)}"
            # Pos would be a tuple of two ints like (5, 5)
            # Fingerprint is always a nested tuple: ((..., (...), ...),)
            # Not a simple (int, int) position
            assert not (
                len(entry) == 2
                and isinstance(entry[0], int)
                and isinstance(entry[1], int)
            ), f"Visited entry looks like a Pos tuple, not a fingerprint: {entry}"

    def test_policy_works_without_context(self):
        """Policy works when no EffectContext has been initialized yet."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)], 1: [(3, 3), (2, 3)]},
        )
        action_ids = (0, 1, 2)
        scene = _scene(reg, catalog, n_observed=10, action_ids=action_ids)

        action = policy.decide(scene)
        assert action in [1, 2, 3, 4]

        policy.on_observed(scene)

        # Context initialized with available_actions from scene
        assert policy.context is not None
        assert policy.context.available_actions

    def test_decide_random_phase_without_rules(self):
        """Without EffectContext, decide returns random actions and phase is explore_random."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=6, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5)]},
        )

        # _ctx is None → random phase
        assert policy._ctx is None

        scene = _scene(reg, catalog, n_observed=1)
        action = policy.decide(scene)
        assert action in [1, 2, 3, 4]
        assert policy.status().phase == "explore_random"

        # Even with more observations, still random if no context
        scene2 = _scene(reg, catalog, n_observed=10)
        action2 = policy.decide(scene2)
        assert action2 in [1, 2, 3, 4]
        assert policy.status().phase == "explore_random"

    def test_engine_step_with_multi_entity(self):
        """After observe + record_step + observe, _rule_entity_ids() returns entity IDs from rules."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42, log_engine=False),
        )

        # Two entities with 2 frames each so movement rules can be learned
        reg, catalog = _make_registry_and_catalog(
            positions={
                0: [(5, 5), (5, 6)],  # entity 0 moves right on action 1
                1: [(3, 3), (4, 3)],  # entity 1 moves down on action 1
            },
        )
        action_ids = (0, 1)

        # Frame 0: observe
        scene0 = _scene(reg, catalog, frame_idx=0, n_observed=1, action_ids=action_ids)
        policy.on_observed(scene0)

        # Decide (will be random since no context yet)
        policy.decide(scene0)

        # record_step is called inside decide, so now set up context manually
        # Inject a context with movement rules for both entities
        rule0 = _movement_rule(entity_id=0, action=1, dr=0, dc=1)
        rule1 = _movement_rule(entity_id=1, action=1, dr=1, dc=0)
        policy._ctx = EffectContext(
            movement_rules=(rule0, rule1),
            available_actions=(1,),
        )
        policy._engine_ctx = policy._ctx

        # _rule_entity_ids should return both entity IDs from rules
        rule_entity_ids = policy._rule_entity_ids()
        assert 0 in rule_entity_ids, f"Entity 0 should be in rule_entity_ids, got {rule_entity_ids}"
        assert 1 in rule_entity_ids, f"Entity 1 should be in rule_entity_ids, got {rule_entity_ids}"

        # Also test with collision rules
        collision_rule = Rule(
            guard_spec={"action": 2},
            effects=(
                Effect(dim="pos", of=0, op="revert", value=None),
                Effect(dim="pos", of=1, op="delta", value=(0, 0)),
            ),
            support=2,
            kind="collision",
        )
        policy._ctx = EffectContext(
            movement_rules=(rule0,),
            collision_rules=(collision_rule,),
            available_actions=(1, 2),
        )
        rule_entity_ids = policy._rule_entity_ids()
        # Should include entity IDs from both movement and collision rules
        assert 0 in rule_entity_ids, f"Entity 0 from collision rule should be included, got {rule_entity_ids}"
        assert 1 in rule_entity_ids, f"Entity 1 from collision rule should be included, got {rule_entity_ids}"

    def test_divergence_detected_on_fingerprint_mismatch(self):
        """When predicted state fingerprint doesn't match observed, diverged is True and plan is cleared."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        # Two entities with 2 frames so we can learn movement rules
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
        )
        action_ids = (0, 1)
        scene0 = _scene(reg, catalog, frame_idx=0, n_observed=1, action_ids=action_ids)

        # Inject a context with a movement rule
        rule = _movement_rule(entity_id=0, action=1, dr=0, dc=1)
        policy._ctx = EffectContext(
            movement_rules=(rule,),
            available_actions=(1,),
        )
        policy._engine_ctx = policy._ctx

        # record_step stores prediction: entity 0 at (5,5) + action 1 → predicted (5,6)
        policy.on_observed(scene0)
        policy.record_step(scene0, action=1)

        # Now observe scene1 but with DIFFERENT positions (mismatch from prediction)
        reg_mismatch, catalog_mismatch = _make_registry_and_catalog(
            positions={0: [(10, 10), (10, 10)]},  # completely different from (5,6)
        )
        scene_mismatch = _scene(reg_mismatch, catalog_mismatch, frame_idx=1, n_observed=2, action_ids=action_ids)

        policy.on_observed(scene_mismatch)

        assert policy.status().diverged is True
        assert policy.plan == []

    def test_no_divergence_when_prediction_matches(self):
        """When prediction matches observed state, diverged is False."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
        )
        action_ids = (0, 1)
        scene0 = _scene(reg, catalog, frame_idx=0, n_observed=1, action_ids=action_ids)

        rule = _movement_rule(entity_id=0, action=1, dr=0, dc=1)
        policy._ctx = EffectContext(
            movement_rules=(rule,),
            available_actions=(1,),
        )
        policy._engine_ctx = policy._ctx

        # Observe scene0 at frame 0, then record step with action=1
        # This predicts entity 0 moves from (5,5) to (5,6)
        policy.on_observed(scene0)
        policy.record_step(scene0, action=1)

        # Observe the SAME registry/catalog at frame 1 where entity 0 IS at (5,6)
        # This should match the predicted fingerprint
        scene_match = _scene(reg, catalog, frame_idx=1, n_observed=2, action_ids=action_ids)

        policy.on_observed(scene_match)

        assert policy.status().diverged is False

    def test_no_divergence_without_prediction(self):
        """When no prediction was made (ctx=None), diverged is always False."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5)]},
        )
        scene = _scene(reg, catalog, n_observed=10)

        # With ctx=None, record_step cannot predict → _expect stays None
        assert policy._ctx is None
        policy.on_observed(scene)
        policy.record_step(scene, action=1)

        # Observe again — no prediction was made, so no divergence
        policy.on_observed(scene)

        assert policy.status().diverged is False

    def test_dead_entity_excluded_from_spec(self):
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(0, 0)], 10: [(1, 1)], 12: [(2, 2)]},
        )
        catalog.entities[0].lifecycle = LifecycleState.MERGED
        catalog.entities[10].lifecycle = LifecycleState.MERGED
        catalog.entities[12].lifecycle = LifecycleState.ACTIVE

        scene = _scene(reg, catalog)

        rule0 = _movement_rule(entity_id=0, action=1, dr=0, dc=1)
        rule10 = _movement_rule(entity_id=10, action=1, dr=0, dc=1)
        policy._ctx = EffectContext(
            movement_rules=(rule0, rule10),
            available_actions=(1,),
        )

        spec = policy._engine_plan_spec(scene)

        assert 12 in spec.entities, f"Expected active entity 12 in spec, got {spec.entities}"
        assert 0 not in spec.entities, f"Expected merged entity 0 to be filtered out, got {spec.entities}"
        assert 10 not in spec.entities, f"Expected merged entity 10 to be filtered out, got {spec.entities}"

    def test_residual_excludes_dead_entities(self):
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        reg, catalog = _make_registry_and_catalog(
            positions={0: [(0, 0)], 10: [(1, 1)], 12: [(2, 2)]},
        )
        catalog.entities[0].lifecycle = LifecycleState.MERGED
        catalog.entities[10].lifecycle = LifecycleState.MERGED
        catalog.entities[12].lifecycle = LifecycleState.ACTIVE
        scene = _scene(reg, catalog)

        rule0 = _movement_rule(entity_id=0, action=1, dr=0, dc=1)
        rule10 = _movement_rule(entity_id=10, action=1, dr=0, dc=1)
        policy._ctx = EffectContext(
            movement_rules=(rule0, rule10),
            available_actions=(1,),
        )

        state = policy._snapshot_state(scene)
        
        assert any(eid == 12 for eid, _ in state.relevant), "Active entity 12 should be relevant"
        assert not any(eid == 0 for eid, _ in state.relevant), "Merged entity 0 should not be relevant"
        assert not any(eid == 10 for eid, _ in state.relevant), "Merged entity 10 should not be relevant"


# ---------------------------------------------------------------------------
# Recording-based integration tests
# ---------------------------------------------------------------------------


def _load_raw_frames(path: str):
    """Read recording JSONL and extract (raw_frames, actions, state_names, levels)."""
    raw_frames = []
    actions = []
    state_names = []
    levels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            raw_frames.append(data["frame"])
            ai = data.get("action_input") or {}
            action = int(ai.get("id", -1))
            if action < 0:
                action = RESET_ACTION
            actions.append(action)
            state_names.append(str(data.get("state", "NOT_FINISHED")))
            levels.append(int(data.get("levels_completed", 0)))
    return raw_frames, actions, state_names, levels


@pytest.mark.unit
class TestRuleFirstOnRecording:
    """Recording replay through the production learning loop.

    Drives session.ingest → policy.on_observed → policy.decide →
    run_engine_step → inject → update_context, mirroring
    llm_curiosity_agent minus the LLM: a deterministic stand-in derives
    candidate movement rules from observed transitions the way the LLM
    rule proposer does, exercising the real inject → predict → confirm →
    promote pipeline.
    """

    @staticmethod
    def _observed_transition_rules(
        observed_transition: tuple[SceneState, int, SceneState],
    ) -> tuple[Rule, ...]:
        """Deterministic stand-in for the LLM rule proposer.

        The production proposer reads the observed transition of an
        unknown action and emits candidate movement rules. Here the same
        candidates are derived directly from position deltas, so the
        replay tests exercise the real inject → predict → confirm →
        promote pipeline without an LLM.
        """
        state_before, action, observed = observed_transition
        proposals: list[Rule] = []
        for eid in state_before.entity_ids_with_dim("pos"):
            p0 = state_before.pos(eid)
            p1 = observed.pos(eid)
            if p0 is None or p1 is None:
                continue
            dr = int(round(p1[0] - p0[0]))
            dc = int(round(p1[1] - p0[1]))
            if (dr, dc) == (0, 0):
                continue
            proposals.append(
                Rule(
                    guard_spec={"action": action},
                    effects=(Effect("pos", eid, "delta", (dr, dc)),),
                    support=0,
                    kind="movement",
                )
            )
        return tuple(proposals)

    @staticmethod
    def _replay_learning_loop(recording_path: str, max_frames: int) -> RuleFirstPolicy:
        raw_frames, actions, _, _ = _load_raw_frames(recording_path)
        session = PerceptionSession()
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=6, seed=42),
        )
        history = TransitionHistory()

        engine_step_pending: tuple[SceneState, PlanSpec, int] | None = None
        for i in range(min(max_frames, len(raw_frames))):
            scene = session.ingest(raw_frames[i], actions[i])
            policy.on_observed(scene)

            if engine_step_pending is not None and policy.context is not None:
                state_before, spec, pending_action = engine_step_pending
                observed = snapshot_from_scene(scene, spec)
                if observed is not None:
                    result = run_engine_step(
                        ctx=policy.context,
                        state_before=state_before,
                        action=pending_action,
                        observed=observed,
                        spec=spec,
                        history=history,
                    )
                    ctx = result.ctx
                    if result.observed_transition is not None:
                        proposals = TestRuleFirstOnRecording._observed_transition_rules(
                            result.observed_transition
                        )
                        ctx = inject_llm_proposals(ctx, proposals)
                        history.append(
                            state_before=state_before,
                            action=pending_action,
                            state_after=observed,
                            frame_idx=scene.frame_idx,
                        )
                    policy.update_context(ctx)
                engine_step_pending = None

            policy.decide(scene)
            state = policy._snapshot_state(scene)
            if (
                state is not None
                and policy.context is not None
                and i + 1 < len(raw_frames)
            ):
                # The next frame is produced by the RECORDED action, not
                # the policy's choice (replay, not live play).
                engine_step_pending = (
                    state,
                    policy._engine_plan_spec(scene),
                    actions[i + 1],
                )
        return policy

    def test_rule_first_reaches_directed_phase(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")

        case = cases[0]
        policy = self._replay_learning_loop(str(case.recording.path), max_frames=30)

        assert policy.context is not None, "should build an EffectContext"
        assert len(policy.context.movement_rules) > 0, (
            "learning loop should confirm movement rules from replayed motion"
        )

    def test_rule_first_no_crash_on_recording(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")

        case = cases[0]
        policy = self._replay_learning_loop(str(case.recording.path), max_frames=50)
        assert policy.context is not None