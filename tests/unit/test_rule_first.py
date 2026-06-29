"""Tests for RuleFirstPolicy: phase transition, fingerprint visited, no controllable_id, engine step."""

from __future__ import annotations

import json

import pytest

from effects.context import EffectContext
from effects.rules import Effect, Rule
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry, Observation, Track
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.heuristics import ExplorationConfig
from planning.rule_first import RuleFirstPolicy
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
    controllable_ids: set[int] | None = None,
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
            "controllable": True if controllable_ids and eid in controllable_ids else None,
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

        # Multi-frame registry → transitions exist → movement rules produced
        # Entity 0 moves from (5,5) to (5,6) between frames
        reg2, catalog2 = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)]},
        )
        action_ids = (0, 1)
        scene2 = _scene(reg2, catalog2, n_observed=10, action_ids=action_ids)
        policy.decide(scene2)

        assert policy.context is not None, "Context should be populated after decide"
        assert len(policy.context.movement_rules) > 0, (
            f"Should have movement rules from transitions, got {policy.context.movement_rules}"
        )
        assert policy.status().phase != "explore_random"

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

    def test_no_controllable_dependency(self):
        """Policy works when controllable_id() returns None."""
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=0, seed=42),
        )

        # Two entities with multi-frame positions, neither controllable
        reg, catalog = _make_registry_and_catalog(
            positions={0: [(5, 5), (5, 6)], 1: [(3, 3), (2, 3)]},
        )
        action_ids = (0, 1, 2)
        scene = _scene(reg, catalog, n_observed=10, action_ids=action_ids)
        assert scene.controllable_id() is None

        action = policy.decide(scene)
        assert action in [1, 2, 3, 4]

        policy.on_observed(scene)

        assert policy.controllable_id is None

        # With transitions, learn_effect_context_multi produces rules
        assert policy.context is not None
        assert len(policy.context.movement_rules) > 0

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

        # Two entities with 2 frames each so learn_effect_context_multi can learn movement
        reg, catalog = _make_registry_and_catalog(
            positions={
                0: [(5, 5), (5, 6)],  # entity 0 moves right on action 1
                1: [(3, 3), (4, 3)],  # entity 1 moves down on action 1
            },
            controllable_ids={0},
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
    """Integration test: RuleFirstPolicy on a recording.

    V2 policy should reach directed phase even when controllable_id
    detection fails (which happens in wa30 due to track fragmentation).
    """

    def test_rule_first_reaches_directed_phase(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")

        # Prefer wa30, fall back to first available
        wa30_cases = [c for c in cases if "wa30" in c.recording.name]
        case = wa30_cases[0] if wa30_cases else cases[0]

        raw_frames, actions, _, _ = _load_raw_frames(str(case.recording.path))
        if not raw_frames:
            pytest.skip("recording file has no frames")

        session = PerceptionSession()
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=6, seed=42),
        )

        max_frames = min(30, len(raw_frames))
        last_action = RESET_ACTION

        for i in range(max_frames):
            scene = session.ingest(raw_frames[i], last_action)
            policy.on_observed(scene)
            action = policy.decide(scene)
            last_action = action

        # V2 should learn movement rules even without controllable_id
        assert policy.context is not None, "V2 should build an EffectContext"
        assert len(policy.context.movement_rules) > 0, (
            "V2 should learn movement rules for multiple entities"
        )
        # V2 never has controllable_id
        assert policy.controllable_id is None

    def test_rule_first_no_crash_on_recording(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")

        case = cases[0]
        raw_frames, actions, _, _ = _load_raw_frames(str(case.recording.path))
        if not raw_frames:
            pytest.skip("recording file has no frames")

        session = PerceptionSession()
        policy = RuleFirstPolicy(
            action_space=[1, 2, 3, 4],
            config=ExplorationConfig(min_random_steps=6, seed=42),
        )

        max_frames = min(50, len(raw_frames))
        last_action = RESET_ACTION

        for i in range(max_frames):
            scene = session.ingest(raw_frames[i], last_action)
            policy.on_observed(scene)
            action = policy.decide(scene)
            last_action = action