"""Effects layer: rules, learning, and predict pipeline (slice 2)."""

from __future__ import annotations

import pytest

from effects import (
    Effect,
    EffectContext,
    Prediction,
    Rule,
    SceneState,
    learn_effect_context,
    load_recording_meta,
    merge_effect_context,
    predict,
)
from effects.dsl import dsl_to_rule, rule_to_dsl
from effects.guard_parse import evaluate_guard, parse_guard_clauses
from effects.state import TERMINAL_GAME_OVER
from perception.session import PerceptionSession
from planning.search import plan_bfs
from tests.perception_fixtures import (
    REPO_ROOT,
    load_effects_expectations,
    load_manifest,
)

G50T_PATH = REPO_ROOT / (
    "recordings/g50t-5849a774.curiosity.200."
    "31c022a3-dd4e-4ebe-8ad9-c237c6053bb1.recording.jsonl"
)


@pytest.mark.unit
class TestLearnEffectContext:
    @pytest.fixture
    def g50t_session(self):
        if not G50T_PATH.is_file():
            pytest.skip("g50t recording missing")
        session, _ = PerceptionSession.from_recording(G50T_PATH)
        return session

    def test_g50t_learns_terminal_rule(self, g50t_session):
        scene = g50t_session.snapshot()
        ctrl = scene.controllable_id()
        assert ctrl is not None
        meta = load_recording_meta(G50T_PATH)
        ctx = learn_effect_context(
            g50t_session.registry,
            scene.catalog,
            list(g50t_session.action_ids),
            meta,
            ctrl,
        )
        assert ctx is not None
        game_over = [
            r for r in ctx.terminal_rules
            if r.kind == "terminal" and any(e.dim == "terminal" and e.value == TERMINAL_GAME_OVER for e in r.effects)
        ]
        assert game_over, "expected GAME_OVER terminal rule on g50t"
        assert any(e.value == TERMINAL_GAME_OVER for e in game_over[0].effects)

    def test_g50t_learns_counter_rules(self, g50t_session):
        scene = g50t_session.snapshot()
        ctrl = scene.controllable_id()
        assert ctrl is not None
        meta = load_recording_meta(G50T_PATH)
        ctx = learn_effect_context(
            g50t_session.registry,
            scene.catalog,
            list(g50t_session.action_ids),
            meta,
            ctrl,
        )
        assert ctx is not None
        growth = [
            r
            for r in ctx.relational_rules
            if r.kind == "delta" and any(e.dim == "size" and e.value == 1 for e in r.effects)
        ]
        assert growth, "expected counter +1 rules on g50t"

    def test_ls20_no_terminal_rules(self):
        cases = [c for c in load_manifest() if c.recording.name == "ls20-random-legal"]
        if not cases:
            pytest.skip("ls20 recording missing")
        path = cases[0].recording.path
        session, _ = PerceptionSession.from_recording(path)
        scene = session.snapshot()
        ctrl = scene.controllable_id()
        assert ctrl is not None
        ctx = learn_effect_context(
            session.registry,
            scene.catalog,
            list(session.action_ids),
            load_recording_meta(path),
            ctrl,
        )
        assert ctx is not None
        assert not ctx.terminal_rules


@pytest.mark.unit
class TestPredictPipeline:
    def test_terminal_rule_sets_game_over(self):
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [5, 5]},
                ]
            },
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=1,
        )
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(movement_rule,), available_actions=(1,), terminal_rules=(rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        nxt = predict(start, 1, ctx)
        assert not nxt.unknown
        assert nxt.state.terminal == TERMINAL_GAME_OVER
        assert nxt.state.pos(0) == (5, 6)

    def test_counter_rule_updates_size(self):
        movement_rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=1,
            kind="movement",
        )
        rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 1),),
            support=3,
        )
        ctx = EffectContext(movement_rules=(movement_rule,), available_actions=(2,), relational_rules=(rule,))
        start = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (3, ("size", 10)),
            )
        )
        nxt = predict(start, 2, ctx)
        assert not nxt.unknown
        assert nxt.state.get(3, "size") == 11
        assert nxt.state.pos(0) == (1, 1)


@pytest.mark.unit
class TestBFSPrunesGameOver:
    def test_game_over_branch_not_expanded(self):
        movement_rule_1 = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=1,
            kind="movement",
        )
        movement_rule_2 = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=1,
            kind="movement",
        )
        dead = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [0, 0]},
                ]
            },
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=1,
        )
        ctx = EffectContext(movement_rules=(movement_rule_1, movement_rule_2), available_actions=(1, 2), terminal_rules=(dead,))
        start = SceneState(relevant=((0, ("pos", (0, 0))),))
        plan, unknowns = plan_bfs(
            start,
            lambda s: s.get(0, "pos") == (1, 0),
            [1, 2],
            ctx,
            max_nodes=20,
        )
        assert plan == [2]


@pytest.mark.unit
class TestEffectsManifest:
    @pytest.fixture(params=load_effects_expectations(), ids=lambda e: e.recording.name)
    def expect(self, request):
        if not request.param.recording.path.is_file():
            pytest.skip("recording missing")
        return request.param

    def test_manifest_effect_expectations(self, expect):
        session, _ = PerceptionSession.from_recording(expect.recording.path)
        scene = session.snapshot()
        ctrl = scene.controllable_id()
        assert ctrl is not None
        ctx = learn_effect_context(
            session.registry,
            scene.catalog,
            list(session.action_ids),
            load_recording_meta(expect.recording.path),
            ctrl,
        )
        assert ctx is not None
        if expect.expect_terminal_rule:
            assert ctx.terminal_rules
        else:
            assert not ctx.terminal_rules
        if expect.expect_counter_rule:
            assert ctx.relational_rules
        else:
            assert not ctx.relational_rules


def _make_rule(**overrides):
    defaults = dict(
        guard_spec={"action": 1},
        effects=(Effect("size", 3, "delta", 1),),
        support=1,
    )
    defaults.update(overrides)
    return Rule(**defaults)


@pytest.mark.unit
class TestEffectContextMovementRules:
    def test_construction_with_explicit_movement_rules(self):
        rule = _make_rule()
        ctx = EffectContext(movement_rules=(rule,))
        assert ctx.movement_rules == (rule,)

    def test_construction_default_movement_rules(self):
        ctx = EffectContext()
        assert ctx.movement_rules == ()

    def test_to_dict_serializes_movement_rules(self):
        rule = _make_rule()
        ctx = EffectContext(movement_rules=(rule,))
        d = ctx.to_dict()
        assert "movement_rules" in d
        assert d["movement_rules"] == [rule_to_dsl(rule)]

    def test_to_dict_empty_movement_rules(self):
        ctx = EffectContext()
        d = ctx.to_dict()
        assert d["movement_rules"] == []

    def test_merge_movement_rules_base_and_override(self):
        base_rule = _make_rule(support=1)
        override_rule = _make_rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 2),),
            support=3,
        )
        base = EffectContext(movement_rules=(base_rule,))
        engine = EffectContext(movement_rules=(override_rule,))
        merged = merge_effect_context(base, engine)
        assert merged.movement_rules == (base_rule, override_rule)

    def test_merge_movement_rules_dedupes(self):
        rule = _make_rule()
        base = EffectContext(movement_rules=(rule,))
        engine = EffectContext(movement_rules=(rule,))
        merged = merge_effect_context(base, engine)
        assert merged.movement_rules == (rule,)

    def test_merge_movement_rules_base_first(self):
        rule_a = _make_rule(support=1)
        rule_b = _make_rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 2),),
            support=2,
        )
        base = EffectContext(movement_rules=(rule_a,))
        engine = EffectContext(movement_rules=(rule_b,))
        merged = merge_effect_context(base, engine)
        assert merged.movement_rules == (rule_a, rule_b)


@pytest.mark.unit
class TestRuleKindField:
    """Rule.kind is a stored optional field with backward-compatible default."""

    def test_default_kind_delta(self):
        """Effects without terminal dim → kind defaults to 'delta'."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 0, "delta", 1),),
            support=0,
        )
        assert rule.kind == "delta"

    def test_default_kind_terminal(self):
        """Effects with terminal dim → kind defaults to 'terminal'."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("terminal", 0, "set", "win"),),
            support=0,
        )
        assert rule.kind == "terminal"

    def test_explicit_kind_overrides_default(self):
        """Explicit kind= is stored and not overridden by __post_init__."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 0, "delta", 1),),
            support=0,
            kind="movement",
        )
        assert rule.kind == "movement"


@pytest.mark.unit
class TestRevertEffect:
    """Effect.op='revert' and Rule.apply revert dispatch."""

    def test_revert_effect_construction(self):
        """Effect with op='revert' constructs without error."""
        eff = Effect("pos", 0, "revert", "before")
        assert eff.op == "revert"

    def test_revert_noop_without_state_before(self):
        """Revert effect with state_before=None → state unchanged (no-op)."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=0,
        )
        state = SceneState(relevant=((0, ("pos", (3, 4))),))
        result = rule.apply(state, 1)
        assert result.pos(0) == (3, 4)

    def test_revert_with_state_before(self):
        """Revert effect restores position from state_before."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=0,
        )
        state_before = SceneState(relevant=((0, ("pos", (1, 1))),))
        state_after = SceneState(relevant=((0, ("pos", (3, 4))),))
        result = rule.apply(state_after, 1, state_before=state_before)
        assert result.pos(0) == (1, 1)


@pytest.mark.unit
class TestOverlapsGuard:
    def test_parse_overlaps_guard_top_level(self):
        clauses = parse_guard_clauses(
            {"overlaps": {"entity_a": 0, "entity_b": 5}}
        )
        assert len(clauses) == 1
        c = clauses[0]
        assert c["has_overlaps"] is True
        assert c["overlaps_entity_ids"] == (0, 5)
        assert c["has_action"] is False
        assert c["has_pos"] is False

    def test_parse_overlaps_guard_inside_all(self):
        clauses = parse_guard_clauses(
            {"all": [
                {"action": 1},
                {"overlaps": {"entity_a": 2, "entity_b": 3}},
            ]}
        )
        assert len(clauses) == 2
        overlaps_clause = [c for c in clauses if c["has_overlaps"]][0]
        assert overlaps_clause["overlaps_entity_ids"] == (2, 3)

    def test_evaluate_overlaps_true(self):
        clause_guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        entity_cells = {
            0: frozenset({(1, 1), (2, 2)}),
            5: frozenset({(1, 1), (3, 3)}),
        }
        assert evaluate_guard(clause_guard, state, 0, entity_cells=entity_cells) is True

    def test_evaluate_overlaps_false(self):
        clause_guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        entity_cells = {
            0: frozenset({(1, 1)}),
            5: frozenset({(2, 2)}),
        }
        assert evaluate_guard(clause_guard, state, 0, entity_cells=entity_cells) is False

    def test_evaluate_overlaps_requires_entity_cells(self):
        clause_guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        with pytest.raises(ValueError, match="overlaps guard requires entity_cells"):
            evaluate_guard(clause_guard, state, 0)

    def test_evaluate_overlaps_missing_entity(self):
        clause_guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        entity_cells = {0: frozenset({(1, 1)})}
        assert evaluate_guard(clause_guard, state, 0, entity_cells=entity_cells) is False

    def test_parse_action_guard_still_works(self):
        clauses = parse_guard_clauses({"action": 3})
        assert len(clauses) == 1
        assert clauses[0]["has_action"] is True
        assert clauses[0]["action"] == 3
        assert clauses[0]["has_overlaps"] is False



def _make_rule(**overrides):
    defaults = dict(
        guard_spec={"action": 1},
        effects=(Effect("size", 3, "delta", 1),),
        support=1,
    )
    defaults.update(overrides)
    return Rule(**defaults)


def _positional_rule(action: int = 1, entity_id: int = 0, pos: tuple[int, int] = (5, 5), **overrides):
    """Build a Rule whose guard includes a positional check (is_positional_guard=True)."""
    defaults = dict(
        guard_spec={
            "all": [
                {"action": action},
                {"dim": "pos", "of": entity_id, "eq": list(pos)},
            ]
        },
        effects=(Effect("size", entity_id, "delta", 1),),
        support=1,
    )
    defaults.update(overrides)
    return Rule(**defaults)


@pytest.mark.unit
class TestEffectContextNewFields:
    """Test collision_rules and available_actions fields on EffectContext."""

    def test_collision_rules_default_empty(self):
        ctx = EffectContext()
        assert ctx.collision_rules == ()

    def test_available_actions_default_empty(self):
        ctx = EffectContext()
        assert ctx.available_actions == ()

    def test_collision_rules_construction(self):
        rule = _positional_rule()
        ctx = EffectContext(collision_rules=(rule,))
        assert ctx.collision_rules == (rule,)

    def test_available_actions_construction(self):
        ctx = EffectContext(available_actions=(1, 2, 3))
        assert ctx.available_actions == (1, 2, 3)

    def test_to_dict_includes_collision_rules(self):
        rule = _positional_rule()
        ctx = EffectContext(collision_rules=(rule,))
        d = ctx.to_dict()
        assert "collision_rules" in d
        assert d["collision_rules"] == [rule_to_dsl(rule)]

    def test_to_dict_includes_available_actions(self):
        ctx = EffectContext(available_actions=(1, 3, 5))
        d = ctx.to_dict()
        assert d["available_actions"] == [1, 3, 5]

    def test_to_dict_empty_collision_rules(self):
        ctx = EffectContext()
        d = ctx.to_dict()
        assert d["collision_rules"] == []
        assert d["available_actions"] == []


@pytest.mark.unit
class TestMergeCollisionRulesAndActions:
    """Test merge_effect_context for collision_rules and available_actions."""

    def test_merge_collision_rules_base_and_engine(self):
        base_rule = _positional_rule(action=1, pos=(5, 5))
        engine_rule = _positional_rule(action=2, pos=(6, 6))
        base = EffectContext(collision_rules=(base_rule,))
        engine = EffectContext(collision_rules=(engine_rule,))
        merged = merge_effect_context(base, engine)
        assert merged.collision_rules == (base_rule, engine_rule)

    def test_merge_collision_rules_dedupes(self):
        rule = _positional_rule()
        base = EffectContext(collision_rules=(rule,))
        engine = EffectContext(collision_rules=(rule,))
        merged = merge_effect_context(base, engine)
        assert merged.collision_rules == (rule,)

    def test_merge_collision_rules_base_first(self):
        rule_a = _positional_rule(action=1, pos=(5, 5))
        rule_b = _positional_rule(action=2, pos=(6, 6))
        base = EffectContext(collision_rules=(rule_a,))
        engine = EffectContext(collision_rules=(rule_b,))
        merged = merge_effect_context(base, engine)
        assert merged.collision_rules == (rule_a, rule_b)

    def test_merge_available_actions_union_sorted(self):
        base = EffectContext(available_actions=(1, 3, 5))
        engine = EffectContext(available_actions=(2, 3, 6))
        merged = merge_effect_context(base, engine)
        assert merged.available_actions == (1, 2, 3, 5, 6)

    def test_merge_available_actions_dedupes(self):
        base = EffectContext(available_actions=(1, 2, 3))
        engine = EffectContext(available_actions=(1, 2, 3))
        merged = merge_effect_context(base, engine)
        assert merged.available_actions == (1, 2, 3)

    def test_merge_available_actions_empty_both(self):
        base = EffectContext()
        engine = EffectContext()
        merged = merge_effect_context(base, engine)
        assert merged.available_actions == ()

    def test_merge_collision_rules_empty_both(self):
        base = EffectContext()
        engine = EffectContext()
        merged = merge_effect_context(base, engine)
        assert merged.collision_rules == ()


@pytest.mark.unit
class TestPrediction:
    """Tests for the Prediction dataclass and collision rule bucket."""

    def test_prediction_unknown_true(self):
        """Prediction(state, unknown=True) construction."""
        state = SceneState(relevant=((0, ("pos", (3, 4))),))
        pred = Prediction(state, unknown=True)
        assert pred.state == state
        assert pred.unknown is True

    def test_prediction_unknown_false(self):
        """Prediction(state, unknown=False) construction."""
        state = SceneState(relevant=((0, ("pos", (3, 4))),))
        pred = Prediction(state, unknown=False)
        assert pred.state == state
        assert pred.unknown is False

    def test_prediction_default_unknown_false(self):
        """Prediction default unknown is False."""
        state = SceneState(relevant=((0, ("pos", (3, 4))),))
        pred = Prediction(state)
        assert pred.unknown is False

    def test_collision_rule_revert_position(self):
        """Collision rule with op='revert' reverts entity's position to state_before."""
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=1,
            kind="movement",
        )
        collision_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=1,
        )
        ctx = EffectContext(
            movement_rules=(movement_rule,),
            collision_rules=(collision_rule,),
            available_actions=(1,),
        )
        # Entity 0 starts at (5, 5), movement rule moves to (5, 6)
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 1, ctx)
        assert not result.unknown
        # Collision revert should restore to (5, 5)
        assert result.state.pos(0) == (5, 5)

    def test_predict_returns_unknown_when_no_delta_fallback(self):
        """predict() returns Prediction(state, unknown=True) when no movement
        rule matches the action (no delta fallback)."""
        # Only action 1 has a movement rule; action 2 has none
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(movement_rule,), available_actions=(1,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 2, ctx)
        assert result.unknown is True
        assert result.state == start

    def test_predict_returns_unknown_when_no_delta_movement_rules_path(self):
        """predict() returns Prediction(state, unknown=True) when no movement
        rule matches and there are no fallback deltas."""
        # A movement rule for action 2 (not action 1)
        movement_rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "set", (9, 9)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(movement_rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        # Action 1: no movement rule matches
        result = predict(start, 1, ctx)
        assert result.unknown is True
        assert result.state == start

    def test_predict_returns_known_when_movement_rule_fires(self):
        """predict() returns Prediction(nxt, unknown=False) when a movement
        rule fires."""
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (9, 9)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(movement_rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 1, ctx)
        assert result.unknown is False
        assert result.state.pos(0) == (9, 9)

    def test_predict_collision_rule_with_overlaps_guard(self):
        """Collision rule with overlaps guard evaluated against post-movement state."""
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=1,
            kind="movement",
        )
        collision_rule = Rule(
            guard_spec={"overlaps": {"entity_a": 0, "entity_b": 5}},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=1,
        )
        ctx = EffectContext(
            movement_rules=(movement_rule,),
            collision_rules=(collision_rule,),
            available_actions=(1,),
        )
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        # Entity 0 cells include (5,6) (overlap with entity 5)
        cells = {
            0: frozenset({(5, 5), (5, 6)}),
            5: frozenset({(5, 6)}),
        }
        result = predict(start, 1, ctx, entity_cells=cells)
        assert not result.unknown
        # Revert should restore entity 0 to (5,5)
        assert result.state.pos(0) == (5, 5)


