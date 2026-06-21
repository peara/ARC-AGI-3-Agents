"""Effects layer: rules, learning, and predict pipeline (slice 2)."""

from __future__ import annotations

import pytest

from effects import (
    Effect,
    EffectContext,
    Rule,
    SceneState,
    learn_effect_context,
    load_recording_meta,
    merge_effect_context,
    predict,
)
from effects.dsl import dsl_to_rule, rule_to_dsl
from effects.guard_parse import evaluate_guard, parse_guard_clauses
from effects.kinematics import MovementModel
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
            non_markovian=True,
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
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 1)},
            known_transitions={},
            known_blocks=frozenset(),
        )
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
        ctx = EffectContext(movement=model, terminal_rules=(rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        nxt = predict(start, 1, ctx)
        assert nxt is not None
        assert nxt.terminal == TERMINAL_GAME_OVER
        assert nxt.pos(0) == (5, 6)

    def test_counter_rule_updates_size(self):
        model = MovementModel(
            entity_id=0,
            motion_by_action={2: (0, 0)},
            known_transitions={((1, 1), 2): (1, 1)},
            known_blocks=frozenset(),
        )
        rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 1),),
            support=3,
        )
        ctx = EffectContext(movement=model, relational_rules=(rule,))
        start = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (3, ("size", 10)),
            )
        )
        nxt = predict(start, 2, ctx)
        assert nxt is not None
        assert nxt.get(3, "size") == 11
        assert nxt.pos(0) == (1, 1)


@pytest.mark.unit
class TestBFSPrunesGameOver:
    def test_game_over_branch_not_expanded(self):
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 0), 2: (1, 0)},
            known_transitions={
                ((0, 0), 1): (0, 0),
                ((0, 0), 2): (1, 0),
            },
            known_blocks=frozenset(),
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
        ctx = EffectContext(movement=model, terminal_rules=(dead,))
        start = SceneState(relevant=((0, ("pos", (0, 0))),))
        plan = plan_bfs(
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
            non_markovian=expect.expect_non_markovian,
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
        assert ctx.non_markovian == expect.expect_non_markovian


def _make_model():
    return MovementModel(
        entity_id=0,
        motion_by_action={1: (0, 1)},
        known_transitions={},
        known_blocks=frozenset(),
    )


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
        model = _make_model()
        rule = _make_rule()
        ctx = EffectContext(movement=model, movement_rules=(rule,))
        assert ctx.movement_rules == (rule,)

    def test_construction_default_movement_rules(self):
        model = _make_model()
        ctx = EffectContext(movement=model)
        assert ctx.movement_rules == ()

    def test_to_dict_serializes_movement_rules(self):
        model = _make_model()
        rule = _make_rule()
        ctx = EffectContext(movement=model, movement_rules=(rule,))
        d = ctx.to_dict()
        assert "movement_rules" in d
        assert d["movement_rules"] == [rule_to_dsl(rule)]

    def test_to_dict_empty_movement_rules(self):
        model = _make_model()
        ctx = EffectContext(movement=model)
        d = ctx.to_dict()
        assert d["movement_rules"] == []

    def test_merge_movement_rules_base_and_override(self):
        model = _make_model()
        base_rule = _make_rule(support=1)
        override_rule = _make_rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 2),),
            support=3,
        )
        base = EffectContext(movement=model, movement_rules=(base_rule,))
        engine = EffectContext(movement=model, movement_rules=(override_rule,))
        merged = merge_effect_context(base, engine)
        assert merged.movement_rules == (base_rule, override_rule)

    def test_merge_movement_rules_dedupes(self):
        model = _make_model()
        rule = _make_rule()
        base = EffectContext(movement=model, movement_rules=(rule,))
        engine = EffectContext(movement=model, movement_rules=(rule,))
        merged = merge_effect_context(base, engine)
        assert merged.movement_rules == (rule,)

    def test_merge_movement_rules_base_first(self):
        model = _make_model()
        rule_a = _make_rule(support=1)
        rule_b = _make_rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 3, "delta", 2),),
            support=2,
        )
        base = EffectContext(movement=model, movement_rules=(rule_a,))
        engine = EffectContext(movement=model, movement_rules=(rule_b,))
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


@pytest.mark.unit
class TestDualPathPredict:
    """Dual-path predict: movement_rules path vs fallback to predict_move."""

    @staticmethod
    def _make_model(**overrides):
        defaults = dict(
            entity_id=0,
            motion_by_action={1: (0, 1)},
            known_transitions={},
            known_blocks=frozenset(),
        )
        defaults.update(overrides)
        return MovementModel(**defaults)

    def test_empty_movement_rules_identical_to_no_movement_rules(self):
        """predict with empty movement_rules tuple == predict without it."""
        model = self._make_model()
        terminal_rule = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [5, 5]},
                ]
            },
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=1,
        )
        ctx_no_mr = EffectContext(movement=model, terminal_rules=(terminal_rule,))
        ctx_empty_mr = EffectContext(
            movement=model, terminal_rules=(terminal_rule,), movement_rules=()
        )
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result_no_mr = predict(start, 1, ctx_no_mr)
        result_empty_mr = predict(start, 1, ctx_empty_mr)
        assert result_no_mr is not None
        assert result_empty_mr is not None
        assert result_no_mr.pos(0) == result_empty_mr.pos(0)
        assert result_no_mr.terminal == result_empty_mr.terminal

    def test_movement_rule_path_applies_movement_rule(self):
        """When movement_rules exist and guard matches, use rule instead of predict_move."""
        model = self._make_model()
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (9, 9)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement=model, movement_rules=(movement_rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 1, ctx)
        assert result is not None
        assert result.pos(0) == (9, 9)

    def test_movement_rule_no_match_falls_back_to_predict_move(self):
        """When movement_rules exist but no guard matches, fall back to predict_move."""
        model = self._make_model()
        movement_rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "set", (9, 9)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement=model, movement_rules=(movement_rule,))
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 1, ctx)
        assert result is not None
        assert result.pos(0) == (5, 6)

    def test_entity_cells_passthrough_to_rule_apply(self):
        """entity_cells kwarg is forwarded to Rule.apply via state_before/entity_cells.

        Uses a movement rule (so we stay in the movement_rules path) and a
        revert effect to verify state_before is passed correctly.  The revert
        should restore entity 0's position to state_before=(5,5), undoing the
        movement rule's set to (10,10).
        """
        model = self._make_model()
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (10, 10)),),
            support=1,
            kind="movement",
        )
        revert_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=1,
        )
        ctx = EffectContext(
            movement=model,
            movement_rules=(movement_rule,),
            relational_rules=(revert_rule,),
        )
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        cells = {0: frozenset({(5, 5)}), 3: frozenset({(1, 1)})}
        result = predict(start, 1, ctx, entity_cells=cells)
        assert result is not None
        assert result.pos(0) == (5, 5)

    def test_backward_compat_predict_without_entity_cells(self):
        """predict(state, action, ctx) still works without entity_cells kwarg."""
        model = self._make_model()
        ctx = EffectContext(movement=model)
        start = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(start, 1, ctx)
        assert result is not None
        assert result.pos(0) == (5, 6)

    def test_rule_apply_backward_compat_without_state_before(self):
        """Rule.apply(state_after, action) still works without kwargs."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (7, 7)),),
            support=1,
        )
        state = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = rule.apply(state, 1)
        assert result.pos(0) == (7, 7)

    def test_movement_rules_then_terminal_then_relational(self):
        """Movement rules, then terminal, then relational all apply in order."""
        model = self._make_model()
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (10, 10)),),
            support=1,
            kind="movement",
        )
        terminal_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=1,
        )
        counter_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 3, "delta", 5),),
            support=1,
        )
        ctx = EffectContext(
            movement=model,
            movement_rules=(movement_rule,),
            terminal_rules=(terminal_rule,),
            relational_rules=(counter_rule,),
        )
        start = SceneState(
            relevant=(
                (0, ("pos", (5, 5))),
                (3, ("size", 10)),
            )
        )
        result = predict(start, 1, ctx)
        assert result is not None
        assert result.pos(0) == (10, 10)
        assert result.terminal == TERMINAL_GAME_OVER
        assert result.get(3, "size") == 15


@pytest.mark.unit
class TestIntegrationDualPath:
    """Integration tests for the full dual-path predict pipeline.

    Covers: movement rule application, fallback, revert effects, overlaps
    guards, engine promotion lifecycle, DSL round-trip, and predict
    ordering.
    """

    @staticmethod
    def _make_model(**overrides):
        defaults = dict(
            entity_id=0,
            motion_by_action={1: (0, 1), 2: (1, 0)},
            known_transitions={},
            known_blocks=frozenset(),
        )
        defaults.update(overrides)
        return MovementModel(**defaults)

    def test_movement_rule_moves_entity_correctly(self):
        """A movement rule with op='set' moves the entity to the given pos."""
        model = self._make_model()
        rule = Rule(
            guard_spec={"action": 0},
            effects=(Effect("pos", 0, "set", (4, 5)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement=model, movement_rules=(rule,))
        state = SceneState(relevant=((0, ("pos", (5, 5))),))
        result = predict(state, 0, ctx)
        assert result is not None
        assert result.pos(0) == (4, 5)

    def test_empty_movement_rules_identical_to_no_movement_rules(self):
        """predict() with movement_rules=() produces the same output as
        predict() with no movement_rules at all (fallback path)."""
        model = self._make_model()
        ctx_no_mr = EffectContext(movement=model)
        ctx_empty_mr = EffectContext(movement=model, movement_rules=())
        state = SceneState(relevant=((0, ("pos", (5, 5))),))
        result_no_mr = predict(state, 1, ctx_no_mr)
        result_empty_mr = predict(state, 1, ctx_empty_mr)
        assert result_no_mr is not None
        assert result_empty_mr is not None
        # Compare fingerprints since SceneState is a frozen dataclass
        assert result_no_mr.fingerprint() == result_empty_mr.fingerprint()
        assert result_no_mr.pos(0) == result_empty_mr.pos(0)

    def test_revert_effect_with_state_before(self):
        """Rule.apply with state_before restores entity position on revert."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=0,
        )
        state_before = SceneState(relevant=((0, ("pos", (1, 1))),))
        state_after = SceneState(relevant=((0, ("pos", (3, 4))),))
        result = rule.apply(state_after, 1, state_before=state_before)
        assert result.pos(0) == (1, 1)

    def test_revert_effect_without_state_before_is_noop(self):
        """Rule.apply with revert but no state_before leaves state unchanged."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", "before"),),
            support=0,
        )
        state_after = SceneState(relevant=((0, ("pos", (3, 4))),))
        result = rule.apply(state_after, 1)
        assert result.pos(0) == (3, 4)

    def test_overlaps_guard_detects_overlap(self):
        """evaluate_guard returns True when entity cells overlap."""
        guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        cells_overlapping = {
            0: frozenset({(1, 1)}),
            5: frozenset({(1, 1)}),
        }
        assert evaluate_guard(guard, state, 0, entity_cells=cells_overlapping) is True

    def test_overlaps_guard_no_overlap(self):
        guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        cells_disjoint = {
            0: frozenset({(1, 1)}),
            5: frozenset({(2, 2)}),
        }
        assert evaluate_guard(guard, state, 0, entity_cells=cells_disjoint) is False

    def test_overlaps_guard_without_entity_cells_raises(self):
        """evaluate_guard raises ValueError when overlaps guard lacks entity_cells."""
        guard = {"overlaps": {"entity_a": 0, "entity_b": 5}}
        state = SceneState(relevant=())
        with pytest.raises(ValueError, match="overlaps guard requires entity_cells"):
            evaluate_guard(guard, state, 0)

    def test_engine_promotion_lifecycle_movement(self):
        """propose → confirm → promote lifecycle for movement rules."""
        from dataclasses import replace

        from effects.engine import _promote_rules, confirm_rules
        from effects.residual import compute_residual

        model = self._make_model()
        proposed_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (1, 1)),),
            support=0,
            kind="movement",
        )
        ctx = EffectContext(
            movement=model,
            proposed_rules=(proposed_rule,),
            confirm_threshold=2,
        )
        assert len(ctx.movement_rules) == 0
        assert len(ctx.proposed_rules) == 1

        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        observed = SceneState(relevant=((0, ("pos", (1, 1))),))
        ctx = confirm_rules(ctx, before, 1, observed)
        assert len(ctx.proposed_rules) == 1
        assert ctx.proposed_rules[0].support == 1
        assert len(ctx.movement_rules) == 0

        ctx = confirm_rules(ctx, before, 1, observed)
        assert len(ctx.proposed_rules) == 0
        assert len(ctx.movement_rules) == 1
        assert ctx.movement_rules[0].support == 2
        assert ctx.movement_rules[0].kind == "movement"

    def test_dsl_round_trip_movement_with_revert(self):
        """rule_to_dsl → dsl_to_rule preserves kind='movement' and revert effects."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(
                Effect("pos", 0, "set", (5, 6)),
                Effect("pos", 0, "revert", "before"),
            ),
            support=3,
            kind="movement",
        )
        dsl = rule_to_dsl(rule)
        assert dsl["kind"] == "movement"
        assert len(dsl["effects"]) == 2
        assert dsl["effects"][1]["op"] == "revert"

        round_tripped = dsl_to_rule(dsl)
        assert round_tripped.kind == "movement"
        assert len(round_tripped.effects) == 2
        assert round_tripped.effects[0] == Effect("pos", 0, "set", (5, 6))
        assert round_tripped.effects[1] == Effect("pos", 0, "revert", "before")
        assert round_tripped == rule

    def test_predict_ordering_movement_terminal_relational(self):
        """predict applies movement rules, then terminal, then relational."""
        model = self._make_model()
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "set", (10, 10)),),
            support=1,
            kind="movement",
        )
        terminal_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("terminal", 0, "set", TERMINAL_GAME_OVER),),
            support=1,
        )
        relational_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 3, "delta", 5),),
            support=1,
        )
        ctx = EffectContext(
            movement=model,
            movement_rules=(movement_rule,),
            terminal_rules=(terminal_rule,),
            relational_rules=(relational_rule,),
        )
        start = SceneState(
            relevant=(
                (0, ("pos", (5, 5))),
                (3, ("size", 10)),
            )
        )
        result = predict(start, 1, ctx)
        assert result is not None
        assert result.pos(0) == (10, 10)  # movement rule overrides model
        assert result.terminal == TERMINAL_GAME_OVER
        assert result.get(3, "size") == 15  # 10 + 5 from delta
