"""Effects slice 3: residual diff and rule engine lifecycle."""

from __future__ import annotations

import pytest

from effects import (
    Effect,
    EffectContext,
    ResidualEntry,
    Rule,
    SceneState,
    compute_residual,
    confirm_rules,
    diff_effect_context,
    engine_step,
    learn_effect_context,
    load_recording_meta,
    predict,
    propose_rules,
    prune_rules,
    should_engine_step,
)
from effects.engine import _bump_support, _iter_managed_rules, _promote_rules
from effects.engine_log import _index_rules, format_rule
from perception.session import PerceptionSession
from planning.adapters import snapshot_from_scene
from planning.search import PlanSpec
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
class TestComputeResidual:
    def test_size_mismatch(self):
        predicted = SceneState(relevant=((17, ("size", 10)),))
        observed = SceneState(relevant=((17, ("size", 8)),))
        residual = compute_residual(
            predicted,
            observed,
            entity_ids=(17,),
            dims=("size",),
        )
        assert len(residual) == 1
        assert residual[0].entity_id == 17
        assert residual[0].dim == "size"
        assert residual[0].predicted == 10
        assert residual[0].observed == 8


@pytest.mark.unit
class TestRuleEngineSynthetic:
    def _ctx(self) -> EffectContext:
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=1,
            kind="movement",
        )
        return EffectContext(movement_rules=(movement_rule,), available_actions=(1,), confirm_threshold=2)

    def test_confirm_promotes_counter_rule(self):
        ctx = self._ctx()
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )
        predicted = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        residual = compute_residual(
            predicted, observed, entity_ids=(17,), dims=("size",)
        )
        ctx = propose_rules(ctx, before, 1, residual)
        assert len(ctx.proposed_rules) == 1
        assert ctx.proposed_rules[0].effects[0].value == -2

        ctx = confirm_rules(ctx, before, 1, observed)
        assert ctx.proposed_rules[0].support == 1
        assert not ctx.relational_rules

        ctx = confirm_rules(ctx, before, 1, observed)
        assert not ctx.proposed_rules
        assert len(ctx.relational_rules) == 1
        assert ctx.relational_rules[0].effects[0].value == -2

        nxt = predict(before, 1, ctx)
        assert not nxt.unknown
        assert nxt.state.get(17, "size") == 8

    def test_prune_removes_mispredicting_rule(self):
        bad = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", +2),),
            support=3,
        )
        movement_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(movement_rule,), available_actions=(1,), relational_rules=(bad,))
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )
        predicted = predict(before, 1, ctx)
        assert not predicted.unknown
        residual = compute_residual(
            predicted.state, observed, entity_ids=(17,), dims=("size",)
        )
        ctx = prune_rules(ctx, before, 1, observed, residual)
        assert not ctx.relational_rules


@pytest.mark.unit
class TestLs20CounterPropose:
    def test_entity_17_decrease_proposed_with_size_in_spec(self):
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

        spec = PlanSpec(entities=[ctrl, 17], dims=("pos", "size"), goal=lambda s: False)
        found = False
        for fidx in range(1, len(session.action_ids)):
            before = snapshot_from_scene(scene, spec, frame_idx=fidx - 1)
            after = snapshot_from_scene(scene, spec, frame_idx=fidx)
            if before is None or after is None:
                continue
            action = int(session.action_ids[fidx])
            predicted = predict(before, action, ctx)
            if predicted.unknown:
                continue
            updated = engine_step(
                ctx,
                before,
                action,
                after,
                entity_ids=(17,),
                dims=("size",),
            )
            for rule in (*updated.proposed_rules, *updated.relational_rules):
                if rule.kind == "delta" and rule.effects[0].of == 17:
                    if rule.effects[0].value == -2:
                        found = True
                        break
            if found:
                break
        assert found, "expected -2 counter rule for ls20 entity #17"


@pytest.mark.unit
class TestG50tAbstain:
    @pytest.fixture
    def g50t_session(self):
        if not G50T_PATH.is_file():
            pytest.skip("g50t recording missing")
        session, _ = PerceptionSession.from_recording(G50T_PATH)
        return session

    def test_action5_abstains_when_not_confirmed(self, g50t_session):
        scene = g50t_session.snapshot()
        ctrl = scene.controllable_id()
        assert ctrl is not None
        ctx = learn_effect_context(
            g50t_session.registry,
            scene.catalog,
            list(g50t_session.action_ids),
            load_recording_meta(G50T_PATH),
            ctrl,
            non_markovian=True,
        )
        assert ctx is not None
        assert ctx.non_markovian
        assert scene.determinism_violations

        uncovered = SceneState(relevant=((ctrl, ("pos", (999, 999))),))
        assert predict(uncovered, 5, ctx).unknown


@pytest.mark.unit
class TestProposeRulesLlmProposals:
    def _ctx(self) -> EffectContext:
        return EffectContext(confirm_threshold=2)

    def test_llm_proposals_added_with_support_zero(self):
        from dataclasses import replace as dc_replace

        ctx = self._ctx()
        before = SceneState(relevant=((17, ("size", 10)),))
        residual = (ResidualEntry(17, "size", 10, 8),)
        llm_rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 17, "delta", -1),),
            support=5,
        )
        result = propose_rules(ctx, before, 1, residual, llm_proposals=(llm_rule,))
        llm_in_result = [r for r in result.proposed_rules if r.key() == llm_rule.key()]
        assert len(llm_in_result) == 1
        assert llm_in_result[0].support == 0

    def test_llm_proposals_dedup_against_existing_proposed(self):
        from dataclasses import replace as dc_replace

        existing = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=0,
        )
        ctx = dc_replace(self._ctx(), proposed_rules=(existing,))
        before = SceneState(relevant=((17, ("size", 10)),))
        residual = (ResidualEntry(17, "size", 10, 8),)
        result = propose_rules(ctx, before, 1, residual, llm_proposals=(existing,))
        assert sum(1 for r in result.proposed_rules if r.key() == existing.key()) == 1

    def test_llm_proposals_dedup_against_relational(self):
        from dataclasses import replace as dc_replace

        relational = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 17, "delta", -1),),
            support=3,
        )
        ctx = dc_replace(self._ctx(), relational_rules=(relational,))
        before = SceneState(relevant=((17, ("size", 10)),))
        residual = (ResidualEntry(17, "size", 10, 8),)
        result = propose_rules(ctx, before, 1, residual, llm_proposals=(relational,))
        assert not any(r.key() == relational.key() for r in result.proposed_rules)

    def test_llm_proposals_merges_with_residual_rules(self):
        ctx = self._ctx()
        before = SceneState(relevant=((17, ("size", 10)),))
        residual = (ResidualEntry(17, "size", 10, 8),)
        llm_rule = Rule(
            guard_spec={"action": 99},
            effects=(Effect("size", 17, "delta", -3),),
            support=5,
        )
        result = propose_rules(ctx, before, 1, residual, llm_proposals=(llm_rule,))
        residual_rule_key = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=0,
        ).key()
        keys = [r.key() for r in result.proposed_rules]
        assert residual_rule_key in keys
        assert llm_rule.key() in keys


@pytest.mark.unit
class TestEngineLog:
    def test_diff_propose_confirm_promote(self):
        ctx = EffectContext(confirm_threshold=2)
        proposed = propose_rules(
            ctx,
            SceneState(relevant=((17, ("size", 10)),)),
            1,
            (ResidualEntry(17, "size", 10, 8),),
        )
        lines = diff_effect_context(ctx, proposed)
        assert any(line.startswith("+ proposed:") for line in lines)

        confirmed = confirm_rules(
            proposed,
            SceneState(relevant=((17, ("size", 10)),)),
            1,
            SceneState(relevant=((17, ("size", 8)),)),
        )
        lines = diff_effect_context(proposed, confirmed)
        assert any("support 0→1" in line for line in lines)

        promoted = confirm_rules(
            confirmed,
            SceneState(relevant=((17, ("size", 10)),)),
            1,
            SceneState(relevant=((17, ("size", 8)),)),
        )
        lines = diff_effect_context(confirmed, promoted)
        assert any("proposed→relational" in line for line in lines)


@pytest.mark.unit
class TestEffectsEngineManifest:
    @pytest.fixture(params=load_effects_expectations(), ids=lambda e: e.recording.name)
    def expect(self, request):
        if not request.param.recording.path.is_file():
            pytest.skip("recording missing")
        return request.param

    def test_abstain_expectations(self, expect):
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
        if not expect.expect_abstain_non_markovian:
            return

        scene = session.snapshot()
        assert ctx.non_markovian
        assert scene.determinism_violations
        uncovered = SceneState(relevant=((ctrl, ("pos", (999, 999))),))
        assert predict(uncovered, 5, ctx).unknown
        assert not should_engine_step(ctx, uncovered, 5)


@pytest.mark.unit
class TestMovementRulePromotionRouting:
    def test_iter_managed_rules_returns_movement_group(self):
        """_iter_managed_rules returns four groups: terminals, counters, movement, collision."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(mv_rule,))
        terminals, counters, movement, collision = _iter_managed_rules(ctx)
        assert len(terminals) == 0
        assert len(counters) == 0
        assert len(movement) == 1
        assert len(collision) == 0
        assert movement[0] == (mv_rule, "movement")

    def test_iter_managed_rules_proposed_movement_routed_to_movement(self):
        """Proposed rules with kind='movement' go to the movement group."""
        proposed_mv = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "delta", (0, 1)),),
            support=0,
            kind="movement",
        )
        ctx = EffectContext(proposed_rules=(proposed_mv,))
        terminals, counters, movement, collision = _iter_managed_rules(ctx)
        assert len(terminals) == 0
        assert len(counters) == 0
        assert len(movement) == 1
        assert len(collision) == 0
        assert movement[0] == (proposed_mv, "proposed")

    def test_promote_rules_routes_movement_to_movement_bucket(self):
        """_promote_rules routes kind='movement' proposed rules to movement_rules."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        ctx = EffectContext(proposed_rules=(mv_rule,), confirm_threshold=2)
        result = _promote_rules(ctx)
        assert len(result.movement_rules) == 1
        assert result.movement_rules[0].key() == mv_rule.key()
        assert len(result.proposed_rules) == 0

    def test_promote_rules_dedup_against_existing_movement(self):
        """_promote_rules dedup movement rules against existing movement_rules."""
        existing = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=3,
            kind="movement",
        )
        proposed = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        ctx = EffectContext(
            movement_rules=(existing,),
            proposed_rules=(proposed,),
            confirm_threshold=2,
        )
        result = _promote_rules(ctx)
        assert len(result.movement_rules) == 1
        assert result.movement_rules[0].support == 3

    def test_promote_rules_below_threshold_stays_proposed(self):
        """Movement proposed rules below threshold stay in proposed."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(proposed_rules=(mv_rule,), confirm_threshold=2)
        result = _promote_rules(ctx)
        assert len(result.movement_rules) == 0
        assert len(result.proposed_rules) == 1

    def test_bump_support_in_movement_bucket(self):
        """_bump_support bumps movement rules in movement_rules bucket."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=3,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(mv_rule,))
        result = _bump_support(ctx, mv_rule)
        assert len(result.movement_rules) == 1
        assert result.movement_rules[0].support == 4
        assert len(result.proposed_rules) == 0

    def test_bump_support_proposed_movement_rule(self):
        """_bump_support bumps movement rules in proposed when not in movement_rules."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=0,
            kind="movement",
        )
        ctx = EffectContext(proposed_rules=(mv_rule,))
        result = _bump_support(ctx, mv_rule)
        assert len(result.movement_rules) == 0
        assert len(result.proposed_rules) == 1
        assert result.proposed_rules[0].support == 1

    def test_confirm_rules_promotes_movement_via_lifecycle(self):
        """Full confirm_rules lifecycle promotes a movement rule from proposed to movement_rules."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=1,
            kind="movement",
        )
        ctx = EffectContext(proposed_rules=(mv_rule,), confirm_threshold=2)
        before = SceneState(relevant=((0, ("pos", (1, 1))), (17, ("size", 10))))
        observed = SceneState(relevant=((0, ("pos", (1, 1))), (17, ("size", 8))))
        result = confirm_rules(ctx, before, 1, observed)
        assert len(result.proposed_rules) == 0
        assert len(result.movement_rules) == 1
        assert result.movement_rules[0].support == 2

    def test_prune_rules_includes_movement(self):
        """prune_rules prunes mispredicting movement rules."""
        bad = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", +2),),
            support=3,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(bad,))
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )
        predicted = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        residual = compute_residual(
            predicted, observed, entity_ids=(17,), dims=("size",)
        )
        result = prune_rules(ctx, before, 1, observed, residual)
        assert len(result.movement_rules) == 0


@pytest.mark.unit
class TestEngineLogMovementKind:
    """engine_log handles kind='movement' correctly."""

    def test_format_rule_movement(self):
        """format_rule produces 'movement ...' label for movement kind."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=3,
            kind="movement",
        )
        result = format_rule(rule)
        assert result.startswith("movement ")
        assert "support=3" in result

    def test_format_rule_movement_with_positional_guard(self):
        """format_rule includes guard for movement rules with positional guard."""
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [3, 4]},
                ]
            },
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        result = format_rule(rule)
        assert result.startswith("movement ")
        assert "guard=" in result

    def test_index_rules_includes_movement(self):
        """_index_rules includes movement_rules bucket."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=3,
            kind="movement",
        )
        ctx = EffectContext(movement_rules=(mv_rule,))
        index = _index_rules(ctx)
        assert mv_rule.key() in index
        assert index[mv_rule.key()].bucket == "movement"

    def test_diff_shows_movement_promotion(self):
        """diff_effect_context shows promotion from proposed to movement."""
        mv_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (1, 0)),),
            support=2,
            kind="movement",
        )
        before_ctx = EffectContext(proposed_rules=(mv_rule,), confirm_threshold=2)
        after_ctx = EffectContext(movement_rules=(mv_rule,), confirm_threshold=2)
        lines = diff_effect_context(before_ctx, after_ctx)
        assert any("proposed→movement" in line for line in lines)


@pytest.mark.unit
class TestCollisionRuleRouting:
    """Collision rule routing mirrors movement rule routing."""

    def test_iter_managed_rules_returns_collision_group(self):
        """_iter_managed_rules returns collision rules in the fourth bucket."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=2,
            kind="collision",
        )
        ctx = EffectContext(collision_rules=(col_rule,))
        terminals, counters, movement, collision = _iter_managed_rules(ctx)
        assert len(terminals) == 0
        assert len(counters) == 0
        assert len(movement) == 0
        assert len(collision) == 1
        assert collision[0] == (col_rule, "collision")

    def test_iter_managed_rules_proposed_collision_routed_to_collision(self):
        """Proposed rules with kind='collision' go to the collision group."""
        proposed_col = Rule(
            guard_spec={"action": 2},
            effects=(Effect("pos", 0, "revert", ""),),
            support=0,
            kind="collision",
        )
        ctx = EffectContext(proposed_rules=(proposed_col,))
        terminals, counters, movement, collision = _iter_managed_rules(ctx)
        assert len(collision) == 1
        assert collision[0] == (proposed_col, "proposed")

    def test_promote_rules_routes_collision_to_collision_bucket(self):
        """_promote_rules routes kind='collision' proposed rules to collision_rules."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=2,
            kind="collision",
        )
        ctx = EffectContext(proposed_rules=(col_rule,), confirm_threshold=2)
        result = _promote_rules(ctx)
        assert len(result.collision_rules) == 1
        assert result.collision_rules[0].key() == col_rule.key()
        assert len(result.proposed_rules) == 0

    def test_promote_rules_dedup_against_existing_collision(self):
        """_promote_rules dedup collision rules against existing collision_rules."""
        existing = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=3,
            kind="collision",
        )
        proposed = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=2,
            kind="collision",
        )
        ctx = EffectContext(
            collision_rules=(existing,),
            proposed_rules=(proposed,),
            confirm_threshold=2,
        )
        result = _promote_rules(ctx)
        assert len(result.collision_rules) == 1
        assert result.collision_rules[0].support == 3

    def test_promote_rules_below_threshold_stays_proposed(self):
        """Collision proposed rules below threshold stay in proposed."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=1,
            kind="collision",
        )
        ctx = EffectContext(proposed_rules=(col_rule,), confirm_threshold=2)
        result = _promote_rules(ctx)
        assert len(result.collision_rules) == 0
        assert len(result.proposed_rules) == 1

    def test_bump_support_in_collision_bucket(self):
        """_bump_support bumps collision rules in collision_rules bucket."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=3,
            kind="collision",
        )
        ctx = EffectContext(collision_rules=(col_rule,))
        result = _bump_support(ctx, col_rule)
        assert len(result.collision_rules) == 1
        assert result.collision_rules[0].support == 4
        assert len(result.proposed_rules) == 0

    def test_bump_support_proposed_collision_rule(self):
        """_bump_support bumps collision rules in proposed when not in collision_rules."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=0,
            kind="collision",
        )
        ctx = EffectContext(proposed_rules=(col_rule,))
        result = _bump_support(ctx, col_rule)
        assert len(result.collision_rules) == 0
        assert len(result.proposed_rules) == 1
        assert result.proposed_rules[0].support == 1

    def test_confirm_rules_promotes_collision_via_lifecycle(self):
        """Full confirm_rules lifecycle promotes a collision rule from proposed to collision_rules."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=1,
            kind="collision",
        )
        ctx = EffectContext(proposed_rules=(col_rule,), confirm_threshold=2)
        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        observed = SceneState(relevant=((0, ("pos", (1, 1))),))
        result = confirm_rules(ctx, before, 1, observed)
        assert len(result.proposed_rules) == 0
        assert len(result.collision_rules) == 1
        assert result.collision_rules[0].support == 2

    def test_prune_rules_includes_collision(self):
        """prune_rules prunes mispredicting collision rules."""
        bad = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", +2),),
            support=3,
            kind="collision",
        )
        ctx = EffectContext(collision_rules=(bad,))
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )
        predicted = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        residual = compute_residual(
            predicted, observed, entity_ids=(17,), dims=("size",)
        )
        result = prune_rules(ctx, before, 1, observed, residual)
        assert len(result.collision_rules) == 0


@pytest.mark.unit
class TestEngineLogCollisionKind:
    """engine_log handles kind='collision' correctly."""

    def test_format_rule_collision(self):
        """format_rule produces 'collision ...' label for collision kind."""
        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=3,
            kind="collision",
        )
        result = format_rule(rule)
        assert result.startswith("collision ")
        assert "support=3" in result

    def test_format_rule_collision_with_positional_guard(self):
        """format_rule includes guard for collision rules with positional guard."""
        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [3, 4]},
                ]
            },
            effects=(Effect("pos", 0, "revert", ""),),
            support=2,
            kind="collision",
        )
        result = format_rule(rule)
        assert result.startswith("collision ")
        assert "guard=" in result

    def test_index_rules_includes_collision(self):
        """_index_rules includes collision_rules bucket."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=3,
            kind="collision",
        )
        ctx = EffectContext(collision_rules=(col_rule,))
        index = _index_rules(ctx)
        assert col_rule.key() in index
        assert index[col_rule.key()].bucket == "collision"

    def test_diff_shows_collision_promotion(self):
        """diff_effect_context shows promotion from proposed to collision."""
        col_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "revert", ""),),
            support=2,
            kind="collision",
        )
        before_ctx = EffectContext(proposed_rules=(col_rule,), confirm_threshold=2)
        after_ctx = EffectContext(collision_rules=(col_rule,), confirm_threshold=2)
        lines = diff_effect_context(before_ctx, after_ctx)
        assert any("proposed→collision" in line for line in lines)
