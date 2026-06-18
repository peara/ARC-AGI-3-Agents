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
from effects.kinematics import MovementModel
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
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 0)},
            known_transitions={((1, 1), 1): (1, 1)},
            known_blocks=frozenset(),
        )
        return EffectContext(movement=model, confirm_threshold=2)

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
        assert nxt is not None
        assert nxt.get(17, "size") == 8

    def test_prune_removes_mispredicting_rule(self):
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 0)},
            known_transitions={((1, 1), 1): (1, 1)},
            known_blocks=frozenset(),
        )
        bad = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", +2),),
            support=3,
        )
        ctx = EffectContext(movement=model, relational_rules=(bad,))
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
        assert predicted is not None
        residual = compute_residual(
            predicted, observed, entity_ids=(17,), dims=("size",)
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
            if predicted is None:
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
        assert predict(uncovered, 5, ctx) is None


@pytest.mark.unit
class TestProposeRulesLlmProposals:
    def _ctx(self) -> EffectContext:
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 0)},
            known_transitions={((1, 1), 1): (1, 1)},
            known_blocks=frozenset(),
        )
        return EffectContext(movement=model, confirm_threshold=2)

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
        model = MovementModel(
            entity_id=0,
            motion_by_action={1: (0, 0)},
            known_transitions={((1, 1), 1): (1, 1)},
            known_blocks=frozenset(),
        )
        ctx = EffectContext(movement=model, confirm_threshold=2)
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
        assert predict(uncovered, 5, ctx) is None
        assert not should_engine_step(ctx, uncovered, 5)
