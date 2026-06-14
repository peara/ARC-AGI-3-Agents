"""Effects layer: rules, learning, and predict pipeline (slice 2)."""

from __future__ import annotations

import pytest

from effects import (
    CounterRule,
    EffectContext,
    SceneState,
    TerminalRule,
    learn_effect_context,
    load_recording_meta,
    predict,
)
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
        game_over = [r for r in ctx.terminal_rules if r.terminal == TERMINAL_GAME_OVER]
        assert game_over, "expected GAME_OVER terminal rule on g50t"
        assert game_over[0].guard_key == ((10, 16), 2)

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
            if isinstance(r, CounterRule) and r.delta_size == 1
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
        rule = TerminalRule(
            entity_id=0,
            guard_key=((5, 5), 1),
            terminal=TERMINAL_GAME_OVER,
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
        rule = CounterRule(entity_id=3, action=2, delta_size=1, support=3)
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
        dead = TerminalRule(
            entity_id=0,
            guard_key=((0, 0), 1),
            terminal=TERMINAL_GAME_OVER,
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
