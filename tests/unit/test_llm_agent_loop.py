"""Unit tests for agents/templates/llm_curiosity_agent.py — LLM agent loop state machine."""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction, GameState

from planning.probe import ProbeGoal

if TYPE_CHECKING:
    from agents.templates.llm_curiosity_agent import LlmCuriosity

# ── Fake frame data ──────────────────────────────────────────────────────────────────────


class _FakeFrameData:
    """Minimal stand-in for FrameData that the agent actually reads."""

    def __init__(
        self,
        state: GameState = GameState.NOT_FINISHED,
        available_actions: list[int] | None = None,
        frame=None,
    ) -> None:
        self.state = state
        self.available_actions = available_actions
        self.frame = frame


# ── Agent factory ───────────────────────────────────────────────────────────────────────


def _make_agent() -> LlmCuriosity:
    """Create an LlmCuriosity agent with mocked-out infrastructure."""
    from agents.templates.llm_curiosity_agent import LlmCuriosity

    with patch("agents.templates.llm_curiosity_agent.LLMClient"), patch(
        "agents.templates.llm_curiosity_agent.PerceptionSession"
    ), patch("agents.templates.llm_curiosity_agent.ExplorationPolicy") as MockPolicy, patch(
        "agents.templates.llm_curiosity_agent.ExplorationConfig"
    ):
        # ExplorationPolicy mock
        mock_policy = MagicMock()
        mock_policy.action_space = [1, 2, 3, 4]
        mock_policy.context = None
        mock_policy.status.return_value = MagicMock(
            phase="init",
            controllable_id=None,
            target=None,
            plan_len=0,
            n_observed=0,
            n_visited=0,
            diverged=False,
        )
        mock_policy.decide.return_value = 1  # simple action ID
        MockPolicy.return_value = mock_policy

        agent = LlmCuriosity(
            card_id="test-card",
            game_id="test-game",
            agent_name="test-agent",
            ROOT_URL="https://example.com",
            record=False,
            arc_env=MagicMock(),
        )
        # Replace llm_call with a mock so no network calls happen
        agent.llm_call = MagicMock(return_value='{"target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}, "max_steps": 50, "reason": "probe entity 17"}')
    return agent


def _make_scene_with_controllable(ctrl_id: int = 0) -> MagicMock:
    """Return a MagicMock scene that reports a controllable entity."""
    scene = MagicMock()
    scene.controllable_id.return_value = ctrl_id
    scene.controllable_pos.return_value = (5, 5)
    return scene


# ===========================================================================
# TestLLmCuriosityAgentLoop
# ===========================================================================


@pytest.mark.unit
class TestLlmCuriosityAgentLoop:
    """Tests for LlmCuriosity choose_action state machine transitions."""

    # -----------------------------------------------------------------------
    # 1. Cold start — no LLM calls
    # -----------------------------------------------------------------------

    def test_cold_start_no_llm_call(self) -> None:
        """When controllable_id is None and policy.context is None, agent stays in 'random' and never calls LLM."""
        agent = _make_agent()
        # Force llm_call to raise if ever called
        agent.llm_call = MagicMock(side_effect=AssertionError("LLM should not be called"))

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])
        # With no scene controllable and no context, phase stays "random"
        agent._phase = "random"
        # Mock scene with no controllable
        mock_scene = MagicMock()
        mock_scene.controllable_id.return_value = None
        agent._scene = mock_scene

        action = agent.choose_action([frame], frame)

        # LLM was never called
        agent.llm_call.assert_not_called()
        # Phase stayed random
        assert agent._phase == "random"
        # Got a valid action (from policy.decide fallback)
        assert isinstance(action, GameAction)

    # -----------------------------------------------------------------------
    # 2. Phase transition — random → llm_directed
    # -----------------------------------------------------------------------

    def test_phase_transition_to_llm_directed(self) -> None:
        """When controllable_id and context become available, phase transitions to 'llm_directed' and LLM is called."""
        agent = _make_agent()
        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        # Set up scene with controllable entity
        scene = _make_scene_with_controllable()
        agent._scene = scene

        # Set policy context so phase gate passes
        agent.policy.context = MagicMock()  # non-None
        agent.policy.status.return_value.diverged = False

        # First call should transition to llm_directed and call LLM
        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner, patch(
            "agents.templates.llm_curiosity_agent.execute_probe"
        ) as mock_execute:
            mock_call_planner.return_value = ProbeGoal(
                target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
                max_steps=50,
                reason="probe entity 17",
            )
            mock_execute.return_value = ([3, 1, 1, 4], [])

            agent.choose_action([frame], frame)

        assert agent._phase == "llm_directed"
        mock_call_planner.assert_called_once()

    # -----------------------------------------------------------------------
    # 3. LLM parse failure → fallback
    # -----------------------------------------------------------------------

    def test_llm_parse_failure_fallback(self) -> None:
        """When LLM returns unparseable text, agent falls back to random.choice(actions) without crashing."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner:
            # call_planner returns None (parse failure)
            mock_call_planner.return_value = None

            action = agent.choose_action([frame], frame)

        # No exception propagated
        assert isinstance(action, GameAction)
        # Action must be one of the available actions (random.choice)
        assert action.value in [1, 2, 3, 4]

    # -----------------------------------------------------------------------
    # 4. Probe execution — plan found
    # -----------------------------------------------------------------------

    def test_probe_execution_plan_found(self) -> None:
        """When LLM returns a valid goal and execute_probe returns a plan, first action is popped and plan is stored."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner, patch(
            "agents.templates.llm_curiosity_agent.execute_probe", return_value=([3, 1, 1, 4], [])
        ):
            mock_call_planner.return_value = goal

            action = agent.choose_action([frame], frame)

        # First action should be 3 (first element popped from plan)
        assert action.value == 3
        # After popping the first element, remaining plan is [1, 1, 4]
        assert agent._probe_plan == [1, 1, 4]
        # Goal is stored
        assert agent._current_goal == goal

    # -----------------------------------------------------------------------
    # 5. Probe exhaustion → re-call LLM
    # -----------------------------------------------------------------------

    def test_probe_exhaustion_triggers_failure_context(self) -> None:
        """When the last probe plan action is consumed, failure_context is set to 'probe_exhausted' and passed to the next LLM call."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = [1]  # one action left
        agent._current_goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner:
            mock_call_planner.return_value = None  # no new goal this time

            action = agent.choose_action([frame], frame)

        # After exhaustion + LLM returning None, action is random.choice(actions)
        assert isinstance(action, GameAction)
        assert action.value in [1, 2, 3, 4]
        # Probe plan is now empty / None
        assert agent._probe_plan is None
        # The failure context "probe_exhausted" was passed to call_planner as failure_context
        call_args = mock_call_planner.call_args
        assert call_args is not None
        # Check keyword argument
        kw_failure = call_args.kwargs.get("failure_context")
        assert kw_failure is not None
        assert kw_failure["type"] == "probe_exhausted"
        # After the LLM call completes (even with None goal), _failure_context is cleared
        assert agent._failure_context is None

    # -----------------------------------------------------------------------
    # 6. Divergence detection → clear plan
    # -----------------------------------------------------------------------

    def test_divergence_clears_plan(self) -> None:
        """When policy.status().diverged is True and no active probe plan, failure_context is 'rule_violation' and probe_plan is cleared."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None  # no active plan — falls through to divergence check
        agent._current_goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = True  # ← divergence!

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner:
            mock_call_planner.return_value = None  # no new goal

            agent.choose_action([frame], frame)

        # Plan remains None
        assert agent._probe_plan is None
        # Divergence failure context was passed to call_planner
        kw_failure = mock_call_planner.call_args.kwargs.get("failure_context")
        assert kw_failure is not None
        assert kw_failure["type"] == "rule_violation"
        assert agent._current_goal is None

    # -----------------------------------------------------------------------
    # 7. RESET clears state
    # -----------------------------------------------------------------------

    def test_reset_clears_state(self) -> None:
        """When frame state is NOT_PLAYED, choose_action returns RESET and clears all internal state."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = [1, 2, 3]
        agent._current_goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )
        agent._failure_context = {"type": "probe_exhausted", "last_action": 1}

        frame = _FakeFrameData(state=GameState.NOT_PLAYED)

        action = agent.choose_action([frame], frame)

        assert action == GameAction.RESET
        assert agent._probe_plan is None
        assert agent._failure_context is None
        assert agent._current_goal is None

    # -----------------------------------------------------------------------
    # 8. LLM call exception → fallback + cooldown
    # -----------------------------------------------------------------------

    def test_llm_exception_fallback_with_cooldown(self) -> None:
        """When LLM call raises an exception, agent falls back to random.choice(actions) and sets _llm_cooldown to 3."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner:
            mock_call_planner.side_effect = RuntimeError("LLM network error")

            action = agent.choose_action([frame], frame)

        # No exception propagated
        assert isinstance(action, GameAction)
        # Cooldown set to 3
        assert agent._llm_cooldown == 3
        # Action must be one of the available actions (random.choice)
        assert action.value in [1, 2, 3, 4]

    # -----------------------------------------------------------------------
    # 9. Unreachable goal stores unknowns in failure_context
    # -----------------------------------------------------------------------

    def test_unreachable_goal_stores_unknowns(self) -> None:
        """When BFS returns None (unreachable), failure_context includes unknowns."""
        from planning.query import UnknownAction
        from effects.state import SceneState

        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )

        fake_unknowns = [
            UnknownAction(action=3, state=SceneState(relevant=((0, ("pos", (1, 2))),))),
        ]

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner, patch(
            "agents.templates.llm_curiosity_agent.execute_probe"
        ) as mock_execute:
            mock_call_planner.return_value = goal
            mock_execute.return_value = (None, fake_unknowns)

            action = agent.choose_action([frame], frame)

        assert isinstance(action, GameAction)
        assert action.value in [1, 2, 3, 4]
        # failure_context was cleared after LLM call but then set by unreachable branch
        assert agent._failure_context is not None
        assert agent._failure_context["type"] == "unreachable"
        assert "unknowns" in agent._failure_context
        assert len(agent._failure_context["unknowns"]) == 1
        assert agent._failure_context["unknowns"][0]["action"] == 3

    # -----------------------------------------------------------------------
    # 10. Goal already met with goal.action → execute that action
    # -----------------------------------------------------------------------

    def test_goal_already_met_with_action(self) -> None:
        """When plan is empty but goal.action is set, execute goal.action directly."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
            action=2,
        )

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner, patch(
            "agents.templates.llm_curiosity_agent.execute_probe"
        ) as mock_execute:
            mock_call_planner.return_value = goal
            # Empty plan means "already at target"
            mock_execute.return_value = ([], [])

            action = agent.choose_action([frame], frame)

        assert action.value == 2
        assert agent._current_goal == goal

    # -----------------------------------------------------------------------
    # 11. Goal already met without action → random choice
    # -----------------------------------------------------------------------

    def test_goal_already_met_without_action(self) -> None:
        """When plan is empty and no goal.action, fall back to random.choice(actions)."""
        agent = _make_agent()
        agent._phase = "llm_directed"
        agent._probe_plan = None
        agent._llm_cooldown = 0

        scene = _make_scene_with_controllable()
        agent._scene = scene
        agent.policy.context = MagicMock()
        agent.policy.status.return_value.diverged = False

        frame = _FakeFrameData(state=GameState.NOT_FINISHED, available_actions=[1, 2, 3, 4])

        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            max_steps=50,
            reason="probe entity 17",
        )

        with patch("agents.templates.llm_curiosity_agent.call_planner") as mock_call_planner, patch(
            "agents.templates.llm_curiosity_agent.execute_probe"
        ) as mock_execute:
            mock_call_planner.return_value = goal
            mock_execute.return_value = ([], [])

            action = agent.choose_action([frame], frame)

        assert isinstance(action, GameAction)
        assert action.value in [1, 2, 3, 4]