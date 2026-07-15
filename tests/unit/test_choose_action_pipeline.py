"""Unit tests for agents/templates/llm_curiosity_agent.py — choose_action pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction, GameState

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

    with (
        patch("agents.templates.llm_curiosity_agent.LLMClient"),
        patch("agents.templates.llm_curiosity_agent.PerceptionSession"),
        patch("agents.templates.llm_curiosity_agent.ExplorationPolicy") as MockPolicy,
        patch("agents.templates.llm_curiosity_agent.ExplorationConfig"),
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
        agent.llm_call = MagicMock()
    return agent


# ===========================================================================
# TestChooseActionPipeline
# ===========================================================================


@pytest.mark.unit
class TestChooseActionPipeline:
    """Tests for the individual stages of LlmCuriosity.choose_action()."""

    def test_reset_clears_state(self) -> None:
        """Verify _reset() clears state and returns GameAction.RESET."""
        agent = _make_agent()
        agent._probe_plan = [1, 2]
        agent._failure_context = {"type": "error"}
        agent._current_goal = MagicMock()
        agent._last_action_id = 5
        agent._tried_fallback_unknowns.add((123, 1))

        action = agent._reset()

        assert action == GameAction.RESET
        assert agent._probe_plan is None
        assert agent._failure_context is None
        assert agent._current_goal is None
        assert agent._last_action_id == 0  # RESET_ACTION
        assert len(agent._tried_fallback_unknowns) == 0

    def test_perceive_returns_frame_context(self) -> None:
        """Verify _perceive() returns FrameContext when a new frame is ingested."""
        agent = _make_agent()
        agent.policy.context = MagicMock()  # Required for FrameContext return
        
        # Mock a frame that is not None and differs in ID from _last_observed_frame_id
        frame = _FakeFrameData(frame=MagicMock())
        
        with patch.object(agent.session, "ingest") as mock_ingest, \
             patch.object(agent._entity_builder, "update", return_value=(MagicMock(), MagicMock())) as mock_update:
            
            from planning.frame_context import FrameContext
            fc = agent._perceive(frame)
            
            assert isinstance(fc, FrameContext)
            mock_ingest.assert_called_once()
            mock_update.assert_called_once()
            assert agent._last_observed_frame_id == id(frame)

    def test_perceive_returns_none_for_duplicate_frame(self) -> None:
        """Verify _perceive() returns None when the frame is a duplicate."""
        agent = _make_agent()
        frame = _FakeFrameData(frame=MagicMock())
        agent._last_observed_frame_id = id(frame)
        
        fc = agent._perceive(frame)
        assert fc is None

    def test_verify_passes_frame_context_through(self) -> None:
        """Verify _verify() returns the FrameContext unchanged."""
        agent = _make_agent()
        from planning.frame_context import FrameContext
        fc = FrameContext(
            scene=MagicMock(),
            ctx=MagicMock(),
            residual=(),
            observed_transition=None,
            unknowns=(),
            diverged=False,
            spec=MagicMock(),
        )
        
        result = agent._verify(fc)
        assert result is fc

    def test_decide_returns_int(self) -> None:
        """Verify _decide() returns an int (action_id) in random phase."""
        agent = _make_agent()
        agent._phase = "random"
        agent.policy.decide.return_value = 42
        
        action_id = agent._decide([1, 2, 3, 42])
        
        assert isinstance(action_id, int)
        assert action_id == 42

    def test_prepare_next_returns_game_action(self) -> None:
        """Verify _prepare_next() returns a GameAction via _record_and_return."""
        agent = _make_agent()
        from planning.frame_context import FrameContext
        fc = FrameContext(
            scene=MagicMock(),
            ctx=MagicMock(),
            residual=(),
            observed_transition=None,
            unknowns=(),
            diverged=False,
            spec=MagicMock(),
        )
        
        action = agent._prepare_next(1, fc)
        
        assert isinstance(action, GameAction)
        assert action.value == 1

    def test_current_frame_context_builds_from_cached_scene(self) -> None:
        """Verify _current_frame_context() builds FrameContext if context is available."""
        agent = _make_agent()
        agent._scene = MagicMock()
        
        # Case 1: No context
        agent.policy.context = None
        assert agent._current_frame_context() is None
        
        # Case 2: Context available
        agent.policy.context = MagicMock()
        from planning.frame_context import FrameContext
        fc = agent._current_frame_context()
        
        assert isinstance(fc, FrameContext)
        assert fc.scene is agent._scene
        assert fc.ctx is agent.policy.context
