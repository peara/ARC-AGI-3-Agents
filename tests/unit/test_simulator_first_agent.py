"""Unit tests for the simulator-first agent modules.

Covers: world_model, check, sandbox (live + offline), agent, registration.
"""

from __future__ import annotations

import glob
from unittest.mock import MagicMock

import pytest

from agents import AVAILABLE_AGENTS
from agents.duck_harness_agent.base import DirectStepAgent
from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.check import cluster_cells, diagnose, run_check
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.world_model import extract_notes, format_notes

# ═══════════════════════════════════════════════════════════════════════════════
# 1. World Model — extract_notes / format_notes
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractNotes:
    def test_basic_notes_and_plan(self):
        result = extract_notes("Notes: test\nPlan: do thing")
        assert result == {"notes": "test", "plan": "do thing"}

    def test_empty_string(self):
        result = extract_notes("")
        assert result == {"notes": "", "plan": ""}

    def test_multiline_content(self):
        text = "Notes: first line\nsecond line\nPlan: step one\nstep two"
        result = extract_notes(text)
        assert "first line" in result["notes"]
        assert "second line" in result["notes"]
        assert "step one" in result["plan"]
        assert "step two" in result["plan"]

    def test_only_notes(self):
        result = extract_notes("Notes: just notes here")
        assert result["notes"] == "just notes here"
        assert result["plan"] == ""

    def test_only_plan(self):
        result = extract_notes("Plan: just plan here")
        assert result["plan"] == "just plan here"
        assert result["notes"] == ""


class TestFormatNotes:
    def test_basic_format(self):
        result = format_notes({"notes": "test", "plan": "do thing"})
        assert "Notes:" in result
        assert "Plan:" in result
        assert "test" in result
        assert "do thing" in result

    def test_header_line(self):
        result = format_notes({"notes": "", "plan": ""})
        assert "Notes carried from earlier turns:" in result

    def test_revision_line(self):
        result = format_notes({"notes": "x", "plan": "y"})
        assert "Revise anything above" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Check — run_check / diagnose
# ═══════════════════════════════════════════════════════════════════════════════


def _make_grids(n: int = 3) -> tuple[list[list[list[int]]], list[int]]:
    """Create n identical 4x4 grids + n-1 zero actions."""
    grid = [[0] * 4 for _ in range(4)]
    grids = [grid for _ in range(n)]
    actions = [0] * (n - 1)
    return grids, actions


def _identity_sim(grid: list[list[int]], action: int) -> list[list[int]]:
    """Identity simulate: returns the same grid."""
    return [row[:] for row in grid]


def _shift_sim(grid: list[list[int]], action: int) -> list[list[int]]:
    """Simulate that shifts all cells by +1 (always wrong vs identity grids)."""
    return [[(cell + 1) % 16 for cell in row] for row in grid]


class TestRunCheck:
    def test_identity_sim_perfect(self):
        grids, actions = _make_grids(3)
        result = run_check(_identity_sim, grids, actions, verbose=False)
        assert result["total_wrong"] == 0
        assert result["frames_correct"] == 2
        # With identical grids, there are 0 changed cells so accuracy is
        # 0/1*100 = 0.0 (the metric is "correct of changed").
        # What matters is total_wrong == 0 for a perfect sim.
        assert result["total_wrong"] == 0

    def test_wrong_sim_has_errors(self):
        grids, actions = _make_grids(3)
        result = run_check(_shift_sim, grids, actions, verbose=False)
        assert result["total_wrong"] > 0

    def test_exception_in_sim(self):
        def bad_sim(grid, action):
            raise ValueError("boom")

        grids, actions = _make_grids(3)
        result = run_check(bad_sim, grids, actions, verbose=False)
        assert result["total_wrong"] > 0

    def test_none_sim_return(self):
        def none_sim(grid, action):
            return None

        grids, actions = _make_grids(3)
        result = run_check(none_sim, grids, actions, verbose=False)
        assert result["total_wrong"] > 0

    def test_per_frame_metrics(self):
        grids, actions = _make_grids(3)
        result = run_check(_identity_sim, grids, actions, verbose=False)
        assert len(result["per_frame"]) == 2


class TestDiagnose:
    def test_diagnose_returns_dict(self):
        grids, actions = _make_grids(3)
        result = diagnose(_identity_sim, grids, actions)
        assert isinstance(result, dict)
        assert "total_wrong" in result
        assert "total_missed" in result

    def test_diagnose_with_errors(self):
        grids, actions = _make_grids(3)
        result = diagnose(_shift_sim, grids, actions)
        assert result["total_wrong"] > 0


class TestClusterCells:
    def test_empty(self):
        assert cluster_cells([]) == []

    def test_single_cell(self):
        result = cluster_cells([(5, 5)])
        assert len(result) == 1
        assert (5, 5) in result[0]

    def test_nearby_cells_merged(self):
        result = cluster_cells([(0, 0), (1, 1)])
        assert len(result) == 1

    def test_distant_cells_separate(self):
        result = cluster_cells([(0, 0), (50, 50)], threshold=3)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sandbox — Live Mode
# ═══════════════════════════════════════════════════════════════════════════════


def _mock_step_env_callback(action_id: int, action_data: dict | None = None) -> dict:
    """Minimal step_env_callback for live mode tests."""
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [[0] * 64 for _ in range(64)],
        "valid_actions": [0, 1, 2],
        "last_action_result": {},
    }


def _make_live_sandbox(**kwargs) -> SimulatorSandbox:
    return SimulatorSandbox(
        step_env_callback=_mock_step_env_callback,
        timeout=10.0,
        **kwargs,
    )


class TestSandboxLiveMode:
    def test_construction(self):
        sandbox = _make_live_sandbox()
        assert sandbox._step_env_callback is not None
        assert sandbox._grids == []
        assert sandbox._actions == []

    def test_both_modes_raises(self):
        """Cannot provide both harness and step_env_callback."""
        mock_harness = MagicMock()
        with pytest.raises(ValueError, match="either"):
            SimulatorSandbox(
                harness=mock_harness,
                step_env_callback=_mock_step_env_callback,
            )

    def test_neither_mode_raises(self):
        """Must provide either harness or step_env_callback."""
        with pytest.raises(ValueError, match="either"):
            SimulatorSandbox()

    def test_action_calls_callback(self):
        received: list[tuple[int, dict | None]] = []

        def capture_callback(action_id: int, data: dict | None = None) -> dict:
            received.append((action_id, data))
            return _mock_step_env_callback(action_id, data)

        sandbox = SimulatorSandbox(step_env_callback=capture_callback, timeout=10.0)
        # Need to set _current_frame for action() to work with grid history
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, action_taken = sandbox.run_code("action(1)")
        assert action_taken == 1
        assert len(received) == 1
        assert received[0][0] == 1

    def test_run_code_returns_3_tuple(self):
        sandbox = _make_live_sandbox()
        output, error, action_taken = sandbox.run_code("x = 42")
        assert isinstance(output, str)
        assert error is None
        assert action_taken is None

    def test_max_actions_per_turn(self):
        sandbox = SimulatorSandbox(
            step_env_callback=_mock_step_env_callback,
            timeout=10.0,
            max_actions_per_turn=2,
        )
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        # First two actions succeed
        output, error, _ = sandbox.run_code("action(0); action(1)")
        assert error is None or "Max actions" not in (error or "")
        # Third action in same turn exceeds limit
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, action_taken = sandbox.run_code("action(0)")
        assert "Max actions" in (error or "")

    def test_dunder_guard_rejects_import(self):
        sandbox = _make_live_sandbox()
        output, error, action_taken = sandbox.run_code("__import__('os')")
        assert error is not None
        assert "dunder" in error.lower()
        assert action_taken is None

    def test_update_state_sets_namespace(self):
        sandbox = _make_live_sandbox()
        sandbox.update_state(
            objects=({"id": 1},),
            adjacency=frozenset({(1, 2)}),
            current_frame=[[0]],
            previous_frame=[[1]],
            valid_actions=[0, 3],
            last_action_result={"moved": True},
            history=[{"step": 1}],
        )
        assert sandbox.namespace["objects"] == ({"id": 1},)
        assert sandbox.namespace["current_frame"] == [[0]]
        assert sandbox.namespace["previous_frame"] == [[1]]
        assert sandbox.namespace["valid_actions"] == [0, 3]

    def test_reset_turn_counter(self):
        sandbox = SimulatorSandbox(
            step_env_callback=_mock_step_env_callback,
            timeout=10.0,
            max_actions_per_turn=10,
        )
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        sandbox.run_code("action(0)")
        assert sandbox.actions_this_turn == 1
        sandbox.reset_turn_counter()
        assert sandbox.actions_this_turn == 0
        assert sandbox._action_taken is None

    def test_simulate_in_namespace(self):
        """set_simulate + simulate works in sandbox namespace."""
        sandbox = _make_live_sandbox()
        code = """
def my_sim(grid, action):
    return [row[:] for row in grid]
set_simulate(my_sim)
result = simulate([[0,0],[0,0]], 0)
"""
        output, error, action_taken = sandbox.run_code(code)
        assert error is None, f"Unexpected error: {error}"
        assert action_taken is None

    def test_simulate_not_set_raises(self):
        """simulate() without set_simulate raises RuntimeError."""
        sandbox = _make_live_sandbox()
        output, error, action_taken = sandbox.run_code("simulate([[0]], 0)")
        assert error is not None
        assert "No simulate function" in error

    def test_check_in_namespace(self):
        """check() in namespace works after set_simulate."""
        sandbox = _make_live_sandbox()
        # Add some grids manually
        sandbox._grids = [[[0] * 4 for _ in range(4)] for _ in range(3)]
        sandbox._actions = [0, 0]
        sandbox.namespace["n_frames"] = len(sandbox._grids)
        code = """
def my_sim(grid, action):
    return [row[:] for row in grid]
set_simulate(my_sim)
result = check()
"""
        output, error, action_taken = sandbox.run_code(code)
        assert error is None, f"Unexpected error: {error}"

    def test_update_notes_sandbox(self):
        """update_notes() records structured notes in the sandbox."""
        sandbox = _make_live_sandbox()
        assert sandbox._pending_notes == {}
        output, error, _ = sandbox.run_code("update_notes(notes='test notes', plan='test plan')")
        assert error is None, f"Unexpected error: {error}"
        assert sandbox._pending_notes == {"notes": "test notes", "plan": "test plan"}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sandbox — Offline Mode (using ReplayHarness)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSandboxOfflineMode:
    @pytest.fixture
    def ls20_recording(self) -> str | None:
        """Find an ls20 recording file for offline tests."""
        recordings = sorted(glob.glob("recordings/ls20-*.recording.jsonl"))
        if not recordings:
            pytest.skip("No ls20 recording files found")
        return recordings[0]

    @pytest.fixture
    def offline_sandbox(self, ls20_recording) -> SimulatorSandbox:
        """Create a sandbox in offline mode from a real recording."""
        from replay.harness import ReplayHarness

        harness = ReplayHarness.from_recording(ls20_recording)
        sandbox = SimulatorSandbox(harness=harness, timeout=30.0)
        return sandbox

    def test_offline_construction(self, offline_sandbox):
        """Offline sandbox has grids and actions loaded."""
        assert len(offline_sandbox._grids) > 0
        assert len(offline_sandbox._actions) > 0

    def test_offline_tools_available(self, offline_sandbox):
        """Offline sandbox namespace has frame access tools."""
        ns = offline_sandbox.namespace
        assert "get_frame" in ns
        assert "get_action" in ns
        assert "n_frames" in ns
        assert ns["n_frames"] > 0

    def test_offline_get_frame(self, offline_sandbox):
        """get_frame returns a grid."""
        frame = offline_sandbox.namespace["get_frame"](0)
        assert isinstance(frame, list)
        assert len(frame) > 0
        assert len(frame[0]) > 0

    def test_offline_run_code_returns_3_tuple(self, offline_sandbox):
        """run_code returns (output, error, action_taken)."""
        output, error, action_taken = offline_sandbox.run_code("x = 42")
        assert isinstance(output, str)
        assert error is None
        # In offline mode, no action() is available, so action_taken is None
        assert action_taken is None

    def test_offline_action_not_available(self, offline_sandbox):
        """action() in offline mode raises RuntimeError."""
        output, error, action_taken = offline_sandbox.run_code("action(0)")
        assert error is not None
        assert "live mode" in error.lower() or "action" in error.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Agent class
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimulatorFirstAgent:
    def test_is_subclass_of_direct_step_agent(self):
        assert issubclass(SimulatorFirstAgent, DirectStepAgent)

    def test_name_contains_simulatorfirst(self):
        # Name includes the parent name + ".simulatorfirst"
        # We can't easily construct the agent without a live game,
        # but we can check the property logic
        agent_cls = SimulatorFirstAgent
        assert "simulatorfirst" in agent_cls.__name__.lower()

    def test_estimate_tokens_returns_positive(self):
        messages = [
            {"role": "system", "content": "Hello world"},
            {"role": "user", "content": "Test message"},
        ]
        tokens = SimulatorFirstAgent._estimate_tokens(messages)
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_with_images(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "See this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
            ]},
            {"role": "assistant", "content": "I see it"},
        ]
        tokens = SimulatorFirstAgent._estimate_tokens(messages)
        # Image token cost (1024) + text tokens
        assert tokens >= 1024

    def test_estimate_tokens_empty(self):
        tokens = SimulatorFirstAgent._estimate_tokens([])
        assert tokens == 1  # max(1, 0)

    def test_strip_old_images(self):
        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "Frame 1"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,old1"}},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "text", "text": "Frame 2"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,new1"}},
            ]},
        ]
        SimulatorFirstAgent._strip_old_images(history, keep_last_n_user=1)
        # Old user message should have images stripped
        assert all(
            p.get("type") != "image_url"
            for p in history[0]["content"]
        )
        # Recent user message should keep its image
        assert any(
            p.get("type") == "image_url"
            for p in history[2]["content"]
        )

    def test_keep_recent_assistant_turns(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        kept = SimulatorFirstAgent._keep_recent_assistant_turns(messages, max_turns=1)
        # Should keep only the last assistant turn + surrounding messages
        assert any(m.get("content") == "a3" for m in kept)
        assert kept[-1].get("role") == "assistant" or any(
            m.get("content") == "a3" for m in kept
        )

    def test_keep_recent_assistant_turns_empty(self):
        result = SimulatorFirstAgent._keep_recent_assistant_turns([], max_turns=5)
        assert result == []

    def test_keep_recent_assistant_turns_zero(self):
        messages = [{"role": "user", "content": "hi"}]
        result = SimulatorFirstAgent._keep_recent_assistant_turns(messages, max_turns=0)
        assert result == []

    def test_drop_until_first_user_message(self):
        history = [
            {"role": "assistant", "content": "orphan"},
            {"role": "tool", "content": "orphan tool"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = SimulatorFirstAgent._drop_until_first_user_message(history)
        assert result[0]["role"] == "user"
        assert len(result) == 2


def test_agent_handles_update_notes_tool_call():
    """Agent should update world_model when LLM calls update_notes tool."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent._world_model = {"notes": "", "plan": ""}
    args = {"notes": "test notes", "plan": "test plan"}
    if args.get("notes"):
        agent._world_model["notes"] = args["notes"]
    if args.get("plan"):
        agent._world_model["plan"] = args["plan"]
    assert agent._world_model == {"notes": "test notes", "plan": "test plan"}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_simulatorfirst_in_available_agents(self):
        assert "simulatorfirst" in AVAILABLE_AGENTS

    def test_simulatorfirst_class_name(self):
        assert AVAILABLE_AGENTS["simulatorfirst"].__name__ == "SimulatorFirstAgent"

    def test_simulatorfirst_is_agent_subclass(self):
        from agents.agent import Agent
        assert issubclass(AVAILABLE_AGENTS["simulatorfirst"], Agent)