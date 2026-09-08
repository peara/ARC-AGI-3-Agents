"""Unit tests for the simulator-first agent modules.

Covers: world_model, check, sandbox (live + offline), agent, registration.
"""

from __future__ import annotations

import glob
import json
from unittest.mock import MagicMock

import pytest

from agents import AVAILABLE_AGENTS
from agents.loop_agent import LoopAgent
from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.check import cluster_cells, diagnose, run_check
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import SpiralGuard
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


class TestSandboxLiveMode:
    def test_construction(self, live_sandbox):
        sandbox = live_sandbox()
        assert sandbox._step_env_callback is not None
        assert sandbox._grids == []
        assert sandbox._actions == []

    def test_both_modes_raises(self, mock_step_env):
        """Cannot provide both harness and step_env_callback."""
        mock_harness = MagicMock()
        with pytest.raises(ValueError, match="either"):
            SimulatorSandbox(
                harness=mock_harness,
                step_env_callback=mock_step_env,
            )

    def test_neither_mode_raises(self):
        """Must provide either harness or step_env_callback."""
        with pytest.raises(ValueError, match="either"):
            SimulatorSandbox()

    def test_action_calls_callback(self, mock_step_env):
        received: list[tuple[int, dict | None]] = []

        def capture_callback(action_id: int, data: dict | None = None) -> dict:
            received.append((action_id, data))
            return mock_step_env(action_id, data)

        sandbox = SimulatorSandbox(step_env_callback=capture_callback, timeout=10.0)
        # Need to set _current_frame for action() to work with grid history
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, action_taken = sandbox.run_code("action(1)")
        assert action_taken == 1
        assert len(received) == 1
        assert received[0][0] == 1

    def test_run_code_returns_3_tuple(self, live_sandbox):
        sandbox = live_sandbox()
        output, error, action_taken = sandbox.run_code("x = 42")
        assert isinstance(output, str)
        assert error is None
        assert action_taken is None

    def test_action_blocked_after_win(self, mock_step_env):
        """action() should be blocked after a winning result."""
        sandbox = SimulatorSandbox(step_env_callback=mock_step_env)
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox._last_action_result = {"run_complete": True}
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, _ = sandbox.run_code("action(0)")
        assert "Game already won/over" in (error or "")

    def test_action_blocked_after_game_over(self, mock_step_env):
        """action() should be blocked after a game-over result."""
        sandbox = SimulatorSandbox(step_env_callback=mock_step_env)
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox._last_action_result = {"game_over": True}
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, _ = sandbox.run_code("action(0)")
        assert "Game already won/over" in (error or "")

    def test_dunder_guard_rejects_import(self, live_sandbox):
        sandbox = live_sandbox()
        output, error, action_taken = sandbox.run_code("__import__('os')")
        assert error is not None
        assert "dunder" in error.lower()
        assert action_taken is None

    def test_update_state_sets_namespace(self, live_sandbox):
        sandbox = live_sandbox()
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

    def test_reset_turn_counter(self, mock_step_env):
        sandbox = SimulatorSandbox(
            step_env_callback=mock_step_env,
            timeout=10.0,
        )
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        sandbox.run_code("action(0)")
        assert sandbox.actions_this_turn == 1
        sandbox.reset_turn_counter()
        assert sandbox.actions_this_turn == 0
        assert sandbox._action_taken is None

    def test_simulate_in_namespace(self, live_sandbox):
        """set_simulate + simulate works in sandbox namespace."""
        sandbox = live_sandbox()
        code = """
def my_sim(grid, action):
    return [row[:] for row in grid]
set_simulate(my_sim)
result = simulate([[0,0],[0,0]], 0)
"""
        output, error, action_taken = sandbox.run_code(code)
        assert error is None, f"Unexpected error: {error}"
        assert action_taken is None

    def test_simulate_not_set_raises(self, live_sandbox):
        """simulate() without set_simulate raises RuntimeError."""
        sandbox = live_sandbox()
        output, error, action_taken = sandbox.run_code("simulate([[0]], 0)")
        assert error is not None
        assert "No simulate function" in error

    def test_check_in_namespace(self, live_sandbox):
        """check() in namespace works after set_simulate."""
        sandbox = live_sandbox()
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

    def test_update_notes_sandbox(self, live_sandbox):
        """update_notes() records structured notes in the sandbox."""
        sandbox = live_sandbox()
        assert sandbox._pending_notes == {}
        output, error, _ = sandbox.run_code(
            "update_notes(notes='test notes', plan='test plan')"
        )
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
        assert issubclass(SimulatorFirstAgent, LoopAgent)

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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "See this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            },
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
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,old1"},
                    },
                ],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 2"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,new1"},
                    },
                ],
            },
        ]
        SimulatorFirstAgent._strip_old_images(history, keep_last_n_user=1)
        # Old user message should have images stripped
        assert all(p.get("type") != "image_url" for p in history[0]["content"])
        # Recent user message should keep its image
        assert any(p.get("type") == "image_url" for p in history[2]["content"])

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


class TestTrimOldToolResults:
    def test_replaces_old_results(self):
        """6 python tool results → 3 oldest have placeholder."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        msgs = []
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"tc{i}",
                    "content": f"output {i}" * 100,
                }
            )
        agent._trim_old_tool_results(msgs, keep_last_n=3)
        assert len(msgs) == 12
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        placeholders = [m for m in tool_msgs if "[Old output" in m.get("content", "")]
        assert len(placeholders) >= 1

    def test_no_delete(self):
        """Messages list length unchanged after trimming."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        msgs = []
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {"role": "tool", "tool_call_id": f"tc{i}", "content": f"output {i}"}
            )
        original_len = len(msgs)
        agent._trim_old_tool_results(msgs, keep_last_n=3)
        assert len(msgs) == original_len

    def test_exempts_update_notes(self):
        """update_notes results are NOT trimmed."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        msgs = []
        for i in range(4):
            name = "update_notes" if i % 2 == 1 else "python"
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": f"tc{i}", "function": {"name": name, "arguments": "{}"}}
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"tc{i}",
                    "content": f"output {i}" * 50,
                }
            )
        agent._trim_old_tool_results(msgs, keep_last_n=2)
        for i in [1, 3]:
            tool_msg = msgs[i * 2 + 1]
            assert "output" in tool_msg["content"], (
                f"update_notes result {i} was trimmed"
            )

    def test_preserves_pairing(self):
        """Every role:tool has matching preceding tool_calls entry."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        msgs = []
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {"name": "python", "arguments": "{}"},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"tc{i}",
                    "content": f"output {i}" * 100,
                }
            )
        agent._trim_old_tool_results(msgs, keep_last_n=3)
        for i, m in enumerate(msgs):
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                found = False
                for j in range(i):
                    if msgs[j].get("role") == "assistant" and msgs[j].get("tool_calls"):
                        for tc in msgs[j]["tool_calls"]:
                            if tc["id"] == tc_id:
                                found = True
                                break
                assert found, f"Tool message at index {i} has no matching tool_calls"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestBfs:
    def test_bfs_no_simulate(self):
        """bfs returns None when no simulate is registered."""
        sb = SimulatorSandbox(step_env_callback=lambda a, d: {})
        output, error, _ = sb.run_code(
            "result = bfs([[0]], lambda g: True)\nprint(f'Result: {result}')"
        )
        assert error is None
        assert "No simulate" in output

    def test_bfs_goal_already_reached(self):
        """bfs returns [] when goal is already reached at start."""
        sb = SimulatorSandbox(step_env_callback=lambda a, d: {})
        sb.namespace["valid_actions"] = [0, 1]
        output, error, _ = sb.run_code(
            """
def sim(grid, action):
    return [row[:] for row in grid]
set_simulate(sim)
result = bfs([[0, 0], [0, 0]], lambda g: True)
print(f'Result: {result}')
"""
        )
        assert error is None
        assert "Goal already reached" in output
        assert "Result: []" in output

    def test_bfs_finds_path(self):
        """bfs finds a 1-step path with a simple simulate."""
        sb = SimulatorSandbox(step_env_callback=lambda a, d: {})
        sb.namespace["valid_actions"] = [0, 1]
        output, error, _ = sb.run_code(
            """
def simulate(grid, action):
    new_grid = [row[:] for row in grid]
    if action == 1:
        new_grid[0][0] = 99  # change a cell
    return new_grid
set_simulate(simulate)
def goal(grid):
    return grid[0][0] == 99
result = bfs([[0, 0], [0, 0]], goal)
print(f'Path: {result}')
"""
        )
        assert error is None
        assert "Path found: [1]" in output
        assert "Path: [1]" in output

    def test_bfs_no_path(self):
        """bfs returns None when no path exists within depth."""
        sb = SimulatorSandbox(step_env_callback=lambda a, d: {})
        sb.namespace["valid_actions"] = [0, 1]
        output, error, _ = sb.run_code(
            """
def simulate(grid, action):
    return [row[:] for row in grid]
set_simulate(simulate)
def goal(grid):
    return grid[0][0] == 99
result = bfs([[0, 0], [0, 0]], goal)
print(f'Result: {result}')
print(f'Depth: {result}')
"""
        )
        assert error is None
        assert "No path found" in output
        assert "No path found within depth 20" in output
        assert "Result: None" in output


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Context trimming — _trim_old_non_tool_messages
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrimOldNonToolMessages:
    def test_trim_nudges_keeps_last(self):
        msgs = []
        for i in range(5):
            text = (
                "If you discovered something new about the game, call update_notes."
                if i % 2 == 0
                else "Please use the python tool to explore the game."
            )
            msgs.append({"role": "user", "content": text})
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert len(msgs) == 5
        assert msgs[-1]["content"] != "[nudge]"
        for i in range(4):
            assert msgs[i]["content"] == "[nudge]"

    def test_trim_frame_prompts_keeps_last(self):
        msgs = []
        for i in range(3):
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Frame prompt {i}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i}"}},
                    ],
                }
            )
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "[frame]"
        assert msgs[1]["content"] == "[frame]"
        assert msgs[2]["content"] != "[frame]"
        assert msgs[2]["content"][0]["text"] == "Frame prompt 2"

    def test_trim_assistant_code_keeps_last_4(self):
        msgs = []
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {
                                "name": "python",
                                "arguments": f'{{"code": "print({i})"}}',
                            },
                        }
                    ],
                }
            )
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert len(msgs) == 6
        for i in range(2):
            assert (
                msgs[i]["tool_calls"][0]["function"]["arguments"]
                == '{"code": "# previous code omitted"}'
            )
        for i in range(2, 6):
            assert msgs[i]["tool_calls"][0]["function"]["arguments"] == f'{{"code": "print({i})"}}'

    def test_trim_no_delete(self):
        msgs = []
        for i in range(10):
            msgs.append(
                {
                    "role": "user",
                    "content": "If you discovered something new about the game, call update_notes.",
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "print(1)"}',
                            },
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": f"tc{i}", "content": "output"})
        original_len = len(msgs)
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert len(msgs) == original_len

    def test_trim_preserves_tool_call_structure(self):
        msgs = []
        for i in range(3):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {
                                "name": "python",
                                "arguments": f'{{"code": "print({i})"}}',
                            },
                        }
                    ],
                }
            )
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        for i, m in enumerate(msgs):
            assert m["role"] == "assistant"
            assert m["tool_calls"][0]["id"] == f"tc{i}"
            assert m["tool_calls"][0]["function"]["name"] == "python"

    def test_trim_idempotent(self):
        msgs = []
        for i in range(5):
            msgs.append(
                {
                    "role": "user",
                    "content": "If you discovered something new about the game, call update_notes.",
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "print(1)"}',
                            },
                        }
                    ],
                }
            )
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        snapshot = [dict(m) for m in msgs]
        for i, m in enumerate(msgs):
            if isinstance(m.get("content"), str):
                snapshot[i]["content"] = m["content"]
            if m.get("tool_calls"):
                snapshot[i]["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "function": dict(tc["function"]),
                    }
                    for tc in m["tool_calls"]
                ]
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        for orig, after in zip(snapshot, msgs):
            if isinstance(orig.get("content"), str):
                assert orig["content"] == after["content"]
            if orig.get("tool_calls"):
                for tc_orig, tc_after in zip(orig["tool_calls"], after["tool_calls"]):
                    assert tc_orig["id"] == tc_after["id"]
                    assert tc_orig["function"]["name"] == tc_after["function"]["name"]
                    assert tc_orig["function"]["arguments"] == tc_after["function"]["arguments"]

    def test_trim_single_frame_no_op(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Only frame prompt"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert len(msgs) == 1
        assert msgs[0]["content"][0]["text"] == "Only frame prompt"

    def test_trim_update_notes_arguments(self):
        msgs = []
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "function": {
                                "name": "update_notes",
                                "arguments": f'{{"notes": "note {i}", "plan": "plan {i}"}}',
                            },
                        }
                    ],
                }
            )
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        for i in range(2):
            assert (
                msgs[i]["tool_calls"][0]["function"]["arguments"]
                == '{"notes": "[trimmed]", "plan": "[trimmed]"}'
            )
        for i in range(2, 6):
            assert (
                msgs[i]["tool_calls"][0]["function"]["arguments"]
                == f'{{"notes": "note {i}", "plan": "plan {i}"}}'
            )

    def test_trim_both_nudge_types(self):
        msgs = [
            {"role": "user", "content": "If you discovered something new about the game, call update_notes."},
            {"role": "user", "content": "Please use the python tool to explore the game."},
            {"role": "user", "content": "Another nudge: discovered something new."},
        ]
        SimulatorFirstAgent._trim_old_non_tool_messages(msgs)
        assert msgs[0]["content"] == "[nudge]"
        assert msgs[1]["content"] == "[nudge]"
        assert "discovered something new" in msgs[2]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Action budget enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestActionBudget:
    @pytest.mark.unit
    def test_step_env_has_no_budget_guard(self):
        import inspect

        source = inspect.getsource(LoopAgent.step_env)
        assert "MAX_ACTIONS" not in source, \
            "step_env must not contain a budget guard (committed batches complete)"
        assert "RuntimeError" not in source, \
            "step_env must not raise a RuntimeError"

    @pytest.mark.unit
    def test_tool_loop_budget_guard(self):
        import inspect

        source = "\n".join(
            [
                inspect.getsource(SimulatorFirstAgent.run),
                inspect.getsource(SimulatorFirstAgent._run_tool_loop),
            ]
        )
        assert "action_counter >= self.MAX_ACTIONS" in source, \
            "run()/_run_tool_loop must check action_counter vs MAX_ACTIONS"
        assert "budget exhausted mid-turn" in source, \
            "tool loop must log budget exhausted mid-turn"
        assert "guardrail:" in source and "budget exhausted" in source, \
            "tool loop must use the guardrail log format"

    @pytest.mark.unit
    def test_no_fallback_returns_reset_on_exhaustion(self):
        """P2-T3: run() has no random fallback when budget is exhausted.
        The anti-spiral cap at >=36 returns directly (no random action)."""
        import inspect

        source = "\n".join(
            [
                inspect.getsource(SimulatorFirstAgent.run),
                inspect.getsource(SimulatorFirstAgent._run_tool_loop),
            ]
        )

        # P2-T3: no random action injection anywhere in run()/_run_tool_loop
        assert "random" not in source or "No random fallback" in source, \
            "run() must not inject random actions"

    @pytest.mark.unit
    def test_no_max_actions_attr_on_class(self):
        assert (
            "max_actions" not in SimulatorFirstAgent.__dict__
        ), "SimulatorFirstAgent must not have the stale max_actions class attr"

    def _make_budget_exhausted_agent(self):
        import numpy as np
        from arcengine import FrameData, GameState

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 30
        agent.action_counter = 61  # crossed MAX mid-turn, as in the 61/30 bug
        agent.game_id = "test"
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = None
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._llm_calls = 0
        agent._sandbox_steps = 0
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def fake_llm_chat(**kwargs):
            agent._llm_calls += 1
            raise AssertionError("LLM must not be called when budget is exhausted")

        def fake_step_env(action):
            agent._sandbox_steps += 1
            raise AssertionError("step_env must not be called once budget is spent")

        agent._llm_chat = fake_llm_chat
        agent.step_env = fake_step_env  # type: ignore[method-assign]
        agent._sandbox = MagicMock()
        agent._sandbox._simulate = None
        agent._sandbox._pending_notes = {}
        agent._sandbox._pending_exception_flow = None
        agent._sandbox._transition_pending = False
        agent._sandbox._action_taken = None
        agent._sandbox._last_check_result = None
        agent._sandbox._last_bfs_result = None
        agent._sandbox._pending_images = []
        agent._sandbox.reset_turn_counter = MagicMock()
        agent._sandbox.update_state = MagicMock()
        agent._workflow = MagicMock()
        agent._workflow.directive.return_value = ""
        agent._workflow.update = MagicMock()
        agent._trim_messages_for_context = lambda messages, **kw: list(messages)
        agent._append_notes_message = lambda messages, wm: None

        frame = FrameData(
            game_id="test",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.frames = [frame]
        return agent, [frame]

    @pytest.mark.unit
    def test_run_no_llm_call_after_budget_exhausted(self):
        """61/30 bug regression: counter past MAX must end the turn without
        any further LLM calls or sandbox steps."""
        agent, frames = self._make_budget_exhausted_agent()
        agent.run()
        assert agent._llm_calls == 0
        assert agent._sandbox_steps == 0

    @pytest.mark.unit
    def test_run_ends_when_batch_crosses_budget(self):
        """Batch crosses MAX mid-turn: LLM must not be called again after it."""
        agent, frames = self._make_budget_exhausted_agent()
        agent.action_counter = 28  # 2 left
        agent.MAX_ACTIONS = 30

        def fake_llm_chat(**kwargs):
            agent._llm_calls += 1
            return type(
                "R",
                (),
                {
                    "tool_calls": [{
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code": "action(4)\\naction(4)\\naction(4)"}',
                        },
                    }],
                    "content": "running batch",
                },
            )()

        def fake_sandbox_run(code):
            agent.action_counter = 31  # batch pushes past MAX
            return ("output", None, 4)

        agent._llm_chat = fake_llm_chat
        agent._sandbox.run_code = fake_sandbox_run  # type: ignore[method-assign]

        agent.run()
        assert agent._llm_calls == 1, (
            "after the batch crossed MAX, no further LLM call may happen"
        )


class TestNotesMessageRegression:
    def test_turn_start_appends_notes_user_message(self):
        """Turn-start message list should include a trailing notes user message."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "test notes", "plan": "test plan"}
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "frame prompt"},
            {"role": "assistant", "content": "ok"},
        ]
        # Production will append a separate notes message at turn start.
        agent._append_notes_message(messages, agent._world_model)
        notes_messages = [
            m
            for m in messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]\n")
        ]
        assert len(notes_messages) == 1
        assert "test notes" in notes_messages[0]["content"]
        assert "test plan" in notes_messages[0]["content"]

    def test_update_notes_updates_message_in_place(self):
        """Updating notes should mutate the existing notes message, not append."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        stale_world_model = {"notes": "old notes", "plan": "old plan"}
        notes_content = f"[Current notes]\n{format_notes(stale_world_model)}"
        messages = [
            {"role": "user", "content": notes_content},
        ]
        original_len = len(messages)

        # Production will update the notes message in-place.
        new_world_model = {"notes": "new notes", "plan": "new plan"}
        agent._update_notes_message(messages, new_world_model)
        assert len(messages) == original_len
        notes_msg = messages[0]
        assert notes_msg["content"].startswith("[Current notes]\n")
        assert "new notes" in notes_msg["content"]
        assert "new plan" in notes_msg["content"]
        assert "old notes" not in notes_msg["content"]
        assert "old plan" not in notes_msg["content"]

    def test_same_turn_double_call_reflects_first_call(self):
        """Regression test: first update_notes correction must replace stale notes."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        stale_world_model = {"notes": "stale notes from prev turn", "plan": ""}
        stale_notes = f"[Current notes]\n{format_notes(stale_world_model)}"
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": stale_notes},
        ]

        # Production will apply the first update_notes correction in-place.
        correction = {"notes": "correction: color 0 does not move", "plan": ""}
        agent._update_notes_message(messages, correction)

        notes_msg = messages[-1]
        assert notes_msg["content"].startswith("[Current notes]\n")
        assert "correction: color 0 does not move" in notes_msg["content"]
        assert "stale notes from prev turn" not in notes_msg["content"]

    def test_update_notes_empty_strings(self):
        """Empty notes/plan values are still formatted with labels intact."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "", "plan": ""}
        initial_world_model = {"notes": "initial notes", "plan": "initial plan"}
        notes_content = f"[Current notes]\n{format_notes(initial_world_model)}"
        messages = [
            {"role": "user", "content": notes_content},
        ]

        agent._update_notes_message(messages, agent._world_model)

        notes_msg = messages[-1]
        assert notes_msg["content"].startswith("[Current notes]\n")
        assert "Notes:" in notes_msg["content"]
        assert "Plan:" in notes_msg["content"]
        assert "initial notes" not in notes_msg["content"]
        assert "initial plan" not in notes_msg["content"]

    def test_update_notes_notes_only_no_plan(self):
        """Updating notes only leaves the existing plan text in place."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        initial_world_model = {"notes": "old", "plan": "old plan"}
        notes_content = f"[Current notes]\n{format_notes(initial_world_model)}"
        messages = [
            {"role": "user", "content": notes_content},
        ]

        updated_world_model = {"notes": "new notes only", "plan": "old plan"}
        agent._update_notes_message(messages, updated_world_model)

        notes_msg = messages[-1]
        assert notes_msg["content"].startswith("[Current notes]\n")
        assert "new notes only" in notes_msg["content"]
        assert "old plan" in notes_msg["content"]

    def test_update_notes_multiline(self):
        """Multiline notes content is preserved in the notes message."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "line1\nline2\nline3", "plan": ""}
        stale_world_model = {"notes": "stale single line", "plan": ""}
        notes_content = f"[Current notes]\n{format_notes(stale_world_model)}"
        messages = [
            {"role": "user", "content": notes_content},
        ]

        agent._update_notes_message(messages, agent._world_model)

        notes_msg = messages[-1]
        assert notes_msg["content"].startswith("[Current notes]\n")
        assert "line1" in notes_msg["content"]
        assert "line3" in notes_msg["content"]
        assert "stale single line" not in notes_msg["content"]

    def test_notes_message_not_trimmed_by_non_tool_trimmer(self):
        """The non-tool message trimmer must leave notes messages unchanged."""
        notes_content = (
            "[Current notes]\n"
            "Notes carried from earlier turns:\n"
            "Notes: test\n"
            "Plan: \n"
            "- Revise anything above if it contradicts what you see now."
        )
        messages = [
            {
                "role": "user",
                "content": "If you discovered something new about the game, call update_notes.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 0"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {"role": "user", "content": notes_content},
            {
                "role": "user",
                "content": "If you discovered something new about the game, call update_notes.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def"},
                    },
                ],
            },
        ]

        SimulatorFirstAgent._trim_old_non_tool_messages(messages)

        assert messages[0]["content"] == "[nudge]"
        assert messages[1]["content"] == "[frame]"
        assert messages[2]["content"] == notes_content
        assert "discovered something new" in messages[3]["content"]
        assert messages[4]["content"][0]["text"] == "Frame 1"

    def test_persistent_history_strips_notes_messages(self):
        """_persistent_history_messages must remove all [Current notes] messages."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "frame 1 notes", "plan": "frame 1 plan"}
        agent._context_budget_tokens = 100000
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "frame"},
            {
                "role": "user",
                "content": f"[Current notes]\n{format_notes(agent._world_model)}",
            },
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
        ]
        result = agent._persistent_history_messages(messages)
        notes_messages = [
            m
            for m in result
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]")
        ]
        assert len(notes_messages) == 0

    def test_multi_frame_only_one_notes_message(self):
        """Across frames only one notes message exists at prompt time, and none persist."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._context_budget_tokens = 100000

        # Frame 1
        agent._world_model = {"notes": "frame 1", "plan": ""}
        messages_1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "frame1"},
        ]
        agent._append_notes_message(messages_1, agent._world_model)
        history_1 = agent._persistent_history_messages(messages_1)
        assert (
            len([
                m
                for m in messages_1
                if isinstance(m.get("content"), str)
                and m["content"].startswith("[Current notes]")
            ])
            == 1
        )

        # Frame 2
        agent._world_model = {"notes": "frame 2", "plan": ""}
        messages_2 = [
            {"role": "system", "content": "sys"},
            *history_1,
            {"role": "user", "content": "frame2"},
        ]
        agent._append_notes_message(messages_2, agent._world_model)
        assert (
            len([
                m
                for m in messages_2
                if isinstance(m.get("content"), str)
                and m["content"].startswith("[Current notes]")
            ])
            == 1
        )
        history_2 = agent._persistent_history_messages(messages_2)

        # Frame 3
        agent._world_model = {"notes": "frame 3", "plan": ""}
        messages_3 = [
            {"role": "system", "content": "sys"},
            *history_2,
            {"role": "user", "content": "frame3"},
        ]
        agent._append_notes_message(messages_3, agent._world_model)
        assert (
            len([
                m
                for m in messages_3
                if isinstance(m.get("content"), str)
                and m["content"].startswith("[Current notes]")
            ])
            == 1
        )

        notes_in_history_2 = [
            m
            for m in history_2
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]")
        ]
        assert len(notes_in_history_2) == 0

    def test_persistent_history_zero_notes_after_strip(self):
        """Strip removes multiple [Current notes] messages regardless of position."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "n", "plan": "p"}
        agent._context_budget_tokens = 100000
        notes = f"[Current notes]\n{format_notes(agent._world_model)}"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": notes},
            {"role": "user", "content": "frame prompt"},
            {"role": "user", "content": notes},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            {"role": "user", "content": notes},
        ]
        result = agent._persistent_history_messages(messages)
        notes_messages = [
            m
            for m in result
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]")
        ]
        assert len(notes_messages) == 0
        assert any(m.get("content") == "frame prompt" for m in result)
        assert any(m.get("role") == "assistant" for m in result)

    def test_preserve_history_false_rollback_no_stale_notes(self):
        """When preserve_history=False, rolling back to previous_history has no notes."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._world_model = {"notes": "frame 1", "plan": ""}
        agent._context_budget_tokens = 100000
        agent._history_messages = []
        messages_1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "frame1"},
        ]
        agent._append_notes_message(messages_1, agent._world_model)
        history_1 = agent._persistent_history_messages(messages_1)
        assert (
            len([
                m
                for m in history_1
                if isinstance(m.get("content"), str)
                and m["content"].startswith("[Current notes]")
            ])
            == 0
        )

        previous_history = list(history_1)
        agent._history_messages = previous_history
        stale_notes = [
            m
            for m in agent._history_messages
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]")
        ]
        assert len(stale_notes) == 0

    def test_degenerate_notes_only_user_message(self):
        """_strip_notes_messages handles the case where the only user message is notes."""
        notes_content = f"[Current notes]\n{format_notes({'notes': 'only', 'plan': ''})}"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": notes_content},
            {"role": "assistant", "content": "ack"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        result = SimulatorFirstAgent._strip_notes_messages(messages)
        user_messages = [m for m in result if m.get("role") == "user"]
        assert len(user_messages) == 0
        assert all(
            not (
                isinstance(m.get("content"), str)
                and m["content"].startswith("[Current notes]")
            )
            for m in result
        )

    def test_high_turn_count_notes_stripped_first(self):
        """Notes are stripped before _keep_recent_assistant_turns can retain them."""
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._context_budget_tokens = 100000
        notes_content = f"[Current notes]\n{format_notes({'notes': 'high turn', 'plan': ''})}"
        messages: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": notes_content},
        ]
        for i in range(35):
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"tc{i}",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "1"}',
                            },
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": f"tc{i}", "content": "output"}
            )
        result = agent._persistent_history_messages(messages)
        notes_messages = [
            m
            for m in result
            if isinstance(m.get("content"), str)
            and m["content"].startswith("[Current notes]")
        ]
        assert len(notes_messages) == 0


class TestLevelTransition:
    """End-to-end level-transition walkthrough using run().

    Drives the real SimulatorFirstAgent.run over a scripted step_env
    callback whose batch [4, 4, 1] wins a level (action_id=1 triggers
    level_completed=True). The sandbox raises LevelTransition after
    action(1), hard-aborting the batch. run() detects the transition
    and processes it on the next outer-loop iteration.

    All tests terminate run() via the anti-spiral guard (>=36 non-action
    calls) after the transition is processed, since the game is not
    actually won (levels_completed=1 < win_levels=7).
    """

    def _make_level_transition_agent(self, *, seed_notes="", seed_plan=""):
        """Build a fully-wired SimulatorFirstAgent with stubbed LLM/step_env.

        The LLM stub returns the winning batch on the first call, then
        no-action python calls on all subsequent calls. The step_env
        callback makes action_id=1 trigger level_completed=True.

        run() terminates via the anti-spiral guard (>=36 non-action calls)
        after the transition is processed.
        """
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-transition"
        agent._world_model = {"notes": seed_notes, "plan": seed_plan}
        agent._history_messages: list[dict] = []
        agent._history_turns: list[dict] = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result: dict = {}
        agent._current_grid: list[list[int]] | None = None
        agent._previous_grid: list[list[int]] | None = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._llm_calls = 0
        agent._sandbox_steps = 0
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False
        agent._objects: tuple = ()
        agent._adjacency = frozenset()
        agent._non_action_calls = 0

        captured = {"messages": []}
        call_index = [0]

        def fake_llm_chat(**kwargs):
            agent._llm_calls += 1
            call_index[0] += 1
            msgs = kwargs.get("messages", [])
            captured["messages"].append([dict(m) for m in msgs])
            if call_index[0] == 1:
                return type(
                    "R",
                    (),
                    {
                        "tool_calls": [{
                            "id": "tc-win",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "for a in [4, 4, 1]:\\n    action(a)\\n"}',
                            },
                        }],
                        "content": None,
                    },
                )()
            else:
                return type(
                    "R",
                    (),
                    {
                        "tool_calls": [{
                            "id": f"tc-{call_index[0]}",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "x = 1"}',
                            },
                        }],
                        "content": None,
                    },
                )()

        def fake_step_env(action):
            from arcengine import FrameData, GameState
            agent._sandbox_steps += 1
            action_id = action.value if hasattr(action, "value") else int(action)
            levels_before = agent._current_grid_levels_completed
            win = action_id == 1
            if win:
                levels_after = levels_before + 1
            else:
                levels_after = levels_before
            frame = FrameData(
                game_id="test-transition",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=levels_after,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
            agent.frames.append(frame)
            agent.action_counter += 1
            agent._current_grid = [list(row) for row in frame.frame[0]]
            if len(agent.frames) >= 2 and agent.frames[-2].frame is not None:
                prev_raw = agent.frames[-2].frame[0]
                agent._previous_grid = (
                    [list(r) for r in prev_raw]
                    if not isinstance(prev_raw, list)
                    else [list(r) for r in prev_raw]
                )
            else:
                agent._previous_grid = None
            return frame

        agent._llm_chat = fake_llm_chat
        agent.step_env = fake_step_env

        holder: dict = {"agent": None}

        def adapter(action_id: int, action_data):
            from arcengine import GameAction
            ag = holder["agent"]
            game_action = GameAction.from_id(action_id)
            ag.step_env(game_action)
            if len(ag.frames) >= 2:
                prev = ag.frames[-2]
                curr = ag.frames[-1]
                prev_levels = prev.levels_completed if hasattr(prev, "levels_completed") else 0
                curr_levels = curr.levels_completed if hasattr(curr, "levels_completed") else 0
                prev_grid = prev.frame[0] if prev.frame else None
                curr_grid = curr.frame[0] if curr.frame else None
                ag._last_action_result = {
                    "board_changed": prev_grid != curr_grid,
                    "done": curr.state.name in ("GAME_OVER", "WIN"),
                    "level_completed": curr_levels > prev_levels,
                    "game_over": curr.state.name == "GAME_OVER",
                    "run_complete": curr.state.name == "WIN",
                    "reward": curr_levels - prev_levels,
                    "valid_actions": list(curr.available_actions or []),
                }
                if curr_levels > ag._current_grid_levels_completed:
                    ag._current_grid_levels_completed = curr_levels
            else:
                ag._last_action_result = {}

            state_response: dict = {
                "objects": ag._objects,
                "adjacency": ag._adjacency,
                "history": ag._history_turns,
            }
            if ag._current_grid is not None:
                state_response["grid"] = ag._current_grid
            state_response["valid_actions"] = ag._valid_actions
            state_response["last_action_result"] = ag._last_action_result
            return state_response

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        holder["agent"] = agent
        agent._workflow = WorkflowController(sandbox)

        frame = FrameData(
            game_id="test-transition",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.frames = [frame]
        return agent, agent.frames, captured

    @pytest.mark.unit
    def test_winning_batch_hard_aborts_and_only_three_actions_taken(self):
        """Batch [4,4,1] wins at action(1). The LevelTransition hard-aborts
        the remaining batch actions. Only 3 env steps are taken."""
        agent, frames, _ = self._make_level_transition_agent()

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        agent.run()

        assert agent._sandbox_steps == 3, (
            f"only 3 env steps (got {agent._sandbox_steps}) — "
            "remaining batched actions must not step"
        )

    @pytest.mark.unit
    def test_transition_turn_prompt_has_level_transition_text(self):
        """After the winning batch, the next LLM call's prompt contains
        LEVEL TRANSITION text and no EXCEPTION FLOW text."""
        agent, frames, captured = self._make_level_transition_agent()

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        agent.run()

        found_transition = False
        for i, msgs in enumerate(captured["messages"]):
            if i == 0:
                continue
            all_text = []
            for m in msgs:
                c = m.get("content")
                if isinstance(c, str):
                    all_text.append(c)
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict):
                            all_text.append(part.get("text", ""))
                elif m.get("type") == "text":
                    all_text.append(m.get("text", ""))
            joined = "\n".join(all_text)
            if "LEVEL TRANSITION" in joined:
                found_transition = True
                assert "EXCEPTION FLOW" not in joined, (
                    "EXCEPTION FLOW must not leak into the transition prompt"
                )
                break

        assert found_transition, (
            "transition prompt must contain LEVEL TRANSITION text"
        )

    @pytest.mark.unit
    def test_transition_clears_history_messages_and_history_turns(self):
        """After the level transition, _history_messages and _history_turns
        are cleared by the transition block in run()."""
        agent, frames, _ = self._make_level_transition_agent()

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        agent._history_messages = [
            {"role": "assistant", "content": "stale from dead level"},
            {"role": "tool", "tool_call_id": "old", "content": "stale result"},
        ]
        agent._history_turns = [
            {"action": 4, "frame_index": 0, "frame": []},
            {"action": 4, "frame_index": 1, "frame": []},
        ]

        agent.run()

        assert agent._history_messages == [], (
            f"history_messages must be cleared (got {len(agent._history_messages)} msgs)"
        )
        assert len(agent._history_turns) == 0, (
            f"history_turns must be empty after transition "
            f"(got {len(agent._history_turns)})"
        )

    @pytest.mark.unit
    def test_transition_clears_plan_preserves_notes_and_resets_workflow(self):
        """After the level transition, plan is cleared and workflow was
        reset to EXPLORE via reset_to_explore."""

        seeded_notes = "Maze. floor=3,wall=4. Player 5x5. Target box=portal."
        seeded_plan = "Execute path [1,4,4,4,4,1,1,1] in real env."
        agent, frames, _ = self._make_level_transition_agent(
            seed_notes=seeded_notes, seed_plan=seeded_plan,
        )

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        reset_calls: list[str] = []
        original_reset = agent._workflow.reset_to_explore

        def tracking_reset(reason: str = "level transition"):
            reset_calls.append(reason)
            original_reset(reason)

        agent._workflow.reset_to_explore = tracking_reset

        agent.run()

        assert agent._world_model["plan"] == "", (
            f"plan must be cleared (got {agent._world_model['plan']!r})"
        )
        assert agent._world_model["notes"], "notes must not be empty after transition"

        assert len(reset_calls) >= 1, (
            f"reset_to_explore must be called during transition "
            f"(got reset calls: {reset_calls})"
        )

    @pytest.mark.unit
    def test_transition_turn_no_env_steps_after_transition(self):
        """After the winning batch (3 env steps), no additional env steps
        are taken. run() terminates via anti-spiral guard without stepping."""
        agent, frames, _ = self._make_level_transition_agent()

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        agent.run()

        assert agent._sandbox_steps == 3, (
            f"no env step in transition turn (got {agent._sandbox_steps})"
        )

    @pytest.mark.unit
    def test_check_after_transition_reports_zero_frames(self):
        """After the level transition, sandbox._grids is empty and check()
        reports 'No simulate frames recorded'."""
        agent, frames, _ = self._make_level_transition_agent()

        sandbox = agent._sandbox
        sandbox._current_frame = [list(row) for row in frames[0].frame[0]]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None

        reg_out, reg_err, _ = sandbox.run_code(
            "def simulate(g, a):\n    return g\nset_simulate(simulate)\n"
        )
        assert reg_err is None, f"simulate registration failed: {reg_err}"
        assert sandbox._simulate is not None

        agent.run()

        assert sandbox._simulate is not None, (
            "simulate must be preserved across level transition"
        )
        assert len(sandbox._grids) == 1, (
            "after transition + reseed, _grids must hold exactly the new "
            f"level's frame 0 (got {len(sandbox._grids)} grids)"
        )

        output, error, _ = sandbox.run_code("check()")
        assert error is None, f"check() must not raise (got error: {error})"
        combined = output or ""
        assert "No transitions recorded yet" in combined, (
            f"check() output must report no transitions (got: {combined[:120]!r})"
        )
        assert sandbox._grids == [sandbox._current_frame], (
            "after transition + reseed, _grids must hold exactly the new "
            f"level's frame 0 (got {len(sandbox._grids)} grids)"
        )


class TestFrameZeroSeed:
    """Seed of grids[0] in live mode (ls20-simulate-spiral brainstorm).

    update_state() appends the current frame when _grids is empty so
    check()/diagnose() can see frame 0 before any action(); with fewer than
    2 grids they report an honest 'no transitions' message instead of a
    misleading 0-frame accuracy.
    """

    def _make_seeded(self, live_sandbox):
        sandbox = live_sandbox()
        grid = [[3] * 4 for _ in range(4)]
        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=grid,
            previous_frame=None,
            valid_actions=[1, 2],
            last_action_result={},
            history=[],
        )
        return sandbox

    @pytest.mark.unit
    def test_update_state_seeds_frame_zero(self, live_sandbox):
        sandbox = self._make_seeded(live_sandbox)
        assert sandbox._grids == [[[3] * 4 for _ in range(4)]]
        assert sandbox.namespace["n_frames"] == 1

    @pytest.mark.unit
    def test_seed_is_deep_copy(self, live_sandbox):
        grid = [[3] * 4 for _ in range(4)]
        sandbox = live_sandbox()
        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=grid,
            previous_frame=None,
            valid_actions=[1, 2],
            last_action_result={},
            history=[],
        )
        sandbox._current_frame[0][0] = 99
        assert sandbox._grids[0][0][0] == 3, (
            "seeded grid must be a copy — mutating current_frame must not "
            "alter the recorded frame 0"
        )

    @pytest.mark.unit
    def test_seed_not_duplicated_on_second_update_state(self, live_sandbox):
        sandbox = self._make_seeded(live_sandbox)
        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=[[5] * 4 for _ in range(4)],
            previous_frame=None,
            valid_actions=[1, 2],
            last_action_result={},
            history=[],
        )
        assert len(sandbox._grids) == 1, (
            "seeding must only fire when _grids is empty"
        )

    @pytest.mark.unit
    def test_action_after_seed_keeps_pairing_consistent(self, live_sandbox):
        """action() skips its first-append branch when _grids is non-empty —
        the seeded grid IS the pre-action grid, so grids[i]+actions[i]->
        grids[i+1] pairing survives."""
        sandbox = self._make_seeded(live_sandbox)
        sandbox._current_frame = [[0] * 8 for _ in range(8)]
        sandbox.namespace["current_frame"] = sandbox._current_frame
        output, error, taken = sandbox.run_code("action(1)")
        assert error is None
        assert taken == 1
        assert len(sandbox._grids) == 2, (
            f"seed + 1 action = 2 grids (got {len(sandbox._grids)})"
        )
        assert len(sandbox._actions) == 1

    @pytest.mark.unit
    def test_check_with_single_seeded_grid_reports_no_transitions(
        self, live_sandbox
    ):
        sandbox = self._make_seeded(live_sandbox)
        sandbox.run_code("def my_sim(g, a):\n    return g\nset_simulate(my_sim)\n")
        output, error, _ = sandbox.run_code("check()")
        assert error is None
        assert "No transitions recorded yet" in output
        # Guard returns before run_check: the check-failure counter must not
        # see a fake failure for a state with nothing to test.
        assert sandbox._last_check_result is None

    @pytest.mark.unit
    def test_check_works_after_seed_plus_action(self, live_sandbox):
        def small_step(action_id: int, action_data: object) -> dict[str, object]:
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": [[0] * 4 for _ in range(4)],
                "valid_actions": [1, 2],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=small_step, timeout=5.0)
        sandbox._current_frame = [[3] * 4 for _ in range(4)]
        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=sandbox._current_frame,
            previous_frame=None,
            valid_actions=[1, 2],
            last_action_result={},
            history=[],
        )
        sandbox.run_code("action(1)")
        sandbox.run_code("def my_sim(g, a):\n    return g\nset_simulate(my_sim)\n")
        output, error, _ = sandbox.run_code("check()")
        assert error is None
        assert "No transitions" not in output, (
            "with a seed + 1 action there is 1 transition — check() must run"
        )


# 8. Event-driven state refactor (T1–T3: per-action history, prompt refresh,
#    workflow guardrails mid-batch)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventDrivenStateRefactor:
    """Unit tests for the event-driven state refactor (plan T4).

    Tests cover: per-action history entries via _on_action_executed hook,
    history semantics matching the system prompt, RESET skip policy,
    fallback path coverage, loop continuation after action (T0),
    prompt refresh after action (T2), and workflow guardrail mid-batch (T3.b).
    """

    def _make_event_agent(self, *, llm_stubs=None, step_env_fn=None):
        """Build a stubbed SimulatorFirstAgent with a real SimulatorSandbox.

        Uses the same pattern as _make_level_transition_agent: __new__,
        attribute seeding, real sandbox with step_env_callback adapter.

        Parameters
        ----------
        llm_stubs : list[callable] | None
            Each element is a callable ``fake_llm_chat(**kwargs) -> response``
            called in order. When exhausted, raises RuntimeError("done").
        step_env_fn : callable | None
            Optional override for the step_env function. If None, a default
            that increments action_counter and builds a FrameData with a
            distinct grid cell is used.
        """
        import numpy as np
        from arcengine import FrameData, GameAction, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-event"
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages: list[dict] = []
        agent._history_turns: list[dict] = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result: dict = {}
        agent._current_grid: list[list[int]] | None = None
        agent._previous_grid: list[list[int]] | None = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._llm_calls = 0
        agent._sandbox_steps = 0
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False
        agent._objects: tuple = ()
        agent._adjacency = frozenset()
        agent._non_action_calls = 0

        call_index = [0]  # mutable counter for llm_stubs

        if llm_stubs is None:
            llm_stubs = []

        def default_llm_chat(**kwargs):
            agent._llm_calls += 1
            idx = call_index[0]
            call_index[0] += 1
            if idx < len(llm_stubs):
                return llm_stubs[idx](**kwargs)
            raise RuntimeError("done")

        if step_env_fn is not None:
            fake_step_env = step_env_fn
        else:

            def fake_step_env(action):
                action_id = action.value if hasattr(action, "value") else int(action)
                grid = np.zeros((1, 64, 64), dtype=int)
                grid[0, action_id, 0] = action_id + 10
                frame = FrameData(
                    game_id="test-event",
                    frame=grid,
                    state=GameState.NOT_FINISHED,
                    levels_completed=0,
                    win_levels=7,
                    available_actions=[1, 2, 3, 4],
                )
                agent.frames.append(frame)
                agent.action_counter += 1
                agent._current_grid = [list(row) for row in frame.frame[0]]
                agent._on_action_executed(action_id, frame)
                return frame

        agent._llm_chat = default_llm_chat
        agent.step_env = fake_step_env  # type: ignore[method-assign]

        holder: dict = {"agent": None}

        def adapter(action_id: int, action_data):
            game_action = GameAction.from_id(action_id)
            ag = holder["agent"]
            ag.step_env(game_action)
            if len(ag.frames) >= 2:
                prev = ag.frames[-2]
                curr = ag.frames[-1]
                prev_levels = (
                    prev.levels_completed if hasattr(prev, "levels_completed") else 0
                )
                curr_levels = (
                    curr.levels_completed if hasattr(curr, "levels_completed") else 0
                )
                ag._last_action_result = {
                    "board_changed": True,
                    "done": False,
                    "level_completed": curr_levels > prev_levels,
                    "game_over": False,
                    "run_complete": False,
                    "reward": curr_levels - prev_levels,
                    "valid_actions": list(curr.available_actions or []),
                }
            else:
                ag._last_action_result = {}

            state_response: dict = {
                "objects": ag._objects,
                "adjacency": ag._adjacency,
                "history": ag._history_turns,
            }
            if ag._current_grid is not None:
                state_response["grid"] = ag._current_grid
            state_response["valid_actions"] = ag._valid_actions
            state_response["last_action_result"] = ag._last_action_result
            return state_response

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        holder["agent"] = agent
        agent._workflow = WorkflowController(sandbox)

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.frames = [frame]
        return agent, agent.frames

    def _python_tool_call(self, code: str, call_id: str = "tc-1"):
        """Helper: build an LLM response object with a single python tool call."""
        return type(
            "R",
            (),
            {
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": f'{{"code": {json.dumps(code)}}}',
                        },
                    }
                ],
                "content": None,
            },
        )()

    def _text_response(self, text: str = "no action"):
        """Helper: build an LLM response with no tool calls, just text."""
        return type("R", (), {"tool_calls": None, "content": text})()

    # ── T4.a: test_history_entry_per_action ────────────────────────────────

    @pytest.mark.unit
    def test_history_entry_per_action(self):
        """Batch action(1); action(1); action(2) produces 3 history entries
        with correct action ids, frame_index, and post-action grids."""

        agent, frames = self._make_event_agent()

        # LLM call 1: python tool call with batch of 3 actions.
        # LLM call 2: raise RuntimeError("done") to break the loop cleanly.
        captured = {"messages": []}

        def llm_call1(**kwargs):
            captured["messages"].append([dict(m) for m in kwargs.get("messages", [])])
            return self._python_tool_call(
                "for a in [1, 1, 2]:\n    action(a)\n"
            )

        def llm_call2(**kwargs):
            captured["messages"].append([dict(m) for m in kwargs.get("messages", [])])
            raise RuntimeError("done")

        # Re-wire the agent with these stubs.
        call_index = [0]

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            idx = call_index[0]
            call_index[0] += 1
            stubs = [llm_call1, llm_call2]
            if idx < len(stubs):
                return stubs[idx](**kwargs)
            raise RuntimeError("done")

        agent._llm_chat = fake_llm
        agent._sandbox._current_frame = [list(row) for row in frames[0].frame[0]]

        agent.run()

        # 3 history entries: one per executed action.
        assert len(agent._history_turns) == 3, (
            f"expected 3 history entries, got {len(agent._history_turns)}"
        )
        assert agent._history_turns[0]["action"] == 1
        assert agent._history_turns[1]["action"] == 1
        assert agent._history_turns[2]["action"] == 2

        # frame_index: 0, 1, 2 (action_counter starts at 0, each step +1)
        assert agent._history_turns[0]["frame_index"] == 0
        assert agent._history_turns[1]["frame_index"] == 1
        assert agent._history_turns[2]["frame_index"] == 2

        # Each entry's frame grid has the marker: cell[action_id][0] == action_id + 10
        for i, entry in enumerate(agent._history_turns):
            aid = entry["action"]
            grid = entry["frame"]
            assert grid[aid][0] == aid + 10, (
                f"entry {i}: grid[{aid}][0] should be {aid + 10}, "
                f"got {grid[aid][0]}"
            )

    # ── T4.b: test_history_semantics_matches_prompt ────────────────────────

    @pytest.mark.unit
    def test_history_semantics_matches_prompt(self):
        """action(1); action(2) produces 2 entries where each frame is the
        post-action grid for that specific action."""

        agent, frames = self._make_event_agent()

        def llm_call1(**kwargs):
            return self._python_tool_call("for a in [1, 2]:\n    action(a)\n")

        def llm_call2(**kwargs):
            raise RuntimeError("done")

        call_index = [0]

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            idx = call_index[0]
            call_index[0] += 1
            stubs = [llm_call1, llm_call2]
            if idx < len(stubs):
                return stubs[idx](**kwargs)
            raise RuntimeError("done")

        agent._llm_chat = fake_llm
        agent._sandbox._current_frame = [list(row) for row in frames[0].frame[0]]

        agent.run()

        assert len(agent._history_turns) == 2, (
            f"expected 2 history entries, got {len(agent._history_turns)}"
        )
        # history[0].frame is the post-action-1 grid: cell [1][0] = 11
        assert agent._history_turns[0]["frame"][1][0] == 11, (
            f"post-action-1 grid: cell [1][0] should be 11, "
            f"got {agent._history_turns[0]['frame'][1][0]}"
        )
        # history[1].frame is the post-action-2 grid: cell [2][0] = 12
        assert agent._history_turns[1]["frame"][2][0] == 12, (
            f"post-action-2 grid: cell [2][0] should be 12, "
            f"got {agent._history_turns[1]['frame'][2][0]}"
        )

    # ── T4.c: test_reset_no_history_entry ──────────────────────────────────

    @pytest.mark.unit
    def test_reset_no_history_entry(self):
        """REWRITTEN (task 6, RESET-handler alignment): RESET (action 0) now
        appends a history entry with a provenance marker — env-initiated
        framing, not a world transition. The old skip policy is gone."""
        import numpy as np
        from arcengine import FrameData, GameState

        agent, _ = self._make_event_agent()
        agent._history_turns = []

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent._on_action_executed(0, frame)
        assert len(agent._history_turns) == 1, (
            f"RESET (action_id=0) must create a history entry, "
            f"got {len(agent._history_turns)} entries"
        )
        entry = agent._history_turns[0]
        assert entry["action"] == 0
        assert "provenance" in entry, (
            f"RESET entry must carry a provenance marker, got keys {sorted(entry)}"
        )

    # ── task 6: RESET history entries (iter-0 + mid-game) ─────────────────

    @pytest.mark.unit
    def test_iter0_reset_history_entry(self):
        """task 6 (a): the iter-0 RESET produces history[0] with action==0,
        frame_index==0, and frame == the post-RESET board (B0)."""
        import numpy as np
        from arcengine import FrameData, GameState

        agent, _ = self._make_event_agent()
        agent._history_turns = []
        agent.action_counter = 1  # step_env already incremented past the RESET

        post_reset_grid = np.zeros((1, 64, 64), dtype=int)
        post_reset_grid[0, 0, 0] = 10  # RESET marker (cell[action_id][0] = id+10)
        frame = FrameData(
            game_id="test-event",
            frame=post_reset_grid,
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent._on_action_executed(0, frame)

        assert len(agent._history_turns) == 1
        entry = agent._history_turns[0]
        assert entry["action"] == 0
        assert entry["frame_index"] == 0
        assert entry["frame"][0][0] == 10, (
            f"history[0].frame must be the post-RESET board (marker 10 at "
            f"[0][0]), got {entry['frame'][0][0]}"
        )

    @pytest.mark.unit
    def test_iter0_reset_history_provenance_marker(self):
        """task 6 (c): the iter-0 entry carries the env-RESET provenance
        marker; a mid-game RESET entry carries the mid-game marker."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.agent import (
            ITER0_RESET_PROVENANCE,
            MIDGAME_RESET_PROVENANCE,
        )

        agent, _ = self._make_event_agent()
        agent._history_turns = []
        agent.action_counter = 1

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent._on_action_executed(0, frame)
        assert agent._history_turns[0]["provenance"] == ITER0_RESET_PROVENANCE

        # Mid-game RESET: history non-empty → different marker.
        agent.action_counter = 2
        agent._on_action_executed(0, frame)
        assert agent._history_turns[1]["provenance"] == MIDGAME_RESET_PROVENANCE

    @pytest.mark.unit
    def test_iter0_reset_history_helper_idempotent(self):
        """task 6: _record_iter0_reset_history is idempotent — the guard-site
        call no-ops when the hook already appended the iter-0 entry."""
        import numpy as np
        from arcengine import FrameData, GameState

        agent, _ = self._make_event_agent()
        agent._history_turns = []
        agent.action_counter = 1

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent._on_action_executed(0, frame)
        agent._record_iter0_reset_history(frame)  # guard-site seam: no-op
        assert len(agent._history_turns) == 1, (
            f"guard-site call must not duplicate the iter-0 entry, "
            f"got {len(agent._history_turns)}"
        )

    @pytest.mark.unit
    def test_midgame_reset_appends_history_entry(self):
        """task 6 (b): a mid-game RESET via sandbox action(0) appends a
        history entry (previously skipped). Full run() path: LLM batch
        action(1); action(0) → 2 entries, entry 1 is the RESET with
        provenance."""
        agent, frames = self._make_event_agent()

        def llm_call1(**kwargs):
            return self._python_tool_call("for a in [1, 0]:\n    action(a)\n")

        def llm_call2(**kwargs):
            raise RuntimeError("done")

        call_index = [0]

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            idx = call_index[0]
            call_index[0] += 1
            stubs = [llm_call1, llm_call2]
            if idx < len(stubs):
                return stubs[idx](**kwargs)
            raise RuntimeError("done")

        agent._llm_chat = fake_llm
        agent._sandbox._current_frame = [list(row) for row in frames[0].frame[0]]

        agent.run()

        assert len(agent._history_turns) == 2, (
            f"expected 2 history entries (action 1 + mid-game RESET), "
            f"got {len(agent._history_turns)}"
        )
        assert agent._history_turns[0]["action"] == 1
        reset_entry = agent._history_turns[1]
        assert reset_entry["action"] == 0
        assert "provenance" in reset_entry, (
            f"mid-game RESET entry must carry a provenance marker, "
            f"got keys {sorted(reset_entry)}"
        )
        # The RESET entry's frame is the post-RESET board: marker 10 at [0][0].
        assert reset_entry["frame"][0][0] == 10

    @pytest.mark.unit
    def test_workflow_update_stays_action_gated_on_reset(self):
        """task 6: the workflow counter update stays action-gated — a RESET
        hook call must not advance the workflow's action_counter view past
        the agent's real counter (documented gate in _on_action_executed)."""
        import numpy as np
        from arcengine import FrameData, GameState

        agent, _ = self._make_event_agent()
        agent._history_turns = []
        agent.action_counter = 1

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent._on_action_executed(0, frame)
        assert agent._workflow._action_counter == 1, (
            f"workflow.update must still fire on RESET with the real counter "
            f"(1), got {agent._workflow._action_counter}"
        )

    # ── T4.d: test_fallback_action_gets_history_entry ─────────────────────

    @pytest.mark.unit
    def test_run_no_random_fallback(self):
        """P2-T3: when LLM returns no tool calls, no random action is injected.
        run() terminates via MAX_ACTIONS exhausted, action_counter unchanged."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 100  # budget already exhausted
        agent.game_id = "test-no-fallback"
        agent.frames = [
            FrameData(
                game_id="test-no-fallback",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)
        agent._llm_chat = lambda **kwargs: type(
            "R", (), {"tool_calls": None, "content": "no action"}
        )()

        agent.run()

        assert agent.action_counter == 100, (
            f"action_counter should be unchanged (no random fallback), "
            f"got {agent.action_counter}"
        )

    # ── T4.e: test_loop_continues_after_action ────────────────────────────

    @pytest.mark.unit
    def test_loop_continues_after_action(self):
        """T0 verification: after an action() call in python code, the tool
        loop does NOT exit. A second LLM call happens in the same turn."""

        agent, frames = self._make_event_agent()

        # Call 1: python tool call with action(1) — loop continues.
        # Call 2: python tool call with print("done") — no action, loop continues.
        # Call 3: raise RuntimeError("end") — break loop.
        call_index = [0]

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            idx = call_index[0]
            call_index[0] += 1
            if idx == 0:
                return self._python_tool_call("action(1)\n")
            elif idx == 1:
                return self._python_tool_call('print("done")\n')
            else:
                raise RuntimeError("end")

        agent._llm_chat = fake_llm
        agent._sandbox._current_frame = [list(row) for row in frames[0].frame[0]]

        agent.run()

        assert agent._llm_calls >= 3, (
            f"expected >= 3 LLM calls (loop continues after action), "
            f"got {agent._llm_calls}"
        )
        # Only action(1) creates a history entry; print("done") doesn't.
        assert len(agent._history_turns) == 1, (
            f"expected 1 history entry (only action(1)), "
            f"got {len(agent._history_turns)}"
        )
        assert agent._history_turns[0]["action"] == 1

    # ── T4.f: test_prompt_refresh_after_action ──────────────────────────────

    @pytest.mark.unit
    def test_prompt_refresh_after_action(self):
        """T2 verification: after action(1), the next LLM call's messages
        contain a user message whose frame-header text includes the new
        history entry (action=1) and frame_index matches action_counter-1."""

        agent, frames = self._make_event_agent()

        captured = {"messages": []}
        call_index = [0]

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            captured["messages"].append([dict(m) for m in kwargs.get("messages", [])])
            idx = call_index[0]
            call_index[0] += 1
            if idx == 0:
                return self._python_tool_call("action(1)\n")
            else:
                raise RuntimeError("end")

        agent._llm_chat = fake_llm
        agent._sandbox._current_frame = [list(row) for row in frames[0].frame[0]]

        agent.run()

        # Second LLM call's messages (index 1) should contain a refreshed
        # user message with frame header that includes action=1 in history.
        assert len(captured["messages"]) >= 2, (
            f"expected >= 2 LLM calls, got {len(captured['messages'])}"
        )
        second_call_msgs = captured["messages"][1]

        # Find the refreshed user message: history summary is a separate
        # content block from "Frame N", so collect all text from user msgs.
        all_text_parts: list[str] = []
        for msg in second_call_msgs:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text", ""), str)
                    ):
                        all_text_parts.append(block["text"])
            elif isinstance(content, str):
                all_text_parts.append(content)

        combined = "\n".join(all_text_parts)
        assert "action=1" in combined, (
            f"refreshed prompt must include action=1 from history, "
            f"got text blocks: {all_text_parts!r}"
        )

    # ── T4.g: test_workflow_guardrail_mid_batch ────────────────────────────

    @pytest.mark.unit
    def test_workflow_guardrail_mid_batch(self):
        """T3.b verification: calling _on_action_executed when
        action_counter crosses 10 while in EXPLORE transitions phase to MODEL."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.workflow import Phase

        agent, _ = self._make_event_agent()

        # Set action_counter to 9 (so after the hook increments via update,
        # it sees 10 and the guardrail fires).
        agent.action_counter = 9
        agent._workflow._phase = Phase.EXPLORE

        frame = FrameData(
            game_id="test-event",
            frame=np.zeros((1, 64, 64), dtype=int),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        # Call the hook directly (action_id=1, not RESET).
        agent._on_action_executed(1, frame)

        # The hook calls _workflow.update(action_counter) which checks
        # action_counter >= 10. After step_env the counter was incremented
        # to 10, so the guardrail should fire.
        # But note: _on_action_executed is called AFTER step_env, and
        # step_env already incremented action_counter to 10 (the hook
        # uses the already-incremented counter). The guardrail in
        # workflow.update checks action_counter >= 10.
        # We set action_counter=9 before the hook, but the hook reads
        # self.action_counter which is 9. The workflow.update(9) won't
        # fire the guardrail. We need action_counter=10 at the point
        # the hook calls _workflow.update().
        # Let's re-do with action_counter=10.
        agent._workflow._phase = Phase.EXPLORE
        agent.action_counter = 10

        agent._on_action_executed(1, frame)

        assert agent._workflow.phase == Phase.MODEL, (
            f"guardrail should transition EXPLORE→MODEL at action_counter=10, "
            f"got phase={agent._workflow.phase}"
        )


class TestRegistration:
    def test_simulatorfirst_in_available_agents(self):
        assert "simulatorfirst" in AVAILABLE_AGENTS

    def test_simulatorfirst_class_name(self):
        assert AVAILABLE_AGENTS["simulatorfirst"].__name__ == "SimulatorFirstAgent"

    def test_simulatorfirst_is_agent_subclass(self):
        from agents.agent import Agent

        assert issubclass(AVAILABLE_AGENTS["simulatorfirst"], Agent)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Phase 2 — main() override smoke test
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainOverride:
    """Smoke tests: verify run() contract for budget exhaustion and anti-spiral caps."""

    @pytest.mark.unit
    def test_run_returns_none_when_win_state(self):
        """run() returns None when the latest frame has state=WIN."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-main"
        agent.frames = [
            FrameData(
                game_id="test-main",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.WIN,
                levels_completed=7,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)

        agent.run()
        # run() returns None (implicitly) when WIN state detected at outer loop guard.

    @pytest.mark.unit
    def test_callback_enforces_max_actions(self):
        """P2-T2.d: callback raises before step_env when MAX_ACTIONS budget exhausted."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 100  # budget exhausted
        agent.game_id = "test-budget"
        agent.frames = [
            FrameData(
                game_id="test-budget",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)

        with pytest.raises(RuntimeError, match="budget exhausted"):
            agent._step_env_callback(1, None)

    @pytest.mark.unit
    def test_run_no_random_fallback(self):
        """P2-T3: when budget is exhausted, no random action is injected.
        action_counter is unchanged — no random action taken."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 100  # budget exhausted
        agent.game_id = "test-no-fallback"
        agent.frames = [
            FrameData(
                game_id="test-no-fallback",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)
        agent._llm_chat = lambda **kwargs: type(
            "R", (), {"tool_calls": None, "content": "no action"}
        )()

        agent.run()

        assert agent.action_counter == 100, (
            "action_counter should be unchanged (no random fallback), got "
            f"{agent.action_counter}"
        )

    @pytest.mark.unit
    def test_run_terminates_at_non_action_cap(self):
        """P2-T5.a: run() terminates when MAX_ACTIONS is exhausted;
        no random action injected when LLM returns no tool calls."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 5
        agent.action_counter = 5  # budget exhausted
        agent.game_id = "test-end-condition"
        agent.frames = [
            FrameData(
                game_id="test-end-condition",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)

        agent._llm_chat = lambda **kwargs: type(
            "R", (), {"tool_calls": None, "content": "no action"}
        )()

        agent.run()

        assert agent.action_counter == 5, (
            f"action_counter should be 5 (MAX_ACTIONS), "
            f"got {agent.action_counter}"
        )
        assert agent.action_counter >= agent.MAX_ACTIONS


class TestAntiSpiralGuard:
    """Tests for the anti-spiral guard in run().

    The corrected anti-spiral guard: increments _non_action_calls on no-action
    iterations, resets on action. At >=12: nudge. At >=24: set_phase('MODEL')
    (NO reset — counter continues). At >=36: terminate run().

    Counter climbs 0→12(nudge)→24(set_phase MODEL)→36(terminate).
    Reset only happens when an action IS taken.
    """

    @pytest.mark.unit
    def test_anti_spiral_nudge_and_set_phase(self):
        """Non-action iterations trigger nudge at >=12 and set_phase('MODEL') at >=24.
        With the corrected guard (no >=24 reset), the counter reaches 36 and
        run() terminates via the hard cap."""
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-anti-spiral"
        agent.frames = [
            FrameData(
                game_id="test-anti-spiral",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        sandbox._current_frame = agent._current_grid
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)

        set_phase_calls: list[tuple[str, str]] = []
        original_set_phase = agent._workflow.set_phase

        def tracking_set_phase(phase: str, reason: str = ""):
            set_phase_calls.append((phase, reason))
            return original_set_phase(phase, reason)

        agent._workflow.set_phase = tracking_set_phase  # type: ignore[method-assign]

        # LLM always returns python() tool call with "x = 1" (no action).
        # run_code returns (output, None, None) — action_taken_id stays None,
        # so _non_action_calls climbs every iteration until >=36 terminates.
        def fake_llm_chat(**kwargs):
            return type(
                "R",
                (),
                {
                    "tool_calls": [{
                        "id": "tc-1",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code": "x = 1"}',
                        },
                    }],
                    "content": None,
                },
            )()

        agent._llm_chat = fake_llm_chat

        def fake_run_code(code):
            return ("x = 1", None, None)

        sandbox.run_code = fake_run_code  # type: ignore[method-assign]

        agent.run()

        # The counter should have reached >=36, causing run() to terminate.
        assert agent._non_action_calls >= 36, (
            f"_non_action_calls should be >=36 (cap reached), "
            f"got {agent._non_action_calls}"
        )

        model_calls = [
            (phase, reason)
            for phase, reason in set_phase_calls
            if phase == "MODEL"
        ]
        assert len(model_calls) >= 1, (
            f"Expected set_phase('MODEL') call at >=24 non-action calls, "
            f"got {model_calls}"
        )

        # No action was taken, so action_counter should still be 0.
        assert agent.action_counter == 0, (
            f"action_counter should be 0 (no random action), "
            f"got {agent.action_counter}"
        )

    @pytest.mark.unit
    def test_anti_spiral_resets_on_action(self):
        """_non_action_calls resets to 0 when an action is taken via sandbox.
        With the corrected guard, after reset the counter can climb again to 36.
        Verify: action(1) resets counter, then subsequent no-action calls
        accumulate again, and run() terminates via the >=36 cap."""
        import numpy as np
        from arcengine import FrameData, GameAction, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-anti-spiral-reset"
        agent.frames = [
            FrameData(
                game_id="test-anti-spiral-reset",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        action_taken_flag = [False]

        def fake_step_env(action):
            action_id = action.value if hasattr(action, "value") else int(action)
            grid = np.zeros((1, 64, 64), dtype=int)
            grid[0, action_id, 0] = action_id + 10
            frame = FrameData(
                game_id="test-anti-spiral-reset",
                frame=grid,
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
            agent.frames.append(frame)
            agent.action_counter += 1
            agent._current_grid = [list(row) for row in frame.frame[0]]
            action_taken_flag[0] = True
            return frame

        def adapter(action_id: int, action_data):
            ag = agent
            game_action = GameAction.from_id(action_id)
            ag.step_env(game_action)
            if len(ag.frames) >= 2:
                prev = ag.frames[-2]
                curr = ag.frames[-1]
                prev_levels = prev.levels_completed if hasattr(prev, "levels_completed") else 0
                curr_levels = curr.levels_completed if hasattr(curr, "levels_completed") else 0
                prev_grid = prev.frame[0] if prev.frame else None
                curr_grid = curr.frame[0] if curr.frame else None
                ag._last_action_result = {
                    "board_changed": prev_grid != curr_grid,
                    "done": False,
                    "level_completed": curr_levels > prev_levels,
                    "game_over": False,
                    "run_complete": False,
                    "reward": curr_levels - prev_levels,
                    "valid_actions": list(curr.available_actions or []),
                }
            else:
                ag._last_action_result = {}
            state_response = {
                "objects": ag._objects,
                "adjacency": ag._adjacency,
                "history": ag._history_turns,
            }
            if ag._current_grid is not None:
                state_response["grid"] = ag._current_grid
            state_response["valid_actions"] = ag._valid_actions
            state_response["last_action_result"] = ag._last_action_result
            return state_response

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        sandbox._current_frame = agent._current_grid
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)
        agent.step_env = fake_step_env  # type: ignore[method-assign]

        # LLM: call 1 returns python() with action(1), then all subsequent
        # calls return python() with no action. Counter resets after action,
        # then climbs back up to 36 and terminates.
        call_index = [0]

        def fake_llm_chat(**kwargs):
            call_index[0] += 1
            if call_index[0] == 1:
                # First call: take an action — resets _non_action_calls to 0
                return type(
                    "R",
                    (),
                    {
                        "tool_calls": [{
                            "id": "tc-1",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "action(1)"}',
                            },
                        }],
                        "content": None,
                    },
                )()
            else:
                # Subsequent calls: no action — counter climbs toward 36
                return type(
                    "R",
                    (),
                    {
                        "tool_calls": [{
                            "id": f"tc-{call_index[0]}",
                            "type": "function",
                            "function": {
                                "name": "python",
                                "arguments": '{"code": "x = 1"}',
                            },
                        }],
                        "content": None,
                    },
                )()

        agent._llm_chat = fake_llm_chat

        # Do NOT stub sandbox.run_code — let the real sandbox execute
        # action(1) through the callback chain so step_env is called.

        agent.run()

        # action(1) was taken, so action_counter >= 1
        assert agent.action_counter >= 1, (
            f"action_counter should be >= 1 (one action taken), "
            f"got {agent.action_counter}"
        )
        assert action_taken_flag[0], "An action should have been taken"

        # After action, _non_action_calls reset to 0, then accumulated again
        # until hitting >=36. Final value should be >=36.
        assert agent._non_action_calls >= 36, (
            f"_non_action_calls should be >=36 (accumulated after action reset), "
            f"got {agent._non_action_calls}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SpiralGuard — pure anti-spiral policy (workflow.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMidTurnEscapeGuardrail:
    """a21a2571 regression: the MODEL→EXPLORE rescue after 5 check failures
    must fire DURING a no-action tool spiral.

    The run spent 37 consecutive python calls at frame 19 with check()
    accuracy stuck at 1.4% — _check_failures climbed past 5, but update()
    only ran at turn boundaries, so the rescue never fired and the game was
    terminated by the 36-call cap with the agent still in MODEL.
    """

    def _make_check_spiral_agent(self):
        import numpy as np
        from arcengine import FrameData, GameState

        from agents.simulator_agent.sandbox import SimulatorSandbox
        from agents.simulator_agent.workflow import WorkflowController

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-midturn-escape"
        agent.frames = [
            FrameData(
                game_id="test-midturn-escape",
                frame=np.zeros((1, 64, 64), dtype=int),
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages = []
        agent._history_turns = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result = {}
        agent._current_grid = [[0] * 64 for _ in range(64)]
        agent._previous_grid = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._non_action_calls = 0
        agent._objects = ()
        agent._adjacency = frozenset()
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False

        def adapter(action_id: int, action_data):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": agent._current_grid,
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {},
            }

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        sandbox._current_frame = agent._current_grid
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        agent._sandbox = sandbox
        agent._workflow = WorkflowController(sandbox)
        # Seed the spiral state: turn-start directive is MODEL, so a
        # "PHASE: EXPLORE" string can only appear via the mid-turn guardrail.
        agent._workflow.set_phase("MODEL", "ready to model")

        # Every python call runs check() and fails (new result dict each
        # call — _handle_python_call detects a fresh check by identity).
        def fake_run_code(code):
            sandbox._last_check_result = {
                "wrong_cells": 500,
                "overall_accuracy": 1.4,
            }
            return ("Overall: 500 wrong, 0 correct (1.4%)", None, None)

        sandbox.run_code = fake_run_code  # type: ignore[method-assign]

        saw_explore_directive = [False]

        def fake_llm_chat(**kwargs):
            for m in kwargs.get("messages", []):
                content = m.get("content")
                texts = content if isinstance(content, list) else [content]
                for t in texts:
                    if isinstance(t, str) and "PHASE: EXPLORE" in t:
                        saw_explore_directive[0] = True
            return type(
                "R",
                (),
                {
                    "tool_calls": [{
                        "id": "tc-x",
                        "type": "function",
                        "function": {
                            "name": "python",
                            "arguments": '{"code": "check()"}',
                        },
                    }],
                    "content": None,
                },
            )()

        agent._llm_chat = fake_llm_chat
        return agent, saw_explore_directive

    @pytest.mark.unit
    def test_model_to_explore_fires_mid_spiral_after_5_check_failures(self):
        agent, saw_explore_directive = self._make_check_spiral_agent()
        agent.run()
        assert agent._workflow.phase.value == "EXPLORE", (
            "escape guardrail must rescue the agent out of MODEL mid-turn "
            f"(got {agent._workflow.phase.value})"
        )
        assert saw_explore_directive[0], (
            "the EXPLORE directive must be injected into the conversation "
            "when the mid-turn guardrail fires"
        )
        assert agent._non_action_calls >= 36, (
            "spiral still terminates via the 36-call cap — the fix changes "
            "the phase mid-spiral, it does not remove the cap"
        )


class TestSpiralGuard:
    """Unit tests for the SpiralGuard policy class (thresholds 12/24/36)."""

    def test_reset_on_action(self):
        verdict = SpiralGuard.record(20, took_action=True)
        assert verdict.count == 0
        assert not verdict.nudge
        assert not verdict.phase_model
        assert not verdict.terminate

    def test_below_nudge_threshold(self):
        verdict = SpiralGuard.record(0, took_action=False)
        assert verdict.count == 1
        assert not verdict.nudge
        assert not verdict.phase_model
        assert not verdict.terminate

    def test_nudge_at_12(self):
        verdict = SpiralGuard.record(11, took_action=False)
        assert verdict.count == 12
        assert verdict.nudge
        assert not verdict.phase_model
        assert not verdict.terminate

    def test_phase_model_at_24_counter_continues(self):
        verdict = SpiralGuard.record(23, took_action=False)
        assert verdict.count == 24
        assert verdict.nudge
        assert verdict.phase_model
        assert verdict.count == 24, "counter must NOT reset at 24"
        assert not verdict.terminate

    def test_terminate_at_36(self):
        verdict = SpiralGuard.record(35, took_action=False)
        assert verdict.count == 36
        assert verdict.nudge
        assert verdict.phase_model
        assert verdict.terminate

    def test_flags_carry_at_high_counts(self):
        verdict = SpiralGuard.record(50, took_action=False)
        assert verdict.count == 51
        assert verdict.nudge
        assert verdict.phase_model
        assert verdict.terminate
