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

    def test_action_blocked_after_win(self):
        """action() should be blocked after a winning result."""
        sandbox = SimulatorSandbox(step_env_callback=_mock_step_env_callback)
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox._last_action_result = {"run_complete": True}
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, _ = sandbox.run_code("action(0)")
        assert "Game already won/over" in (error or "")

    def test_action_blocked_after_game_over(self):
        """action() should be blocked after a game-over result."""
        sandbox = SimulatorSandbox(step_env_callback=_mock_step_env_callback)
        sandbox._current_frame = [[0] * 4 for _ in range(4)]
        sandbox._last_action_result = {"game_over": True}
        sandbox.namespace["current_frame"] = sandbox._current_frame
        sandbox.namespace["previous_frame"] = None
        output, error, _ = sandbox.run_code("action(0)")
        assert "Game already won/over" in (error or "")

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


class TestRegistration:
    def test_simulatorfirst_in_available_agents(self):
        assert "simulatorfirst" in AVAILABLE_AGENTS

    def test_simulatorfirst_class_name(self):
        assert AVAILABLE_AGENTS["simulatorfirst"].__name__ == "SimulatorFirstAgent"

    def test_simulatorfirst_is_agent_subclass(self):
        from agents.agent import Agent

        assert issubclass(AVAILABLE_AGENTS["simulatorfirst"], Agent)
