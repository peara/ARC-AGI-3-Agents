"""Unit tests for the [Simulator state] block builder.

Incident 5681a14a (ls20, 2026-09-10): the model never saw its own sandbox
state — 29 LLM calls / ~25 min in one turn, 4 simulate rewrites, and a
self-inflicted ignore-mask shrink (128 → 98 cells) it could not observe.
``SimulatorFirstAgent._build_sim_state_block`` is the foundation of the fix:
a pure function of sandbox state that renders the registered simulate source
and the active ignore mask for injection into the LLM context.

Covers:
- skip-when-empty (RATIFIED): no simulate + empty mask → None
- MagicMock robustness (B1): isinstance-guarded reads never raise
- full source verbatim rendering + "(source unavailable)" fallback
- deterministic, sorted ignore-mask rendering (LM Studio prefix-cache
  stability) with the 6-group cap and hidden-cell remainder
"""

from __future__ import annotations

import itertools
import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.simulator_agent.agent import SIM_STATE_HEADER, SimulatorFirstAgent

_HEADER = f"{SIM_STATE_HEADER} (authoritative; refreshed every call)"


def _make_agent(sandbox: Any) -> SimulatorFirstAgent:
    """Build an agent via __new__ with only _sandbox seeded (builder-only)."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent._sandbox = sandbox
    return agent


def _make_fake_sandbox(
    simulate: Any = None,
    source: Any = "",
    mask: Any = None,
) -> Any:
    """Lightweight fake sandbox: plain attributes, no MagicMock auto-attrs."""

    class _FakeSandbox:
        pass

    s = _FakeSandbox()
    s._simulate = simulate
    s._simulate_source = source
    s._ignore_mask = mask if mask is not None else set()
    return s


@pytest.mark.unit
def test_empty_state_returns_none() -> None:
    """No simulate (None) + empty set mask → skip-when-empty (RATIFIED)."""
    agent = _make_agent(_make_fake_sandbox(simulate=None, source="", mask=set()))
    assert agent._build_sim_state_block() is None


@pytest.mark.unit
def test_magicmock_sandbox_never_raises() -> None:
    """B1: MagicMock auto-attrs are truthy and non-str/set — the isinstance
    guards must degrade them to absent instead of raising, and no mock repr
    may leak into the block."""
    agent = _make_agent(MagicMock())
    result = agent._build_sim_state_block()
    assert result is None or isinstance(result, str)
    if result is not None:
        assert "MagicMock" not in result


@pytest.mark.unit
def test_full_source_rendered_verbatim() -> None:
    """Registered simulate with a real source: header + registration line +
    FULL untruncated source body, block ≤ 4 KB."""
    source = "def simulate(grid, action):\n" + "    # pad\n" * 100 + "    return grid\n"
    assert len(source) > 900  # ~1 KB fixture
    agent = _make_agent(_make_fake_sandbox(simulate=lambda g, a: g, source=source))
    block = agent._build_sim_state_block()
    assert block is not None
    assert block.startswith(_HEADER)
    assert "simulate: registered (frozen copy — helpers fixed at registration)" in block
    assert source in block
    assert len(block) <= 4096


@pytest.mark.unit
def test_source_unavailable_fallback() -> None:
    """Offline corpora carry the "(source unavailable)" sentinel — the block
    must show the fallback line and never leak the sentinel string."""
    agent = _make_agent(
        _make_fake_sandbox(simulate=lambda g, a: g, source="(source unavailable)")
    )
    block = agent._build_sim_state_block()
    assert block is not None
    assert "simulate: registered (source not captured this session)" in block
    assert "(source unavailable)" not in block


@pytest.mark.unit
def test_two_single_cells_canonical() -> None:
    """Incident 5681a14a seq-44 shape: two isolated cells render canonically
    and byte-stable across calls (no set-hash order leak)."""
    mask = {(61, 19), (62, 19)}
    first = SimulatorFirstAgent._render_ignore_mask(mask)
    second = SimulatorFirstAgent._render_ignore_mask(mask)
    assert first == "2 cells — row 61, col 19; row 62, col 19"
    assert first == second


@pytest.mark.unit
def test_full_width_band() -> None:
    """128-cell contiguous band (rows 61-62, all cols) renders as two
    full-width row groups."""
    mask = {(row, col) for row in (61, 62) for col in range(64)}
    assert SimulatorFirstAgent._render_ignore_mask(mask) == (
        "128 cells — row 61, cols 0-63; row 62, cols 0-63"
    )


@pytest.mark.unit
def test_scattered_cap() -> None:
    """8 groups × 50 cells: 6 groups shown, remainder summarized by CELL
    count (not group count) — '+100 cells in 2 more groups'."""
    mask = {(g, c) for g in range(8) for c in range(50)}
    rendered = SimulatorFirstAgent._render_ignore_mask(mask)
    assert rendered.startswith("400 cells — ")
    assert "+100 cells in 2 more groups" in rendered
    assert rendered.count(";") == 6  # 6 shown groups + remainder line
    block = _make_agent(
        _make_fake_sandbox(simulate=lambda g, a: g, source="", mask=mask)
    )._build_sim_state_block()
    assert block is not None
    assert len(block) <= 4096


@pytest.mark.unit
def test_deterministic_rendering() -> None:
    """Prefix-cache stability: identical mask → byte-identical rendering,
    regardless of set iteration order."""
    mask = {(row, (row * 7 + col * 13) % 64) for row in range(64) for col in range(5)}
    assert len(mask) >= 200
    first = SimulatorFirstAgent._render_ignore_mask(mask)
    second = SimulatorFirstAgent._render_ignore_mask(mask)
    assert first == second


@pytest.mark.unit
def test_single_row_two_col_runs() -> None:
    """A row with two non-contiguous col-runs must render 'row 61' — the
    grouping logic appends the same row twice, so len(rows)==1 is wrong."""
    mask = {(61, 5), (61, 20), (61, 21)}
    assert SimulatorFirstAgent._render_ignore_mask(mask) == (
        "3 cells — row 61, col 5, cols 20-21"
    )


@pytest.mark.unit
def test_mask_only_state() -> None:
    """Mask-only state (simulate never registered) still shows the block:
    'simulate: not registered' + the ignore line."""
    agent = _make_agent(
        _make_fake_sandbox(simulate=None, source="", mask={(61, 19)})
    )
    block = agent._build_sim_state_block()
    assert block is not None
    assert "simulate: not registered" in block
    assert "ignore: 1 cells — row 61, col 19" in block


# ── Injection hook tests (Task 2) ────────────────────────────────────────
#
# These drive the REAL _run_tool_loop with a scripted fake LLM
# (test_agent.py:1002-1069 _make_budget_exhausted_agent pattern: __new__
# construction + attribute seeding, no mocking of the loop under test).


def _make_tool_loop_agent(sandbox: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Build an agent via __new__ seeded for _run_tool_loop, mirroring
    test_agent.py:_make_budget_exhausted_agent (only the attributes the
    tool loop actually touches). Returns (agent, snapshots) where snapshots
    collects one deep-ish copy of `messages` per _llm_chat call."""
    from arcengine import FrameData, GameState

    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.MAX_ACTIONS = 30
    agent.action_counter = 0
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
    agent._board_reset_ended_turn = False

    agent._sandbox = sandbox
    agent._workflow = MagicMock()
    agent._workflow.directive.return_value = ""
    agent._workflow.apply_escape_guardrails.return_value = False
    agent._workflow.on_check_result = MagicMock()
    agent._workflow.on_bfs_result = MagicMock()
    agent._workflow.on_exception_flow = MagicMock()
    agent._workflow.update = MagicMock()
    agent.step_env = MagicMock(return_value=None)  # type: ignore[method-assign]

    frame = FrameData(
        game_id="test",
        frame=np.zeros((1, 64, 64), dtype=int),
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=7,
        available_actions=[1, 2, 3, 4],
    )
    agent.frames = [frame]

    snapshots: list[list[dict[str, Any]]] = []
    return agent, snapshots


def _wire_fake_llm(
    agent: Any,
    snapshots: list[list[dict[str, Any]]],
    script: list[Any],
) -> None:
    """Attach a scripted fake _llm_chat that snapshots messages per call.

    ``script`` entries are either an exception instance (raised) or a
    response factory called with no args. The script is consumed in order;
    when exhausted, the last entry repeats (loop keeps calling until an
    end condition fires — tests must bound the loop themselves).
    """

    def fake_llm_chat(**kwargs: Any) -> Any:
        agent._llm_calls += 1
        snapshots.append([dict(m) for m in kwargs["messages"]])
        entry = script[min(agent._llm_calls - 1, len(script) - 1)]
        if isinstance(entry, Exception):
            raise entry
        return entry()

    agent._llm_chat = fake_llm_chat


def _python_response(code: str, content: str = "running") -> Any:
    """Build an LLM response carrying one `python` tool call."""
    return lambda: type(
        "R",
        (),
        {
            "tool_calls": [
                {
                    "id": f"tc{next(_tc_counter)}",
                    "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                }
            ],
            "content": content,
        },
    )()


_tc_counter = itertools.count(1)

def _bind_budget_end(sandbox: Any, agent: Any) -> None:
    """Wire the fake sandbox so the FIRST action() call ends the turn via the
    mid-turn budget guard: bump agent.action_counter to MAX_ACTIONS when
    run_code sees action(. Mirrors production budget semantics (the loop's
    action branch checks action_counter >= MAX_ACTIONS and returns) without
    needing step_env frame advancement in the fixture."""
    original_run_code = sandbox.run_code

    def run_code_with_budget_end(code: str) -> tuple[str, str | None, int | None]:
        result = original_run_code(code)
        if "action(" in code:
            agent.action_counter = agent.MAX_ACTIONS
        return result

    sandbox.run_code = run_code_with_budget_end  # type: ignore[method-assign]




def _blocks_in(messages: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """All (index, content) of sim-state blocks: role user + str + header."""
    return [
        (i, m["content"])
        for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(SIM_STATE_HEADER)
    ]


def _make_recording_sandbox() -> Any:
    """Fake sandbox whose run_code() executes set_ignore(...) calls against
    plain-attribute state (mirrors sandbox.set_ignore semantics: REPLACE).
    action() codes return action_taken=1 so the tool loop treats them as
    actions (SpiralGuard reset, budget guard, prompt refresh)."""

    class _RecordingSandbox:
        def __init__(self) -> None:
            self._simulate = None
            self._simulate_source = ""
            self._ignore_mask: set[tuple[int, int]] = set()
            self._pending_notes: dict[str, str] = {}
            self._pending_exception_flow = None
            self._transition_pending = False
            self._action_taken = None
            self._last_check_result = None
            self._last_bfs_result = None
            self.pending_images: list[dict[str, Any]] = []
            self.executed: list[str] = []

        def run_code(self, code: str) -> tuple[str, str | None, int | None]:
            self.executed.append(code)
            if "action(" in code:
                self._action_taken = 1
                return ("ok", None, 1)
            for name, cells in (
                ("set_ignore({(61, 19), (62, 19)})", {(61, 19), (62, 19)}),
                ("set_ignore({(61, 19)})", {(61, 19)}),
            ):
                if name in code:
                    self._ignore_mask = set(cells)
            return ("ok", None, None)

    return _RecordingSandbox()


@pytest.mark.unit
def test_hook_injects_at_index1_and_refreshes() -> None:
    """Real _run_tool_loop, scripted fake LLM: call1 set_ignore(2 cells),
    call2 set_ignore(1 cell), call3 action(1). Every call's messages[1] is
    the block; exactly one block per snapshot; call3's block reflects
    call2's mask (refresh works — the incident's blind spot)."""
    sandbox = _make_recording_sandbox()
    agent, snapshots = _make_tool_loop_agent(sandbox)
    _bind_budget_end(sandbox, agent)
    _wire_fake_llm(
        agent,
        snapshots,
        [
            _python_response("set_ignore({(61, 19), (62, 19)})"),
            _python_response("set_ignore({(61, 19)})"),
            _python_response("action(1)"),
        ],
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [{"type": "text", "text": "Frame 0"}]},
    ]
    agent._run_tool_loop(messages, grid_b64="")

    assert len(snapshots) == 3, f"expected 3 LLM calls, got {len(snapshots)}"
    # Call 1: stateless at injection time (set_ignore runs AFTER the call) —
    # skip-when-empty means NO block yet.
    assert _blocks_in(snapshots[0]) == [], "call 1 is stateless: no block expected"
    for call_idx, snap in enumerate(snapshots[1:], start=2):
        blocks = _blocks_in(snap)
        assert len(blocks) == 1, (
            f"call {call_idx}: expected exactly one block, got {len(blocks)}"
        )
        idx, content = blocks[0]
        assert idx == 1, f"call {call_idx}: block at index {idx}, expected 1"
        assert snap[1]["role"] == "user"
        assert isinstance(snap[1]["content"], str)
        assert snap[1]["content"].startswith(SIM_STATE_HEADER)

    assert "ignore: 2 cells" in snapshots[1][1]["content"], (
        "call 2 must reflect call 1's 2-cell mask"
    )
    assert "ignore: 1 cells" in snapshots[2][1]["content"], (
        "call 3 must reflect call 2's 1-cell mask (refresh)"
    )
    assert "ignore: 2 cells" not in snapshots[2][1]["content"]


@pytest.mark.unit
def test_hook_no_block_when_stateless() -> None:
    """Fresh sandbox, no simulate, no mask: the hook must be a no-op —
    no block anywhere, and the message list byte-identical to the
    pre-change shape (system, frame user, then loop-appended messages)."""
    sandbox = _make_recording_sandbox()
    sandbox._ignore_mask = set()
    agent, snapshots = _make_tool_loop_agent(sandbox)
    _bind_budget_end(sandbox, agent)
    _wire_fake_llm(
        agent,
        snapshots,
        [_python_response("action(1)")],
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [{"type": "text", "text": "Frame 0"}]},
    ]
    expected_first_snapshot = [dict(m) for m in messages]
    agent._run_tool_loop(messages, grid_b64="")

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert _blocks_in(snap) == [], "no block may appear when state is empty"
    assert snap == expected_first_snapshot, (
        "stateless prompt must be byte-identical to the pre-change shape"
    )
    assert _blocks_in(messages) == []


@pytest.mark.unit
def test_hook_overflow_retry_keeps_block() -> None:
    """First _llm_chat raises a context_length exception; the retry path
    (continue → loop top → trims → hook) must produce exactly one block at
    messages[1] on the successful call."""
    sandbox = _make_recording_sandbox()
    sandbox._ignore_mask = {(61, 19), (62, 19)}
    agent, snapshots = _make_tool_loop_agent(sandbox)
    _bind_budget_end(sandbox, agent)
    # Shrink the budget so the overflow-retry trim actually drops history
    # (62-message history ~ 620 estimated tokens -> budget 300 forces drops).
    agent._context_budget_tokens = 300
    _wire_fake_llm(
        agent,
        snapshots,
        [
            Exception("Request failed: context_length exceeded"),
            _python_response("action(1)"),
        ],
    )

    # Tiny history can't shrink (len(trimmed) == len(messages)) so the retry
    # path would bail — pad with droppable history so the overflow-retry
    # continue actually fires.
    messages: list[dict[str, Any]] = [{"role": "system", "content": "system prompt"}]
    for i in range(30):
        messages.append({"role": "user", "content": f"old frame {i}"})
        messages.append({"role": "assistant", "content": f"old reply {i}"})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Frame 0"}]})

    agent._run_tool_loop(messages, grid_b64="")

    assert len(snapshots) == 2, f"expected 2 calls (fail + retry), got {len(snapshots)}"
    assert len(_blocks_in(snapshots[0])) == 1, (
        "the failed call must still have seen the block (injected before the raise)"
    )
    blocks = _blocks_in(snapshots[1])
    assert len(blocks) == 1, (
        f"retry call: expected exactly one block, got {len(blocks)}"
    )
    idx, _ = blocks[0]
    assert idx == 1
    assert snapshots[1][1]["content"].startswith(SIM_STATE_HEADER)


@pytest.mark.unit
def test_hook_position_always_index1() -> None:
    """Large history forcing trim_messages_for_context drops: the block
    must remain at messages[1] in every snapshot (never drift, never
    duplicate)."""
    sandbox = _make_recording_sandbox()
    sandbox._ignore_mask = {(61, 19)}
    agent, snapshots = _make_tool_loop_agent(sandbox)
    agent._context_budget_tokens = 60
    _wire_fake_llm(
        agent,
        snapshots,
        [_python_response("action(1)")],
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": "system prompt"}]
    for i in range(40):
        messages.append({"role": "user", "content": f"old frame {i}"})
        messages.append({"role": "assistant", "content": f"old reply {i}"})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Frame 0"}]})

    agent._run_tool_loop(messages, grid_b64="")

    assert len(snapshots) >= 1
    for call_idx, snap in enumerate(snapshots, start=1):
        blocks = _blocks_in(snap)
        assert len(blocks) == 1, (
            f"call {call_idx}: expected exactly one block, got {len(blocks)}"
        )
        idx, _ = blocks[0]
        assert idx == 1, (
            f"call {call_idx}: block at index {idx} under trim pressure, expected 1"
        )