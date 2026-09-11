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
- cross-call registration linecache limitation (S3): getsource resolves
  against the CURRENT run_code snippet — documented, never crashes
- two-turn e2e (S4): block stripped from persistent history, re-injected
  fresh exactly once on the next turn's first LLM call
- level-transition pin (S5b): reset_for_level_transition clears the mask,
  preserves _simulate/_simulate_source — the block mirrors exactly that
- incident-shape replay (5681a14a): every sandbox state change (simulate
  rewrite, mask shrink) is visible on the NEXT LLM call's snapshot
"""

from __future__ import annotations

import ast
import itertools
import json
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from agents.simulator_agent.agent import SIM_STATE_HEADER, SimulatorFirstAgent
from agents.simulator_agent.conversation import is_sim_state_block

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

def _eval_setcomp(expr: str) -> set:
    """Evaluate a set-comprehension literal (e.g. '{(r, c) for r in ...}').

    ast.literal_eval cannot handle comprehensions; eval the expression
    instead (test fixture over agent-authored code, not arbitrary input).
    """
    return set(eval(expr))  # noqa: S307 - fixture-controlled expression



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
            # Generic set_ignore matcher: brace-depth scan from the
            # set_ignore( call to its closing paren, then evaluate the
            # literal (set literals AND set-comprehensions — comprehensions
            # are evaluated directly by exec since ast.literal_eval cannot).
            idx = code.find("set_ignore(")
            if idx != -1:
                depth = 0
                start = idx + len("set_ignore(")
                end = None
                for i in range(start, len(code)):
                    ch = code[i]
                    if ch in "{([":
                        depth += 1
                    elif ch in "})]":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                expr = code[start:end]
                try:
                    self._ignore_mask = ast.literal_eval(expr)
                except (ValueError, SyntaxError):
                    self._ignore_mask = set(_eval_setcomp(expr))
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


# ── Trim exemption + persistent-history strip (Task 3) ───────────────────
#
# These exercise the REAL conversation.py pipeline functions directly
# (pure functions — no agent construction needed). The block counts toward
# estimate_tokens but is never selected for dropping; persistent history
# strips it before the budget trim.


def _block_msg() -> dict[str, Any]:
    """A minimal [Simulator state] block message (header prefix is what
    is_sim_state_block keys on — content body is irrelevant here)."""
    return {"role": "user", "content": f"{SIM_STATE_HEADER} (authoritative; refreshed every call)\nignore: none"}


def _tool_pairing_ok(messages: list[dict[str, Any]]) -> bool:
    """Every role:tool message must be preceded by its assistant tool_calls
    partner (the API pairing invariant conversation.py pins)."""
    prev_role: str | None = None
    prev_tool_calls: Any = None
    for m in messages:
        if m.get("role") == "tool":
            if prev_role != "assistant" or not prev_tool_calls:
                return False
        prev_role = m.get("role")
        prev_tool_calls = m.get("tool_calls")
    return True


@pytest.mark.unit
def test_trim_exemption_keeps_block() -> None:
    """Budget-forced trim with the block at history head: the block survives
    at index 1 of the trimmed output, the oldest post-block history drops
    first, and no tool message appears before its assistant tool_calls
    partner."""
    from agents.simulator_agent.conversation import trim_messages_for_context

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system prompt"},
        _block_msg(),
    ]
    for i in range(10):
        messages.append({"role": "user", "content": f"old frame {i} " + "x" * 300})
        messages.append({"role": "assistant", "content": f"old reply {i}"})
    messages.append({"role": "user", "content": [{"type": "text", "text": "Frame 0"}]})

    trimmed = trim_messages_for_context(messages, budget_tokens=400)

    assert trimmed[0]["role"] == "system"
    assert is_sim_state_block(trimmed[1]), (
        "block must survive at index 1 under budget pressure"
    )
    assert len(trimmed) < len(messages), "trim must actually have dropped history"
    assert _tool_pairing_ok(trimmed)


@pytest.mark.unit
def test_block_counts_toward_budget() -> None:
    """The block is exempt from SELECTION only — estimate_tokens must still
    count it (real budget cost)."""
    from agents.simulator_agent.conversation import estimate_tokens

    with_block = [
        {"role": "system", "content": "system"},
        _block_msg(),
        {"role": "user", "content": "frame"},
    ]
    without_block = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "frame"},
    ]
    assert estimate_tokens(with_block) > estimate_tokens(without_block)


@pytest.mark.unit
def test_persistent_history_strips_block() -> None:
    """persistent_history_messages output contains zero sim-state blocks;
    the [Current notes] strip still works (existing behavior unchanged)."""
    from agents.simulator_agent.conversation import persistent_history_messages

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        _block_msg(),
        {"role": "user", "content": "[Current notes]\nold notes"},
        {"role": "user", "content": "user 0"},
        {"role": "assistant", "content": "assistant 0"},
        {"role": "user", "content": "user 1"},
        {"role": "assistant", "content": "assistant 1"},
        {"role": "user", "content": "user 2"},
        {"role": "assistant", "content": "assistant 2"},
    ]
    history = persistent_history_messages(messages, budget_tokens=50_000)

    assert history, "pipeline must keep history"
    assert all(not is_sim_state_block(m) for m in history), (
        "no stale sim-state block may survive the save path"
    )
    assert all(
        not (m.get("role") == "user" and m.get("content", "").startswith("[Current notes]"))
        for m in history
    ), "notes strip must still work"


@pytest.mark.unit
def test_trim_old_non_tool_messages_leaves_block() -> None:
    """trim_old_non_tool_messages must never rewrite the block: its nudge
    patterns ('discovered something new', 'Please use the python tool') and
    list-content frame targeting structurally cannot match a str user
    message with the sim-state header — pinned here with a defensive-skip
    guard in the selector."""
    from agents.simulator_agent.conversation import trim_old_non_tool_messages

    block = _block_msg()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        block,
        {"role": "user", "content": "discovered something new: nudge 1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "Please use the python tool: nudge 2"},
        {"role": "assistant", "content": "a2"},
    ]
    trim_old_non_tool_messages(messages)

    assert messages[1] is block
    assert messages[1]["content"] == block["content"], (
        "block content must be byte-identical after the trimmer runs"
    )
    # nudge_indices[:-1] keeps the LAST nudge (existing behavior) — the
    # earlier nudge is rewritten.
    assert messages[2]["content"] == "[nudge]"
    assert messages[4]["content"] == "Please use the python tool: nudge 2"


@pytest.mark.unit
def test_drop_oldest_skips_block_pairing_intact() -> None:
    """History [block, assistant(tool_calls), tool, user, ...] under budget
    pressure: the block is skipped for selection, the assistant+tool pair
    drops TOGETHER (tool never orphaned), and the block survives."""
    from agents.simulator_agent.conversation import drop_oldest_history_block

    history: list[dict[str, Any]] = [
        _block_msg(),
        {
            "role": "assistant",
            "content": "old reply",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "python", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": "old result"},
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]
    changed = drop_oldest_history_block(history, preserve_recent=1)

    assert changed is True
    assert is_sim_state_block(history[0]), "block must survive at head"
    assert history == [
        _block_msg(),
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]
    assert _tool_pairing_ok(history)


@pytest.mark.unit
def test_drop_oldest_all_blocks_returns_false() -> None:
    """When the only removable messages are sim-state blocks, nothing is
    droppable — return False and leave the history untouched."""
    from agents.simulator_agent.conversation import drop_oldest_history_block

    history: list[dict[str, Any]] = [_block_msg(), _block_msg()]
    original = [dict(m) for m in history]
    changed = drop_oldest_history_block(history, preserve_recent=1)

    assert changed is False
    assert history == original


@pytest.mark.unit
def test_drop_oldest_block_becomes_head_survives_cleanup_scans() -> None:
    """After popping the selected non-block message, if the block becomes
    the new head, the leading tool/non-user cleanup scans must NOT consume
    it (it is a user message, but the scans pop non-user heads — the
    explicit is_sim_state_block guard keeps it)."""
    from agents.simulator_agent.conversation import drop_oldest_history_block

    history: list[dict[str, Any]] = [
        _block_msg(),
        {"role": "assistant", "content": "orphan assistant"},
        {"role": "user", "content": "real user"},
        {"role": "assistant", "content": "recent"},
    ]
    changed = drop_oldest_history_block(history, preserve_recent=1)

    assert changed is True
    assert is_sim_state_block(history[0]), "block must survive as new head"
    assert history == [
        _block_msg(),
        {"role": "user", "content": "real user"},
        {"role": "assistant", "content": "recent"},
    ]

# ── S3: cross-call registration linecache limitation (Task 4) ────────────
#
# run_code() seeds linecache.cache["<sandbox>"] with the CURRENT call's
# snippet before exec (sandbox.py:1094-1103). When the def happens in one
# call and set_simulate() in a LATER call, inspect.getsource(func) resolves
# func's co_firstlineno against the LATER snippet — capturing the WRONG
# lines. This is a documented limitation, pinned here so it can never
# surprise anyone again.


@pytest.mark.unit
def test_cross_call_registration_documents_linecache_limit(
    seeded_live_sandbox, mock_step_env
) -> None:
    """S3 (incident 5681a14a): def in run_code call A, set_simulate in call B.

    DOCUMENTED LIMITATION: linecache.cache["<sandbox>"] is overwritten by
    every run_code() call, so at call B getsource(my_sim) resolves
    my_sim's co_firstlineno against call B's snippet lines. Observed
    behavior (probed empirically): the capture returns call B's snippet
    lines starting at my_sim's co_firstlineno — here the single line
    "set_simulate(my_sim)" — i.e. the WRONG source (it is not my_sim's
    body). The builder must not crash on it; the block shows whatever was
    captured verbatim. This test pins that behavior as a known limitation,
    not a hidden surprise.
    """
    sandbox = seeded_live_sandbox(mock_step_env)

    # Call A: define the function only (no registration).
    out_a, err_a, _ = sandbox.run_code(
        "def my_sim(grid, action):\n    return copy_grid(grid)\n"
    )
    assert err_a is None
    assert sandbox._simulate is None, "call A must not register anything"
    assert sandbox._simulate_source == ""

    # Call B: register the function defined in call A (cross-call).
    out_b, err_b, _ = sandbox.run_code("set_simulate(my_sim)\n")
    assert err_b is None
    assert sandbox._simulate is not None
    captured = sandbox._simulate_source

    # Whatever was captured, the builder must not raise on it.
    agent = _make_agent(sandbox)
    block = agent._build_sim_state_block()
    assert block is not None, "registered simulate must produce a block"
    assert block.startswith(_HEADER)
    assert "simulate: registered" in block
    assert "ignore: none" in block

    # Document precisely what was captured: the linecache-"<sandbox>"
    # overwrite means getsource resolved my_sim's co_firstlineno (line 1
    # of call A's snippet) against call B's snippet — so the capture is
    # call B's line 1, NOT my_sim's body. Record it with a clear message;
    # if the capture mechanism ever changes, this assertion documents the
    # new behavior and the docstring must be updated.
    assert captured == "set_simulate(my_sim)\n", (
        "DOCUMENTED LIMITATION (S3): cross-call capture resolved my_sim's "
        f"co_firstlineno against call B's linecache-seeded snippet and got "
        f"{captured!r} — the WRONG source (call B's snippet line, not "
        "my_sim's body). If this assertion fails, the capture behavior "
        "changed: update this pin + the module docstring accordingly."
    )
    # The block renders the captured (wrong) source verbatim — the model
    # at least sees SOMETHING deterministic rather than crashing.
    assert captured in block


# ── S4: two-turn e2e — no stale duplicate across turns (Task 4) ──────────
#
# Turn 1 registers simulate + sets a mask via the real tool loop; the
# block must be STRIPPED from the saved persistent history. Turn 2
# re-assembles [system, *saved_history, frame_user] and runs the hook:
# exactly ONE block, reflecting LIVE sandbox state, no stale copy deeper
# in the history.


def _make_recording_sandbox_v2() -> Any:
    """_RecordingSandbox extended for the Task-4 arcs: run_code() also
    executes set_simulate(...) registrations against plain-attribute state
    (mirrors sandbox.set_simulate capture semantics: the def-to-set_simulate
    span of the CURRENT run_code snippet becomes _simulate_source)."""
    base = _make_recording_sandbox()

    class _RecordingSandboxV2(type(base)):
        pass

    s = _RecordingSandboxV2()
    s.__dict__.update(base.__dict__)

    original_run_code = s.run_code

    def run_code_with_simulate(code: str) -> tuple[str, str | None, int | None]:
        result = original_run_code(code)
        marker = "set_simulate("
        if marker in code and "def " in code:
            # Mirror sandbox.set_simulate source capture: the def-to-
            # set_simulate span of the current snippet (same-call case).
            lines = code.splitlines(True)
            stop = next(i for i, ln in enumerate(lines) if marker in ln)
            s._simulate = lambda g, a: g  # frozen-copy stand-in
            s._simulate_source = "".join(lines[: stop + 1])
        return result

    s.run_code = run_code_with_simulate  # type: ignore[method-assign]
    return s


def test_incident_shape_5681a14a() -> None:
    """Incident 5681a14a replay — the block the model never had.

    Arc (recording .llm.jsonl): v1 simulate registered at seq 11 (~857
    chars); 128-cell timer mask set at seq 36; v4 rewrite at seq 44 with
    the mask SHRUNK to 98 cells (set_ignore REPLACE); the model never saw
    any of it. Here the fake-LLM arc drives the real loop and asserts
    every state change is visible in the NEXT call's snapshot.
    """
    v1_source = "def simulate(grid, action):\n" + "    # v1 timer model\n" * 38 + "    return grid\n"
    v2_source = "def simulate(grid, action):\n" + "    # v4 5x5 origin model\n" * 50 + "    return grid\n"
    assert len(v1_source) > 500 and len(v2_source) > 500

    sandbox = _make_recording_sandbox_v2()
    agent, snapshots = _make_tool_loop_agent(sandbox)
    _bind_budget_end(sandbox, agent)
    _wire_fake_llm(
        agent,
        snapshots,
        [
            _python_response("set_ignore({(r, c) for r in (61, 62) for c in range(64)})"),
            _python_response(
                v2_source + "set_simulate(simulate)\n"
            ),
            _python_response("set_ignore({(r, c) for r in (61, 62) for c in range(15, 64)})"),
            _python_response("action(1)"),
        ],
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [{"type": "text", "text": "Frame 0"}]},
    ]
    returned, _ = agent._run_tool_loop(messages, grid_b64="")

    assert len(snapshots) == 4, f"expected 4 calls, got {len(snapshots)}"
    # call 1: stateless -> no block (skip-when-empty).
    assert _blocks_in(snapshots[0]) == []
    # call 2: mask set in call 1 -> visible (band).
    blocks2 = _blocks_in(snapshots[1])
    assert len(blocks2) == 1
    assert "ignore: 128 cells" in blocks2[0][1]
    # call 3: simulate registered in call 2 -> v2 source visible; mask still 128.
    blocks3 = _blocks_in(snapshots[2])
    assert len(blocks3) == 1
    assert "# v4 5x5 origin model" in blocks3[0][1], (
        "call 3 shows the v2 source registered in call 2 (no re-derivation needed)"
    )
    assert "ignore: 128 cells" in blocks3[0][1]
    # call 4: mask shrunk in call 3 (REPLACE semantics) -> visible shrink.
    blocks4 = _blocks_in(snapshots[3])
    assert len(blocks4) == 1
    assert "ignore: 98 cells" in blocks4[0][1], (
        "the shrink the incident model never saw is now visible"
    )
    # Largest realistic block stays under 4 KB.
    assert len(blocks4[0][1]) < 4096


def test_two_turn_e2e_no_stale_duplicate() -> None:
    """S4: block stripped on save, re-injected fresh exactly once next turn.

    Turn 1 (real _run_tool_loop): call1 registers simulate via
    set_simulate, call2 sets a 2-cell mask, call3 acts (budget-end). The
    saved history must contain ZERO sim-state blocks. Turn 2: assemble
    [system, *saved_history, frame_user], run the hook — EXACTLY ONE
    block at index 1 reflecting LIVE sandbox state, none stale deeper.

    NOTE: _run_tool_loop REBINDS messages at the trim step (agent.py:448);
    the returned list is the live one (production uses the return value
    too, agent.py:357)."""
    sandbox = _make_recording_sandbox_v2()
    agent, snapshots = _make_tool_loop_agent(sandbox)
    _bind_budget_end(sandbox, agent)
    _wire_fake_llm(
        agent,
        snapshots,
        [
            _python_response(
                "def my_sim(grid, action):\n"
                "    return copy_grid(grid)\n"
                "set_simulate(my_sim)\n"
            ),
            _python_response("set_ignore({(61, 19), (62, 19)})"),
            _python_response("action(1)"),
        ],
    )

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [{"type": "text", "text": "Frame 0"}]},
    ]
    returned_messages, _terminated = agent._run_tool_loop(messages, grid_b64="")
    assert len(snapshots) == 3, f"expected 3 LLM calls, got {len(snapshots)}"

    # Turn-1 snapshots: call1 stateless (no block), call2 shows source,
    # call3 shows the mask — state visible on the NEXT call.
    assert _blocks_in(snapshots[0]) == [], "call 1 is stateless: no block"
    assert "def my_sim(grid, action):" in snapshots[1][1]["content"], (
        "call 2 must show the source registered in call 1"
    )
    assert "ignore: 2 cells" in snapshots[2][1]["content"], (
        "call 3 must show the mask set in call 2"
    )

    # Save history exactly as the agent does at turn end.
    saved_history = agent._persistent_history_messages(returned_messages)
    assert saved_history, "save path must keep history"
    assert _blocks_in(saved_history) == [], (
        "ZERO sim-state blocks may survive the persistent-history save"
    )
    assert any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in saved_history
    ), "the substantive turn-1 tool exchange survives the strip"

    # Turn 2: fresh agent over the same LIVE sandbox.
    agent2, _snapshots2 = _make_tool_loop_agent(sandbox)
    turn2_messages = [
        {"role": "system", "content": "system prompt"},
        *saved_history,
        {"role": "user", "content": [{"type": "text", "text": "Frame 1"}]},
    ]
    agent2._inject_sim_state_block(turn2_messages)

    blocks = _blocks_in(turn2_messages)
    assert len(blocks) == 1, f"turn 2 must carry EXACTLY ONE block, got {len(blocks)}"
    idx, content = blocks[0]
    assert idx == 1, f"turn-2 block at index {idx}, expected 1"
    assert "def my_sim(grid, action):" in content, (
        "turn-2 block reflects the LIVE sandbox's registered source"
    )
    assert "ignore: 2 cells" in content, "turn-2 block reflects the LIVE mask"
    assert all(
        not is_sim_state_block(m) for m in turn2_messages[2:]
    ), "no stale block deeper in the turn-2 history"


def test_level_transition_block_semantics(seeded_live_sandbox, win_callback) -> None:
    """S5b: reset_for_level_transition clears the ignore mask but preserves
    the simulate function + captured source (sandbox.py:1030-1044). The
    block must reflect exactly that: preserved source, 'ignore: none'."""
    s = seeded_live_sandbox(win_callback, current_frame=[[0] * 8 for _ in range(8)])
    code = (
        "def my_sim(grid, action):\n"
        "    return copy_grid(grid)\n"
        "set_simulate(my_sim)\n"
        "set_ignore({(61, 19), (62, 19)})\n"
    )
    output, error, _ = s.run_code(code)
    assert error is None
    assert s._simulate is not None
    assert s._ignore_mask == {(61, 19), (62, 19)}

    s.reset_for_level_transition()
    assert s._ignore_mask == set(), "mask must clear on level transition"
    assert s._simulate is not None, "simulate must survive level transition"
    assert "def my_sim" in (s._simulate_source or ""), (
        "captured source must survive level transition"
    )

    agent = _make_agent(s)
    block = agent._build_sim_state_block()
    assert block is not None
    assert "def my_sim(grid, action):" in block
    assert "ignore: none" in block
