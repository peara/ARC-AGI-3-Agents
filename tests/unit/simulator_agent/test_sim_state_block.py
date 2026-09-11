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

from typing import Any
from unittest.mock import MagicMock

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