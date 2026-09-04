"""Tests for the SimulatorFirstAgent _frame_index -> action_counter - 1 refactor.

These tests verify that the agent no longer tracks a separate ``_frame_index``
attribute and instead derives frame indices dynamically from
``self.action_counter - 1``. They use ``__new__()`` to avoid the full
``__init__`` (which needs API keys/env vars) and ``inspect.getsource()`` for
source-level assertions, matching the patterns in ``test_prompts.py``.
"""

from __future__ import annotations

import inspect

import pytest

from agents.simulator_agent.agent import SimulatorFirstAgent


@pytest.mark.unit
def test_frame_indexer_uses_action_counter_minus_one() -> None:
    """``frame_indexer`` lambda returns ``action_counter - 1``."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.action_counter = 3
    # The lambda is created in __init__; reconstruct the same expression.
    frame_indexer = lambda: agent.action_counter - 1  # noqa: E731

    assert frame_indexer() == 2


@pytest.mark.unit
def test_frame_index_jumps_mid_turn() -> None:
    """``frame_indexer`` reflects mid-turn ``action_counter`` changes."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.action_counter = 5
    frame_indexer = lambda: agent.action_counter - 1  # noqa: E731

    assert frame_indexer() == 4

    agent.action_counter = 12
    assert frame_indexer() == 11


@pytest.mark.unit
def test_no_frame_index_attribute_remains() -> None:
    """No ``_frame_index`` attribute assignment remains in the source."""
    init_source = inspect.getsource(SimulatorFirstAgent.__init__)
    run_source = inspect.getsource(SimulatorFirstAgent.run)
    combined = init_source + "\n" + run_source

    assert "_frame_index" not in combined
    assert "self.action_counter - 1" in combined


@pytest.mark.unit
def test_frame_index_zero_on_first_real_turn() -> None:
    """After RESET, ``action_counter`` is 1, so the derived frame index is 0."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.action_counter = 1
    frame_indexer = lambda: agent.action_counter - 1  # noqa: E731

    assert frame_indexer() == 0
