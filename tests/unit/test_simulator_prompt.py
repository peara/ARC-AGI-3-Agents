"""Content assertions for simulator agent prompt edits.

These tests verify the Task 1 edits to agents/simulator_agent/prompts.py
without importing schemas or other constants, and without modifying prompts.py.
"""

from __future__ import annotations

import re

from agents.simulator_agent.prompts import (
    AGENT_GAME_OVERVIEW_ADDENDUM,
    AGENT_SIMULATOR_TOOLS_ADDENDUM,
    AGENT_SYSTEM_PROMPT,
    AGENT_WORKFLOW_ADDENDUM,
)


def test_find_objects_documented():
    assert "find_objects" in AGENT_SIMULATOR_TOOLS_ADDENDUM


def test_atoms_ordering_documented():
    text = AGENT_SIMULATOR_TOOLS_ADDENDUM
    assert ("scan order" in text) or ("row-major" in text)


def test_background_heuristic_removed():
    assert "most common color" not in AGENT_GAME_OVERVIEW_ADDENDUM.lower()


def test_bfs_uses_is_not_none():
    assert "if path is not None" in AGENT_WORKFLOW_ADDENDUM


def test_bfs_goal_reached_handled():
    assert "Goal already reached" in AGENT_WORKFLOW_ADDENDUM


def test_bfs_if_path_colon_absent():
    assert re.search(r"if path:\s", AGENT_WORKFLOW_ADDENDUM) is None


def test_system_prompt_assembles():
    assert len(AGENT_SYSTEM_PROMPT) > 1000


def test_find_objects_description_accurate():
    text = AGENT_SIMULATOR_TOOLS_ADDENDUM
    start = text.find("find_objects(grid")
    assert start != -1, "find_objects description not found"
    end = text.find("\n", start)
    if end == -1:
        end = len(text)
    description = text[start:end]
    assert ("sorted by size" in description) or ("size descending" in description)
    assert ("proximity" in description) or ("within 2 cells" in description)
