"""Content assertions for simulator agent prompt strings.

These tests verify documented properties of agents/simulator_agent/prompts.py
without modifying prompts.py.
"""

from __future__ import annotations

import pytest

from agents.simulator_agent.prompts import (
    AGENT_GAME_OVERVIEW_ADDENDUM,
    AGENT_PHASES_OVERVIEW,
    AGENT_PYTHON_TOOL_ADDENDUM,
    AGENT_SIMULATOR_TOOLS_ADDENDUM,
    AGENT_SYSTEM_PROMPT,
    AGENT_WORLD_MODEL_ADDENDUM,
)


@pytest.mark.unit
def test_find_objects_documented():
    assert "find_objects" in AGENT_SIMULATOR_TOOLS_ADDENDUM


@pytest.mark.unit
def test_atoms_ordering_documented():
    text = AGENT_SIMULATOR_TOOLS_ADDENDUM
    assert ("scan order" in text) or ("row-major" in text)


@pytest.mark.unit
def test_background_heuristic_removed():
    assert "most common color" not in AGENT_GAME_OVERVIEW_ADDENDUM.lower()


@pytest.mark.unit
def test_phases_overview_documents_workflow():
    text = AGENT_PHASES_OVERVIEW
    assert "EXPLORE → MODEL → PLAN → EXECUTE" in text
    assert "set_phase" in text


@pytest.mark.unit
def test_world_model_addendum_documents_update_notes_and_done():
    text = AGENT_WORLD_MODEL_ADDENDUM
    assert "update_notes" in text
    assert "DONE" in text


@pytest.mark.unit
def test_python_tool_addendum_documents_limits():
    text = AGENT_PYTHON_TOOL_ADDENDUM
    assert "4096 characters" in text
    assert "30 seconds" in text


@pytest.mark.unit
def test_system_prompt_assembles_all_addenda():
    prompt = AGENT_SYSTEM_PROMPT
    assert len(prompt) > 1000
    assert "Runtime state" in prompt
    assert "Visual game" in prompt
    assert "Python tool" in prompt
    assert "Sandbox tools" in prompt
    assert "How to work" in prompt
    assert "Notes and Plan" in prompt


@pytest.mark.unit
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
