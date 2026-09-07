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
    AGENT_RUNTIME_STATE_ADDENDUM,
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


@pytest.mark.unit
def test_simulate_redefinition_guidance_present():
    """a21a2571: the model defined `def simulate(...)` 6 times, silently
    discarded each time. The prompt must explain the protected name and
    show the register-an-alias pattern."""
    text = AGENT_SIMULATOR_TOOLS_ADDENDUM
    assert "NEVER define a function named `simulate`" in text
    assert "set_simulate(" in text
    assert "any name except 'simulate'" in text


@pytest.mark.unit
def test_history_transition_pairing_documented():
    """a21a2571: the model paired history[i]["action"] with the transition
    TO history[i+1]["frame"], corrupting the action->direction mapping.
    The runtime-state addendum must state the pairing formula."""
    text = AGENT_RUNTIME_STATE_ADDENDUM
    assert 'before = history[i-1]["frame"]' in text
    assert 'after  = history[i]["frame"]' in text
    assert "sim(before, a_i) must equal after" in text
    assert "off-by-one" in text


@pytest.mark.unit
def test_diagnose_usage_example_documented():
    """a21a2571: diagnose() was never called (0 of 72 LLM calls). The
    VALIDATION section must explain MISSED/SPURIOUS/WRONG_VALUE and when
    to run it."""
    text = AGENT_SIMULATOR_TOOLS_ADDENDUM
    assert "MISSED" in text
    assert "SPURIOUS" in text
    assert "WRONG_VALUE" in text
    assert "call diagnose() BEFORE rewriting" in text
