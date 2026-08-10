"""Unit tests for parsers used by the unified node.

These tests cover:
- The search-anywhere action parser (``_parse_action_id_anywhere`` from the
  unified node module) that finds ACTION even after reasoning text.
- The vision-agent parser functions for EXPECT/REFLECT (``plan.py``) and
- MECHANICS/TACTICAL (``reflect.py``) which are reused by the unified node.
"""

from __future__ import annotations

import pytest

from agents.langgraph_unified_agent.nodes.unified import _parse_action_id_anywhere
from agents.langgraph_vision_agent.nodes.plan import (
    _parse_expectation,
    _parse_reflect_flag,
)
from agents.langgraph_vision_agent.nodes.reflect import _parse_response


@pytest.mark.unit
def test_parse_action_from_start():
    """ACTION at the start of the response is parsed correctly."""
    assert _parse_action_id_anywhere("ACTION 3 because adjacent") == 3


@pytest.mark.unit
def test_parse_action_from_middle():
    """ACTION after reasoning text on a later line is parsed correctly.

    This is the key experiment finding: the LLM may place ACTION after
    preliminary reasoning, not only at the start of the response.
    """
    text = (
        "The player can move.\n"
        "ACTION 4 because the target is above."
    )
    assert _parse_action_id_anywhere(text) == 4


@pytest.mark.unit
def test_parse_action_not_found():
    """No ACTION in response returns None."""
    assert _parse_action_id_anywhere("I think we should wait") is None


@pytest.mark.unit
def test_parse_reflect_yes():
    """REFLECT: yes returns True."""
    assert _parse_reflect_flag("Some text\nREFLECT: yes") is True


@pytest.mark.unit
def test_parse_reflect_no():
    """REFLECT: no returns False."""
    assert _parse_reflect_flag("Some text\nREFLECT: no") is False


@pytest.mark.unit
def test_parse_reflect_missing():
    """Missing REFLECT line defaults to False."""
    assert _parse_reflect_flag("ACTION 2 because safe") is False


@pytest.mark.unit
def test_parse_mechanics_full():
    """MECHANICS + MECHANICS_SUMMARY sections are parsed correctly."""
    text = (
        "MECHANICS:\n"
        "- gravity pulls objects down\n"
        "- keys open doors\n\n"
        "MECHANICS_SUMMARY: objects fall unless supported\n\n"
        "TACTICAL:\n"
        "- go right first\n\n"
        "TACTICAL_SUMMARY: explore east"
    )
    result = _parse_response(text)
    assert result is not None
    mechanics, mechanics_summary, tactical, tactical_summary = result
    assert "gravity pulls objects down" in mechanics
    assert "keys open doors" in mechanics
    assert mechanics_summary == "objects fall unless supported"
    assert "go right first" in tactical
    assert tactical_summary == "explore east"


@pytest.mark.unit
def test_parse_tactical_full():
    """TACTICAL + TACTICAL_SUMMARY sections are parsed correctly."""
    text = (
        "MECHANICS:\n"
        "- player can move\n\n"
        "MECHANICS_SUMMARY: basic movement\n\n"
        "TACTICAL:\n"
        "- avoid red cells\n"
        "- reach green portal\n\n"
        "TACTICAL_SUMMARY: stay safe and exit"
    )
    result = _parse_response(text)
    assert result is not None
    mechanics, mechanics_summary, tactical, tactical_summary = result
    assert mechanics == ["player can move"]
    assert mechanics_summary == "basic movement"
    assert tactical == ["avoid red cells", "reach green portal"]
    assert tactical_summary == "stay safe and exit"


@pytest.mark.unit
def test_parse_mechanics_missing():
    """Missing MECHANICS section returns None gracefully."""
    text = (
        "TACTICAL:\n"
        "- go left\n\n"
        "TACTICAL_SUMMARY: move west"
    )
    assert _parse_response(text) is None


@pytest.mark.unit
def test_parse_all_sections():
    """Full response with ACTION + EXPECT + REFLECT + MECHANICS + TACTICAL."""
    text = (
        "Reasoning: the agent should move up.\n"
        "ACTION 2 because the exit is above\n"
        "EXPECT: player moves toward exit\n"
        "REFLECT: yes\n\n"
        "MECHANICS:\n"
        "- up moves the agent north\n\n"
        "MECHANICS_SUMMARY: vertical movement works\n\n"
        "TACTICAL:\n"
        "- head to the top\n\n"
        "TACTICAL_SUMMARY: reach the exit"
    )
    assert _parse_action_id_anywhere(text) == 2
    assert _parse_expectation(text) == "player moves toward exit"
    assert _parse_reflect_flag(text) is True
    result = _parse_response(text)
    assert result is not None
    mechanics, mechanics_summary, tactical, tactical_summary = result
    assert mechanics == ["up moves the agent north"]
    assert mechanics_summary == "vertical movement works"
    assert tactical == ["head to the top"]
    assert tactical_summary == "reach the exit"
