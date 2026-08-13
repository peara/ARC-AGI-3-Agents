"""Unit tests for the unified agent tool schemas.

Validates the static JSON-schema tool definitions (INSPECT_TOOL, DECIDE_TOOL,
UNIFIED_TOOLS) exported from ``agents.langgraph_unified_agent.tools``.
"""

from __future__ import annotations

import pytest

from agents.langgraph_unified_agent.tools import (
    DECIDE_TOOL,
    DECIDE_V2_TOOL,
    INSPECT_TOOL,
    UNIFIED_TOOLS,
    UNIFIED_TOOLS_V2,
)


@pytest.mark.unit
def test_inspect_tool_schema():
    """INSPECT_TOOL has type='function', function.name='inspect'."""
    assert INSPECT_TOOL["type"] == "function"
    assert INSPECT_TOOL["function"]["name"] == "inspect"


@pytest.mark.unit
def test_inspect_tool_code_param():
    """INSPECT_TOOL parameters has a 'code' field of type 'string'."""
    params = INSPECT_TOOL["function"]["parameters"]
    assert "code" in params["properties"]
    assert params["properties"]["code"]["type"] == "string"


@pytest.mark.unit
def test_inspect_tool_code_required():
    """'code' is in the required list for INSPECT_TOOL."""
    params = INSPECT_TOOL["function"]["parameters"]
    assert "code" in params["required"]


@pytest.mark.unit
def test_inspect_tool_description_mentions_sandbox():
    """INSPECT_TOOL description mentions 'sandbox' or 'inspect'."""
    description = INSPECT_TOOL["function"]["description"].lower()
    assert "sandbox" in description or "inspect" in description


@pytest.mark.unit
def test_decide_tool_schema():
    """DECIDE_TOOL has type='function', function.name='decide'."""
    assert DECIDE_TOOL["type"] == "function"
    assert DECIDE_TOOL["function"]["name"] == "decide"


@pytest.mark.unit
def test_decide_tool_action_id_param():
    """DECIDE_TOOL action_id is integer type."""
    props = DECIDE_TOOL["function"]["parameters"]["properties"]
    assert props["action_id"]["type"] == "integer"


@pytest.mark.unit
def test_decide_tool_required_params():
    """DECIDE_TOOL required params are action_id, expectation, reflect."""
    params = DECIDE_TOOL["function"]["parameters"]
    assert sorted(params["required"]) == ["action_id", "expectation", "reflect"]


@pytest.mark.unit
def test_decide_tool_mechanics_param():
    """DECIDE_TOOL mechanics is array of strings type."""
    props = DECIDE_TOOL["function"]["parameters"]["properties"]
    assert props["mechanics"]["type"] == "array"
    assert props["mechanics"]["items"]["type"] == "string"


@pytest.mark.unit
def test_decide_tool_optional_fields():
    """mechanics_summary, tactical, tactical_summary are NOT in required."""
    required = DECIDE_TOOL["function"]["parameters"]["required"]
    for field in ("mechanics_summary", "tactical", "tactical_summary"):
        assert field not in required, f"{field} should not be required"


@pytest.mark.unit
def test_unified_tools_list_length():
    """UNIFIED_TOOLS has exactly 2 items."""
    assert len(UNIFIED_TOOLS) == 2


@pytest.mark.unit
def test_unified_tools_contains_inspect_and_decide():
    """UNIFIED_TOOLS[0] is INSPECT_TOOL, UNIFIED_TOOLS[1] is DECIDE_TOOL."""
    assert UNIFIED_TOOLS[0] is INSPECT_TOOL
    assert UNIFIED_TOOLS[1] is DECIDE_TOOL


# ------------------------------------------------------------------
# V2 schema tests (DECIDE_V2_TOOL / UNIFIED_TOOLS_V2)
# ------------------------------------------------------------------


@pytest.mark.unit
def test_decide_v2_tool_schema():
    """DECIDE_V2_TOOL has type='function', function.name='decide'."""
    assert DECIDE_V2_TOOL["type"] == "function"
    assert DECIDE_V2_TOOL["function"]["name"] == "decide"


@pytest.mark.unit
def test_decide_v2_world_model_actions():
    """DECIDE_V2_TOOL world_model has actions array of strings."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert "actions" in wm["properties"]
    assert wm["properties"]["actions"]["type"] == "array"
    assert wm["properties"]["actions"]["items"]["type"] == "string"


@pytest.mark.unit
def test_decide_v2_world_model_actions_required():
    """actions is in world_model.required."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert "actions" in wm["required"]


@pytest.mark.unit
def test_decide_v2_world_model_required_fields():
    """world_model.required includes actions, goal, goal_status, mechanics, tactical."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert sorted(wm["required"]) == ["actions", "goal", "goal_status", "mechanics", "tactical"]


@pytest.mark.unit
def test_decide_v2_world_model_goal():
    """DECIDE_V2_TOOL world_model has goal string field."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert "goal" in wm["properties"]
    assert wm["properties"]["goal"]["type"] == "string"


@pytest.mark.unit
def test_decide_v2_world_model_goal_status():
    """DECIDE_V2_TOOL world_model has goal_status string enum field."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert "goal_status" in wm["properties"]
    assert wm["properties"]["goal_status"]["type"] == "string"
    assert "enum" in wm["properties"]["goal_status"]
    assert "blocked" in wm["properties"]["goal_status"]["enum"]


@pytest.mark.unit
def test_decide_v2_world_model_required_with_goal():
    """world_model.required includes actions, goal, goal_status, mechanics, tactical."""
    wm = DECIDE_V2_TOOL["function"]["parameters"]["properties"]["world_model"]
    assert sorted(wm["required"]) == ["actions", "goal", "goal_status", "mechanics", "tactical"]


@pytest.mark.unit
def test_unified_tools_v2_list():
    """UNIFIED_TOOLS_V2 has 2 items (inspect + decide_v2)."""
    assert len(UNIFIED_TOOLS_V2) == 2
    assert UNIFIED_TOOLS_V2[0]["function"]["name"] == "inspect"
    assert UNIFIED_TOOLS_V2[1]["function"]["name"] == "decide"