"""Unit tests for Duck Harness world model parser."""

from __future__ import annotations

from agents.duck_harness_agent.world_model import (
    ALL_KEYS,
    CANONICAL_LABELS,
    clear_world_model,
    extract_labeled_blocks,
    extract_world_model,
    extract_world_model_strict,
    format_world_model,
)

# ── Tests ──────────────────────────────────────────────────────────────────


def test_parse_wellformed():
    """7-block response with all labels parsed into correct keys."""
    content = """\
World model: The grid has a 3x3 pattern
Goal model: Reach the green square
Action model: Move right when path is clear
Recent findings: Blue entity moved left on frame 5
Open questions: What does the red switch do?
Plan: Move right 3 times
Cross-level notes: This level shares mechanics with level 1"""

    result = extract_world_model(content)
    assert result["world_model"] == "The grid has a 3x3 pattern"
    assert result["goal_model"] == "Reach the green square"
    assert result["action_model"] == "Move right when path is clear"
    assert result["recent_findings"] == "Blue entity moved left on frame 5"
    assert result["open_questions"] == "What does the red switch do?"
    assert result["current_plan"] == "Move right 3 times"
    assert result["cross_level_notes"] == "This level shares mechanics with level 1"


def test_parse_missing_labels():
    """Missing labels → empty strings for those keys."""
    content = """\
World model: Grid is 64x64
Plan: Move left"""

    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Move left"
    assert result["goal_model"] == ""
    assert result["action_model"] == ""
    assert result["recent_findings"] == ""
    assert result["open_questions"] == ""
    assert result["cross_level_notes"] == ""


def test_fallback_labels():
    """Hypothesis → world_model, History check → recent_findings, Next test → current_plan."""
    content = """\
Hypothesis: Entities rotate after each action
History check: Action 3 caused entity to disappear
Next test: Try action 5 near the wall"""

    result = extract_world_model(content)
    assert result["world_model"] == "Entities rotate after each action"
    assert result["recent_findings"] == "Action 3 caused entity to disappear"
    assert result["current_plan"] == "Try action 5 near the wall"

    # Canonical labels take priority over fallbacks when both present
    content_both = """\
World model: Canonical WM
Hypothesis: Fallback WM"""
    result_both = extract_world_model(content_both)
    assert result_both["world_model"] == "Canonical WM"


def test_format_world_model():
    """format_world_model produces header + labels + footer."""
    model = {
        "world_model": "Grid is 64x64",
        "goal_model": "Reach exit",
        "action_model": "",
        "recent_findings": "Frame 5 showed movement",
        "open_questions": "",
        "current_plan": "Go right",
        "cross_level_notes": "",
    }
    text = format_world_model(model)
    assert text.startswith("Working world model carried from earlier turns:")
    assert text.endswith("- Revise any item above immediately if current_frame contradicts it.")
    assert "World model: Grid is 64x64" in text
    assert "Action model:" in text  # empty value → just label with colon
    assert "Goal model: Reach exit" in text


def test_clear_world_model():
    """clear_world_model returns all values as empty strings."""
    model = {
        "world_model": "something",
        "goal_model": "else",
        "action_model": "move",
        "recent_findings": "found",
        "open_questions": "what",
        "current_plan": "go",
        "cross_level_notes": "note",
    }
    cleared = clear_world_model(model)
    for key in ALL_KEYS:
        assert cleared[key] == ""
    # Original model should be unchanged
    assert model["world_model"] == "something"


def test_multiline_blocks():
    """Blocks spanning multiple lines are captured fully."""
    content = """\
World model:
  Line 1 of world model.
  Line 2 of world model.
Goal model:
  Line 1 of goal.
  Line 2 of goal."""

    result = extract_world_model(content)
    assert "Line 1 of world model" in result["world_model"]
    assert "Line 2 of world model" in result["world_model"]
    assert "Line 1 of goal" in result["goal_model"]
    assert "Line 2 of goal" in result["goal_model"]


def test_leading_dash_stripped():
    """'- World model: text' parsed correctly (leading dash stripped)."""
    content = "- World model: Dashed label content\n- Plan: Dashed plan"
    result = extract_world_model(content)
    assert result["world_model"] == "Dashed label content"
    assert result["current_plan"] == "Dashed plan"


def test_case_insensitive():
    """'WORLD MODEL: text' parsed correctly (case-insensitive)."""
    content = "WORLD MODEL: Uppercase label\nGOAL MODEL: Uppercase goal"
    result = extract_world_model(content)
    assert result["world_model"] == "Uppercase label"
    assert result["goal_model"] == "Uppercase goal"


def test_star_prefix_stripped():
    """'* World model: text' parsed correctly (leading star stripped)."""
    content = "* World model: Starred content\n* Plan: Starred plan"
    result = extract_world_model(content)
    assert result["world_model"] == "Starred content"
    assert result["current_plan"] == "Starred plan"


def test_extract_labeled_blocks_custom_labels():
    """extract_labeled_blocks works with arbitrary label names."""
    content = "Name: Alice\nAge: 30\nCity: NYC"
    result = extract_labeled_blocks(content, ["Name", "Age", "City", "Country"])
    assert result["name"] == "Alice"
    assert result["age"] == "30"
    assert result["city"] == "NYC"
    assert result["country"] == ""


def test_format_world_model_empty():
    """format_world_model with all-empty values shows labels with no content."""
    model = {key: "" for key in ALL_KEYS}
    text = format_world_model(model)
    assert text.startswith("Working world model carried from earlier turns:")
    # Each label should appear with just a colon (no trailing content)
    for label, _ in CANONICAL_LABELS:
        assert f"{label}:" in text


# ── Markdown bold stripping tests ────────────────────────────────────────────


def test_markdown_bold_label():
    """**World model**: content → label='World model', not 'World model**'."""
    content = "**World model**: Grid is 64x64\n**Plan**: Go right"
    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Go right"


def test_list_markdown_bold_label():
    """- **World model**: content → label='World model'."""
    content = "- **World model**: Grid is 64x64\n- **Plan**: Go right"
    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Go right"


def test_plain_label_still_works():
    """World model: content (no markdown) still parsed correctly."""
    content = "World model: Grid is 64x64\nPlan: Go right"
    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Go right"


def test_dash_label_still_works():
    """- World model: content (dash prefix) still parsed correctly."""
    content = "- World model: Grid is 64x64\n- Plan: Go right"
    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Go right"


def test_star_label_still_works():
    """* World model: content (star prefix) still parsed correctly."""
    content = "* World model: Grid is 64x64\n* Plan: Go right"
    result = extract_world_model(content)
    assert result["world_model"] == "Grid is 64x64"
    assert result["current_plan"] == "Go right"


# ── extract_world_model_strict tests ─────────────────────────────────────────


def test_strict_all_present():
    """All 7 labels found → parsed dict full, missing empty."""
    content = """\
World model: The grid has a 3x3 pattern
Goal model: Reach the green square
Action model: Move right when path is clear
Recent findings: Blue entity moved left
Open questions: What does the red switch do?
Plan: Move right 3 times
Cross-level notes: Shares mechanics with level 1"""
    parsed, missing = extract_world_model_strict(content)
    assert parsed["world_model"] == "The grid has a 3x3 pattern"
    assert parsed["goal_model"] == "Reach the green square"
    assert missing == []


def test_strict_missing_labels():
    """Missing labels reported in missing list with display labels."""
    content = "World model: Grid is 64x64\nPlan: Go right"
    parsed, missing = extract_world_model_strict(content)
    assert parsed["world_model"] == "Grid is 64x64"
    assert parsed["current_plan"] == "Go right"
    assert parsed["goal_model"] == ""
    assert "Goal model" in missing
    assert "Action model" in missing
    assert "Recent findings" in missing
    assert "Open questions" in missing
    assert "Cross-level notes" in missing
    assert "World model" not in missing
    assert "Plan" not in missing


def test_strict_none_preserved_as_string():
    """LLM writes 'None' → preserved as 'None' string, not missing."""
    content = "World model: None\nPlan: Go right"
    parsed, missing = extract_world_model_strict(content)
    assert parsed["world_model"] == "None"
    assert "World model" not in missing


def test_strict_empty_block_not_missing():
    """Label present with empty value is not missing (value may include next block)."""
    content = "World model: None\nPlan: Go right"
    parsed, missing = extract_world_model_strict(content)
    assert parsed["world_model"] == "None"
    assert "World model" not in missing


def test_strict_fallback_found_not_missing():
    """Fallback label found → canonical key not in missing list."""
    content = "Hypothesis: Entities rotate\nNext test: Try action 5"
    parsed, missing = extract_world_model_strict(content)
    assert parsed["world_model"] == "Entities rotate"
    assert parsed["current_plan"] == "Try action 5"
    assert "World model" not in missing
    assert "Plan" not in missing