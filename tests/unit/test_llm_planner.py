"""Unit tests for planning/llm_planner.py — prompt construction, response parsing, goal validation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from planning.llm_planner import (
    _build_messages,
    _parse_response,
    _validate_goal,
    call_planner,
)
from planning.probe import ProbeGoal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_call(response: str) -> MagicMock:
    """Return a MagicMock llm_call that returns *response*."""
    mock = MagicMock()
    mock.return_value = response
    return mock


def _bundle(
    entities: dict[str, dict[str, object]] | list[dict[str, object]] | None = None,
    *,
    entities_format: str = "dict",
    unknowns: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build a realistic scene bundle.

    entities_format="dict" uses string-keyed dict (old test format).
    entities_format="list" uses list-of-dicts (actual scene.summary() format).
    """
    if entities is None:
        entities = (
            [
                {"id": 0, "role": "controllable", "pos": [32, 16]},
                {"id": 17, "role": "counter", "pos": [12, 36]},
            ]
            if entities_format == "list"
            else {
                "0": {"role": "controllable", "pos": [32, 16]},
                "17": {"role": "counter", "pos": [12, 36]},
            }
        )
    bundle: dict[str, object] = {"scene": {"entities": entities}}
    if unknowns is not None:
        bundle["unknowns"] = unknowns
    return bundle


# ===========================================================================
# TestLLMPlanner
# ===========================================================================


@pytest.mark.unit
class TestLLMPlanner:
    """Tests for LLM planner prompt construction, response parsing, and validation."""

    # -----------------------------------------------------------------------
    # _build_messages
    # -----------------------------------------------------------------------

    def test_build_messages_structure(self) -> None:
        """Produces exactly 2 messages: system + user."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_messages_system_has_schema(self) -> None:
        """System message contains target schema and 'near'."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        content = messages[0]["content"]
        assert "target" in content
        assert "near" in content

    def test_build_messages_system_has_examples(self) -> None:
        """System message contains all 3 example JSONs."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        content = messages[0]["content"]
        # Example 1: near with relative entity ref (of: 17)
        assert '"of": 17' in content
        # Example 2: near with coordinate list [5, 32]
        assert "5, 32" in content
        # Example 3: near with small radius (of: 8)
        assert '"of": 8' in content

    def test_build_messages_system_has_instructions(self) -> None:
        """System message contains key instruction phrases."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        content = messages[0]["content"]
        assert "Always have an opinion" in content
        assert "reason" in content

    def test_build_messages_system_has_action_field(self) -> None:
        """System message describes the optional action field."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        content = messages[0]["content"]
        assert "action" in content
        assert "unknown" in content.lower() or "Unknown" in content

    def test_build_messages_system_has_unknowns_example(self) -> None:
        """System message contains the unknowns example with action field."""
        messages = _build_messages(_bundle(), [0, 1, 2, 3])
        content = messages[0]["content"]
        assert '"action": 3' in content
        assert "unknown" in content.lower()

    def test_build_messages_user_has_bundle(self) -> None:
        """User message contains the bundle JSON."""
        bundle = _bundle()
        messages = _build_messages(bundle, [0, 1, 2, 3])
        user_content = messages[1]["content"]
        bundle_json = json.dumps(bundle, indent=2)
        assert bundle_json in user_content

    def test_build_messages_user_has_actions(self) -> None:
        """User message contains the available actions list."""
        actions = [0, 1, 2, 3]
        messages = _build_messages(_bundle(), actions)
        user_content = messages[1]["content"]
        assert str(actions) in user_content

    def test_build_messages_with_failure_context(self) -> None:
        """Failure context appears in user message and contains 'prediction_failure'."""
        failure = {"type": "prediction_failure", "detail": "entity 17 not found"}
        messages = _build_messages(_bundle(), [0, 1], failure_context=failure)
        user_content = messages[1]["content"]
        assert "Failure context" in user_content
        assert "prediction_failure" in user_content

    def test_build_messages_without_failure_context(self) -> None:
        """No 'Failure context' section when failure_context is None."""
        messages = _build_messages(_bundle(), [0, 1], failure_context=None)
        user_content = messages[1]["content"]
        assert "Failure context" not in user_content

    def test_build_messages_with_unknowns_in_bundle(self) -> None:
        """Unknown actions in bundle appear in user message."""
        unknowns = [
            {"action": 3, "state": "fingerprint_abc"},
            {"action": 5, "state": "fingerprint_def"},
        ]
        bundle = _bundle(unknowns=unknowns)
        messages = _build_messages(bundle, [0, 1, 2, 3])
        user_content = messages[1]["content"]
        assert "Unknown actions" in user_content
        assert "fingerprint_abc" in user_content

    def test_build_messages_without_unknowns(self) -> None:
        """No 'Unknown actions' section when bundle has no unknowns."""
        bundle = _bundle()  # no unknowns key
        messages = _build_messages(bundle, [0, 1, 2, 3])
        user_content = messages[1]["content"]
        assert "Unknown actions" not in user_content

    def test_build_messages_empty_unknowns_list(self) -> None:
        """No 'Unknown actions' section when unknowns is an empty list."""
        bundle = _bundle(unknowns=[])
        messages = _build_messages(bundle, [0, 1, 2, 3])
        user_content = messages[1]["content"]
        assert "Unknown actions" not in user_content

    # -----------------------------------------------------------------------
    # _parse_response
    # -----------------------------------------------------------------------

    def test_parse_markdown_json_block(self) -> None:
        """Extracts JSON from ```json ... ``` block."""
        raw = 'Here is the plan:\n```json\n{"target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}, "max_steps": 50, "reason": "test"}\n```'
        result = _parse_response(raw)
        assert result is not None
        assert result["target"]["of"] == 0
        assert result["max_steps"] == 50

    def test_parse_raw_json(self) -> None:
        """Extracts raw JSON string (no markdown wrapper)."""
        raw = '{"target": {"dim": "pos", "of": 0, "eq": [5, 10]}, "max_steps": 30, "reason": "raw test"}'
        result = _parse_response(raw)
        assert result is not None
        assert result["target"]["dim"] == "pos"
        assert result["max_steps"] == 30

    def test_parse_garbage_returns_none(self) -> None:
        """Unparseable text returns None."""
        result = _parse_response("I don't know what to do here")
        assert result is None

    def test_parse_multiple_blocks_first_valid(self) -> None:
        """Multiple code blocks: takes first valid JSON."""
        raw = '```json\n{"first": true}\n```\nSome text\n```json\n{"second": true}\n```'
        result = _parse_response(raw)
        assert result is not None
        assert result == {"first": True}

    def test_parse_empty_string_returns_none(self) -> None:
        """Empty string returns None."""
        result = _parse_response("")
        assert result is None

    def test_parse_invalid_json_in_block_fallback(self) -> None:
        """Invalid JSON in markdown block falls through to next block or raw parse."""
        raw = '```json\n{invalid json}\n```\n```json\n{"valid": true}\n```'
        result = _parse_response(raw)
        assert result is not None
        assert result == {"valid": True}

    # -----------------------------------------------------------------------
    # _validate_goal
    # -----------------------------------------------------------------------

    def test_validate_valid_goal(self) -> None:
        """Valid entity IDs → returns ProbeGoal with correct fields."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            "max_steps": 50,
            "reason": "Entity 17 is unexplored",
        }
        scene_entities = {0, 17, 5}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target == goal_dict["target"]
        assert result.max_steps == 50
        assert result.reason == "Entity 17 is unexplored"

    def test_validate_rejects_old_predicate_key(self) -> None:
        """Goals using old 'predicate' key (instead of 'target') → None."""
        goal_dict = {
            "predicate": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
            "max_steps": 50,
            "reason": "Entity 17 is unexplored",
        }
        scene_entities = {0, 17, 5}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    def test_validate_invalid_entity_in_of(self) -> None:
        """Invalid entity ID in 'of' → None."""
        goal_dict = {
            "target": {"dim": "pos", "of": 99, "eq": [5, 10]},
            "max_steps": 50,
            "reason": "navigate",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    def test_validate_invalid_entity_in_near(self) -> None:
        """Invalid entity ID in 'near' dict → None."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": {"of": 99, "radius": 3}},
            "max_steps": 50,
            "reason": "navigate",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    def test_validate_missing_required_keys(self) -> None:
        """Missing 'reason' or 'max_steps' → None."""
        scene_entities = {0, 17}
        # Missing "reason"
        goal_no_reason = {
            "target": {"dim": "pos", "of": 0, "eq": [5, 10]},
            "max_steps": 50,
        }
        assert _validate_goal(goal_no_reason, scene_entities) is None

        # Missing "max_steps"
        goal_no_steps = {
            "target": {"dim": "pos", "of": 0, "eq": [5, 10]},
            "reason": "navigate",
        }
        assert _validate_goal(goal_no_steps, scene_entities) is None

    def test_validate_target_not_dict(self) -> None:
        """Target is not a dict → None."""
        goal_dict = {
            "target": "not a dict",
            "max_steps": 50,
            "reason": "navigate",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    def test_validate_all_conjunction(self) -> None:
        """'all' conjunction with valid entity IDs → ProbeGoal."""
        goal_dict = {
            "target": {
                "all": [
                    {"dim": "pos", "of": 0, "eq": [5, 10]},
                    {"dim": "size", "of": 17, "eq": 8},
                ]
            },
            "max_steps": 100,
            "reason": "navigate and check size",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target == goal_dict["target"]
        assert result.max_steps == 100
        assert result.reason == "navigate and check size"

    def test_validate_with_valid_action(self) -> None:
        """Valid 'action' field (integer) → ProbeGoal (action validated but not stored yet)."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
            "action": 3,
            "max_steps": 50,
            "reason": "probe action 3 near entity",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target == goal_dict["target"]
        assert result.max_steps == 50

    def test_validate_without_action(self) -> None:
        """No 'action' field → ProbeGoal returned (action is optional)."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
            "max_steps": 50,
            "reason": "navigate to area",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target == goal_dict["target"]

    def test_validate_action_null(self) -> None:
        """'action': null → ProbeGoal returned (action=None)."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
            "action": None,
            "max_steps": 50,
            "reason": "navigate to area",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is not None
        assert isinstance(result, ProbeGoal)

    def test_validate_rejects_string_action(self) -> None:
        """'action' as string → None (invalid type)."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
            "action": "three",
            "max_steps": 50,
            "reason": "navigate to area",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    def test_validate_rejects_float_action(self) -> None:
        """'action' as float → None (invalid type)."""
        goal_dict = {
            "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
            "action": 3.5,
            "max_steps": 50,
            "reason": "navigate to area",
        }
        scene_entities = {0, 17}
        result = _validate_goal(goal_dict, scene_entities)
        assert result is None

    # -----------------------------------------------------------------------
    # call_planner (end-to-end with mock llm_call)
    # -----------------------------------------------------------------------

    def test_call_planner_valid_response(self) -> None:
        """Mock returns valid JSON → ProbeGoal."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
                "max_steps": 50,
                "reason": "Entity 17 is unexplored — navigate close to observe",
            }
        )
        bundle = _bundle()
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target["of"] == 0
        assert result.target["near"]["of"] == 17
        assert result.max_steps == 50
        assert "unexplored" in result.reason

    def test_call_planner_list_entities_format(self) -> None:
        """Entity list format (actual scene.summary() output) validates correctly."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
                "max_steps": 50,
                "reason": "Entity 17 is unexplored — navigate close to observe",
            }
        )
        bundle = _bundle(entities_format="list")
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target["of"] == 0
        assert result.target["near"]["of"] == 17

    def test_call_planner_list_entities_rejects_invalid(self) -> None:
        """List entity format rejects goals referencing nonexistent entities."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 99, "near": {"of": 17, "radius": 3}},
                "max_steps": 50,
                "reason": "navigate to nonexistent entity 99",
            }
        )
        bundle = _bundle(entities_format="list")
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is None

    def test_call_planner_garbage_response(self) -> None:
        """Mock returns garbage → None."""
        bundle = _bundle()
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call("I am not JSON"))
        assert result is None

    def test_call_planner_invalid_entity(self) -> None:
        """Mock returns JSON with invalid entity → None."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 99, "eq": [5, 10]},
                "max_steps": 50,
                "reason": "navigate to nonexistent",
            }
        )
        bundle = _bundle()  # has entities 0 and 17 only
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is None

    def test_call_planner_with_failure_context(self) -> None:
        """Failure context is passed through to messages."""
        failure = {"type": "prediction_failure", "detail": "no path found"}
        bundle = _bundle()
        llm_call = _mock_llm_call(
            json.dumps(
                {
                    "target": {"dim": "pos", "of": 0, "eq": [5, 10]},
                    "max_steps": 30,
                    "reason": "retry",
                }
            )
        )

        result = call_planner(bundle, [0, 1, 2, 3], llm_call, failure_context=failure)

        # Verify llm_call was invoked and the messages contain failure context
        llm_call.assert_called_once()
        messages = llm_call.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "Failure context" in user_msg
        assert "prediction_failure" in user_msg
        assert result is not None

    def test_call_planner_missing_keys(self) -> None:
        """Mock returns JSON missing 'reason' → None."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 0, "eq": [5, 10]},
                "max_steps": 50,
            }
        )
        bundle = _bundle()
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is None

    def test_call_planner_with_action(self) -> None:
        """Mock returns valid JSON with action field → ProbeGoal."""
        response_json = json.dumps(
            {
                "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
                "action": 3,
                "max_steps": 50,
                "reason": "probe action 3",
            }
        )
        bundle = _bundle()
        result = call_planner(bundle, [0, 1, 2, 3], _mock_llm_call(response_json))
        assert result is not None
        assert isinstance(result, ProbeGoal)
        assert result.target["of"] == 0
