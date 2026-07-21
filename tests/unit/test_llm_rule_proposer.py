"""Tests for planning/llm_rule_proposer.py — real implementations."""

from __future__ import annotations

import json

from effects.rules import Effect, Rule
from planning.llm_planner import call_rule_proposer
from planning.llm_rule_proposer import (
    NULL_RULE_PROPOSER,
    SYSTEM_PROMPT,
    make_rule_proposer,
    parse_proposals,
    validate_proposal,
)

# ---------------------------------------------------------------------------
# parse_proposals
# ---------------------------------------------------------------------------


def test_parse_proposals_valid_json() -> None:
    """Valid JSON with a rules key returns a list of 1 dict."""
    raw = json.dumps(
        {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 3},
                    "effect": {"dim": "size", "of": 5, "delta": 1},
                    "support": 4,
                }
            ]
        }
    )
    result = parse_proposals(raw)
    assert len(result) == 1
    assert result[0]["kind"] == "delta"


def test_parse_proposals_with_fences() -> None:
    """Markdown fences around JSON should still be extracted."""
    raw = '```json\n{"rules": [{"kind": "delta", "guard": {"action": 1}, "effect": {"dim": "size", "of": 2, "delta": -1}, "support": 3}]}\n```'
    result = parse_proposals(raw)
    assert len(result) == 1
    assert result[0]["kind"] == "delta"


def test_parse_proposals_malformed_json() -> None:
    """Malformed JSON input returns empty list."""
    result = parse_proposals("not json at all")
    assert result == []


def test_parse_proposals_missing_rules_key() -> None:
    """JSON missing the 'rules' key returns empty list."""
    result = parse_proposals(json.dumps({"data": []}))
    assert result == []


# ---------------------------------------------------------------------------
# validate_proposal
# ---------------------------------------------------------------------------


def test_validate_proposal_valid_delta() -> None:
    """A well-formed delta proposal against valid entities returns a Rule."""
    proposal: dict = {
        "kind": "delta",
        "guard": {"action": 3},
        "effect": {"dim": "size", "of": 5, "delta": 1},
        "support": 4,
    }
    scene_entities: dict[int, dict] = {5: {"dim": "size"}}
    rule = validate_proposal(proposal, scene_entities)
    assert rule is not None
    assert isinstance(rule, Rule)
    assert rule.kind == "delta"
    assert rule.support == 4


def test_validate_proposal_invalid_entity_id() -> None:
    """A proposal referencing a non-existent entity returns None."""
    proposal: dict = {
        "kind": "delta",
        "guard": {"action": 3},
        "effect": {"dim": "size", "of": 99, "delta": 1},
        "support": 1,
    }
    scene_entities: dict[int, dict] = {5: {"dim": "size"}}
    result = validate_proposal(proposal, scene_entities)
    assert result is None


def test_validate_proposal_invalid_guard() -> None:
    """A proposal with an invalid guard structure returns None (parse_guard_clauses raises)."""
    # parse_guard_clauses expects "action" or "all" keys — a dict with neither
    # is still valid per guard_parse (it returns a clause with has_action=False).
    # So test with a kind that's invalid instead.
    proposal: dict = {
        "kind": "invalid_kind",
        "guard": {"action": 3},
        "effect": {"dim": "size", "of": 5, "delta": 1},
        "support": 1,
    }
    scene_entities: dict[int, dict] = {5: {"dim": "size"}}
    result = validate_proposal(proposal, scene_entities)
    assert result is None


# ---------------------------------------------------------------------------
# call_rule_proposer (integration)
# ---------------------------------------------------------------------------


def test_call_rule_proposer_returns_list_of_rules() -> None:
    """call_rule_proposer with a valid LLM response returns Rule objects."""
    valid_response = json.dumps(
        {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 3},
                    "effect": {"dim": "size", "of": 5, "delta": 1},
                    "support": 2,
                }
            ]
        }
    )

    def mock_llm_call(messages: list[dict[str, str]]) -> str:
        return valid_response

    bundle = {
        "scene": {"entities": [{"id": 5, "row": 0, "col": 0}]},
        "engine_rules": {"confirmed": []},
    }
    residual: list[dict[str, object]] = []

    result = call_rule_proposer(bundle, residual, mock_llm_call)
    assert len(result) == 1
    assert isinstance(result[0], Rule)
    assert result[0].kind == "delta"


def test_call_rule_proposer_dedups_existing() -> None:
    """call_rule_proposer deduplicates proposals that match existing confirmed rules."""
    # Create a confirmed rule: action 3 → delta size +1 on entity 5
    existing_rule = Rule(
        guard_spec={"action": 3},
        effects=(Effect("size", 5, "delta", 1),),
        support=3,
    )
    # The LLM proposes the same rule
    valid_response = json.dumps(
        {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 3},
                    "effect": {"dim": "size", "of": 5, "delta": 1},
                    "support": 2,
                }
            ]
        }
    )

    def mock_llm_call(messages: list[dict[str, str]]) -> str:
        return valid_response

    bundle = {
        "scene": {"entities": [{"id": 5, "row": 0, "col": 0}]},
        "engine_rules": {"confirmed": [existing_rule]},
    }
    residual: list[dict[str, object]] = []

    result = call_rule_proposer(bundle, residual, mock_llm_call)
    # Should be deduped — same key as existing confirmed rule
    assert len(result) == 0


# ---------------------------------------------------------------------------
# NULL_RULE_PROPOSER
# ---------------------------------------------------------------------------


def test_null_rule_proposer_returns_empty_list() -> None:
    """NULL_RULE_PROPOSER always returns an empty list."""
    assert NULL_RULE_PROPOSER() == []
    assert NULL_RULE_PROPOSER({}, [], None) == []


# ---------------------------------------------------------------------------
# make_rule_proposer
# ---------------------------------------------------------------------------


def test_make_rule_proposer_returns_callable() -> None:
    """make_rule_proposer returns a callable that delegates to call_rule_proposer."""
    call_count = 0

    valid_response = json.dumps(
        {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 3},
                    "effect": {"dim": "size", "of": 5, "delta": 1},
                    "support": 2,
                }
            ]
        }
    )

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return valid_response

    proposer = make_rule_proposer(mock_llm, cooldown=0.0)

    bundle = {
        "scene": {"entities": [{"id": 5, "row": 0, "col": 0}]},
        "engine_rules": {"confirmed": []},
    }
    residual: list[dict[str, object]] = []

    result = proposer(bundle, residual)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Rule)
    assert result[0].kind == "delta"
    assert call_count == 1


def test_make_rule_proposer_cooldown_returns_empty() -> None:
    """make_rule_proposer returns [] when called within the cooldown window."""
    call_count = 0

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps({"rules": []})

    proposer = make_rule_proposer(mock_llm, cooldown=60.0)

    _ = proposer({}, [])
    assert call_count == 1

    result2 = proposer({}, [])
    assert result2 == []
    assert call_count == 1  # LLM was NOT called again


# ---------------------------------------------------------------------------
# Prompt text: orientation uses integers, not compass letters
# ---------------------------------------------------------------------------


def test_prompt_orientation_uses_integers_not_compass() -> None:
    """SYSTEM_PROMPT must describe orientation as integer 0-3, not compass letters."""
    prompt_lower = SYSTEM_PROMPT.lower()
    # Must mention "integer" or "0-3" for orientation
    assert "integer" in prompt_lower or "0-3" in SYSTEM_PROMPT, (
        "Prompt should describe orientation as integer 0-3"
    )
    # Must NOT mention compass direction names
    assert "north" not in prompt_lower, "Prompt should not mention 'North'"
    assert "east" not in prompt_lower, "Prompt should not mention 'East'"
    assert "south" not in prompt_lower, "Prompt should not mention 'South'"
    assert "west" not in prompt_lower, "Prompt should not mention 'West'"


def test_prompt_presents_set_and_delta_neutrally() -> None:
    """SYSTEM_PROMPT must present both set and delta for orientation without bias."""
    # Both "set" and "delta" should appear in orientation description context
    assert '"set"' in SYSTEM_PROMPT, "Prompt should describe set option"
    assert '"delta"' in SYSTEM_PROMPT, "Prompt should describe delta option"
    # Should not say "prefer set" or "prefer delta"
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "prefer" not in prompt_lower or "prefer a" in prompt_lower, (
        "Prompt should not bias toward set or delta"
    )


# ---------------------------------------------------------------------------
# validate_proposal: orientation effects with integer values
# ---------------------------------------------------------------------------


def test_validate_proposal_orientation_set_integer() -> None:
    """Orientation effect with integer value should validate successfully."""
    proposal: dict = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "orientation", "of": 1, "op": "set", "value": 2}],
        "support": 3,
    }
    scene_entities: dict[int, dict] = {1: {"dim": "orientation"}}
    rule = validate_proposal(proposal, scene_entities)
    assert rule is not None
    assert isinstance(rule, Rule)
    assert rule.kind == "movement"
    assert rule.effects[0].dim == "orientation"
    assert rule.effects[0].value == 2


def test_validate_proposal_orientation_delta_integer() -> None:
    """Orientation delta effect with integer value should validate."""
    proposal: dict = {
        "kind": "movement",
        "guard": {"action": 2},
        "effects": [{"dim": "orientation", "of": 1, "op": "delta", "value": 1}],
        "support": 3,
    }
    scene_entities: dict[int, dict] = {1: {"dim": "orientation"}}
    rule = validate_proposal(proposal, scene_entities)
    assert rule is not None
    assert isinstance(rule, Rule)
    assert rule.effects[0].op == "delta"
    assert rule.effects[0].value == 1


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT sections (Task 8)
# ---------------------------------------------------------------------------


def test_system_prompt_contains_refuted_rules_section() -> None:
    """SYSTEM_PROMPT must include guidance about refuted rules."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "refuted" in prompt_lower, "SYSTEM_PROMPT missing 'refuted' section"
    assert "refuted_rules" in prompt_lower or "refuted rules" in prompt_lower, (
        "SYSTEM_PROMPT missing 'refuted_rules' or 'Refuted rules' section header"
    )


def test_system_prompt_contains_counter_evidence_section() -> None:
    """SYSTEM_PROMPT must include guidance about counter-evidence."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "counter-evidence" in prompt_lower or "counter evidence" in prompt_lower, (
        "SYSTEM_PROMPT missing 'counter-evidence' section"
    )


def test_system_prompt_contains_over_specialization_guidance() -> None:
    """SYSTEM_PROMPT must warn against over-specialization / overfitting."""
    prompt_lower = SYSTEM_PROMPT.lower()
    assert "general" in prompt_lower, "SYSTEM_PROMPT missing 'general' keyword"
    assert (
        "overfit" in prompt_lower
        or "over-specif" in prompt_lower
        or "specific guard" in prompt_lower
    ), "SYSTEM_PROMPT missing over-specialization / overfitting warning"


# ---------------------------------------------------------------------------
# Validation loop (Task 7)
# ---------------------------------------------------------------------------


def _make_llm_response(rule_dict: dict) -> str:
    """Wrap a rule dict in the expected JSON response format."""
    return json.dumps({"rules": [rule_dict]})


def test_call_rule_proposer_backward_compat_no_history() -> None:
    """Without history/ctx/spec, call_rule_proposer skips validation."""
    mv_rule = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [-1, 0]}],
        "support": 3,
    }
    bundle = {
        "scene": {"frame_idx": 0, "n_observed": 1, "entities": [{"id": 0, "pos": [5, 5]}]},
        "engine_rules": {"confirmed": [], "proposed": []},
    }
    call_count = 0

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return _make_llm_response(mv_rule)

    result = call_rule_proposer(bundle, [], mock_llm)
    assert len(result) == 1
    assert result[0].kind == "movement"
    assert call_count == 1, "Should call LLM exactly once without validation"


def test_validation_loop_passes_first_try() -> None:
    """When proposed rules pass validation, no retries needed."""
    from effects.context import EffectContext
    from effects.engine import _ProjectionSpec
    from effects.transition_history import TransitionHistory
    from effects.state import SceneState

    mv_rule_dict = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [-1, 0]}],
        "support": 3,
    }
    bundle = {
        "scene": {"frame_idx": 0, "n_observed": 1, "entities": [{"id": 0, "pos": [5, 5]}]},
        "engine_rules": {"confirmed": [], "proposed": []},
    }
    call_count = 0

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return _make_llm_response(mv_rule_dict)

    mv_rule = Rule(
        guard_spec={"action": 1},
        effects=(Effect("pos", 0, "delta", (-1, 0)),),
        support=1,
        kind="movement",
    )
    ctx = EffectContext(movement_rules=(mv_rule,), available_actions=(1,))
    history = TransitionHistory()
    history.append(
        state_before=SceneState(relevant=((0, ("pos", (5, 5))),)),
        action=1,
        state_after=SceneState(relevant=((0, ("pos", (4, 5))),)),
        frame_idx=0,
    )
    spec = _ProjectionSpec(
        entities=(0,), dims=("pos",), include_terminal=False
    )

    result = call_rule_proposer(
        bundle, [], mock_llm,
        history=history, ctx=ctx, spec=spec,
    )
    assert len(result) >= 1, "Should return at least one validated rule"
    assert call_count == 1, "Should call LLM exactly once when validation passes"


def test_validation_loop_retries_with_counter_evidence() -> None:
    """When rules fail validation, LLM is called again with counter-evidence."""
    from effects.context import EffectContext
    from effects.engine import _ProjectionSpec
    from effects.transition_history import TransitionHistory
    from effects.state import SceneState

    wrong_rule_dict = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [99, 99]}],
        "support": 3,
    }
    correct_rule_dict = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [-1, 0]}],
        "support": 3,
    }
    bundle = {
        "scene": {"frame_idx": 0, "n_observed": 1, "entities": [{"id": 0, "pos": [5, 5]}]},
        "engine_rules": {"confirmed": [], "proposed": []},
    }
    call_count = 0

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(wrong_rule_dict)
        return _make_llm_response(correct_rule_dict)

    # Existing movement rule so predict() returns known (not unknown)
    existing_rule = Rule(
        guard_spec={"action": 1},
        effects=(Effect("pos", 0, "delta", (-1, 0)),),
        support=1,
        kind="movement",
    )
    ctx = EffectContext(movement_rules=(existing_rule,), available_actions=(1,))
    history = TransitionHistory()
    history.append(
        state_before=SceneState(relevant=((0, ("pos", (5, 5))),)),
        action=1,
        state_after=SceneState(relevant=((0, ("pos", (4, 5))),)),
        frame_idx=0,
    )
    spec = _ProjectionSpec(
        entities=(0,), dims=("pos",), include_terminal=False
    )

    result = call_rule_proposer(
        bundle, [], mock_llm,
        history=history, ctx=ctx, spec=spec,
    )
    assert call_count >= 2, f"Should retry LLM on validation failure, got {call_count} calls"


def test_validation_loop_drops_after_max_retries() -> None:
    """When LLM always returns wrong rules, max retries reached, returns empty."""
    from effects.context import EffectContext
    from effects.engine import _ProjectionSpec
    from effects.transition_history import TransitionHistory
    from effects.state import SceneState

    wrong_rule_dict = {
        "kind": "movement",
        "guard": {"action": 1},
        "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [99, 99]}],
        "support": 3,
    }
    bundle = {
        "scene": {"frame_idx": 0, "n_observed": 1, "entities": [{"id": 0, "pos": [5, 5]}]},
        "engine_rules": {"confirmed": [], "proposed": []},
    }
    call_count = 0

    def mock_llm(messages: list[dict[str, str]]) -> str:
        nonlocal call_count
        call_count += 1
        return _make_llm_response(wrong_rule_dict)

    existing_rule = Rule(
        guard_spec={"action": 1},
        effects=(Effect("pos", 0, "delta", (-1, 0)),),
        support=1,
        kind="movement",
    )
    ctx = EffectContext(movement_rules=(existing_rule,), available_actions=(1,))
    history = TransitionHistory()
    history.append(
        state_before=SceneState(relevant=((0, ("pos", (5, 5))),)),
        action=1,
        state_after=SceneState(relevant=((0, ("pos", (4, 5))),)),
        frame_idx=0,
    )
    spec = _ProjectionSpec(
        entities=(0,), dims=("pos",), include_terminal=False
    )

    result = call_rule_proposer(
        bundle, [], mock_llm,
        history=history, ctx=ctx, spec=spec,
    )
    assert call_count <= 3, f"Should not exceed max 3 retries, got {call_count}"
    # After max retries with failing rules, result should be empty
    # (all proposed rules fail validation)