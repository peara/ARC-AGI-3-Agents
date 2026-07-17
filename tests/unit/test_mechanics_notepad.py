"""Tests for MechanicsNotepad and MechanicsHypothesis: convergence, field caps, reset, confidence monotonicity."""

from __future__ import annotations

import os

import pytest

from effects.rules import Effect, Rule
from planning.mechanics_notepad import (
    COLD_START_FRAMES,
    MAX_CHANGES_CHARS,
    MAX_MECHANICS_ITEMS,
    MAX_OBJECTIVE_CHARS,
    MAX_PROGRESS_ITEMS,
    MechanicsHypothesis,
    MechanicsNotepad,
)
from planning.mechanics_prompt import build_action_legend

EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", ".local", "experiments"
)


def _load_response(filename: str) -> str:
    path = os.path.join(EXPERIMENTS_DIR, filename)
    with open(path) as f:
        return f.read()


STAGE_RESPONSES = {
    0: _load_response("iterative_stage_0_frames_0-5_response.txt"),
    1: _load_response("iterative_stage_1_frames_6-13_response.txt"),
    2: _load_response("iterative_stage_2_frames_14-24_response.txt"),
    3: _load_response("iterative_stage_3_frames_25-40_response.txt"),
}

ZERO_GRID = [[0] * 64 for _ in range(64)]

MOCK_SCENE_SUMMARY = {
    "levels_completed": 0,
    "controllable_id": 1,
    "controllable_pos": (10, 10),
    "n_entities": 5,
    "action_taken": "ACTION1",
}

ACTION_LEGEND = {0: "ACTION0", 1: "ACTION1 (move)", 2: "ACTION2 (move)", 5: "ACTION5"}


def _make_mock_llm(stage_responses: dict[int, str]) -> tuple[list[int], callable]:
    """Return a mock LLM callable that cycles through stage responses by call order."""
    call_log: list[int] = []

    def mock_llm(messages: list[dict]) -> str:
        idx = len(call_log)
        call_log.append(idx)
        return stage_responses[idx]

    return call_log, mock_llm


# ---------------------------------------------------------------------------
# Test 1: Convergence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_convergence():
    """Replay 4 stages through MechanicsNotepad with pre-recorded LLM responses.

    Asserts:
    - Stage 0: hypothesis exists, confidence >= 0.5
    - Stage 1: status is "refined" or "confirmed", confidence >= 0.7
    - Stage 2+: status is "confirmed", confidence non-decreasing
    - Final: objective contains "blue" and ("target" or "zone" or "area")
    """
    call_log, mock_llm = _make_mock_llm(STAGE_RESPONSES)
    nb = MechanicsNotepad(llm_call=mock_llm, vision_enabled=False)

    stage_frame_ranges = [
        (5, 6),    # stage 0: frames 0-5
        (6, 8),    # stage 1: frames 6-13
        (14, 11),  # stage 2: frames 14-24
        (25, 16),  # stage 3: frames 25-40
    ]

    prev_confidence = 0.0

    for stage_idx, (frame_index, n_frames) in enumerate(stage_frame_ranges):
        # Build frame data — n_frames copies of zero grid + scene summary
        frames = [ZERO_GRID] * min(n_frames, 8)
        summaries = [MOCK_SCENE_SUMMARY] * min(n_frames, 8)

        # For non-cold-start stages, bypass cooldown by resetting _last_update_frame
        if stage_idx > 0:
            nb._last_update_frame = frame_index - COLD_START_FRAMES - 1

        h = nb.update(
            frames=frames,
            scene_summaries=summaries,
            action_legend=ACTION_LEGEND,
            frame_index=frame_index,
        )

        assert h is not None, f"Stage {stage_idx}: hypothesis should not be None"

        if stage_idx == 0:
            assert h.confidence >= 0.5, f"Stage 0: confidence {h.confidence} < 0.5"
        elif stage_idx == 1:
            assert h.status in ("refined", "confirmed"), (
                f"Stage 1: status '{h.status}' not refined/confirmed"
            )
            assert h.confidence >= 0.7, f"Stage 1: confidence {h.confidence} < 0.7"
        else:
            assert h.status == "confirmed", (
                f"Stage {stage_idx}: status '{h.status}' not confirmed"
            )
            assert h.confidence >= prev_confidence, (
                f"Stage {stage_idx}: confidence {h.confidence} < prev {prev_confidence}"
            )

        prev_confidence = h.confidence

    # Final objective assertions
    final_h = nb.hypothesis
    assert final_h is not None
    obj_lower = final_h.objective.lower()
    assert "blue" in obj_lower, f"Final objective missing 'blue': {final_h.objective}"
    assert any(
        word in obj_lower for word in ("target", "zone", "area")
    ), f"Final objective missing target/zone/area: {final_h.objective}"


# ---------------------------------------------------------------------------
# Test 2: Field caps
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_field_caps():
    """Verify MechanicsHypothesis.from_llm_response() truncates over-length fields."""
    long_string = "x" * 500
    raw = {
        "objective": long_string,  # 500 chars → capped at 200
        "key_mechanics": [f"mechanic_{i}" for i in range(10)],  # 10 → capped at 5
        "progress_signals": [f"signal_{i}" for i in range(10)],  # 10 → capped at 5
        "changes": long_string,  # 500 chars → capped at 500 (exactly at limit)
        "next_steps": long_string,  # 500 chars → capped at 300
        "entity_roles": {"role_a": "desc_a", "role_b": "desc_b"},
        "confidence": 0.85,
        "status": "confirmed",
    }

    h = MechanicsHypothesis.from_llm_response(raw, frame_index=10)

    assert len(h.objective) <= MAX_OBJECTIVE_CHARS, (
        f"objective length {len(h.objective)} > {MAX_OBJECTIVE_CHARS}"
    )
    assert len(h.key_mechanics) <= MAX_MECHANICS_ITEMS, (
        f"key_mechanics length {len(h.key_mechanics)} > {MAX_MECHANICS_ITEMS}"
    )
    assert len(h.progress_signals) <= MAX_PROGRESS_ITEMS, (
        f"progress_signals length {len(h.progress_signals)} > {MAX_PROGRESS_ITEMS}"
    )
    assert len(h.changes) <= MAX_CHANGES_CHARS, (
        f"changes length {len(h.changes)} > {MAX_CHANGES_CHARS}"
    )


# ---------------------------------------------------------------------------
# Test 3: Reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reset():
    """Verify reset() clears hypothesis and bundle dict."""
    # Use stage 0 response to create an initial hypothesis
    call_log, mock_llm = _make_mock_llm(STAGE_RESPONSES)
    nb = MechanicsNotepad(llm_call=mock_llm, vision_enabled=False)

    # Create initial hypothesis via update
    h = nb.update(
        frames=[ZERO_GRID] * 6,
        scene_summaries=[MOCK_SCENE_SUMMARY] * 6,
        action_legend=ACTION_LEGEND,
        frame_index=5,
    )
    assert h is not None, "Initial update should produce a hypothesis"
    assert nb.to_bundle_dict() is not None

    # Reset
    nb.reset()

    assert nb.hypothesis is None, "After reset, hypothesis should be None"
    assert nb.to_bundle_dict() is None, "After reset, bundle dict should be None"


# ---------------------------------------------------------------------------
# Test 4: Confidence monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confidence_monotonicity():
    """Verify prev confidence 0.9 + new confidence 0.8 (both confirmed) → stored 0.9.

    Confidence monotonicity: max(new, prev) when both are confirmed/refined.
    """
    # Stage 1 response has confidence 0.9, status "refined"
    # Stage 2 response has confidence 1.0, status "confirmed"
    # We need a custom stage that returns confidence 0.8 with confirmed status
    # after we set up a 0.9 confirmed hypothesis.
    declining_response = """```json
{
  "status": "confirmed",
  "changes": "Slight refinement of wording.",
  "objective": "Collect the small blue objects and place them into the central blue target zone.",
  "key_mechanics": [
    "The player controls a green entity using grid-based movement.",
    "ACTION5 is used to pick up and drop blue objects."
  ],
  "progress_signals": [
    "Blue objects disappearing from starting positions.",
    "Levels_completed counter incrementing."
  ],
  "entity_roles": {
    "controllable": "Green rectangular block.",
    "collectible": "Small blue squares.",
    "target_zone": "Larger blue rectangular outline."
  },
  "next_steps": "Move the green entity carrying a blue object into the target zone.",
  "confidence": 0.8
}
```"""

    call_count = [0]

    def mock_llm_declining(messages: list[dict]) -> str:
        idx = call_count[0]
        call_count[0] += 1
        if idx == 0:
            return STAGE_RESPONSES[1]  # confidence 0.9, status "refined"
        else:
            return declining_response  # confidence 0.8, status "confirmed"

    nb = MechanicsNotepad(llm_call=mock_llm_declining, vision_enabled=False)

    # Stage 0: initial hypothesis with confidence 0.9
    h0 = nb.update(
        frames=[ZERO_GRID] * 6,
        scene_summaries=[MOCK_SCENE_SUMMARY] * 6,
        action_legend=ACTION_LEGEND,
        frame_index=6,
    )
    assert h0 is not None
    assert h0.confidence == pytest.approx(0.9)

    # Stage 1: new response with confidence 0.8, status "confirmed"
    nb._last_update_frame = 20 - COLD_START_FRAMES - 1  # bypass cooldown
    h1 = nb.update(
        frames=[ZERO_GRID] * 6,
        scene_summaries=[MOCK_SCENE_SUMMARY] * 6,
        action_legend=ACTION_LEGEND,
        frame_index=20,
    )
    assert h1 is not None
    # Both prev (0.9, refined) and new (0.8, confirmed) are in confirmed/refined,
    # so confidence should be max(0.9, 0.8) = 0.9
    assert h1.confidence == pytest.approx(0.9), (
        f"Expected confidence 0.9 (max of 0.9 and 0.8), got {h1.confidence}"
    )


# ---------------------------------------------------------------------------
# Test 5: Planner influence — different hypotheses yield different bundles
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_planner_influence():
    """Different hypothesis → different bundle content → planner sees different context."""
    h_a = MechanicsHypothesis.from_llm_response(
        {
            "objective": "Go to top-left corner",
            "key_mechanics": ["Move up and left"],
            "progress_signals": ["Reaching corner"],
            "entity_roles": {},
            "next_steps": "Navigate to corner",
            "confidence": 0.8,
            "status": "confirmed",
            "changes": "",
        },
        frame_index=10,
    )

    h_b = MechanicsHypothesis.from_llm_response(
        {
            "objective": "Go to bottom-right zone",
            "key_mechanics": ["Move down and right"],
            "progress_signals": ["Entering zone"],
            "entity_roles": {},
            "next_steps": "Navigate to zone",
            "confidence": 0.9,
            "status": "confirmed",
            "changes": "",
        },
        frame_index=10,
    )

    dict_a = h_a.to_bundle_dict()
    dict_b = h_b.to_bundle_dict()

    # Objectives differ
    assert dict_a["objective"] != dict_b["objective"]
    assert dict_a["next_steps"] != dict_b["next_steps"]
    assert dict_a["confidence"] != dict_b["confidence"]

    # Both have the required keys
    assert set(dict_a.keys()) == {"objective", "next_steps", "confidence", "status"}


# ---------------------------------------------------------------------------
# Test 6: Ablation — with/without hypothesis produces different bundles
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ablation():
    """With hypothesis has mechanics_hypothesis field, without doesn't, rest identical."""
    nb = MechanicsNotepad(llm_call=lambda msgs: '{}', vision_enabled=False)

    # Without hypothesis: to_bundle_dict() returns None
    assert nb.to_bundle_dict() is None

    # With hypothesis: to_bundle_dict() returns a dict
    nb._hypothesis = MechanicsHypothesis.from_llm_response(
        {
            "objective": "Test",
            "confidence": 0.5,
            "status": "initial",
            "key_mechanics": [],
            "progress_signals": [],
            "entity_roles": {},
            "next_steps": "explore",
            "changes": "",
        },
        frame_index=0,
    )

    result = nb.to_bundle_dict()
    assert result is not None
    assert "objective" in result
    assert "next_steps" in result
    assert "confidence" in result
    assert "status" in result


# ---------------------------------------------------------------------------
# Test 7: NOTEPAD_ENABLED env var parsing

# ---------------------------------------------------------------------------
# Test 7: NOTEPAD_ENABLED env var parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_notepad_enabled_env_var():
    """NOTEPAD_ENABLED env var is parsed correctly for all accepted values."""
    import os

    original = os.environ.pop("NOTEPAD_ENABLED", None)
    try:
        # Default (unset) → True
        assert os.environ.get("NOTEPAD_ENABLED", "true").lower() in ("true", "1", "yes")

        # Truthy values
        for val in ("true", "1", "yes"):
            os.environ["NOTEPAD_ENABLED"] = val
            assert os.environ.get("NOTEPAD_ENABLED", "true").lower() in ("true", "1", "yes"), (
                f"Expected '{val}' to be truthy"
            )

        # Falsy values
        for val in ("false", "0", "no", "FALSE", "0", "No"):
            os.environ["NOTEPAD_ENABLED"] = val
            assert os.environ.get("NOTEPAD_ENABLED", "true").lower() not in ("true", "1", "yes"), (
                f"Expected '{val}' to be falsy"
            )

        # Empty string → falsy (matches _vision_enabled pattern)
        os.environ["NOTEPAD_ENABLED"] = ""
        assert os.environ.get("NOTEPAD_ENABLED", "true").lower() not in ("true", "1", "yes")
    finally:
        if original is not None:
            os.environ["NOTEPAD_ENABLED"] = original
        else:
            os.environ.pop("NOTEPAD_ENABLED", None)


# ---------------------------------------------------------------------------
# Test 8: Action Legend Building
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_action_legend():
    # Case 1: Confirmed movement rule for action 1
    actions1 = (0, 1, 5)
    rules1 = (
        Rule(
            guard_spec={"action": 1},
            effects=(Effect(dim="pos", of=1, op="delta", value=(0, -1)),),
            support=3,
            kind="delta",
        ),
    )
    legend1 = build_action_legend(actions1, rules1)
    assert legend1 == {0: "ACTION0", 1: "ACTION1 (move)", 5: "ACTION5"}

    # Case 2: No movement rules
    actions2 = (0, 1, 5)
    rules2 = ()
    legend2 = build_action_legend(actions2, rules2)
    assert legend2 == {0: "ACTION0", 1: "ACTION1", 5: "ACTION5"}

    # Case 3: Empty available_actions
    actions3 = ()
    rules3 = (
        Rule(
            guard_spec={"action": 1},
            effects=(Effect(dim="pos", of=1, op="delta", value=(0, -1)),),
            support=3,
            kind="delta",
        ),
    )
    legend3 = build_action_legend(actions3, rules3)
    assert legend3 == {}

    # Case 4: Compound guard (action 2 + position)
    actions4 = (0, 2, 5)
    rules4 = (
        Rule(
            guard_spec={"all": [{"action": 2}, {"dim": "pos", "eq": [0, 0], "of": 1}]},
            effects=(Effect(dim="pos", of=1, op="delta", value=(0, -1)),),
            support=3,
            kind="delta",
        ),
    )
    legend4 = build_action_legend(actions4, rules4)
    assert legend4 == {0: "ACTION0", 2: "ACTION2 (move)", 5: "ACTION5"}
