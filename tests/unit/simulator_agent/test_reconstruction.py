"""ReplayMarker + embedded-state reader + flow-message parser tests.

Anchor: 2026-09-08 run eba2a894-9ba7-4dc2-967c-c00eaf3c0f55 (ls20,
simulatorfirst). The recording is a gitignored dev artifact — tests over its
real data skip with an explicit reason when absent. Pure-logic tests
(marker construction, parser on synthetic messages) run unconditionally.

Replaces the dead 1786060d incident tests (recording deleted).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.reconstruction import (
    EmbeddedState,
    FlowFacts,
    ReplayMarker,
    _embedded_state,
    _row_tool_calls,
    load_llm_rows,
    parse_flow_message,
)

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "eba2a894-9ba7-4dc2-967c-c00eaf3c0f55.recording.jsonl"
)

pytestmark = pytest.mark.unit


# ── Fixtures ──────────────────────────────────────────────────────────────


def _load_lines() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _RECORDING.read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def recording_lines() -> list[dict[str, Any]]:
    if not _RECORDING.exists():
        pytest.skip("gitignored dev artifact: eba2a894 recording not present")
    return _load_lines()


@pytest.fixture(scope="module")
def llm_rows() -> list[dict[str, Any]]:
    llm_path = Path(str(_RECORDING).replace(".recording.jsonl", ".llm.jsonl"))
    if not llm_path.exists():
        pytest.skip("gitignored dev artifact: eba2a894 llm log not present")
    return load_llm_rows(llm_path)


# ── ReplayMarker (pure logic) ─────────────────────────────────────────────


class TestReplayMarker:
    def test_constructs_with_both_fields(self):
        marker = ReplayMarker(turn_frame=12, turn_seq=18)
        assert marker.turn_frame == 12
        assert marker.turn_seq == 18

    def test_no_defaults(self):
        with pytest.raises(TypeError):
            ReplayMarker()  # type: ignore[call-arg]

    def test_turn_frame_required(self):
        with pytest.raises(TypeError):
            ReplayMarker(turn_seq=18)  # type: ignore[call-arg]

    def test_turn_seq_required(self):
        with pytest.raises(TypeError):
            ReplayMarker(turn_frame=12)  # type: ignore[call-arg]

    def test_frozen(self):
        marker = ReplayMarker(turn_frame=12, turn_seq=18)
        with pytest.raises(Exception):
            marker.turn_frame = 13  # type: ignore[misc]


# ── _embedded_state (real eba2a894 data) ──────────────────────────────────


class TestEmbeddedState:
    def test_line12_all_six_fields_populated(self, recording_lines):
        state = _embedded_state(recording_lines[12])
        assert state is not None
        assert state.has_simulate is True
        assert state.n_collected_frames == 13
        assert state.history_turns == 12  # int COUNT, not a list
        lcr = state.last_check_result
        assert lcr is not None
        assert (lcr["frames_correct"], lcr["frames_total"]) == (5, 5)
        assert state.simulate_source is not None
        assert len(state.simulate_source) == 724
        wm = state.world_model
        assert isinstance(wm, dict)
        assert isinstance(wm.get("notes"), str) and len(wm["notes"]) > 0

    def test_line0_degenerate_no_exception(self, recording_lines):
        state = _embedded_state(recording_lines[0])
        assert state is not None
        assert state.has_simulate is False
        assert state.n_collected_frames == 0
        assert state.history_turns == 0
        assert state.simulate_source == ""
        assert state.last_check_result is None

    def test_absent_block_returns_none(self):
        assert _embedded_state({}) is None
        assert _embedded_state({"data": {}}) is None
        assert _embedded_state({"data": {"scene_state": {}}}) is None
        assert _embedded_state({"data": {"scene_state": {"simulator_state": None}}}) is None

    def test_missing_keys_read_as_none(self):
        state = _embedded_state(
            {"data": {"scene_state": {"simulator_state": {"has_simulate": True}}}}
        )
        assert state == EmbeddedState(
            world_model=None,
            history_turns=None,
            has_simulate=True,
            simulate_source=None,
            last_check_result=None,
            n_collected_frames=None,
        )


# ── parse_flow_message (real + synthetic) ─────────────────────────────────


class TestParseFlowMessage:
    def test_seq18_real_flow_message(self, llm_rows):
        by_seq = {r["seq"]: r for r in llm_rows}
        facts = parse_flow_message(by_seq[18]["messages"])
        assert facts == FlowFacts(
            n_cells=49,
            n_regions=2,
            regions=(
                # bbox ordering: (row_start, col_start, row_end, col_end)
                _region((10, 34, 19, 38), 47, ("3->9", "9->5")),
                _region((61, 23, 62, 23), 2, ("3->11",)),
            ),
        )

    def test_seq12_no_flow_message_returns_none(self, llm_rows):
        by_seq = {r["seq"]: r for r in llm_rows}
        assert parse_flow_message(by_seq[12]["messages"]) is None

    def test_empty_messages_returns_none(self):
        assert parse_flow_message([]) is None

    def test_last_flow_message_wins(self):
        messages = [
            {"role": "user", "content": "EXCEPTION FLOW — differs from reality by 3 cells in 1 regions:\n  rows 0-0 cols 0-0: 3 cells"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "EXCEPTION FLOW — differs from reality by 9 cells in 2 regions:\n  rows 1-2 cols 3-4: 5 cells (1->2)\n  rows 5-5 cols 6-6: 4 cells"},
        ]
        facts = parse_flow_message(messages)
        assert facts is not None
        assert facts.n_cells == 9
        assert facts.n_regions == 2
        assert [r.bbox for r in facts.regions] == [(1, 3, 2, 4), (5, 6, 5, 6)]

    def test_region_without_transitions(self):
        messages = [
            {"role": "user", "content": "EXCEPTION FLOW — differs from reality by 4 cells in 1 regions:\n  rows 2-3 cols 4-4: 4 cells"},
        ]
        facts = parse_flow_message(messages)
        assert facts is not None
        assert facts.regions[0].transitions == ()

    def test_trailing_period_tolerated(self):
        messages = [
            {"role": "user", "content": "EXCEPTION FLOW — differs from reality by 2 cells in 1 regions:\n  rows 0-1 cols 2-2: 2 cells (0->1)."},
        ]
        facts = parse_flow_message(messages)
        assert facts is not None
        assert facts.n_cells == 2
        assert facts.regions[0].transitions == ("0->1",)

    def test_regions_preserve_message_order_not_sorted(self):
        messages = [
            {"role": "user", "content": "EXCEPTION FLOW — differs from reality by 6 cells in 2 regions:\n  rows 50-51 cols 1-2: 4 cells\n  rows 10-10 cols 3-3: 2 cells"},
        ]
        facts = parse_flow_message(messages)
        assert facts is not None
        assert [r.bbox for r in facts.regions] == [(50, 1, 51, 2), (10, 3, 10, 3)]


# ── _row_tool_calls (real + synthetic) ────────────────────────────────────


class TestRowToolCalls:
    def test_seq18_python_call(self, llm_rows):
        by_seq = {r["seq"]: r for r in llm_rows}
        calls = _row_tool_calls(by_seq[18])
        assert len(calls) == 1
        name, args = calls[0]
        assert name == "python"
        assert "code" in args

    def test_null_tool_calls(self):
        assert _row_tool_calls({"tool_calls": None}) == []
        assert _row_tool_calls({}) == []

    def test_non_dict_entries_skipped(self):
        row = {"tool_calls": ["garbage", 42, {"function": {"name": "python", "arguments": '{"code": "x=1"}'}}]}
        calls = _row_tool_calls(row)
        assert calls == [("python", {"code": "x=1"})]

    def test_multiple_calls_per_row(self):
        row = {
            "tool_calls": [
                {"function": {"name": "python", "arguments": '{"code": "a=1"}'}},
                {"function": {"name": "inspect", "arguments": "{}"}},
            ]
        }
        calls = _row_tool_calls(row)
        assert [name for name, _ in calls] == ["python", "inspect"]

    def test_malformed_arguments_become_empty_dict(self):
        row = {"tool_calls": [{"function": {"name": "python", "arguments": "not json"}}]}
        assert _row_tool_calls(row) == [("python", {})]


# ── Helper ────────────────────────────────────────────────────────────────


def _region(
    bbox: tuple[int, int, int, int], n_cells: int, transitions: tuple[str, ...]
) -> Any:
    from agents.simulator_agent.reconstruction import FlowRegion

    return FlowRegion(bbox=bbox, n_cells=n_cells, transitions=transitions)