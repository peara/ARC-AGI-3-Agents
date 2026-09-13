"""ReplayMarker + embedded-state reader + flow-message parser tests.

Anchor: 2026-09-08 run eba2a894-9ba7-4dc2-967c-c00eaf3c0f55 (ls20,
simulatorfirst). The recording is a gitignored dev artifact — tests over its
real data skip with an explicit reason when absent. Pure-logic tests
(marker construction, parser on synthetic messages) run unconditionally.

Replaces the dead first-generation incident tests (that recording was deleted).
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


# ── Machinery tests (mini fixture) ────────────────────────────────────────
#
# Run UNCONDITIONALLY against the committed synthetic fixtures in
# ``fixtures/`` (generated by ``fixtures/gen_mini.py`` over the real local
# ls20 env — no real recording needed, no LLM calls, no network). The
# fixture carries NO embedded simulator_state blocks, so per-frame
# verification degrades gracefully (all _embedded_state reads return None).
#
# Marker for every test: ReplayMarker(turn_frame=5, turn_seq=11).
# Expected walk outcome: _actions == [0, 1, 1, 2, 1, 1] (RESET + 5),
# len(_grids) == 7 (virtual pair + 5), set_ignore_source is None,
# _simulate_source == set_simulate #2's def block, notes == mini notes/plan,
# workflow phase == MODEL.

import logging  # noqa: E402

from agents.simulator_agent.reconstruction import (  # noqa: E402
    reconstruct,
    verify_reconstruction,
)
from agents.simulator_agent.reset_policy import RESET_ACTION  # noqa: E402
from agents.simulator_agent.workflow import Phase  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_MINI_RECORDING = _FIXTURES / "mini.recording.jsonl"
_MINI_LLM = _FIXTURES / "mini.llm.jsonl"
_MINI_MARKER = ReplayMarker(turn_frame=5, turn_seq=11)

# The def block inspect.getsource captures for set_simulate #2 (the trailing
# set_simulate(simulate) call is not part of the function's source).
_SIM2_DEF = (
    "def simulate(grid, action):\n"
    "    g = [row[:] for row in grid]\n"
    "    if action == 1:\n"
    "        g[0][0] = (g[0][0] + 1) % 16\n"
    "    return g\n"
)


@pytest.fixture(scope="module")
def mini_state() -> Any:
    """Reconstructed state from the committed mini fixtures (walk once)."""
    if not _MINI_RECORDING.exists() or not _MINI_LLM.exists():
        pytest.fail(
            "mini fixtures missing — regenerate with "
            "uv run python tests/unit/simulator_agent/fixtures/gen_mini.py"
        )
    return reconstruct(_MINI_RECORDING, marker=_MINI_MARKER, seed=0)


class TestMachinery:
    def test_machinery_batch_row_consumes_two_lines(self, mini_state):
        """The seq-3 batch row (action(1); action(1)) consumes recording
        lines 1 AND 2 — per-action line consumption, not per-row."""
        assert mini_state.sandbox._actions == [RESET_ACTION, 1, 1, 2, 1, 1]  # noqa: SLF001
        assert len(mini_state.sandbox._grids) == 7  # noqa: SLF001

    def test_machinery_end_to_end_mini_fixture(self, mini_state):
        """reconstruct(mini) runs clean; no embedded state → zero mismatches
        (graceful degradation); verify_reconstruction also runs without
        raising (emb=None paths)."""
        # reconstruct() already raised if the walk mismatched; assert the
        # corpus shape the corrected mapping predicts at turn_frame=5.
        assert len(mini_state.sandbox._grids) == 7  # noqa: SLF001
        assert len(mini_state.sandbox._actions) == 6  # noqa: SLF001
        assert len(mini_state.history_turns) == 6
        checks = verify_reconstruction(mini_state)
        assert checks["corpus_shape"]["ok"] is True
        assert checks["corpus_shape"]["embedded_n_collected_frames"] is None

    def test_machinery_no_set_ignore_source_is_none(self, mini_state):
        """No set_ignore python call in the fixture → set_ignore_source None."""
        assert mini_state.set_ignore_source is None

    def test_machinery_second_set_simulate_replaces_first(self, mini_state):
        """set_simulate #2 replaces #1: _simulate_source == source #2."""
        assert mini_state.simulate_source == _SIM2_DEF
        assert mini_state.simulate_source != (
            "def simulate(grid, action):\n    return [row[:] for row in grid]\n"
        )

    def test_machinery_update_notes_and_set_phase_land(self, mini_state):
        """update_notes → state.notes (world_model semantics — NOT
        sandbox._pending_notes, which run_code wipes); set_phase → MODEL."""
        assert mini_state.notes == {"notes": "mini notes", "plan": "mini plan"}
        assert mini_state.workflow._phase == Phase.MODEL  # noqa: SLF001

    def test_machinery_null_tool_calls_row_skipped(self, mini_state):
        """Seq-7 row (tool_calls: null) skipped without crash or state effect:
        the walk completed and the corpus matches the 5 scripted actions."""
        assert mini_state.sandbox._actions == [RESET_ACTION, 1, 1, 2, 1, 1]  # noqa: SLF001
        assert mini_state.messages == [{"role": "user", "content": "mini marked turn"}]

    def test_machinery_ok_false_row_skipped_with_warning(
        self, mini_state, caplog
    ):
        """Seq-8 ok=False row skipped with a logger.warning; its code never
        ran (no 'should not run' side effect possible — actions still match)."""
        with caplog.at_level(logging.WARNING, logger="agents.simulator_agent.reconstruction"):
            state = reconstruct(_MINI_RECORDING, marker=_MINI_MARKER, seed=0)
        assert any(
            "ok=False" in rec.message and "8" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]
        assert state.sandbox._actions == [RESET_ACTION, 1, 1, 2, 1, 1]  # noqa: SLF001

    def test_machinery_action_id_guard_loud_error(self, tmp_path):
        """Corrupted fixture copy (line 3's action id 2→3) → loud
        RuntimeError 'action-id mismatch' from the walker's guard."""
        events = [
            json.loads(line)
            for line in _MINI_RECORDING.read_text().splitlines()
            if line.strip()
        ]
        assert events[3]["data"]["action_input"]["id"] == 2
        events[3]["data"]["action_input"]["id"] = 3
        corrupt_rec = tmp_path / "mini.recording.jsonl"
        corrupt_rec.write_text("\n".join(json.dumps(ev) for ev in events) + "\n")
        corrupt_llm = tmp_path / "mini.llm.jsonl"
        corrupt_llm.write_text(_MINI_LLM.read_text())

        with pytest.raises(RuntimeError, match="action-id mismatch"):
            reconstruct(corrupt_rec, marker=_MINI_MARKER, seed=0)

    def test_machinery_queue_drain_falls_back_to_env_step(self, mini_state):
        """After the walk, sandbox.action(1) steps past turn_frame → the
        queue-drain env.step fallback fires → corpus grows to 8 grids."""
        out, err, _ = mini_state.sandbox.run_code("action(1)\n")
        assert err is None
        assert len(mini_state.sandbox._grids) == 8  # noqa: SLF001
        assert mini_state.sandbox._actions[-1] == 1  # noqa: SLF001


# ── Fidelity tests (eba2a894 recording) ───────────────────────────────────
#
# Data-driven: expected values are parsed FROM the recording (embedded
# simulator_state at the marked line, flow facts from the marked turn's
# messages). Pinned literals {49, 2, 5, 724, 53, MODEL} are tripwires
# guarding against reader/parser bugs — everything else derives.

_MARKER = ReplayMarker(turn_frame=12, turn_seq=18)


@pytest.fixture(scope="module")
def fidelity_state():
    if not _RECORDING.exists():
        pytest.skip(
            "gitignored dev artifact: eba2a894 recording not present; "
            "machinery tests in fixtures/ run without it"
        )
    return reconstruct(_RECORDING, marker=_MARKER, seed=0)


@pytest.fixture(scope="module")
def emb12(recording_lines):
    emb = _embedded_state(recording_lines[_MARKER.turn_frame])
    assert emb is not None, "marked line must carry embedded simulator_state"
    return emb


@pytest.fixture(scope="module")
def marked_messages(llm_rows):
    by_seq = {r["seq"]: r for r in llm_rows}
    return by_seq[_MARKER.turn_seq]["messages"]


class TestFidelity:
    def test_corpus_shape_matches_pinned_mapping(self, fidelity_state, emb12):
        """POST-append offsets: the marked line's own transition appends
        AFTER the embedded capture (PRE-action timing), so grids = n+1,
        actions = n, history = h+1."""
        assert len(fidelity_state.sandbox._grids) == emb12.n_collected_frames + 1  # noqa: SLF001
        assert len(fidelity_state.sandbox._actions) == emb12.n_collected_frames  # noqa: SLF001
        assert len(fidelity_state.history_turns) == emb12.history_turns + 1
        assert verify_reconstruction(fidelity_state)["corpus_shape"]["ok"] is True

    def test_simulate_source_byte_identical(self, fidelity_state, emb12):
        """Walker-registered simulate == embedded source, byte-equal."""
        assert fidelity_state.sandbox._simulate_source == emb12.simulate_source  # noqa: SLF001
        assert len(fidelity_state.sandbox._simulate_source) == 724  # noqa: SLF001

    def test_stale_check_is_cached_not_recomputed(self, fidelity_state, emb12):
        """The LLM sees the CACHED check, not a re-check — the cached
        _last_check_result must equal the embedded result (the stale
        5/5@100.0 snapshot, not a fresh 8/8 recomputation)."""
        cached = fidelity_state.sandbox._last_check_result  # noqa: SLF001
        assert cached is not None
        for key in ("overall_accuracy", "frames_correct", "frames_total"):
            assert cached[key] == emb12.last_check_result[key]
        assert (
            cached["frames_correct"],
            cached["frames_total"],
            cached["overall_accuracy"],
        ) == (5, 5, 100.0)

    def test_no_ignore_mask_before_incident(self, fidelity_state):
        """No set_ignore ran before the marked turn → source None, mask empty."""
        assert fidelity_state.set_ignore_source is None
        assert len(fidelity_state.sandbox._ignore_mask) == 0  # noqa: SLF001

    def test_conversation_prefix_verbatim(self, fidelity_state, marked_messages):
        """Conversation prefix is byte-equal to the recorded seq-18 messages."""
        assert fidelity_state.messages == marked_messages
        assert len(fidelity_state.messages) == 53
        assert parse_flow_message(fidelity_state.messages) is not None

    def test_pending_flow_matches_recorded_message(self, fidelity_state):
        """Re-fire predict_and_compare on the final transition: the pending
        flow (n_diff, n_regions) equals the parsed recorded flow facts."""
        from agents.simulator_agent.frame_layers import settled_board

        sandbox = fidelity_state.sandbox
        parsed = parse_flow_message(fidelity_state.messages)
        assert parsed is not None
        sandbox._pending_exception_flow = None  # noqa: SLF001
        sandbox.predict_and_compare(
            settled_board(fidelity_state.harness.frames[-2].frame),
            settled_board(fidelity_state.harness.frames[-1].frame),
            sandbox._actions[-1],  # noqa: SLF001
        )
        pending = sandbox._pending_exception_flow  # noqa: SLF001
        assert pending is not None
        assert (pending["n_diff"], pending["n_regions"]) == (
            parsed.n_cells,
            parsed.n_regions,
        )
        assert (pending["n_diff"], pending["n_regions"]) == (49, 2)

    def test_phase_derived_from_set_phase_calls(self, fidelity_state):
        """Phase comes from the walk's set_phase calls, cross-checked
        (advisory) against the recorded PHASE directive."""
        assert fidelity_state.workflow._phase == Phase.MODEL  # noqa: SLF001
        assert verify_reconstruction(fidelity_state)["phase_crosscheck"] == {
            "found": "MODEL",
            "matches": True,
        }

    def test_verify_dict_green_on_incident(self, fidelity_state):
        """Every verify_reconstruction check green on the real incident."""
        checks = verify_reconstruction(fidelity_state)
        assert checks["simulate_source_match"] is True
        assert checks["stale_check_match"] is True
        assert checks["has_simulate"] is True
        assert checks["flow_match"]["ok"] is True
        assert checks["corpus_shape"]["ok"] is True
        assert checks["phase"] == "MODEL"
        assert checks["n_messages"] == 53
        assert checks["last_user_has_exception"] is True

    def test_reconstruction_is_deterministic(self):
        """Two fresh reconstruct() runs produce identical state cores and
        identical verify dicts."""
        if not _RECORDING.exists():
            pytest.skip(
                "gitignored dev artifact: eba2a894 recording not present; "
                "machinery tests in fixtures/ run without it"
            )
        s1 = reconstruct(_RECORDING, marker=_MARKER, seed=0)
        s2 = reconstruct(_RECORDING, marker=_MARKER, seed=0)
        assert s1.sandbox._grids == s2.sandbox._grids  # noqa: SLF001
        assert s1.sandbox._actions == s2.sandbox._actions  # noqa: SLF001
        assert s1.messages == s2.messages
        assert s1.history_turns == s2.history_turns
        assert s1.notes == s2.notes
        assert s1.workflow._phase == s2.workflow._phase  # noqa: SLF001
        assert s1.sandbox._simulate_source == s2.sandbox._simulate_source  # noqa: SLF001
        assert verify_reconstruction(s1) == verify_reconstruction(s2)

    def test_per_frame_verification_all_lines(self, recording_lines, fidelity_state):
        """Ground truth existed at EVERY line 0..12, so the walker's
        per-line verification actually compared against real data;
        fidelity_state existing proves the walk raised zero mismatches
        (collect-then-raise)."""
        for i in range(_MARKER.turn_frame + 1):
            assert _embedded_state(recording_lines[i]) is not None, f"line {i}"