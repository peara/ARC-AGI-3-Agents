"""Board extraction policy tests (class vs continuity).

Anchors:
- abb14eca (2026-09-22 cosmetic-claim incident) — 13 multi-layer stacks:
  11 HUD/blocked-move highlight flashes (game continues from the LAST
  layer — the class rule ingests the transient overlay as board state),
  2 full-screen board-reset flashes (FLASH_RESET — both policies keep the
  class path, no flash events).
- d91cdde0 (2026-09-08, via frame_layers docstring profiles) — highlights
  that continue from layer 0: continuity must stay byte-identical to the
  class rule there (synthetic stacks with the recorded profiles).

The recording is a gitignored dev artifact — anchored tests skip loudly
when absent; pure-logic tests run unconditionally.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.board_extraction import (
    FlashEvent,
    extract_board,
    flash_event_text,
    flash_log_text,
    region_key,
)
from agents.simulator_agent.frame_layers import (
    classify_stack,
    settled_board,
)
from agents.simulator_agent.sandbox import SimulatorSandbox

pytestmark = pytest.mark.unit

_ABB_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "abb14eca-e587-4aa0-ad7f-ec5cdb712e53.recording.jsonl"
)


def _load_abb_lines() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _ABB_RECORDING.read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def abb_lines() -> list[dict[str, Any]]:
    if not _ABB_RECORDING.exists():
        pytest.skip("gitignored dev artifact: abb14eca recording not present")
    return _load_abb_lines()


# ── synthetic d91cdde0-style highlight (continues from layer 0) ────────────


def _highlight_from_layer0() -> tuple[list[list[list[int]]], list[list[int]]]:
    """6-layer stack, all animation layers == base (layer 0), plus one
    highlighted last layer — the d91cdde0 shape. The game continues from
    layer 0 (profile [0,0,0,0,0,76]): continuity must pick layer 0 and
    emit NO flash event."""
    base = [[3 if (r + c) % 7 else 4 for c in range(64)] for r in range(64)]
    highlight = [row[:] for row in base]
    for r in range(10, 14):
        for c in range(30, 34):
            highlight[r][c] = 0
    stack = [[row[:] for row in base] for _ in range(5)] + [highlight]
    return stack, base


class TestContinuityOnSyntheticShapes:
    def test_d91cdde0_highlight_continues_from_layer0_no_event(self):
        stack, base = _highlight_from_layer0()
        assert classify_stack(stack).value == "highlight"
        board, event = extract_board(stack, base, "continuity")
        assert board == base  # layer 0 — identical to the class rule
        assert event is None

    def test_abb_shape_flash_picked_stable_and_reported(self):
        """abb14eca shape: 5 identical FLASHED layers + stable last layer.
        prev_board == the stable board (a blocked move: nothing moved)."""
        stable = [[3 if (r + c) % 5 else 4 for c in range(64)] for r in range(64)]
        flashed = [row[:] for row in stable]
        for r in range(9, 16):
            for c in range(34, 39):
                flashed[r][c] = 0  # white overlay
        stack = [[row[:] for row in flashed] for _ in range(5)] + [stable]
        assert classify_stack(stack).value == "highlight"

        board, event = extract_board(stack, stable, "continuity")
        assert board == stable  # NOT the flashed layer
        assert event is not None
        assert event.n_cells == 35
        assert event.regions == ((9, 15, 34, 38),)
        expected = Counter(
            (stable[r][c], 0) for r in range(9, 16) for c in range(34, 39)
        )
        assert event.transitions() == expected

    def test_no_anchor_falls_back_to_class_rule(self):
        stack, _ = _highlight_from_layer0()
        board, event = extract_board(stack, None, "continuity")
        assert board == settled_board(stack)
        assert event is None

    def test_malformed_anchor_falls_back_to_class_rule(self):
        stack, _ = _highlight_from_layer0()
        board, event = extract_board(stack, [[1, 2], [3]], "continuity")
        assert board == settled_board(stack)
        assert event is None

    def test_single_layer_identical_under_both_policies(self):
        base = [[1] * 4 for _ in range(4)]
        for policy in ("class", "continuity"):
            board, event = extract_board([base], base, policy)
            assert board == base
            assert event is None

    def test_empty_and_malformed_stack_total(self):
        assert extract_board(None, None, "continuity") == ([], None)
        assert extract_board([], None, "continuity") == ([], None)
        assert extract_board([["x"]], None, "continuity") == (["x"], None)

    def test_class_policy_never_emits_events(self):
        stable = [[3] * 8 for _ in range(8)]
        flashed = [row[:] for row in stable]
        flashed[0][0] = 0
        stack = [[row[:] for row in flashed] for _ in range(5)] + [stable]
        board, event = extract_board(stack, stable, "class")
        assert board == flashed  # the historical (transient) pick
        assert event is None


# ── abb14eca real events ──────────────────────────────────────────────────


class TestAbb14ecaExtraction:
    """Every multi-layer stack in the recording, walked with a running
    extraction anchor — the exact continuity walk the agent will do."""

    HIGHLIGHT_LINES = (17, 36, 37, 40, 41, 42, 56, 57, 66, 75, 76)
    FLASH_RESET_LINES = (49, 97)

    def test_all_hIGHLIGHT_flashes_pick_stable_layer(self, abb_lines):
        prev: list[list[int]] | None = None
        events: dict[int, FlashEvent] = {}
        boards: dict[int, list[list[int]]] = {}
        for i, line in enumerate(abb_lines):
            stack = line["data"]["frame"]
            if len(stack) == 1:
                board, _ = extract_board(stack, prev, "continuity")
                prev = board
                continue
            board, event = extract_board(stack, prev, "continuity")
            if event is not None:
                events[i] = event
                assert board != settled_board(stack), (
                    f"line {i}: continuity picked the class pick (overlay!)"
                )
            boards[i] = board
            prev = board

        # All 11 highlight flashes fired as events
        assert set(events) == set(self.HIGHLIGHT_LINES), sorted(events)

        # The two top-box blocked-entry flashes (the incident's signal):
        # rows 9-15 painted white on blocked top-box entry
        for i in (17, 66):
            ev = events[i]
            top_box = [c for c in ev.cells if 9 <= c[0] <= 15]
            assert top_box, f"line {i}: no rows-9-15 overlay cells"
            assert all(o == 0 for _, _, _, o in top_box)  # white overlay

        # The 9 timer flashes: rows 53-62 only (HUD bottom bar)
        for i in (36, 37, 40, 41, 42, 56, 57, 75, 76):
            ev = events[i]
            assert all(53 <= r <= 62 for r, c, s, o in ev.cells), (
                f"line {i}: overlay outside rows 53-62"
            )

        # Continuity boards are the layers the game continues from:
        # next frame's extraction differs from the stable pick by only the
        # next move's own changes (spot-check the blocked-move flash, line 17)
        stack17 = abb_lines[17]["data"]["frame"]
        next_frame = abb_lines[18]["data"]["frame"]
        prev16 = settled_board(abb_lines[16]["data"]["frame"])
        board17, ev17 = extract_board(stack17, prev16, "continuity")
        assert ev17 is not None
        assert board17 == stack17[-1]
        assert board17 != stack17[0]
        # line 18 is a plain move: its settled board continues from line 17's
        # stable layer — 52 cells of movement, NOT 112 of flash+move
        moved = sum(
            1
            for r in range(64)
            for c in range(64)
            if board17[r][c] != next_frame[0][r][c]
        )
        assert moved == 52

    def test_flash_reset_stacks_keep_class_path(self, abb_lines):
        for i in self.FLASH_RESET_LINES:
            stack = abb_lines[i]["data"]["frame"]
            prev = settled_board(abb_lines[i - 1]["data"]["frame"])
            board, event = extract_board(stack, prev, "continuity")
            assert board == settled_board(stack)  # restored board, last layer
            assert event is None

    def test_walk_is_pure(self, abb_lines):
        """extract_board never mutates the stack (boards alias layers, but
        the caller's frames must never be written through)."""
        for i in self.HIGHLIGHT_LINES:
            before = json.dumps(abb_lines[i]["data"]["frame"])
            extract_board(abb_lines[i]["data"]["frame"], None, "continuity")
            assert json.dumps(abb_lines[i]["data"]["frame"]) == before


# ── FlashEvent rendering and round-trip ────────────────────────────────────


class TestFlashEventRender:
    def test_text_mentions_cells_regions_and_probe_hint(self):
        ev = FlashEvent(
            cells=((9, 34, 3, 0), (9, 35, 3, 0), (53, 5, 5, 0)),
            regions=((9, 9, 34, 35), (53, 53, 5, 5)),
        )
        text = flash_event_text(ev)
        assert text.startswith("FLASH: 3 cells")
        assert "rows 9-9 cols 34-35" in text
        assert "rows 53-53 cols 5-5" in text
        assert "state feedback" in text

    def test_dict_round_trip(self):
        ev = FlashEvent(
            cells=((9, 34, 3, 0),),
            regions=((9, 9, 34, 34),),
        )
        rebuilt = FlashEvent.from_dict(ev.as_dict())
        assert rebuilt == ev

    def test_from_dict_malformed_is_none(self):
        assert FlashEvent.from_dict(None) is None
        assert FlashEvent.from_dict({"cells": "no"}) is None
        assert FlashEvent.from_dict({"cells": [["x"]], "regions": []}) is None


# ── Region dedup (timer-flash identity) ────────────────────────────────────


class TestRegionDedup:
    def test_overlaps_detects_shifted_timer_digits(self):
        """Timer digits shift ±1 cell between flashes — identity is overlap."""
        ev = FlashEvent(
            cells=((53, 5, 5, 0), (53, 6, 5, 0)),
            regions=((53, 53, 5, 6),),
        )
        assert ev.overlaps((53, 53, 6, 7)) is True  # shifted by one cell
        assert ev.overlaps((53, 53, 8, 9)) is False  # disjoint digits
        assert ev.overlaps((0, 0, 0, 0)) is False

    def test_region_key_is_sorted_bboxes(self):
        ev = FlashEvent(
            cells=((53, 5, 5, 0), (9, 34, 3, 0)),
            regions=((53, 53, 5, 5), (9, 9, 34, 34)),
        )
        assert region_key(ev) == ((9, 9, 34, 34), (53, 53, 5, 5))


# ── Flash log aggregate (flash_log_text) ────────────────────────────────────


def _timer_event(col: int = 5) -> FlashEvent:
    """HUD-timer-shaped flash: rows 53-62, one digit at ``col``."""
    cells = tuple(
        (r, c, 5, 0)
        for r in range(53, 63)
        for c in range(col, col + 2)
        if (r + c) % 3  # digit-ish pattern
    )
    return FlashEvent(
        cells=cells,
        regions=((53, 62, col, col + 1),),
    )


class TestFlashLogText:
    def test_empty_store_renders_empty(self):
        assert flash_log_text({}, []) == ""

    def test_repeated_timer_flashes_collapse_to_one_region_row(self):
        """Three timer flashes at shifted columns + one top-box flash →
        one timer row (3 fires) + one top-box row (1 fire), sorted by fire
        count."""
        events: dict[int, FlashEvent] = {
            4: _timer_event(5),
            9: _timer_event(6),  # shifted digit — same region
            14: _timer_event(5),
            20: FlashEvent(
                cells=((9, 34, 3, 0),),
                regions=((9, 9, 34, 34),),
            ),
        }
        actions = [2] * 25
        text = flash_log_text(events, actions)
        assert text.startswith("Flash log (4 transient overlay events")
        assert "fired 3x" in text
        assert "transitions [4, 9, 14]" in text
        assert "fired 1x" in text
        # Most-fired region first
        timer_pos = text.find("fired 3x")
        box_pos = text.find("fired 1x")
        assert timer_pos < box_pos
        # Both surfaces carry the not-board-state caveat
        assert "NOT board state" in text

    def test_actions_histogram_per_region(self):
        events = {
            0: _timer_event(5),
            3: _timer_event(6),
        }
        actions = [2, 2, 2, 7, 7]
        text = flash_log_text(events, actions)
        assert "actions {2: 1, 7: 1}" in text

    def test_out_of_range_transition_index_degrades(self):
        """A flash stored at a transition beyond the actions list (the
        unrecorded-transition guard in sandbox.py) must render, not crash."""
        events = {12: _timer_event(5)}
        text = flash_log_text(events, [2, 2])
        assert "fired 1x" in text

    def test_purity_over_the_store(self):
        events = {4: _timer_event(5)}
        snapshot = dict(events)
        flash_log_text(events, [2])
        assert events == snapshot


# ── abb14eca real aggregate ────────────────────────────────────────────────


class TestAbb14ecaFlashLog:
    def test_timer_flashes_collapse_top_box_stays_distinct(self, abb_lines):
        """The real recording: 11 events — the 9 timer flashes collapse to
        ~2 digit regions, the 2 top-box flashes stay distinct rows."""
        prev: list[list[int]] | None = None
        events: dict[int, FlashEvent] = {}
        actions = [line["data"].get("action_taken", 2) or 2 for line in abb_lines]
        for i, line in enumerate(abb_lines):
            stack = line["data"]["frame"]
            board, event = extract_board(stack, prev, "continuity")
            prev = board
            if event is not None:
                events[i] = event
        assert len(events) == 11
        text = flash_log_text(events, actions)
        assert "11 transient overlay events" in text
        rows = [ln for ln in text.splitlines() if ln.startswith("  rows")]
        # Top-box (2 fires) ranks above every timer digit (1 fire each,
        # since digit shifts move them out of overlap) — pre-registered
        # expectation: 9 timer events, 2 top-box; dedup can only merge
        # shifted-overlapping digits.
        assert any("fired 2x" in ln and "rows 9-15" in ln for ln in rows)


# ── Sandbox surfacing surfaces (flag + dedup print) ────────────────────────


def _live_sandbox(
    responses: list[dict[str, Any]], surface: bool = True
) -> SimulatorSandbox:
    """Live-mode sandbox over scripted callback responses."""
    queue = list(responses)

    def callback(action_id: int, action_data: Any) -> dict[str, Any]:
        return queue.pop(0) if queue else {"grid": _GRID}

    return SimulatorSandbox(
        step_env_callback=callback, surface_flash_events=surface
    )


_GRID = [[3] * 64 for _ in range(64)]
_FLASH = FlashEvent(
    cells=((9, 34, 3, 0), (10, 34, 3, 0)),
    regions=((9, 10, 34, 34),),
)
_FLASH_RESPONSE = {"grid": _GRID, "flash_event": _FLASH.as_dict()}


def _seeded(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Response list where the flash rides the SECOND action.

    The first live action has prev_grid None (no corpus anchor — the real
    agent seeds _current_frame via update_state() before the tool loop),
    so a flash there is unrecorded and correctly not stored."""
    return [{"grid": _GRID}, response]


class TestSandboxFlashSurfacing:
    """Surfacing asserts on run_code's RETURNED output — the worker thread
    swaps sys.stdout to its own buffer, so capsys never sees tool prints
    (the real seam is what the LLM's tool result shows)."""

    def test_flag_off_stores_but_never_prints(self):
        sb = _live_sandbox(
            _seeded(_FLASH_RESPONSE) + [{"grid": _GRID}], surface=False
        )
        out, _, _ = sb.run_code("action(2)")
        out, _, _ = sb.run_code("action(2)")
        assert sb._flash_events == {0: _FLASH}
        assert "FLASH:" not in out

    def test_flag_on_prints_flash_line(self):
        sb = _live_sandbox(_seeded(_FLASH_RESPONSE))
        sb.run_code("action(2)")
        out, _, _ = sb.run_code("action(2)")
        assert "FLASH: 2 cells" in out

    def test_repeat_region_suppresses_second_line_but_keeps_store(self):
        sb = _live_sandbox(_seeded(dict(_FLASH_RESPONSE)) + [dict(_FLASH_RESPONSE)])
        sb.run_code("action(2)")
        out1, _, _ = sb.run_code("action(2)")
        out2, _, _ = sb.run_code("action(2)")
        assert "FLASH:" in out1
        assert "FLASH:" not in out2
        assert sorted(sb._flash_events) == [0, 1]

    def test_new_region_prints_again(self):
        other = {
            "grid": _GRID,
            "flash_event": FlashEvent(
                cells=((53, 5, 5, 0),),
                regions=((53, 53, 5, 5),),
            ).as_dict(),
        }
        sb = _live_sandbox(_seeded(dict(_FLASH_RESPONSE)) + [other])
        sb.run_code("action(2)")
        _, _, _ = sb.run_code("action(2)")
        out2, _, _ = sb.run_code("action(2)")
        assert "FLASH:" in out2

    def test_flash_on_unrecorded_transition_not_stored(self):
        """First action with prev_grid None: no corpus entry, no store key
        (the -1 corruption guard)."""
        sb = _live_sandbox([dict(_FLASH_RESPONSE)])
        out, _, _ = sb.run_code("action(2)")
        assert sb._flash_events == {}
        assert "FLASH:" not in out


class TestSandboxFlashLogInCheck:
    def test_check_prints_flash_log_when_surfacing_on(self):
        sb = _live_sandbox(_seeded(_FLASH_RESPONSE))
        sb.run_code("action(2)")
        sb.run_code("action(2)")
        sb.run_code("set_simulate(lambda g, a: g)")
        out, _, _ = sb.run_code("check()")
        assert "Flash log" in out
        assert sb._last_check_result["flash_log"] != ""

    def test_check_flash_log_suppressed_when_flag_off(self):
        sb = _live_sandbox(_seeded(_FLASH_RESPONSE), surface=False)
        sb.run_code("action(2)")
        sb.run_code("action(2)")
        sb.run_code("set_simulate(lambda g, a: g)")
        out, _, _ = sb.run_code("check()")
        assert "Flash log" not in out
        assert sb._last_check_result["flash_log"] == ""

    def test_sim_state_block_carries_flash_log(self):
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._sandbox = _live_sandbox(_seeded(_FLASH_RESPONSE))
        agent._sandbox.run_code("action(2)")
        agent._sandbox.run_code("action(2)")
        agent._sandbox.run_code("set_simulate(lambda g, a: g)")
        block = agent._build_sim_state_block()
        assert block is not None
        assert "Flash log" in block

    def test_sim_state_block_flash_log_off_via_sandbox_flag(self):
        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent._sandbox = _live_sandbox(_seeded(_FLASH_RESPONSE), surface=False)
        agent._sandbox.run_code("action(2)")
        agent._sandbox.run_code("action(2)")
        agent._sandbox.run_code("set_simulate(lambda g, a: g)")
        block = agent._build_sim_state_block()
        assert block is not None
        assert "Flash log" not in block
