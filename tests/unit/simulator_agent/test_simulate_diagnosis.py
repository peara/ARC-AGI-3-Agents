"""Unit tests for the simulator exception-flow diagnosis infrastructure.

These tests exercise the production code paths: ``predict_and_compare()`` on
``SimulatorSandbox`` and the namespace freeze in the real ``set_simulate`` —
not re-implementations of the logic under test.
"""

from __future__ import annotations

import pytest

from agents.simulator_agent.check import diagnose, run_check
from agents.simulator_agent.prompts import EXCEPTION_FLOW_TEXT


@pytest.mark.unit
def test_predict_crash_sets_error_key(plain_sandbox) -> None:
    s = plain_sandbox()

    def bad_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        raise ValueError("intentional test crash")

    s._simulate = bad_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[1] * 64 for _ in range(64)]

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["error"]
    assert "ValueError" in pending["error"]
    assert "intentional test crash" in pending["error"]
    assert pending["action_id"] == 4
    assert pending["n_diff"] == 0
    assert pending["regions"] == []
    # Crash path: no diff image (no meaningful diff regions)
    assert s.pending_images == []


@pytest.mark.unit
def test_predict_diff_sets_regions(plain_sandbox) -> None:
    s = plain_sandbox()

    # simulate predicts "nothing happens"; reality changes one cell 0->7
    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    curr[5][5] = 7

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["error"] is None
    assert pending["n_diff"] == 1
    assert pending["n_regions"] == 1
    assert len(pending["regions"]) == 1
    assert pending["regions"][0]["bbox"] == (5, 5, 5, 5)
    assert pending["regions"][0]["n_cells"] == 1
    assert "0->7" in pending["regions"][0]["transitions"]
    # Visual diff image rendered into pending_images
    assert len(s.pending_images) == 1
    assert "SIMULATION DIFF after action 4" in s.pending_images[0]["caption"]
    assert s.pending_images[0]["b64"]


@pytest.mark.unit
def test_region_cap_at_4(plain_sandbox) -> None:
    s = plain_sandbox()

    cells = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    for r, c in cells:
        curr[r][c] = 1

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["n_regions"] == 6  # total regions found
    # But only top 4 stored (by area — all are 1x1 here, so any 4)
    assert len(pending["regions"]) == 4


@pytest.mark.unit
def test_ignore_mask_hides_hud_tick_diff(plain_sandbox) -> None:
    s = plain_sandbox()

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    curr[61][27] = 3  # HUD timer tick, same shape as ls20's 11->3
    curr[5][5] = 7  # real sprite movement the LLM failed to model

    s._ignore_mask = {(61, 27)}
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["n_diff"] == 1, "ignored HUD cell must not count into n_diff"
    assert pending["n_regions"] == 1
    assert pending["regions"][0]["bbox"] == (5, 5, 5, 5)


@pytest.mark.unit
def test_ignore_mask_full_cover_produces_no_exception_flow(plain_sandbox) -> None:
    s = plain_sandbox()

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    curr[61][27] = 3
    curr[62][27] = 3

    s._ignore_mask = {(61, 27), (62, 27)}
    s.predict_and_compare(prev, curr, action_id=4)

    assert s._pending_exception_flow is None, (
        "diff entirely inside ignore mask must not fire exception flow"
    )
    assert s.pending_images == []


@pytest.mark.unit
def test_exception_flow_text_format_crash() -> None:
    text = EXCEPTION_FLOW_TEXT.format(
        action_id=1,
        action_name="up",
        diagnosis="Your simulate() CRASHED on this action:\nFile <sandbox>, line 5\nValueError: test",
        diagnosis_hint="Fix the bug and re-register.",
    )
    assert "EXCEPTION FLOW" in text
    assert "action 1 (up)" in text
    assert "ValueError: test" in text
    assert "Fix the bug" in text


@pytest.mark.unit
def test_exception_flow_text_format_diff() -> None:
    text = EXCEPTION_FLOW_TEXT.format(
        action_id=4,
        action_name="right",
        diagnosis="Your simulate() predicted a result that differs from reality by 67 cells in 4 regions:\n  rows 9-15 cols 33-39: 24 cells (0->5)",
        diagnosis_hint="A red-boxed image of the diff was attached.",
    )
    assert "67 cells" in text
    assert "rows 9-15 cols 33-39" in text
    assert "0->5" in text
    assert "red-boxed image" in text


@pytest.mark.unit
def test_frozen_simulate_survives_helper_redefinition(plain_sandbox) -> None:
    s = plain_sandbox()
    # _build_namespace eagerly reads _grids/_actions (n_frames init)
    s._grids = []
    s._actions = []
    s.namespace = s._build_namespace()
    ns = s.namespace

    ns["my_helper"] = lambda g: g  # noqa: E731
    exec("def my_simulate(g, a):\n    return my_helper(g)\n", ns)

    # Register via the REAL set_simulate (which freezes the namespace)
    ns["set_simulate"](ns["my_simulate"])
    assert s._simulate is not None
    assert s._simulate is not ns["my_simulate"], (
        "set_simulate did not freeze-wrap the function"
    )

    g = [[1, 2], [3, 4]]
    assert s._simulate(g, 1) == g

    # Poison the live namespace — the frozen snapshot must be unaffected
    ns["my_helper"] = lambda g: "POISONED"  # noqa: E731
    assert s._simulate(g, 1) == g, (
        "freeze failed — frozen simulate saw poisoned helper"
    )


@pytest.mark.unit
def test_set_simulate_captures_source_from_run_code(seeded_live_sandbox, win_callback) -> None:
    """Incident 5681a14a: _simulate_source stayed empty forever.

    Commit 24a8cc8 accidentally deleted the ``self._simulate_source``
    assignment while restructuring ``set_simulate`` for the freeze
    mechanism, and ``inspect.getsource`` cannot read exec'd code anyway
    (co_filename "<sandbox>" is not a real file). Net effect: every
    recording since Aug 30 recorded ``simulate_source: ""`` and the
    set_simulate ack always printed "(source unavailable)".

    Regression: the real run_code → set_simulate path must capture the
    actual function body.
    """
    s = seeded_live_sandbox(win_callback, current_frame=[[0] * 8 for _ in range(8)])

    code = (
        "def my_simulate(grid, action):\n"
        "    return copy_grid(grid)\n"
        "set_simulate(my_simulate)\n"
    )
    output, error, _ = s.run_code(code)

    assert error is None
    assert "[set_simulate] registered" in output
    assert "(source unavailable)" not in output, (
        "ack could not read the source — linecache seeding broken"
    )
    assert "def my_simulate(grid, action):" in output, (
        "ack first line should show the real source now"
    )
    assert "def my_simulate(grid, action):" in s._simulate_source
    assert "return copy_grid(grid)" in s._simulate_source
    assert s._simulate_source != "(source unavailable)"


@pytest.mark.unit
def test_tool_protection_restores_overwrite(plain_sandbox) -> None:
    s = plain_sandbox()
    s.namespace = {
        "find_objects": "ORIGINAL",
        "action": lambda a: a,
        "current_frame": [],
    }
    s._protected_tools = {
        "find_objects": "ORIGINAL",
        "action": s.namespace["action"],
    }

    s.namespace["find_objects"] = "POISONED"

    output = s._protect_tool_names("hello")
    assert s.namespace["find_objects"] == "ORIGINAL"
    assert (
        "WARNING: 'find_objects' is a builtin tool and was protected from redefinition"
        in output
    )
    assert output.count("WARNING") == 1


# ── Level-transition hard-abort + state clearing (WS1 Task 1) ──────────────


@pytest.mark.unit
def test_win_action_sets_transition_pending(seeded_live_sandbox, win_callback) -> None:
    initial = [[0] * 8 for _ in range(8)]
    s = seeded_live_sandbox(win_callback, current_frame=initial)

    output, error, action_taken = s.run_code("action(1)")

    assert s._transition_pending is True
    assert error is None
    assert output == "[LEVEL COMPLETED — board transitioning; stop]"
    assert action_taken == 1


@pytest.mark.unit
def test_transition_clears_grids_and_check_state(seeded_live_sandbox, win_callback) -> None:
    initial = [[0] * 8 for _ in range(8)]

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s = seeded_live_sandbox(win_callback, simulate=static_simulate, current_frame=initial)
    s._simulate_source = "def simulate(g, a): return g"
    s._pending_notes = {"notes": "win-trigger conjecture", "plan": ""}

    s._grids = [initial, initial, initial, initial]
    s._actions = [4, 4, 1]
    s._last_check_result = {"total_wrong": 5}
    s._ignore_mask = {(0, 0), (1, 1)}

    s.reset_for_level_transition()

    assert s._grids == []
    assert s._actions == []
    assert s._last_check_result is None
    assert s._ignore_mask == set()
    assert s._pending_exception_flow is None
    assert s._current_frame is None
    assert s._previous_grid is None
    assert s._last_action_result == {}
    assert s._simulate is not None
    assert s._simulate_source == "def simulate(g, a): return g"
    assert s._pending_notes == {"notes": "win-trigger conjecture", "plan": ""}
    assert s._transition_pending is False
    assert s.namespace["current_frame"] is None
    assert s.namespace["history"] == []

    output, error, _ = s.run_code("check()")
    assert error is None
    assert "No transitions recorded yet" in output


@pytest.mark.unit
def test_predict_and_compare_suppressed_on_win_step(seeded_live_sandbox, win_callback) -> None:
    initial = [[0] * 8 for _ in range(8)]

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s = seeded_live_sandbox(win_callback, simulate=static_simulate, current_frame=initial)

    s.run_code("action(1)")

    assert s._pending_exception_flow is None


@pytest.mark.unit
def test_check_with_empty_history_after_clear(seeded_live_sandbox, win_callback) -> None:
    initial = [[0] * 8 for _ in range(8)]

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s = seeded_live_sandbox(win_callback, simulate=static_simulate, current_frame=initial)
    s.reset_for_level_transition()

    output, error, _ = s.run_code("check()")
    assert error is None
    assert "No transitions recorded yet" in output


# ── Degenerate RESET surfacing in run_check (task 8) ───────────────────────


def _identity_simulate(grid: list[list[int]], action: int) -> list[list[int]]:
    """Trivial simulate: returns the input grid unchanged."""
    return [row[:] for row in grid]


def _reset_first_corpus() -> tuple[
    list[list[list[int]]],
    list[int],
]:
    """Corpus mirroring the live RESET-seeded shape: (B0,0)→B0 then (B0,1)→B1.

    Transition 0 is the virtual RESET pair: identical boards, action 0 —
    trivially correct under any simulate, hence the frames_correct inflation
    this task surfaces.
    """
    base = [[0] * 4 for _ in range(4)]
    b0 = [row[:] for row in base]
    b1 = [row[:] for row in base]
    b1[0][0] = 3
    return [b0, b0, b1], [0, 1]


@pytest.mark.unit
def test_check_surfaces_degenerate_reset_count_and_key() -> None:
    """Identity simulate on a RESET-first corpus: degenerate RESET surfaced.

    The (B0,0)→B0 identity transition is counted as a correct frame (metric
    UNCHANGED) but must be tagged per-frame, counted in the summary line, and
    returned as ``degenerate_reset_frames``.
    """
    grids, actions = _reset_first_corpus()

    result = run_check(_identity_simulate, grids, actions, verbose=True)

    # Metrics untouched: transition 0 trivially correct, transition 1 wrong.
    assert result["frames_total"] == 2
    assert result["frames_correct"] == 1
    assert result["total_wrong"] == 1
    assert result["per_frame"][0]["wrong"] == 0
    assert result["per_frame"][0]["changed"] == 0
    assert result["per_frame"][1]["wrong"] == 1
    assert result["per_frame"][1]["changed"] == 1

    # New surfacing: per-frame tag + aggregate key.
    assert result["per_frame"][0]["degenerate_reset"] is True
    assert result["per_frame"][1].get("degenerate_reset", False) is False
    assert result["degenerate_reset_frames"] == 1


@pytest.mark.unit
def test_check_summary_line_contains_degenerate_reset_count(capsys) -> None:
    """Verbose summary line carries the degenerate RESET count."""
    grids, actions = _reset_first_corpus()

    run_check(_identity_simulate, grids, actions, verbose=True)
    out = capsys.readouterr().out

    assert "Frames correct: 1/2 (1 degenerate RESET)" in out


@pytest.mark.unit
def test_check_no_reset_transitions_no_degenerate_surfacing(capsys) -> None:
    """Corpus without action-0 transitions: no tag, no count, no suffix."""
    base = [[0] * 4 for _ in range(4)]
    g0 = [row[:] for row in base]
    g1 = [row[:] for row in base]
    g1[0][0] = 3
    g2 = [row[:] for row in g1]
    g2[1][1] = 5
    grids = [g0, g1, g2]
    actions = [1, 2]

    result = run_check(_identity_simulate, grids, actions, verbose=True)
    out = capsys.readouterr().out

    assert "degenerate" not in out
    assert result["degenerate_reset_frames"] == 0
    for frame_result in result["per_frame"]:
        assert frame_result.get("degenerate_reset", False) is False


# ── skip_transitions: scoring-only exclusion (task 7) ──────────────────────


def _three_transition_corpus() -> tuple[
    list[list[list[int]]],
    list[int],
]:
    """3 grids / 2 transitions; transition 1 is the 'flash' stand-in.

    Transition 0: (g0, 1) → g1 — one cell 0→3.
    Transition 1: (g1, 2) → g2 — one cell 0→5 (identity simulate misses it).
    """
    base = [[0] * 4 for _ in range(4)]
    g0 = [row[:] for row in base]
    g1 = [row[:] for row in base]
    g1[0][0] = 3
    g2 = [row[:] for row in g1]
    g2[1][1] = 5
    return [g0, g1, g2], [1, 2]


@pytest.mark.unit
def test_check_skip_transitions_excluded_from_totals() -> None:
    """Skipped index: no per-frame entry, no aggregate contribution."""
    grids, actions = _three_transition_corpus()

    result = run_check(
        _identity_simulate, grids, actions, verbose=False,
        skip_transitions={1},
    )

    assert result["frames_total"] == 1
    assert result["frames_correct"] == 0
    assert [fr["frame"] for fr in result["per_frame"]] == [0]
    assert result["total_wrong"] == 1
    assert result["total_changed"] == 1
    assert result["total_correct"] == 0
    assert result["skipped_transitions"] == [1]
    assert result["wrong_cells"] == 1


@pytest.mark.unit
def test_check_skip_transitions_empty_set_equals_none() -> None:
    """skip_transitions=set() behaves exactly like None (old behavior)."""
    grids, actions = _three_transition_corpus()

    baseline = run_check(_identity_simulate, grids, actions, verbose=False)
    empty = run_check(
        _identity_simulate, grids, actions, verbose=False,
        skip_transitions=set(),
    )

    assert empty == baseline


@pytest.mark.unit
def test_check_skip_transitions_out_of_range_ignored() -> None:
    """Out-of-range indices in the skip set are ignored gracefully."""
    grids, actions = _three_transition_corpus()

    result = run_check(
        _identity_simulate, grids, actions, verbose=False,
        skip_transitions={5, 99, -1},
    )

    assert result["frames_total"] == 2
    assert result["skipped_transitions"] == []
    assert result["total_wrong"] == 2


@pytest.mark.unit
def test_check_skip_transitions_reset_transition_still_scored() -> None:
    """RESET-action transitions are NOT excludable via the skip seam's
    sibling path — check_includes_reset() policy is untouched: a skipped
    set that omits the RESET transition leaves it scored (pin)."""
    grids, actions = _reset_first_corpus()

    result = run_check(
        _identity_simulate, grids, actions, verbose=False,
        skip_transitions=set(),
    )

    assert result["frames_total"] == 2
    assert result["per_frame"][0]["degenerate_reset"] is True
    assert result["degenerate_reset_frames"] == 1


@pytest.mark.unit
def test_diagnose_skip_transitions_excluded_from_totals(capsys) -> None:
    """diagnose() skip: skipped index contributes to no aggregate."""
    grids, actions = _three_transition_corpus()

    result = diagnose(
        _identity_simulate, grids, actions, skip_transitions={1}
    )
    out = capsys.readouterr().out

    assert result["total_wrong"] == 1
    assert result["total_missed"] == 1
    assert result["frames_total"] == 1
    assert result["skipped_transitions"] == [1]
    assert "Frame 1" not in out


@pytest.mark.unit
def test_diagnose_skip_transitions_empty_equals_none(capsys) -> None:
    """diagnose() empty skip set == None == old behavior."""
    grids, actions = _three_transition_corpus()

    baseline = diagnose(_identity_simulate, grids, actions)
    baseline_out = capsys.readouterr().out
    empty = diagnose(
        _identity_simulate, grids, actions, skip_transitions=set()
    )
    empty_out = capsys.readouterr().out

    assert empty == baseline
    assert empty_out == baseline_out


@pytest.mark.unit
def test_diagnose_skip_transitions_out_of_range_ignored() -> None:
    """diagnose() out-of-range skip indices are ignored gracefully."""
    grids, actions = _three_transition_corpus()

    result = diagnose(
        _identity_simulate, grids, actions, skip_transitions={7, 42}
    )

    assert result["frames_total"] == 2
    assert result["skipped_transitions"] == []
    assert result["total_wrong"] == 2
