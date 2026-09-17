"""UNKNOWN-abstention unit coverage for sandbox.py + workflow.py (plan task 8).

The L1 ladder of the simulator-unknown-abstention plan: the sandbox-side
contracts that replaced set_ignore masking — predict_and_compare UNKNOWN
suppression, the bfs abstention warning, the workflow guardrail's
abstention-immunity, and the UNKNOWN namespace plumbing.

Real-path style throughout (house rules, .agents/skills/test-writing):
plain ``SimulatorSandbox`` instances from the conftest factories, real
registered sims, real method calls — no MagicMock anywhere in this file.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable

import pytest

from agents.simulator_agent.check import UNKNOWN
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController

# ── Shared grid helpers ─────────────────────────────────────────────────────


def _grid64(fill: int = 0) -> list[list[int]]:
    """Fresh 64x64 grid (find_changed_regions hardcodes 64x64 — smaller
    synthetic grids crash with IndexError inside predict_and_compare)."""
    return [[fill] * 64 for _ in range(64)]


def _abstaining_sim(
    abstained: set[tuple[int, int]],
) -> Callable[[list[list[int]], int], list[list[int]]]:
    """Sim that copies the grid and writes UNKNOWN on *abstained* cells."""

    def simulate(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        for r, c in abstained:
            out[r][c] = UNKNOWN
        return out

    return simulate


# ═══════════════════════════════════════════════════════════════════════════
# (a) predict_and_compare: UNKNOWN suppression vs modeled-cell firing
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_abstained_only_diff_suppresses_exception_flow_and_counts(
    plain_sandbox,
) -> None:
    """Given a sim that abstains on the only cells reality changed,

    When predict_and_compare runs,

    Then no exception flow fires (the quieting property) and the
    persistent _suppressed_abstained_diffs counter increments by the
    abstained-changed cell count.
    """
    s = plain_sandbox()
    s._simulate = _abstaining_sim({(61, 27), (62, 27)})

    prev = _grid64()
    new = _grid64()
    new[61][27] = 3  # reality changed both abstained cells
    new[62][27] = 3

    s.predict_and_compare(prev, new, action_id=4)

    assert s._pending_exception_flow is None, (
        "diff confined to abstained cells must not fire exception flow"
    )
    assert s.pending_images == []
    assert s._suppressed_abstained_diffs == 2


@pytest.mark.unit
def test_abstained_stable_diff_does_not_bump_counter(plain_sandbox) -> None:
    """Given a sim that abstains on a cell reality left unchanged,

    When predict_and_compare runs,

    Then the counter stays at 0 — only abstained cells that ACTUALLY
    changed count (mirrors check.py's abstained_changed bucket).
    """
    s = plain_sandbox()
    s._simulate = _abstaining_sim({(61, 27)})

    prev = _grid64()
    new = _grid64()  # reality changed nothing

    s.predict_and_compare(prev, new, action_id=4)

    assert s._pending_exception_flow is None
    assert s._suppressed_abstained_diffs == 0


@pytest.mark.unit
def test_modeled_cell_diff_fires_exception_flow_alongside_abstention(
    plain_sandbox,
) -> None:
    """Given a sim that abstains on one changing cell but is wrong on a
    modeled cell,

    When predict_and_compare runs,

    Then the exception flow fires on the modeled cell exactly as before,
    and the abstained cell still increments the suppression counter.
    """
    s = plain_sandbox()
    s._simulate = _abstaining_sim({(61, 27)})

    prev = _grid64()
    new = _grid64()
    new[61][27] = 3  # abstained cell changed -> counter
    new[5][5] = 7  # modeled cell changed, sim predicted no change -> wrong

    s.predict_and_compare(prev, new, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None, "modeled-cell diff must still fire"
    assert pending["error"] is None
    assert pending["n_diff"] == 1, "abstained cell must not count into n_diff"
    assert pending["n_regions"] == 1
    assert pending["regions"][0]["bbox"] == (5, 5, 5, 5)
    assert s._suppressed_abstained_diffs == 1


@pytest.mark.unit
def test_suppression_counter_accumulates_across_calls(plain_sandbox) -> None:
    """Given two consecutive abstained-only diffs,

    When predict_and_compare runs twice,

    Then the counter accumulates (persistent, never reset by
    predict_and_compare itself).
    """
    s = plain_sandbox()
    s._simulate = _abstaining_sim({(61, 27)})

    prev = _grid64()
    new = _grid64()
    new[61][27] = 3

    s.predict_and_compare(prev, new, action_id=4)
    s.predict_and_compare(prev, new, action_id=4)

    assert s._pending_exception_flow is None
    assert s._suppressed_abstained_diffs == 2


# ═══════════════════════════════════════════════════════════════════════════
# (b) bfs warning: one block per call; zero lines + byte-identical when clean
# ═══════════════════════════════════════════════════════════════════════════


def _bfs_sandbox(
    simulate: Callable[[list[list[int]], int], list[list[int]]] | None,
    valid_actions: list[int],
) -> SimulatorSandbox:
    """Offline-shaped sandbox with a registered sim and valid actions.

    Uses the real ``_build_namespace`` so ``bfs`` is the production
    closure; grids are seeded so ``n_frames`` is non-zero (bfs reads
    ``self.namespace['valid_actions']``).
    """
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s.harness = None
    s.timeout = 30.0
    s._exec_budget = 30_000_000
    s._step_env_callback = None
    s.actions_this_turn = 0
    s._action_taken = None
    s._grids = [_grid64(), _grid64()]
    s._actions = [1]
    s._current_frame = None
    s._previous_grid = None
    s._valid_actions = []
    s._last_action_result = {}
    s._simulate = simulate
    s._simulate_source = ""
    s._last_check_result = None
    s._last_bfs_result = None
    s._pending_exception_flow = None
    s._suppressed_abstained_diffs = 0
    s._prev_correct_frames = set()
    s.pending_images = []
    s._pending_notes = {}
    s._transition_pending = False
    s._board_reset_pending = False
    s._engine_event_transitions = set()
    s._worker_stdout = None
    s._stdout_before_worker = None
    s.namespace = s._build_namespace()
    s._protected_tools = {
        k: s.namespace[k]
        for k in (
            "set_simulate",
            "simulate",
            "check",
            "diagnose",
            "bfs",
            "action",
            "update_notes",
            "show_frame",
            "show_grid",
            "atoms",
            "find_objects",
            "find_color",
            "diff",
            "compute_delta",
            "copy_grid",
            "count_color",
            "print_region",
            "move_region",
            "get_bbox",
            "get_frame",
            "get_action",
        )
        if k in s.namespace
    }
    s.namespace["valid_actions"] = valid_actions
    return s


def _moving_sim() -> Callable[[list[list[int]], int], list[list[int]]]:
    """Fully-modeled sim: action 1 moves the 0->2 cell at (0,0) right."""

    def simulate(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        if action == 1:
            for c in range(63, 0, -1):
                if out[0][c - 1] == 2:
                    out[0][c - 1] = 0
                    out[0][c] = 2
                    break
        return out

    return simulate


def _goal_cell_2() -> Callable[[list[list[int]]], bool]:
    """Goal: any 2 in row 0 at col >= 2."""

    def goal(grid: list[list[int]]) -> bool:
        return any(cell == 2 for cell in grid[0][2:])

    return goal


@pytest.mark.unit
def test_bfs_abstaining_sim_exactly_one_warning_block() -> None:
    """Given a sim that abstains on a cell in every expanded child,

    When bfs runs and finds a path,

    Then exactly ONE warning block prints (assert [bfs] line count == 1)
    and the path is still returned (never a refusal).
    """
    s = _bfs_sandbox(_abstaining_sim({(30, 30)}), valid_actions=[1, 2])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](_grid64(), _goal_cell_2())

    out = buf.getvalue()
    assert out.count("[bfs]") == 1, f"expected exactly one [bfs] line, got:\n{out}"
    assert "states contained UNKNOWN cells (your simulate abstains there)" in out
    assert "Execute in small batches and diff() frames after each" in out
    # Never a refusal: bfs still returns (a path or None), never raises.
    assert path is None or isinstance(path, list)


@pytest.mark.unit
def test_bfs_fully_modeled_sim_zero_warning_lines_byte_identical() -> None:
    """Given a fully-modeled sim on a small deterministic scenario,

    When bfs runs and finds a 2-step path,

    Then ZERO warning lines print and the output is byte-identical to the
    pinned pre-change baseline (no abstention plumbing altered the
    non-abstaining path).
    """
    s = _bfs_sandbox(_moving_sim(), valid_actions=[1, 2])

    start = _grid64()
    start[0][0] = 2  # 2 steps of action 1 to reach col >= 2

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](start, _goal_cell_2())

    out = buf.getvalue()
    assert path == [1, 1]
    assert "[bfs]" not in out
    assert "UNKNOWN" not in out
    # Byte-identical pin: goal-already-checked, path-found, no warning.
    assert out == "Path found: [1, 1] (2 steps)\n"


@pytest.mark.unit
def test_bfs_goal_already_reached_no_warning_when_clean() -> None:
    """Given a fully-modeled sim and a start state already at the goal,

    When bfs runs,

    Then the goal-already-reached return path prints no warning (the
    0/0-count guard turns warn_unknown into a no-op).
    """
    s = _bfs_sandbox(_moving_sim(), valid_actions=[1, 2])

    start = _grid64()
    start[0][5] = 2  # already at goal

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](start, _goal_cell_2())

    out = buf.getvalue()
    assert path == []
    assert "[bfs]" not in out
    assert out == "Goal already reached. Path: [] (0 steps)\n"


# ═══════════════════════════════════════════════════════════════════════════
# (f) bfs path-through-UNKNOWN: bracketed sentence present vs absent
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_bfs_path_through_unknown_carries_bracketed_sentence() -> None:
    """Given an abstaining sim whose returned path crosses abstained cells,

    When bfs runs,

    Then the warning contains the bracketed unsoundness sentence.
    """

    # Sim: action 1 moves the 2 right (modeled), but abstains on (0, 63)
    # — every child state carries UNKNOWN, so any path is unsound.
    def unsound_moving_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        out[0][63] = UNKNOWN
        if action == 1:
            for c in range(63, 0, -1):
                if out[0][c - 1] == 2:
                    out[0][c - 1] = 0
                    out[0][c] = 2
                    break
        return out

    s = _bfs_sandbox(unsound_moving_sim, valid_actions=[1, 2])

    start = _grid64()
    start[0][0] = 2

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](start, _goal_cell_2())

    out = buf.getvalue()
    assert path == [1, 1], "bfs must still return the path (never refuses)"
    assert out.count("[bfs]") == 1
    assert (
        "[The returned path passes through unmodeled cells — it may be unsound.]" in out
    )


@pytest.mark.unit
def test_bfs_children_unknown_but_path_clean_no_bracketed_sentence() -> None:
    """Given a sim where SOME expanded children contain UNKNOWN but the
    returned path's states do not,

    When bfs runs,

    Then the warning prints (children-with-UNKNOWN > 0) but WITHOUT the
    bracketed unsoundness sentence.
    """

    # Sim: action 1 moves the 2 right (fully modeled); action 2 abstains
    # on (30, 30) and changes nothing else. bfs expands both branches —
    # the action-2 child carries UNKNOWN — but the returned path [1, 1]
    # never touches an abstained state.
    def mixed_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        if action == 1:
            for c in range(63, 0, -1):
                if out[0][c - 1] == 2:
                    out[0][c - 1] = 0
                    out[0][c] = 2
                    break
        else:
            out[30][30] = UNKNOWN
        return out

    s = _bfs_sandbox(mixed_sim, valid_actions=[1, 2])

    start = _grid64()
    start[0][0] = 2

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](start, _goal_cell_2())

    out = buf.getvalue()
    assert path == [1, 1]
    assert out.count("[bfs]") == 1, (
        "children with UNKNOWN must still warn (once), even on a clean path"
    )
    assert "states contained UNKNOWN cells (your simulate abstains there)" in out
    assert (
        "[The returned path passes through unmodeled cells — it may be unsound.]"
        not in out
    )


# ═══════════════════════════════════════════════════════════════════════════
# (c) Workflow guardrail: abstention-only checks never demote MODEL→EXPLORE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_abstention_only_check_resets_failure_counter_at_model(plain_sandbox) -> None:
    """Given a MODEL-phase controller one failure away from the 5-strike
    guardrail,

    When a schema-2 abstention-only check result arrives (wrong_cells 0,
    50 abstained-changed),

    Then the failure counter RESETS to 0 and the phase stays MODEL.
    """
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready to model")
    controller._check_failures = 4  # one failure away from the guardrail

    controller.on_check_result({"wrong_cells": 0, "abstained_changed": 50, "schema": 2})

    assert controller._check_failures == 0
    assert controller.phase == Phase.MODEL


@pytest.mark.unit
def test_six_abstention_only_checks_never_demote_from_model(plain_sandbox) -> None:
    """Given a MODEL-phase controller fed six consecutive abstention-only
    check results,

    When the escape guardrails are evaluated after each,

    Then the phase stays MODEL — the 5-strike demotion can never fire on
    abstention (wrong_cells redefined as predicted-wrong only).
    """
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready to model")

    for _ in range(6):
        controller.on_check_result(
            {"wrong_cells": 0, "abstained_changed": 50, "schema": 2}
        )
        assert controller.apply_escape_guardrails() is False
        assert controller.phase == Phase.MODEL

    assert controller._check_failures == 0


# ═══════════════════════════════════════════════════════════════════════════
# (d) set_simulate accepts UNKNOWN outputs; (e) UNKNOWN global in namespace
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_set_simulate_accepts_unknown_outputs(plain_sandbox) -> None:
    """Given a sim whose output grid contains UNKNOWN cells,

    When set_simulate registers it,

    Then registration succeeds (no output validation exists — none was
    added) and the frozen simulate round-trips UNKNOWN values.
    """
    s = plain_sandbox()
    s._grids = []
    s._actions = []
    s.namespace = s._build_namespace()

    def unknown_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        out[0][0] = UNKNOWN
        return out

    s.namespace["set_simulate"](unknown_sim)

    assert s._simulate is not None
    g = [[0, 0], [0, 0]]
    result = s._simulate(g, 1)
    assert result[0][0] == UNKNOWN


@pytest.mark.unit
def test_unknown_global_prints_minus_one_via_run_code(seeded_live_sandbox) -> None:
    """Given the sandbox namespace,

    When run_code executes ``print(UNKNOWN)``,

    Then the output is ``-1`` — the abstention sentinel is a named
    constant the LLM's code can reference.
    """
    s = seeded_live_sandbox(
        lambda action_id, action_data: {
            "grid": [[0] * 8 for _ in range(8)],
            "valid_actions": [1],
            "last_action_result": {},
            "history": [],
            "objects": (),
            "adjacency": frozenset(),
        },
        current_frame=[[0] * 8 for _ in range(8)],
    )

    output, error, _ = s.run_code("print(UNKNOWN)")

    assert error is None
    assert output == "-1\n"


# ═══════════════════════════════════════════════════════════════════════════
# (g) F2 review fixes: diagnose abstention semantics, counter reset, bfs flag
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_diagnose_unknown_over_stable_region_not_spurious(capsys) -> None:
    """Given a sim that abstains on a region reality leaves stable,

    When diagnose runs,

    Then those abstained cells do NOT count as SPURIOUS (F1: raw grid_diff
    put every UNKNOWN cell into pred_set — phantom errors the LLM saw).
    """
    from agents.simulator_agent.check import diagnose

    g0 = [[0] * 8 for _ in range(8)]
    g1 = [[0] * 8 for _ in range(8)]
    g1[0][0] = 3  # one real change, correctly predicted below

    def sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        out[0][0] = 3  # model the real change
        out[5][5] = UNKNOWN  # abstain on a stable cell
        out[5][6] = UNKNOWN
        return out

    result = diagnose(sim, [g0, g1], [1])

    assert result["total_spurious"] == 0, (
        "abstained cells over a stable region must not be SPURIOUS"
    )
    assert result["total_wrong"] == 0
    assert result["total_abstained"] == 2


@pytest.mark.unit
def test_diagnose_unknown_over_changed_region_not_missed_not_spurious(
    capsys,
) -> None:
    """Given a sim that abstains on the very cells reality changed,

    When diagnose runs,

    Then those cells are neither MISSED nor SPURIOUS — abstained means
    not judged (mirrors run_check's abstained_changed bucket).
    """
    from agents.simulator_agent.check import diagnose

    g0 = [[0] * 8 for _ in range(8)]
    g1 = [[0] * 8 for _ in range(8)]
    g1[2][2] = 3  # reality changes (2,2); sim abstains on it

    def sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        out[2][2] = UNKNOWN
        return out

    result = diagnose(sim, [g0, g1], [1])

    assert result["total_missed"] == 0
    assert result["total_spurious"] == 0
    assert result["total_wrong"] == 0
    assert result["total_abstained"] == 1


@pytest.mark.unit
def test_check_resets_suppressed_abstained_diffs_counter(plain_sandbox) -> None:
    """Given a sandbox with a non-zero _suppressed_abstained_diffs counter,

    When the namespace check() runs,

    Then the counter resets to 0 — the NOTE window is genuinely 'since
    your last check()' (F2: the counter was never reset, so the NOTE
    lied about the window).
    """
    s = plain_sandbox()
    s._simulate = _abstaining_sim({(61, 27)})
    s._suppressed_abstained_diffs = 5
    # plain_sandbox ships an empty namespace; build the real one so the
    # production check() closure is exercised.
    s._grids = [_grid64(), _grid64()]
    s._actions = [1]
    s._engine_event_transitions = set()
    s.namespace = s._build_namespace()

    prev = _grid64()
    new = _grid64()
    new[61][27] = 3
    s.predict_and_compare(prev, new, action_id=4)
    assert s._suppressed_abstained_diffs == 6

    with contextlib.redirect_stdout(io.StringIO()):
        s.namespace["check"]()

    assert s._suppressed_abstained_diffs == 0, (
        "check() must consume the suppressed-diff counter (window = since last check)"
    )


@pytest.mark.unit
def test_reset_for_level_transition_clears_suppressed_counter(plain_sandbox) -> None:
    """Given a sandbox with a non-zero suppressed-diff counter,

    When reset_for_level_transition runs,

    Then the counter resets to 0 — level transitions start a fresh
    NOTE window (F2).
    """
    s = plain_sandbox()
    s._suppressed_abstained_diffs = 9

    s.reset_for_level_transition()

    assert s._suppressed_abstained_diffs == 0


@pytest.mark.unit
def test_bfs_intermediate_unknown_state_carries_bracketed_sentence() -> None:
    """Given a 2-step path whose INTERMEDIATE state contains UNKNOWN but
    whose final parent+child states do not,

    When bfs runs,

    Then the bracketed unsoundness sentence still prints (F4: the old
    final-parent+child check missed intermediate abstained states).
    """

    # Action 1: move the 2 at (0,0) right AND abstain on (30, 30) — but
    # ONLY on the first step (when the 2 is still at col 0/1). The second
    # step's states are fully modeled. The returned path [1, 1] therefore
    # has UNKNOWN only in its intermediate state.
    def intermediate_unknown_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        if action == 1:
            if out[0][0] == 2:  # first step from the start state
                out[30][30] = UNKNOWN
            for c in range(63, 0, -1):
                if out[0][c - 1] == 2:
                    out[0][c - 1] = 0
                    out[0][c] = 2
                    break
        return out

    s = _bfs_sandbox(intermediate_unknown_sim, valid_actions=[1, 2])

    start = _grid64()
    start[0][0] = 2

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        path = s.namespace["bfs"](start, _goal_cell_2())

    out = buf.getvalue()
    assert path == [1, 1], "bfs must still return the path (never refuses)"
    assert out.count("[bfs]") == 1
    assert (
        "[The returned path passes through unmodeled cells — it may be unsound.]" in out
    ), "intermediate-state UNKNOWN must set the path flag"
