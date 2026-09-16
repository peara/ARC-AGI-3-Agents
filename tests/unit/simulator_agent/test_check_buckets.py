"""UNKNOWN-abstention bucket semantics for run_check (plan task 1, TDD).

Anchor: the f9e3e301 incident (docs/reports/simulator-agent.md §14) — the
agent used set_ignore() to mask the key-box region out of scoring, and the
masked cells were never reported again. The UNKNOWN(-1) abstention contract
replaces masking: the model writes UNKNOWN into cells it cannot predict,
check() scores committed cells only, and every abstention is counted.

Real-path tests: the production ``run_check``/``diagnose`` on synthetic
corpora — no mocks of the logic under test. ``UNKNOWN`` is accessed lazily
via ``check.UNKNOWN`` so the pre-change red run fails per-test
(AttributeError / KeyError), not at collection time.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable

import pytest

from agents.simulator_agent import check
from agents.simulator_agent.check import diagnose, run_check


def _grid4() -> list[list[int]]:
    """Fresh 4x4 all-zero grid."""
    return [[0] * 4 for _ in range(4)]


def _sim_painting(
    overrides: dict[tuple[int, int], int],
) -> Callable[[list[list[int]], int], list[list[int]]]:
    """Simulate that copies the input grid then paints *overrides*.

    Override values may be ``check.UNKNOWN`` — that is how a model abstains.
    """

    def simulate(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        for (r, c), value in overrides.items():
            out[r][c] = value
        return out

    return simulate


# ── UNKNOWN constant + seam removal ─────────────────────────────────────────


@pytest.mark.unit
def test_unknown_constant_exported() -> None:
    """UNKNOWN is -1 and exported via __all__ (single source of truth)."""
    assert check.UNKNOWN == -1
    assert "UNKNOWN" in check.__all__


@pytest.mark.unit
def test_ignore_mask_params_removed() -> None:
    """Contract pin: the ignore_mask seam is gone from both entrypoints."""
    for fn in (run_check, diagnose):
        assert "ignore_mask" not in inspect.signature(fn).parameters, (
            f"{fn.__name__} still carries the ignore_mask parameter"
        )


# ── Bucket semantics (plan cases a-g) ──────────────────────────────────────


@pytest.mark.unit
def test_abstained_over_changed_counts_abstained_changed_not_wrong() -> None:
    """Given reality changes a cell the model abstains on,

    When check runs,

    Then the cell is abstained_changed — never wrong, never spurious —
    while reality's change is still measured in total_changed.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3  # reality changes the abstained cell
    sim = _sim_painting({(0, 0): unknown})

    result = run_check(sim, [g0, g1], [1], verbose=False)

    assert result["abstained_changed"] == 1
    assert result["abstained_stable"] == 0
    assert result["total_wrong"] == 0
    assert result["total_spurious"] == 0
    assert result["total_correct"] == 0
    assert result["total_changed"] == 1
    assert result["per_frame"][0]["abstained_changed"] == 1
    assert result["per_frame"][0]["wrong"] == 0
    # Nothing committed -> accuracy defined as 100, coverage flags the gap.
    assert result["overall_accuracy"] == 100.0
    assert result["coverage"] == 0.0


@pytest.mark.unit
def test_abstained_over_stable_counts_abstained_stable_not_spurious() -> None:
    """Given the model abstains on a cell reality leaves unchanged,

    While committing a correct prediction elsewhere,

    Then the abstained cell is abstained_stable — never spurious.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[1][1] = 3  # reality changes a cell the model predicts correctly
    sim = _sim_painting({(0, 0): unknown, (1, 1): 3})

    result = run_check(sim, [g0, g1], [1], verbose=False)

    assert result["abstained_stable"] == 1
    assert result["abstained_changed"] == 0
    assert result["total_spurious"] == 0
    assert result["total_correct"] == 1
    assert result["total_wrong"] == 0


@pytest.mark.unit
def test_mixed_frame_partial_commitment_all_buckets() -> None:
    """Given one transition hitting every bucket at once,

    When check runs,

    Then each bucket counts exactly its own cells and accuracy/coverage
    are computed over committed cells only.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3  # changed + abstained      -> abstained_changed
    g1[1][1] = 3  # changed + predicted 3     -> correct
    g1[2][2] = 3  # changed + predicted 5     -> wrong
    # (3, 3) unchanged + predicted 7          -> spurious (and wrong: 7 != 0)
    # (0, 1) unchanged + abstained            -> abstained_stable
    sim = _sim_painting(
        {
            (0, 0): unknown,
            (0, 1): unknown,
            (1, 1): 3,
            (2, 2): 5,
            (3, 3): 7,
        }
    )

    result = run_check(sim, [g0, g1], [1], verbose=False)

    frame = result["per_frame"][0]
    assert frame["abstained_changed"] == 1
    assert frame["abstained_stable"] == 1
    assert frame["correct"] == 1
    assert frame["wrong"] == 2
    assert frame["spurious"] == 1
    # accuracy = 1 correct / (1 correct + 2 wrong) over committed cells only.
    assert result["overall_accuracy"] == 33.3
    # coverage = (3 changed - 1 abstained-changed) / 3 changed.
    assert result["coverage"] == 66.7


@pytest.mark.unit
def test_wrong_cells_excludes_abstained() -> None:
    """Given one abstained-over-changed transition and one genuinely wrong
    transition,

    Then wrong_cells counts only the committed wrong cell.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3  # transition 0: abstained-over-changed
    g2 = [row[:] for row in g1]
    g2[1][1] = 3  # transition 1: identity copy misses it -> wrong
    sim = _sim_painting({(0, 0): unknown})

    result = run_check(sim, [g0, g1, g2], [1, 1], verbose=False)

    assert result["abstained_changed"] == 1
    assert result["total_wrong"] == 1
    assert result["wrong_cells"] == 1
    assert result["per_frame"][0]["wrong"] == 0
    assert result["per_frame"][1]["wrong"] == 1


@pytest.mark.unit
def test_coverage_partial_commitment_over_changed_cells() -> None:
    """Given reality changes two cells and the model abstains on one,

    Then coverage is the committed share of changed cells (50.0%).
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3
    g1[1][1] = 3
    sim = _sim_painting({(0, 0): unknown, (1, 1): 3})

    result = run_check(sim, [g0, g1], [1], verbose=False)

    assert result["coverage"] == 50.0
    assert result["abstained_changed"] == 1
    assert result["total_correct"] == 1


@pytest.mark.unit
def test_coverage_none_when_nothing_changed() -> None:
    """Given a corpus where reality never changes a cell,

    Then coverage is None (no denominator), not a crash.
    """
    g0 = _grid4()
    g1 = _grid4()
    sim = _sim_painting({})  # full commitment, nothing changes

    result = run_check(sim, [g0, g1], [1], verbose=False)

    assert result["coverage"] is None
    assert result["total_changed"] == 0


@pytest.mark.unit
def test_all_unknown_predict_is_not_a_crash() -> None:
    """Given the model abstains on EVERY cell while reality changes some,

    Then the zero denominator is defined as accuracy 100% (nothing was
    committed), coverage is 0, and every changed cell lands in
    abstained_changed — no crash, no division by zero.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3
    g1[1][1] = 5

    def abstain_everything(grid: list[list[int]], action: int) -> list[list[int]]:
        return [[unknown] * 4 for _ in range(4)]

    result = run_check(abstain_everything, [g0, g1], [1], verbose=False)

    assert result["total_wrong"] == 0
    assert result["overall_accuracy"] == 100.0
    assert result["per_frame"][0]["accuracy"] == 100.0
    assert result["coverage"] == 0.0
    assert result["abstained_changed"] == 2
    assert result["abstained_stable"] == 14
    assert result["frames_correct"] == 1


@pytest.mark.unit
def test_unknown_over_unknown_stays_abstained_stable() -> None:
    """Given the model abstains on a cell reality never changed, across
    multiple transitions,

    Then it accumulates abstained_stable and never leaks into wrong or
    spurious.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[1][1] = 3  # transition 0 (action 1): real change, model commits 3
    g2 = [row[:] for row in g1]
    g2[2][2] = 5  # transition 1 (action 2): real change, model commits 5

    def staged_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        out = [row[:] for row in grid]
        out[0][0] = unknown  # abstain on (0,0) every transition
        if action == 1:
            out[1][1] = 3
        else:
            out[2][2] = 5
        return out

    result = run_check(staged_sim, [g0, g1, g2], [1, 2], verbose=False)

    assert result["abstained_stable"] == 2  # one per transition
    assert result["abstained_changed"] == 0
    assert result["total_wrong"] == 0
    assert result["total_spurious"] == 0
    assert result["total_correct"] == 2


@pytest.mark.unit
def test_skip_transitions_excluded_from_all_aggregates() -> None:
    """Given a skipped transition carrying abstained and wrong cells,

    Then none of its cells reach any aggregate — buckets included — and
    its simulate call never runs.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3  # transition 0: abstained-over-changed
    g2 = [row[:] for row in g1]
    g2[1][1] = 3  # transition 1 (SKIPPED): would be a committed wrong
    g3 = [row[:] for row in g2]
    g3[2][2] = 3  # transition 2: committed wrong (identity copy misses it)
    sim = _sim_painting({(0, 0): unknown})

    seen_actions: list[int] = []

    def counting_sim(grid: list[list[int]], action: int) -> list[list[int]]:
        seen_actions.append(action)
        return sim(grid, action)

    result = run_check(
        counting_sim,
        [g0, g1, g2, g3],
        [1, 2, 3],
        verbose=False,
        skip_transitions={1},
    )

    assert result["frames_total"] == 2
    assert result["skipped_transitions"] == [1]
    assert [fr["frame"] for fr in result["per_frame"]] == [0, 2]
    assert seen_actions == [1, 3]  # skipped transition never ran simulate
    # Transition 1's cells reach no bucket:
    assert result["abstained_changed"] == 1  # transition 0's only
    assert result["abstained_stable"] == 1  # transition 2's (0,0) stable
    assert result["total_wrong"] == 1
    assert result["total_changed"] == 2
    assert result["coverage"] == 50.0
    assert result["frames_correct"] == 1


# ── Accuracy formula + preserved semantics ─────────────────────────────────


@pytest.mark.unit
def test_accuracy_over_committed_cells_only() -> None:
    """Given 2 correct + 1 wrong committed predictions plus 1 abstained
    changed cell,

    Then accuracy = 2/(2+1) — abstained cells are outside the denominator
    (no mask denominator anywhere).
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3  # abstained-over-changed
    g1[1][1] = 3  # correct
    g1[2][2] = 3  # correct
    g1[3][3] = 3  # wrong (predicted 5)
    sim = _sim_painting({(0, 0): unknown, (1, 1): 3, (2, 2): 3, (3, 3): 5})

    result = run_check(sim, [g0, g1], [1], verbose=False)

    assert result["total_correct"] == 2
    assert result["total_wrong"] == 1
    assert result["overall_accuracy"] == 66.7
    assert result["coverage"] == 75.0


@pytest.mark.unit
def test_degenerate_reset_unchanged_under_abstention() -> None:
    """Given a virtual RESET pair (identical boards, action 0) where the
    sim abstains on every cell,

    Then the transition is still tagged degenerate RESET and still counts
    as a correct frame (degenerate-RESET semantics preserved).
    """
    unknown = check.UNKNOWN
    b0 = _grid4()

    def abstain_everything(grid: list[list[int]], action: int) -> list[list[int]]:
        return [[unknown] * 4 for _ in range(4)]

    result = run_check(
        abstain_everything, [b0, [row[:] for row in b0]], [0], verbose=False
    )

    assert result["per_frame"][0]["degenerate_reset"] is True
    assert result["degenerate_reset_frames"] == 1
    assert result["frames_correct"] == 1


@pytest.mark.unit
def test_result_dict_schema_and_cluster_key() -> None:
    """Given any scored corpus,

    Then the result carries schema 2 and an abstained_clusters key (empty
    list until the cluster log lands).
    """
    g0 = _grid4()
    g1 = _grid4()

    result = run_check(_sim_painting({}), [g0, g1], [1], verbose=False)

    assert result["schema"] == 2
    assert result["abstained_clusters"] == []


@pytest.mark.unit
def test_run_check_does_not_mutate_inputs() -> None:
    """Given grids containing cells the sim abstains on,

    When run_check runs,

    Then the input grids and actions are byte-identical afterwards.
    """
    unknown = check.UNKNOWN
    g0 = _grid4()
    g0[3][3] = 7
    g1 = _grid4()
    g1[0][0] = 3
    grids = [g0, g1]
    actions = [1]
    grids_snapshot = copy.deepcopy(grids)
    actions_snapshot = list(actions)

    run_check(_sim_painting({(0, 0): unknown}), grids, actions, verbose=False)

    assert grids == grids_snapshot
    assert actions == actions_snapshot


@pytest.mark.unit
def test_diagnose_runs_without_mask() -> None:
    """Non-regression pin: diagnose still runs on a plain corpus after the
    ignore_mask seam is removed."""
    g0 = _grid4()
    g1 = _grid4()
    g1[0][0] = 3

    result = diagnose(_sim_painting({}), [g0, g1], [1])

    assert result["frames_total"] == 1
    assert result["total_wrong"] == 1
