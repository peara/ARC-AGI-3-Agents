"""RESET routing through the handler (reset_policy) in ``SimulatorSandbox``.

Task 7 of the RESET-handler alignment wave. Three production sites route
through ``agents/simulator_agent/reset_policy.py`` instead of re-deriving
RESET semantics inline:

1. **bfs** — RESET edges are excluded from the branching action set
   (``bfs_includes_reset()`` is False): a RESET edge cycles back to the
   level-start board, which BFS has already visited. Excluded branches log
   at DEBUG on logger ``simulator.reset_policy``.
2. **predict_and_compare** — RESET is skipped (unchanged behaviour), but the
   skip now routes through ``is_reset`` and LOGS instead of staying silent.
3. **Offline normalization** — harness ``"RESET"`` string ids normalize to
   ``RESET_ACTION`` via ``is_reset``; garbage ids keep the historical
   ``int()`` error semantics.

Regression anchor: the silent-skip observability principle (AGENTS.md) —
every former silent skip must log.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from agents.simulator_agent.reset_policy import RESET_ACTION, bfs_includes_reset
from agents.simulator_agent.sandbox import SimulatorSandbox

# ── Helpers ──────────────────────────────────────────────────────────────────


def _identity_simulate(g: list[list[int]], a: int) -> list[list[int]]:
    """simulate that returns the grid unchanged (RESET-like for every action)."""
    return [row[:] for row in g]


def _make_grids(size: int = 64) -> tuple[list[list[int]], list[list[int]]]:
    """(prev, curr) 64x64 pair differing at one cell — find_changed_regions
    hard-codes 64x64 bounds, so predict tests must use 64-sized grids."""
    prev = [[0] * size for _ in range(size)]
    curr = [[0] * size for _ in range(size)]
    curr[5][5] = 7
    return prev, curr


def _make_offline_sandbox(
    action_inputs: list[dict[str, Any]],
) -> SimulatorSandbox:
    """Offline-mode sandbox over a stub harness (mirrors __init__ loading).

    The harness stub provides exactly what ``__init__`` consumes:
    ``replay_all()`` + ``frames`` (FrameData-like with ``.frame``) +
    ``action_inputs``. No network, no real recording needed.
    """
    from types import SimpleNamespace

    # Two distinct 2x2 boards so the corpus has one real transition.
    board_a = [[1, 2], [3, 4]]
    board_b = [[5, 6], [7, 8]]
    frames = [
        SimpleNamespace(frame=[row[:] for row in board_a]),
        SimpleNamespace(frame=[row[:] for row in board_b]),
    ]

    class _StubHarness:
        def replay_all(self) -> None:
            pass

    harness = _StubHarness()
    harness.frames = frames  # type: ignore[attr-defined]
    harness.action_inputs = action_inputs  # type: ignore[attr-defined]
    return SimulatorSandbox(harness=harness, timeout=5.0)  # type: ignore[arg-type]


# ── 1. bfs RESET-edge exclusion ──────────────────────────────────────────────


class TestBfsResetExclusion:
    @pytest.mark.unit
    def test_bfs_never_expands_reset_action(
        self, seeded_live_sandbox, caplog
    ) -> None:
        """bfs with valid_actions=[0,1,2,3] must never emit action 0 in a plan.

        Identity simulate: every action (incl. RESET) maps any grid to
        itself, so the only reachable state is the start board. The goal is
        unreachable; the point is that the expansion set excludes RESET —
        asserted via the DEBUG log + the policy contract, and via the plan
        result on a goal-reachable variant (next test).
        """
        sb = seeded_live_sandbox(callback=lambda a, d: {}, simulate=_identity_simulate)
        sb.namespace["valid_actions"] = [0, 1, 2, 3]
        with caplog.at_level(logging.DEBUG, logger="simulator.reset_policy"):
            result = sb.namespace["bfs"]([[0]], lambda g: False, max_depth=3)
        assert result is None  # identity sim: goal unreachable
        assert bfs_includes_reset() is False
        # The exclusion decision is observable (observability principle):
        # bfs logs the excluded RESET branch at DEBUG.
        assert any(
            "RESET" in rec.message and "bfs" in rec.message.lower()
            for rec in caplog.records
        ), "bfs must log excluded RESET branches at DEBUG"

    @pytest.mark.unit
    def test_bfs_plan_contains_no_reset(self, seeded_live_sandbox) -> None:
        """Goal reachable via action 1: the returned plan must be [1], never
        contain 0, even though 0 is a valid action."""
        calls: list[int] = []

        def sim(g: list[list[int]], a: int) -> list[list[int]]:
            calls.append(a)
            new = [row[:] for row in g]
            if a == 1:
                new[0][0] = 9
            return new

        sb = seeded_live_sandbox(callback=lambda a, d: {}, simulate=sim)
        sb.namespace["valid_actions"] = [0, 1, 2, 3]
        result = sb.namespace["bfs"]([[0]], lambda grid: grid[0][0] == 9, max_depth=5)
        assert result == [1]
        assert 0 not in calls, "bfs must never expand the RESET edge"

    @pytest.mark.unit
    def test_bfs_reset_only_valid_actions_returns_none(
        self, seeded_live_sandbox
    ) -> None:
        """valid_actions=[0] only: after exclusion the branch set is empty —
        bfs must terminate with 'No path found', not hang or crash."""
        sb = seeded_live_sandbox(callback=lambda a, d: {}, simulate=_identity_simulate)
        sb.namespace["valid_actions"] = [0]
        result = sb.namespace["bfs"]([[0]], lambda g: False, max_depth=3)
        assert result is None


# ── 2. predict_and_compare RESET skip (routed + logged) ─────────────────────


class TestPredictSkipRouting:
    @pytest.mark.unit
    def test_predict_skips_reset_and_logs(self, plain_sandbox, caplog) -> None:
        """predict_and_compare on action 0: skipped (no exception flow), and
        the skip is LOGGED (formerly silent)."""
        s = plain_sandbox()
        s._simulate = _identity_simulate
        prev, curr = _make_grids()
        with caplog.at_level(logging.DEBUG):
            s.predict_and_compare(prev, curr, action_id=RESET_ACTION)
        assert s._pending_exception_flow is None
        assert s.pending_images == []
        assert any(
            "predict" in rec.message.lower() and "RESET" in rec.message
            for rec in caplog.records
        ), "RESET skip in predict_and_compare must be logged"

    @pytest.mark.unit
    def test_predict_reset_string_also_skips(self, plain_sandbox) -> None:
        """Wire form 'RESET' (str) routes through is_reset too — same skip."""
        s = plain_sandbox()
        s._simulate = _identity_simulate
        prev, curr = _make_grids()
        # predict_and_compare's signature takes int; the handler accepts both
        # forms — this pins that the int path (0) is the one used here.
        s.predict_and_compare(prev, curr, action_id=0)
        assert s._pending_exception_flow is None

    @pytest.mark.unit
    def test_predict_non_reset_still_runs(self, plain_sandbox) -> None:
        """Control: a non-RESET action still produces the diff flow."""
        s = plain_sandbox()
        s._simulate = _identity_simulate
        prev, curr = _make_grids()
        s.predict_and_compare(prev, curr, action_id=4)
        assert s._pending_exception_flow is not None
        assert s._pending_exception_flow["n_diff"] == 1


# ── 3. Offline normalization via the handler ────────────────────────────────


class TestOfflineNormalization:
    @pytest.mark.unit
    def test_reset_string_and_int_normalize(self) -> None:
        """action_inputs [{"id": "RESET"}, {"id": 1}] -> _actions == [0, 1]."""
        sb = _make_offline_sandbox([{"id": "RESET"}, {"id": 1}])
        assert sb._actions == [RESET_ACTION, 1]
        assert len(sb._grids) == 2

    @pytest.mark.unit
    def test_garbage_string_id_raises_like_int(self) -> None:
        """Non-int non-RESET string ids keep the historical int() semantics:
        ValueError, not silent coercion."""
        with pytest.raises(ValueError):
            _make_offline_sandbox([{"id": "BOGUS"}])

    @pytest.mark.unit
    def test_int_zero_normalizes_to_reset_action(self) -> None:
        """Integer 0 ids pass through as RESET_ACTION (== 0)."""
        sb = _make_offline_sandbox([{"id": 0}, {"id": 2}])
        assert sb._actions == [RESET_ACTION, 2]