"""Unit tests for ``agents/simulator_agent/reset_policy.py`` — the single
owner of all RESET (action 0) semantics for the simulator-agent ecosystem.

Written FIRST under strict TDD (RED before implementation). The module under
test does not exist yet at the time these tests were authored; every import
below is expected to fail during the RED phase.

Contract pinned here (consumed by tasks 4-7 of the RESET-handler alignment
wave — these API names are contractual):

- ``is_reset(action_id)`` — True for int 0 and the exact string "RESET".
- ``virtual_reset_pair(b0)`` — the ``([B0, B0], [0])`` corpus shape that
  mirrors offline loading (sandbox.py:190-207): the [0]/[1] board duplicate
  IS the RESET transition. Naive ``actions=[0]`` without the duplicate grid
  is THE corruption trap (a1's result would be attributed to RESET).
- ``ResetSeedTracker`` — agent-side one-time seed flag. Levels 2+ start via
  level-completion with no RESET, so ``should_seed`` is False for them
  ALWAYS; idempotent under double-mark.
- Policy constants: check/diagnose include RESET transitions as trained
  data; BFS excludes RESET edges as wasteful cycle-backs.
- ``producing_action_caption`` — state-keyed caption: the action that
  PRODUCED frame i is ``actions[i-1]`` (``history[i] ↔ transition i ↔
  actions[i]``). Out-of-range is graceful ("unknown"), never raises.
"""

from __future__ import annotations

import pytest

from agents.simulator_agent.reset_policy import (
    ResetSeedTracker,
    bfs_includes_reset,
    check_includes_reset,
    diagnose_includes_reset,
    is_reset,
    producing_action_caption,
    virtual_reset_pair,
)
from perception.session import RESET_ACTION

# ── Scenario 1: predicate, both forms ────────────────────────────────────


class TestIsReset:
    def test_int_zero_is_reset(self) -> None:
        assert is_reset(0) is True

    def test_string_reset_is_reset(self) -> None:
        assert is_reset("RESET") is True

    def test_string_lowercase_not_reset(self) -> None:
        """Case-sensitive: only the exact "RESET" spelling counts."""
        assert is_reset("reset") is False
        assert is_reset("Reset") is False

    def test_other_ints_not_reset(self) -> None:
        for aid in (1, 2, 3, 4, 5, 6, 7, -1, 100):
            assert is_reset(aid) is False

    def test_other_strings_not_reset(self) -> None:
        for s in ("ACTION1", "ACTION5", "", "RESET ", " RESET", "0"):
            assert is_reset(s) is False

    def test_matches_perception_session_constant(self) -> None:
        """The predicate must agree with the canonical constant imported
        from perception.session (no 4th constant copy in this repo)."""
        assert is_reset(RESET_ACTION) is True


# ── Scenario 2: virtual pair shape + mutation immunity ───────────────────


class TestVirtualResetPair:
    def test_exact_shape(self) -> None:
        b0 = [[1, 2], [3, 4]]
        grids, actions = virtual_reset_pair(b0)
        assert grids == [b0, b0]
        assert actions == [RESET_ACTION]
        assert len(grids) == 2
        assert len(actions) == 1

    def test_deep_copy_immune_to_caller_mutation(self) -> None:
        """Both returned boards must be deep copies: mutating the caller's
        b0 (or either returned board) must not affect the other copies."""
        b0 = [[5, 6], [7, 8]]
        grids, _ = virtual_reset_pair(b0)
        assert grids[0] is not b0
        assert grids[1] is not b0
        assert grids[0] is not grids[1]

        b0[0][0] = 99
        assert grids[0][0][0] == 5, "caller mutation leaked into copy 0"
        assert grids[1][0][0] == 5, "caller mutation leaked into copy 1"

        grids2, _ = virtual_reset_pair([[1]])
        grids2[0][0][0] = 42
        assert grids2[1][0][0] == 1, "returned boards share state"

    def test_mirrors_offline_corpus_shape(self) -> None:
        """Task 2's learnings: offline corpus for both pinned recordings has
        grids[0] == grids[1] and actions[0] == 0. The virtual pair mirrors
        this shape exactly — that symmetry IS the design."""
        b0 = [[0] * 4 for _ in range(4)]
        grids, actions = virtual_reset_pair(b0)
        assert grids[0] == grids[1]
        assert actions == [0]
        # len(actions) == len(grids) - 1 invariant from the corpus pinning
        assert len(actions) == len(grids) - 1


# ── Scenario 3: tracker one-time seed + idempotence ──────────────────────


class TestResetSeedTracker:
    def test_level1_should_seed_before_marking(self) -> None:
        tracker = ResetSeedTracker()
        assert tracker.should_seed(0) is True

    def test_level1_no_seed_after_marking(self) -> None:
        tracker = ResetSeedTracker()
        tracker.mark_seeded(0)
        assert tracker.should_seed(0) is False

    def test_levels_2plus_never_seed(self) -> None:
        """Levels 2+ start via level-completion with no RESET — should_seed
        is False for them ALWAYS, before AND after marking."""
        tracker = ResetSeedTracker()
        for level in (2, 3, 10):
            assert tracker.should_seed(level) is False
        tracker.mark_seeded(2)
        for level in (2, 3, 10):
            assert tracker.should_seed(level) is False

    def test_double_mark_idempotent(self) -> None:
        tracker = ResetSeedTracker()
        tracker.mark_seeded(0)
        tracker.mark_seeded(0)
        assert tracker.should_seed(0) is False
        assert tracker.is_seeded_for(0) is True

    def test_is_seeded_for_alias(self) -> None:
        """Readability alias: same truth as ``not should_seed`` for level 0."""
        tracker = ResetSeedTracker()
        assert tracker.is_seeded_for(0) is False
        tracker.mark_seeded(0)
        assert tracker.is_seeded_for(0) is True
        # Levels 2+ are never "seeded via RESET" — the alias tracks the
        # one-time flag only.
        assert tracker.is_seeded_for(2) is False
        tracker.mark_seeded(2)
        assert tracker.is_seeded_for(2) is True

    def test_multiple_levels_tracked_independently(self) -> None:
        tracker = ResetSeedTracker()
        tracker.mark_seeded(0)
        assert tracker.should_seed(0) is False
        # A different level's flag does not leak.
        assert tracker.is_seeded_for(1) is False


# ── Scenario 4: policy constants ─────────────────────────────────────────


class TestPolicyConstants:
    def test_check_includes_reset(self) -> None:
        assert check_includes_reset() is True

    def test_diagnose_includes_reset(self) -> None:
        assert diagnose_includes_reset() is True

    def test_bfs_excludes_reset(self) -> None:
        assert bfs_includes_reset() is False


# ── Scenario 5: producing-action caption ─────────────────────────────────


class TestProducingActionCaption:
    def test_frame_zero_is_reset(self) -> None:
        assert (
            producing_action_caption(0, [1, 2, 3])
            == "action that produced frame 0: RESET (id 0)"
        )

    def test_frame_zero_with_empty_actions(self) -> None:
        assert (
            producing_action_caption(0, [])
            == "action that produced frame 0: RESET (id 0)"
        )

    def test_state_keyed_pairing(self) -> None:
        """The action that PRODUCED frame i is actions[i-1] — state-keyed,
        not action-keyed. history[i] ↔ transition i ↔ actions[i]."""
        assert (
            producing_action_caption(1, [7, 2, 3])
            == "action that produced frame 1: 7"
        )
        assert (
            producing_action_caption(2, [7, 2, 3])
            == "action that produced frame 2: 2"
        )
        assert (
            producing_action_caption(3, [7, 2, 3])
            == "action that produced frame 3: 3"
        )

    def test_out_of_range_graceful(self) -> None:
        """Never raises; reports 'unknown' for indices beyond the corpus."""
        assert (
            producing_action_caption(4, [7, 2, 3])
            == "action that produced frame 4: unknown"
        )
        assert (
            producing_action_caption(100, [])
            == "action that produced frame 100: unknown"
        )

    def test_negative_index_graceful(self) -> None:
        """Negative frame indices are out-of-range too (never raises)."""
        assert (
            producing_action_caption(-1, [7, 2, 3])
            == "action that produced frame -1: unknown"
        )


@pytest.mark.unit
class TestModuleContract:
    """Contract pinning per house conventions (inspect.getsource)."""

    def test_module_imports_reset_action_from_perception(self) -> None:
        """Pin: this module imports RESET_ACTION from perception.session —
        the ONLY simulator_agent module allowed to (no 4th constant copy)."""
        import inspect

        import agents.simulator_agent.reset_policy as rp

        src = inspect.getsource(rp)
        assert "from perception.session import RESET_ACTION" in src
        # No local redefinition of the constant.
        assert "RESET_ACTION = 0" not in src
        assert "RESET_ACTION: int = 0" not in src

    def test_logger_name(self) -> None:
        import logging

        import agents.simulator_agent.reset_policy as rp

        assert rp.logger.name == "simulator.reset_policy"
        assert isinstance(rp.logger, logging.Logger)