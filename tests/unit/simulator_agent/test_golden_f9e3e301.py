"""Golden test (L2 ladder gate): the f9e3e301 abstained log surfaces the
key-box signal the real run threw away.

Anchor: 2026-09-04 run f9e3e301-7f2f-4703-ae19-85f051412259 (ls20,
simulatorfirst).  In that run the agent registered a pure movement sim
(llm.jsonl seq 10, frame 4) and masked the key box + timer EXTERNALLY via
``set_ignore`` — permanently deleting the most important signal on the
board as "noise".  This test proves the UNKNOWN-abstention replacement
(.omo/plans/simulator-unknown-abstention.md) surfaces that signal: the
same sim, abstaining INSIDE the model, produces an abstained-cluster log
whose single merged cluster contains the key-box region, changed on the
flash transitions, with the blocked UP entries in its action histogram.

INDEX SPACE (plan task 9, reviewer-measured — do not re-derive): the
fixture builds from SETTLED RECORDING BOARDS directly (no virtual RESET
pair), so fixture transition i = recording transition i = ``settled(frame
i) --actions[i]--> settled(frame i+1)``.  The flash transitions in THIS
index space are 10, 20, 22 (the recorded-check corpus space is +1 — its
"frames 10,11,20,21,23" anchor does NOT apply here).

Measured cluster shapes (reviewer-verified twice; also re-measured while
building the fixture — see .omo/evidence/simulator-unknown-abstention/
task-9-golden.txt):

- settled diffs: settled(10)→settled(11) = 76, settled(11)→settled(12)
  = 0, settled(20)→settled(21) = 76, settled(21)→settled(22) = 0,
  settled(22)→settled(23) = 78.
- ONE merged abstained cluster (``cluster_cells`` threshold=3 merges the
  key-box rows 53-62 cols 1-10 with the timer rows 61-62): bbox (53, 1,
  62, 31), cells_per_frame_max 78.
- Transition 22's flash was produced by action 3 (not 1) — the histogram
  is {1: 13, 2: 5, 3: 2, 4: 1}, NOT all-1.

Fixtures (committed, built from the real recording):

- ``fixtures/f9e3e301_simulate.py`` — the verbatim seq-10 sim source
  (``_my_sim_verbatim``) + ``my_sim`` wrapper adding ONLY the UNKNOWN
  writes (rows 61-62 all cols; rows 53-62 cols 1-10) on every return
  path.  The flash transitions are blocked moves — the verbatim sim
  early-returns there, so end-of-body-only writes would silently skip
  them and the key-box signal would NOT surface.
- ``fixtures/f9e3e301_grids.json`` — 24 settled boards (recording frames
  0-23, settled via ``frame_layers.settled_board``; frames 11/12/21/22
  are 6-layer HIGHLIGHT stacks settling to layer 0) + the 23 actions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.check import UNKNOWN, run_check

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(_FIXTURES) not in sys.path:
    sys.path.insert(0, str(_FIXTURES))

from f9e3e301_simulate import _my_sim_verbatim, my_sim  # noqa: E402

pytestmark = pytest.mark.unit

# Key-box region (the run's set_ignore target) and the flash transitions,
# in FIXTURE index space (= recording transition space).
_KEYBOX_CONTAINS = (53, 1, 62, 10)  # bbox must CONTAIN this region
_FLASH_TRANSITIONS = {10, 20, 22}
_ZERO_DIFF_TRANSITIONS = {11, 21}  # the zero-diff duplicate frames


def _load_corpus() -> tuple[list[list[list[int]]], list[int]]:
    """Load the committed fixture: 24 settled boards + 23 actions."""
    payload = json.loads(
        (_FIXTURES / "f9e3e301_grids.json").read_text(encoding="utf-8")
    )
    grids = payload["grids"]
    actions = payload["actions"]
    assert len(grids) == 24, "fixture must carry 24 settled boards (frames 0-23)"
    assert len(actions) == 23, "fixture must carry 23 actions (transitions 0-22)"
    return grids, actions


@pytest.fixture(scope="module")
def golden_result() -> dict[str, Any]:
    """``run_check`` of the abstaining sim over the real corpus (once)."""
    grids, actions = _load_corpus()
    return run_check(my_sim, grids, actions, verbose=False)


@pytest.fixture(scope="module")
def negative_result() -> dict[str, Any]:
    """Same check with the ORIGINAL sim style (no UNKNOWN writes)."""
    grids, actions = _load_corpus()
    return run_check(_my_sim_verbatim, grids, actions, verbose=False)


def _covers_keybox(cluster: dict[str, Any]) -> bool:
    """True iff the cluster's bbox CONTAINS the key-box region."""
    r0, c0, r1, c1 = cluster["bbox"]
    return (
        r0 <= _KEYBOX_CONTAINS[0]
        and c0 <= _KEYBOX_CONTAINS[1]
        and r1 >= _KEYBOX_CONTAINS[2]
        and c1 >= _KEYBOX_CONTAINS[3]
    )


class TestGoldenF9e3e301:
    """The abstained log surfaces the key-box signal on real data."""

    def test_single_merged_cluster_contains_keybox(self, golden_result):
        """(1) ONE merged abstained cluster whose bbox CONTAINS rows 53-62
        cols 1-10 — the key-box rows merge with the timer rows 61-62 under
        ``cluster_cells`` threshold=3 (measured: rows 53-62, cols ~1-31)."""
        clusters = golden_result["abstained_clusters"]
        assert len(clusters) == 1, (
            f"expected exactly ONE merged abstained cluster "
            f"(key-box + timer merge under threshold=3), got {len(clusters)}: "
            f"{clusters}"
        )
        assert _covers_keybox(clusters[0]), (
            f"cluster bbox {clusters[0]['bbox']} must CONTAIN the key-box "
            f"region rows 53-62 cols 1-10"
        )

    def test_cluster_changed_on_flash_transitions(self, golden_result):
        """(2) The merged cluster's ``changed_on_transitions`` ⊇ {10, 20,
        22} — the flash transitions in fixture index space."""
        clusters = golden_result["abstained_clusters"]
        assert len(clusters) == 1
        changed_on = set(clusters[0]["changed_on_transitions"])
        missing = _FLASH_TRANSITIONS - changed_on
        assert not missing, (
            f"cluster must have changed on flash transitions "
            f"{sorted(_FLASH_TRANSITIONS)}; missing {sorted(missing)} "
            f"(changed_on={sorted(changed_on)})"
        )

    def test_cluster_action_histogram(self, golden_result):
        """(3) The blocked UP entries: action 1 AND action 3 are among the
        cluster's histogram actions.  Transition 22's flash was produced by
        action 3 — the histogram is NOT all-1 (measured: {1: 13, 2: 5,
        3: 2, 4: 1})."""
        clusters = golden_result["abstained_clusters"]
        assert len(clusters) == 1
        actions_hist = clusters[0]["actions"]
        assert 1 in actions_hist, (
            f"action 1 (blocked UP) must be in the cluster histogram, "
            f"got {actions_hist}"
        )
        assert 3 in actions_hist, (
            f"action 3 (produced transition 22's flash) must be in the "
            f"cluster histogram, got {actions_hist}"
        )

    def test_per_frame_abstained_changed(self, golden_result):
        """(4) Per-frame: ``abstained_changed > 0`` on transitions 10, 20,
        22 AND ``abstained_changed == 0`` on the zero-diff duplicates 11,
        21 (measured: 76 / 0 / 76 / 0 / 78)."""
        by_frame = {pf["frame"]: pf for pf in golden_result["per_frame"]}
        for t in _FLASH_TRANSITIONS:
            assert t in by_frame, f"transition {t} missing from per_frame"
            assert by_frame[t]["abstained_changed"] > 0, (
                f"transition {t}: abstained_changed must be > 0, got "
                f"{by_frame[t]['abstained_changed']}"
            )
        for t in _ZERO_DIFF_TRANSITIONS:
            assert t in by_frame, f"transition {t} missing from per_frame"
            assert by_frame[t]["abstained_changed"] == 0, (
                f"transition {t} (zero-diff duplicate): abstained_changed "
                f"must be 0, got {by_frame[t]['abstained_changed']}"
            )

    def test_negative_control_original_sim_no_abstained_cluster(self, negative_result):
        """(5) NEGATIVE CONTROL: the ORIGINAL sim style (rows painted as
        constant values instead of UNKNOWN — the sim predicts the
        bar/box cells as unchanged) produces NO abstained cluster covering
        rows 53-62 — proving the log is abstention-driven, not
        diff-driven."""
        clusters = negative_result["abstained_clusters"]
        covering = [cl for cl in clusters if _covers_keybox(cl)]
        assert not clusters or not covering, (
            f"original sim must produce no abstained cluster covering the "
            f"key-box region; got {len(clusters)} clusters, "
            f"{len(covering)} covering: {clusters}"
        )

    # ── Fixture integrity (the corpus the golden assertions stand on) ────

    def test_fixture_sim_abstains_on_every_return_path(self):
        """The UNKNOWN writes must survive the blocked-move early return —
        the flash transitions (10/20/22) are all blocked moves in the real
        recording, so an end-of-body-only addition would silently skip
        them and the key-box signal would NOT surface."""
        grids, actions = _load_corpus()
        # Transition 10: blocked UP move (player at rows 15-19, wall above).
        predicted = my_sim(grids[10], actions[10])
        assert predicted[61][0] == UNKNOWN, (
            "timer row 61 must be UNKNOWN even on the blocked-move path"
        )
        assert predicted[53][1] == UNKNOWN, (
            "key-box cell (53,1) must be UNKNOWN even on the blocked-move path"
        )

    def test_fixture_verbatim_sim_is_unmasked(self):
        """The verbatim movement sim carries NO abstention — the negative
        control's premise (original style = constant values, no UNKNOWN)."""
        grids, actions = _load_corpus()
        predicted = _my_sim_verbatim(grids[10], actions[10])
        assert predicted[61][0] != UNKNOWN and predicted[53][1] != UNKNOWN, (
            "the verbatim sim must not write UNKNOWN anywhere"
        )
        n_unknown = sum(row.count(UNKNOWN) for row in predicted)
        assert n_unknown == 0, f"verbatim sim wrote {n_unknown} UNKNOWN cells"

    def test_fixture_action_pairing(self):
        """Fixture transition i is produced by ``actions[i]`` = the
        ``action_input.id`` at recording line i+1 (send-then-record).
        Transition 10 (frame 10→11) must be action 1; transition 22
        (frame 22→23) must be action 3 — the pairing the histogram
        assertions depend on."""
        _, actions = _load_corpus()
        assert actions[10] == 1, (
            f"transition 10 must be produced by action 1, got {actions[10]}"
        )
        assert actions[22] == 3, (
            f"transition 22 must be produced by action 3, got {actions[22]}"
        )
