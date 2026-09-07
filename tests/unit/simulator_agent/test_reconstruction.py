"""Reconstruction fidelity tests for the ls20 spiral incident.

Anchor: 2026-09-07 run 1786060d-b75e-48f7-b0fb-ac224be8c18a (ls20,
simulatorfirst). At spiral-turn start (recording frame 11) the sandbox held:
a registered 5x5-block simulate (check() cached 100% over 7 transitions),
a 14-cell yellow-HUD ignore mask, and a pending 123-cell exception flow
from the last action of a 4x-up batch (player walked into the goal box).

These tests pin the reconstruction contract so later prompt-variant
experiments resume from a bit-identical incident state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.simulator_agent.reconstruction import (
    IncidentMarker,
    reconstruct,
    verify_reconstruction,
)

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "1786060d-b75e-48f7-b0fb-ac224be8c18a.recording.jsonl"
)
_MARKER = IncidentMarker(
    spiral_frame=11,
    simulate_seq=29,
    ignore_seq=30,
    spiral_seq=35,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def state():
    if not _RECORDING.exists():
        pytest.skip("incident recording not present")
    return reconstruct(_RECORDING, marker=_MARKER)


class TestReconstruction:
    def test_sandbox_transitions_match_recording(self, state):
        sandbox = state.sandbox
        assert len(sandbox._grids) == 12
        assert len(sandbox._actions) == 11
        assert sandbox._actions == [1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1]
        assert sandbox.namespace["n_frames"] == 12

    def test_stale_check_is_the_frame7_cached_result(self, state):
        cached = state.sandbox._last_check_result
        assert cached is not None
        assert cached["overall_accuracy"] == 100.0
        assert (cached["frames_correct"], cached["frames_total"]) == (7, 7)

    def test_simulate_registered_from_seq29_source(self, state):
        assert state.sandbox._simulate is not None
        assert "my_sim" in state.simulate_source
        assert "set_simulate(my_sim)" in state.simulate_source

    def test_ignore_mask_is_the_seq30_hud_region(self, state):
        assert len(state.sandbox._ignore_mask) == 14
        expected = {(61, c) for c in range(13, 20)} | {(62, c) for c in range(13, 20)}
        assert state.sandbox._ignore_mask == expected

    def test_conversation_prefix_is_seq35_verbatim(self, state):
        assert state.frame_index == 11
        assert len(state.messages) == 101
        assert state.messages[0]["role"] == "system"
        user_texts = [
            m["content"] for m in state.messages
            if m["role"] == "user" and isinstance(m["content"], str)
        ]
        assert any("EXCEPTION FLOW" in t for t in user_texts)

    def test_batch_diff_images_match_recorded_captions(self, state):
        captions = [img["caption"] for img in state.diff_images]
        assert len(captions) == 4
        assert captions[0] == (
            "SIMULATION DIFF after action 1 (2 cells in 1 regions, red = wrong)"
        )
        assert captions[-1] == (
            "SIMULATION DIFF after action 1 (123 cells in 2 regions, red = wrong)"
        )
        assert state.sandbox.pending_images == []

    def test_exception_flow_reproduces_123_cell_divergence(self, state):
        checks = verify_reconstruction(state)
        assert checks["exception_n_diff"] == 123
        assert checks["exception_n_regions"] == 2
        assert checks["exception_regions"] == [
            ((53, 1, 62, 10), 76),
            ((10, 34, 19, 38), 47),
        ]

    def test_workflow_at_model_phase(self, state):
        assert state.workflow.phase.value == "MODEL"

    def test_reconstruction_is_deterministic(self):
        if not _RECORDING.exists():
            pytest.skip("incident recording not present")
        first = verify_reconstruction(reconstruct(_RECORDING, marker=_MARKER))
        second = verify_reconstruction(reconstruct(_RECORDING, marker=_MARKER))
        assert first == second