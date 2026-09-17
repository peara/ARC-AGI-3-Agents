"""Perception boundary contract tests (multi-game)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from perception.objects import n_subframes, to_grid
from perception.session import PerceptionSession
from tests.perception_fixtures import load_perception_expectations


def _existing_expectations():
    return [e for e in load_perception_expectations() if e.recording.path.is_file()]


def perception_case_id(expect) -> str:
    return expect.recording.name


@pytest.mark.unit
class TestSettledSubframeExtraction:
    def test_to_grid_defaults_to_last_subframe(self):
        stack = np.stack(
            [
                np.zeros((4, 4), dtype=np.int16),
                np.ones((4, 4), dtype=np.int16),
            ]
        )
        assert int(to_grid(stack)[0, 0]) == 1
        assert int(to_grid(stack, layer=0)[0, 0]) == 0
        assert n_subframes(stack) == 2


@pytest.mark.unit
class TestPerceptionContract:
    @pytest.fixture(params=_existing_expectations(), ids=perception_case_id)
    def expect(self, request):
        return request.param

    def test_summary_is_json_serializable(self, expect):
        session, _ = PerceptionSession.from_recording(expect.recording.path)
        summary = session.snapshot().summary()
        encoded = json.dumps(summary, sort_keys=True)
        roundtrip = json.loads(encoded)
        assert roundtrip["frame_idx"] == summary["frame_idx"]
        assert "determinism" in roundtrip
        assert "events" in roundtrip
        assert "globals" in roundtrip

    def test_counter_expectation(self, expect):
        session, _ = PerceptionSession.from_recording(expect.recording.path)
        summary = session.snapshot().summary()
        counters = [e for e in summary["entities"] if e.get("role") == "counter"]
        assert len(counters) >= expect.min_counters

    def test_animation_events(self, expect):
        session, _ = PerceptionSession.from_recording(expect.recording.path)
        summary = session.snapshot().summary()
        animations = [
            e for e in summary["events"] if e.get("kind") == "animation"
        ]
        assert len(animations) >= expect.min_animation_events


