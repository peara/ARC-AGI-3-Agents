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
    def test_last_subframe_is_settled_state(self):
        """g50t: last sub-frame continues into the next step's first sub-frame."""
        path = next(
            e.recording.path
            for e in load_perception_expectations()
            if e.recording.name == "g50t-curiosity"
        )
        import json

        raw_frames = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line).get("data", {})
                if data.get("frame") is not None:
                    raw_frames.append(np.asarray(data["frame"]))

        # find a multi-subframe step followed by another frame
        for t in range(len(raw_frames) - 1):
            if raw_frames[t].ndim == 3 and raw_frames[t].shape[0] > 1:
                last = raw_frames[t][-1]
                first_next = raw_frames[t + 1][0]
                diff = int(np.sum(last != first_next))
                assert diff <= 1, (
                    f"settled continuity broken at step {t}: "
                    f"{diff} cells differ between last and next-first"
                )
                break
        else:
            pytest.skip("no multi-subframe pair found")

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

    def test_controllable_expectation(self, expect):
        session, _ = PerceptionSession.from_recording(expect.recording.path)
        scene = session.snapshot()
        if expect.controllable_entity_id is None:
            assert scene.controllable_id() is None
        else:
            assert scene.controllable_id() == expect.controllable_entity_id

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


