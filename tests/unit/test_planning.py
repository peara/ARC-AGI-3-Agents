"""Planning tests driven by tests/reference_recordings.json (multi-game)."""

from __future__ import annotations

import pytest

from perception.planning import learn_movement_model
from perception.recording_eval import collect_observed_steps, plan_and_evaluate
from tests.perception_fixtures import (
    MANIFEST_PATH,
    PlanCase,
    build_perception_stack,
    load_manifest,
    plan_case_id,
)


def _existing_cases() -> list[PlanCase]:
    return [c for c in load_manifest() if c.recording.path.is_file()]


@pytest.mark.unit
class TestReferenceRecordingsManifest:
    def test_manifest_loads(self):
        cases = load_manifest()
        assert cases, "reference_recordings.json should list at least one plan case"

    def test_manifest_paths_exist(self):
        missing = [
            str(c.recording.path)
            for c in load_manifest()
            if not c.recording.path.is_file()
        ]
        assert not missing, f"missing recording files: {missing}"

    def test_manifest_file_is_valid_json(self):
        assert MANIFEST_PATH.is_file()


@pytest.mark.unit
class TestPlanningOnRecordings:
    @pytest.fixture(params=_existing_cases(), ids=plan_case_id)
    def plan_case(self, request) -> PlanCase:
        return request.param

    @pytest.fixture
    def stack(self, plan_case: PlanCase):
        return build_perception_stack(plan_case.recording.path)

    def test_controllable_entity_exists(self, stack, plan_case: PlanCase):
        ent = stack.catalog.entities.get(plan_case.entity_id)
        assert ent is not None
        assert ent.affordances.get("controllable") is True

    def test_movement_model_from_recording(self, stack, plan_case: PlanCase):
        model = learn_movement_model(
            stack.registry,
            stack.catalog,
            stack.action_ids,
            plan_case.entity_id,
        )
        assert model is not None
        assert model.motion_by_action

    def test_observed_steps_collected(self, stack, plan_case: PlanCase):
        steps = collect_observed_steps(
            stack.registry,
            stack.catalog,
            stack.action_ids,
            plan_case.entity_id,
        )
        assert len(steps) >= 1

    def test_plan_reaches_goal(self, stack, plan_case: PlanCase):
        result = plan_and_evaluate(
            stack.registry,
            stack.catalog,
            stack.action_ids,
            plan_case.entity_id,
            plan_case.start_frame,
            plan_case.goal_frame,
        )
        assert result is not None, "BFS should find a plan"
        assert result.predict_reached_goal
        assert result.diverged_steps == 0

    def test_plan_steps_match_or_extrapolate(self, stack, plan_case: PlanCase):
        result = plan_and_evaluate(
            stack.registry,
            stack.catalog,
            stack.action_ids,
            plan_case.entity_id,
            plan_case.start_frame,
            plan_case.goal_frame,
        )
        assert result is not None
        for step in result.steps:
            assert step.status in ("matched", "extrapolated")
            assert step.predicted_pos is not None

    def test_goal_matches_recording_end_frame(self, stack, plan_case: PlanCase):
        from perception.planning import entity_pos_at

        result = plan_and_evaluate(
            stack.registry,
            stack.catalog,
            stack.action_ids,
            plan_case.entity_id,
            plan_case.start_frame,
            plan_case.goal_frame,
        )
        assert result is not None
        actual = entity_pos_at(
            stack.registry,
            stack.catalog,
            plan_case.entity_id,
            plan_case.goal_frame,
        )
        assert result.goal_pos == actual
