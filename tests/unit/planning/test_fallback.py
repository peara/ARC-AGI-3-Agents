"""Unit tests for planning/fallback.py — pure fallback selection functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from effects.state import SceneState
from planning.fallback import build_fallback_goal, pick_fallback_unknown, tried_key
from planning.probe import ProbeGoal
from planning.query import UnknownAction


def _make_scene(pos: tuple[int, int] | None = (5, 5)) -> MagicMock:
    scene = MagicMock()
    scene.controllable_pos.return_value = pos
    return scene


def _make_unknown(action: int, pos: tuple[int, int] | None) -> UnknownAction:
    if pos is None:
        state = SceneState(relevant=((0, ("alive", 1)),))
    else:
        state = SceneState(relevant=((0, ("pos", pos)),))
    return UnknownAction(action=action, state=state)


@pytest.mark.unit
class TestPickFallbackUnknown:
    def test_empty_unknowns_returns_none(self) -> None:
        assert pick_fallback_unknown([], set(), _make_scene()) is None

    def test_all_tried_returns_none(self) -> None:
        ua = _make_unknown(3, (10, 10))
        tried = {tried_key(ua)}
        assert pick_fallback_unknown([ua], tried, _make_scene()) is None

    def test_single_fresh_unknown_returned(self) -> None:
        ua = _make_unknown(3, (10, 10))
        result = pick_fallback_unknown([ua], set(), _make_scene((0, 0)))
        assert result is ua

    def test_picks_nearest_by_manhattan(self) -> None:
        near = _make_unknown(1, (6, 5))
        far = _make_unknown(2, (20, 20))
        result = pick_fallback_unknown([far, near], set(), _make_scene((5, 5)))
        assert result is near

    def test_filters_tried_picks_next_nearest(self) -> None:
        near = _make_unknown(1, (6, 5))
        mid = _make_unknown(2, (8, 8))
        far = _make_unknown(3, (20, 20))
        tried = {tried_key(near)}
        result = pick_fallback_unknown([far, near, mid], tried, _make_scene((5, 5)))
        assert result is mid

    def test_same_action_different_state_both_fresh(self) -> None:
        ua_a = _make_unknown(5, (5, 5))
        ua_b = _make_unknown(5, (10, 10))
        tried = {tried_key(ua_a)}
        result = pick_fallback_unknown([ua_a, ua_b], tried, _make_scene((9, 9)))
        assert result is ua_b

    def test_no_pos_defaults_distance_zero(self) -> None:
        ua_no_pos = _make_unknown(1, None)
        ua_far = _make_unknown(2, (50, 50))
        result = pick_fallback_unknown([ua_far, ua_no_pos], set(), _make_scene((0, 0)))
        assert result is ua_no_pos

    def test_scene_controllable_none_defaults_zero(self) -> None:
        ua_a = _make_unknown(1, (50, 50))
        ua_b = _make_unknown(2, (1, 1))
        result = pick_fallback_unknown([ua_a, ua_b], set(), _make_scene(None))
        assert result in (ua_a, ua_b)


@pytest.mark.unit
class TestBuildFallbackGoal:
    def test_builds_goal_with_pos(self) -> None:
        ua = _make_unknown(3, (47, 31))
        goal = build_fallback_goal(ua)
        assert isinstance(goal, ProbeGoal)
        assert goal.action == 3
        assert "all" in goal.target
        pred = goal.target["all"][0]
        assert pred["dim"] == "pos"
        assert pred["of"] == 0
        assert list(pred["eq"]) == [47, 31]

    def test_builds_goal_non_tuple_value(self) -> None:
        ua = _make_unknown(3, None)
        goal = build_fallback_goal(ua)
        pred = goal.target["all"][0]
        assert pred["dim"] == "alive"
        assert pred["eq"] == 1

    def test_reason_includes_action(self) -> None:
        ua = _make_unknown(7, (1, 1))
        goal = build_fallback_goal(ua)
        assert "7" in goal.reason


@pytest.mark.unit
class TestTriedKey:
    def test_key_is_fingerprint_and_action(self) -> None:
        ua = _make_unknown(5, (5, 5))
        key = tried_key(ua)
        assert key == (ua.state.fingerprint(), 5)

    def test_different_states_different_keys(self) -> None:
        ua_a = _make_unknown(5, (5, 5))
        ua_b = _make_unknown(5, (10, 10))
        assert tried_key(ua_a) != tried_key(ua_b)

    def test_same_state_different_actions_different_keys(self) -> None:
        ua_a = _make_unknown(5, (5, 5))
        ua_b = _make_unknown(3, (5, 5))
        assert tried_key(ua_a) != tried_key(ua_b)