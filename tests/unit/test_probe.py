"""Unit tests for planning/probe.py — ProbeGoal DSL, compile_goal, resolve_predicate, derive_spec_from_predicate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from effects.state import SceneState
from planning.probe import (
    ProbeGoal,
    compile_goal,
    derive_spec_from_predicate,
    execute_probe,
    resolve_predicate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(entity_id: int, dim: str, value: object) -> SceneState:
    """Build a minimal SceneState with one relevant entry."""
    return SceneState(relevant=((entity_id, (dim, value)),), terminal="alive")


def _mock_scene(entity_pos: dict[int, tuple[int, int] | None] | None = None) -> MagicMock:
    """Build a mock SceneSnapshot with configurable entity_pos results."""
    scene = MagicMock()
    if entity_pos is None:
        entity_pos = {}

    def _pos(eid: int) -> tuple[int, int] | None:
        return entity_pos.get(eid)

    scene.entity_pos = _pos
    return scene


# ===========================================================================
# TestProbeGoal
# ===========================================================================


@pytest.mark.unit
class TestProbeGoal:
    """Tests for ProbeGoal dataclass and the goal predicate compiler."""

    # -----------------------------------------------------------------------
    # ProbeGoal dataclass construction
    # -----------------------------------------------------------------------

    def test_default_construction_predicate_only(self) -> None:
        """ProbeGoal requires only target; other fields have defaults."""
        goal = ProbeGoal(target={"dim": "pos", "of": 0, "eq": [5, 10]})
        assert goal.target == {"dim": "pos", "of": 0, "eq": [5, 10]}
        assert goal.entities is None
        assert goal.dims is None
        assert goal.max_steps == 20
        assert goal.reason == ""

    def test_explicit_entities_and_dims(self) -> None:
        """ProbeGoal accepts explicit entities and dims tuples."""
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            entities=(0, 17),
            dims=("pos", "size"),
            max_steps=50,
            reason="test goal",
        )
        assert goal.entities == (0, 17)
        assert goal.dims == ("pos", "size")
        assert goal.max_steps == 50
        assert goal.reason == "test goal"

    def test_frozen_dataclass_immutable(self) -> None:
        """ProbeGoal is frozen — attribute assignment raises."""
        goal = ProbeGoal(target={"action": 1})
        with pytest.raises(AttributeError):
            goal.target = {}  # type: ignore[misc]
        with pytest.raises(AttributeError):
            goal.max_steps = 99  # type: ignore[misc]

    # -----------------------------------------------------------------------
    # compile_goal — position equality (list → tuple conversion)
    # -----------------------------------------------------------------------

    def test_compile_pos_eq_list_value(self) -> None:
        """Position equality with list value: list auto-converted to tuple."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "eq": [5, 10]})
        state = _state(0, "pos", (5, 10))
        assert goal_fn(state) is True

    def test_compile_pos_eq_tuple_value(self) -> None:
        """Position equality with tuple value already stored."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "eq": (5, 10)})
        state = _state(0, "pos", (5, 10))
        assert goal_fn(state) is True

    def test_compile_pos_eq_mismatch(self) -> None:
        """Position equality fails when values differ."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "eq": [5, 10]})
        state = _state(0, "pos", (3, 7))
        assert goal_fn(state) is False

    # -----------------------------------------------------------------------
    # compile_goal — scalar equality
    # -----------------------------------------------------------------------

    def test_compile_size_eq_scalar(self) -> None:
        """Scalar equality on dim 'size'."""
        goal_fn = compile_goal({"dim": "size", "of": 17, "eq": 8})
        state = _state(17, "size", 8)
        assert goal_fn(state) is True

    def test_compile_size_eq_mismatch(self) -> None:
        """Scalar equality fails when values differ."""
        goal_fn = compile_goal({"dim": "size", "of": 17, "eq": 8})
        state = _state(17, "size", 5)
        assert goal_fn(state) is False

    def test_compile_size_eq_missing_dim(self) -> None:
        """Scalar equality returns False when dim is absent (get returns None)."""
        goal_fn = compile_goal({"dim": "size", "of": 17, "eq": 8})
        # State has entity 0, not 17 — get returns None
        state = _state(0, "size", 5)
        assert goal_fn(state) is False

    # -----------------------------------------------------------------------
    # compile_goal — near comparator (within radius)
    # -----------------------------------------------------------------------

    def test_compile_near_within_radius(self) -> None:
        """Near predicate: position within Manhattan radius."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "near": [5, 10], "radius": 2})
        # (6, 10) is distance 1 from (5, 10) — within radius 2
        state = _state(0, "pos", (6, 10))
        assert goal_fn(state) is True

    def test_compile_near_exactly_on_target(self) -> None:
        """Near predicate: exactly on target position (radius 0)."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "near": [5, 10], "radius": 0})
        state = _state(0, "pos", (5, 10))
        assert goal_fn(state) is True

    def test_compile_near_outside_radius(self) -> None:
        """Near predicate: position outside Manhattan radius."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "near": [5, 10], "radius": 2})
        # (8, 10) is distance 3 from (5, 10) — outside radius 2
        state = _state(0, "pos", (8, 10))
        assert goal_fn(state) is False

    def test_compile_near_default_radius_zero(self) -> None:
        """Near predicate without explicit radius defaults to 0 (exact match)."""
        goal_fn = compile_goal({"dim": "pos", "of": 0, "near": [5, 10]})
        state = _state(0, "pos", (5, 10))
        assert goal_fn(state) is True
        state_miss = _state(0, "pos", (5, 11))
        assert goal_fn(state_miss) is False

    # -----------------------------------------------------------------------
    # compile_goal — conjunction (all)
    # -----------------------------------------------------------------------

    def test_compile_all_conjunction(self) -> None:
        """Conjunction: all sub-predicates must match."""
        goal_fn = compile_goal(
            {
                "all": [
                    {"dim": "pos", "of": 0, "eq": [5, 10]},
                    {"dim": "size", "of": 17, "eq": 8},
                ]
            }
        )
        state = SceneState(
            relevant=(
                (0, ("pos", (5, 10))),
                (17, ("size", 8)),
            ),
            terminal="alive",
        )
        assert goal_fn(state) is True

    def test_compile_all_one_fails(self) -> None:
        """Conjunction: returns False when one sub-predicate fails."""
        goal_fn = compile_goal(
            {
                "all": [
                    {"dim": "pos", "of": 0, "eq": [5, 10]},
                    {"dim": "size", "of": 17, "eq": 8},
                ]
            }
        )
        state = SceneState(
            relevant=(
                (0, ("pos", (5, 10))),
                (17, ("size", 5)),  # wrong size
            ),
            terminal="alive",
        )
        assert goal_fn(state) is False

    def test_compile_all_vacuously_true(self) -> None:
        """Conjunction with empty list is vacuously True."""
        goal_fn = compile_goal({"all": []})
        state = _state(0, "pos", (1, 1))
        assert goal_fn(state) is True

    # -----------------------------------------------------------------------
    # compile_goal — action guard
    # -----------------------------------------------------------------------

    def test_compile_action_always_true(self) -> None:
        """Action guards are ignored for goal predicates — always True."""
        goal_fn = compile_goal({"action": 3})
        state = _state(0, "pos", (0, 0))
        assert goal_fn(state) is True

    # -----------------------------------------------------------------------
    # compile_goal — relative near raises ValueError
    # -----------------------------------------------------------------------

    def test_compile_relative_near_raises(self) -> None:
        """Relative near reference must be resolved before compile_goal."""
        with pytest.raises(ValueError, match="relative near references must be resolved"):
            compile_goal({"dim": "pos", "of": 0, "near": {"of": 5, "radius": 2}})

    # -----------------------------------------------------------------------
    # compile_goal — unknown predicate form raises ValueError
    # -----------------------------------------------------------------------

    def test_compile_unknown_predicate_raises(self) -> None:
        """Unknown predicate form raises ValueError."""
        with pytest.raises(ValueError, match="unknown predicate form"):
            compile_goal({"invalid_key": 42})

    # -----------------------------------------------------------------------
    # resolve_predicate
    # -----------------------------------------------------------------------

    def test_resolve_relative_near(self) -> None:
        """resolve_predicate replaces relative near dict with concrete position."""
        scene = _mock_scene(entity_pos={5: (10, 20)})
        predicate: dict[str, object] = {
            "dim": "pos",
            "of": 0,
            "near": {"of": 5, "radius": 2},
        }
        resolved = resolve_predicate(predicate, scene)
        assert resolved["near"] == [10, 20]
        assert resolved["radius"] == 2
        assert resolved["of"] == 0
        assert resolved["dim"] == "pos"

    def test_resolve_nested_all_with_relative_near(self) -> None:
        """resolve_predicate handles nested all with relative near."""
        scene = _mock_scene(entity_pos={5: (10, 20)})
        predicate: dict[str, object] = {
            "all": [
                {"dim": "pos", "of": 0, "near": {"of": 5, "radius": 2}},
                {"dim": "size", "of": 17, "eq": 8},
            ]
        }
        resolved = resolve_predicate(predicate, scene)
        children = resolved["all"]
        assert isinstance(children, list)
        assert len(children) == 2
        assert children[0]["near"] == [10, 20]
        assert children[0]["radius"] == 2
        assert children[1] == {"dim": "size", "of": 17, "eq": 8}

    def test_resolve_entity_no_position_raises(self) -> None:
        """resolve_predicate raises ValueError when entity has no position."""
        scene = _mock_scene(entity_pos={5: None})
        predicate: dict[str, object] = {
            "dim": "pos",
            "of": 0,
            "near": {"of": 5, "radius": 2},
        }
        with pytest.raises(ValueError, match="entity 5 has no position"):
            resolve_predicate(predicate, scene)

    def test_resolve_no_relative_near_returns_as_is(self) -> None:
        """resolve_predicate returns predicate as-is when no relative near."""
        scene = _mock_scene()
        predicate: dict[str, object] = {"dim": "pos", "of": 0, "eq": [5, 10]}
        resolved = resolve_predicate(predicate, scene)
        assert resolved == predicate

    # -----------------------------------------------------------------------
    # derive_spec_from_predicate
    # -----------------------------------------------------------------------

    def test_derive_spec_simple(self) -> None:
        """derive_spec_from_predicate collects entity IDs and dim names."""
        predicate: dict[str, object] = {"dim": "pos", "of": 0, "eq": [5, 10]}
        entities, dims = derive_spec_from_predicate(predicate)
        assert entities == (0,)
        assert dims == ("pos",)

    def test_derive_spec_multiple_dims(self) -> None:
        """derive_spec_from_predicate collects multiple entities and dims from all conjunction."""
        predicate: dict[str, object] = {
            "all": [
                {"dim": "pos", "of": 0, "eq": [5, 10]},
                {"dim": "size", "of": 17, "eq": 8},
            ]
        }
        entities, dims = derive_spec_from_predicate(predicate)
        assert entities == (0, 17)
        assert dims == ("pos", "size")

    def test_derive_spec_includes_relative_near_ref(self) -> None:
        """derive_spec_from_predicate includes entity IDs from relative near dicts."""
        predicate: dict[str, object] = {
            "dim": "pos",
            "of": 0,
            "near": {"of": 5, "radius": 2},
        }
        entities, dims = derive_spec_from_predicate(predicate)
        assert 0 in entities
        assert 5 in entities
        assert dims == ("pos",)

    def test_derive_spec_empty_conjunction(self) -> None:
        """derive_spec_from_predicate returns empty tuples for empty conjunction."""
        predicate: dict[str, object] = {"all": []}
        entities, dims = derive_spec_from_predicate(predicate)
        assert entities == ()
        assert dims == ()

    # -----------------------------------------------------------------------
    # Integration: resolve + compile
    # -----------------------------------------------------------------------

    def test_resolve_then_compile_near(self) -> None:
        """End-to-end: resolve relative near then compile and evaluate."""
        scene = _mock_scene(entity_pos={5: (10, 20)})
        predicate: dict[str, object] = {
            "dim": "pos",
            "of": 0,
            "near": {"of": 5, "radius": 3},
        }
        resolved = resolve_predicate(predicate, scene)
        goal_fn = compile_goal(resolved)
        # Entity 0 at (12, 19) — Manhattan distance |12-10|+|19-20| = 3 ≤ 3
        state = _state(0, "pos", (12, 19))
        assert goal_fn(state) is True

    def test_resolve_then_compile_near_fail(self) -> None:
        """End-to-end: resolved near fails when outside radius."""
        scene = _mock_scene(entity_pos={5: (10, 20)})
        predicate: dict[str, object] = {
            "dim": "pos",
            "of": 0,
            "near": {"of": 5, "radius": 2},
        }
        resolved = resolve_predicate(predicate, scene)
        goal_fn = compile_goal(resolved)
        # Entity 0 at (15, 20) — Manhattan distance 5 > radius 2
        state = _state(0, "pos", (15, 20))
        assert goal_fn(state) is False

    # -----------------------------------------------------------------------
    # Integration: execute_probe
    # -----------------------------------------------------------------------

    def test_execute_probe_returns_plan(self) -> None:
        """execute_probe resolves predicate, builds spec, and returns BFS result."""
        from unittest.mock import patch

        scene = _mock_scene(entity_pos={5: (10, 20)})
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "near": {"of": 5, "radius": 3}},
            max_steps=100,
        )
        fake_start = _state(0, "pos", (12, 19))
        fake_ctx = MagicMock()
        fake_actions = [0, 1, 2, 3]

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=([1, 2], [])) as mock_bfs,
        ):
            result = execute_probe(goal, scene, fake_ctx, fake_actions)

        assert result == ([1, 2], [])
        mock_bfs.assert_called_once()
        call_kwargs = mock_bfs.call_args
        assert call_kwargs[1]["max_nodes"] == 100

    def test_execute_probe_returns_none_when_no_start(self) -> None:
        """execute_probe returns None when snapshot_from_scene returns None."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(target={"dim": "pos", "of": 0, "eq": [5, 10]})
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=None),
            patch("planning.probe.plan_bfs") as mock_bfs,
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1])

        assert result == (None, [])
        mock_bfs.assert_not_called()

    def test_execute_probe_returns_none_when_no_plan(self) -> None:
        """execute_probe returns None when plan_bfs finds no plan."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(target={"dim": "pos", "of": 0, "eq": [5, 10]})
        fake_start = _state(0, "pos", (0, 0))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=(None, [])),
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == (None, [])

    def test_execute_probe_with_explicit_entities_dims(self) -> None:
        """execute_probe uses explicit entities/dims when provided."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            entities=(0, 17),
            dims=("pos", "size"),
            max_steps=50,
        )
        fake_start = _state(0, "pos", (5, 10))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=([], [])) as mock_bfs,
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1])

        assert result == ([], [])
        assert mock_bfs.call_args[1]["max_nodes"] == 50


# ===========================================================================
# TestExecuteProbeWithAction
# ===========================================================================


@pytest.mark.unit
class TestExecuteProbeWithAction:
    """Tests for execute_probe when ProbeGoal has an action field."""

    # -----------------------------------------------------------------------
    # ProbeGoal construction with action
    # -----------------------------------------------------------------------

    def test_probegoal_with_action_construction(self) -> None:
        """ProbeGoal accepts action field and defaults to None."""
        goal_no_action = ProbeGoal(target={"dim": "pos", "of": 0, "eq": [5, 10]})
        assert goal_no_action.action is None

        goal_with_action = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=3,
        )
        assert goal_with_action.action == 3

    # -----------------------------------------------------------------------
    # action=3 and already at target → returns ([action], [])
    # -----------------------------------------------------------------------

    def test_action_already_at_target_returns_action_only(self) -> None:
        """When goal.action is set and start state satisfies target, returns ([action], [])."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=3,
            max_steps=50,
        )
        # Start state already satisfies the goal predicate
        fake_start = _state(0, "pos", (5, 10))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs") as mock_bfs,
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == ([3], [])
        mock_bfs.assert_not_called()

    # -----------------------------------------------------------------------
    # action=3 and BFS finds plan → returns (plan + [action], [])
    # -----------------------------------------------------------------------

    def test_action_bfs_finds_plan_appends_action(self) -> None:
        """When goal.action is set and BFS finds a plan, returns plan + [action]."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=3,
            max_steps=50,
        )
        # Start state does NOT satisfy the goal (wrong position)
        fake_start = _state(0, "pos", (0, 0))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=([1, 1, 2], [])),
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == ([1, 1, 2, 3], [])

    # -----------------------------------------------------------------------
    # action=3 and BFS fails → returns (None, unknowns)
    # -----------------------------------------------------------------------

    def test_action_bfs_fails_returns_none_unknowns(self) -> None:
        """When goal.action is set and BFS fails, returns (None, unknowns)."""
        from unittest.mock import patch

        from planning.query import UnknownAction

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=3,
            max_steps=50,
        )
        fake_start = _state(0, "pos", (0, 0))
        fake_ctx = MagicMock()
        unknown_state = _state(0, "pos", (3, 3))
        unknowns = [UnknownAction(action=4, state=unknown_state)]

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=(None, unknowns)),
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == (None, unknowns)
        # The unknowns are passed through as-is; action is NOT appended to a None plan

    # -----------------------------------------------------------------------
    # action=None (default) and BFS succeeds → returns (plan, [])
    # -----------------------------------------------------------------------

    def test_no_action_bfs_succeeds_returns_plan(self) -> None:
        """When goal.action is None and BFS succeeds, returns (plan, []) unchanged."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=None,
            max_steps=50,
        )
        fake_start = _state(0, "pos", (0, 0))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=([1, 2], [])),
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == ([1, 2], [])

    # -----------------------------------------------------------------------
    # action=None (default) and BFS fails → returns (None, unknowns)
    # -----------------------------------------------------------------------

    def test_no_action_bfs_fails_returns_none_unknowns(self) -> None:
        """When goal.action is None and BFS fails, returns (None, unknowns)."""
        from unittest.mock import patch

        from planning.query import UnknownAction

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=None,
            max_steps=50,
        )
        fake_start = _state(0, "pos", (0, 0))
        fake_ctx = MagicMock()
        unknowns = [UnknownAction(action=7, state=_state(0, "pos", (1, 1)))]

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs", return_value=(None, unknowns)),
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == (None, unknowns)

    # -----------------------------------------------------------------------
    # plan_bfs NOT called when already at target with action
    # -----------------------------------------------------------------------

    def test_plan_bfs_not_called_when_already_at_target_with_action(self) -> None:
        """Verify plan_bfs is never called when start state already satisfies target and action is set."""
        from unittest.mock import patch

        scene = _mock_scene()
        goal = ProbeGoal(
            target={"dim": "pos", "of": 0, "eq": [5, 10]},
            action=3,
            max_steps=50,
        )
        fake_start = _state(0, "pos", (5, 10))
        fake_ctx = MagicMock()

        with (
            patch("planning.probe.snapshot_from_scene", return_value=fake_start),
            patch("planning.probe.plan_bfs") as mock_bfs,
        ):
            result = execute_probe(goal, scene, fake_ctx, [0, 1, 2, 3])

        assert result == ([3], [])
        mock_bfs.assert_not_called()