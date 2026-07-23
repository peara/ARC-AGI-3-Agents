"""Tests for effects.residual — approximate float comparison."""

from effects.residual import ResidualEntry, _values_differ, compute_residual
from effects.state import SceneState


class TestValuesDiffer:
    def test_float_close_values_no_diff(self):
        assert not _values_differ(31.499999999999996, 31.5)

    def test_float_truly_different(self):
        assert _values_differ(31.0, 32.0)

    def test_float_small_diff(self):
        assert _values_differ(31.0, 31.001)

    def test_int_same(self):
        assert not _values_differ(3, 3)

    def test_int_different(self):
        assert _values_differ(3, 4)

    def test_tuple_close_floats_no_diff(self):
        assert not _values_differ((27.25, 31.499999999999996), (27.25, 31.5))

    def test_tuple_truly_different(self):
        assert _values_differ((31.0,), (32.0,))

    def test_tuple_mixed_int_float_no_diff(self):
        assert not _values_differ((5, 3.000000000000001), (5, 3.0))

    def test_tuple_different_lengths(self):
        assert _values_differ((1,), (1, 2))

    def test_none_both_none(self):
        assert not _values_differ(None, None)

    def test_none_vs_value(self):
        assert _values_differ(None, 1.0)
        assert _values_differ(1.0, None)

    def test_str_same(self):
        assert not _values_differ("alive", "alive")

    def test_str_different(self):
        assert _values_differ("alive", "game_over")

    def test_frozenset_different(self):
        assert _values_differ(frozenset([(0, 0)]), frozenset([(0, 1)]))

    def test_frozenset_same(self):
        s = frozenset([(0, 0)])
        assert not _values_differ(s, s)


class TestComputeResidualFloatTolerance:
    def _make_state(self, *entries: tuple[int, str, object]) -> SceneState:
        return SceneState(relevant=tuple((eid, (dim, val)) for eid, dim, val in entries))

    def test_pos_float_artifact_not_flagged(self):
        pred = self._make_state((1, "pos", (27.25, 31.499999999999996)))
        obs = self._make_state((1, "pos", (27.25, 31.5)))
        result = compute_residual(pred, obs, entity_ids=(1,), dims=("pos",))
        assert result == ()

    def test_pos_truly_different_flagged(self):
        pred = self._make_state((1, "pos", (31.0,)))
        obs = self._make_state((1, "pos", (32.0,)))
        result = compute_residual(pred, obs, entity_ids=(1,), dims=("pos",))
        assert len(result) == 1
        assert result[0].dim == "pos"

    def test_orientation_exact_comparison(self):
        pred = self._make_state((1, "orientation", 2))
        obs = self._make_state((1, "orientation", 2))
        result = compute_residual(pred, obs, entity_ids=(1,), dims=("orientation",))
        assert result == ()

    def test_orientation_mismatch_flagged(self):
        pred = self._make_state((1, "orientation", 2))
        obs = self._make_state((1, "orientation", 3))
        result = compute_residual(pred, obs, entity_ids=(1,), dims=("orientation",))
        assert len(result) == 1

    def test_size_float_artifact_not_flagged(self):
        pred = self._make_state((1, "size", 4.000000000000001))
        obs = self._make_state((1, "size", 4.0))
        result = compute_residual(pred, obs, entity_ids=(1,), dims=("size",))
        assert result == ()

    def test_terminal_exact_comparison(self):
        pred = SceneState(relevant=(), terminal="alive")
        obs = SceneState(relevant=(), terminal="alive")
        result = compute_residual(pred, obs, entity_ids=(), dims=(), include_terminal=True)
        assert result == ()