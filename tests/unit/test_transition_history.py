"""Tests for effects.transition_history."""

from __future__ import annotations

from effects.state import SceneState
from effects.transition_history import TransitionHistory


def _state(pos: tuple[int, int]) -> SceneState:
    return SceneState(relevant=((0, ("pos", pos)),))


def test_empty_history():
    h = TransitionHistory()
    assert len(h) == 0
    assert h.last_n(5) == []
    assert list(h) == []


def test_append_and_index():
    h = TransitionHistory()
    s0, s1, s2 = _state((1, 1)), _state((1, 2)), _state((1, 3))
    h.append(state_before=s0, action=1, state_after=s1, frame_idx=0)
    h.append(state_before=s1, action=2, state_after=s2, frame_idx=1)

    assert len(h) == 2
    assert h[0].frame_idx == 0
    assert h[0].action == 1
    assert h[0].state_before is s0
    assert h[0].state_after is s1
    assert h[-1].frame_idx == 1


def test_last_n():
    h = TransitionHistory()
    for i in range(5):
        h.append(
            state_before=_state((i, 0)),
            action=i,
            state_after=_state((i + 1, 0)),
            frame_idx=i,
        )
    assert len(h.last_n(3)) == 3
    assert h.last_n(3)[0].frame_idx == 2
    assert h.last_n(3)[2].frame_idx == 4
    assert len(h.last_n(100)) == 5
    assert h.last_n(0) == []


def test_filter_by_action():
    h = TransitionHistory()
    for i in range(4):
        h.append(
            state_before=_state((i, 0)),
            action=1 if i % 2 == 0 else 2,
            state_after=_state((i + 1, 0)),
            frame_idx=i,
        )
    action1 = list(h.filter(action=1))
    assert len(action1) == 2
    assert all(t.action == 1 for t in action1)
    assert list(h.filter(action=99)) == []
    all_t = list(h.filter())
    assert len(all_t) == 4


def test_iteration():
    h = TransitionHistory()
    for i in range(3):
        h.append(
            state_before=_state((i, 0)),
            action=i,
            state_after=_state((i + 1, 0)),
            frame_idx=i,
        )
    ids = [t.frame_idx for t in h]
    assert ids == [0, 1, 2]