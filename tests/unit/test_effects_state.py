"""SceneState v2: extendable dims and terminal field."""

from __future__ import annotations

import pytest

from effects import SceneState


@pytest.mark.unit
class TestSceneStateV2:
    def test_default_terminal_is_alive(self):
        state = SceneState(relevant=((0, ("pos", (1, 2))),))
        assert state.terminal == "alive"

    def test_get_set_dim(self):
        state = SceneState(relevant=())
        state = state.set_dim(0, "pos", (3, 4))
        assert state.get(0, "pos") == (3, 4)
        state = state.set_dim(0, "size", 10)
        assert state.get(0, "size") == 10
        assert state.get(0, "pos") == (3, 4)

    def test_set_dim_replaces_existing(self):
        state = SceneState(relevant=((0, ("pos", (1, 1))),))
        state = state.set_dim(0, "pos", (2, 2))
        assert state.get(0, "pos") == (2, 2)
        assert len(state.relevant) == 1

    def test_with_pos_matches_set_dim(self):
        state = SceneState(relevant=())
        via_pos = state.with_pos(1, (5, 6))
        via_set = state.set_dim(1, "pos", (5, 6))
        assert via_pos.relevant == via_set.relevant
        assert via_pos.terminal == via_set.terminal

    def test_with_terminal(self):
        state = SceneState(relevant=())
        dead = state.with_terminal("game_over")
        assert dead.terminal == "game_over"
        assert dead.relevant == state.relevant

    def test_fingerprint_default_excludes_terminal(self):
        a = SceneState(relevant=((0, ("pos", (1, 2))),), terminal="alive")
        b = SceneState(relevant=((0, ("pos", (1, 2))),), terminal="game_over")
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_include_terminal(self):
        a = SceneState(relevant=(), terminal="alive")
        b = SceneState(relevant=(), terminal="game_over")
        assert a.fingerprint(include_terminal=True) != b.fingerprint(
            include_terminal=True
        )
        assert a.fingerprint(include_terminal=True) == ((), "alive")

    def test_latent_dim_round_trip(self):
        state = SceneState(relevant=()).set_dim(0, "pass_wall", True)
        assert state.get(0, "pass_wall") is True
