"""Curiosity exploration loop tests."""

from __future__ import annotations

import numpy as np
import pytest

from effects import learn_movement_model
from perception.session import RESET_ACTION, PerceptionSession
from planning import ExplorationConfig, ExplorationPolicy
from tests.perception_fixtures import load_manifest

SIM_STEP = 3
SIM_MOTION = {
    1: (-SIM_STEP, 0),
    2: (SIM_STEP, 0),
    3: (0, -SIM_STEP),
    4: (0, SIM_STEP),
}
BG = 0
PLAYER = 9
WALL = 8


class GridWorld:
    def __init__(self, rows: int = 30, cols: int = 30) -> None:
        self.rows = rows
        self.cols = cols
        self.walls: set[tuple[int, int]] = set()
        for r in range(6, rows):
            for c in range(6, cols):
                interior = 9 <= r <= 20 and 9 <= c <= 20
                frame = (6 <= r <= 23) and (6 <= c <= 23)
                if frame and not interior:
                    self.walls.add((r, c))
        self.player = (9, 9)

    def _player_cells(self, top: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = top
        return [(r + dr, c + dc) for dr in range(3) for dc in range(3)]

    def _legal(self, top: tuple[int, int]) -> bool:
        for r, c in self._player_cells(top):
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False
            if (r, c) in self.walls:
                return False
        return True

    def step(self, action_id: int) -> None:
        delta = SIM_MOTION.get(int(action_id))
        if delta is None:
            return
        cand = (self.player[0] + delta[0], self.player[1] + delta[1])
        if self._legal(cand):
            self.player = cand

    def frame(self) -> np.ndarray:
        grid = np.full((self.rows, self.cols), BG, dtype=np.int16)
        for r, c in self.walls:
            grid[r, c] = WALL
        for r, c in self._player_cells(self.player):
            grid[r, c] = PLAYER
        return grid


def _run_loop(steps: int, seed: int = 0):
    world = GridWorld()
    session = PerceptionSession(grid_rows=world.rows, grid_cols=world.cols)
    policy = ExplorationPolicy(
        action_space=list(SIM_MOTION),
        config=ExplorationConfig(seed=seed, min_random_steps=6),
        grid_rows=world.rows,
        grid_cols=world.cols,
    )
    last_action = RESET_ACTION
    statuses = []
    scene = session.snapshot()
    for _ in range(steps):
        scene = session.ingest(world.frame(), last_action)
        policy.on_observed(scene)
        action = policy.decide(scene, list(SIM_MOTION))
        statuses.append(policy.status())
        world.step(action)
        last_action = action
    return session, policy, scene, statuses


@pytest.mark.unit
class TestExplorationLoopSimulated:
    def test_controllable_detected(self):
        _, policy, scene, _ = _run_loop(steps=40)
        assert scene.controllable_id() is not None
        assert policy.controllable_id is not None

    def test_learned_motion_matches_truth(self):
        _, policy, _, _ = _run_loop(steps=40)
        model = policy.model
        assert model is not None and model.motion_by_action
        for action, disp in model.motion_by_action.items():
            assert disp == SIM_MOTION[action]

    def test_enters_bfs_phase(self):
        _, _, _, statuses = _run_loop(steps=40)
        phases = {s.phase for s in statuses}
        assert "explore_random" in phases
        assert "frontier" in phases

    def test_verify_loop_triggers_replan_on_wall(self):
        _, policy, _, statuses = _run_loop(steps=60)
        assert any(s.diverged for s in statuses)
        model = policy.model
        assert model is not None
        assert len(model.known_blocks) >= 1

    def test_exploration_visits_many_cells(self):
        _, policy, _, _ = _run_loop(steps=60)
        assert len(policy.visited) >= 6


@pytest.mark.unit
class TestPerceptionSession:
    def test_from_recording_recovers_controllable(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")
        session, _ = PerceptionSession.from_recording(cases[0].recording.path)
        scene = session.snapshot()
        assert scene.controllable_id() is not None
        model = learn_movement_model(
            scene.registry,
            scene.catalog,
            list(scene.action_ids),
            scene.controllable_id(),
        )
        assert model is not None and model.motion_by_action
        for disp in model.motion_by_action.values():
            assert max(abs(disp[0]), abs(disp[1])) == 5

    def test_summary_is_llm_ready(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")
        session, _ = PerceptionSession.from_recording(cases[0].recording.path)
        summary = session.snapshot().summary()
        assert "entities" in summary
        assert summary["controllable_id"] is not None


@pytest.mark.unit
class TestExplorationOnRecording:
    def test_session_plus_policy_recovers_controllable(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")
        session, frames = PerceptionSession.from_recording(cases[0].recording.path)
        policy = ExplorationPolicy(action_space=[1, 2, 3, 4])
        scene = session.snapshot()
        policy.on_observed(scene)
        policy.decide(scene, [1, 2, 3, 4])
        assert policy.controllable_id is not None
        assert policy.model is not None
