"""Curiosity exploration loop tests.

Two angles:
  * A tiny simulated ``GridWorld`` drives the full online loop (random cold
    start -> controllable detection -> BFS toward the unknown -> verify/replan
    on a wall hit). No network, no recording needed.
  * The reference recording validates that ``observe()`` wires the perception
    stack together on real data and recovers the documented controllable map.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.exploration import RESET_ACTION, ExplorationConfig, ExplorationPlanner
from tests.perception_fixtures import build_perception_stack, load_manifest

# Action ids -> (dr, dc) used by the simulator. Matches ls20 semantics.
SIM_STEP = 3
SIM_MOTION = {
    1: (-SIM_STEP, 0),  # up
    2: (SIM_STEP, 0),   # down
    3: (0, -SIM_STEP),  # left
    4: (0, SIM_STEP),   # right
}
BG = 0       # background
PLAYER = 9   # controllable blob
WALL = 8     # blocking structure


class GridWorld:
    """A boxed room. A 3x3 player moves on a step lattice; walls block it.

    Ground truth the planner must *discover*: which blob is the player, how each
    action moves it, and where the (initially unknown) walls are.
    """

    def __init__(self, rows: int = 30, cols: int = 30) -> None:
        self.rows = rows
        self.cols = cols
        # Interior free area rows/cols 9..20; a thick wall frame encloses it.
        self.walls: set[tuple[int, int]] = set()
        for r in range(6, rows):
            for c in range(6, cols):
                interior = 9 <= r <= 20 and 9 <= c <= 20
                frame = (6 <= r <= 23) and (6 <= c <= 23)
                if frame and not interior:
                    self.walls.add((r, c))
        self.player = (9, 9)  # top-left of the 3x3 player

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
    planner = ExplorationPlanner(
        action_space=list(SIM_MOTION),
        config=ExplorationConfig(seed=seed, min_random_steps=6),
        grid_rows=world.rows,
        grid_cols=world.cols,
    )
    last_action = RESET_ACTION
    statuses = []
    for _ in range(steps):
        planner.observe(world.frame(), last_action)
        action = planner.decide(list(SIM_MOTION))
        statuses.append(planner.status())
        world.step(action)
        last_action = action
    return planner, statuses


@pytest.mark.unit
class TestExplorationLoopSimulated:
    def test_controllable_detected(self):
        planner, _ = _run_loop(steps=40)
        assert planner.controllable_id is not None

    def test_learned_motion_matches_truth(self):
        planner, _ = _run_loop(steps=40)
        model = planner.model
        assert model is not None and model.motion_by_action
        # Every learned action→displacement must match the simulator's truth.
        for action, disp in model.motion_by_action.items():
            assert disp == SIM_MOTION[action]

    def test_enters_bfs_phase(self):
        _, statuses = _run_loop(steps=40)
        phases = {s.phase for s in statuses}
        assert "explore_random" in phases  # cold start happened
        assert "frontier" in phases  # BFS toward the unknown kicked in

    def test_verify_loop_triggers_replan_on_wall(self):
        planner, statuses = _run_loop(steps=60)
        # Hitting an unknown wall must surface as a divergence...
        assert any(s.diverged for s in statuses)
        # ...and the surprise must be absorbed into the movement model.
        assert planner.model is not None
        assert len(planner.model.known_blocks) >= 1

    def test_exploration_visits_many_cells(self):
        planner, _ = _run_loop(steps=60)
        # Curiosity should spread the player across the room, not idle.
        assert len(planner.visited) >= 6


@pytest.mark.unit
class TestExplorationOnRecording:
    def _stack(self):
        cases = [c for c in load_manifest() if c.recording.path.is_file()]
        if not cases:
            pytest.skip("no reference recordings available")
        return build_perception_stack(cases[0].recording.path)

    def test_observe_recovers_controllable(self):
        stack = self._stack()
        planner = ExplorationPlanner(action_space=[1, 2, 3, 4])
        for grid, action in zip(stack.frames, stack.action_ids):
            planner.observe(grid, action if action >= 0 else RESET_ACTION)
        assert planner.controllable_id is not None
        # A decision builds the movement model from the observed trajectory.
        planner.decide([1, 2, 3, 4])
        assert planner.model is not None
        # Recorded controllable moves in 5-cell steps (see report).
        assert planner.model.motion_by_action
        for disp in planner.model.motion_by_action.values():
            assert max(abs(disp[0]), abs(disp[1])) == 5
