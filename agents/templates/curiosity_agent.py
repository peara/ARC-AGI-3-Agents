"""Curiosity agent: random cold start -> perception -> BFS verify/replan loop.

A thin online driver around ``perception.exploration.ExplorationPlanner``. The
agent owns the env handshake (RESET / done / complex-action args); the planner
owns perception and decision-making, so the same logic is exercised offline by
``tests/unit/test_exploration.py``.

The agent does not hardcode which blob it controls. It probes randomly until the
perception layer confirms a controllable entity, then lets BFS steer that entity
toward the unknown, verifying each step against the next frame.
"""

import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from perception.exploration import RESET_ACTION, ExplorationConfig, ExplorationPlanner

from ..agent import Agent


class Curiosity(Agent):
    """Perception-driven explorer with an online BFS verify/replan loop."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        self.planner = ExplorationPlanner(
            action_space=[a.value for a in GameAction if a is not GameAction.RESET],
            config=ExplorationConfig(seed=seed),
        )
        self._last_action_id = RESET_ACTION
        self._last_observed_frame_id: int | None = None

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._last_action_id = RESET_ACTION
            return GameAction.RESET

        # Observe the frame our previous action produced, exactly once. A failed
        # step returns no new frame object, so id() guards against double-ingest.
        if latest_frame.frame and id(latest_frame) != self._last_observed_frame_id:
            self.planner.observe(latest_frame.frame, self._last_action_id)
            self._last_observed_frame_id = id(latest_frame)

        available = latest_frame.available_actions or None
        action_id = self.planner.decide(available)
        action = GameAction.from_id(action_id)
        self._last_action_id = action_id

        status = self.planner.status()
        reason = (
            f"{status.phase} ctrl={status.controllable_id} "
            f"target={status.target} plan={status.plan_len} "
            f"visited={status.n_visited}"
        )
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"phase": status.phase, "note": reason}
        else:
            action.reasoning = reason
        return action
