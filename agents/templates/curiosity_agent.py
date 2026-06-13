"""Curiosity agent: perception session + exploration policy.

The agent owns orchestration (env handshake, recording). Perception lives in
``PerceptionSession``; action selection in ``ExplorationPolicy``. A future LLM
planner swaps in at the policy slot without touching the session.
"""

import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning import (
    ExplorationConfig,
    ExplorationPolicy,
    PlannerStatus,
)

from ..agent import Agent


class Curiosity(Agent):
    """Perception session + curiosity exploration policy."""

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        self.session = PerceptionSession()
        self.policy = ExplorationPolicy(
            action_space=[a.value for a in GameAction if a is not GameAction.RESET],
            config=ExplorationConfig(seed=seed),
        )
        self._last_action_id = RESET_ACTION
        self._last_observed_frame_id: int | None = None
        self._scene: SceneSnapshot | None = None

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

        if latest_frame.frame and id(latest_frame) != self._last_observed_frame_id:
            self._scene = self.session.ingest(latest_frame.frame, self._last_action_id)
            self.policy.on_observed(self._scene)
            self._last_observed_frame_id = id(latest_frame)

        scene = self._scene or self.session.snapshot()
        available = latest_frame.available_actions or None
        action_id = self.policy.decide(scene, available)
        action = GameAction.from_id(action_id)
        self._last_action_id = action_id

        status = self.policy.status()
        reason = _format_status(status)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"phase": status.phase, "note": reason}
        else:
            action.reasoning = reason
        return action


def _format_status(status: PlannerStatus) -> str:
    return (
        f"{status.phase} ctrl={status.controllable_id} "
        f"target={status.target} plan={status.plan_len} "
        f"visited={status.n_visited}"
    )
