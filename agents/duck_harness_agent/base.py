"""DirectStepAgent — base class for agents that step the environment from choose_action().

Unlike the default Agent.main() loop which calls take_action() after choose_action(),
DirectStepAgent.main() only calls choose_action() and increments the counter. The
concrete subclass is expected to call step_env() from within choose_action() (or from
a sandbox callback) to advance the environment.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from arcengine import FrameData, GameAction

from ..agent import Agent, _active_log_path
from ..tracing import trace_agent_session

logger = logging.getLogger()


class DirectStepAgent(Agent, ABC):
    """Agent whose main loop does NOT call take_action().

    The loop body is:
        action = choose_action(frames, latest_frame)
        action_counter += 1

    Subclasses call step_env(action) when they are ready to advance the
    environment — typically from within choose_action() or from a sandbox
    action callback.

    step_env(action) wraps take_action() + append_frame() and returns the
    resulting FrameData, so the subclass never needs to call those directly.
    """

    @trace_agent_session
    def main(self) -> None:  # noqa: D401 — imperative mood
        """Main loop: choose_action only — no take_action call."""
        self.timer = time.time()
        if self._log_handler is not None:
            _active_log_path.set(self._log_handler.baseFilename)

        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            self.choose_action(
                self.frames,
                self._convert_raw_frame_data(
                    self.arc_env.observation_space if self.arc_env else None
                ),
            )
            self.action_counter += 1

        self.cleanup()

    def step_env(self, action: GameAction) -> Optional[FrameData]:
        """Step the environment and record the result.

        Calls take_action() then append_frame(), returning the new frame.
        Returns ``None`` if the frame data failed validation.
        """
        frame = self.take_action(action)
        if frame is not None:
            self.append_frame(frame, action)
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, "
                f"levels completed {frame.levels_completed}, avg fps {self.fps})"
            )
        return frame

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take.

        Subclasses typically call ``self.step_env(action)`` inside this method
        (or from a sandbox callback) to advance the environment.
        """
        raise NotImplementedError