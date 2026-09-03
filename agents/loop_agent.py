"""LoopAgent — Agent subclass that owns its own action loop via run().

Unlike Agent's choose_action-per-turn contract (or DirectStepAgent which
inherits it), LoopAgent subclasses implement run() — the full game loop
with their own end conditions, persistent conversation, and tool dispatch.

The Agent ABC's is_done/choose_action are vestigial here: they exist only
to satisfy the abstract contract. main() calls run() (which owns end
conditions), not the is_done-guarded loop.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, final

from arcengine import FrameData, GameAction

from .agent import Agent, _active_log_path
from .tracing import trace_agent_session

logger = logging.getLogger()


class LoopAgent(Agent, ABC):
    """Agent whose main loop delegates all action-taking to ``run``.

    The base provides:
    - ``step_env(action)`` — single-action advance: take_action + append_frame
      + action_counter++ + log. Subclasses call this from inside run() when
      the agent decides to act (e.g. from a sandbox action callback).
    - ``@trace_agent_session @final main()`` — final entry point; sets up
      tracing, timer, and log routing, calls self.run(), then cleanup().

    Subclasses implement ``run()`` — the full game loop. ``run()`` owns all
    end conditions (WIN, MAX_ACTIONS, custom caps), persistent conversation
    state, and tool dispatch. The Agent ABC's ``is_done``/``choose_action``
    are provided as non-abstract stubs (return False / raise NotImplementedError)
    to satisfy instantiation; they are NEVER called on the live path.
    """

    @trace_agent_session  # preserves AgentOps tracing; types agent_instance: Agent
    @final                # prevent subclass override — main() owns tracing + cleanup
    def main(self) -> None:
        """Final main entry point. Sets up tracing + log routing, calls run(),
        then cleanup() (even if run() raises)."""
        self.timer = time.time()
        if self._log_handler is not None:
            _active_log_path.set(self._log_handler.baseFilename)
        try:
            self.run()
        finally:
            self.cleanup()

    @abstractmethod
    def run(self) -> None:
        """The full game loop. Subclass owns all end conditions and action taking.

        Call ``self.step_env(action)`` to advance the environment. The base
        class handles counter increment + frame append + logging.
        """
        raise NotImplementedError

    def step_env(self, action: GameAction) -> Optional[FrameData]:
        """Step the environment by one action. Mirrors ``DirectStepAgent.step_env``.

        Returns the new frame, or ``None`` if the frame failed validation.
        ``action_counter`` is incremented here (NOT in main()) so that
        multi-action batching — where the sandbox calls ``action()`` multiple
        times in a single ``run()`` turn — is counted correctly.
        """
        frame = self.take_action(action)
        if frame is not None:
            self.action_counter += 1
            self.append_frame(frame, action)
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, "
                f"levels completed {frame.levels_completed}, avg fps {self.fps})"
            )
        return frame

    # ── Vestigial stubs — satisfy Agent ABC. Never called on the live path. ──

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Vestigial stub. LoopAgent subclasses own end conditions in run()."""
        return False

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Vestigial stub. LoopAgent subclasses implement run() instead."""
        raise NotImplementedError(
            "LoopAgent uses run(), not choose_action(). "
            "Subclass run() to define agent behavior."
        )
