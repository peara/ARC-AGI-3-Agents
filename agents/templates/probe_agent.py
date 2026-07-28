"""Deterministic probe agent — cycles through available actions in sorted order.

Designed for cold-start mechanics experiments: produces a clean recording where
each available action is taken exactly once (in ascending order), giving the
mechanics prompt one observation per action with no overlap.

Usage:
    uv run main.py --agent=probe --game=wa30 --max-actions 6

The first frame is always RESET (env handshake). Subsequent frames take
``available_actions`` in sorted order, cycling if the budget exceeds the action
count. The agent stops as soon as every available action has been taken at
least once (or when ``MAX_ACTIONS`` is reached).
"""

from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent


class Probe(Agent):
    """Take each available action once, in sorted order, then stop."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._probe_queue: list[int] = []
        self._probe_started = False
        self._seen_actions: set[int] = set()

    @property
    def name(self) -> str:
        return f"{super().name}.probe"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Done once we've taken every available action at least once AND
        # at least one extra action beyond the full set (so the recording
        # captures a transition for the last action in the set).
        if not self._probe_started:
            return False
        available = self._current_available(latest_frame)
        if not available:
            return False
        seen_all = available.issubset(self._seen_actions)
        extra = len(self._seen_actions) > len(available - {0})
        return seen_all and extra

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        available = self._current_available(latest_frame)
        if not available:
            return GameAction.RESET

        # On the first real frame, build the probe queue from the sorted
        # available actions (excluding RESET).
        if not self._probe_started:
            self._probe_queue = sorted(available - {0})
            self._probe_started = True

        # Refill if we've exhausted the queue but haven't seen all actions yet
        # (can happen if available_actions changed mid-game).
        if not self._probe_queue:
            remaining = sorted(available - self._seen_actions - {0})
            self._probe_queue = remaining if remaining else sorted(available - {0})

        action_id = self._probe_queue.pop(0)
        self._seen_actions.add(action_id)
        action = GameAction.from_id(action_id)
        if action.is_simple():
            action.reasoning = f"probe: action {action_id}"
        return action

    @staticmethod
    def _current_available(latest_frame: FrameData) -> set[int]:
        raw = getattr(latest_frame, "available_actions", None)
        if raw:
            return set(raw)
        return {a.value for a in GameAction if a is not GameAction.RESET}