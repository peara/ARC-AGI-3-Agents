"""Agent wrapper for the LangGraph vision-agent workflow."""

import os
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState
from langgraph.pregel import Pregel

from ..agent import Agent
from .config import VisionAgentConfig, load_config
from .graph import build_workflow
from .logging import extract_state_for_recording, log_frame
from .services import AgentServices, create_services
from .state import GameState as LangGraphState


class LangGraphVisionAgent(Agent):
    """A perception-first LangGraph agent for ARC-AGI-3."""

    MAX_ACTIONS = 60

    _workflow: Pregel
    _services: AgentServices
    _config: VisionAgentConfig
    _frame_index: int
    _state: dict[str, Any] | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        config = load_config(os.getenv("LANGGRAPH_VISION_CONFIG"))
        self._config = config
        self._services = create_services(
            recorder=self.recorder,
            frame_indexer=lambda: self._frame_index,
            config=config,
        )
        self._workflow = build_workflow(self._services)
        self._frame_index = 0
        self._state = None

        if kwargs.get("max_actions") is None:
            self.MAX_ACTIONS = config.max_actions

    @property
    def name(self) -> str:
        return f"{super().name}.langgraphvision"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return True when the level is won or the action budget is exhausted."""
        return latest_frame.state is GameState.WIN or self.action_counter >= self.MAX_ACTIONS

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Run the LangGraph workflow and return the chosen action."""
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        self._frame_index += 1

        state_dict: LangGraphState = {
            **(self._state or {}),
            "latest_frame": latest_frame,
            "available_actions": latest_frame.available_actions or [],
            "frame_index": self._frame_index,
            "node_path": [],
        }

        start_time = time.time()
        output: LangGraphState = self._workflow.invoke(state_dict)
        latency_ms = int((time.time() - start_time) * 1000)

        self._state = dict(output)
        action = self._state.get("action")
        if action is not None:
            self._state["last_action_id"] = action.value

        # Inject reasoning into the ARC engine log
        if action is not None and isinstance(action, GameAction) and action != GameAction.RESET:
            action.reasoning = {
                "plan": str(self._state.get("plan", ""))[:8000],
                "action_id": action.value,
                "expectation": str(self._state.get("expectation", ""))[:2000],
                "needs_reflection": bool(self._state.get("needs_reflection", False)),
            }

        log_frame(
            frame_index=self._frame_index,
            node_path=self._state.get("node_path", []),
            action=action if isinstance(action, GameAction) else None,
            uncertain=bool(self._state.get("uncertain_about")),
            reason=str(self._state.get("plan", "")),
            latency_ms=latency_ms,
        )

        if not isinstance(action, GameAction):
            return GameAction.RESET
        return action

    def _extra_record_data(self) -> dict[str, Any]:
        """Attach a serialisable snapshot of LangGraph state to each frame."""
        return {"langgraph_state": extract_state_for_recording(self._state or {})}
