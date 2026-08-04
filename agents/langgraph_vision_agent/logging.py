"""Node-level logging infrastructure for the LangGraph vision agent.

Uses the standard library ``logging`` module.  The existing
``Agent._install_log_handler`` attaches a file handler to the root logger
that routes all records to the recording ``.logs.log`` sidecar, so no new
file handlers or sidecars are needed here.
"""

import logging
from typing import Any

from arcengine import GameAction

frame_logger = logging.getLogger("langgraph.frame")
node_logger = logging.getLogger("langgraph.node")

frame_logger.setLevel(logging.DEBUG)
node_logger.setLevel(logging.DEBUG)


def log_frame(
    frame_index: int,
    node_path: list[str],
    action: GameAction | None,
    uncertain: bool,
    reason: str,
    latency_ms: int,
) -> None:
    """Emit one INFO-level log line summarising the frame decision."""
    action_repr = f"{action.value}" if action is not None else "None"
    path_repr = "/".join(node_path) if node_path else "none"
    frame_logger.info(
        "frame=%s path=%s action=%s uncertain=%s reason=%r latency_ms=%s",
        frame_index,
        path_repr,
        action_repr,
        uncertain,
        reason,
        latency_ms,
    )


def log_node(frame_index: int, node_name: str, **diffs: Any) -> None:
    """Emit a DEBUG-level log line describing state changes inside a node."""
    diff_parts = ", ".join(f"{k}={v!r}" for k, v in diffs.items())
    node_logger.debug("frame=%s node=%s %s", frame_index, node_name, diff_parts)


def extract_state_for_recording(state: dict[str, Any]) -> dict[str, Any]:
    """Return a serialisable subset of LangGraph state for the recording."""
    return {
        "mechanics": state.get("mechanics", []),
        "mechanics_summary": state.get("mechanics_summary", ""),
        "tactical": state.get("tactical", []),
        "tactical_summary": state.get("tactical_summary", ""),
        "plan": state.get("plan", ""),
        "uncertain_about": state.get("uncertain_about"),
        "needs_reflection": state.get("needs_reflection", False),
        "expectation": state.get("expectation", ""),
        "history": state.get("history", []),
        "node_path": state.get("node_path", []),
    }
