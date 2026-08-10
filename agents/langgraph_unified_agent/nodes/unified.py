"""Temporary stub for the unified LangGraph node.

This node combines planning and reflection into a single LLM call.  The
implementation here is intentionally minimal and will be replaced by the
real unified node in a follow-up task.
"""

from __future__ import annotations

from typing import Any, Callable

from arcengine import GameAction

from ..services import AgentServices


def make_unified_node(services: AgentServices) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a stub unified node that emits a fixed default action."""

    def unified_node(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "action": GameAction.from_id(0),
            "plan": "stub unified plan",
            "node_path": list(state.get("node_path", [])),
        }

    return unified_node
