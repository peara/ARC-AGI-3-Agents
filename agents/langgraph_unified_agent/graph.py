"""LangGraph StateGraph builder for the unified agent workflow.

The unified workflow has only two real nodes: ``observe`` and ``unified``.
Observation is reused from the vision agent.  The unified node (stub until
Task 4) is responsible for both reflection and planning in a single LLM call.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command

from agents.langgraph_vision_agent.observe import make_observe_node

from .nodes.unified import make_unified_node
from .services import AgentServices
from .state import UnifiedState


def _with_path_tracking(name: str, node_fn):
    """Wrap a node function so it appends *name* to ``node_path``."""

    def wrapper(state: dict) -> dict | Command:
        path = list(state.get("node_path", []))
        path.append(name)
        result = node_fn(state)

        if isinstance(result, Command):
            update = dict(result.update) if result.update else {}
            update["node_path"] = path
            return Command(goto=result.goto, update=update)
        merged = dict(result) if result else {}
        merged["node_path"] = path
        return merged

    return wrapper


def build_workflow(services: AgentServices) -> Pregel:
    """Build and compile the LangGraph unified-agent workflow.

    Returns a compiled :class:`Pregel` graph ready for ``.invoke()``.
    """
    graph = StateGraph(
        UnifiedState,
        input_schema=UnifiedState,
        output_schema=UnifiedState,
    )

    graph.add_node(
        "observe", _with_path_tracking("observe", make_observe_node(services))
    )
    graph.add_node(
        "unified", _with_path_tracking("unified", make_unified_node(services))
    )

    graph.add_edge(START, "observe")
    graph.add_edge("observe", "unified")
    graph.add_edge("unified", END)

    return graph.compile()
