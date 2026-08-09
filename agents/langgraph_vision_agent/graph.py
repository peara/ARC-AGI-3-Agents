"""LangGraph StateGraph builder for the vision-agent workflow.

Wires together the four nodes (observe → reflect → plan → experiment)
with conditional routing from the plan node.

* Plan returns ``Command(goto="experiment", update={...})`` when uncertain
  → routes to the experiment node.
* Plan returns a plain ``dict`` when confident → routes to END.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.types import Command

from .nodes.experiment import make_experiment_node
from .nodes.plan import make_plan_node
from .nodes.planner_v2 import make_planner_v2_node
from .nodes.reflect import make_reflect_node
from .observe import make_observe_node
from .services import AgentServices
from .state import GameState


def _plan_router(output: dict | Command) -> str:
    """Route plan node output to the next node name.

    * ``Command`` → ``"experiment"``  (uncertain, wants exploration)
    * ``dict``    → ``"__end__"``     (confident, action decided)
    """
    if isinstance(output, Command):
        return "experiment"
    return END


def _with_path_tracking(name: str, node_fn):
    """Wrap a node function so it appends *name* to ``node_path``."""

    def wrapper(state: dict) -> dict | Command:
        path = list(state.get("node_path", []))
        path.append(name)
        # Merge the path update *into* the node's own output so
        # LangGraph reducers see a single dict / Command.
        result = node_fn(state)

        if isinstance(result, Command):
            # Command.update is a dict; merge node_path into it.
            update = dict(result.update) if result.update else {}
            update["node_path"] = path
            return Command(goto=result.goto, update=update)
        # Plain dict return — merge node_path.
        merged = dict(result) if result else {}
        merged["node_path"] = path
        return merged

    return wrapper


def build_workflow(services: AgentServices) -> Pregel:
    """Build and compile the LangGraph vision-agent workflow.

    Returns a compiled :class:`Pregel` graph ready for ``.invoke()``.
    """
    graph = StateGraph(
        GameState,
        input_schema=GameState,
        output_schema=GameState,
    )

    # --- nodes (wrapped with path tracking) ---
    graph.add_node(
        "observe", _with_path_tracking("observe", make_observe_node(services))
    )
    graph.add_node(
        "reflect", _with_path_tracking("reflect", make_reflect_node(services))
    )
    graph.add_node(
        "plan",
        _with_path_tracking(
            "plan",
            make_planner_v2_node(services)
            if services.config.use_planner_v2
            else make_plan_node(services),
        ),
    )

    # --- edges ---
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "reflect")
    graph.add_edge("reflect", "plan")

    if services.config.use_planner_v2:
        graph.add_edge("plan", END)
    else:
        graph.add_node(
            "experiment",
            _with_path_tracking("experiment", make_experiment_node(services)),
        )

        # Conditional: plan → experiment (uncertain) or END (confident)
        graph.add_conditional_edges(
            "plan",
            _plan_router,
            {"experiment": "experiment", END: END},
        )

        graph.add_edge("experiment", END)

    return graph.compile()


def draw_mermaid() -> str:
    """Convenience: build a default workflow and return its Mermaid diagram."""
    from .config import VisionAgentConfig
    from .services import create_services

    services = create_services(
        recorder=None,
        frame_indexer=lambda: 0,
        config=VisionAgentConfig(),
    )
    workflow = build_workflow(services)
    return workflow.get_graph().draw_mermaid()
