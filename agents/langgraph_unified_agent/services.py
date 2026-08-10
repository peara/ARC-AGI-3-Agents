"""Services container and factory for the unified LangGraph agent.

The unified agent reuses the collapsible :class:`AgentServices` from the
vision agent.  Only the planner LLM callable is wired up (as ``unified_call``);
the reflector and experimenter slots are no-op placeholders because this
workflow has only two real nodes: ``observe`` and ``unified``.
"""

from __future__ import annotations

from typing import Callable

from agents.langgraph_unified_agent.config import UnifiedAgentConfig
from agents.langgraph_vision_agent.services import (
    AgentServices,
    _make_callable,
)
from agents.llm_client import LLMClient
from agents.recorder import Recorder
from agents.templates.llm_logging import LlmCallLogger


def _noop_callable(_messages: list[dict[str, str]]) -> str:
    """Placeholder for unused LLM callables."""
    return ""


def create_services(
    recorder: Recorder | None,
    frame_indexer: Callable[[], int],
    config: UnifiedAgentConfig,
) -> "AgentServices":
    """Build the service container for a single unified agent session.

    The planner callable is reused as the unified LLM callable, with
    ``kind="unified"`` in the LLM logger so unified calls can be distinguished
    from the vision agent's ``planner`` calls.
    """
    llm_client = LLMClient()
    llm_logger: LlmCallLogger | None = None
    if recorder is not None:
        llm_logger = LlmCallLogger(
            guid=recorder.guid,
            path=recorder.llm_log_path(),
            frame_indexer=frame_indexer,
        )

    images_dir = recorder.images_dir_path() if recorder is not None else None

    unified_call = _make_callable(
        llm_client,
        llm_logger,
        kind="unified",
        thinking=config.llm_thinking,
        max_tokens=config.unified_max_tokens,
    )

    return AgentServices(
        llm_client=llm_client,
        llm_logger=llm_logger,
        images_dir=images_dir,
        planner_call=unified_call,
        reflector_call=_noop_callable,
        experimenter_call=_noop_callable,
        config=config,
    )
