"""Services container and factory for the LangGraph vision agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agents.llm_client import ChatResponse, LLMClient
from agents.recorder import Recorder
from agents.templates.llm_logging import LlmCallLogger, wrap_llm_call

from .config import VisionAgentConfig


@dataclass
class AgentServices:
    """Collapsible dependency bundle shared by LangGraph nodes."""

    llm_client: LLMClient
    llm_logger: LlmCallLogger | None
    planner_call: Callable
    reflector_call: Callable
    experimenter_call: Callable
    config: VisionAgentConfig


def _make_callable(
    llm_client: LLMClient,
    logger: LlmCallLogger | None,
    kind: str,
    thinking: bool,
    max_tokens: int,
) -> Callable[..., ChatResponse | str]:
    """Return a logging-wrapped LLM callable, or the raw chat when no logger."""
    if logger is None:
        return llm_client.chat
    return wrap_llm_call(
        llm_client.chat,
        logger,
        kind=kind,
        thinking=thinking,
        max_tokens=max_tokens,
    )


def create_services(
    recorder: Recorder | None,
    frame_indexer: Callable[[], int],
    config: VisionAgentConfig,
) -> AgentServices:
    """Build the service container for a single agent session."""
    llm_client = LLMClient()
    llm_logger: LlmCallLogger | None = None
    if recorder is not None:
        llm_logger = LlmCallLogger(
            guid=recorder.guid,
            path=recorder.llm_log_path(),
            frame_indexer=frame_indexer,
        )

    planner_call = _make_callable(
        llm_client,
        llm_logger,
        kind="planner",
        thinking=config.llm_thinking,
        max_tokens=config.planner_max_tokens,
    )
    reflector_call = _make_callable(
        llm_client,
        llm_logger,
        kind="reflector",
        thinking=config.llm_thinking,
        max_tokens=config.reflector_max_tokens,
    )
    experimenter_call = _make_callable(
        llm_client,
        llm_logger,
        kind="experimenter",
        thinking=config.llm_thinking,
        max_tokens=config.experimenter_max_tokens,
    )

    return AgentServices(
        llm_client=llm_client,
        llm_logger=llm_logger,
        planner_call=planner_call,
        reflector_call=reflector_call,
        experimenter_call=experimenter_call,
        config=config,
    )
