"""Services container and factory for the Duck harness agent.

The Duck harness agent wires a single LLM callable into a sandboxed
 Duck script.  The service layer is responsible for building the
LLM client, optional call logger, and the wrapped ``llm_chat`` callable.
Tools are NOT injected here; the agent passes them per-call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agents.duck_harness_agent.config import DuckAgentConfig
from agents.llm_client import LLMClient
from agents.recorder import Recorder
from agents.templates.llm_logging import LlmCallLogger, wrap_llm_call


@dataclass
class DuckServices:
    """Service container for one Duck harness agent session."""

    llm_client: LLMClient
    llm_logger: LlmCallLogger | None
    llm_chat: Callable[..., Any]
    config: DuckAgentConfig


def create_services(
    recorder: Recorder | None,
    frame_indexer: Callable[[], int],
    config: DuckAgentConfig,
) -> DuckServices:
    """Build the service container for a single Duck harness agent session.

    ``llm_chat`` is the LLM callable the sandboxed Duck script invokes.
    When a recorder is present, calls are wrapped with ``kind="duck"`` and
    logged to the per-recording ``.llm.jsonl`` sidecar.  No tools are bound
    here; the Duck script/agent supplies them when it calls ``llm_chat``.
    """
    llm_client = LLMClient()
    llm_logger: LlmCallLogger | None = None
    if recorder is not None:
        llm_logger = LlmCallLogger(
            guid=recorder.guid,
            path=recorder.llm_log_path(),
            frame_indexer=frame_indexer,
        )

    if llm_logger is None:
        llm_chat = llm_client.chat
    else:
        llm_chat = wrap_llm_call(
            llm_client.chat,
            llm_logger,
            kind="duck",
            thinking=config.llm_thinking,
            max_tokens=None,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
        )

    return DuckServices(
        llm_client=llm_client,
        llm_logger=llm_logger,
        llm_chat=llm_chat,
        config=config,
    )
