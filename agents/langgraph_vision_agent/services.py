"""Services container and factory for the LangGraph vision agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from agents.llm_client import ChatResponse, LLMClient
from agents.recorder import Recorder
from agents.templates.llm_logging import LlmCallLogger, wrap_llm_call

from .config import VisionAgentConfig


@dataclass
class AgentServices:
    """Collapsible dependency bundle shared by LangGraph nodes."""

    llm_client: LLMClient
    llm_logger: LlmCallLogger | None
    images_dir: str | None
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

    images_dir = recorder.images_dir_path() if recorder is not None else None

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
        images_dir=images_dir,
        planner_call=planner_call,
        reflector_call=reflector_call,
        experimenter_call=experimenter_call,
        config=config,
    )


T = TypeVar("T")


def call_with_retry(
    llm_call: Callable[[list[dict[str, Any]]], Any],
    messages: list[dict[str, Any]],
    parse_fn: Callable[[str], T | None],
    *,
    max_attempts: int = 3,
    nudge_prefix: str = "Your previous response did not match the expected format",
) -> tuple[T | None, str, int]:
    """Call an LLM, parse the response, and retry with a nudge on parse failure.

    Modeled on the validation loop in planning/llm_planner.py.

    Args:
        llm_call: Callable that takes messages and returns an LLM response
            (str or ChatResponse with .content attribute).
        messages: Initial message list to send.
        parse_fn: Function that parses raw response text. Returns parsed result
            on success, None on parse failure.
        max_attempts: Maximum number of LLM calls (default 3 = 1 initial + 2 retries).
        nudge_prefix: Prefix for the nudge message on retry.

    Returns:
        (parsed_result_or_None, raw_response_text, attempts_used).
        If all attempts fail to parse, returns (None, last_raw, max_attempts).
    """
    current_messages = list(messages)
    raw_str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm_call(current_messages)
        except Exception:
            # LLM exceptions are not retried here — let the caller handle them
            raise

        raw_str = response if isinstance(response, str) else getattr(response, "content", str(response))
        parsed = parse_fn(raw_str)

        if parsed is not None:
            return parsed, raw_str, attempt

        # Parse failed — nudge and retry
        if attempt < max_attempts:
            current_messages = current_messages + [
                {"role": "assistant", "content": raw_str},
                {"role": "user", "content": f"{nudge_prefix}. Please output in the expected format."},
            ]

    return None, raw_str, max_attempts
