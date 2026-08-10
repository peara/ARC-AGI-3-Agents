"""LLM call logging — records every LLM call (messages + raw response + metadata)
to a dedicated JSONL file for offline analysis.

Usage::

    logger = LlmCallLogger(guid=recorder.guid, path=recorder.llm_log_path(),
                           frame_indexer=lambda: agent._frame_index)
    wrapped = wrap_llm_call(agent.llm_call, logger, kind="planner")
    # wrapped(messages) -> ChatResponse | str  (same signature as llm_call)
    # Each call appends one JSONL line to logger.path.

Truncation: any single message ``content`` longer than ``MAX_CONTENT_CHARS``
is truncated and a ``[...truncated N chars]`` marker is appended. The top-level
event gets ``truncated: true`` so consumers can flag partial payloads.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from agents.llm_client import ChatResponse

log = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 40_000


class LLMTruncationError(Exception):
    """Raised when an LLM response is truncated (finish_reason=='length')
    and ``LLM_STRICT_MODE`` is enabled."""


def _is_strict_mode() -> bool:
    return os.environ.get("LLM_STRICT_MODE", "").strip().lower() in ("true", "1", "yes")


class LlmCallable(Protocol):
    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = ...,
        max_tokens: int | None = ...,
        tools: list[dict] | None = ...,
        tool_choice: str | None = ...,
    ) -> ChatResponse | str: ...


class LlmCallLogger:
    """Appends one JSONL event per LLM call to ``path`` (lazy open).

    ``frame_indexer`` is a zero-arg callable returning the current frame
    index (int).  The caller is responsible for maintaining that counter;
    this logger just reads it at call time.
    """

    def __init__(
        self,
        guid: str,
        path: str,
        frame_indexer: Callable[[], int],
    ) -> None:
        self.guid = guid
        self.path = path
        self._frame_indexer = frame_indexer
        self._seq = 0
        self._fp: Any = None
        self.trigger: str = ""

    def _ensure_open(self) -> None:
        if self._fp is None:
            self._fp = open(self.path, "a", encoding="utf-8")

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._ensure_open()
            self._fp.write(json.dumps(event, ensure_ascii=False))
            self._fp.write("\n")
            self._fp.flush()
        except Exception:
            log.exception("LlmCallLogger.emit failed")


def _truncate_content(content: str, limit: int) -> tuple[str, bool]:
    """Return ``(content_or_truncated, was_truncated)``."""
    if len(content) <= limit:
        return content, False
    dropped = len(content) - limit
    return f"{content[:limit]}[...truncated {dropped} chars]", True


def _truncate_messages(
    messages: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], bool]:
    truncated_any = False
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            new_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    new_blocks.append({"type": "text", "text": "[image omitted]"})
                    truncated_any = True
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if len(text) > limit:
                        text, did = _truncate_content(text, limit)
                        truncated_any = truncated_any or did
                    new_blocks.append({"type": "text", "text": text})
                else:
                    new_blocks.append(block)
            out.append({**msg, "content": new_blocks})
        elif isinstance(content, str) and len(content) > limit:
            content, did = _truncate_content(content, limit)
            truncated_any = truncated_any or did
            out.append({**msg, "content": content})
        else:
            out.append(msg)
    return out, truncated_any


def wrap_llm_call(
    llm_call: LlmCallable,
    logger: LlmCallLogger,
    kind: str,
    *,
    thinking: bool | None = None,
    max_tokens: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> Callable[..., ChatResponse | str]:
    """Wrap ``llm_call`` so every invocation is logged to ``logger.path``.

    The returned callable has the same signature as ``llm_call``:
    ``(messages, *, thinking=None, max_tokens=None, tools=None, tool_choice=None) -> ChatResponse | str``.

    ``thinking`` / ``max_tokens`` set here act as per-kind defaults applied
    to every call through this wrapper. They may be overridden per-call by
    passing the same kwargs to the returned callable. Precedence:
    per-call kwarg > wrap-time default > env var > server default.

    Truncation detection: if the LLM response has ``finish_reason == "length"``
    the event is marked ``truncated: true``.  In ``LLM_STRICT_MODE`` this
    raises :class:`LLMTruncationError`; otherwise a WARNING is logged.
    """

    def wrapped(
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = thinking,
        max_tokens: int | None = max_tokens,
        tools: list[dict] | None = tools,
        tool_choice: str | None = tool_choice,
    ) -> ChatResponse | str:
        seq = logger.next_seq()
        frame_index = logger._frame_indexer()
        trigger = logger.trigger or kind
        t0 = time.perf_counter()
        ok = True
        error: str | None = None
        raw = ""
        finish_reason: str | None = None
        tool_calls_data: list[dict] | None = None
        try:
            result = llm_call(
                messages,
                thinking=thinking,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )
            if isinstance(result, ChatResponse):
                raw = result.content
                finish_reason = result.finish_reason
                tool_calls_data = result.tool_calls
            else:
                # Backward compat: bare-string return from fakes / old callables.
                raw = result
                finish_reason = None
            return result
        except Exception as exc:
            ok = False
            error = repr(exc)
            raise
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            trunc_msgs, truncated = _truncate_messages(messages, MAX_CONTENT_CHARS)
            trunc_raw, raw_truncated = (
                _truncate_content(raw, MAX_CONTENT_CHARS) if raw else ("", False)
            )
            finish_truncated = finish_reason == "length"
            is_truncated = truncated or raw_truncated or finish_truncated
            if finish_truncated:
                log.warning(
                    "LLM response truncated (finish_reason='length') "
                    "kind=%s frame=%d seq=%d",
                    kind, frame_index, seq,
                )
                if _is_strict_mode():
                    truncation_exc = LLMTruncationError(
                        f"LLM response truncated: finish_reason='length' "
                        f"(kind={kind}, frame={frame_index}, seq={seq})"
                    )
                    event: dict[str, Any] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "guid": logger.guid,
                        "seq": seq,
                        "frame_index": frame_index,
                        "kind": kind,
                        "trigger": trigger,
                        "messages": trunc_msgs,
                        "response_raw": trunc_raw,
                        "tool_calls": tool_calls_data,
                        "latency_ms": latency_ms,
                        "ok": ok,
                        "error": error,
                        "truncated": is_truncated,
                        "finish_reason": finish_reason,
                    }
                    logger.emit(event)
                    raise truncation_exc
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "guid": logger.guid,
                "seq": seq,
                "frame_index": frame_index,
                "kind": kind,
                "trigger": trigger,
                "messages": trunc_msgs,
                "response_raw": trunc_raw,
                "tool_calls": tool_calls_data,
                "latency_ms": latency_ms,
                "ok": ok,
                "error": error,
                "truncated": is_truncated,
                "finish_reason": finish_reason,
            }
            logger.emit(event)

    return wrapped