"""LLM call logging — records every LLM call (messages + raw response + metadata)
to a dedicated JSONL file for offline analysis.

Usage::

    logger = LlmCallLogger(guid=recorder.guid, path=recorder.llm_log_path(),
                           frame_indexer=lambda: agent._frame_index)
    wrapped = wrap_llm_call(agent.llm_call, logger, kind="planner")
    # wrapped(messages) -> str  (same signature as llm_call)
    # Each call appends one JSONL line to logger.path.

Truncation: any single message ``content`` longer than ``MAX_CONTENT_CHARS``
is truncated and a ``[...truncated N chars]`` marker is appended. The top-level
event gets ``truncated: true`` so consumers can flag partial payloads.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)

MAX_CONTENT_CHARS = 20_000


class LlmCallable(Protocol):
    def __call__(self, messages: list[dict[str, str]]) -> str: ...


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
        # Mutable trigger label; caller sets this before each call site.
        # Defaults to the ``kind`` passed at wrap time.
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
            # Logging must never break the agent loop.
            log.exception("LlmCallLogger.emit failed")


def _truncate_content(content: str, limit: int) -> tuple[str, bool]:
    """Return ``(content_or_truncated, was_truncated)``."""
    if len(content) <= limit:
        return content, False
    dropped = len(content) - limit
    return f"{content[:limit]}[...truncated {dropped} chars]", True


def _truncate_messages(
    messages: list[dict[str, str]],
    limit: int,
) -> tuple[list[dict[str, str]], bool]:
    truncated_any = False
    out: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > limit:
            content, did = _truncate_content(content, limit)
            truncated_any = truncated_any or did
        out.append({**msg, "content": content})
    return out, truncated_any


def wrap_llm_call(
    llm_call: LlmCallable,
    logger: LlmCallLogger,
    kind: str,
) -> Callable[[list[dict[str, str]]], str]:
    """Wrap ``llm_call`` so every invocation is logged to ``logger.path``.

    The returned callable has the same signature as ``llm_call``:
    ``(messages: list[dict[str, str]]) -> str``.
    """

    def wrapped(messages: list[dict[str, str]]) -> str:
        seq = logger.next_seq()
        frame_index = logger._frame_indexer()
        trigger = logger.trigger or kind
        t0 = time.perf_counter()
        ok = True
        error: str | None = None
        raw = ""
        try:
            raw = llm_call(messages)
            return raw
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
            event: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "guid": logger.guid,
                "seq": seq,
                "frame_index": frame_index,
                "kind": kind,
                "trigger": trigger,
                "messages": trunc_msgs,
                "response_raw": trunc_raw,
                "latency_ms": latency_ms,
                "ok": ok,
                "error": error,
                "truncated": truncated or raw_truncated,
            }
            logger.emit(event)

    return wrapped