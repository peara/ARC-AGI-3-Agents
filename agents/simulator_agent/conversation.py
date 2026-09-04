"""Pure message-list trimming pipeline for LLM chat conversations.

Extracted verbatim from ``SimulatorFirstAgent`` (``agent.py``) so the pipeline
is unit-testable without constructing an agent. The agent class keeps thin
delegating wrappers (``SimulatorFirstAgent._estimate_tokens`` etc.) because
tests pin those names, and one budget regression test monkeypatches
``agent._trim_messages_for_context`` on an instance built via ``__new__()`` —
so the agent's ``run()`` must keep calling it through ``self.*``.

Two entry points take the only piece of agent state the pipeline ever used
explicitly: ``trim_messages_for_context`` and ``persistent_history_messages``
take ``budget_tokens``.

Invariants (pinned by ``tests/unit/simulator_agent/test_agent.py``):
- Mutators (``drop_oldest_history_block``, ``strip_old_images``,
  ``trim_old_tool_results``, ``trim_old_non_tool_messages``) never delete
  messages — the API pairing requirement (every ``role:tool`` message keeps a
  preceding ``tool_calls`` entry) is preserved.
- ``trim_old_non_tool_messages`` is idempotent.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "IMAGE_TOKEN_COST",
    "drop_oldest_history_block",
    "drop_until_first_user_message",
    "estimate_tokens",
    "keep_recent_assistant_turns",
    "persistent_history_messages",
    "strip_notes_messages",
    "strip_old_images",
    "trim_messages_for_context",
    "trim_old_non_tool_messages",
    "trim_old_tool_results",
]

IMAGE_TOKEN_COST = 1024


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate token count — text via char heuristic, images via fixed cost."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        total += IMAGE_TOKEN_COST
                    elif part.get("type") == "text":
                        total += len(part.get("text", "")) // 3
        elif isinstance(content, str):
            total += len(content) // 3
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += len(json.dumps(tool_calls, default=str)) // 3
    return max(1, total)


def drop_oldest_history_block(
    history: list[dict[str, Any]], *, preserve_recent: int
) -> bool:
    removable = len(history) - preserve_recent
    if removable <= 0:
        return False
    history.pop(0)
    while (
        history
        and history[0].get("role") == "tool"
        and len(history) > preserve_recent
    ):
        history.pop(0)
    while (
        history
        and history[0].get("role") != "user"
        and len(history) > preserve_recent
    ):
        history.pop(0)
    return True


def keep_recent_assistant_turns(
    messages: list[dict[str, Any]], *, max_turns: int
) -> list[dict[str, Any]]:
    if max_turns <= 0 or not messages:
        return []
    kept_reversed: list[dict[str, Any]] = []
    assistant_turns = 0
    for message in reversed(messages):
        kept_reversed.append(message)
        if message.get("role") == "assistant":
            assistant_turns += 1
            if assistant_turns >= max_turns:
                break
    kept = list(reversed(kept_reversed))
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    return kept


def drop_until_first_user_message(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trimmed = list(history)
    while trimmed and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    return trimmed


def trim_messages_for_context(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int,
    preserve_recent: int = 1,
    extra_safety_tokens: int = 0,
) -> list[dict[str, Any]]:
    if not messages:
        return []
    system_message = messages[0]
    history = list(messages[1:])
    budget = max(1, budget_tokens - extra_safety_tokens)
    while history and estimate_tokens([system_message, *history]) > budget:
        if not drop_oldest_history_block(history, preserve_recent=preserve_recent):
            break
    history = drop_until_first_user_message(history)
    return [system_message, *history]


def persistent_history_messages(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int,
) -> list[dict[str, Any]]:
    stripped = strip_notes_messages(messages)
    trimmed = trim_messages_for_context(stripped, budget_tokens=budget_tokens)
    if not trimmed:
        return []
    trimmed_history = trimmed[1:]
    history = keep_recent_assistant_turns(
        trimmed_history,
        max_turns=30,
    )
    if (
        history
        and history[0].get("role") != "user"
        and len(trimmed_history) > len(history)
    ):
        prev_idx = len(trimmed_history) - len(history) - 1
        if prev_idx >= 0 and trimmed_history[prev_idx].get("role") == "user":
            history = [trimmed_history[prev_idx], *history]
    history = drop_until_first_user_message(history)
    strip_old_images(history, keep_last_n_user=2)
    return history


def strip_notes_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list with all [Current notes] user messages removed.

    Notes messages are injected fresh each frame from self._world_model.
    Carrying stale notes messages in persistent history causes duplicates
    and bloat. Strip them before the trim pipeline so history never keeps
    old notes.
    """
    return [
        msg
        for msg in messages
        if not (
            msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith("[Current notes]")
        )
    ]


def strip_old_images(
    history: list[dict[str, Any]], *, keep_last_n_user: int
) -> None:
    """Remove image_url blocks from all but the last N user messages.

    Vision models tokenize images as hundreds/thousands of tokens
    regardless of file size. Carrying old frame images in persistent
    history quickly exhausts the context window. Only the most recent
    frames need images — older ones are text-only.
    """
    user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
    cutoff_indices = (
        set(user_indices[-keep_last_n_user:]) if keep_last_n_user > 0 else set()
    )
    for i in user_indices:
        if i in cutoff_indices:
            continue
        content = history[i].get("content")
        if not isinstance(content, list):
            continue
        history[i]["content"] = [
            p
            for p in content
            if not (isinstance(p, dict) and p.get("type") == "image_url")
        ]


def trim_old_tool_results(
    messages: list[dict[str, Any]], keep_last_n: int = 3
) -> None:
    """Replace old tool result content with placeholders, keeping only last N.

    Mutates messages in-place. Never deletes messages (API pairing requirement).

    Exemptions:
    - update_notes results: always kept (tiny, ~50 chars)
    - action() result: the last tool result that triggered an action is pinned
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_last_n:
        return

    exempt_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "update_notes":
                    exempt_ids.add(tc["id"])

    if tool_indices:
        last_tool_id = messages[tool_indices[-1]].get("tool_call_id")
        if last_tool_id:
            exempt_ids.add(last_tool_id)

    trimmable = [
        i for i in tool_indices if messages[i].get("tool_call_id") not in exempt_ids
    ]
    to_trim = trimmable[:-keep_last_n] if len(trimmable) > keep_last_n else []

    for i in to_trim:
        msg = messages[i]
        content = msg.get("content", "")
        if isinstance(content, str):
            original_chars = len(content)
        elif isinstance(content, list):
            original_chars = sum(
                len(b.get("text", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            original_chars = len(str(content))
        msg["content"] = f"[Old output ({original_chars} chars, trimmed)]"


def trim_old_non_tool_messages(messages: list[dict[str, Any]]) -> None:
    """Trim old user prompts, nudge messages, and assistant code.

    Idempotent — safe to run on already-trimmed messages.
    Mutates in-place. Never deletes messages (API pairing requirement).
    """
    nudge_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and (
            "discovered something new" in m.get("content", "")
            or "Please use the python tool" in m.get("content", "")
        )
    ]
    for i in nudge_indices[:-1]:
        messages[i]["content"] = "[nudge]"

    frame_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "user" and isinstance(m.get("content"), list)
    ]
    for i in frame_indices[:-1]:
        messages[i] = {**messages[i], "content": "[frame]"}

    assistant_tc_indices = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    for i in assistant_tc_indices[:-4]:
        for tc in messages[i]["tool_calls"]:
            name = tc["function"]["name"]
            if name == "python":
                tc["function"]["arguments"] = '{"code": "# previous code omitted"}'
            elif name == "update_notes":
                tc["function"]["arguments"] = (
                    '{"notes": "[trimmed]", "plan": "[trimmed]"}'
                )