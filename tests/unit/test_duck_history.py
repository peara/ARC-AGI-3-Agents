"""Unit tests for Duck Harness history trimming and context management."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from agents.duck_harness_agent.agent import DuckHarnessAgent

if TYPE_CHECKING:
    pass


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_agent(context_budget_tokens: int = 100_000, max_history_turns: int = 30) -> DuckHarnessAgent:
    """Return a minimal DuckHarnessAgent instance without running __init__."""
    agent = object.__new__(DuckHarnessAgent)
    agent._context_budget_tokens = context_budget_tokens
    agent._config = SimpleNamespace(max_history_turns=max_history_turns)
    return agent


def _assistant_turn(index: int) -> list[dict[str, Any]]:
    """Return one assistant/tool result turn pair."""
    return [
        {"role": "assistant", "content": f"assistant {index}"},
        {"role": "tool", "tool_call_id": f"tc{index}", "content": f"result {index}"},
    ]


def _is_context_overflow(error_text: str) -> bool:
    """Mirror the context-overflow detection in DuckHarnessAgent.choose_action."""
    lowered = error_text.lower()
    return "context_length" in lowered or "context length" in lowered or "too long" in lowered


# ── Tests ────────────────────────────────────────────────────────────────────


def test_estimate_tokens_basic():
    """Token estimate is at least 1 and grows with message list size."""
    empty = DuckHarnessAgent._estimate_tokens([])
    small = DuckHarnessAgent._estimate_tokens([{"role": "user", "content": "hi"}])
    large = DuckHarnessAgent._estimate_tokens([{"role": "user", "content": "x" * 1000}])
    assert empty >= 1
    assert large > small


def test_estimate_tokens_empty():
    """Empty message list still returns a positive token count."""
    assert DuckHarnessAgent._estimate_tokens([]) >= 1


def test_drop_oldest_history_block():
    """Dropping the oldest block removes the first user/assistant/tool group."""
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "tool", "tool_call_id": "tc1", "content": "old tool"},
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]
    changed = DuckHarnessAgent._drop_oldest_history_block(history, preserve_recent=1)
    assert changed is True
    assert history == [
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new assistant"},
    ]


def test_drop_oldest_preserve_recent():
    """When history length is within the preserved window, no drop occurs."""
    history = [{"role": "user", "content": "only"}]
    original = list(history)
    changed = DuckHarnessAgent._drop_oldest_history_block(history, preserve_recent=2)
    assert changed is False
    assert history == original


def test_keep_recent_assistant_turns():
    """Keep only the most recent N assistant turns (assistant + tool pairs)."""
    messages: list[dict[str, Any]] = []
    for i in range(5):
        messages.extend(_assistant_turn(i))
    kept = DuckHarnessAgent._keep_recent_assistant_turns(messages, max_turns=3)
    assistant_roles = [m for m in kept if m.get("role") == "assistant"]
    assert len(assistant_roles) == 3
    assert kept[0].get("role") == "assistant"
    assert "assistant 2" in kept[0]["content"]
    assert any("assistant 4" in m["content"] for m in assistant_roles)


def test_keep_recent_zero():
    """max_turns=0 yields an empty message list."""
    messages = [{"role": "assistant", "content": "a"}]
    assert DuckHarnessAgent._keep_recent_assistant_turns(messages, max_turns=0) == []


def test_drop_until_first_user():
    """Leading assistant/tool messages are stripped until the first user message."""
    messages = [
        {"role": "assistant", "content": "lead"},
        {"role": "tool", "tool_call_id": "tc", "content": "lead tool"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "last"},
    ]
    trimmed = DuckHarnessAgent._drop_until_first_user_message(messages)
    assert trimmed == [
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "last"},
    ]


def test_drop_until_first_user_already_user():
    """A list already starting with a user message is unchanged."""
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    trimmed = DuckHarnessAgent._drop_until_first_user_message(messages)
    assert trimmed == messages


def test_trim_messages_for_context():
    """A large history is trimmed to fit a small context budget."""
    agent = _make_agent(context_budget_tokens=200)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system prompt"},
    ]
    for i in range(10):
        messages.append({"role": "user", "content": f"user message {i} " + "x" * 500})
    trimmed = agent._trim_messages_for_context(messages)
    assert len(trimmed) < len(messages)
    assert trimmed[0]["role"] == "system"
    assert all(m.get("role") != "system" for m in trimmed[1:])


def test_trim_preserves_system():
    """The system message is always retained at the front after trimming."""
    agent = _make_agent(context_budget_tokens=200)
    messages = [
        {"role": "system", "content": "critical system prompt"},
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 1000},
    ]
    trimmed = agent._trim_messages_for_context(messages)
    assert trimmed[0] == {"role": "system", "content": "critical system prompt"}


def test_persistent_history_pipeline():
    """Full pipeline trims context, keeps recent turns, and starts with a user message."""
    agent = _make_agent(context_budget_tokens=50_000, max_history_turns=3)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
    ]
    for i in range(6):
        messages.append({"role": "user", "content": f"user {i}"})
        messages.extend(_assistant_turn(i))
    history = agent._persistent_history_messages(messages)
    assert history
    assert history[0]["role"] == "user"
    assistant_count = sum(1 for m in history if m.get("role") == "assistant")
    assert assistant_count <= 3


def test_persistent_history_empty():
    """Empty input to the persistent-history pipeline returns an empty list."""
    agent = _make_agent()
    assert agent._persistent_history_messages([]) == []


def test_context_overflow_detection():
    """Context-length / too-long error strings are detected correctly."""
    assert _is_context_overflow("This model's maximum context_length is 4097 tokens")
    assert _is_context_overflow("context length exceeded")
    assert _is_context_overflow("input is too long")
    assert not _is_context_overflow("rate limit exceeded")
    assert not _is_context_overflow("connection error")


def test_drop_oldest_history_block_with_leading_tool():
    """Leading tool messages after the popped block are also removed."""
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "tool", "tool_call_id": "tc1", "content": "old tool"},
        {"role": "tool", "tool_call_id": "tc2", "content": "another old tool"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new assistant"},
    ]
    changed = DuckHarnessAgent._drop_oldest_history_block(history, preserve_recent=1)
    assert changed is True
    assert history == [
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new assistant"},
    ]
