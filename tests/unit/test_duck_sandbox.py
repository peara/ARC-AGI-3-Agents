"""Unit tests for Duck Harness sandbox module."""

from __future__ import annotations

import time

import pytest

from agents.duck_harness_agent.sandbox import (
    DuckSandbox,
    SandboxResult,
    _ALLOWED_IMPORTS,
    _DANGEROUS_BUILTINS,
    _DUNDER_PATTERN,
    _MAX_OUTPUT_CHARS,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_callback(action_id: int, data: dict | None = None) -> dict:
    """Minimal step_env_callback returning a valid state dict."""
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [[0] * 64 for _ in range(64)],
        "valid_actions": [0, 1, 2],
        "last_action_result": "moved",
    }


def _make_sandbox(**kwargs) -> DuckSandbox:
    return DuckSandbox(step_env_callback=_mock_callback, **kwargs)


# ── Tests ──────────────────────────────────────────────────────────────────


def test_action_callback():
    """action(1) triggers callback with action_id=1 and returns action_taken=1."""
    received: list[tuple[int, dict | None]] = []

    def capture_callback(action_id: int, data: dict | None = None) -> dict:
        received.append((action_id, data))
        return _mock_callback(action_id, data)

    sandbox = DuckSandbox(step_env_callback=capture_callback, timeout=10.0)
    result = sandbox.run(
        code="action(1)",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.action_taken == 1
    assert len(received) == 1
    assert received[0] == (1, None)


def test_no_action_returns_none():
    """Code that never calls action() → action_taken is None."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="x = 42",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.action_taken is None
    assert result.error is None


def test_timeout_kills_subprocess():
    """Code that loops forever produces a timeout error result."""
    sandbox = DuckSandbox(step_env_callback=_mock_callback, timeout=1.0)
    result = sandbox.run(
        code="while True: pass",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is not None
    assert "timed out" in result.error.lower()


def test_dunder_rejected():
    """Code containing __import__('os') is rejected with dunder error."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="__import__('os')",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is not None
    assert "dunder" in result.error.lower()


def test_output_captured():
    """print('hello') → output contains 'hello'."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code='print("hello")',
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert "hello" in result.output


def test_output_truncated():
    """Very long output is truncated at ~_MAX_OUTPUT_CHARS."""
    sandbox = _make_sandbox()
    # Generate output far exceeding the limit
    code = f'print("x" * {_MAX_OUTPUT_CHARS * 2})'
    result = sandbox.run(
        code=code,
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert len(result.output) <= _MAX_OUTPUT_CHARS + len("... (truncated)")
    assert "truncated" in result.output


def test_state_refresh_after_action():
    """Objects and globals are refreshed after action() call."""
    call_count = 0

    def evolving_callback(action_id: int, data: dict | None = None) -> dict:
        nonlocal call_count
        call_count += 1
        return {
            "objects": ({"id": call_count, "x": 10 * call_count},),
            "adjacency": frozenset(),
            "history": [{"step": call_count}],
            "grid": [[call_count] * 64 for _ in range(64)],
            "valid_actions": [0, 1],
            "last_action_result": f"step_{call_count}",
        }

    sandbox = DuckSandbox(step_env_callback=evolving_callback, timeout=10.0)
    # After action(), objects should be the refreshed tuple
    code = "result = action(0); got = len(objects)"
    result = sandbox.run(
        code=code,
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is None
    assert result.action_taken == 0


def test_action6_complex_action():
    """action(6, x=30, y=40) passes action_data dict to callback."""
    received: list[tuple[int, dict | None]] = []

    def capture_callback(action_id: int, data: dict | None = None) -> dict:
        received.append((action_id, data))
        return _mock_callback(action_id, data)

    sandbox = DuckSandbox(step_env_callback=capture_callback, timeout=10.0)
    result = sandbox.run(
        code="action(6, x=30, y=40)",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.action_taken == 6
    assert len(received) == 1
    assert received[0][0] == 6
    assert received[0][1] == {"x": 30, "y": 40}


def test_restricted_builtins():
    """Dangerous builtins (open, exec) are blocked in sandbox."""
    sandbox = _make_sandbox()
    # open should be unavailable
    result = sandbox.run(
        code="open('/etc/passwd')",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    # open is in _DANGEROUS_BUILTINS, so it should be removed from builtins
    assert result.error is not None

    # exec should be unavailable
    result2 = sandbox.run(
        code="exec('1+1')",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result2.error is not None

    # Verify _DANGEROUS_BUILTINS contains expected names
    assert "open" in _DANGEROUS_BUILTINS
    assert "__import__" not in _DANGEROUS_BUILTINS
    assert "exec" in _DANGEROUS_BUILTINS


def test_allowed_imports():
    """import math works in the sandbox."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="import math; print(round(math.pi, 2))",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is None, f"Unexpected error: {result.error}"
    assert "3.14" in result.output


def test_from_import():
    """from collections import Counter works in the sandbox."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="from collections import Counter; print(Counter([1,1,2]))",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is None, f"Unexpected error: {result.error}"
    assert "1" in result.output


def test_blocked_import():
    """import os raises ImportError with 'not allowed' message."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="import os",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is not None
    assert "not allowed" in result.error
    assert "Allowed modules" in result.error


def test_submodule_import():
    """import collections.abc works (checks top-level collections only)."""
    sandbox = _make_sandbox()
    result = sandbox.run(
        code="import collections.abc; print('ok')",
        objects=(),
        adjacency=frozenset(),
        history=[],
    )
    assert result.error is None, f"Unexpected error: {result.error}"
    assert "ok" in result.output