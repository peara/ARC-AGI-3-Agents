"""Structured DSL serialization for CounterRule and TerminalRule."""

from __future__ import annotations

from typing import Any

from .guard_parse import parse_guard_clauses
from .rules import CounterRule, TerminalRule

# TypedDict-style aliases for clarity (plain dicts at runtime).
DslRule = dict[str, Any]


def rule_to_dsl(rule: CounterRule | TerminalRule) -> DslRule:
    """Convert a CounterRule or TerminalRule to a structured DSL dict."""

    if isinstance(rule, CounterRule):
        if rule.delta_size == 0:
            raise ValueError("delta_size must not be 0")
        if rule.guard_pos is not None and rule.controllable_id is None:
            raise ValueError(
                "guard_pos requires controllable_id to be set"
            )

        effect: dict[str, Any] = {
            "dim": "size",
            "of": rule.entity_id,
            "delta": rule.delta_size,
        }

        guard: dict[str, Any]
        if rule.guard_pos is not None and rule.controllable_id is not None:
            r, c = rule.guard_pos
            guard = {
                "all": [
                    {"action": rule.action},
                    {"dim": "pos", "of": rule.controllable_id, "eq": [r, c]},
                ]
            }
        else:
            guard = {"action": rule.action}

        return {
            "kind": "delta",
            "entity_id": rule.entity_id,
            "action": rule.action,
            "effect": effect,
            "guard": guard,
            "support": rule.support,
        }

    # TerminalRule
    pos, act = rule.guard_key
    r, c = pos
    guard = {
        "all": [
            {"action": act},
            {"dim": "pos", "of": rule.entity_id, "eq": [r, c]},
        ]
    }

    return {
        "kind": "terminal",
        "entity_id": rule.entity_id,
        "guard": guard,
        "effect": {"terminal": rule.terminal},
        "support": rule.support,
    }


def dsl_to_rule(dsl: DslRule) -> CounterRule | TerminalRule:
    """Reconstruct a CounterRule or TerminalRule from a DSL dict."""

    kind = dsl.get("kind")
    if kind not in ("delta", "terminal"):
        raise ValueError(f"unknown kind: {kind!r}")

    if kind == "delta":
        entity_id: int = dsl["entity_id"]
        action: int = dsl["action"]
        delta_size: int = dsl["effect"]["delta"]
        support: int = dsl["support"]

        if delta_size == 0:
            raise ValueError("delta_size must not be 0")

        guard_pos: tuple[int, int] | None = None
        controllable_id: int | None = None

        for gc in parse_guard_clauses(dsl["guard"]):
            if gc["has_pos"]:
                guard_pos = gc["pos"]
                controllable_id = gc["entity_id"]
        if guard_pos is not None and controllable_id is None:
            raise ValueError(
                "guard_pos requires controllable_id to be set"
            )

        return CounterRule(
            entity_id=entity_id,
            action=action,
            delta_size=delta_size,
            support=support,
            controllable_id=controllable_id,
            guard_pos=guard_pos,
        )

    # terminal
    entity_id = dsl["entity_id"]
    terminal = dsl["effect"]["terminal"]
    support = dsl["support"]

    t_guard_pos: tuple[int, int] | None = None
    action_id: int | None = None

    for gc in parse_guard_clauses(dsl["guard"]):
        if gc["has_action"]:
            action_id = gc["action"]
        if gc["has_pos"]:
            t_guard_pos = gc["pos"]

    if t_guard_pos is None or action_id is None:
        raise ValueError("terminal rule guard missing pos or action clause")

    return TerminalRule(
        entity_id=entity_id,
        guard_key=(t_guard_pos, action_id),
        terminal=terminal,
        support=support,
    )