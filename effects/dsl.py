"""Structured DSL serialization for Rule."""

from __future__ import annotations

from typing import Any

from .guard_parse import parse_guard_clauses
from .rules import Effect, Rule

# TypedDict-style aliases for clarity (plain dicts at runtime).
DslRule = dict[str, Any]


def rule_to_dsl(rule: Rule) -> DslRule:
    """Convert a Rule to a structured DSL dict."""
    if rule.kind == "delta":
        effect_dict: dict[str, Any] = {}
        for e in rule.effects:
            if e.dim == "size" and e.op == "delta":
                effect_dict = {"dim": "size", "of": e.of, "delta": e.value}
        return {
            "kind": "delta",
            "guard": rule.guard_spec,
            "effect": effect_dict,
            "support": rule.support,
        }

    # Terminal rule
    effect_dict = {}
    for e in rule.effects:
        if e.dim == "terminal":
            effect_dict = {"terminal": e.value}
    return {
        "kind": "terminal",
        "guard": rule.guard_spec,
        "effect": effect_dict,
        "support": rule.support,
    }


def dsl_to_rule(dsl: DslRule) -> Rule:
    """Reconstruct a Rule from a DSL dict."""
    kind = dsl.get("kind")
    if kind not in ("delta", "terminal"):
        raise ValueError(f"unknown kind: {kind!r}")

    guard: dict[str, Any] = dsl["guard"]
    support: int = dsl["support"]

    if kind == "delta":
        eff = dsl["effect"]
        entity_id: int = eff["of"]
        delta_size: int = eff["delta"]
        if delta_size == 0:
            raise ValueError("delta_size must not be 0")
        return Rule(
            guard_spec=guard,
            effects=(Effect("size", entity_id, "delta", delta_size),),
            support=support,
        )

    # terminal
    eff = dsl["effect"]
    terminal = eff["terminal"]
    entity_id = 0
    for gc in parse_guard_clauses(guard):
        if gc["has_pos"]:
            entity_id = gc.get("entity_id") or 0
    return Rule(
        guard_spec=guard,
        effects=(Effect("terminal", entity_id, "set", terminal),),
        support=support,
    )