"""Shared guard-clause parsing for DSL deserialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from .state import SceneState


class GuardClause(TypedDict):
    """A normalised guard clause extracted from a DSL guard dict."""

    has_action: bool
    action: int | None
    has_pos: bool
    entity_id: int | None
    pos: tuple[int, int] | None


def parse_guard_clauses(guard: dict[str, Any]) -> list[GuardClause]:
    """Parse a DSL guard dict into a list of normalised GuardClause dicts.

    Handles two formats:
    - ``{"action": N}`` → single action clause
    - ``{"all": [...]}`` → conjunction of clauses (action and/or pos)
    - ``{"all": []}`` → empty list
    """
    if "all" not in guard:
        # Simple single-action guard: {"action": N}
        action_val: int | None = guard.get("action")
        return [
            GuardClause(
                has_action=action_val is not None,
                action=action_val,
                has_pos=False,
                entity_id=None,
                pos=None,
            )
        ]

    clauses = guard["all"]
    result: list[GuardClause] = []
    for clause in clauses:
        if "action" in clause:
            result.append(
                GuardClause(
                    has_action=True,
                    action=clause["action"],
                    has_pos=False,
                    entity_id=None,
                    pos=None,
                )
            )
        if "dim" in clause and clause["dim"] == "pos":
            eq = clause["eq"]
            result.append(
                GuardClause(
                    has_action=False,
                    action=None,
                    has_pos=True,
                    entity_id=clause.get("of"),
                    pos=(eq[0], eq[1]),
                )
            )
    return result


def evaluate_guard(
    guard: dict[str, object], state: SceneState, action: int
) -> bool:
    """Evaluate a guard dict against the current state and action.

    Placeholder implementation — full DSL evaluation coming in a later slice.
    """
    clauses = parse_guard_clauses(guard)
    if not clauses:
        return True
    for clause in clauses:
        if clause["has_action"] and clause["action"] != action:
            return False
        if clause["has_pos"]:
            eid = clause.get("entity_id")
            pos = clause.get("pos")
            if eid is not None and pos is not None:
                if state.pos(eid) != pos:
                    return False
    return True