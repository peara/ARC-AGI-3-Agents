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
    has_overlaps: bool
    overlaps_entity_ids: tuple[int, int] | None


def parse_guard_clauses(guard: dict[str, Any]) -> list[GuardClause]:
    """Parse a DSL guard dict into a list of normalised GuardClause dicts.

    Handles formats:
    - ``{"action": N}`` → single action clause
    - ``{"all": [...]}`` → conjunction of clauses (action and/or pos and/or overlaps)
    - ``{"overlaps": {"entity_a": N, "entity_b": M}}`` → overlaps clause
    - ``{"all": []}`` → empty list
    """
    # Top-level overlaps guard (not inside "all")
    if "overlaps" in guard and "all" not in guard:
        ov = guard["overlaps"]
        return [
            GuardClause(
                has_action=False,
                action=None,
                has_pos=False,
                entity_id=None,
                pos=None,
                has_overlaps=True,
                overlaps_entity_ids=(ov["entity_a"], ov["entity_b"]),
            )
        ]

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
                has_overlaps=False,
                overlaps_entity_ids=None,
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
                    has_overlaps=False,
                    overlaps_entity_ids=None,
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
                    has_overlaps=False,
                    overlaps_entity_ids=None,
                )
            )
        if "overlaps" in clause:
            ov = clause["overlaps"]
            result.append(
                GuardClause(
                    has_action=False,
                    action=None,
                    has_pos=False,
                    entity_id=None,
                    pos=None,
                    has_overlaps=True,
                    overlaps_entity_ids=(ov["entity_a"], ov["entity_b"]),
                )
            )
    return result


def evaluate_guard(
    guard: dict[str, object],
    state: SceneState,
    action: int,
    *,
    entity_cells: dict[int, frozenset[tuple[int, int]]] | None = None,
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
        if clause["has_overlaps"]:
            if entity_cells is None:
                msg = "overlaps guard requires entity_cells"
                raise ValueError(msg)
            ids = clause["overlaps_entity_ids"]
            if ids is None:
                return False
            cells_a = entity_cells.get(ids[0])
            cells_b = entity_cells.get(ids[1])
            if cells_a is None or cells_b is None:
                return False
            if not (cells_a & cells_b):
                return False
    return True