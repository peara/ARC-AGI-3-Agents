"""Goal-directed probe: compile DSL predicates into BFS goals and execute plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from effects import EffectContext, SceneState
from perception.session import SceneSnapshot

from .adapters import snapshot_from_scene
from .heuristics import within
from .query import UnknownAction
from .search import PlanSpec, plan_bfs


@dataclass(frozen=True)
class ProbeGoal:
    """Declarative goal specification for a single probe invocation."""

    target: dict[str, object]  # DSL predicate dict
    entities: tuple[int, ...] | None = None  # auto-derived if None
    dims: tuple[str, ...] | None = None  # auto-derived if None
    max_steps: int = 20  # BFS node limit
    reason: str = ""  # logging only


def compile_goal(predicate: dict[str, Any]) -> Callable[[SceneState], bool]:
    """Compile a DSL predicate dict into a callable goal over SceneState.

    Note: ``resolve_predicate``, ``compile_goal``, and ``derive_spec_from_predicate``
    all operate on plain DSL dicts (the value stored in ``ProbeGoal.target``), not on
    the ``ProbeGoal`` dataclass field itself.  Their ``predicate`` parameter is a
    dict, not a ProbeGoal attribute.
    """

    # Conjunction of sub-predicates
    if "all" in predicate:
        sub_predicates = predicate["all"]
        if not sub_predicates:
            return lambda _s: True
        compiled = [compile_goal(c) for c in sub_predicates]
        return lambda s: all(g(s) for g in compiled)

    # Action guard — ignored for goal predicates
    if "action" in predicate:
        return lambda _s: True

    # Position near predicate
    if "dim" in predicate and predicate["dim"] == "pos" and "near" in predicate:
        eid: int = predicate["of"]
        near_val = predicate["near"]
        if isinstance(near_val, dict):
            raise ValueError(
                "relative near references must be resolved first via resolve_predicate"
            )
        radius: int = predicate.get("radius", 0)
        target: tuple[int, int] = tuple(near_val)

        def _near_goal(s: SceneState) -> bool:
            return within(s.pos(eid), target, radius)

        return _near_goal

    # Dim equality predicate
    if "dim" in predicate and "of" in predicate and "eq" in predicate:
        dim_name: str = predicate["dim"]
        entity_id: int = predicate["of"]
        value = predicate["eq"]
        if isinstance(value, list):
            value = tuple(value)

        def _eq_goal(s: SceneState) -> bool:
            got = s.get(entity_id, dim_name)
            return bool(got == value)

        return _eq_goal

    raise ValueError(f"unknown predicate form: {predicate!r}")


def resolve_predicate(
    predicate: dict[str, Any], scene: SceneSnapshot
) -> dict[str, object]:
    """Resolve relative entity references in a predicate dict to concrete values.

    Deep-walks the predicate, replacing ``{"near": {"of": ref_eid, "radius": N}}``
    with ``{"near": [r, c], "radius": N}`` using the scene's entity positions.
    """

    # Conjunction — recurse into each child
    if "all" in predicate:
        resolved_children = [
            resolve_predicate(c, scene) for c in predicate["all"]
        ]
        return {"all": resolved_children}

    # Relative near reference
    if "near" in predicate and isinstance(predicate["near"], dict):
        near_dict = predicate["near"]
        ref_eid: int = near_dict["of"]
        radius: int = near_dict.get("radius", 0)
        pos = scene.entity_pos(ref_eid)
        if pos is None:
            raise ValueError(f"entity {ref_eid} has no position")
        return {
            k: v for k, v in predicate.items() if k != "near"
        } | {"near": list(pos), "radius": radius}

    # No relative references — return as-is (with list→tuple for eq values)
    return dict(predicate)


def derive_spec_from_predicate(
    predicate: dict[str, Any],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Walk a predicate dict and collect all entity IDs and dim names."""

    entity_ids: set[int] = set()
    dim_names: set[str] = set()
    _walk_predicate(predicate, entity_ids, dim_names)
    return (tuple(sorted(entity_ids)), tuple(sorted(dim_names)))


def _walk_predicate(
    pred: dict[str, Any],
    entity_ids: set[int],
    dim_names: set[str],
) -> None:
    """Recursive helper for derive_spec_from_predicate."""

    if "all" in pred:
        for child in pred["all"]:
            _walk_predicate(child, entity_ids, dim_names)
        return

    if "of" in pred:
        eid = pred["of"]
        if isinstance(eid, int):
            entity_ids.add(eid)
    if "dim" in pred:
        dim = pred["dim"]
        if isinstance(dim, str):
            dim_names.add(dim)
    # Also check nested "near" dicts
    if "near" in pred and isinstance(pred["near"], dict):
        ref_eid = pred["near"].get("of")
        if isinstance(ref_eid, int):
            entity_ids.add(ref_eid)


def execute_probe(
    goal: ProbeGoal,
    scene: SceneSnapshot,
    ctx: EffectContext,
    actions: list[int],
) -> tuple[list[int] | None, list[UnknownAction]]:
    """Resolve predicate, build spec, and run BFS to find an action sequence."""

    resolved = resolve_predicate(goal.target, scene)

    if goal.entities is not None and goal.dims is not None:
        entities = goal.entities
        dims = goal.dims
    else:
        derived_entities, derived_dims = derive_spec_from_predicate(resolved)
        entities = goal.entities if goal.entities is not None else derived_entities
        dims = goal.dims if goal.dims is not None else derived_dims

    compiled_goal = compile_goal(resolved)
    spec = PlanSpec(
        entities=list(entities),
        dims=dims,
        goal=compiled_goal,
        include_terminal=False,
    )

    start = snapshot_from_scene(scene, spec)
    if start is None:
        return (None, [])

    return plan_bfs(start, compiled_goal, actions, ctx, max_nodes=goal.max_steps)