"""Entity identity layer: re-identification + composition + roles.

Sits between raw tracking (``perception.registry``) and semantic grouping
(``grouping``).  Produces stable entity identities that survive rotation,
color change, and disappearance/reappearance — without any LLM or network.

Public API::

    from entity import EntityBuilder, EntityBuilderConfig

    builder = EntityBuilder()
    logical_registry, catalog = builder.update(registry, action_ids)
    # logical_registry: merged tracks (stable across death/birth events)
    # catalog: entities built from merged tracks, with roles assigned
"""

from __future__ import annotations

from .builder import EntityBuilder, EntityBuilderConfig
from .logical_registry import LogicalRegistry
from .reconciler import Reconciler, ReconcilerConfig


def __getattr__(name: str) -> object:
    if name in {
        "HeuristicRoleAssignerV1", "RoleAssigner", "RolePatch",
        "apply_patches", "assign_roles", "detect_agent",
        "detect_controllable", "detect_counter",
    }:
        from . import roles
        return getattr(roles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EntityBuilder",
    "EntityBuilderConfig",
    "HeuristicRoleAssignerV1",
    "LogicalRegistry",
    "Reconciler",
    "ReconcilerConfig",
    "RoleAssigner",
    "RolePatch",
    "apply_patches",
    "assign_roles",
    "detect_agent",
    "detect_controllable",
    "detect_counter",
]