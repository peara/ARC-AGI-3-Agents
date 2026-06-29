"""Raw track-level helpers for role detection.

Entity-level role classes and functions have moved to ``entity.roles``.
This module re-exports the public API from ``entity.roles`` for backward
compatibility and also re-exports the low-level helpers from
``perception._roles_helpers`` so that existing ``from perception.roles
import _controllable_tracks`` (etc.) calls continue to work.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Re-exports from entity.roles (backward compatibility)
# ---------------------------------------------------------------------------
from entity.roles import (  # noqa: F401  re-export
    HeuristicRoleAssignerV1,
    RoleAssigner,
    RolePatch,
    apply_patches,
    assign_roles,
    detect_agent,
    detect_controllable,
    detect_counter,
)

# ---------------------------------------------------------------------------
# Raw track-level helpers (delegated to _roles_helpers to avoid circular import)
# ---------------------------------------------------------------------------
from ._roles_helpers import (  # noqa: F401
    _RESET_ACTION,
    _controllable_tracks,
    _is_counter_track,
    _is_structural,
    _track_action_displacements,
)

__all__ = [
    "HeuristicRoleAssignerV1",
    "RoleAssigner",
    "RolePatch",
    "apply_patches",
    "assign_roles",
    "detect_agent",
    "detect_controllable",
    "detect_counter",
]