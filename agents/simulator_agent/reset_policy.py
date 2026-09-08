"""Single owner of all RESET (action 0) semantics for the simulator agent.

Canonical mapping table (this module is the single owner — cite it, do not
restate the constants elsewhere):

- ``grids[j] == frames[t_level + j]`` board-wise, where ``t_level`` is the
  frames-index of the level start: ``t_level = 0`` for level 1 under the
  frames[0] fill (the synthetic env.reset() frame duplicates the recorded
  RESET result), and the level's first frame index for levels 2+ (which
  start via level-completion with no RESET).
- ``history[i] ↔ transition i ↔ actions[i]``: transition i maps
  ``grids[i] --actions[i]--> grids[i+1]``. State-keyed pairing: the action
  that PRODUCED frame ``i`` is ``actions[i-1]`` (frame 0 is produced by
  RESET itself, not by any corpus action).

Why the virtual pair ``([B0, B0], [0])`` exists: offline loading
(``sandbox.py:190-207``) replays ALL harness frames including the synthetic
reset at index 0, so both pinned recordings show ``grids[0] == grids[1]``
and ``actions[0] == 0`` — the [0]/[1] board duplicate IS the RESET
transition. Naive seeding of ``actions=[0]`` without the duplicate grid is
THE corruption trap: transition 0 would become ``(B0, RESET) -> B1`` and
a1's result would be attributed to RESET. The pair shape prevents it.

Policy rationale (why the three includes_* constants differ):

- ``check`` / ``diagnose`` INCLUDE RESET transitions: the RESET transition
  is trained data like any other — the corpus legitimately contains it and
  scoring over it is meaningful.
- ``bfs`` EXCLUDES RESET edges: a RESET edge cycles back to the level-start
  board, which BFS has already visited — expanding it is pure waste.

Consumers (tasks 4-7 of the RESET-handler alignment wave) must call these
APIs instead of re-deriving RESET semantics inline. This is the ONLY
simulator_agent module allowed to import ``RESET_ACTION`` from
``perception.session`` — no 4th constant copy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from perception.session import RESET_ACTION

logger = logging.getLogger("simulator.reset_policy")

__all__ = [
    "RESET_ACTION",
    "ResetSeedTracker",
    "bfs_includes_reset",
    "check_includes_reset",
    "diagnose_includes_reset",
    "is_reset",
    "producing_action_caption",
    "virtual_reset_pair",
]


def is_reset(action_id: int | str) -> bool:
    """True iff *action_id* is the RESET action.

    Accepts both wire forms: the integer id (``0``) and the harness string
    form (``"RESET"``, case-sensitive — only the exact uppercase spelling).
    Mirrors the offline normalization at ``sandbox.py:201-202``.
    """
    if isinstance(action_id, str):
        return action_id == "RESET"
    return action_id == RESET_ACTION


def virtual_reset_pair(
    b0: list[list[int]],
) -> tuple[list[list[list[int]]], list[int]]:
    """Build the RESET-seeding corpus pair ``([B0, B0], [RESET_ACTION])``.

    Both boards are deep copies of *b0* — immune to caller mutation and to
    each other. The duplicate is intentional: it IS the RESET transition
    (see module docstring). Shape invariant: ``len(actions) == len(grids)-1``.
    """
    grids: list[list[list[int]]] = [
        [row[:] for row in b0],
        [row[:] for row in b0],
    ]
    logger.debug("virtual_reset_pair: built ([B0, B0], [%d])", RESET_ACTION)
    return grids, [RESET_ACTION]


class ResetSeedTracker:
    """Agent-side one-time seed flag per game level.

    Level 1 (``game_level == 0``) starts from a RESET, so the sandbox corpus
    must be seeded exactly once with :func:`virtual_reset_pair`. Levels 2+
    start via level-completion with no RESET — ``should_seed`` is False for
    them ALWAYS. Marking is idempotent under double-mark.
    """

    def __init__(self) -> None:
        self._seeded_levels: set[int] = set()

    def mark_seeded(self, game_level: int) -> None:
        """Record that *game_level* has been RESET-seeded (idempotent)."""
        before = game_level in self._seeded_levels
        self._seeded_levels.add(game_level)
        if not before:
            logger.info("reset seed marked for game_level=%d", game_level)
        else:
            logger.debug(
                "reset seed re-mark (idempotent) for game_level=%d", game_level
            )

    def should_seed(self, game_level: int) -> bool:
        """True only for level 0 before marking; False for levels 2+ always."""
        if game_level != 0:
            # Levels 2+ start via level-completion with no RESET.
            logger.debug(
                "should_seed(game_level=%d) -> False (non-first level)", game_level
            )
            return False
        result = game_level not in self._seeded_levels
        logger.debug("should_seed(game_level=0) -> %s", result)
        return result

    def is_seeded_for(self, game_level: int) -> bool:
        """Readability alias: True iff *game_level* has been marked seeded."""
        return game_level in self._seeded_levels


def check_includes_reset() -> bool:
    """``check()`` scores RESET transitions as trained data (True).

    The offline corpus legitimately contains the RESET transition
    (``grids[0] == grids[1]``, ``actions[0] == 0``) — it is data like any
    other and scoring over it is meaningful.
    """
    return True


def diagnose_includes_reset() -> bool:
    """``diagnose()`` includes RESET transitions as trained data (True).

    Same rationale as :func:`check_includes_reset`.
    """
    return True


def bfs_includes_reset() -> bool:
    """BFS excludes RESET edges (False).

    A RESET edge cycles back to the level-start board, which BFS has
    already visited — expanding it is a wasteful cycle-back.
    """
    return False


def producing_action_caption(frame_index: int, actions: Sequence[int]) -> str:
    """Caption naming the action that PRODUCED *frame_index*.

    State-keyed pairing (``history[i] ↔ transition i ↔ actions[i]``): the
    action that produced frame ``i`` is ``actions[i-1]``. Frame 0 is
    produced by RESET itself. Out-of-range indices (including negatives)
    report ``unknown`` gracefully — this never raises.
    """
    if frame_index == 0:
        return f"action that produced frame 0: RESET (id {RESET_ACTION})"
    if 0 < frame_index <= len(actions):
        return f"action that produced frame {frame_index}: {actions[frame_index - 1]}"
    logger.debug(
        "producing_action_caption: frame_index=%d out of range for %d actions",
        frame_index,
        len(actions),
    )
    return f"action that produced frame {frame_index}: unknown"