"""In-process sandbox with persistent namespace for the simulator experiment.

The namespace persists across LLM turns so that ``set_simulate(func)``
registers a function that stays available for ``simulate()`` and
``check()`` in subsequent turns.

Supports two modes:
- **Offline** (ReplayHarness): replay a recording, access frames by index.
- **Live** (step_env_callback): take actions in real-time via the
  ``action()`` tool which calls the callback directly (no IPC).

Security: restricted builtins, dunder guard, import whitelist — same
pattern as ``duck_harness_agent/sandbox.py`` but in-process (no IPC
needed since there is no ``action()`` callback in offline mode, and
in live mode the callback is called directly in-process).
"""

from __future__ import annotations

import contextlib
import contextvars
import io
import linecache
import logging
import re
import sys
import threading
import traceback
import types  # noqa: F401  # reserved for Task 2 freeze + tool-protection logic
from collections import Counter, deque
from collections.abc import Callable
from io import StringIO
from typing import Any

from agents.simulator_agent.check import diagnose as diagnose_fn
from agents.simulator_agent.check import run_check
from agents.simulator_agent.frame_layers import (
    diff_count,
    is_board_reset,
    settled_board,
)
from agents.simulator_agent.reset_policy import (
    RESET_ACTION,
    bfs_includes_reset,
    is_reset,
    producing_action_caption,
    virtual_reset_pair,
)
from agents.simulator_agent.tools import (
    compute_delta_tool,
    count_color,
    find_color,
    find_objects_from_atoms,
    get_bbox,
    grid_diff,
    move_region,
    print_region,
    segment_atoms,
)
from replay.harness import ReplayHarness
from vision.render import (
    draw_boxes_on_grid,
    find_changed_regions,
    grid_to_image,
    image_to_base64,
)

logger = logging.getLogger(__name__)

# single RESET-decision log channel — see reset_policy.py
RESET_POLICY_LOGGER = logging.getLogger("simulator.reset_policy")

# board-reset detection channel — see frame_layers.is_board_reset
BOARD_RESET_LOGGER = logging.getLogger("simulator.board_reset")

# run_code() output marker when a batch was hard-aborted by a board reset;
# agent.py's tool loop detects this substring to end the turn — the raise
# site (sandbox.py) and the detector (agent.py) must never drift apart.
BOARD_RESET_MARKER = "[BOARD RESET — level budget exhausted; board restored to start; stop acting]"

# ── Sandbox security ──────────────────────────────────────────────────────

_ALLOWED_IMPORTS = frozenset(
    {
        "copy",
        "math",
        "re",
        "collections",
        "itertools",
        "functools",
        "json",
        "string",
        "random",
    }
)

_DANGEROUS_BUILTINS = frozenset(
    {
        "open",
        "compile",
        "eval",
        "exec",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
        "object",
    }
)

_DUNDER_PATTERN = re.compile(r"__\w+__")

_MAX_OUTPUT_CHARS = 4096

# Line events per second of ``timeout`` for sandbox-tagged code (see
# _DEFAULT_EXEC_BUDGET note above SandboxBudgetExceeded). Sized for honest
# heavy turns: full-board simulate loops and bfs() over ~50k nodes burn
# ~1e6-1e7 line events; a pure-Python busy loop burns ~1e6/second, so the
# budget fires in roughly the same wall-clock as ``timeout``. The tracer
# raise is a fast path, not the last line of defense — see
# ``run_code()`` for the wall-clock containment layer.
_DEFAULT_EXEC_BUDGET_PER_SECOND = 1_000_000


def _make_quarantine_raiser(name: str) -> Callable[..., Any]:
    """Build a neutered tool for an orphaned (zombie-occupied) namespace."""

    def _quarantined(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"[quarantined] '{name}' is unavailable: a previous execution "
            f"timed out and its sandbox was reset. Start fresh."
        )

    return _quarantined


# Builtins for a quarantined namespace: empty on purpose — any builtin use
# in the zombie raises (contained), and unbound memory growth via print is
# impossible because quarantine also neuters the orphan's "print" entry.
_QUARANTINE_SAFE_BUILTINS_SNAPSHOT: dict[str, Any] = {}


class LevelTransition(Exception):
    """Raised inside ``action()`` when ``last_action_result.level_completed`` is True.

    Hard-aborts the current ``run_code()`` batch: remaining batched actions
    never execute (they would step on the new level's fresh board). Caught
    explicitly in ``run_code()`` BEFORE the broad ``except Exception`` and
    converted to the marker output ``"[LEVEL COMPLETED — board transitioning; stop]"``
    with the winning action id preserved in ``self._action_taken``.
    """


class BoardReset(BaseException):
    """Raised inside ``action()`` when the response stack is a board-reset flash.

    Engine-initiated level-budget exhaustion: the API restored the board to
    the level start after an action (incident d91cdde0 f48/f92 — 6-layer
    stacks, 5 uniform animation layers over the restored board). This is
    NOT a level completion (``levels_completed`` unchanged — that is
    ``LevelTransition``'s job) and NOT an action-0 RESET (``reset_policy``
    owns that predicate). Detection is class-aware via
    ``frame_layers.is_board_reset`` against the level-start reference
    ``self._grids[0]``.

    Derives from ``BaseException`` (same reasoning as
    ``SandboxBudgetExceeded`` below): sandboxed code must not be able to
    swallow the batch-abort with ``except Exception`` and keep spinning
    refused calls — the unwind must be unconditional.

    Hard-aborts the current ``run_code()`` batch exactly like
    ``LevelTransition``: remaining batched actions never execute (they
    would step on the restored board mid-flash). Caught explicitly in
    ``run_code()`` BEFORE the broad ``except Exception`` and converted to
    the marker output ``BOARD_RESET_MARKER`` with the triggering action id
    preserved in ``self._action_taken``. ``_board_reset_pending`` stays
    ALIVE for agent-side consumption at the next turn boundary (Task 8).

    Refusal ordering (incident 6685d7d2): the pre-flag check in ``action()``
    raises BEFORE ``step_env_callback`` — a refused call must not step the
    real environment.
    """


class SandboxBudgetExceeded(BaseException):
    """Trace-hook budget interrupt for runaway sandbox code.

    Derives from ``BaseException`` (not ``Exception``) so sandboxed code
    cannot swallow it with ``except Exception`` — same reasoning as
    ``KeyboardInterrupt``. Raised from the line tracer when a code object
    tagged ``<sandbox>`` (the exec'd snippet or any function it registered)
    exceeds the per-``run_code()`` line-event budget.

    ``run_code()`` converts it to an error string before it can reach any
    other handler, so the BaseException never escapes the sandbox boundary.
    """


class SimulatorSandbox:
    """In-process sandbox with persistent namespace for the simulator experiment.

    The namespace persists across LLM turns so that ``set_simulate(func)``
    registers a function that stays available for ``simulate()`` and
    ``check()`` in subsequent turns.

    Two construction modes:

    1. **Offline** (default): pass ``harness`` to replay a recording.
    2. **Live**: pass ``step_env_callback`` to enable the ``action()``
       tool for real-time environment interaction.
    """

    def __init__(
        self,
        harness: ReplayHarness | None = None,
        timeout: float = 30.0,
        max_frames: int | None = None,
        step_env_callback: Callable[[int, dict[str, Any] | None], dict[str, Any]]
        | None = None,
    ) -> None:
        # ── Determine mode ──────────────────────────────────────────────
        if harness is not None and step_env_callback is not None:
            raise ValueError(
                "Provide either harness (offline) or step_env_callback (live), not both"
            )
        if harness is None and step_env_callback is None:
            raise ValueError(
                "Provide either harness (offline) or step_env_callback (live)"
            )

        self.harness = harness
        self.timeout = timeout
        self._exec_budget = int(float(timeout) * _DEFAULT_EXEC_BUDGET_PER_SECOND)
        self._step_env_callback: (
            Callable[[int, dict[str, Any] | None], dict[str, Any]] | None
        ) = step_env_callback
        self.actions_this_turn: int = 0
        self._action_taken: int | None = None

        # ── Load grids and actions ──────────────────────────────────────
        if harness is not None:
            # Offline mode: extract from replayed recording
            harness.replay_all()
            self._grids: list[list[list[int]]] = []
            self._actions: list[int] = []
            for fd in harness.frames:
                # settled_board: class-aware layer pick — flash frames
                # (d91cdde0 f48/f92) carry the restored board, not the
                # uniform phantom layer 0. FrameData.frame is already
                # [layer][row][col]; settled_board returns the settled
                # layer directly (plain lists — no tolist() needed).
                self._grids.append(
                    [[cell for cell in row] for row in settled_board(fd.frame)]
                )
            for ai in harness.action_inputs:
                action_id = ai["id"]
                if is_reset(action_id):
                    # RESET semantics owned by reset_policy — do not re-inline.
                    action_id = RESET_ACTION
                self._actions.append(int(action_id))

            if max_frames is not None and max_frames < len(self._grids):
                self._grids = self._grids[:max_frames]
                self._actions = self._actions[: max_frames - 1]
        else:
            # Live mode: start empty, grow as actions are taken
            self._grids = []
            self._actions = []

        # ── Live mode state tracking ────────────────────────────────────
        self._current_frame: list[list[int]] | None = None
        self._previous_grid: list[list[int]] | None = None
        self._valid_actions: list[int] = []
        self._last_action_result: dict[str, Any] = {}

        # Current simulator state
        self._simulate: Callable[[list[list[int]], int], list[list[int]]] | None = None
        self._simulate_source: str = ""
        self._last_check_result: dict[str, Any] | None = None
        self._last_bfs_result: list[int] | None = None
        self._pending_exception_flow: dict[str, int] | None = None
        self._ignore_mask: set[tuple[int, int]] = set()
        self._prev_correct_frames: set[int] = set()
        self.pending_images: list[dict[str, Any]] = []

        # Structured world model notes recorded via update_notes()
        self._pending_notes: dict[str, str] = {}

        # Level-transition hard-abort flag (set when a winning action is
        # detected inside action(); consumed by run_code's except chain).
        self._transition_pending: bool = False

        # Board-reset hard-abort flag (set when a flash stack is detected
        # inside action(); consumed by run_code's return path — the flag
        # itself stays alive for agent-side consumption at the next turn
        # boundary, mirroring _transition_pending's cross-turn semantics).
        self._board_reset_pending: bool = False

        # Transition indices produced by engine-initiated board resets
        # (the level-budget flash). Scoring-only: check()/diagnose() skip
        # them (no simulate can model the unmodelable restore), but the
        # corpus retains the transitions. RESET-action transitions are
        # still scored per check_includes_reset().
        self._engine_event_transitions: set[int] = set()

        # Worker-scoped stdout capture (run_code): the buffer installed as
        # sys.stdout for the exec worker, and the stream it replaced.
        self._worker_stdout: StringIO | None = None
        self._stdout_before_worker: Any = None

        # Build persistent namespace
        self.namespace: dict[str, Any] = self._build_namespace()

        self._protected_tools: dict[str, Any] = {
            k: self.namespace[k]
            for k in (
                "set_simulate", "simulate", "check", "diagnose", "bfs", "action",
                "set_ignore", "update_notes", "show_frame", "show_grid",
                "atoms", "find_objects", "find_color", "diff", "compute_delta",
                "copy_grid", "count_color", "print_region", "move_region",
                "get_bbox", "get_frame", "get_action",
            )
            if k in self.namespace
        }

    # ── Namespace construction ──────────────────────────────────────────

    def _build_namespace(self) -> dict[str, Any]:
        ns: dict[str, Any] = {}

        # ── Recording access ───────────────────────────────────────────
        ns["get_frame"] = lambda i: self._grids[i]
        ns["get_action"] = lambda i: self._actions[i]
        ns["n_frames"] = len(self._grids)

        # ── Simulator control ──────────────────────────────────────────
        def set_simulate(
            func: Callable[[list[list[int]], int], list[list[int]]],
        ) -> None:
            """Register a simulate(grid, action) -> next_grid function.

            The function's globals are snapshotted at registration time, so later
            redefinitions of helper functions do NOT affect the registered simulate.
            Call set_simulate() again to pick up new helper definitions.
            """
            frozen = types.FunctionType(
                func.__code__,
                dict(func.__globals__),
                func.__name__,
                func.__defaults__,
                func.__closure__,
            )
            self._simulate = frozen
            try:
                import inspect as _inspect
                src = _inspect.getsource(func)
                self._simulate_source = src
                first_line = src.splitlines()[0] if src else "(no source)"
            except Exception:
                self._simulate_source = "(source unavailable)"
                first_line = "(source unavailable)"
            print(
                f"[set_simulate] registered (helper definitions frozen at "
                f"registration; call set_simulate() again after redefining helpers)\n"
                f"[set_simulate] source first line: {first_line.strip()}"
            )

        ns["set_simulate"] = set_simulate

        def simulate(grid: list[list[int]], action: int) -> list[list[int]]:
            """Run the current simulate function (for self-testing)."""
            if self._simulate is None:
                raise RuntimeError(
                    "No simulate function set. Call set_simulate(func) first."
                )
            return self._simulate(grid, action)

        ns["simulate"] = simulate

        # ── Action tool (live mode only) ───────────────────────────────
        def action(action_id: int, **kwargs: Any) -> dict[str, Any]:
            """Take an action in the live environment.

            Calls ``step_env_callback(action_id, action_data)`` directly
            (in-process, no IPC). After the callback returns, the sandbox
            namespace is refreshed with the new state and the action is
            appended to the grid/action history.

            Returns the callback response dict so LLM code can inspect
            ``last_action_result``.
            """
            if self._step_env_callback is None:
                raise RuntimeError(
                    "action() is only available in live mode (step_env_callback required)"
                )

            # Build action_data from kwargs
            action_data: dict[str, Any] | None = dict(kwargs) if kwargs else None

            # Check if game is already over from previous action
            if self._last_action_result.get(
                "run_complete"
            ) or self._last_action_result.get("game_over"):
                raise RuntimeError("Game already won/over — no more actions allowed")

            # Level-transition hard-abort: a previous action in this batch
            # already completed a level. Remaining actions would step on the
            # new level's fresh board — refuse them.
            if self._transition_pending:
                raise LevelTransition(
                    f"Level already completed this batch (action {action_id} skipped)"
                )

            # Board-reset hard-abort: a previous action in this batch already
            # triggered the engine's level-budget flash. Remaining actions
            # would step on the restored board mid-flash — refuse them.
            # Both refusals sit BEFORE step_env_callback: stepping then
            # discarding burns real action budget, records frames the
            # sandbox corpus never ingests (env ↔ corpus divergence), and
            # leaves _action_taken=None so SpiralGuard miscounts real
            # actions as non-action calls (incident 6685d7d2 — 14 real
            # steps wasted post-flash).
            if self._board_reset_pending:
                raise BoardReset("Board already reset this batch — stop acting")

            self.actions_this_turn += 1

            # Remember the grid before action for history tracking
            prev_grid = self._current_frame

            # Call the callback directly (in-process)
            response = self._step_env_callback(action_id, action_data)

            # Append to grid/action history
            # We need _grids to have len(_actions) + 1 entries for check() to work:
            # _grids[i] = grid before action i, _grids[i+1] = grid after action i
            if prev_grid is not None:
                if not self._grids:
                    # First action: append the initial grid as the starting state
                    self._grids.append(prev_grid)
                new_grid = response.get("grid")
                if new_grid is not None:
                    self._grids.append(new_grid)
                self._actions.append(action_id)

            # Track last action taken
            self._action_taken = action_id

            # Refresh namespace variables from callback response
            ns["objects"] = response.get("objects", ns.get("objects", ()))
            ns["adjacency"] = response.get(
                "adjacency", ns.get("adjacency", frozenset())
            )
            ns["current_frame"] = response.get("grid", ns.get("current_frame"))
            ns["previous_frame"] = (
                prev_grid if prev_grid is not None else ns.get("previous_frame")
            )
            ns["valid_actions"] = response.get(
                "valid_actions", ns.get("valid_actions", [])
            )
            ns["last_action_result"] = response.get(
                "last_action_result", ns.get("last_action_result", {})
            )
            ns["history"] = response.get("history", ns.get("history", []))

            # Also update instance attributes for update_state() to work with
            self._current_frame = ns["current_frame"]
            self._previous_grid = prev_grid
            self._valid_actions = ns["valid_actions"]
            self._last_action_result = ns["last_action_result"]

            # Level-transition detection: the winning action's callback
            # returned level_completed=True. Suppress predict_and_compare
            # (the win-pose diff vs. the old level's simulate is meaningless),
            # flag the pending transition, and hard-abort the batch so the
            # remaining actions never step on the new level's fresh board.
            if self._last_action_result.get("level_completed"):
                self._transition_pending = True
                raise LevelTransition(
                    f"Level completed after action {action_id}"
                )

            # Board-reset detection: the response stack is the engine's
            # level-budget flash (uniform animation layers over a board
            # restored to the level start). The raise happens BEFORE
            # predict_and_compare, so the flash diff never enters the
            # exception flow — structural skip per reset-parity, no dead
            # branch inside predict_and_compare. Detection is independent
            # of simulate registration (works in EXPLORE phase too).
            frame_layers_raw = response.get("frame_layers")
            if frame_layers_raw and self._grids and is_board_reset(
                frame_layers_raw, self._grids[0], action_is_reset=is_reset(action_id)
            ):
                self._board_reset_pending = True
                self._engine_event_transitions.add(len(self._actions) - 1)
                BOARD_RESET_LOGGER.info(
                    "frame=%d board_reset detected action_id=%s layers=%d diff=%d cells",
                    len(self._grids) - 1, action_id, len(frame_layers_raw),
                    diff_count(frame_layers_raw[-1], self._grids[0]),
                )
                BOARD_RESET_LOGGER.debug(
                    "board-reset transition excluded from predict-and-compare per reset-parity"
                )
                raise BoardReset(
                    f"Level budget exhausted after action {action_id}; board restored to start"
                )

            # ── Predict-and-compare (extracted to predict_and_compare method) ─
            self.predict_and_compare(prev_grid, new_grid, action_id)
            # Update n_frames since grids grew
            ns["n_frames"] = len(self._grids)

            return response

        ns["action"] = action

        # ── Ignore mask ───────────────────────────────────────────────
        def set_ignore(
            cells: list[tuple[int, int]] | None = None,
            colors: list[int] | None = None,
        ) -> None:
            """Declare cells to skip in check() and diagnose().

            Pass cells as a list of (row, col) positions, or colors as a list
            of color indices, or both. All matching cells are excluded from
            accuracy metrics.
            """
            mask: set[tuple[int, int]] = set()
            if cells:
                mask.update(cells)
            if colors:
                for grid in self._grids:
                    for r in range(len(grid)):
                        for c in range(len(grid[0])):
                            if grid[r][c] in colors:
                                mask.add((r, c))
            self._ignore_mask = mask
            print(
                f"[set_ignore] {len(self._ignore_mask)} cells will be ignored in check()/diagnose()"
            )

        ns["set_ignore"] = set_ignore

        # ── Inspection tools (delegated to tools.py) ───────────────────
        ns["atoms"] = segment_atoms
        ns["find_color"] = find_color
        ns["find_objects"] = find_objects_from_atoms
        ns["get_bbox"] = get_bbox
        ns["diff"] = grid_diff
        ns["compute_delta"] = compute_delta_tool
        ns["copy_grid"] = lambda g: [row[:] for row in g]
        ns["count_color"] = count_color
        ns["print_region"] = print_region
        ns["move_region"] = move_region

        # ── Visual inspection tools ───────────────────────────────────
        def show_frame(frame_index: int, *, label: str = "") -> str:
            """Render a frame as an image for visual inspection."""
            if 0 <= frame_index < len(self._grids):
                grid = self._grids[frame_index]
                img = grid_to_image(grid, scale=8)
                b64 = image_to_base64(img)
                caption = (
                    f"Frame {frame_index} "
                    f"({producing_action_caption(frame_index, self._actions)})"
                )
                if label:
                    caption += f" — {label}"
                self.pending_images.append(
                    {
                        "b64": b64,
                        "caption": caption,
                    }
                )
                msg = f"[show_frame] Frame {frame_index} rendered — image will appear in next message"
                print(msg)
                return msg
            return f"Frame {frame_index} out of range (0-{len(self._grids) - 1})"

        ns["show_frame"] = show_frame

        def show_grid(grid: list[list[int]], *, label: str = "grid") -> str:
            """Render an arbitrary grid (e.g. your simulate output) as an image."""
            img = grid_to_image(grid, scale=8)
            b64 = image_to_base64(img)
            self.pending_images.append({"b64": b64, "caption": label})
            print("[show_grid] Rendered — image will appear in next message")
            return "[show_grid] rendered"

        ns["show_grid"] = show_grid

        # ── World model notes ───────────────────────────────────────────
        def update_notes(notes: str, plan: str = "") -> None:
            """Record your world model for next turn."""
            self._pending_notes = {"notes": notes, "plan": plan}
            print(
                f"[update_notes] recorded: notes={len(notes)} chars, plan={len(plan)} chars"
            )

        ns["update_notes"] = update_notes

        # ── Prebuilt check ─────────────────────────────────────────────
        def check(
            simulate_fn: Callable[[list[list[int]], int], list[list[int]]]
            | None = None,
        ) -> dict[str, Any]:
            """Test a simulate function against all recorded frames.

            If simulate_fn is None, uses the currently registered simulate.
            Prints a per-frame summary and returns aggregate metrics.
            """
            fn = simulate_fn if simulate_fn is not None else self._simulate
            if fn is None:
                print("No simulate function set. Call set_simulate(func) first.")
                return {"error": "no simulate function"}
            if len(self._grids) < 2:
                print(
                    "No transitions recorded yet — take an action first "
                    "(check() needs a before/after grid pair)."
                )
                return {"error": "no frames recorded"}
            result = run_check(
                fn,
                self._grids,
                self._actions,
                verbose=True,
                ignore_mask=self._ignore_mask,
                skip_transitions=self._engine_event_transitions,
            )
            self._last_check_result = result
            return result

        ns["check"] = check

        def diagnose(
            simulate_fn: Callable[[list[list[int]], int], list[list[int]]]
            | None = None,
        ) -> dict[str, Any]:
            """Test simulate and print semantic error analysis.

            For each wrong frame, classifies errors as:
            - MISSED: cells that changed in reality but simulate didn't change
            - SPURIOUS: cells simulate changed but shouldn't have
            - WRONG_VALUE: cells simulate changed but to the wrong color

            Groups errors by spatial cluster to identify which object is wrong.
            """
            fn = simulate_fn if simulate_fn is not None else self._simulate
            if fn is None:
                print("No simulate function set. Call set_simulate(func) first.")
                return {"error": "no simulate function"}
            if len(self._grids) < 2:
                print(
                    "No transitions recorded yet — take an action first "
                    "(diagnose() needs a before/after grid pair)."
                )
                return {"error": "no frames recorded"}
            return diagnose_fn(
                fn,
                self._grids,
                self._actions,
                ignore_mask=self._ignore_mask,
                skip_transitions=self._engine_event_transitions,
            )

        ns["diagnose"] = diagnose

        def bfs(
            start_grid: list[list[int]],
            goal_fn: Callable[[list[list[int]]], bool],
            max_depth: int = 20,
        ) -> list[int] | None:
            """Search for a path from start_grid to a goal state using simulate().

            Args:
                start_grid: The grid to search from.
                goal_fn: A function(grid) -> bool. Returns True when the goal
                    is reached. Any print() output from goal_fn at the goal
                    state is captured and shown.
                max_depth: Maximum path length (default 20).

            Returns:
                A list of action IDs, or None if no path found.
                Requires set_simulate() to be called first.
            """
            if self._simulate is None:
                print("No simulate function set. Call set_simulate(func) first.")
                return None

            valid_acts = self.namespace.get("valid_actions", [])
            if not valid_acts:
                print("No valid actions available.")
                return None

            if not bfs_includes_reset():
                # Policy (reset_policy): RESET edges cycle back to the
                # level-start board BFS already visited — pure waste.
                reset_branches = [a for a in valid_acts if is_reset(a)]
                if reset_branches:
                    valid_acts = [a for a in valid_acts if not is_reset(a)]
                    RESET_POLICY_LOGGER.debug(
                        "bfs: excluded RESET branch(es) %s per reset_policy "
                        "(bfs_includes_reset=False); branching on %s",
                        reset_branches,
                        valid_acts,
                    )

            def grid_hash(g: list[list[int]]) -> tuple[tuple[int, ...], ...]:
                return tuple(tuple(row) for row in g)

            def deep_copy(g: list[list[int]]) -> list[list[int]]:
                return [row[:] for row in g]

            start_hash = grid_hash(start_grid)
            visited: set[tuple[tuple[int, ...], ...]] = {start_hash}
            queue: deque[tuple[list[list[int]], list[int]]] = deque(
                [(deep_copy(start_grid), [])]
            )
            max_nodes = 50_000
            nodes_expanded = 0

            try:
                if goal_fn(deep_copy(start_grid)):
                    goal_buf = io.StringIO()
                    with contextlib.redirect_stdout(goal_buf):
                        goal_fn(deep_copy(start_grid))
                    print("Goal already reached. Path: [] (0 steps)")
                    goal_output = goal_buf.getvalue().strip()
                    if goal_output:
                        print(goal_output)
                    return []
            except Exception as e:
                print(f"Error in goal_fn: {e}")
                return None

            while queue and nodes_expanded < max_nodes:
                grid, path = queue.popleft()
                nodes_expanded += 1

                if len(path) >= max_depth:
                    continue

                for action_id in valid_acts:
                    try:
                        next_grid: Any = self._simulate(deep_copy(grid), action_id)
                    except Exception:
                        continue

                    if next_grid is None:
                        continue

                    nh = grid_hash(next_grid)
                    if nh in visited:
                        continue
                    visited.add(nh)

                    new_path = path + [action_id]

                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            is_goal = goal_fn(deep_copy(next_grid))
                    except Exception:
                        continue

                    if is_goal:
                        goal_buf = io.StringIO()
                        with contextlib.redirect_stdout(goal_buf):
                            goal_fn(deep_copy(next_grid))
                        print(f"Path found: {new_path} ({len(new_path)} steps)")
                        goal_output = goal_buf.getvalue().strip()
                        if goal_output:
                            print(goal_output)
                        self._last_bfs_result = new_path
                        return new_path

                    queue.append((next_grid, new_path))

            print(f"No path found within depth {max_depth}.")
            self._last_bfs_result = None
            return None

        ns["bfs"] = bfs

        return ns

    def predict_and_compare(
        self,
        prev_grid: list[list[int]] | None,
        new_grid: list[list[int]] | None,
        action_id: int,
    ) -> None:
        """Run simulate on prev_grid+action_id, compare with new_grid.

        On diff: populates self._pending_exception_flow with region fingerprint
        and renders a boxed visual diff into self.pending_images.

        Cells in self._ignore_mask (set via set_ignore) are hidden from the
        comparison, matching check()/diagnose() semantics — an action whose
        only diffs are ignored (e.g. HUD timer ticks) produces no exception
        flow.

        On crash: populates self._pending_exception_flow with the traceback tail.

        On match: clears self._pending_exception_flow (no exception flow).

        Skipped when simulate isn't registered, the action is RESET (policy
        owned by ``agents/simulator_agent/reset_policy.py`` — ``is_reset``),
        or game is over. The RESET skip is logged (never silent).
        """
        if is_reset(action_id):
            RESET_POLICY_LOGGER.debug(
                "predict_and_compare: skipping RESET action %d per reset_policy",
                action_id,
            )
            return
        if (
            self._simulate is None
            or prev_grid is None
            or new_grid is None
            or self._last_action_result.get("run_complete")
            or self._last_action_result.get("game_over")
        ):
            return
        try:
            grid_copy = [row[:] for row in prev_grid]
            predicted = self._simulate(grid_copy, action_id)
            if (
                predicted is not None
                and len(predicted) == len(new_grid)
                and all(
                    len(pr) == len(nr)
                    for pr, nr in zip(predicted, new_grid)
                )
            ):
                # Hide ignored cells by making the prediction agree with
                # reality there — every downstream diff computation then
                # covers only cells the LLM hasn't declared unimportant.
                if self._ignore_mask:
                    mask = self._ignore_mask
                    predicted = [
                        [
                            new_cell if (r, c) in mask else cell
                            for c, (cell, new_cell) in enumerate(
                                zip(row, new_row)
                            )
                        ]
                        for r, (row, new_row) in enumerate(
                            zip(predicted, new_grid)
                        )
                    ]
                if predicted != new_grid:
                    n_diff = sum(
                        1
                        for r in range(len(predicted))
                        for c in range(len(predicted[0]))
                        if predicted[r][c] != new_grid[r][c]
                    )
                    # Region fingerprint (top 4 by cell count)
                    raw_regions = find_changed_regions(predicted, new_grid)

                    def _area(r):
                        return (r[1] - r[0] + 1) * (r[3] - r[2] + 1)

                    sorted_regions = sorted(
                        raw_regions, key=_area, reverse=True
                    )[:4]
                    region_info: list[dict[str, Any]] = []
                    for r0, r1, c0, c1 in sorted_regions:
                        trans: Counter = Counter()
                        for r in range(r0, r1 + 1):
                            for c in range(c0, c1 + 1):
                                if predicted[r][c] != new_grid[r][c]:
                                    trans[
                                        (predicted[r][c], new_grid[r][c])
                                    ] += 1
                        top = ", ".join(
                            f"{o}->{n}"
                            for (o, n), _ in trans.most_common(2)
                        )
                        region_info.append(
                            {
                                "bbox": (r0, c0, r1, c1),
                                "n_cells": sum(trans.values()),
                                "transitions": top,
                            }
                        )
                    self._pending_exception_flow = {
                        "action_id": action_id,
                        "n_diff": n_diff,
                        "n_regions": len(raw_regions),
                        "regions": region_info,
                        "error": None,
                    }
                    # Visual diff image (boxed actual frame)
                    try:
                        diff_img = draw_boxes_on_grid(new_grid, raw_regions)
                        self.pending_images.append(
                            {
                                "b64": image_to_base64(diff_img),
                                "caption": (
                                    f"SIMULATION DIFF after action {action_id} "
                                    f"({n_diff} cells in {len(raw_regions)} regions, red = wrong)"
                                ),
                            }
                        )
                    except Exception as img_exc:
                        logger.warning(
                            f"failed to render diff image: {img_exc}"
                        )
                else:
                    self._pending_exception_flow = None
        except Exception as exc:
            # Crash path: capture traceback tail for the LLM
            tb = traceback.format_exc()
            tail = "\n".join(tb.strip().splitlines()[-3:])[:500]
            logger.warning(
                f"exception_flow: simulate crashed during predict-and-compare: {exc}"
            )
            self._pending_exception_flow = {
                "action_id": action_id,
                "n_diff": 0,
                "n_regions": 0,
                "regions": [],
                "error": tail,
            }

    def _protect_tool_names(self, output: str) -> str:
        """Restore any protected tool names that the LLM code reassigned.

        Appends a warning to output for each one. Returns the updated output.
        """
        for name, original in self._protected_tools.items():
            current = self.namespace.get(name)
            if current is not original:
                self.namespace[name] = original
                warning = (
                    f"[sandbox] WARNING: '{name}' is a builtin tool and was "
                    f"protected from redefinition"
                )
                output += "\n" + warning
                print(warning)
        return output

    # ── Live mode: state refresh ────────────────────────────────────────

    def update_state(
        self,
        objects: Any,
        adjacency: Any,
        current_frame: list[list[int]] | None,
        previous_frame: list[list[int]] | None,
        valid_actions: list[int],
        last_action_result: dict[str, Any],
        history: list[Any],
        reset_seeded: bool = False,
    ) -> None:
        """Set namespace variables from the agent before each ``run_code()`` call.

        Called by the live agent to push fresh environment state into the
        sandbox namespace so LLM code can access ``objects``, ``adjacency``,
        ``current_frame``, etc.

        ``reset_seeded`` (agent-side decision, see
        ``reset_policy.ResetSeedTracker``): when True and the corpus is
        empty, seed the virtual RESET pair ``([B0, B0], [0])`` via
        ``reset_policy.virtual_reset_pair`` — level 1 starts from a RESET,
        so transition 0 must be ``(B0, RESET) -> B0`` (the duplicate board
        IS the RESET transition; naive ``grids=[B0]``/``actions=[0]``
        seeding would corrupt it into ``(B0, RESET) -> B1``). When False
        (default), seed grids-only — the levels-2+ path, which start via
        level-completion with no RESET.
        """
        self.namespace["objects"] = objects
        self.namespace["adjacency"] = adjacency
        self.namespace["current_frame"] = current_frame
        self.namespace["previous_frame"] = previous_frame
        self.namespace["valid_actions"] = valid_actions
        self.namespace["last_action_result"] = last_action_result
        self.namespace["history"] = history

        # Also store as instance attributes for action() to update
        self._current_frame = current_frame
        self._previous_grid = previous_frame
        self._valid_actions = valid_actions
        self._last_action_result = last_action_result

        # Seed frame 0 so check()/diagnose() have data before any action();
        # action()'s first-append branch only fires when _grids is empty,
        # so seeding here keeps grids[i]+actions[i]->grids[i+1] consistent.
        #
        # Pair semantics (reset_seeded=True): the seed is the virtual RESET
        # pair ([B0, B0], [0]) — transition 0 is (B0, RESET) -> B0. The
        # first real action then appends (B1, a1) as transition 1, giving
        # _grids == [B0, B0, B1] / _actions == [0, a1]. The pair shape is
        # what kills the corruption trap (B0, RESET) -> B1.
        if not self._grids and current_frame is not None:
            if reset_seeded:
                pair_grids, pair_actions = virtual_reset_pair(current_frame)
                self._grids.extend(pair_grids)
                self._actions.extend(pair_actions)
                logger.info(
                    "simulatorfirst: seeded virtual RESET pair into sandbox "
                    "corpus (grids=%d, actions=%d)",
                    len(self._grids),
                    len(self._actions),
                )
            else:
                self._grids.append([row[:] for row in current_frame])
            self.namespace["n_frames"] = len(self._grids)

    def reset_turn_counter(self) -> None:
        """Reset the per-turn action counter and action_taken tracker.

        Called by the agent at the start of each turn before ``run_code()``.
        """
        self.actions_this_turn = 0
        self._action_taken = None

    def reset_for_level_transition(self) -> None:
        """Clear per-level state after a ``LevelTransition`` hard-abort.

        Clears the 12 per-level structures (grids, actions, check state,
        ignore mask, correct-frame cache, pending exception flow,
        engine-event transition set, pending images, current frame,
        previous grid, last action result, and the matching namespace
        vars). PRESERVES ``_simulate``, ``_simulate_source``
        (the LLM's carried-forward hypothesis) and ``_pending_notes`` (notes
        recorded on the transition turn survive the clear). The
        ``_transition_pending`` flag is intentionally NOT cleared here — it
        is a cross-turn signal consumed by the agent on the next turn.

        Reseed on levels 2+ is grids-only (``update_state`` with
        ``reset_seeded=False``): levels 2+ start via level-completion with
        no RESET, so no virtual RESET pair is fabricated for them (see
        ``reset_policy.ResetSeedTracker.should_seed``).
        """
        self._grids = []
        self._actions = []
        self._last_check_result = None
        self._ignore_mask = set()
        self._prev_correct_frames = set()
        self._pending_exception_flow = None
        self._engine_event_transitions = set()
        self.pending_images = []
        self._current_frame = None
        self._previous_grid = None
        self._last_action_result = {}
        self.namespace["current_frame"] = None
        self.namespace["previous_frame"] = None
        self.namespace["history"] = []
        self.namespace["last_action_result"] = {}
        self.namespace["n_frames"] = 0

    # ── Code execution ─────────────────────────────────────────────────

    def run_code(self, code: str) -> tuple[str, str | None, int | None]:
        """Exec *code* in the sandbox namespace with timeout.

        Returns ``(stdout_output, error_message, action_taken)``.

        ``action_taken`` is the last action ID taken during this code
        execution, or ``None`` if no action was called.  It is reset
        to ``None`` at the start of each ``run_code()`` call.

        Two-layer runaway protection (replaces the SIGALRM timeout, which
        was structurally dead: agents run in daemon threads and CPython
        delivers signals only in the main thread's bytecode loop):

        1. **Tracer budget** (fast path): a per-thread line tracer meters
           ``<sandbox>``-tagged code and raises ``SandboxBudgetExceeded``
           when the budget (``timeout`` × 1e6 line events) is exhausted.
           Covers plain loops and ``except Exception`` swallows.
        2. **Wall-clock containment** (last resort): the exec runs in a
           worker thread and the caller joins with ``timeout``. If the
           worker is still alive (bare ``except:`` can swallow the tracer
           raise — CPython 3.12 pathology at high event counts), the
           orphaned namespace is quarantined (tools neutered, fresh
           namespace swapped in) and ``run_code`` returns a timeout error.
           The zombie keeps spinning but can no longer reach the
           environment or the live namespace; the agent loop resumes.
        """
        # Reset action tracker for this execution
        self._action_taken = None
        # Only the latest run_code() call's notes persist
        self._pending_notes = {}

        if _DUNDER_PATTERN.search(code):
            return ("", "Error: dunder attributes are not allowed", None)

        # Seed linecache so inspect.getsource() works for exec'd code
        # (co_filename "<sandbox>" is not a real file). Incident 5681a14a:
        # without this, set_simulate() never captured the source and every
        # recording recorded an empty _simulate_source.
        linecache.cache["<sandbox>"] = (
            len(code),
            None,
            code.splitlines(True),
            "<sandbox>",
        )

        # Restricted builtins
        raw_builtins = __builtins__
        if isinstance(raw_builtins, dict):
            safe_builtins = {
                k: v for k, v in raw_builtins.items() if k not in _DANGEROUS_BUILTINS
            }
            real_import = raw_builtins["__import__"]
        else:
            safe_builtins = {
                k: v
                for k, v in vars(raw_builtins).items()
                if k not in _DANGEROUS_BUILTINS
            }
            real_import = raw_builtins.__import__

        def safe_import(
            name: str,
            globals: dict[str, Any] | None = None,  # noqa: A002
            locals: dict[str, Any] | None = None,  # noqa: A002
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            top = name.split(".")[0]
            if top not in _ALLOWED_IMPORTS:
                allowed = ", ".join(sorted(_ALLOWED_IMPORTS))
                raise ImportError(
                    f"import of '{name}' is not allowed. Allowed: {allowed}"
                )
            return real_import(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = safe_import

        buf = StringIO()
        result: dict[str, Any] = {"output": "", "error": None, "action": None}
        old_builtins = self.namespace.get("__builtins__")

        # Tool closures (check/bfs/show_frame/set_simulate/...) print via the
        # module-level print, so capture needs the process stdout swap (the
        # pre-composite mechanism) — a namespace-scoped shadow print cannot
        # reach them. The steal bug from the SIGALRM era stays fixed: the
        # worker restores stdout in its finally, and _quarantine_zombie
        # restores it when the worker never gets there.
        self._worker_stdout = buf
        self._stdout_before_worker = sys.stdout

        def _worker() -> None:
            """Exec the code on this worker thread; never raises.

            The tracer is installed per-thread from inside, so arming it
            here targets exactly the sandbox execution — no main-thread
            requirement, no effect on the caller's trace state (e.g.
            pytest/coverage on the agent thread). The namespace dict is
            captured once: after a quarantine swap the zombie's finally
            block would otherwise mutate the fresh namespace.
            """
            ns = self.namespace
            worker_trace = sys.gettrace()
            sys.settrace(self._make_budget_tracer())
            ns["__builtins__"] = safe_builtins
            sys.stdout = buf
            try:
                exec(compile(code, "<sandbox>", "exec"), ns)
            except SandboxBudgetExceeded as exc:
                result["error"] = (
                    f"{type(exc).__name__}: your code exceeded the execution "
                    f"budget ({self._exec_budget} line events for sandbox "
                    f"code). Bound your loops (e.g. `for _ in range(200)`); "
                    f"never loop on simulate() output without a step cap."
                )
                logger.warning(f"simulatorfirst: sandbox budget exceeded: {exc}")
            except LevelTransition:
                result["error"] = None
            except BoardReset:
                result["error"] = None
            except BaseException as exc:  # noqa: BLE001 — sandbox boundary
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                sys.settrace(worker_trace)
                if sys.stdout is buf:
                    sys.stdout = self._stdout_before_worker
                if old_builtins is not None:
                    ns["__builtins__"] = old_builtins
                else:
                    ns.pop("__builtins__", None)
                result["output"] = buf.getvalue()

        # Incident a21a2571 (after 24e7dce): _active_log_path is a ContextVar
        # (agents/agent.py) — thread-local, so this worker got a FRESH context
        # and every log line emitted from sandbox code (ACTION counts, phase=,
        # check results) was silently dropped by the recording filter.
        worker_ctx = contextvars.copy_context()
        worker = threading.Thread(
            target=lambda: worker_ctx.run(_worker),
            name="sandbox-exec",
            daemon=True,
        )
        worker.start()
        # Containment bound: 4x timeout gives honest slow turns (BFS with
        # real engine calls burn wall-clock but few line events) room to
        # finish after the tracer fast-path would have fired on a pure spin.
        containment_timeout = max(4.0 * self.timeout, 2.0)
        worker.join(timeout=containment_timeout)

        timed_out = worker.is_alive()
        if timed_out:
            self._quarantine_zombie()
            error = (
                f"SandboxTimeout: your code did not finish within "
                f"{containment_timeout:g}s (it also survived the line-event "
                f"budget, likely via a bare `except:`). The sandbox state "
                f"was reset; bound your loops (e.g. `for _ in range(200)`) "
                f"and re-run."
            )
            output = "(no output — sandbox timed out and was reset)"
            logger.critical(
                "simulatorfirst: sandbox exec exceeded %gs and survived the "
                "tracer budget — zombie worker quarantined, namespace reset",
                containment_timeout,
            )
        else:
            error = result["error"]
            output = self._protect_tool_names(result["output"])
            if len(output) > _MAX_OUTPUT_CHARS:
                output = (
                    output[:_MAX_OUTPUT_CHARS]
                    + f"\n... (output capped at {_MAX_OUTPUT_CHARS} chars — you printed too much. "
                    "Use smaller queries: atoms() for overview, find_color() for specific colors, "
                    "print_region with max 20×20 areas.)"
                )

        if self._transition_pending and not timed_out:
            self.reset_for_level_transition()
            return (
                "[LEVEL COMPLETED — board transitioning; stop]",
                None,
                self._action_taken,
            )

        if self._board_reset_pending and not timed_out:
            # No state clear here (RESET-parity): the corpus already holds
            # the reset transition (pre-flash grid + action + restored
            # grid). _board_reset_pending stays ALIVE for agent-side
            # consumption at the next turn boundary — only the marker is
            # returned.
            return (
                BOARD_RESET_MARKER,
                None,
                self._action_taken,
            )

        return (output, error, self._action_taken)

    def _quarantine_zombie(self) -> None:
        """Contain a runaway exec worker that survived the tracer budget.

        The abandoned worker thread keeps a reference to the *old* namespace
        dict as its exec globals, so mutation of that dict is harmless — but
        its tool entries still close over ``self`` and could step the real
        environment. Replace every protected tool in the orphaned namespace
        with a raiser, then swap ``self.namespace`` for a fresh build so the
        next ``run_code()``/``update_state()`` sees pristine bindings.
        """
        orphan = self.namespace
        orphan["action"] = _make_quarantine_raiser("action")
        for name in self._protected_tools:
            orphan[name] = _make_quarantine_raiser(name)
        orphan["print"] = _make_quarantine_raiser("print")
        orphan["__builtins__"] = _QUARANTINE_SAFE_BUILTINS_SNAPSHOT
        worker_buf = getattr(self, "_worker_stdout", None)
        before = getattr(self, "_stdout_before_worker", None)
        if worker_buf is not None and before is not None and sys.stdout is worker_buf:
            sys.stdout = before
        self.namespace = self._build_namespace()
        self._protected_tools = {
            k: self.namespace[k]
            for k in self._protected_tools
            if k in self.namespace
        }

    def _make_budget_tracer(self) -> Callable[[Any, str, Any], Any]:
        """Build a line tracer metering only ``<sandbox>``-tagged code objects.

        On budget exhaustion the tracer disarms itself globally
        (``sys.settrace(None)``) and goes silent before raising: line events
        keep firing while the exception unwinds through user ``except:``
        handlers, and re-raising from those events traps the interpreter in
        an unwinding loop that never terminates (reproduced on 3.12.3 with
        budgets >=~50k line events + a bare-except loop). Going silent
        guarantees unwind-to-``run_code``'s except chain in one pass.

        Returns ``None`` for non-sandbox frames, which disables tracing for
        that whole call subtree — engine calls and HTTP work inside
        ``action()`` consume zero budget and add zero tracer overhead there.
        """
        budget_remaining = self._exec_budget
        fired = False

        def tracer(frame: Any, event: str, arg: Any) -> Any:
            nonlocal fired
            if frame.f_code.co_filename != "<sandbox>":
                return None
            if fired:
                return None
            nonlocal budget_remaining
            budget_remaining -= 1
            if budget_remaining <= 0:
                fired = True
                sys.settrace(None)
                raise SandboxBudgetExceeded(
                    f"line-event budget of {self._exec_budget} exceeded"
                )
            return tracer

        return tracer

    # ── Measurement (called by the harness after each turn) ────────────

    def measure(self) -> dict[str, Any]:
        """Run the current simulate on all frames.  Returns metrics dict.

        In live mode, returns a sensible default with ``accuracy: None``
        since the grid history grows dynamically.
        """
        # In live mode or with no simulate function, return defaults
        if self._simulate is None:
            return {
                "accuracy": None,
                "wrong_cells": 0,
                "frames_correct": 0,
                "frames_total": max(len(self._grids) - 1, 0),
                "regressions": 0,
            }

        # Need at least 2 grids (before + after) and matching actions
        if len(self._grids) < 2 or len(self._actions) < 1:
            return {
                "accuracy": None,
                "wrong_cells": 0,
                "frames_correct": 0,
                "frames_total": max(len(self._grids) - 1, 0),
                "regressions": 0,
            }

        result = run_check(
            self._simulate,
            self._grids,
            self._actions,
            verbose=False,
            ignore_mask=self._ignore_mask,
            skip_transitions=self._engine_event_transitions,
        )

        if "error" in result:
            return {
                "accuracy": 0.0,
                "wrong_cells": result["total_wrong"],
                "frames_correct": 0,
                "frames_total": len(self._grids) - 1,
                "regressions": 0,
            }

        current_correct = {
            fr["frame"] for fr in result["per_frame"] if fr.get("wrong", 999) == 0
        }
        regressions = len(self._prev_correct_frames - current_correct)
        self._prev_correct_frames = current_correct

        return {
            "accuracy": result["overall_accuracy"],
            "wrong_cells": result["total_wrong"],
            "frames_correct": result["frames_correct"],
            "frames_total": result["frames_total"],
            "regressions": regressions,
        }
