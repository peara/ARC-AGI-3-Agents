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
import io
import re
import signal
import sys
import threading
from collections import deque
from collections.abc import Callable
from io import StringIO
from typing import Any

from agents.simulator_agent.check import diagnose as diagnose_fn
from agents.simulator_agent.check import run_check
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
from perception.objects import to_grid
from replay.harness import ReplayHarness
from vision.render import grid_to_image, image_to_base64

# ── Sandbox security ──────────────────────────────────────────────────────

_ALLOWED_IMPORTS = frozenset(
    {
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
                grid_2d = to_grid(fd.frame)
                self._grids.append(grid_2d.tolist())
            for ai in harness.action_inputs:
                action_id = ai["id"]
                if isinstance(action_id, str) and action_id == "RESET":
                    action_id = 0
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

        # Build persistent namespace
        self.namespace: dict[str, Any] = self._build_namespace()

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
            """Register a simulate(grid, action) -> next_grid function."""
            self._simulate = func
            try:
                import inspect as _inspect

                self._simulate_source = _inspect.getsource(func)
            except Exception:
                self._simulate_source = "(source unavailable)"
            print("[set_simulate] registered. simulate(grid, action) is now available.")

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

            # ── Predict-and-compare: run simulate on prev grid + action, compare with actual
            if (
                self._simulate is not None
                and action_id != 0
                and prev_grid is not None
                and new_grid is not None
                and not self._last_action_result.get("run_complete")
                and not self._last_action_result.get("game_over")
            ):
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
                        if predicted != new_grid:
                            n_diff = sum(
                                1
                                for r in range(len(predicted))
                                for c in range(len(predicted[0]))
                                if predicted[r][c] != new_grid[r][c]
                            )
                            self._pending_exception_flow = {
                                "action_id": action_id,
                                "n_diff": n_diff,
                            }
                        else:
                            self._pending_exception_flow = None
                except Exception as exc:
                    import logging
                    logging.getLogger("sandbox").warning(
                        f"exception_flow: simulate crashed during predict-and-compare: {exc}"
                    )

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
                action = (
                    self._actions[frame_index]
                    if frame_index < len(self._actions)
                    else "?"
                )
                caption = f"Frame {frame_index} (action={action})"
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
            result = run_check(
                fn,
                self._grids,
                self._actions,
                verbose=True,
                ignore_mask=self._ignore_mask,
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
            return diagnose_fn(
                fn, self._grids, self._actions, ignore_mask=self._ignore_mask
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
    ) -> None:
        """Set namespace variables from the agent before each ``run_code()`` call.

        Called by the live agent to push fresh environment state into the
        sandbox namespace so LLM code can access ``objects``, ``adjacency``,
        ``current_frame``, etc.
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

    def reset_turn_counter(self) -> None:
        """Reset the per-turn action counter and action_taken tracker.

        Called by the agent at the start of each turn before ``run_code()``.
        """
        self.actions_this_turn = 0
        self._action_taken = None

    # ── Code execution ─────────────────────────────────────────────────

    def run_code(self, code: str) -> tuple[str, str | None, int | None]:
        """Exec *code* in the sandbox namespace with timeout.

        Returns ``(stdout_output, error_message, action_taken)``.

        ``action_taken`` is the last action ID taken during this code
        execution, or ``None`` if no action was called.  It is reset
        to ``None`` at the start of each ``run_code()`` call.
        """
        # Reset action tracker for this execution
        self._action_taken = None
        # Only the latest run_code() call's notes persist
        self._pending_notes = {}

        if _DUNDER_PATTERN.search(code):
            return ("", "Error: dunder attributes are not allowed", None)

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

        old_builtins = self.namespace.get("__builtins__")
        self.namespace["__builtins__"] = safe_builtins

        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        error: str | None = None

        in_main_thread = threading.current_thread() is threading.main_thread()

        old_handler: Any = None
        if in_main_thread:

            def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
                raise TimeoutError(f"Sandbox timed out after {self.timeout}s")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(max(int(self.timeout), 1))

        try:
            exec(compile(code, "<sandbox>", "exec"), self.namespace)  # noqa: S102
        except TimeoutError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if in_main_thread:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            sys.stdout = old_stdout
            if old_builtins is not None:
                self.namespace["__builtins__"] = old_builtins
            else:
                self.namespace.pop("__builtins__", None)

        output = buf.getvalue()
        if len(output) > _MAX_OUTPUT_CHARS:
            output = (
                output[:_MAX_OUTPUT_CHARS]
                + f"\n... (output capped at {_MAX_OUTPUT_CHARS} chars — you printed too much. "
                "Use smaller queries: atoms() for overview, find_color() for specific colors, "
                "print_region with max 20×20 areas.)"
            )

        return (output, error, self._action_taken)

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
