"""In-process sandbox with persistent namespace for the simulator experiment.

The namespace persists across LLM turns so that ``set_simulate(func)``
registers a function that stays available for ``simulate()`` and
``check()`` in subsequent turns.

Security: restricted builtins, dunder guard, import whitelist — same
pattern as ``duck_harness_agent/sandbox.py`` but in-process (no IPC
needed since there is no ``action()`` callback).
"""

from __future__ import annotations

import re
import signal
import sys
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

_ALLOWED_IMPORTS = frozenset({
    "math", "re", "collections", "itertools", "functools",
    "json", "string", "random",
})

_DANGEROUS_BUILTINS = frozenset({
    "open", "compile", "eval", "exec",
    "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "type", "object",
})

_DUNDER_PATTERN = re.compile(r"__\w+__")

_MAX_OUTPUT_CHARS = 4096


class SimulatorSandbox:
    """In-process sandbox with persistent namespace for the simulator experiment.

    The namespace persists across LLM turns so that ``set_simulate(func)``
    registers a function that stays available for ``simulate()`` and
    ``check()`` in subsequent turns.
    """

    def __init__(
        self, harness: ReplayHarness, timeout: float = 30.0, max_frames: int | None = None
    ) -> None:
        self.harness = harness
        self.timeout = timeout

        # Extract grids and actions from the replayed recording
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

        # Current simulator state
        self._simulate: Callable[[int, int], list[list[int]]] | None = None
        self._prev_correct_frames: set[int] = set()
        self.pending_images: list[dict[str, Any]] = []

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
        def set_simulate(func: Callable[[int, int], list[list[int]]]) -> None:
            """Register a simulate(frame_index, action) -> next_grid function."""
            self._simulate = func

        ns["set_simulate"] = set_simulate

        def simulate(i: int, action: int) -> list[list[int]]:
            """Run the current simulate function (for self-testing)."""
            if self._simulate is None:
                raise RuntimeError(
                    "No simulate function set. Call set_simulate(func) first."
                )
            return self._simulate(i, action)

        ns["simulate"] = simulate

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
                action = self._actions[frame_index] if frame_index < len(self._actions) else "?"
                caption = f"Frame {frame_index} (action={action})"
                if label:
                    caption += f" — {label}"
                self.pending_images.append({
                    "b64": b64,
                    "caption": caption,
                })
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

        # ── Prebuilt check ─────────────────────────────────────────────
        def check(simulate_fn: Callable[[int, int], list[list[int]]] | None = None) -> dict[str, Any]:
            """Test a simulate function against all recorded frames.

            If simulate_fn is None, uses the currently registered simulate.
            Prints a per-frame summary and returns aggregate metrics.
            """
            fn = simulate_fn if simulate_fn is not None else self._simulate
            if fn is None:
                print("No simulate function set. Call set_simulate(func) first.")
                return {"error": "no simulate function"}
            return run_check(fn, self._grids, self._actions, verbose=True)  # type: ignore[no-any-return]

        ns["check"] = check

        def diagnose(simulate_fn: Callable[[int, int], list[list[int]]] | None = None) -> dict[str, Any]:
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
            return diagnose_fn(fn, self._grids, self._actions)  # type: ignore[no-any-return]

        ns["diagnose"] = diagnose

        return ns

    # ── Code execution ─────────────────────────────────────────────────

    def run_code(self, code: str) -> tuple[str, str | None]:
        """Exec *code* in the sandbox namespace with timeout.

        Returns ``(stdout_output, error_message)``.
        """
        if _DUNDER_PATTERN.search(code):
            return ("", "Error: dunder attributes are not allowed")

        # Restricted builtins
        raw_builtins = __builtins__
        if isinstance(raw_builtins, dict):
            safe_builtins = {
                k: v for k, v in raw_builtins.items() if k not in _DANGEROUS_BUILTINS
            }
            real_import = raw_builtins["__import__"]
        else:
            safe_builtins = {
                k: v for k, v in vars(raw_builtins).items()
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
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            sys.stdout = old_stdout
            if old_builtins is not None:
                self.namespace["__builtins__"] = old_builtins
            else:
                self.namespace.pop("__builtins__", None)

        output = buf.getvalue()
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"

        return (output, error)

    # ── Measurement (called by the harness after each turn) ────────────

    def measure(self) -> dict[str, Any]:
        """Run the current simulate on all frames.  Returns metrics dict."""
        if self._simulate is None:
            return {
                "accuracy": None,
                "wrong_cells": 0,
                "frames_correct": 0,
                "frames_total": len(self._grids) - 1,
                "regressions": 0,
            }

        result = run_check(
            self._simulate, self._grids, self._actions, verbose=False
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