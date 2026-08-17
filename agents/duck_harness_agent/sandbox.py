"""Sandboxed code execution with bidirectional IPC for Duck Harness agent.

Uses multiprocessing.Process + Pipe for parent-child communication so that
`action()` calls inside sandbox code propagate to the parent process and
invoke the step_env_callback, with the result fed back to the child.

After each ``action()`` call the child refreshes its ``objects``,
``adjacency``, ``history``, ``current_frame``, ``previous_frame``,
``valid_actions``, and ``last_action_result`` globals from the parent's
state response, so subsequent sandbox code sees the updated world state.
"""

from __future__ import annotations

import logging
import multiprocessing
import multiprocessing.connection
import re
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Callable

logger = logging.getLogger(__name__)

# ── Reuse from vision agent sandbox ──────────────────────────────────────

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

_MAX_OUTPUT_CHARS = 4096  # ≈ tool_output_tokens * 4 (1024 * 4)


# ── Result type ─────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    """Result of executing code inside the sandbox."""

    output: str = ""
    action_taken: int | None = None
    error: str | None = None


# ── Child process entry point ───────────────────────────────────────────

def _child_process(  # noqa: C901 – isolated subprocess, complexity acceptable
    code: str,
    objects: tuple[dict, ...],
    adjacency: frozenset[tuple[int, int]],
    history: list[dict],
    current_frame: list[list[int]],
    previous_frame: list[list[int]],
    valid_actions: list[int],
    last_action_result: str,
    pipe_conn: multiprocessing.connection.Connection,  # type: ignore[attr-defined]
) -> None:
    """Execute *code* in a restricted namespace; communicate via *pipe_conn*.

    IPC protocol (child → parent → child):
      1. Child calls ``action(action_id)`` or ``action(action_id, **kwargs)``
         → sends ``{"type": "action", "action_id": action_id,
         "action_data": {...} | None}`` on pipe.
      2. Parent receives, calls ``step_env_callback(action_id, action_data)``,
         sends back a state dict containing refreshed ``objects``,
         ``adjacency``, ``history``, ``grid``, ``last_action_result``,
         ``valid_actions``.
      3. Child receives response and refreshes all sandbox globals so
         subsequent code sees the updated world state.
      4. On completion (or error), child sends ``{"type": "result", "output":
         ..., "action_taken": ..., "error": ...}`` and exits.
    """
    # Capture the real __import__ before we filter builtins
    _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[attr-defined]

    safe_builtins = {
        k: v for k, v in __builtins__.items()  # type: ignore[attr-defined]
        if k not in _DANGEROUS_BUILTINS
    }

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        if top_level not in _ALLOWED_IMPORTS:
            allowed = ", ".join(sorted(_ALLOWED_IMPORTS))
            raise ImportError(
                f"import of '{name}' is not allowed. "
                f"Allowed modules: {allowed}"
            )
        return _real_import(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = _safe_import

    # Mutable container so the nested action() can write back the action_id
    _action_taken: list[int | None] = [None]

    def action(action_id: int, **kwargs: int) -> dict:  # type: ignore[override]
        """Called by sandbox code to take an action in the environment.

        Supports three calling conventions:
          - ``action(action_id)`` — simple action, action_data is None
          - ``action(action_id, x=30, y=40)`` — complex action with kwargs
          - ``action({"id": 6, "x": 30, "y": 40})`` — dict form

        Sends the action to the parent, waits for a state response, refreshes
        all sandbox globals from the response, and returns the state dict.
        """
        # Dict form: action({"id": 6, "x": 30, "y": 40})
        if isinstance(action_id, dict):
            action_data = {k: v for k, v in action_id.items() if k != "id"}
            action_id = action_id["id"]  # type: ignore[assignment]

        # Kwargs form: action(6, x=30, y=40)
        elif kwargs:
            action_data = dict(kwargs)
        else:
            action_data = None

        _action_taken[0] = action_id  # type: ignore[assignment]
        pipe_conn.send({
            "type": "action",
            "action_id": action_id,
            "action_data": action_data,
        })
        state_response = pipe_conn.recv()

        # ── Refresh sandbox globals from state response ──────────────
        if isinstance(state_response, dict) and "objects" in state_response:
            namespace["objects"] = state_response["objects"]
            namespace["adjacency"] = state_response["adjacency"]
            namespace["history"] = state_response["history"]
            # previous_frame becomes what current_frame was before this action
            namespace["previous_frame"] = namespace["current_frame"]
            namespace["current_frame"] = state_response.get(
                "grid", namespace["current_frame"]
            )
            namespace["valid_actions"] = state_response.get(
                "valid_actions", namespace["valid_actions"]
            )
            namespace["last_action_result"] = state_response.get(
                "last_action_result", namespace["last_action_result"]
            )

        return state_response

    namespace: dict = {
        "__builtins__": safe_builtins,
        "objects": objects,
        "adjacency": adjacency,
        "history": history,
        "current_frame": current_frame,
        "previous_frame": previous_frame,
        "valid_actions": valid_actions,
        "last_action_result": last_action_result,
        "action": action,
    }

    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    error_msg: str | None = None
    try:
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, namespace)  # noqa: S102
    except Exception as exc:
        error_msg = str(exc)
    finally:
        sys.stdout = old_stdout

    output = buf.getvalue()
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "... (truncated)"

    pipe_conn.send({
        "type": "result",
        "output": output,
        "action_taken": _action_taken[0],
        "error": error_msg,
    })
    pipe_conn.close()


# ── Parent-side API ──────────────────────────────────────────────────────

class DuckSandbox:
    """Run untrusted code in a subprocess with bidirectional IPC.

    When sandbox code calls ``action(action_id)`` or
    ``action(action_id, x=..., y=...)``, the parent's
    ``step_env_callback`` is invoked with the action_id and optional
    action_data dict, and the result is fed back to the child process.
    """

    def __init__(
        self,
        step_env_callback: Callable[[int, dict | None], dict],
        timeout: float = 30.0,
    ) -> None:
        self.step_env_callback = step_env_callback
        self.timeout = timeout

    def run(
        self,
        code: str,
        objects: tuple[dict, ...],
        adjacency: frozenset[tuple[int, int]],
        history: list[dict],
        current_frame: list[list[int]] | None = None,
        previous_frame: list[list[int]] | None = None,
        valid_actions: list[int] | None = None,
        last_action_result: str = "",
    ) -> SandboxResult:
        """Execute *code* in a sandboxed subprocess.

        Returns a ``SandboxResult`` with captured output, the action_id
        if ``action()`` was called, or an error message on failure.
        """
        # Reject dunder patterns before spawning a process
        if _DUNDER_PATTERN.search(code):
            return SandboxResult(error="Error: dunder attributes are not allowed")

        # Default mutable containers — empty but valid
        if current_frame is None:
            current_frame = []
        if previous_frame is None:
            previous_frame = []
        if valid_actions is None:
            valid_actions = []

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        proc = multiprocessing.Process(
            target=_child_process,
            args=(
                code, objects, adjacency, history,
                current_frame, previous_frame, valid_actions,
                last_action_result, child_conn,
            ),
            daemon=True,
        )
        proc.start()
        child_conn.close()  # Parent doesn't need the child end

        action_taken: int | None = None
        try:
            # Read messages from the child until it sends a result or times out
            while True:
                # Use a per-message timeout slightly larger than the overall
                # timeout so we still honour the global deadline.
                if not parent_conn.poll(timeout=self.timeout):
                    # Timeout waiting for a message
                    proc.terminate()
                    proc.join(timeout=2.0)
                    logger.warning(f"sandbox: timed out after {self.timeout}s")
                    return SandboxResult(error="Error: sandbox timed out")

                msg = parent_conn.recv()

                if msg["type"] == "action":
                    # Child called action() — invoke callback and send state back
                    action_id = msg["action_id"]
                    action_data = msg.get("action_data")
                    action_taken = action_id
                    try:
                        state_response = self.step_env_callback(action_id, action_data)
                    except Exception as exc:
                        state_response = {"type": "state", "error": str(exc)}
                    parent_conn.send(state_response)

                elif msg["type"] == "result":
                    # Child finished — collect and return
                    output = msg.get("output", "")
                    child_action = msg.get("action_taken")
                    error = msg.get("error")
                    # Prefer the action we observed via callback, fall back
                    # to what the child reports
                    final_action = action_taken if action_taken is not None else child_action
                    return SandboxResult(
                        output=output,
                        action_taken=final_action,
                        error=error,
                    )

                else:
                    # Unknown message type — ignore (defensive)
                    continue

        except EOFError:
            # Pipe closed unexpectedly
            proc.terminate()
            proc.join(timeout=2.0)
            logger.warning("sandbox: process closed unexpectedly")
            return SandboxResult(error="Error: sandbox process closed unexpectedly")

        except Exception as exc:
            proc.terminate()
            proc.join(timeout=2.0)
            logger.warning(f"sandbox: error: {exc}")
            return SandboxResult(error=f"Error: {exc}")

        finally:
            # Ensure process is cleaned up
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2.0)
            parent_conn.close()