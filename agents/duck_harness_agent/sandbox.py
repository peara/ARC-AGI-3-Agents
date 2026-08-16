"""Sandboxed code execution with bidirectional IPC for Duck Harness agent.

Uses multiprocessing.Process + Pipe for parent-child communication so that
`action()` calls inside sandbox code propagate to the parent process and
invoke the step_env_callback, with the result fed back to the child.

This is the highest-risk component — prove IPC works before building the
full agent around it.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.connection
import re
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Callable

# ── Reuse from vision agent sandbox ──────────────────────────────────────

_DANGEROUS_BUILTINS = frozenset({
    "__import__", "open", "compile", "eval", "exec",
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
    pipe_conn: multiprocessing.connection.Connection,  # type: ignore[attr-defined]
) -> None:
    """Execute *code* in a restricted namespace; communicate via *pipe_conn*.

    IPC protocol (child → parent → child):
      1. Child calls ``action(action_id)`` → sends ``{"type": "action",
         "action_id": action_id}`` on pipe.
      2. Parent receives, calls ``step_env_callback(action_id)``, sends back
         ``{"type": "state", ...}``.  (For this PoC the state payload is an
         acknowledgement dict; full state refresh comes in Task 7.)
      3. Child receives response and continues.
      4. On completion (or error), child sends ``{"type": "result", "output":
         ..., "action_taken": ..., "error": ...}`` and exits.
    """
    safe_builtins = {
        k: v for k, v in __builtins__.items()  # type: ignore[attr-defined]
        if k not in _DANGEROUS_BUILTINS
    }

    # Mutable container so the nested action() can write back the action_id
    _action_taken: list[int | None] = [None]

    def action(action_id: int) -> dict:
        """Called by sandbox code to take an action in the environment.

        Sends the action_id to the parent, waits for a state response,
        and returns the state dict so sandbox code can inspect it.
        """
        _action_taken[0] = action_id
        pipe_conn.send({"type": "action", "action_id": action_id})
        state_response = pipe_conn.recv()
        return state_response

    namespace: dict = {
        "__builtins__": safe_builtins,
        "objects": objects,
        "adjacency": adjacency,
        "history": history,
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

    When sandbox code calls ``action(action_id)``, the parent's
    ``step_env_callback`` is invoked and the result is fed back to the
    child process.
    """

    def __init__(self, step_env_callback: Callable[[int], dict], timeout: float = 30.0) -> None:
        self.step_env_callback = step_env_callback
        self.timeout = timeout

    def run(
        self,
        code: str,
        objects: tuple[dict, ...],
        adjacency: frozenset[tuple[int, int]],
        history: list[dict],
    ) -> SandboxResult:
        """Execute *code* in a sandboxed subprocess.

        Returns a ``SandboxResult`` with captured output, the action_id
        if ``action()`` was called, or an error message on failure.
        """
        # Reject dunder patterns before spawning a process
        if _DUNDER_PATTERN.search(code):
            return SandboxResult(error="Error: dunder attributes are not allowed")

        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
        proc = multiprocessing.Process(
            target=_child_process,
            args=(code, objects, adjacency, history, child_conn),
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
                    return SandboxResult(error="Error: sandbox timed out")

                msg = parent_conn.recv()

                if msg["type"] == "action":
                    # Child called action() — invoke callback and send state back
                    action_id = msg["action_id"]
                    action_taken = action_id
                    try:
                        state_response = self.step_env_callback(action_id)
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
            return SandboxResult(error="Error: sandbox process closed unexpectedly")

        except Exception as exc:
            proc.terminate()
            proc.join(timeout=2.0)
            return SandboxResult(error=f"Error: {exc}")

        finally:
            # Ensure process is cleaned up
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=2.0)
            parent_conn.close()