"""Regression: sandbox worker thread must inherit the caller's contextvars.

Incident a21a2571 (ls20, 2026-09-04): after 24e7dce moved sandbox exec into a
worker thread, every log line emitted from sandbox code (ACTION counts,
workflow ``phase=`` lines, check results) silently vanished from the
``.logs.log`` sidecar. Root cause: per-recording log routing relies on the
``_active_log_path`` ContextVar (agents/agent.py), which is thread-local — the
worker got a fresh context, so ``_RecordingLogFilter`` saw an empty value and
dropped the records. ``run_code()`` now runs the worker inside a copy of the
caller's context.
"""

from __future__ import annotations

import inspect
import logging

from agents.agent import _active_log_path, _RecordingLogFilter
from agents.simulator_agent.sandbox import SimulatorSandbox


class TestWorkerContextPropagation:
    def test_log_from_action_callback_reaches_recording_handler(self, tmp_path):
        """A log line emitted inside run_code (worker thread) through the
        agent's recording filter must land in the sidecar file. Without
        context propagation the filter sees an empty contextvar and the
        file stays empty."""
        log_path = tmp_path / "run.logs.log"
        handler = logging.FileHandler(log_path, mode="w")
        handler.addFilter(_RecordingLogFilter(str(log_path)))
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.INFO)
        token = _active_log_path.set(str(log_path))
        try:

            def cb(action_id: int, action_data: object) -> dict[str, object]:
                logging.getLogger().info(
                    "ls20-9607627b - ACTION1: count 2, levels completed 0"
                )
                return {
                    "objects": (),
                    "adjacency": frozenset(),
                    "history": [],
                    "grid": [[0] * 8 for _ in range(8)],
                    "valid_actions": [1],
                    "last_action_result": {},
                }

            sb = SimulatorSandbox(step_env_callback=cb, timeout=5.0)
            sb._current_frame = [[0] * 8 for _ in range(8)]
            sb.namespace["current_frame"] = sb._current_frame
            sb.namespace["previous_frame"] = None
            output, error, taken = sb.run_code("action(1)\n")
            assert error is None
            assert taken == 1
        finally:
            _active_log_path.reset(token)
            root.removeHandler(handler)
            root.setLevel(old_level)
            handler.close()

        content = log_path.read_text()
        assert "ACTION1: count 2" in content

    def test_log_dropped_without_context_set(self, tmp_path):
        """Control: with an empty contextvar the same record is dropped by
        the filter — proves the assertion above discriminates (a handler
        without the filter would pass either way)."""
        log_path = tmp_path / "run.logs.log"
        handler = logging.FileHandler(log_path, mode="w")
        handler.addFilter(_RecordingLogFilter(str(log_path)))
        root = logging.getLogger()
        root.addHandler(handler)
        old_level = root.level
        root.setLevel(logging.INFO)
        try:
            # No _active_log_path.set — simulates the pre-fix worker context.
            logging.getLogger().info("orphan record")
        finally:
            root.removeHandler(handler)
            root.setLevel(old_level)
            handler.close()

        assert log_path.read_text() == ""

    def test_run_code_propagates_caller_context(self):
        """Contract pin: run_code must run the worker inside a copy of the
        caller's context (guards against accidental removal of the fix)."""
        source = inspect.getsource(SimulatorSandbox.run_code)
        assert "copy_context" in source
        assert "worker_ctx.run" in source