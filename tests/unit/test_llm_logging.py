"""Unit tests for agents/templates/llm_logging.py — LlmCallLogger and wrap_llm_call."""

from __future__ import annotations

import json
import os

import pytest

from agents.llm_client import ChatResponse
from agents.recorder import LLM_LOG_SUFFIX, Recorder
from agents.templates.llm_logging import (
    MAX_CONTENT_CHARS,
    LlmCallLogger,
    LLMTruncationError,
    wrap_llm_call,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(
    tmp_path, guid: str = "test-guid-1234", frame_index: int = 0
) -> LlmCallLogger:
    """Create an LlmCallLogger writing to a JSONL file under *tmp_path*."""
    path = str(tmp_path / "test.llm.jsonl")
    return LlmCallLogger(
        guid=guid,
        path=path,
        frame_indexer=lambda fi=frame_index: fi,
    )


def _read_jsonl(path: str) -> list[dict]:
    """Read all JSONL lines from *path* and return parsed dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ok_llm(messages: list[dict[str, str]], *, thinking=None, max_tokens=None) -> str:
    return "ok"


def _hello_llm(messages: list[dict[str, str]], *, thinking=None, max_tokens=None) -> str:
    return "hello world"


def _four_llm(messages: list[dict[str, str]], *, thinking=None, max_tokens=None) -> str:
    return "4"


def _raising_llm(messages: list[dict[str, str]], *, thinking=None, max_tokens=None) -> str:
    raise ValueError("bad model")


def _truncated_llm(messages: list[dict[str, str]], *, thinking=None, max_tokens=None) -> ChatResponse:
    return ChatResponse(content="partial response...", finish_reason="length")


# ===========================================================================
# TestLlmCallLogger
# ===========================================================================


@pytest.mark.unit
class TestLlmCallLogger:
    """Tests for LlmCallLogger and wrap_llm_call."""

    # -------------------------------------------------------------------
    # 1. Successful call
    # -------------------------------------------------------------------

    def test_successful_call_records_event(self, tmp_path: pytest.Path) -> None:
        """Wrapped callable returns raw string; JSONL has one line with
        ok=true, response_raw matching, kind/trigger correct, etc."""
        logger = _make_logger(tmp_path, guid="abc", frame_index=7)
        fake_llm = _hello_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        messages = [{"role": "user", "content": "go"}]
        result = wrapped(messages)

        assert result == "hello world"

        events = _read_jsonl(logger.path)
        assert len(events) == 1
        ev = events[0]
        assert ev["ok"] is True
        assert ev["response_raw"] == "hello world"
        assert ev["kind"] == "planner"
        assert ev["trigger"] == "planner"  # defaults to kind when logger.trigger is ""
        assert ev["latency_ms"] >= 0
        assert ev["seq"] == 1
        assert ev["frame_index"] == 7
        assert ev["guid"] == "abc"
        assert ev["truncated"] is False
        assert ev["messages"] == messages
        assert ev["error"] is None

    def test_trigger_from_logger_attribute(self, tmp_path: pytest.Path) -> None:
        """When logger.trigger is set, it is used instead of kind."""
        logger = _make_logger(tmp_path, frame_index=0)
        logger.trigger = "explore"
        fake_llm = _ok_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped([{"role": "user", "content": "go"}])

        events = _read_jsonl(logger.path)
        assert events[0]["trigger"] == "explore"

    # -------------------------------------------------------------------
    # 2. Raising call
    # -------------------------------------------------------------------

    def test_raising_call_records_error_and_reraises(
        self, tmp_path: pytest.Path
    ) -> None:
        """A fake llm_call that raises ValueError; wrapped re-raises;
        JSONL line has ok=false, error containing ValueError,
        response_raw empty string."""
        logger = _make_logger(tmp_path)
        fake_llm = _raising_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        with pytest.raises(ValueError, match="bad model"):
            wrapped([{"role": "user", "content": "go"}])

        events = _read_jsonl(logger.path)
        assert len(events) == 1
        ev = events[0]
        assert ev["ok"] is False
        assert "ValueError" in ev["error"]
        assert ev["response_raw"] == ""
        assert ev["seq"] == 1

    # -------------------------------------------------------------------
    # 3. Truncation
    # -------------------------------------------------------------------

    def test_truncation_of_long_content(self, tmp_path: pytest.Path) -> None:
        """A message with content exceeding MAX_CONTENT_CHARS; JSONL truncated=true,
        stored content is truncated with marker suffix."""
        logger = _make_logger(tmp_path)
        long_content = "x" * (MAX_CONTENT_CHARS + 5_000)
        messages = [{"role": "user", "content": long_content}]
        fake_llm = _ok_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped(messages)

        events = _read_jsonl(logger.path)
        ev = events[0]
        assert ev["truncated"] is True

        stored_content = ev["messages"][0]["content"]
        assert stored_content[:MAX_CONTENT_CHARS] == "x" * MAX_CONTENT_CHARS
        assert "[...truncated " in stored_content
        assert len(stored_content) > MAX_CONTENT_CHARS

    def test_short_content_not_truncated(self, tmp_path: pytest.Path) -> None:
        """Content under the limit is stored verbatim, truncated=false."""
        logger = _make_logger(tmp_path)
        short_content = "hello"
        messages = [{"role": "user", "content": short_content}]
        fake_llm = _ok_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped(messages)

        events = _read_jsonl(logger.path)
        ev = events[0]
        assert ev["truncated"] is False
        assert ev["messages"][0]["content"] == short_content

    def test_finish_reason_length_marks_truncated(self, tmp_path: pytest.Path) -> None:
        """When finish_reason is 'length', the event is marked truncated."""
        logger = _make_logger(tmp_path)
        fake_llm = _truncated_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        result = wrapped([{"role": "user", "content": "go"}])

        assert isinstance(result, ChatResponse)
        assert result.finish_reason == "length"

        events = _read_jsonl(logger.path)
        ev = events[0]
        assert ev["truncated"] is True
        assert ev["finish_reason"] == "length"

    def test_strict_mode_raises_on_truncation(self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_STRICT_MODE=true causes LLMTruncationError on finish_reason='length'."""
        monkeypatch.setenv("LLM_STRICT_MODE", "true")
        logger = _make_logger(tmp_path)
        fake_llm = _truncated_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        with pytest.raises(LLMTruncationError, match="truncated"):
            wrapped([{"role": "user", "content": "go"}])

        events = _read_jsonl(logger.path)
        assert events[0]["truncated"] is True
        assert events[0]["finish_reason"] == "length"

    def test_strict_mode_off_does_not_raise(self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without LLM_STRICT_MODE, truncation is logged but no exception raised."""
        monkeypatch.delenv("LLM_STRICT_MODE", raising=False)
        logger = _make_logger(tmp_path)
        fake_llm = _truncated_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        result = wrapped([{"role": "user", "content": "go"}])
        assert isinstance(result, ChatResponse)
        assert result.finish_reason == "length"

    # -------------------------------------------------------------------
    # 4. Monotonic seq
    # -------------------------------------------------------------------

    def test_seq_monotonically_increments(self, tmp_path: pytest.Path) -> None:
        """A second call gets seq == 2."""
        logger = _make_logger(tmp_path)
        fake_llm = _ok_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped([{"role": "user", "content": "first"}])
        wrapped([{"role": "user", "content": "second"}])

        events = _read_jsonl(logger.path)
        assert len(events) == 2
        assert events[0]["seq"] == 1
        assert events[1]["seq"] == 2

    # -------------------------------------------------------------------
    # Additional edge cases
    # -------------------------------------------------------------------

    def test_messages_echoed(self, tmp_path: pytest.Path) -> None:
        """Messages passed to the wrapper are recorded in the event."""
        logger = _make_logger(tmp_path)
        messages = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        fake_llm = _four_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped(messages)

        events = _read_jsonl(logger.path)
        assert events[0]["messages"] == messages

    def test_frame_index_changes_between_calls(self, tmp_path: pytest.Path) -> None:
        """Frame indexer returning different values per call is reflected."""
        call_count = 0

        def frame_indexer():
            nonlocal call_count
            val = call_count * 10
            call_count += 1
            return val

        path = str(tmp_path / "test.llm.jsonl")
        logger = LlmCallLogger(
            guid="g", path=path, frame_indexer=frame_indexer
        )
        fake_llm = _ok_llm
        wrapped = wrap_llm_call(fake_llm, logger, kind="planner")

        wrapped([{"role": "user", "content": "a"}])
        wrapped([{"role": "user", "content": "b"}])

        events = _read_jsonl(path)
        assert events[0]["frame_index"] == 0
        assert events[1]["frame_index"] == 10


# ===========================================================================
# TestRecorderLlmLogPath
# ===========================================================================


@pytest.mark.unit
class TestRecorderLlmLogPath:
    """Tests for Recorder.llm_log_path() producing the .llm.jsonl sibling."""

    def test_llm_log_path_sibling_of_recording(
        self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Recorder.llm_log_path() returns a path ending in .llm.jsonl
        that is a sibling of the *.recording.jsonl filename."""
        monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
        recorder = Recorder(prefix="test-game", guid="abcd-1234")
        log_path = recorder.llm_log_path()

        assert log_path.endswith(LLM_LOG_SUFFIX)
        assert log_path.endswith(".llm.jsonl")
        # The recording filename and llm log should share the same directory
        # and base name, differing only in suffix.
        recording_basename = os.path.basename(recorder.filename)
        log_basename = os.path.basename(log_path)
        assert recording_basename.replace(".recording.jsonl", ".llm.jsonl") == log_basename

    def test_llm_log_path_with_custom_filename(
        self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When constructed with an explicit filename, llm_log_path replaces suffix."""
        filename = "mygame.myagent.50.guid123.recording.jsonl"
        monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
        recorder = Recorder(prefix="x", filename=filename)
        log_path = recorder.llm_log_path()

        assert log_path.endswith("mygame.myagent.50.guid123.llm.jsonl")
        assert not log_path.endswith(".recording.jsonl")