"""Unit tests for truncation detection and strict mode (T1 + T2).

T1: max_tokens=2048 for grouping wrap_llm_call.
T2: ChatResponse, truncation detection, LLM_STRICT_MODE, LlmGroupingEngine.adjudicate().
"""

from __future__ import annotations

import inspect
import json

import pytest

from agents.llm_client import ChatResponse
from agents.templates.llm_logging import (
    LlmCallLogger,
    LLMTruncationError,
    _is_strict_mode,
    wrap_llm_call,
)
from grouping.features import EntityFeature
from grouping.llm_engine import LlmGroupingEngine
from grouping.proposal import GroupProposal
from perception.registry import ObjectRegistry, Observation, Track

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logger(
    tmp_path, guid: str = "test-guid-t2", frame_index: int = 0
) -> LlmCallLogger:
    path = str(tmp_path / "test_t2.llm.jsonl")
    return LlmCallLogger(
        guid=guid, path=path, frame_indexer=lambda fi=frame_index: fi,
    )


def _make_feature(
    entity_id: int = 0,
    *,
    ever_moves: bool = True,
    displacements: list[tuple[int, int] | None] | None = None,
    action_displacements: dict[int, list[tuple[int, int]]] | None = None,
    role: str | None = None,
) -> EntityFeature:
    return EntityFeature(
        entity_id=entity_id,
        role=role,
        composition="singleton",
        n_members=1,
        n_observations=5,
        positions=[(10.0, 10.0)],
        bboxes=[(5, 5, 15, 15)],
        displacements=displacements or [],
        action_displacements=action_displacements or {},
        frame_displacements={},
        ever_moves=ever_moves,
        shape_keys=[frozenset()],
        shape_key_stable=True,
        unique_shape_keys=[frozenset()],
        sizes=[100],
        size_range=(100, 100),
        cell_counts=[100],
    )


def _make_registry(*alive_ids: int) -> ObjectRegistry:
    reg = ObjectRegistry()
    for tid in alive_ids:
        obs = Observation(
            frame_idx=0, color=1, size=10, centroid=(0.0, 0.0),
            bbox=(0, 0, 1, 1), shape_key=frozenset(), cells=frozenset(),
            match_rule="new", displacement=None, structural=False,
        )
        reg.tracks[tid] = Track(id=tid, color=1, observations=[obs])
    return reg


# ===========================================================================
# T1: max_tokens for grouping wrap_llm_call
# ===========================================================================


@pytest.mark.unit
class TestGroupingMaxTokens:
    """T1: Verify grouping uses max_tokens=2048."""

    def test_grouping_max_tokens_2048(self) -> None:
        """The wrap_llm_call for grouping uses max_tokens=2048."""
        from agents.templates import llm_curiosity_agent as mod

        source = inspect.getsource(mod)
        # Verify the source contains the grouping wrap_llm_call with max_tokens=2048
        assert "max_tokens=2048" in source, (
            "Expected max_tokens=2048 in grouping wrap_llm_call"
        )


# ===========================================================================
# T2: ChatResponse, truncation detection, strict mode, LlmGroupingEngine
# ===========================================================================


@pytest.mark.unit
class TestChatResponse:
    """T2: ChatResponse dataclass tests."""

    def test_chat_response_fields(self) -> None:
        """ChatResponse has content and finish_reason fields."""
        resp = ChatResponse(content="hello", finish_reason="stop")
        assert resp.content == "hello"
        assert resp.finish_reason == "stop"

    def test_chat_response_str(self) -> None:
        """str(ChatResponse) returns content for backward compatibility."""
        resp = ChatResponse(content="world", finish_reason="stop")
        assert str(resp) == "world"

    def test_chat_response_frozen(self) -> None:
        """ChatResponse is frozen (immutable)."""
        resp = ChatResponse(content="x", finish_reason="stop")
        with pytest.raises(AttributeError):
            resp.content = "y"  # type: ignore[misc]

    def test_chat_response_finish_reason_length(self) -> None:
        """finish_reason='length' signals truncation."""
        resp = ChatResponse(content="partial...", finish_reason="length")
        assert resp.finish_reason == "length"


@pytest.mark.unit
class TestTruncationDetection:
    """T2: wrap_llm_call truncation detection."""

    def test_finish_reason_length_marks_truncated_in_event(
        self, tmp_path: pytest.Path
    ) -> None:
        """When finish_reason='length', the JSONL event has truncated=true."""
        logger = _make_logger(tmp_path)

        def truncated_llm(messages, *, thinking=None, max_tokens=None):
            return ChatResponse(content="partial response", finish_reason="length")

        wrapped = wrap_llm_call(truncated_llm, logger, kind="planner")
        result = wrapped([{"role": "user", "content": "go"}])

        assert isinstance(result, ChatResponse)
        assert result.finish_reason == "length"

        import json as _json
        with open(logger.path, encoding="utf-8") as f:
            ev = _json.loads(f.readline())
        assert ev["truncated"] is True
        assert ev["finish_reason"] == "length"

    def test_finish_reason_stop_not_truncated(
        self, tmp_path: pytest.Path
    ) -> None:
        """When finish_reason='stop', the event is not truncated."""
        logger = _make_logger(tmp_path)

        def normal_llm(messages, *, thinking=None, max_tokens=None):
            return ChatResponse(content="hello", finish_reason="stop")

        wrapped = wrap_llm_call(normal_llm, logger, kind="planner")
        result = wrapped([{"role": "user", "content": "go"}])

        assert isinstance(result, ChatResponse)
        assert result.finish_reason == "stop"

        import json as _json
        with open(logger.path, encoding="utf-8") as f:
            ev = _json.loads(f.readline())
        assert ev["truncated"] is False


@pytest.mark.unit
class TestStrictMode:
    """T2: LLM_STRICT_MODE tests."""

    def test_strict_mode_env_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_STRICT_MODE=true enables strict mode."""
        monkeypatch.setenv("LLM_STRICT_MODE", "true")
        assert _is_strict_mode() is True

    def test_strict_mode_env_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_STRICT_MODE=1 enables strict mode."""
        monkeypatch.setenv("LLM_STRICT_MODE", "1")
        assert _is_strict_mode() is True

    def test_strict_mode_env_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLM_STRICT_MODE=false disables strict mode."""
        monkeypatch.delenv("LLM_STRICT_MODE", raising=False)
        assert _is_strict_mode() is False

    def test_strict_mode_env_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty LLM_STRICT_MODE disables strict mode."""
        monkeypatch.setenv("LLM_STRICT_MODE", "")
        assert _is_strict_mode() is False

    def test_strict_mode_raises_on_truncation(
        self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM_STRICT_MODE=true causes LLMTruncationError on finish_reason='length'."""
        monkeypatch.setenv("LLM_STRICT_MODE", "true")
        logger = _make_logger(tmp_path)

        def truncated_llm(messages, *, thinking=None, max_tokens=None):
            return ChatResponse(content="partial", finish_reason="length")

        wrapped = wrap_llm_call(truncated_llm, logger, kind="planner")

        with pytest.raises(LLMTruncationError, match="truncated"):
            wrapped([{"role": "user", "content": "go"}])

    def test_strict_mode_off_logs_warning(
        self, tmp_path: pytest.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without strict mode, truncation logs warning but no exception."""
        monkeypatch.delenv("LLM_STRICT_MODE", raising=False)
        logger = _make_logger(tmp_path)

        def truncated_llm(messages, *, thinking=None, max_tokens=None):
            return ChatResponse(content="partial", finish_reason="length")

        wrapped = wrap_llm_call(truncated_llm, logger, kind="planner")
        result = wrapped([{"role": "user", "content": "go"}])

        # No exception raised, result returned normally
        assert isinstance(result, ChatResponse)
        assert result.finish_reason == "length"


@pytest.mark.unit
class TestLlmGroupingEngineTruncation:
    """T2: LlmGroupingEngine.adjudicate() truncation handling."""

    ZERO_GRID: list[list[int]] = [[0] * 64 for _ in range(64)]

    def _make_engine_with_llm(self, llm_response, *, vision: bool = False):
        """Create LlmGroupingEngine with a mock LLM that returns the given response."""

        def mock_llm(messages):
            return llm_response

        return LlmGroupingEngine(
            llm_call=mock_llm, vision=vision, image_scale=4,
        )

    def test_adjudicate_detects_truncated_response_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When LLM returns finish_reason='length', adjudicate falls back."""
        monkeypatch.delenv("LLM_STRICT_MODE", raising=False)

        engine = self._make_engine_with_llm(
            ChatResponse(content="truncated json", finish_reason="length"),
        )
        proposals = [
            GroupProposal(
                group_id=0, member_ids={0, 1},
                heuristic="adjacency", evidence={}, support=1,
            ),
        ]
        features = {0: _make_feature(0), 1: _make_feature(1)}

        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self.ZERO_GRID,
            curr_grid=self.ZERO_GRID,
            entities_data=[],
            proposals=proposals,
            confirmed_groups=[],
            features=features,
        )

        # Fallback: all proposals confirmed
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "confirm"
        assert verdicts[0].reason == "fallback"

    def test_adjudicate_strict_mode_raises_on_truncation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM_STRICT_MODE=true causes LLMTruncationError on truncated LLM response."""
        monkeypatch.setenv("LLM_STRICT_MODE", "true")

        engine = self._make_engine_with_llm(
            ChatResponse(content="truncated", finish_reason="length"),
        )
        proposals = [
            GroupProposal(
                group_id=0, member_ids={0, 1},
                heuristic="adjacency", evidence={}, support=1,
            ),
        ]
        features = {0: _make_feature(0), 1: _make_feature(1)}

        with pytest.raises(LLMTruncationError):
            engine.adjudicate(
                prev_grid=self.ZERO_GRID,
                curr_grid=self.ZERO_GRID,
                entities_data=[],
                proposals=proposals,
                confirmed_groups=[],
                features=features,
            )

    def test_adjudicate_strict_mode_raises_on_parse_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM_STRICT_MODE=true causes LLMTruncationError on unparseable response."""
        monkeypatch.setenv("LLM_STRICT_MODE", "true")

        engine = self._make_engine_with_llm(
            ChatResponse(content="not json at all", finish_reason="stop"),
        )
        proposals = [
            GroupProposal(
                group_id=0, member_ids={0, 1},
                heuristic="adjacency", evidence={}, support=1,
            ),
        ]
        features = {0: _make_feature(0), 1: _make_feature(1)}

        with pytest.raises(LLMTruncationError, match="parse failure"):
            engine.adjudicate(
                prev_grid=self.ZERO_GRID,
                curr_grid=self.ZERO_GRID,
                entities_data=[],
                proposals=proposals,
                confirmed_groups=[],
                features=features,
            )

    def test_adjudicate_normal_response_works(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal (non-truncated) ChatResponse with valid JSON works."""
        monkeypatch.delenv("LLM_STRICT_MODE", raising=False)

        response_json = json.dumps([{
            "proposal_id": 0,
            "verdict": "confirm",
            "relation": "merge",
            "members": [{"id": 0, "role": "core", "label": "A"}],
            "reason": "test",
        }])

        engine = self._make_engine_with_llm(
            ChatResponse(content=response_json, finish_reason="stop"),
        )
        proposals = [
            GroupProposal(
                group_id=0, member_ids={0, 1},
                heuristic="adjacency", evidence={}, support=1,
            ),
        ]
        features = {0: _make_feature(0), 1: _make_feature(1)}

        verdicts, _ = engine.adjudicate(
            prev_grid=self.ZERO_GRID,
            curr_grid=self.ZERO_GRID,
            entities_data=[],
            proposals=proposals,
            confirmed_groups=[],
            features=features,
        )

        assert len(verdicts) == 1
        assert verdicts[0].verdict == "confirm"
        assert verdicts[0].relation == "merge"