"""Tests for FrameContext dataclass."""
from __future__ import annotations

from dataclasses import fields, is_dataclass

from planning.frame_context import FrameContext


class TestFrameContext:
    """Test the FrameContext dataclass."""

    def test_is_dataclass(self) -> None:
        """FrameContext should be a dataclass."""
        assert is_dataclass(FrameContext)

    def test_frozen(self) -> None:
        """FrameContext should be frozen (immutable)."""
        assert FrameContext.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_has_eight_fields(self) -> None:
        """FrameContext should have exactly 8 fields."""
        fc_fields = {f.name for f in fields(FrameContext)}
        expected = {
            "scene",
            "ctx",
            "residual",
            "observed_transition",
            "unknowns",
            "diverged",
            "spec",
            "next_spec",
        }
        assert fc_fields == expected, f"Field mismatch: {fc_fields ^ expected}"

    def test_next_spec_defaults_to_none(self) -> None:
        """next_spec should default to None."""
        fc_field = next(f for f in fields(FrameContext) if f.name == "next_spec")
        assert fc_field.default is None
