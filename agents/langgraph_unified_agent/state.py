"""LangGraph state schema for the unified agent."""

from typing import TypedDict

from arcengine import FrameData, GameAction


class UnifiedState(TypedDict, total=False):
    """State carried through the unified LangGraph workflow."""

    available_actions: list[int]
    frame_index: int
    observation: str
    mechanics: list[str]
    mechanics_summary: str
    tactical: list[str]
    tactical_summary: str
    plan: str
    history: list[str]
    action: GameAction | None
    node_path: list[str]
    last_action_id: int
    prev_grid: list[list[int]] | None
    prev_levels_completed: int | None
    expectation: str
    frames: list[FrameData]
    needs_reflection: bool
