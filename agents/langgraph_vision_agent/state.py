"""LangGraph state schema for the vision agent."""

from typing import TypedDict

from arcengine import FrameData, GameAction


class GameState(TypedDict, total=False):
    """State carried through the LangGraph vision-agent workflow."""

    latest_frame: FrameData
    available_actions: list[int]
    frame_index: int
    observation: str
    mechanics: list[str]
    mechanics_summary: str
    tactical: list[str]
    tactical_summary: str
    plan: str
    history: list[str]
    uncertain_about: str | None
    needs_reflection: bool
    action: GameAction | None
    node_path: list[str]
    last_action_id: int
    prev_grid: list[list[int]] | None
    prev_levels_completed: int | None
    expectation: str
    prev_frame: FrameData | None
