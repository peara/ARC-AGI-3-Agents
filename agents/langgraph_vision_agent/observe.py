"""Observation rendering and history writing for the LangGraph vision agent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from arcengine import FrameData

from vision.render import grid_to_image, image_to_base64, make_image_block

from .logging import log_node
from .services import AgentServices


def _is_empty_frame(frame: FrameData | None) -> bool:
    if frame is None or not frame.frame:
        return True
    return False


def render_observation(
    frame: FrameData, frame_index: int = 0, render_scale: int = 8
) -> str | list[dict[str, Any]]:
    """Render a frame grid into a multimodal image block."""
    if _is_empty_frame(frame):
        raise ValueError("Vision is mandatory but frame is empty")

    grid = frame.frame
    inner_grid = grid[0] if len(grid) == 1 else grid

    caption = f"Frame {frame_index}"

    img = grid_to_image(inner_grid, scale=render_scale)  # type: ignore[arg-type]
    b64 = image_to_base64(img)
    image_block = make_image_block(b64)
    text_block = {"type": "text", "text": caption}
    return [image_block, text_block]


def _count_changed_cells(
    grid_a: Sequence[Sequence[int]], grid_b: Sequence[Sequence[int]]
) -> int:
    changed = 0
    for row_a, row_b in zip(grid_a, grid_b):
        for cell_a, cell_b in zip(row_a, row_b):
            if cell_a != cell_b:
                changed += 1
    return changed


def _unwrap_grid(grid: Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]) -> Sequence[Sequence[int]]:
    if len(grid) == 1:
        first = grid[0]
        if first and isinstance(first[0], Sequence):
            return first  # type: ignore[return-value]
    return grid  # type: ignore[return-value]


def make_observe_node(services: AgentServices) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return the LangGraph observe node function for the vision agent."""

    def observe_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)
        frames_list = state.get("frames", [])
        if not frames_list:
            raise ValueError("frames is empty — cannot observe")
        frame: FrameData = frames_list[-1]
        prev_frame = frames_list[-2] if len(frames_list) >= 3 else None
        history: list[str] = list(state.get("history", []))
        prev_grid = state.get("prev_grid")
        render_scale = services.config.render_scale
        prev_levels_completed = state.get("prev_levels_completed")
        expectation = state.get("expectation", "none")

        is_first_frame = prev_grid is None
        grid = _unwrap_grid(frame.frame)

        if prev_frame is not None:
            prev_obs = render_observation(
                prev_frame, frame_index=frame_index - 1, render_scale=render_scale
            )
            curr_obs = render_observation(
                frame, frame_index=frame_index, render_scale=render_scale
            )
            caption = f"Action taken: {state.get('last_action_id')}. You expected: {expectation}"
            observation = prev_obs + [{"type": "text", "text": caption}] + curr_obs
        else:
            observation = render_observation(
                frame, frame_index=frame_index, render_scale=render_scale
            )

        grid_changed = False
        cells_changed = 0

        if not is_first_frame and prev_grid is not None:
            cells_changed = _count_changed_cells(prev_grid, grid)
            grid_changed = cells_changed > 0

            history_line = (
                f"frame {frame_index - 1}: "
                f"action={state.get('last_action_id')}"
            )
            history.append(history_line)
            max_history = services.config.max_history
            if len(history) > max_history:
                history = history[-max_history:]

        levels_completed = getattr(frame, "levels_completed", 0)
        level_changed = False
        if prev_levels_completed is None:
            state["prev_levels_completed"] = levels_completed
        elif prev_levels_completed != levels_completed:
            state["prev_levels_completed"] = levels_completed
            level_changed = True

        observe_signal = bool(is_first_frame or level_changed)
        plan_signal = state.get("needs_reflection", False)
        needs_reflection = observe_signal or plan_signal

        log_node(
            frame_index,
            "observe",
            grid_changed=grid_changed,
            cells_changed=cells_changed,
            level_changed=level_changed,
            needs_reflection=needs_reflection,
        )

        return {
            "observation": observation,
            "history": history,
            "prev_grid": [list(row) for row in grid],
            "prev_levels_completed": state["prev_levels_completed"],
            "needs_reflection": needs_reflection,
            "frame_index": frame_index + 1,
        }

    return observe_node
