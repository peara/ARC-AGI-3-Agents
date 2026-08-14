"""Unified node merging reflector and planner into a single tool-using node.

Runs up to ``unified_max_tool_calls`` (default 12) iterations of a tool loop:
  1. Call LLM with ``inspect`` and ``decide`` tools
  2. If ``response.tool_calls`` contains ``inspect`` → sandbox execution, loop
  3. If ``response.tool_calls`` contains ``decide`` → parse and return dict
  4. If no tool calls → nudge (1 retry), then fallback to random
  5. If both ``inspect`` and ``decide`` in same response → process inspect only, loop

Always returns a dict — never a Command.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from arcengine import GameAction

from agents.langgraph_vision_agent.logging import log_node
from agents.langgraph_vision_agent.sandbox import (
    atoms_to_dicts,
    compute_adjacency,
    run_sandboxed,
)
from agents.langgraph_vision_agent.services import AgentServices

from ..config import UnifiedAgentConfig
from ..prompts import UNIFIED_SYSTEM_PROMPT
from ..tools import UNIFIED_TOOLS_V2, UNIFIED_TOOLS_V3

REFLECT_TOOL_NAME = "reflect"

logger = logging.getLogger(__name__)


def _frame_to_grid(frame_data: Any) -> np.ndarray | None:
    """Extract a 2D numpy grid from a FrameData object."""
    raw = frame_data.frame
    if len(raw) == 1 and raw[0] and isinstance(raw[0][0], list | Sequence):
        raw = raw[0]
    return np.array(raw, dtype=np.int64)


def _build_user_content(
    state: dict[str, Any],
    config: UnifiedAgentConfig,
    objects: tuple[dict[str, Any], ...],
    adjacency: frozenset[tuple[int, int]],
    force_reflect: bool,
    reflect_reason: str | None,
) -> list[dict[str, Any]]:
    """Build the user content blocks for the unified node prompt.

    Returns a list of content blocks (text + image_url) suitable for
    a multimodal user message.
    """
    frame_index: int = state.get("frame_index", 0)
    mechanics: list[str] = state.get("mechanics", [])
    mechanics_summary: str = state.get("mechanics_summary", "")
    tactical: list[str] = state.get("tactical", [])
    tactical_summary: str = state.get("tactical_summary", "")
    actions: list[str] = state.get("actions", [])
    goal: str = state.get("goal", "")
    goal_status: str = state.get("goal_status", "")
    history: list[str] = state.get("history", [])
    expectation: str = state.get("expectation", "")
    available_actions: list[int] = state.get("available_actions", [])

    mechanics_bullets = "\n".join(f"- {m}" for m in mechanics) if mechanics else "(none yet)"
    tactical_bullets = "\n".join(f"- {t}" for t in tactical) if tactical else "(none yet)"
    actions_bullets = "\n".join(f"- {a}" for a in actions) if actions else "(none yet)"
    recent_history = history[-5:] if history else []

    parts: list[str] = [
        f"Frame: {frame_index}",
        f"Available actions: {available_actions}",
        f"Last expectation: {expectation or '(none)'}",
        f"Recent actions: {recent_history}",
        "",
        f"## Current mechanics (max 10)\n{mechanics_bullets}",
        f"## Mechanics summary\n{mechanics_summary or '(none yet)'}",
        f"## Current tactical (max 10)\n{tactical_bullets}",
        f"## Tactical summary\n{tactical_summary or '(none yet)'}",
        f"## Actions (max {config.max_action_entries})\n{actions_bullets}",
        f"## Goal\n{goal or '(none)'}",
        f"## Goal status\n{goal_status or '(none)'}",
        "",
    ]

    if force_reflect:
        parts.append("## REFLECTION REQUIRED THIS FRAME")
        if reflect_reason:
            parts.append(f"Reason: {reflect_reason}")
        if config.use_v3_tools:
            parts.append("You MUST call reflect() this frame before decide().")
        else:
            parts.append("You MUST set reflect=true in your decide() call and include mechanics and tactical observations.")
        parts.append("")

    parts.append("Use inspect() to examine the state, then call decide() with your action.")

    text_prompt = "\n".join(parts)

    # Build multimodal message with observation images + text
    observation = state.get("observation", "")
    content_blocks: list[dict[str, Any]] = []

    if isinstance(observation, list):
        # observation is already a list of content blocks (images + text)
        content_blocks.extend(observation)
        content_blocks.append({"type": "text", "text": text_prompt})
    else:
        content_blocks.append({"type": "text", "text": f"Observation: {observation}\n{text_prompt}"})

    return content_blocks


def _deduplicate_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the first tool call per function name."""
    seen_names: set[str] = set()
    unique: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        if name not in seen_names:
            seen_names.add(name)
            unique.append(tc)
    return unique


def make_unified_node(services: AgentServices) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangGraph node function that plans with a tool loop.

    Merges reflection and planning into a single LLM call cycle using
    native OpenAI-compatible tool calling (inspect + decide).
    """
    # AgentServices.config is VisionAgentConfig at the type level, but the unified
    # agent stores a UnifiedAgentConfig there at runtime (see services.py).
    config: UnifiedAgentConfig = services.config  # pyright: ignore
    # Per-game history cache: list of {"action": int, "objects": tuple, "adjacency": frozenset}
    # Persists across turns (closure), dies when the graph dies.
    history_cache: list[dict[str, Any]] = []

    def unified_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)
        frames = state.get("frames", [])

        # ---- Grid extraction and atom segmentation ----
        grid: np.ndarray | None = None
        if frames:
            grid = _frame_to_grid(frames[-1])
        elif state.get("prev_grid") is not None:
            grid = np.array(state["prev_grid"], dtype=np.int64)

        objects: tuple[dict[str, Any], ...] = ()
        adjacency: frozenset[tuple[int, int]] = frozenset()
        if grid is not None:
            from optitrack.atoms import extract_atoms

            atoms = extract_atoms(grid)
            objects = atoms_to_dicts(atoms)
            adjacency = compute_adjacency(atoms)

        # ---- Read needs_reflection from state → set force_reflect ----
        needs_reflect_flag: bool = state.get("needs_reflection", False)
        force_reflect = needs_reflect_flag
        reflect_reason: str | None = state.get("reflect_reason", None)

        # ---- 5-repeat action guard: check action history ----
        action_history: list[str] = state.get("history", [])

        # ---- Build messages ----
        system_message = {"role": "system", "content": UNIFIED_SYSTEM_PROMPT}
        user_content = _build_user_content(
            state, config, objects, adjacency, force_reflect, reflect_reason,
        )
        messages = [system_message, {"role": "user", "content": user_content}]

        max_tool_calls = config.unified_max_tool_calls
        sandbox_timeout = config.unified_sandbox_timeout

        # Existing scene/mechanics/tactical for graceful degradation
        prev_mechanics: list[str] = list(state.get("mechanics", []))
        prev_mechanics_summary: str = state.get("mechanics_summary", "")
        prev_tactical: list[str] = list(state.get("tactical", []))
        prev_tactical_summary: str = state.get("tactical_summary", "")
        prev_actions: list[str] = list(state.get("actions", []))
        prev_goal: str = state.get("goal", "")
        prev_goal_status: str = state.get("goal_status", "")
        prev_reflect_reason: str = state.get("reflect_reason", "")

        available_actions: list[int] = state.get("available_actions", [1])
        if not available_actions:
            available_actions = [1]

        # ---- Tool loop ----
        nudge_count = 0
        tools = UNIFIED_TOOLS_V3 if config.use_v3_tools else UNIFIED_TOOLS_V2
        for call_idx in range(max_tool_calls):
            try:
                response = services.planner_call(
                    messages, tools=tools, tool_choice="auto",
                )
            except Exception:
                break  # LLM error → fallback below

            # Extract tool calls from response
            raw_tool_calls: list[dict[str, Any]] | None = None
            raw_content: str = ""
            if isinstance(response, str):
                raw_content = response
            else:
                raw_content = getattr(response, "content", "") or ""
                raw_tool_calls = getattr(response, "tool_calls", None)

            # ---- No tool calls → nudge or fallback ----
            if not raw_tool_calls:
                nudge_count += 1
                if nudge_count >= 2:
                    # Second nudge failed → fallback
                    break
                # First nudge: append nudge message and continue
                nudge_hint = "Please call inspect() or decide()." if not config.use_v3_tools else "Please call inspect(), reflect(), or decide()."
                messages = messages + [
                    {"role": "assistant", "content": raw_content},
                    {"role": "user", "content": nudge_hint},
                ]
                continue

            # ---- Deduplicate tool calls (keep first per function name) ----
            tool_calls_list = _deduplicate_tool_calls(raw_tool_calls)
            function_names = {tc["function"]["name"] for tc in tool_calls_list}

            # ---- V2: both inspect and decide → inspect only, loop ----
            # ---- V3: both inspect and decide → inspect only, loop ----
            #          both inspect and reflect → inspect only, loop
            if "inspect" in function_names and ("decide" in function_names or REFLECT_TOOL_NAME in function_names):
                # Keep only inspect, ignore decide/reflect this round
                tool_calls_list = [
                    tc for tc in tool_calls_list if tc["function"]["name"] == "inspect"
                ]
                function_names = {"inspect"}

            # ---- Process inspect tool call ----
            if "inspect" in function_names:
                tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == "inspect")
                try:
                    args = json.loads(tc["function"]["arguments"])
                    code = args.get("code", "")
                except (json.JSONDecodeError, AttributeError):
                    # Bad JSON → append error as tool result, continue
                    error_msg = tc.get("function", {}).get("arguments", "") if isinstance(tc, dict) else ""
                    messages = messages + [
                        {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                        {"role": "tool", "tool_call_id": tc["id"], "content": f"Error: could not parse inspect arguments. {error_msg}"},
                    ]
                    continue

                if not code:
                    messages = messages + [
                        {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                        {"role": "tool", "tool_call_id": tc["id"], "content": "Error: inspect() requires a 'code' argument."},
                    ]
                    continue

                # Run sandbox
                result = run_sandboxed(
                    code, objects, adjacency, list(history_cache), timeout=sandbox_timeout,
                )
                messages = messages + [
                    {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                    {"role": "tool", "tool_call_id": tc["id"], "content": result},
                ]
                continue

            # ============================================================
            # V3 3-tool dispatch: reflect and/or decide
            # ============================================================
            if config.use_v3_tools:
                # Determine order of reflect/decide calls
                has_reflect = REFLECT_TOOL_NAME in function_names
                has_decide = "decide" in function_names

                reflect_before_decide = False
                if has_reflect and has_decide:
                    # Check order: does reflect appear before decide?
                    for tc in tool_calls_list:
                        name = tc["function"]["name"]
                        if name == REFLECT_TOOL_NAME:
                            reflect_before_decide = True
                            break
                        if name == "decide":
                            break

                # --- Extract reflect data if present and relevant ---
                reflect_reason_out: str = prev_reflect_reason
                new_mechanics: list[str] = prev_mechanics
                new_mechanics_summary: str = prev_mechanics_summary
                new_tactical: list[str] = prev_tactical
                new_tactical_summary: str = prev_tactical_summary
                new_actions: list[str] = prev_actions
                new_goal: str = prev_goal
                new_goal_status: str = prev_goal_status

                if has_reflect and (not has_decide or reflect_before_decide):
                    # Process reflect: extract world model fields
                    reflect_tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == REFLECT_TOOL_NAME)
                    try:
                        r_args = json.loads(reflect_tc["function"]["arguments"])
                    except (json.JSONDecodeError, AttributeError):
                        # Bad reflect args → if no decide, continue loop
                        if not has_decide:
                            continue
                        r_args = {}

                    # Extract reflect fields
                    r_reason = r_args.get("reason", "")
                    r_goal = r_args.get("goal", "")
                    r_goal_status = r_args.get("goal_status", "")
                    r_actions = r_args.get("actions", [])
                    r_mechanics = r_args.get("mechanics", [])
                    r_mechanics_summary = r_args.get("mechanics_summary", "")
                    r_tactical = r_args.get("tactical", [])
                    r_tactical_summary = r_args.get("tactical_summary", "")

                    # Apply with max-entry limits
                    max_mechanics = config.max_mechanics
                    max_tactical = config.max_tactical
                    max_action_entries = config.max_action_entries
                    if r_reason:
                        reflect_reason_out = r_reason
                    if r_goal:
                        new_goal = r_goal
                    if r_goal_status:
                        new_goal_status = r_goal_status
                    if r_actions:
                        new_actions = r_actions[-max_action_entries:] if len(r_actions) > max_action_entries else r_actions
                    if r_mechanics:
                        new_mechanics = r_mechanics[-max_mechanics:] if len(r_mechanics) > max_mechanics else r_mechanics
                    if r_mechanics_summary:
                        new_mechanics_summary = r_mechanics_summary
                    if r_tactical:
                        new_tactical = r_tactical[-max_tactical:] if len(r_tactical) > max_tactical else r_tactical
                    if r_tactical_summary:
                        new_tactical_summary = r_tactical_summary

                # --- reflect-only: loop continues ---
                if has_reflect and not has_decide:
                    # Persist reflect data so decide() on the next iteration
                    # can use it instead of stale prev_* values
                    prev_mechanics = new_mechanics
                    prev_mechanics_summary = new_mechanics_summary
                    prev_tactical = new_tactical
                    prev_tactical_summary = new_tactical_summary
                    prev_actions = new_actions
                    prev_goal = new_goal
                    prev_goal_status = new_goal_status
                    prev_reflect_reason = reflect_reason_out

                    # Append reflect tool result and continue loop
                    # The LLM will call decide() on the next iteration
                    messages = messages + [
                        {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                        {"role": "tool", "tool_call_id": reflect_tc["id"], "content": "Reflection recorded. Now call decide() with your action."},
                    ]
                    continue

                # --- decide (with or without prior reflect) → terminate ---
                if has_decide:
                    decide_tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == "decide")
                    try:
                        d_args = json.loads(decide_tc["function"]["arguments"])
                    except (json.JSONDecodeError, AttributeError):
                        break  # Bad JSON → fallback

                    action_id = d_args.get("action_id")
                    expectation = d_args.get("expectation", "")

                    # Validate action_id
                    if action_id is None or action_id not in available_actions:
                        break  # Invalid action → fallback

                    # 5-repeat action guard
                    needs_reflection = has_reflect  # reflect was called → needs_reflection
                    consecutive = 0
                    for h in reversed(action_history):
                        if f"action={action_id}" in h:
                            consecutive += 1
                        else:
                            break
                    if consecutive >= 5:
                        needs_reflection = True
                        logger.info(
                            "frame=%s action=%s repeated %d times; forcing reflection",
                            frame_index,
                            action_id,
                            consecutive,
                        )

                    log_node(frame_index, "unified", action=action_id, tool_calls=call_idx + 1)

                    # Cache this frame for next turn's history
                    history_cache.append({
                        "action": action_id,
                        "objects": objects,
                        "adjacency": adjacency,
                    })

                    return {
                        "action": GameAction.from_id(action_id),
                        "plan": f"Action {action_id}: {expectation}",
                        "expectation": expectation,
                        "needs_reflection": needs_reflection,
                        "mechanics": new_mechanics,
                        "mechanics_summary": new_mechanics_summary,
                        "tactical": new_tactical,
                        "tactical_summary": new_tactical_summary,
                        "actions": new_actions,
                        "goal": new_goal,
                        "goal_status": new_goal_status,
                        "reflect_reason": reflect_reason_out,
                    }

                # No inspect, reflect, or decide → nudge
                continue

            # ============================================================
            # V2 2-tool dispatch: decide (original logic)
            # ============================================================
            # ---- Process decide tool call ----
            if "decide" in function_names:
                tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == "decide")
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, AttributeError):
                    # Bad JSON → fallback
                    break

                action_id = args.get("action_id")
                expectation = args.get("expectation", "")
                reflect = args.get("reflect", False)

                wm = args.get("world_model")
                if wm is not None:
                    mechanics_list = wm.get("mechanics", [])
                    mechanics_summary = wm.get("mechanics_summary", "")
                    tactical_list = wm.get("tactical", [])
                    tactical_summary = wm.get("tactical_summary", "")
                    actions_list = wm.get("actions", [])
                    goal = wm.get("goal", "")
                    goal_status = wm.get("goal_status", "")
                else:
                    mechanics_list = args.get("mechanics", [])
                    mechanics_summary = args.get("mechanics_summary", "")
                    tactical_list = args.get("tactical", [])
                    tactical_summary = args.get("tactical_summary", "")
                    actions_list = args.get("actions", [])
                    goal = args.get("goal", "")
                    goal_status = args.get("goal_status", "")

                # Validate action_id against available_actions
                if action_id is None or action_id not in available_actions:
                    break  # Invalid action → fallback

                # force_reflect overrides reflect to True
                if force_reflect:
                    reflect = True

                # Build output mechanics/tactical
                mechanics_out: list[str] = prev_mechanics
                mechanics_summary_out: str = prev_mechanics_summary
                tactical_out: list[str] = prev_tactical
                tactical_summary_out: str = prev_tactical_summary
                actions_out: list[str] = prev_actions
                goal_out: str = prev_goal
                goal_status_out: str = prev_goal_status

                if reflect:
                    new_mech = mechanics_list if mechanics_list else prev_mechanics
                    new_mech_sum = mechanics_summary if mechanics_summary else prev_mechanics_summary
                    new_tac = tactical_list if tactical_list else prev_tactical
                    new_tac_sum = tactical_summary if tactical_summary else prev_tactical_summary
                    new_actions = actions_list if actions_list else prev_actions
                    new_goal = goal if goal else prev_goal
                    new_goal_status = goal_status if goal_status else prev_goal_status
                    max_mechanics = config.max_mechanics
                    max_tactical = config.max_tactical
                    max_action_entries = config.max_action_entries
                    if len(new_mech) > max_mechanics:
                        new_mech = new_mech[-max_mechanics:]
                    if len(new_tac) > max_tactical:
                        new_tac = new_tac[-max_tactical:]
                    if len(new_actions) > max_action_entries:
                        new_actions = new_actions[-max_action_entries:]
                    mechanics_out = new_mech
                    mechanics_summary_out = new_mech_sum
                    tactical_out = new_tac
                    tactical_summary_out = new_tac_sum
                    actions_out = new_actions
                    goal_out = new_goal
                    goal_status_out = new_goal_status

                # 5-repeat action guard: count consecutive same-action entries
                needs_reflection = reflect
                consecutive = 0
                for h in reversed(action_history):
                    if f"action={action_id}" in h:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 5:
                    needs_reflection = True
                    logger.info(
                        "frame=%s action=%s repeated %d times; forcing reflection",
                        frame_index,
                        action_id,
                        consecutive,
                    )

                log_node(frame_index, "unified", action=action_id, tool_calls=call_idx + 1)

                # Cache this frame for next turn's history
                history_cache.append({
                    "action": action_id,
                    "objects": objects,
                    "adjacency": adjacency,
                })

                return {
                    "action": GameAction.from_id(action_id),
                    "plan": f"Action {action_id}: {expectation}",
                    "expectation": expectation,
                    "needs_reflection": needs_reflection,
                    "mechanics": mechanics_out,
                    "mechanics_summary": mechanics_summary_out,
                    "tactical": tactical_out,
                    "tactical_summary": tactical_summary_out,
                    "actions": actions_out,
                    "goal": goal_out,
                    "goal_status": goal_status_out,
                    "reflect_reason": prev_reflect_reason,
                }

        # ---- Fallback: max tool calls exhausted or LLM error ----
        random_action_id = random.choice(available_actions)
        logger.warning(
            "frame=%s unified exhausted %d tool calls; random fallback action=%s",
            frame_index,
            max_tool_calls,
            random_action_id,
        )
        log_node(frame_index, "unified", action=random_action_id, fallback=True)

        # Cache this frame even on fallback
        history_cache.append({
            "action": random_action_id,
            "objects": objects,
            "adjacency": adjacency,
        })

        return {
            "action": GameAction.from_id(random_action_id),
            "plan": "unified fallback, random action",
            "expectation": "",
            "needs_reflection": False,
            "mechanics": prev_mechanics,
            "mechanics_summary": prev_mechanics_summary,
            "tactical": prev_tactical,
            "tactical_summary": prev_tactical_summary,
            "actions": prev_actions,
            "goal": prev_goal,
            "goal_status": prev_goal_status,
            "reflect_reason": prev_reflect_reason,
        }

    return unified_node