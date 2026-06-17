"""LLM-backed planner: prompt construction, response parsing, and goal validation.

This module is network-free — ``call_planner`` takes a ``Callable`` for the LLM
invocation, keeping ``planning/`` free of API client dependencies.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .probe import ProbeGoal

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an exploration planner for a grid-based game. Your job is to choose the
next probe goal given the current scene and any failure context from previous
attempts.

## Predicate schema

Goals are expressed as predicate dicts with these forms:

1. **near** — navigate the player entity to a position or near another entity:
   {"dim": "pos", "of": <player_eid>, "near": {"of": <target_eid>, "radius": N}}
   {"dim": "pos", "of": <player_eid>, "near": [row, col], "radius": N}

2. **eq** — test equality of a dimension on an entity:
   {"dim": "<dim_name>", "of": <eid>, "eq": <value>}

3. **all** — conjunction of sub-predicates:
   {"all": [<sub_pred1>, <sub_pred2>, ...]}

4. **action** — action guard (ignored for goal satisfaction):
   {"action": <action_id>}

**Prefer `near` predicates** — navigation goals are the primary exploration
pattern. Use `eq` only when you have a specific property hypothesis to test.

## Examples

Example 1 — Go probe an entity (near with relative entity ref):
```json
{
  "predicate": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
  "max_steps": 50,
  "reason": "Entity 17 (counter at row 12) is unexplored — navigate close to observe size changes"
}
```

Example 2 — Explore an area (near with coordinate list):
```json
{
  "predicate": {"dim": "pos", "of": 0, "near": [5, 32], "radius": 5},
  "max_steps": 100,
  "reason": "Top of map is undiscovered — check for entities or interactions in that area"
}
```

Example 3 — Test if an object is solid (near with small radius):
```json
{
  "predicate": {"dim": "pos", "of": 0, "near": {"of": 8, "radius": 1}},
  "max_steps": 50,
  "reason": "Navigate adjacent to entity 8 at (58,6) — next turn, move into it to test if it's solid or walkable"
}
```

## Instructions

- Always have an opinion. If unsure, pick an unexplored entity to navigate toward.
- Your `reason` should explain what you're doing AND what you'll do next — it helps you continue your plan across turns.

## Output format

Respond with a single JSON object:
```json
{
  "predicate": { ... },
  "max_steps": <int>,
  "reason": "<string>"
}
```"""

# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


def _build_messages(
    bundle: dict[str, object],
    available_actions: list[int],
    failure_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Build the system + user messages for the LLM planner call.

    Returns exactly two messages: one with role "system" and one with role
    "user".
    """
    user_parts: list[str] = []

    # Scene bundle as JSON
    user_parts.append(f"## Scene bundle\n```json\n{json.dumps(bundle, indent=2)}\n```")

    # Available actions
    user_parts.append(f"## Available actions\n{available_actions}")

    # Optional failure context
    if failure_context is not None:
        user_parts.append(
            f"## Failure context (previous attempt failed)\n```json\n"
            f"{json.dumps(failure_context, indent=2)}\n```"
        )

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _parse_response(raw: str) -> dict[str, object] | None:
    """Extract a JSON dict from an LLM response string.

    Tries markdown ```json ... ``` blocks first, then falls back to parsing
    the raw string directly.  Returns ``None`` on any parse failure.
    """
    # Try fenced JSON blocks
    for match in _JSON_BLOCK_RE.finditer(raw):
        try:
            result: dict[str, object] = json.loads(match.group(1))
            return result
        except json.JSONDecodeError:
            continue

    # Fallback: try the whole string
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# _validate_goal
# ---------------------------------------------------------------------------


def _walk_predicate_ids(predicate: dict[str, object]) -> set[int]:
    """Collect all entity IDs referenced in a predicate dict."""
    ids: set[int] = set()

    # Conjunction — recurse
    if "all" in predicate:
        children = predicate["all"]
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    ids |= _walk_predicate_ids(child)
        return ids

    # "of" at top level
    of_val = predicate.get("of")
    if isinstance(of_val, int):
        ids.add(of_val)

    # Nested "near" dict with its own "of"
    near_val = predicate.get("near")
    if isinstance(near_val, dict):
        ref_eid = near_val.get("of")
        if isinstance(ref_eid, int):
            ids.add(ref_eid)

    return ids


def _validate_goal(
    goal_dict: dict[str, object],
    scene_entities: set[int],
) -> ProbeGoal | None:
    """Validate a parsed goal dict and construct a ProbeGoal, or None.

    Checks that required keys exist and that all entity IDs referenced in
    the predicate are present in ``scene_entities``.
    """
    required_keys = {"predicate", "max_steps", "reason"}
    if not required_keys.issubset(goal_dict.keys()):
        return None

    predicate = goal_dict["predicate"]
    if not isinstance(predicate, dict):
        return None

    # Collect entity IDs referenced in the predicate
    try:
        referenced_ids = _walk_predicate_ids(predicate)
    except Exception:
        return None

    # Verify all referenced entities exist in the scene
    if not referenced_ids.issubset(scene_entities):
        return None

    try:
        return ProbeGoal(
            predicate=predicate,
            max_steps=int(goal_dict["max_steps"]),  # type: ignore[call-overload]
            reason=str(goal_dict["reason"]),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# call_planner
# ---------------------------------------------------------------------------


def call_planner(
    bundle: dict[str, object],
    available_actions: list[int],
    llm_call: Callable[[list[dict[str, str]]], str],
    failure_context: dict[str, object] | None = None,
) -> ProbeGoal | None:
    """Orchestrate LLM-based planning: build prompt → call LLM → parse → validate.

    Parameters
    ----------
    bundle:
        Scene bundle dict (from ``QueryInterface.bundle()``).
    available_actions:
        List of action IDs available in the current frame.
    llm_call:
        Callable that takes a list of message dicts and returns the raw LLM
        response string.  The planning module does **not** import any API
        client — this callable is injected by the caller.
    failure_context:
        Optional dict describing why a previous plan failed, appended to the
        user message.

    Returns
    -------
    ProbeGoal or None
        A validated goal, or None if the LLM response could not be parsed or
        the predicate references entities not present in the scene.
    """
    messages = _build_messages(bundle, available_actions, failure_context)
    raw = llm_call(messages)
    parsed = _parse_response(raw)
    if parsed is None:
        return None

    # Entity IDs in bundle JSON: scene.summary() returns entities as a list
    # of dicts (each with an "id" key), not a dict keyed by ID.
    scene_entities_raw = bundle.get("scene", {})
    entities_val: object = (
        scene_entities_raw.get("entities", []) if isinstance(scene_entities_raw, dict) else []
    )
    if isinstance(entities_val, list):
        scene_entities: set[int] = {
            int(e["id"]) for e in entities_val if isinstance(e, dict) and "id" in e
        }
    elif isinstance(entities_val, dict):
        scene_entities = {int(eid) for eid in entities_val.keys()}
    else:
        scene_entities = set()

    goal = _validate_goal(parsed, scene_entities)
    return goal