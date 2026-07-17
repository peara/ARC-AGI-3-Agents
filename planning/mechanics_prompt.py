from __future__ import annotations

import json
from typing import Any

from effects.guard_parse import parse_guard_clauses
from effects.rules import Rule
from vision.render import grid_to_image, image_to_base64, make_image_block

MECHANICS_SYSTEM_PROMPT = """\
You are a game-mechanics analyst for an interactive grid-based puzzle game.

You are given a sequence of frames from a single game level. Each frame is a
64×64 grid of color indices (0–15). You see both the raw grid image AND a
symbolic scene description (entities, their positions, colors, sizes, and
composition).

Your job is to infer the **game mechanics** — the unwritten rules that govern
how the game works and what the player must do to complete the level. This is
NOT about picking the next action. It is about understanding the game's
objective and the key interactions.

## What to look for

1. **Objective** — What is the player trying to achieve? What advances
   `levels_completed`? Look for patterns: objects being moved to specific
   locations, objects being collected/depleted, objects being arranged in a
   pattern, reaching a specific position, etc.

2. **Key mechanics** — What interactions exist?
   - Can the player pick up / carry / drop objects?
   - Can the player push / pull objects?
   - Are there transient signals (a color flash indicating "ready to interact")?
   - What happens when the player touches / moves into an object?

3. **Progress signals** — How can the player tell they are making progress?
   - `levels_completed` increments (definite progress)
   - Objects shrinking / disappearing
   - Spatial configuration changes (objects entering/leaving regions,
     reaching positions, forming patterns)
   - New visual elements appearing

4. **Entities and roles** — What types of objects exist? The system
   already labels `controllable` (the player) and `counter` (HUD step
   counter). Infer any additional roles you can observe:
   - Collectible / movable objects
   - Target zone / goal area
   - Obstacles / walls
   - Other HUD elements

## Output format

Respond with a single JSON object:

```json
{
  "objective": "<one-sentence description of the level's goal>",
  "key_mechanics": [
    "<mechanic 1>",
    "<mechanic 2>",
    "..."
  ],
  "progress_signals": [
    "<signal 1>",
    "<signal 2>"
  ],
  "entity_roles": {
    "<role name>": "<how to identify it — color, shape, behavior>"
  },
  "next_steps": "<what the player should do next to advance toward the objective>",
  "confidence": <float 0.0-1.0>
}
```

Your `objective` must be ≤200 chars, `key_mechanics` ≤5 items, `next_steps` ≤300 chars. These limits exist because your output is passed as advisory context to a downstream action planner with a limited context window. Be concise and actionable.
"""

REFINE_SYSTEM_PROMPT = MECHANICS_SYSTEM_PROMPT + """
\
## Iterative refinement

You may receive a **previous hypothesis** — a mechanics inference from
earlier frames. Your job is to evaluate it against the new evidence and
either confirm, refine, or refute it.

- **confirmed**: the new evidence supports the previous hypothesis. Keep
  it as-is (or with minor wording improvements). Boost confidence.
- **refined**: the new evidence partially supports but also adds nuance
  or corrections. Update the relevant fields and explain what changed.
- **refuted**: the new evidence contradicts the previous hypothesis.
  Propose a new one and explain why the old one was wrong.

The `changes` field must explain what you updated and why, referencing
specific evidence from the new frames.

When refining, the output JSON must also include:

```json
{
  "status": "confirmed | refined | refuted",
  "changes": "<what you updated and why, referencing specific evidence>",
  ...all other fields as above...
}
```
"""

def build_action_legend(
    available_actions: tuple[int, ...],
    movement_rules: tuple[Rule, ...],
) -> dict[int, str]:
    """Build a legend mapping action IDs to descriptions.
    
    Actions associated with confirmed movement rules are labeled as '(move)'.
    Others are labeled as plain 'ACTION{N}'.
    """
    confirmed_move_actions: set[int] = set()
    for rule in movement_rules:
        clauses = parse_guard_clauses(rule.guard_spec)
        for clause in clauses:
            if clause["has_action"] and clause["action"] is not None:
                confirmed_move_actions.add(clause["action"])
    
    return {
        action_id: f"ACTION{action_id} (move)" if action_id in confirmed_move_actions else f"ACTION{action_id}"
        for action_id in available_actions
    }

def build_messages(
    frames: list[list[list[int]]],
    scene_summaries: list[dict[str, Any]],
    action_legend: dict[int, str],
    prev_hypothesis: dict[str, Any] | None = None,
    levels_completed_delta: int = 0,
    vision_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Build the system and user messages for mechanics inference.
    
    Args:
        frames: List of 64x64 grids (most recent first).
        scene_summaries: List of scene metadata dicts (most recent first).
        action_legend: Map of action_id to description.
        prev_hypothesis: Previous LLM hypothesis for refinement.
        levels_completed_delta: Change in levels completed since last call.
        vision_enabled: Whether to include grid images.
        
    Returns:
        List of two messages: [system, user].
    """
    # Cap to last 8 frames
    frames = frames[:8]
    scene_summaries = scene_summaries[:8]

    user_content: list[dict[str, Any]] = []

    # 1. Intro / Hypothesis context
    if prev_hypothesis is None:
        user_content.append({
            "type": "text",
            "text": (
                "You are observing frames from an ARC-AGI-3 puzzle game. "
                "Each frame shows a 64×64 grid.\n\n"
                "Action legend:\n"
                + "\n".join(f"  {k}: {v}" for k, v in action_legend.items())
                + "\n\nInfer the game mechanics."
            ),
        })
    else:
        user_content.append({
            "type": "text",
            "text": (
                "New evidence has arrived from the game. Your previous "
                "mechanics hypothesis is below.\n\n"
                "## Previous hypothesis\n```json\n"
                + json.dumps(prev_hypothesis, indent=2)
                + "\n```\n\n"
                "Evaluate it against the new frames below. Output "
                "`status: confirmed/refined/refuted` and explain what "
                "changed in the `changes` field."
            ),
        })

    # 2. Frames sequence (most recent first)
    for i, (grid, summary) in enumerate(zip(frames, scene_summaries)):
        # Note: In the recording, frames are indexed by absolute frame number.
        # Here we just treat them as "Frame 0" (most recent) etc. or preserve indices if known.
        # Since build_messages only gets a list, we use relative indices.
        user_content.append({
            "type": "text",
            "text": f"\n--- Frame {i} ---",
        })

        if vision_enabled:
            img = grid_to_image(grid)
            b64 = image_to_base64(img)
            user_content.append(make_image_block(b64))

        # Summary text
        summary_text = (
            f"Action taken: {summary.get('action_taken', 'unknown')}\n"
            f"Levels completed: {summary.get('levels_completed')}\n"
            f"Controllable entity: {summary.get('controllable_id')} "
            f"at {summary.get('controllable_pos')}\n"
            f"Entity count: {summary.get('n_entities')}"
        )
        user_content.append({
            "type": "text",
            "text": summary_text,
        })

    # 3. Progress signals
    if levels_completed_delta > 0:
        user_content.append({
            "type": "text",
            "text": f"\n⚠ LEVEL ADVANCED: delta={levels_completed_delta}",
        })

    # 4. Final footer
    user_content.append({
        "type": "text",
        "text": "\n--- Your analysis ---\nOutput the JSON object.",
    })

    system_prompt = REFINE_SYSTEM_PROMPT if prev_hypothesis is not None else MECHANICS_SYSTEM_PROMPT

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

__all__ = ["MECHANICS_SYSTEM_PROMPT", "REFINE_SYSTEM_PROMPT", "build_messages"]
