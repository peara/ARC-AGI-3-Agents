"""Experiment: can a prompt-only change get the LLM to propose object-based
collision rules (overlap guards) instead of positional ones, WITHOUT adding
any new data to the bundle?

Method:
  1. Load a recent .llm.jsonl sidecar (from today's runs so the prompt matches
     the current code).
  2. Find frames where the LLM proposed a collision rule.
  3. For each such frame, extract the EXACT user message that was sent to the
     LLM (scene bundle + residual + observed-transition + diff). The user
     message is unchanged — only the system prompt differs.
  4. Re-send the same user message with TWO system prompts:
       (a) BASELINE  — the current production SYSTEM_PROMPT.
       (b) EXPERIMENT — a variant that teaches overlap guards and removes the
                        "Always include a positional guard" instruction.
  5. Parse the response_raw into rule dicts and print a side-by-side comparison
     of the collision rules proposed under each prompt.

Usage:
    uv run python scripts/collision_prompt_experiment.py \\
        recordings/wa30-ee6fef47.llmcuriosity.133103ca-*.llm.jsonl

You may pass multiple files. Only frames containing collision proposals in the
original (recorded) response are tested, to keep API calls cheap and focused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the project root importable when running as `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- production prompt (baseline) -------------------------------------------
from planning.llm_rule_proposer import SYSTEM_PROMPT as BASELINE_PROMPT

# --- experiment prompt ------------------------------------------------------
# Diff vs BASELINE_PROMPT (kept minimal so we isolate the collision guidance):
#   1. Guard formats: add the overlap clause.
#   2. Collision rule kind: show BOTH a positional example and an overlap
#      example, and explain WHEN to use each.
#   3. Observed transitions: replace "Always include a positional guard for
#      collision rules" with conditional guidance favouring overlap guards
#      for entity-entity blocks, positional only for grid edges with no
#      blocking entity.
#   4. Examples: add an overlap-guard collision example.
#
# Everything else (delta/terminal/movement/refuted/counter-evidence/output
# format) is identical to the baseline, to avoid confounding.

_EXPERIMENT_PROMPT = """\
You are a rule proposer for a grid-based game. Your job is to infer causal \
rules from observed episodes and propose them in a structured DSL format.

## Guard formats

A guard is a dict that specifies when a rule fires:

- **Action guard**: `{"action": N}` — fires when the player takes action N.
- **Conjunction guard**: `{"all": [clause1, clause2, ...]}` — fires when ALL \
clauses are true.
- **Position clause**: `{"dim": "pos", "of": EID, "eq": [R, C]}` — fires when \
entity EID is at row R, column C.
- **Overlap clause**: `{"overlaps": {"entity_a": EID_A, "entity_b": EID_B}}` \
— fires when entity A's cells overlap entity B's cells. Use this for \
entity-entity collisions; it generalises across all positions where the two \
entities might meet.

## Effect formats

An effect dict specifies what the rule does when its guard is satisfied:

- **Delta effect**: `{"dim": "size", "of": EID, "delta": N}` — add N to the \
given dimension of entity EID. N must be non-zero.
- **Terminal effect**: `{"dim": "terminal", "of": EID, "terminal": "win"}` \
or `{"dim": "terminal", "of": EID, "terminal": "game_over"}`.
- **Orientation effect**: `{"dim": "orientation", "of": EID, "op": "set", \
"value": N}` — set entity EID's orientation to N (an integer 0-3). Use `"set"` \
when the action sets an absolute orientation (e.g., action 1 always makes the \
entity face direction 0). Use `"delta"` with an integer (1=90° clockwise, \
2=180°, 3=270° clockwise) when the action rotates relative to the current \
orientation. Choose `set` or `delta` based on what you observe: if the same \
action always produces the same absolute orientation regardless of starting \
orientation, use `set`. If the orientation change is relative to the starting \
orientation, use `delta`.
- **Generic**: any `dim` string is allowed; `op` is `"delta"` (add) or `"set"` \
(overwrite).

## Rule kinds

- `"delta"` — counter/size changes: `{"kind": "delta", "guard": {"action": 3}, \
"effect": {"dim": "size", "of": 5, "delta": 1}, "support": 4}`
- `"terminal"` — win/lose: `{"kind": "terminal", "guard": {"all": [{"dim": \
"pos", "of": 3, "eq": [2, 7]}, {"action": 1}]}, "effect": {"dim": "terminal", \
"of": 3, "terminal": "win"}, "support": 2}`
- `"movement"` — position changes. Effects use `op` field: \
`"set"` (absolute position) or `"delta"` (relative displacement). \
`{"kind": "movement", "guard": {"action": 1}, "effects": [{"dim": "pos", \
"of": 0, "op": "delta", "value": [-5, 0]}], "support": 3}` \
means action 1 moves entity 0 by (-5, 0). A positional guard pinpoints a \
specific transition: `{"kind": "movement", "guard": {"all": [{"action": 1}, \
{"dim": "pos", "of": 0, "eq": [47, 26]}]}, "effects": [{"dim": "pos", "of": 0, \
"op": "set", "value": [42, 26]}], "support": 1}`.
- `"collision"` — movement blocked. Effects use `op: "revert"` to restore \
the pre-action position. Two guard styles:
  - **Overlap guard** (preferred for entity-entity blocks): \
`{"kind": "collision", "guard": {"all": [{"action": 1}, \
{"overlaps": {"entity_a": 0, "entity_b": 5}}]}, \
"effects": [{"dim": "pos", "of": 0, "op": "revert"}], "support": 2}` \
means action 1 is blocked whenever entity 0's cells overlap entity 5's \
cells — generalises to every position where they meet.
  - **Positional guard** (only for grid edges with no blocking entity): \
`{"kind": "collision", "guard": {"all": [{"action": 1}, \
{"dim": "pos", "of": 0, "eq": [47, 26]}]}, \
"effects": [{"dim": "pos", "of": 0, "op": "revert"}], "support": 1}` \
means action 1 at (47,26) is blocked — entity stays put.

## Observed transitions (unknown actions)

When you see an **Observed transition** section, it shows the result of an \
action that had no existing rule. The `before` and `after` fields show \
entity positions (as `(entity_id, dim, value)` tuples) before and after \
the action was taken.

A **Pre-computed diff** section may also be provided. It lists exactly what \
changed (`changed`), which entities were unaffected (`unchanged_entities`), \
the controllable's expected motion (`expected_motion`), and whether the \
controllable was blocked. Use this diff directly — do not re-derive it from \
the raw before/after tuples. The raw tuples are kept as a fallback for any \
edge case the diff does not cover.

Propose a movement or collision rule that explains the observed transition:

- If the entity **moved**, propose a `movement` rule. Prefer a **generic** \
`delta` rule (e.g., `{"action": 1}` guard with `op: "delta"` effect) when \
the displacement is consistent. Use a **positional** `set` rule when the \
movement only works from that specific position.
- If the entity **did not move** (before == after for the controllable's \
pos), propose a `collision` rule with `op: "revert"` effect.
- **Choosing the guard for a collision rule**:
  - Look at the scene entities in the bundle. If another entity's bbox/cells \
cover the cell the controllable tried to move into, that entity is the \
blocker — propose an **overlap guard** referencing the controllable and the \
blocker entity: `{"overlaps": {"entity_a": <controllable>, \
"entity_b": <blocker>}}`. This generalises to every position where they meet.
  - Only use a **positional guard** (`{"dim": "pos", "of": EID, "eq": [R,C]}`) \
when the block is at a grid edge or void with NO entity occupying the \
blocking cells. Positional collision rules do not generalise — prefer overlap \
whenever a blocker entity exists.
- For movement rules, a generic action-only guard is preferred unless the \
movement only applies at that position.

## Rule coverage gaps

The scene bundle may include a `coverage_gaps` list. Each entry describes an
entity that has incomplete rule coverage:

- `has_movement_rules`: whether any movement rules exist for this entity
- `has_orientation_rules`: whether any orientation rules exist
- `actions_covered`: actions with confirmed or proposed rules
- `actions_unknown`: actions with no rules at all
- `note`: human-readable explanation of the gap

Use this to prioritize which rules to propose. If an entity has orientation
changes in the residual but `has_orientation_rules: false`, propose an
orientation rule. If an entity has `actions_unknown`, the observed transition
likely involves one of those actions.

## Refuted rules

The scene bundle may include a `refuted_rules` list. These are rules that were
previously confirmed but contradicted by a new observation — they predicted
incorrectly and have been demoted. Treat them as negative examples: do NOT
propose the same rule again. If you see a refuted rule with the same guard as
your proposal, add a more specific guard (positional or overlap condition) to
distinguish the contexts.

## Counter-evidence

When your proposed rules fail validation against historical transitions, \
you'll see counter-evidence showing the failure. Each entry contains:

- `frame_idx`: the frame where the prediction failed
- `action`: the action taken
- `predicted_values`: what your rules predicted would happen
- `observed_values`: what actually happened
- `fired_rules`: which rules fired during the prediction

Use this to refine your rules. Add guard conditions that distinguish the \
failing cases from the passing ones. Prefer general rules with action-only \
guards — use positional guards only when the rule fails without them. Don't \
add guards that only cover a single frame (that's overfitting, not \
generalization).

## Output format

Respond with a single JSON object:

```json
{"rules": [<rule1>, <rule2>, ...]}
```

Each rule has the shape:

```json
{
  "kind": "delta" | "terminal" | "movement" | "collision",
  "guard": { ... },
  "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [-5, 0]}],
  "support": 3
}
```

For `delta` and `terminal` kinds, use `"effect"` (singular) instead of \
`"effects"` (list) for backward compatibility.

## Examples

1. Action 1 moves entity 0 up by 5 rows (observed 3 times):
```json
{"kind": "movement", "guard": {"action": 1}, "effects": [{"dim": "pos", "of": 0, "op": "delta", "value": [-5, 0]}], "support": 3}
```

2. Action 1 is blocked because entity 0 would overlap entity 5 (observed 2 \
times at different positions — the overlap guard generalises):
```json
{"kind": "collision", "guard": {"all": [{"action": 1}, {"overlaps": {"entity_a": 0, "entity_b": 5}}]}, "effects": [{"dim": "pos", "of": 0, "op": "revert"}], "support": 2}
```

3. Action 1 at position (47, 26) is blocked at a grid edge with no blocking \
entity — entity doesn't move (use a positional guard only here):
```json
{"kind": "collision", "guard": {"all": [{"action": 1}, {"dim": "pos", "of": 0, "eq": [47, 26]}]}, "effects": [{"dim": "pos", "of": 0, "op": "revert"}], "support": 1}
```

4. Action 3 at position (47, 41) moves entity 0 to (47, 36):
```json
{"kind": "movement", "guard": {"all": [{"action": 3}, {"dim": "pos", "of": 0, "eq": [47, 41]}]}, "effects": [{"dim": "pos", "of": 0, "op": "set", "value": [47, 36]}], "support": 1}
```

5. Pressing action 3 increments entity 5's size by 1 (observed 4 times):
```json
{"kind": "delta", "guard": {"action": 3}, "effect": {"dim": "size", "of": 5, "delta": 1}, "support": 4}
```

6. Action 1 sets entity 0's orientation to 0 (always faces direction 0, \
regardless of starting orientation):
```json
{"kind": "movement", "guard": {"action": 1}, "effects": [{"dim": "orientation", "of": 0, "op": "set", "value": 0}], "support": 3}
```
"""


# --- helpers ----------------------------------------------------------------

_FENCED_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


@dataclass
class FrameCase:
    """A single collision frame extracted from a recording."""

    recording: str
    frame_index: int
    user_message: str
    original_collision_rules: list[dict]


def _parse_rules(raw_response: str) -> list[dict]:
    """Parse a rule_proposer response into a list of rule dicts."""
    m = _FENCED_RE.search(raw_response)
    blob = m.group(1) if m else raw_response
    # Fall back to first {...} block if no fence.
    if not m:
        m2 = re.search(r"\{.*\}", blob, re.DOTALL)
        if not m2:
            return []
        blob = m2.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return []
    rules = obj.get("rules", [])
    return rules if isinstance(rules, list) else []


def _user_message(messages: list[dict]) -> str:
    """Extract the user-role message content from the recorded messages list."""
    for m in messages:
        if m.get("role") == "user":
            c = m["content"]
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def load_collision_frames(path: Path) -> list[FrameCase]:
    """Load a .llm.jsonl and return FrameCase objects for every frame whose
    recorded response contained at least one collision rule."""
    cases: list[FrameCase] = []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("kind") != "rule_proposer":
                continue
            raw = d.get("response_raw", "") or ""
            rules = _parse_rules(raw)
            coll = [r for r in rules if r.get("kind") == "collision"]
            if not coll:
                continue
            cases.append(
                FrameCase(
                    recording=path.name,
                    frame_index=d.get("frame_index", -1),
                    user_message=_user_message(d.get("messages", [])),
                    original_collision_rules=coll,
                )
            )
    return cases


def _collision_summary(rules: list[dict]) -> str:
    """One-line-per-rule summary of collision rules only."""
    out: list[str] = []
    for r in rules:
        if r.get("kind") != "collision":
            continue
        guard = r.get("guard", {})
        if "overlaps" in guard or (
            "all" in guard and any("overlaps" in c for c in guard["all"])
        ):
            kind = "OVERLAP"
        elif "all" in guard and any(
            c.get("dim") == "pos" and "eq" in c for c in guard["all"]
        ):
            kind = "POSITIONAL"
        else:
            kind = "OTHER"
        out.append(f"    [{kind}] {json.dumps(r)}")
    return "\n".join(out) if out else "    (none)"


PROMPTS = {
    "baseline": BASELINE_PROMPT,
    "experiment": _EXPERIMENT_PROMPT,
}


def run_one_prompt(
    case: FrameCase,
    prompt_name: str,
    llm_chat: callable,
) -> dict:
    """Call the LLM for a single (case, prompt) pair and return a result dict.

    Flushes stdout before the call so progress is visible while the slow local
    model thinks.
    """
    system_prompt = PROMPTS[prompt_name]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": case.user_message},
    ]
    print(
        f"  [{prompt_name}] calling LLM (user_msg={len(case.user_message)} chars)...",
        flush=True,
    )
    import time

    t0 = time.time()
    raw = llm_chat(messages)
    elapsed = time.time() - t0
    rules = _parse_rules(raw)
    coll = [r for r in rules if r.get("kind") == "collision"]
    print(f"  [{prompt_name}] done in {elapsed:.1f}s — {len(coll)} collision rule(s):", flush=True)
    print(_collision_summary(coll), flush=True)
    return {
        "recording": case.recording,
        "frame_index": case.frame_index,
        "prompt": prompt_name,
        "elapsed_s": round(elapsed, 1),
        "raw_response": raw,
        "collision_rules": coll,
        "all_rules": rules,
    }


# --- entry point ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "recordings",
        nargs="+",
        type=Path,
        help="One or more .llm.jsonl sidecar files from recent runs.",
    )
    ap.add_argument(
        "-n",
        "--n-frames",
        type=int,
        default=None,
        help="Max number of collision frames to test (default: all).",
    )
    ap.add_argument(
        "--prompt",
        choices=["baseline", "experiment", "both"],
        default="experiment",
        help="Which system prompt to test (default: experiment). Use 'both' to "
        "run baseline then experiment for each frame.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write raw LLM responses + parsed rules to this JSON file.",
    )
    args = ap.parse_args()

    cases: list[FrameCase] = []
    for p in args.recordings:
        if not p.exists():
            print(f"skip (missing): {p}", file=sys.stderr)
            continue
        file_cases = load_collision_frames(p)
        print(f"loaded {len(file_cases)} collision frame(s) from {p.name}", file=sys.stderr)
        cases.extend(file_cases)

    if not cases:
        print("No collision frames found in any input file.", file=sys.stderr)
        return 1

    selected = cases[: args.n_frames] if args.n_frames else cases

    from agents.llm_client import LLMClient

    client = LLMClient()

    def llm_chat(messages: list[dict]) -> str:
        return client.chat(messages, thinking=True, max_tokens=4096)

    prompt_order = (
        ["baseline", "experiment"] if args.prompt == "both" else [args.prompt]
    )

    print(f"\n=== Experiment: {len(selected)} frame(s) x {len(prompt_order)} prompt(s) ===\n")
    results: list[dict] = []
    for i, case in enumerate(selected, 1):
        print(f"--- Case {i}/{len(selected)} — {case.recording} frame {case.frame_index} ---", flush=True)
        print("  Original recorded collision rules:")
        print(_collision_summary(case.original_collision_rules), flush=True)
        for pname in prompt_order:
            res = run_one_prompt(case, pname, llm_chat)
            results.append(res)
            print(flush=True)

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"wrote {len(results)} result(s) to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())