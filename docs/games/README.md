# Game Reference Sheets

Per-game quick-reference docs for fast log investigation. Each sheet captures
the game's mechanics, visual vocabulary, action set, and recording inventory
so we can triage `.recording.jsonl` / `.llm.jsonl` logs without re-deriving the
rules from scratch.

## Conventions

Each game gets one file named after its game id prefix (e.g. `wa30.md`). A
sheet must answer four questions, fast:

1. **What is the player?** (color, shape, how to spot it in a raw frame)
2. **What do the actions do?** (mapping action id -> effect)
3. **What is the win/loss mechanic?** (what advances `levels_completed`, what
   ends the episode)
4. **What are the visual signals?** (transient colors, HUD elements, anything
   that looks like noise but carries meaning)

Include a "Log triage tips" section with concrete `jq` recipes for that game's
recordings.

## Inventory

| Game | Sheet | Recordings | Notes |
|---|---|---|---|
| wa30 | [wa30.md](wa30.md) | 1 human, 3 llmcuriosity | 2D carry-puzzle; compound player (body+head); ready-state color signal; carry transitions break controllable-id stability (see §3b) |

## How to add a sheet

1. Play the game (human mode) or run `--agent=random --game=<id>` to generate a
   recording.
2. Extract the facts from the recording with `jq` + a small python snippet
   (color histogram, bbox per color, action->delta). Do **not** rely on memory
   — verify against the frames.
3. Fill in the four sections above + log triage tips.
4. Add a row to the inventory table here.
5. If the recording reveals a mechanic that contradicts a `docs/diary/` or
   `docs/reports/` entry, note the contradiction in the sheet and update the
   old entry (or annotate it) — don't leave silent contradictions.