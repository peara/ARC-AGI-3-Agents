---
name: eda
description: "Exploratory data analysis for a single ARC-AGI-3 recording. Runs scripts/eda_recording.py to produce a markdown overview report from the .recording.jsonl, .llm.jsonl, and .logs.log sidecars. Use when the user asks to 'EDA a recording', 'overview of recording X', 'summarize what happened in this run', 'get the big picture of a recording', or needs a quick triage before diving into frame-by-frame debugging. Triggers: 'eda', 'eda recording', 'overview recording', 'summarize recording', 'big picture', 'triage recording', 'what happened in this run'."
---

# EDA — Recording Exploratory Data Analysis

Produces a markdown overview report from the three sidecar files of one
recording so you can skim 2 minutes and know which frames to investigate,
without reading every log line or LLM call by hand.

Works for all agent types. Some sections are agent-specific (see below).

## When to use

- **Starting point** before `recording-debug` (frame-by-frame) — EDA tells
  you WHICH frames to look at; recording-debug tells you WHAT happened there.
- User asks "what happened in this run" or "overview of recording X".
- Triage: "agent didn't complete any levels — why?"

## How to run

```bash
uv run python scripts/eda_recording.py <recording>.recording.jsonl
uv run python scripts/eda_recording.py <recording>.recording.jsonl -o report.md
```

The recording path is the only required argument. The script auto-discovers
the sibling `.llm.jsonl` and `.logs.log` sidecars by replacing the
`.recording.jsonl` suffix. Missing sidecars degrade gracefully (sections
show "*No ... sidecar found*").

## What the report contains

1. **Executive Summary** — game id, frame count, final state, action
   distribution, LLM call count + latency, log line count, warning count.
   *(all agents)*
2. **Key Frames** — table of frames flagged as interesting: grid cell-change
   spikes, flatline frames, LLM latency spikes, LLM errors.
   Perception-pipeline agents also show: controllable-id changes, 0/N proposal
   failures, rule promotions/refutations. *(all agents; perception agents
   have extra row types)*
3. **Rules** — lifecycle counts (injected/bumped/promoted/refuted/pruned),
   confirmed-rule-counts-over-time table, confirmed rules at final frame,
   refuted rules, sample proposed rules.
   *(perception-pipeline only; all zeros for simulator-first agents)*
4. **LLM Call Summary** — by kind/trigger, truncated count, prompt-size
   breakdown, per-frame call table. *(all agents)*
5. **Entity Lifecycle** — controllable-id changes, entity count over time.
   *(perception-pipeline only; "No controllable ID changes" and entity
   count 0 for simulator-first agents)*
6. **Warnings** — deduped WARNING-level log lines with timestamps.
   *(all agents; different loggers per agent type)*
7. **Grid Diff Stats** — per-frame cell-change count + ASCII sparkline.
   *(all agents)*

## Agent-specific triage

### Perception-pipeline agents (llmcuriosity, langgraphvision, langgraphunified)

- Rule churn? → Section 3 (Rules): are confirmed rules accumulating or churning?
- 0/N proposal failures? → Section 2 (Key Frames): look for "0/N proposals survived"
- Controllable-id instability? → Section 5 (Entity Lifecycle): count ID changes
- Spurious relational rules? → Section 3: check for HUD/structural correlations

### Simulator-first agents (simulatorfirst)

The EDA script does not yet surface `simulator_state` data directly. After
running EDA, check these manually:

- **Simulate built?** → Run: `python -c "import json; lines=open('<recording>.recording.jsonl').readlines(); [print(f'f{i}: has_simulate={json.loads(l)[\"data\"][\"scene_state\"].get(\"simulator_state\",{}).get(\"has_simulate\",False)}') for i,l in enumerate(lines)]"`
- **check() accuracy?** → Look for `last_check_result` in `simulator_state` at later frames.
- **Fallback fired?** → Section 6 (Warnings): grep for "falling back to random"
- **LLM timeouts?** → Section 4 (LLM Call Summary): check for `ok: false` + latency spikes
- **Flatline frames?** → Section 7 (Grid Diff Stats): 0 cells changed = agent stuck (not acting or action has no effect)
- **High tool_calls?** → Check `action_input.reasoning.tool_calls` per frame in the recording

### Langgraph agents (langgraphvision, langgraphunified)

The EDA script does not surface `langgraph_state` data directly. After
running EDA, check these manually:

- **Goal identified?** → Check `langgraph_state.goal` and `goal_status` per frame.
- **Expectations met?** → Compare `langgraph_state.expectation` with actual grid diff.
- **Reflection firing?** → Check `needs_reflection` and `reflect_reason` fields.
- **Stuck in loop?** → Check `node_path` across frames for repetition.
- **LLM errors?** → Section 4 (LLM Call Summary): check for `ok: false`.

## Workflow

1. Run the script on the recording path.
2. Read section 1 (Executive Summary) — does the agent look healthy?
3. For perception-pipeline: read section 3 (Rules) — are rules accumulating or churning?
   For simulator-first: check `simulator_state` manually (see triage above).
4. Read section 2 (Key Frames) — which frames need deeper investigation?
5. For each flagged frame, switch to `recording-debug` skill for
   frame-by-frame analysis.

## Output

The report goes to stdout by default. Use `-o <path>` to write to a file.
The report is pure markdown — safe to commit, diff, or paste into issues.