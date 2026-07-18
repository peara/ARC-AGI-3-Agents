---
name: eda
description: "Exploratory data analysis for a single ARC-AGI-3 recording. Runs scripts/eda_recording.py to produce a markdown overview report from the .recording.jsonl, .llm.jsonl, and .logs.log sidecars. Use when the user asks to 'EDA a recording', 'overview of recording X', 'summarize what happened in this run', 'get the big picture of a recording', or needs a quick triage before diving into frame-by-frame debugging. Triggers: 'eda', 'eda recording', 'overview recording', 'summarize recording', 'big picture', 'triage recording', 'what happened in this run'."
---

# EDA — Recording Exploratory Data Analysis

Produces a markdown overview report from the three sidecar files of one
recording so you can skim 2 minutes and know which frames to investigate,
without reading every log line or LLM call by hand.

## When to use

- **Starting point** before `recording-debug` (frame-by-frame) — EDA tells
  you WHICH frames to look at; recording-debug tells you WHAT happened there.
- User asks "what happened in this run" or "overview of recording X".
- Triage: "agent didn't complete any levels — why?" → EDA shows rule churn,
  0/N proposal failures, controllable-id instability at a glance.

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
2. **Key Frames** — table of frames flagged as interesting: controllable-id
   changes, 0/N proposal failures (with raw LLM JSON snippet), rule
   promotions/refutations, grid cell-change spikes, flatline frames, LLM
   latency spikes.
3. **Rules** — lifecycle counts (injected/bumped/promoted/refuted/pruned),
   confirmed-rule-counts-over-time table, confirmed rules at final frame,
   refuted rules (confirmed then disproven), sample proposed rules.
4. **LLM Call Summary** — by kind/trigger, truncated count, prompt-size
   breakdown (largest call's per-message sizes), per-frame call table.
5. **Entity Lifecycle** — controllable-id changes, entity count over time.
6. **Warnings** — deduped WARNING-level log lines with timestamps.
7. **Grid Diff Stats** — per-frame cell-change count + ASCII sparkline.

## Workflow

1. Run the script on the recording path.
2. Read section 1 (Executive Summary) — does the agent look healthy?
3. Read section 3 (Rules) — are confirmed rules accumulating or churning?
4. Read section 2 (Key Frames) — which frames need deeper investigation?
5. For each flagged frame, switch to `recording-debug` skill for
   frame-by-frame analysis.

## Output

The report goes to stdout by default. Use `-o <path>` to write to a file.
The report is pure markdown — safe to commit, diff, or paste into issues.