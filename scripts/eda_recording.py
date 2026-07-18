"""EDA (Exploratory Data Analysis) for a single ARC-AGI-3 recording.

Reads the three sidecar files for one recording:
  <recording>.recording.jsonl  — grid frames + scene_state + actions
  <recording>.llm.jsonl        — one JSON line per LLM call
  <recording>.logs.log         — structured logs from subsystems

Outputs a markdown overview report to stdout (or a file with -o) so you
can skim 2 minutes and know which frames to investigate.

Usage::

    uv run python scripts/eda_recording.py <recording>.recording.jsonl
    uv run python scripts/eda_recording.py <recording>.recording.jsonl -o report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter
from typing import Any

try:
    import numpy as np
except ImportError:
    sys.exit("numpy required: uv run python scripts/eda_recording.py ...")


# --- Parsers ------------------------------------------------------------------

def load_recording(path: str) -> list[dict[str, Any]]:
    """Load .recording.jsonl; return list of data dicts (skip scorecard-only last line)."""
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            data = d.get("data", d)
            if "action_input" in data or ("frame" in data and "state" in data):
                out.append(data)
    return out


def load_llm_log(path: str) -> list[dict[str, Any]]:
    """Load .llm.jsonl; return list of call dicts."""
    if not os.path.isfile(path):
        return []
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


_LOG_SPLIT_RE = re.compile(r" \| ")


def load_structured_logs(path: str) -> list[dict[str, str]]:
    """Load .logs.log; return list of {ts, level, logger, msg}.

    Handles multi-line log entries: lines not matching the
    ``ts | LEVEL | logger | msg`` pattern are appended to the previous
    entry's message (e.g. the raw JSON snippet after ``raw snippet:``).
    """
    if not os.path.isfile(path):
        return []
    out: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            parts = _LOG_SPLIT_RE.split(line, 3)
            if len(parts) < 4:
                # Continuation of previous multi-line entry
                if out:
                    out[-1]["msg"] += "\n" + line
                continue
            out.append({
                "ts": parts[0].strip(),
                "level": parts[1].strip(),
                "logger": parts[2].strip(),
                "msg": parts[3],
            })
    return out


# --- Analysis helpers ---------------------------------------------------------

def grid_diffs(frames: list[dict[str, Any]]) -> list[int]:
    """Cell-change count between consecutive frames (first = 0)."""
    diffs: list[int] = []
    prev: np.ndarray | None = None
    for d in frames:
        grid = np.array(d["frame"][0], dtype=np.int8)
        if prev is not None:
            diffs.append(int((grid != prev).sum()))
        else:
            diffs.append(0)
        prev = grid
    return diffs


_ACTION_RE = re.compile(r"ACTION(\d+).*count (\d+)")


def root_action_frames(logs: list[dict[str, str]]) -> list[int]:
    """Map root ACTION log lines to frame indices (by count N -> frame N)."""
    frames: list[int] = []
    for log in logs:
        if log["logger"] != "root":
            continue
        m = _ACTION_RE.search(log["msg"])
        if m:
            frames.append(int(m.group(2)))
    return frames


def assign_frames_to_engine_logs(
    logs: list[dict[str, str]],
) -> list[tuple[str, str, int]]:
    """Correlate effects.engine / planning.llm_planner logs to frame indices.

    Uses root ACTION log lines as frame boundaries (count N -> frame N).
    Each non-root log is assigned the frame of the most recent preceding ACTION.
    """
    # Build (ts, frame) boundaries from root ACTION lines
    boundaries: list[tuple[str, int]] = []
    for log in logs:
        if log["logger"] != "root":
            continue
        m = _ACTION_RE.search(log["msg"])
        if m:
            boundaries.append((log["ts"], int(m.group(2))))
    # Assign frame by binary search on timestamp
    out: list[tuple[str, str, int]] = []
    for log in logs:
        if log["logger"] in ("effects.engine", "planning.llm_planner",
                              "planning.llm_rule_proposer"):
            frame = _find_frame(log["ts"], boundaries)
            out.append((log["logger"], log["msg"], frame))
    return out


def _find_frame(ts: str, boundaries: list[tuple[str, int]]) -> int:
    """Return frame for the most recent boundary <= ts (or -1)."""
    lo, hi = 0, len(boundaries)
    while lo < hi:
        mid = (lo + hi) // 2
        if boundaries[mid][0] <= ts:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return -1
    return boundaries[lo - 1][1]


def _msg_chars(m: dict[str, Any]) -> int:
    """Char count for a message content (string or multimodal block list)."""
    content = m.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def _fmt_rule(r: dict[str, Any]) -> str:
    """One-line human-readable rule summary."""
    kind = r.get("kind", "?")
    guard = r.get("guard_spec") or r.get("guard", {})
    g = json.dumps(guard, separators=(",", ":")) if guard else "always"
    effects = r.get("effects") or [r.get("effect", {})]
    parts = []
    for e in effects:
        if not e:
            continue
        dim = e.get("dim", "?")
        of = e.get("of", "?")
        op = e.get("op", "")
        val = e.get("value")
        if val is None:
            val = e.get("delta")
            if val is not None:
                op = "delta"
        if val is None:
            val = e.get("set")
            if val is not None:
                op = "set"
        parts.append(f"{dim}[{of}].{op}={val}")
    eff = ", ".join(parts) if parts else "?"
    return f"{kind} guard={g} -> {eff}"


# --- Report -------------------------------------------------------------------

def render_report(
    rec_path: str,
    frames: list[dict[str, Any]],
    llm_calls: list[dict[str, Any]],
    logs: list[dict[str, str]],
) -> str:
    """Render the EDA markdown report."""
    lines: list[str] = []
    w = lines.append

    # --- 1. Executive summary ---
    w("# Recording EDA Report")
    w("")
    w("## 1. Executive Summary")
    w("")
    game_id = frames[0].get("game_id", "?") if frames else "?"
    n_frames = len(frames)
    last_state = frames[-1].get("state", "?") if frames else "?"
    last_levels = frames[-1].get("levels_completed", "?") if frames else "?"
    actions = [d["action_input"]["id"] for d in frames if "action_input" in d]
    action_dist = Counter(actions)
    llm_ok = sum(1 for c in llm_calls if c.get("ok"))
    llm_err = len(llm_calls) - llm_ok
    n_warns = sum(1 for log in logs if log["level"] == "WARNING")
    w(f"- **Recording**: `{os.path.basename(rec_path)}`")
    w(f"- **Game ID**: {game_id}")
    w(f"- **Total frames**: {n_frames}")
    w(f"- **Final state**: {last_state}")
    w(f"- **Levels completed (last)**: {last_levels}")
    w(f"- **Action distribution**: {dict(sorted(action_dist.items()))}")
    w(f"- **LLM calls**: {len(llm_calls)} (ok: {llm_ok}, error: {llm_err})")
    if llm_calls:
        lat = [c["latency_ms"] for c in llm_calls]
        w(f"- **LLM latency**: min={min(lat)}ms, median={statistics.median(lat):.0f}ms, max={max(lat)}ms")
    w(f"- **Log lines**: {len(logs)} ({n_warns} WARNING)")
    w("")

    # --- 2. Timeline / key-frame index ---
    w("## 2. Key Frames")
    w("")

    diffs = grid_diffs(frames)
    flagged: list[tuple[int, str]] = []

    # 2a. Controllable ID changes (from warnings)
    for log in logs:
        if "CONTROLLABLE ID CHANGED" in log["msg"]:
            m = re.search(r"frame=(\d+)", log["msg"])
            if m:
                flagged.append((int(m.group(1)), f"CONTROLLABLE ID CHANGED: {log['msg'].split('CHANGED:')[1].strip()[:60]}"))

    # 2b. Rule proposer 0/N survived (correlated via assign_frames_to_engine_logs,
    # since planning.llm_planner logs lack frame=N)
    engine_events_all = assign_frames_to_engine_logs(logs)
    for logger, msg, frame in engine_events_all:
        if "0/" in msg and "proposals survived" in msg:
            m = re.search(r"(\d+/\d+) proposals survived", msg)
            ratio = m.group(1) if m else "0/N"
            snippet_match = re.search(r"raw snippet: (.*)", msg, re.DOTALL)
            snippet = snippet_match.group(1).strip()[:200] if snippet_match else ""
            snippet = snippet.replace("\n", " ").replace("`", "")[:120]
            flagged.append((frame, f"0/N proposals survived ({ratio}): {snippet}"))

    # 2c. LLM errors / latency spikes
    for c in llm_calls:
        if not c.get("ok"):
            flagged.append((c["frame_index"], f"LLM {c['kind']} ERROR: {c.get('error','')[:80]}"))
    if llm_calls:
        lat_vals = [c["latency_ms"] for c in llm_calls]
        med = statistics.median(lat_vals)
        for c in llm_calls:
            if c["latency_ms"] > med * 3:
                flagged.append((c["frame_index"], f"LLM latency spike: {c['latency_ms']}ms ({c['kind']})"))

    # 2d. Grid cell-change spikes (top 5% by diff)
    if diffs:
        threshold_idx = max(int(len(diffs) * 0.95), 1)
        sorted_diffs = sorted(diffs)
        threshold = sorted_diffs[threshold_idx - 1] if threshold_idx <= len(sorted_diffs) else sorted_diffs[-1]
        for i, d in enumerate(diffs):
            if d >= threshold and d > 0:
                flagged.append((i, f"high-activity grid: {d} cells changed"))

    # 2e. Zero-diff (flatline) frames
    for i, d in enumerate(diffs):
        if d == 0 and i > 0:
            flagged.append((i, "flatline (0 cells changed)"))

    # 2f. Rule lifecycle milestones (promotion / refutation) — reuse engine_events_all
    for logger, msg, frame in engine_events_all:
        if "confirm_rules: promotion" in msg:
            flagged.append((frame, f"rule PROMOTED: {msg[len('confirm_rules: promotion '):][:80]}"))
        elif "refute_rules: moved" in msg:
            m = re.search(r"moved (\d+)", msg)
            n = m.group(1) if m else "?"
            flagged.append((frame, f"{n} rules REFUTED"))

    # Dedupe and sort
    seen: set[tuple[int, str]] = set()
    unique_flagged: list[tuple[int, str]] = []
    for f, r in flagged:
        key = (f, r)
        if key not in seen:
            seen.add(key)
            unique_flagged.append((f, r))
    unique_flagged.sort(key=lambda x: (x[0], x[1]))

    if unique_flagged:
        w("| Frame | Event |")
        w("|-------|-------|")
        for f, r in unique_flagged[:60]:
            w(f"| {f} | {r} |")
        if len(unique_flagged) > 60:
            w(f"\n*({len(unique_flagged) - 60} more events not shown)*")
    else:
        w("*No key frames flagged.*")
    w("")

    w("## 3. Rules")
    w("")
    inject_count = sum(1 for log in logs if log["logger"] == "effects.engine" and "inject_llm_proposals" in log["msg"])
    bump_count = sum(1 for log in logs if log["logger"] == "effects.engine" and "confirm_rules: bumped" in log["msg"])
    promote_count = sum(1 for log in logs if "confirm_rules: promotion" in log["msg"])
    refute_count = sum(1 for log in logs if "refute_rules" in log["msg"])
    prune_count = sum(1 for log in logs if "prune_rules" in log["msg"])
    w(f"- **Proposals injected**: {inject_count}")
    w(f"- **Rules bumped (support incremented)**: {bump_count}")
    w(f"- **Rules promoted (proposed -> confirmed)**: {promote_count}")
    w(f"- **Rules refuted (confirmed -> refuted)**: {refute_count}")
    w(f"- **Rules pruned**: {prune_count}")
    w("")

    w("### Confirmed rule counts over time")
    w("")
    w("| Frame | movement | collision | terminal | relational | proposed | refuted |")
    w("|-------|----------|-----------|----------|------------|----------|---------|")
    checkpoints = sorted(set([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]))
    for i in checkpoints:
        if i >= len(frames):
            continue
        ec = frames[i].get("scene_state", {}).get("effect_context", {})
        if not ec:
            continue
        row = [str(i)]
        for bucket in ("movement_rules", "collision_rules", "terminal_rules",
                        "relational_rules", "proposed_rules", "refuted_rules"):
            row.append(str(len(ec.get(bucket, []))))
        w("| " + " | ".join(row) + " |")
    w("")

    if frames:
        ec = frames[-1].get("scene_state", {}).get("effect_context", {})
        if ec:
            w("### Confirmed rules (final frame)")
            w("")
            any_confirmed = False
            for bucket in ("movement_rules", "collision_rules", "terminal_rules", "relational_rules"):
                rules = ec.get(bucket, [])
                if rules:
                    any_confirmed = True
                    w(f"**{bucket}** ({len(rules)}):")
                    w("")
                    for r in rules[:10]:
                        w(f"- `{_fmt_rule(r)}`")
                    if len(rules) > 10:
                        w(f"- ... ({len(rules) - 10} more)")
                    w("")
            if not any_confirmed:
                w("*No confirmed rules at any bucket.* The agent learned nothing stable.")
                w("")

    if frames:
        ec = frames[-1].get("scene_state", {}).get("effect_context", {})
        refuted = ec.get("refuted_rules", [])
        if refuted:
            w("### Refuted rules")
            w("")
            w(f"**{len(refuted)} rules** were confirmed then refuted:")
            w("")
            for r in refuted[:15]:
                w(f"- `{_fmt_rule(r)}` (support={r.get('support', '?')})")
            if len(refuted) > 15:
                w(f"- ... ({len(refuted) - 15} more)")
            w("")

    if frames:
        ec = frames[-1].get("scene_state", {}).get("effect_context", {})
        proposed = ec.get("proposed_rules", [])
        if proposed:
            w("### Proposed rules (never confirmed)")
            w("")
            w(f"**{len(proposed)} rules** sit in proposed state. Sample (first 10):")
            w("")
            for r in proposed[:10]:
                w(f"- `{_fmt_rule(r)}` (support={r.get('support', 0)})")
            if len(proposed) > 10:
                w(f"- ... ({len(proposed) - 10} more)")
            w("")

    # --- 4. LLM call summary ---
    w("## 4. LLM Call Summary")
    w("")
    if not llm_calls:
        w("*No .llm.jsonl sidecar found.*")
        w("")
    else:
        kind_dist = Counter(c["kind"] for c in llm_calls)
        trigger_dist = Counter(c.get("trigger", "") for c in llm_calls)
        w(f"- **By kind**: {dict(sorted(kind_dist.items()))}")
        w(f"- **By trigger**: {dict(sorted(trigger_dist.items()))}")
        w(f"- **Truncated**: {sum(1 for c in llm_calls if c.get('truncated'))}")
        # Per-frame call count (top frames)
        frame_calls = Counter(c["frame_index"] for c in llm_calls)
        busy = frame_calls.most_common(5)
        w(f"- **Most LLM calls per frame**: {dict(busy)}")
        w("")
        # Prompt size breakdown
        w("### Prompt size breakdown")
        w("")
        trunc_calls = [c for c in llm_calls if c.get("truncated")]
        if trunc_calls:
            sizes = [sum(_msg_chars(m) for m in c["messages"]) for c in trunc_calls]
            w(f"- **Truncated calls prompt size**: min={min(sizes)}, median={statistics.median(sizes):.0f}, max={max(sizes)} chars")
            biggest = max(trunc_calls, key=lambda c: sum(_msg_chars(m) for m in c["messages"]))
            total = sum(_msg_chars(m) for m in biggest["messages"])
            w(f"- **Largest call**: frame={biggest['frame_index']} kind={biggest['kind']} total={total} chars")
            for i, m in enumerate(biggest["messages"]):
                role = m.get("role", "?")
                cs = _msg_chars(m)
                w(f"  - msg[{i}] role={role}: {cs} chars")
            w("")
        w("| Frame | Kind | Trigger | Latency(ms) | Truncated |")
        w("|-------|------|---------|------------|-----------|")
        for c in llm_calls[:40]:
            w(f"| {c['frame_index']} | {c['kind']} | {c.get('trigger','')[:30]} | {c['latency_ms']} | {'yes' if c.get('truncated') else ''} |")
        if len(llm_calls) > 40:
            w(f"\n*({len(llm_calls) - 40} more calls not shown — use jq on .llm.jsonl for details)*")
        w("")

    # --- 5. Entity lifecycle ---
    w("## 5. Entity Lifecycle")
    w("")
    controllable_changes: list[str] = []
    seen_changes: set[str] = set()
    for log in logs:
        if "CONTROLLABLE ID CHANGED" in log["msg"]:
            key = log["msg"][:120]
            if key not in seen_changes:
                seen_changes.add(key)
                controllable_changes.append(f"- {key}")
    if controllable_changes:
        w("**Controllable ID changes:**")
        w("")
        for c in controllable_changes:
            w(c)
    else:
        w("*No controllable ID changes.*")
    w("")
    # Entity count over time (from scene_state)
    if frames:
        counts: list[int] = []
        for d in frames:
            scene = d.get("scene_state", {}).get("scene", {})
            counts.append(scene.get("n_entities", 0))
        if counts:
            w(f"- **Entity count**: min={min(counts)}, max={max(counts)}, last={counts[-1]}")
            # Frames where entity count changed
            changes = [(i, counts[i]) for i in range(1, len(counts)) if counts[i] != counts[i-1]]
            if changes:
                w(f"- **Entity count changes**: {len(changes)} frames")
                for i, c in changes[:10]:
                    w(f"  - frame={i}: {counts[i-1]} -> {c}")
            w("")

    # --- 6. Anomalies / warnings ---
    w("## 6. Warnings")
    w("")
    # Dedupe by (logger, msg): entity.builder emits each WARNING twice per frame
    seen_warns: set[tuple[str, str]] = set()
    warns: list[tuple[str, str, str]] = []
    for log in logs:
        if log["level"] != "WARNING":
            continue
        key = (log["logger"], log["msg"])
        if key in seen_warns:
            continue
        seen_warns.add(key)
        warns.append((log["ts"], log["logger"], log["msg"]))
    if warns:
        w("| Timestamp | Logger | Message |")
        w("|-----------|--------|---------|")
        for ts, lg, m in warns:
            w(f"| {ts[11:]} | {lg} | {m[:160]} |")
    else:
        w("*No WARNING-level log lines.*")
    w("")

    # --- 7. Grid diff stats ---
    w("## 7. Grid Diff Stats")
    w("")
    if diffs:
        w(f"- **Cell changes per frame**: min={min(diffs)}, median={statistics.median(diffs):.0f}, max={max(diffs)}, mean={sum(diffs)/len(diffs):.1f}")
        zero_frames = [i for i, d in enumerate(diffs) if d == 0]
        w(f"- **Zero-diff (flatline) frames**: {len(zero_frames)} ({zero_frames[:15]})")
        w("")
        # Mini ASCII sparkline of diffs
        w("```")
        max_d = max(diffs) if max(diffs) > 0 else 1
        bar_width = 60
        for i, d in enumerate(diffs):
            bar = "#" * int(d / max_d * bar_width) if d > 0 else ""
            if i % 5 == 0 or d == max_d or d == 0:
                w(f"f{i:3d} [{d:4d}] {bar}")
        w("```")
        w("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for a single ARC-AGI-3 recording.")
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument("-o", "--output", help="Write report to file (default: stdout)")
    args = parser.parse_args()

    rec_path = args.recording
    if not os.path.isfile(rec_path):
        sys.exit(f"Recording not found: {rec_path}")

    # Derive sidecar paths
    base = rec_path[: -len(".recording.jsonl")] if rec_path.endswith(".recording.jsonl") else rec_path
    llm_path = base + ".llm.jsonl"
    logs_path = base + ".logs.log"

    frames = load_recording(rec_path)
    llm_calls = load_llm_log(llm_path)
    logs = load_structured_logs(logs_path)

    report = render_report(rec_path, frames, llm_calls, logs)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()