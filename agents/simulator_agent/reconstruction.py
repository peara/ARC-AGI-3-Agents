"""Timeline-driven reconstruction of the agent/sandbox state at a marked turn.

Given a recording and its sibling ``.llm.jsonl`` log, rebuild the world
(via ``ReplayHarness``), the sandbox (registered simulate, corpus), the
workflow phase, and the LLM conversation exactly as they stood at the
start of the marked turn.

Legacy mask era: recordings made before the ``set_ignore`` removal may
contain recorded python tool calls that invoke ``set_ignore(``. Those
calls NameError in the current sandbox, so the walker (``replay_timeline.
reconstruct``) SKIPS their re-execution, marks the run
``legacy_mask_era`` on the returned state, and reports the affected
turns' fidelity as ``unsupported (legacy mask era)`` — loudly, never as
a crash.

Design: **timeline walker** — walk recording lines 0..turn_frame in order,
re-executing every recorded python tool call at its recorded frame against
the corpus as it stood then (seq order within a frame). The walker itself
lives in ``replay_timeline.py``; this module owns the marker/state contract,
readers, parser, mapping table, and verify. Frames are served
from the deterministic replay (a queue over ``harness.frames``, read-only);
the corpus grows via the sandbox's own ``action()`` machinery, exactly as
the live agent grew it. No incident constants: the marker
(``ReplayMarker(turn_frame, turn_seq)``) is required, and every expected
value is derived from the recording's own embedded
``scene_state.simulator_state`` ground truth.

Index conventions (frames ↔ grids ↔ actions, transition pairing, the
virtual RESET pair) are owned by ``agents/simulator_agent/reset_policy.py``
— its module docstring is the canonical mapping table; cite it, never
restate it here.

Per-frame verification: at each recording line carrying an embedded
``simulator_state`` block, the walker compares its own state against the
recording's claim (corpus length, history length, simulate registration,
cached check result) and collects mismatches; all mismatches are raised
together at the end of the walk as a single table (collect-then-raise —
drift is caught at the earliest line without losing the full picture).

Not reconstructed (deliberately):
  - ``_objects``/``_adjacency`` segmentation caches (agent-side, cosmetic;
    the sandbox does not consume them)
  - trimmed-message internals beyond what the log preserved
  - image re-rendering (``[image omitted]`` placeholders stay)
  - phase crosscheck beyond the walk-derived phase (``phase_crosscheck``
    in ``verify_reconstruction`` is best-effort advisory, never a hard
    gate)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.reset_policy import RESET_ACTION
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import WorkflowController
from replay.harness import ReplayHarness

logger = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayMarker:
    """Position of the replayed turn inside the recording + LLM log.

    ``turn_frame`` is the recording line index of the turn's observation
    grid; ``turn_seq`` is the LLM-log seq of the turn's opening
    conversation. Both required — no defaults (no incident anchors).
    """

    turn_frame: int
    turn_seq: int


@dataclass
class ReconstructedState:
    """Everything needed to resume the agent mid-run at the marked turn."""

    harness: ReplayHarness
    sandbox: SimulatorSandbox
    workflow: WorkflowController
    messages: list[dict[str, Any]]  # conversation prefix (marked seq, verbatim)
    frame_index: int
    llm_calls: int  # LLM calls consumed before this turn
    simulate_source: str  # the simulate code registered during the walk
    notes: dict[str, str]  # world-model notes at turn start
    history_turns: list[dict[str, Any]]  # agent-owned history entries
    diff_images: list[dict[str, str]]  # batch predict-and-compare diffs
    marker: ReplayMarker  # the recording this state was reconstructed from
    recording_path: Path
    verify: dict[str, Any] = field(default_factory=dict)
    legacy_mask_era: bool = False  # recorded set_ignore( calls found (pre-removal era)
    legacy_skipped_seqs: tuple[int, ...] = ()  # seqs whose re-execution was skipped


# ── Artifact loading ──────────────────────────────────────────────────────


def load_llm_rows(llm_path: Path) -> list[dict[str, Any]]:
    """Load all rows from a .llm.jsonl sidecar."""
    rows: list[dict[str, Any]] = []
    with open(llm_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tool_code(row: dict[str, Any], tool_name: str) -> str:
    """Extract the ``code`` argument of the first *tool_name* call in a row."""
    for tc in row.get("tool_calls") or []:
        fn = tc.get("function", {})
        if fn.get("name") == tool_name:
            code: Any = json.loads(fn.get("arguments", "{}")).get("code", "")
            return str(code)
    raise ValueError(f"no {tool_name} call found in row")


def _row_tool_calls(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(tool_name, arguments)`` for every tool call in a row.

    Tolerates ``tool_calls: null``, non-dict entries, and multiple calls
    per row (iterate — do not assume exactly one python call).
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for tc in row.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        calls.append((name, args if isinstance(args, dict) else {}))
    return calls


def is_legacy_mask_code(code: str) -> bool:
    """True when a recorded python snippet invokes the removed ``set_ignore``.

    Substring match on ``set_ignore(`` — the tool is gone from the sandbox
    (UNKNOWN abstention replaced it), so re-executing such a snippet would
    NameError. The walker skips these calls and flags the run
    ``legacy_mask_era`` instead.
    """
    return "set_ignore(" in code


# ── Embedded state + flow-message readers ─────────────────────────────────


@dataclass(frozen=True)
class EmbeddedState:
    """The 6 ``simulator_state`` fields the agent embeds in each recording line.

    Field names match the writer (``agent._extra_record_data``) byte-for-byte.
    ``history_turns`` is a COUNT (``len()``), not a list. All fields are
    ``None`` when the line carries no ``simulator_state`` block (early lines,
    non-simulator agents).
    """

    world_model: dict[str, Any] | None = None
    history_turns: int | None = None
    has_simulate: bool | None = None
    simulate_source: str | None = None
    last_check_result: dict[str, Any] | None = None
    n_collected_frames: int | None = None


@dataclass(frozen=True)
class FlowRegion:
    """One diff region from an exception-flow message, in message order."""

    bbox: tuple[int, int, int, int]  # (row_start, col_start, row_end, col_end)
    n_cells: int
    transitions: tuple[str, ...]  # "N->M" colour transitions, verbatim order


@dataclass(frozen=True)
class FlowFacts:
    """Parsed exception-flow message: total diff cells + per-region facts."""

    n_cells: int
    n_regions: int
    regions: tuple[FlowRegion, ...]


def _embedded_state(line_data: dict[str, Any]) -> EmbeddedState | None:
    """Read the 6 ``simulator_state`` fields from a recording line.

    Path: ``line["data"]["scene_state"]["simulator_state"]``. Returns
    ``None`` when the block is absent; never raises on schema drift
    (missing keys read as ``None`` fields).
    """
    ss = line_data.get("data", {}).get("scene_state", {}).get("simulator_state")
    if not isinstance(ss, dict):
        return None
    return EmbeddedState(
        world_model=ss.get("world_model"),
        history_turns=ss.get("history_turns"),
        has_simulate=ss.get("has_simulate"),
        simulate_source=ss.get("simulate_source"),
        last_check_result=ss.get("last_check_result"),
        n_collected_frames=ss.get("n_collected_frames"),
    )


_FLOW_HEADER_RE = re.compile(r"differs from reality by (\d+) cells in (\d+) regions?:")
_FLOW_REGION_RE = re.compile(
    r"rows (\d+)-(\d+) cols (\d+)-(\d+): (\d+) cells(?: \(([^)]*)\))?"
)


def parse_flow_message(messages: list[dict[str, Any]]) -> FlowFacts | None:
    """Parse the LAST user message containing "EXCEPTION FLOW".

    Extracts ``(n_cells, n_regions, regions)`` from the recorded format
    "differs from reality by N cells in M regions: rows 10-19 cols 34-38:
    47 cells (3->9, 9->5); ...". Regions keep message order (NOT sorted).
    Returns ``None`` when no flow message is present.
    """
    content: str | None = None
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and "EXCEPTION FLOW" in msg["content"]
        ):
            content = msg["content"]  # keep scanning — last one wins
    if content is None:
        return None

    header = _FLOW_HEADER_RE.search(content)
    if header is None:
        return None
    n_cells, n_regions = int(header.group(1)), int(header.group(2))

    regions: list[FlowRegion] = []
    for m in _FLOW_REGION_RE.finditer(content):
        transitions = (
            tuple(t.strip() for t in m.group(6).split(",") if t.strip())
            if m.group(6)
            else ()
        )
        regions.append(
            FlowRegion(
                bbox=(
                    int(m.group(1)),
                    int(m.group(3)),
                    int(m.group(2)),
                    int(m.group(4)),
                ),
                n_cells=int(m.group(5)),
                transitions=transitions,
            )
        )
    return FlowFacts(n_cells=n_cells, n_regions=n_regions, regions=tuple(regions))


def _count_llm_calls(rows: list[dict[str, Any]], upto_seq: int) -> int:
    """Count LLM calls with seq <= upto_seq (inclusive)."""
    return sum(1 for r in rows if r.get("seq", 0) <= upto_seq)


# ── Corpus/history mapping contract (pinned empirically, task 1) ──────────
#
# Recording line N → sandbox corpus / agent history lengths, pinned against
# recording eba2a894 (ls20-9607627b, 100 lines; verified at lines 0, 4, 5,
# 12, 13 and swept across all 100 lines — evidence:
# ``.omo/evidence/task-1-mapping-pin.txt``).
#
# Index conventions (frames ↔ grids ↔ actions, transition pairing, the
# virtual RESET pair) are owned by ``agents/simulator_agent/reset_policy.py``
# — its module docstring is the canonical mapping table; cite it, never
# restate it here.
#
# The embedded counts (``.data.scene_state.simulator_state``, written by
# ``agent._extra_record_data``) are LIVE-VERBATIM lengths:
#
#   n_collected_frames == len(sandbox._grids)        (agent.py:1182)
#   history_turns      == len(agent._history_turns)  (agent.py:1178)
#
# so the derived lengths are identities, not estimates:
#
#   len(_grids)          == n_collected_frames
#   len(_actions)        == n_collected_frames - 1   (pair-seeded shape
#                          invariant len(actions) == len(grids) - 1)
#   len(history_turns)   == history_turns            (embedded count INCLUDES
#                          the iter-0 RESET provenance entry — same list the
#                          agent appends to; no off-by-one vs the agent list)
#
# Line-index formulas (eba2a894, level 1, no mid-run level transition):
#
#   line 0 (the RESET observation, full_reset=True): n_collected_frames == 0,
#     history_turns == 0 — the recorder hook fires BEFORE the turn's
#     update_state seeding, so the virtual RESET pair is not yet in the corpus.
#   line N >= 1: n_collected_frames == N + 1; history_turns == min(N, 30)
#     (max_hist=30 trim, agent.py:826 — eba2a894 lines 31+ pin h at 30 while
#     n keeps growing).
#
# Timing (docs/reports/recording-format.md §3): the embedded counts are
# PRE-action for line N — scene_state[N] is the pre-action perception pass,
# frame[N] is the post-action grid produced by action_input[N].id. The
# post-action corpus at the END of turn N has len(_grids) == N + 2 (virtual
# RESET pair + N action grids).
#
# Plan-hypothesis refutation (recorded, live behavior wins): the plan's
# ``len(_grids) == n_collected_frames + 1`` / ``len(_actions) ==
# history_turns + 1`` does NOT hold on the live path. The virtual RESET pair
# is seeded by ``update_state(reset_seeded=True)`` (sandbox.py:1000-1013)
# BEFORE any ``action()`` append, so action()'s first-append branch
# (sandbox.py:442-444) never fires and n_collected_frames already counts the
# pair. Prior convention datapoint (old test: 13 grids / 12 actions at
# turn_frame=11) is CONSISTENT with this table — line 12 embeds n=13, h=12 →
# grids=13, actions=12 — superseded only in provenance (live pair-seed, not
# offline replay-all).
#
# Reconstruction sources per line N (timeline walker inputs):
#
#   _current_frame  ← settled_board(recording line N's ``frame``) — the
#                     post-action grid of turn N-1's action, which is the
#                     observation turn N opens with
#   _previous_grid  ← settled_board(recording line N-1's ``frame``)
#                     (None at line 0 — no prior frame)


# ── Main entry ────────────────────────────────────────────────────────────


def verify_reconstruction(state: ReconstructedState) -> dict[str, Any]:
    """Run all fidelity checks against the recording's embedded ground truth.

    Every expected value is read from artifacts at runtime — the marked
    line's embedded ``simulator_state`` and the parsed flow message in the
    marked turn's conversation. Zero incident literals. Returns a dict and
    NEVER raises on mismatch: checks report values; tests assert.

    Checks and their artifact sources:
      - corpus_shape: walker-final corpus/history lengths vs the pinned
        formulas at ``turn_frame`` (mapping table above), cross-checked
        against the embedded PRE-append counts (offset: the marked line's
        own transition is appended after the embedded capture).
      - simulate_source_match: ``sandbox._simulate_source`` byte-equal to
        the embedded ``simulate_source`` at the marked line.
      - stale_check_match: the CACHED ``_last_check_result`` key fields ==
        the embedded ``last_check_result`` (the LLM sees the cached check,
        not a re-check; verify never calls check(), which would overwrite
        the cache).
      - has_simulate: walker registration == embedded ``has_simulate``.
      - flow_match: predict_and_compare re-fired on the final transition
        (action derived from the corpus tail, not hardcoded) vs the parsed
        flow message; region bboxes included for diagnosis.
      - phase: walk-derived workflow phase; ``phase_crosscheck`` is a
        best-effort advisory parse of the phase directive in the marked
        turn's recorded prompt (never a hard gate).
      - n_messages / last_user_has_exception: conversation shape.
    """
    sandbox = state.sandbox
    marker = state.marker
    checks: dict[str, Any] = {}

    # Marked line's embedded state (ground truth for this turn).
    emb: EmbeddedState | None = None
    with open(state.recording_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > marker.turn_frame:
                break
            if i == marker.turn_frame and line.strip():
                emb = _embedded_state(json.loads(line))
                break

    # 1. corpus_shape — POST-append walker lengths vs pinned formulas.
    tf = marker.turn_frame
    grids_len = len(sandbox._grids)  # noqa: SLF001
    actions_len = len(sandbox._actions)  # noqa: SLF001
    history_len = len(state.history_turns)
    expected = (0, 0, 0) if tf == 0 else (tf + 2, tf + 1, min(tf + 1, 30))
    checks["corpus_shape"] = {
        "ok": (grids_len, actions_len, history_len) == expected,
        "grids": (grids_len, expected[0]),
        "actions": (actions_len, expected[1]),
        "history": (history_len, expected[2]),
        "embedded_n_collected_frames": emb.n_collected_frames if emb else None,
        "embedded_history_turns": emb.history_turns if emb else None,
    }

    # 2. simulate_source_match — byte equality vs embedded.
    emb_src = emb.simulate_source if emb else None
    walker_src = sandbox._simulate_source or ""  # noqa: SLF001
    checks["simulate_source_match"] = (
        walker_src == emb_src if emb_src is not None else bool(walker_src)
    )
    checks["simulate_source_lengths"] = (
        len(walker_src),
        len(emb_src) if emb_src is not None else 0,
    )

    # 3. stale_check_match — CACHED check result vs embedded.
    #    Key set: the schema-2 intersection — the three legacy keys always
    #    compared; ``abstained_changed`` compared only when BOTH dicts carry
    #    it (legacy embedded dicts predate schema 2 and simply lack the key;
    #    ``.get`` on both sides means a missing key never raises).
    cached = sandbox._last_check_result or {}  # noqa: SLF001
    emb_lcr = emb.last_check_result if emb else None
    if emb_lcr is not None:
        keys = ["overall_accuracy", "frames_correct", "frames_total"]
        if "abstained_changed" in cached and "abstained_changed" in emb_lcr:
            keys.append("abstained_changed")
        checks["stale_check_match"] = all(cached.get(k) == emb_lcr.get(k) for k in keys)
    else:
        checks["stale_check_match"] = not cached
    checks["stale_check_pair"] = (
        (
            cached.get("overall_accuracy"),
            cached.get("frames_correct"),
            cached.get("frames_total"),
            cached.get("abstained_changed"),
        ),
        (
            emb_lcr.get("overall_accuracy") if emb_lcr else None,
            emb_lcr.get("frames_correct") if emb_lcr else None,
            emb_lcr.get("frames_total") if emb_lcr else None,
            emb_lcr.get("abstained_changed") if emb_lcr else None,
        ),
    )

    # 4. has_simulate — walker registration vs embedded claim.
    walker_has = sandbox._simulate is not None  # noqa: SLF001
    checks["has_simulate"] = (
        walker_has == emb.has_simulate
        if emb is not None and emb.has_simulate is not None
        else walker_has
    )

    # 5. flow_match — re-fire predict_and_compare on the final transition
    #    (the last appended corpus action produced the marked frame).
    action_id = (
        sandbox._actions[-1] if sandbox._actions else RESET_ACTION  # noqa: SLF001
    )
    prev = [list(r) for r in settled_board(state.harness.frames[-2].frame)]
    curr = [list(r) for r in settled_board(state.harness.frames[-1].frame)]
    sandbox._pending_exception_flow = None  # noqa: SLF001
    sandbox.predict_and_compare(prev, curr, action_id)
    pending = sandbox._pending_exception_flow  # noqa: SLF001
    parsed = parse_flow_message(state.messages)
    refired = (
        (pending.get("n_diff"), pending.get("n_regions")) if pending else (None, None)
    )
    parsed_pair = (parsed.n_cells, parsed.n_regions) if parsed else (None, None)
    checks["flow_match"] = {
        "ok": refired == parsed_pair,
        "n_cells": refired[0],
        "n_regions": refired[1],
        "parsed_n_cells": parsed_pair[0],
        "parsed_n_regions": parsed_pair[1],
        "regions": [
            (tuple(r["bbox"]), r["n_cells"])
            for r in (pending or {}).get("regions", [])  # type: ignore[call-overload]
        ],
    }

    # 6. phase — walk-derived; crosscheck advisory only.
    phase = state.workflow._phase  # noqa: SLF001
    checks["phase"] = phase.name if hasattr(phase, "name") else str(phase)

    def _msg_text(m: dict[str, Any]) -> str:
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(
                p.get("text", "")
                for p in c
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
        return ""

    user_text = " ".join(
        _msg_text(m)
        for m in state.messages
        if isinstance(m, dict) and m.get("role") == "user"
    )
    directive = re.search(r"PHASE[:\s]*([A-Z_]+)", user_text)
    checks["phase_crosscheck"] = {
        "found": directive.group(1) if directive else None,
        "matches": (directive.group(1) == checks["phase"]) if directive else None,
    }

    # 7. conversation shape.
    checks["n_messages"] = len(state.messages)
    last_user = [
        m for m in state.messages if isinstance(m, dict) and m.get("role") == "user"
    ]
    checks["last_user_has_exception"] = any(
        isinstance(m.get("content"), str) and "EXCEPTION FLOW" in m["content"]
        for m in last_user[-2:]
    )

    return checks
