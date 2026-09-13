"""Timeline-driven reconstruction of the agent/sandbox state at a marked turn.

Given a recording and its sibling ``.llm.jsonl`` log, rebuild the world
(via ``ReplayHarness``), the sandbox (registered simulate, corpus, ignore
mask), the workflow phase, and the LLM conversation exactly as they stood
at the start of the marked turn.

Design: **timeline walker** — walk recording lines 0..turn_frame in order,
re-executing every recorded python tool call at its recorded frame against
the corpus as it stood then (seq order within a frame). Frames are served
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.simulator_agent.agent import ITER0_RESET_PROVENANCE
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.reset_policy import RESET_ACTION, is_reset
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController
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
    set_ignore_source: str | None  # the set_ignore code, None when never run
    notes: dict[str, str]  # world-model notes at turn start
    history_turns: list[dict[str, Any]]  # agent-owned history entries
    diff_images: list[dict[str, str]]  # batch predict-and-compare diffs
    marker: ReplayMarker
    recording_path: Path  # the recording this state was reconstructed from
    verify: dict[str, Any] = field(default_factory=dict)


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


# ── Queue-serving step callback ───────────────────────────────────────────


def _make_timeline_step_callback(
    harness: ReplayHarness,
    sandbox: SimulatorSandbox,
    history_turns: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    *,
    turn_frame: int,
    mismatches: list[str],
) -> Callable[[int, dict[str, Any] | None], dict[str, Any]]:
    """Build a step_env_callback that serves recorded frames from a queue.

    Serves ``harness.frames[next_line + 1]`` (read-only; frames[i+1] ==
    recording line i) for prefix actions; falls back to ``harness.env.step``
    after the queue drains (post-incident continuation). Response dict shape
    mirrors ``agent._step_env_callback`` (objects/adjacency/grid/
    valid_actions/last_action_result/history/frame_layers) so
    ``sandbox.action()`` behaves like the live agent's.
    """
    cursor = {"next_line": 1}  # line 0 is produced by the RESET (pair-seed)

    def _verify_at_line(m: int) -> None:
        emb = _embedded_state(lines[m])
        if emb is None:
            return
        expected_n = 0 if m == 0 else m + 1
        expected_h = 0 if m == 0 else min(m, 30)
        n = len(sandbox._grids)  # noqa: SLF001
        h = len(history_turns)
        if n != expected_n:
            mismatches.append(
                f"line {m}: n_collected_frames embedded={expected_n} walker={n}"
            )
        if h != expected_h:
            mismatches.append(
                f"line {m}: history_turns embedded={expected_h} walker={h}"
            )
        if emb.has_simulate is not None and emb.has_simulate != (
            sandbox._simulate is not None  # noqa: SLF001
        ):
            mismatches.append(
                f"line {m}: has_simulate embedded={emb.has_simulate} "
                f"walker={sandbox._simulate is not None}"  # noqa: SLF001
            )
        if emb.simulate_source and sandbox._simulate_source != emb.simulate_source:  # noqa: SLF001
            mismatches.append(
                f"line {m}: simulate_source len embedded={len(emb.simulate_source)} "
                f"walker={len(sandbox._simulate_source or '')}"  # noqa: SLF001
            )
        if emb.last_check_result is not None:
            lcr = sandbox._last_check_result or {}  # noqa: SLF001
            for key in ("frames_correct", "frames_total", "overall_accuracy"):
                if emb.last_check_result.get(key) != lcr.get(key):
                    mismatches.append(
                        f"line {m}: last_check_result.{key} "
                        f"embedded={emb.last_check_result.get(key)} walker={lcr.get(key)}"
                    )

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        m = cursor["next_line"]
        if m > turn_frame:
            # Queue drained — post-incident continuation steps the live env.
            from arcengine import GameAction

            game_action = GameAction.from_id(action_id)
            frame = harness.env.step(game_action)
            if frame is None:
                raise RuntimeError(f"env.step returned None for action {action_id}")
            curr = [list(row) for row in settled_board(frame.frame)]
            prev = [list(row) for row in settled_board(harness.frames[-1].frame)]
            prev_levels = harness.frames[-1].levels_completed or 0
            curr_levels = frame.levels_completed or 0
            last = {
                "board_changed": prev != curr,
                "done": frame.state.value in ("GAME_OVER", "WIN"),
                "level_completed": curr_levels > prev_levels,
                "game_over": frame.state.value == "GAME_OVER",
                "run_complete": frame.state.value == "WIN",
                "reward": curr_levels - prev_levels,
                "valid_actions": list(frame.available_actions)
                if frame.available_actions
                else [],
            }
            return {
                "objects": (),
                "adjacency": frozenset(),
                "grid": curr,
                "valid_actions": last.get("valid_actions", [1, 2, 3, 4]),
                "last_action_result": last,
                "history": history_turns,
            }
        recorded = lines[m]["data"]["action_input"]["id"]
        expected = RESET_ACTION if is_reset(recorded) else int(recorded)
        if expected != action_id:
            raise RuntimeError(
                f"action-id mismatch at recording line {m}: "
                f"recorded {expected!r} != executed {action_id!r}"
            )
        cursor["next_line"] = m + 1
        frame = harness.frames[m + 1]  # frames[i+1] == recording line i
        curr = [list(row) for row in settled_board(frame.frame)]
        prev = [list(row) for row in settled_board(harness.frames[m].frame)]
        # Per-line verification BEFORE appends (live recorder-hook timing:
        # the embedded counts are PRE-action for line m).
        _verify_at_line(m)
        # History append (replicates agent._on_action_executed, inside the
        # callback — the live hook fires from step_env before the recorder).
        history_turns.append(
            {"action": expected, "frame_index": m, "frame": [row[:] for row in curr]}
        )
        if len(history_turns) > 30:  # max_hist=30 trim (agent.py:826)
            del history_turns[:-30]
        prev_levels = harness.frames[m].levels_completed or 0
        curr_levels = frame.levels_completed or 0
        last = {
            "board_changed": prev != curr,
            "done": frame.state.value in ("GAME_OVER", "WIN"),
            "level_completed": curr_levels > prev_levels,
            "game_over": frame.state.value == "GAME_OVER",
            "run_complete": frame.state.value == "WIN",
            "reward": curr_levels - prev_levels,
            "valid_actions": list(frame.available_actions)
            if frame.available_actions
            else [],
        }
        return {
            "objects": (),
            "adjacency": frozenset(),
            "grid": curr,
            "valid_actions": last.get("valid_actions", [1, 2, 3, 4]),
            "last_action_result": last,
            "history": history_turns,
            "frame_layers": [list(layer) for layer in frame.frame],
        }

    return step


# ── Main entry ────────────────────────────────────────────────────────────


def reconstruct(
    recording_path: Path,
    *,
    marker: ReplayMarker,
    seed: int = 0,
) -> ReconstructedState:
    """Rebuild the exact agent/sandbox state at the marked turn.

    Timeline walker: walks recording lines 0..marker.turn_frame in order,
    re-executing every recorded python tool call at its recorded frame
    against the corpus as it stood then (seq order within a frame). Frames
    are served from the deterministic replay (queue over harness.frames,
    read-only); the corpus grows via the sandbox's own action() machinery.
    Per-frame verification against the recording's embedded simulator_state
    runs at each line (collect-then-raise).
    """
    llm_path = Path(str(recording_path).replace(".recording.jsonl", ".llm.jsonl"))
    rows_all = load_llm_rows(llm_path)
    by_seq = {r["seq"]: r for r in rows_all}
    rows_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows_all:
        fi = row.get("frame_index")
        if isinstance(fi, int) and 0 <= fi <= marker.turn_frame:
            rows_by_frame.setdefault(fi, []).append(row)
    for fi in rows_by_frame:
        rows_by_frame[fi].sort(key=lambda r: r.get("seq", 0))

    lines: list[dict[str, Any]] = []
    with open(recording_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > marker.turn_frame:
                break
            if line.strip():
                lines.append(json.loads(line))
    if len(lines) < marker.turn_frame + 1:
        raise RuntimeError(
            f"recording has {len(lines)} lines, need {marker.turn_frame + 1}"
        )

    harness = ReplayHarness.from_recording(recording_path, seed=seed)
    harness.replay_to(marker.turn_frame + 1)

    mismatches: list[str] = []
    history_turns: list[dict[str, Any]] = []

    # Line 0 embedded state (pre-seeding: recorder hook fired before
    # update_state seeding — n=0, h=0).
    emb0 = _embedded_state(lines[0])
    if emb0 is not None:
        if emb0.n_collected_frames not in (None, 0):
            mismatches.append(
                f"line 0: n_collected_frames embedded={emb0.n_collected_frames} walker=0"
            )
        if emb0.history_turns not in (None, 0):
            mismatches.append(
                f"line 0: history_turns embedded={emb0.history_turns} walker=0"
            )

    # Virtual RESET pair (update_state(reset_seeded=True) equivalent):
    # B0 = line 0's frame == frames[1] == frames[0] board-wise.
    b0 = [list(row) for row in settled_board(harness.frames[1].frame)]
    holder: dict[str, Any] = {}
    sandbox = SimulatorSandbox(step_env_callback=lambda a, d: holder["step"](a, d))
    holder["step"] = _make_timeline_step_callback(
        harness,
        sandbox,
        history_turns,
        lines,
        turn_frame=marker.turn_frame,
        mismatches=mismatches,
    )
    sandbox._grids = [[row[:] for row in b0], [row[:] for row in b0]]  # noqa: SLF001
    sandbox._actions = [RESET_ACTION]  # noqa: SLF001
    sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
    history_turns.append(
        {
            "action": RESET_ACTION,
            "frame_index": 0,
            "frame": [row[:] for row in b0],
            "provenance": ITER0_RESET_PROVENANCE,
        }
    )
    sandbox.namespace["history"] = history_turns  # noqa: SLF001
    sandbox._current_frame = [row[:] for row in b0]  # noqa: SLF001
    sandbox.namespace["current_frame"] = sandbox._current_frame  # noqa: SLF001
    sandbox._previous_grid = None  # noqa: SLF001
    sandbox.namespace["previous_frame"] = None  # noqa: SLF001
    sandbox._last_action_result = {  # noqa: SLF001
        "board_changed": False,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "reward": 0,
        "valid_actions": [1, 2, 3, 4],
    }
    sandbox.namespace["last_action_result"] = sandbox._last_action_result  # noqa: SLF001
    sandbox.namespace["valid_actions"] = [1, 2, 3, 4]  # noqa: SLF001

    workflow = WorkflowController(sandbox)
    world_model: dict[str, str] = {"notes": "", "plan": ""}

    for n in range(0, marker.turn_frame + 1):
        for row in rows_by_frame.get(n, []):
            if n == marker.turn_frame and row.get("seq", 0) >= marker.turn_seq:
                continue  # the marked turn itself and later rows are the future
            if row.get("ok") is False:
                logger.warning("reconstruct: seq %s ok=False — skipped", row.get("seq"))
                continue
            for name, args in _row_tool_calls(row):
                if name == "python":
                    code = str(args.get("code", ""))
                    output, error, _ = sandbox.run_code(code)
                    if error:
                        mismatches.append(
                            f"seq {row.get('seq')}: python error {error!r}"
                        )
                    if (
                        "set_simulate" in code
                        and "[set_simulate] registered" not in output
                    ):
                        mismatches.append(
                            f"seq {row.get('seq')}: set_simulate did not register "
                            f"(output={output!r})"
                        )
                elif name == "update_notes":
                    if args.get("notes"):
                        world_model["notes"] = str(args["notes"])
                    if args.get("plan"):
                        world_model["plan"] = str(args["plan"])
                elif name == "set_phase":
                    phase_name = str(args.get("phase", "")).upper()
                    workflow._phase = Phase[phase_name]  # noqa: SLF001
                # inspect/reflect/decide/other tools: no sandbox state effect

    if mismatches:
        raise RuntimeError("per-frame verification failed:\n" + "\n".join(mismatches))

    messages = by_seq[marker.turn_seq]["messages"]
    diff_images = list(sandbox.pending_images)

    return ReconstructedState(
        harness=harness,
        sandbox=sandbox,
        workflow=workflow,
        messages=messages,
        frame_index=marker.turn_frame,
        llm_calls=_count_llm_calls(rows_all, marker.turn_seq - 1),
        simulate_source=sandbox._simulate_source or "",  # noqa: SLF001
        set_ignore_source=None,
        notes=dict(world_model),
        history_turns=history_turns,
        diff_images=diff_images,
        marker=marker,
        recording_path=recording_path,
        verify={},
    )


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
    cached = sandbox._last_check_result or {}  # noqa: SLF001
    emb_lcr = emb.last_check_result if emb else None
    if emb_lcr is not None:
        checks["stale_check_match"] = all(
            cached.get(k) == emb_lcr.get(k)
            for k in ("overall_accuracy", "frames_correct", "frames_total")
        )
    else:
        checks["stale_check_match"] = not cached
    checks["stale_check_pair"] = (
        (
            cached.get("overall_accuracy"),
            cached.get("frames_correct"),
            cached.get("frames_total"),
        ),
        (
            emb_lcr.get("overall_accuracy") if emb_lcr else None,
            emb_lcr.get("frames_correct") if emb_lcr else None,
            emb_lcr.get("frames_total") if emb_lcr else None,
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
        _msg_text(m) for m in state.messages if isinstance(m, dict) and m.get("role") == "user"
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
