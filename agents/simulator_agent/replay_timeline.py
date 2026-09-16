"""The timeline walker: rebuild the agent/sandbox state at a marked turn.

Verbatim-moved from ``reconstruction.py`` (guardrail split): ``reconstruct()``
walks recording lines 0..turn_frame in order, re-executing every recorded
python tool call at its recorded frame against the corpus as it stood then,
serving frames from the deterministic replay (queue over ``harness.frames``,
read-only) and growing the corpus via the sandbox's own ``action()``
machinery. ``reconstruction.py`` owns the marker/state contract, readers,
parser, mapping table, and ``verify_reconstruction``.

Index conventions (frames ↔ grids ↔ actions, transition pairing, the
virtual RESET pair) are owned by ``agents/simulator_agent/reset_policy.py``
— its module docstring is the canonical mapping table; cite it, never
restate it here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.simulator_agent.agent import ITER0_RESET_PROVENANCE
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.reconstruction import (
    ReconstructedState,
    ReplayMarker,
    _count_llm_calls,
    _embedded_state,
    _row_tool_calls,
    is_legacy_mask_code,
    load_llm_rows,
)
from agents.simulator_agent.reset_policy import RESET_ACTION, is_reset
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController
from replay.harness import ReplayHarness

logger = logging.getLogger(__name__)

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
            keys = ["frames_correct", "frames_total", "overall_accuracy"]
            if (
                "abstained_changed" in emb.last_check_result
                and "abstained_changed" in lcr
            ):
                keys.append("abstained_changed")
            for key in keys:
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
    legacy_skipped: list[int] = []

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
                    if is_legacy_mask_code(code):
                        seq = row.get("seq")
                        legacy_skipped.append(seq if isinstance(seq, int) else -1)
                        logger.warning(
                            "reconstruct: seq %s legacy mask era — set_ignore call "
                            "NOT re-executed (tool removed); fidelity for affected "
                            "turns: unsupported (legacy mask era)",
                            seq,
                        )
                        continue
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

    if legacy_skipped:
        print(
            f"[reconstruct] LEGACY MASK ERA: {len(legacy_skipped)} recorded "
            f"set_ignore call(s) at seq {legacy_skipped} were NOT re-executed "
            "(tool removed from the sandbox). Fidelity verdict for affected "
            "turns: unsupported (legacy mask era).",
            flush=True,
        )

    return ReconstructedState(
        harness=harness,
        sandbox=sandbox,
        workflow=workflow,
        messages=messages,
        frame_index=marker.turn_frame,
        llm_calls=_count_llm_calls(rows_all, marker.turn_seq - 1),
        simulate_source=sandbox._simulate_source or "",  # noqa: SLF001
        notes=dict(world_model),
        history_turns=history_turns,
        diff_images=diff_images,
        marker=marker,
        recording_path=recording_path,
        verify={},
        legacy_mask_era=bool(legacy_skipped),
        legacy_skipped_seqs=tuple(legacy_skipped),
    )
