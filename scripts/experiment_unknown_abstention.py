"""L3/L3b harness: UNKNOWN-abstention probe + offline full-level replay.

Two modes (--mode):

- ``probe`` (default, task 11): seed a real SimulatorSandbox at the
  recorded f20 state of run f9e3e301 (the corpus is pair-seeded
  [B0copy, B0, ...], so seeding "to recording transition 20" means 22
  corpus entries [0..21] — the seeding mirrors replay_timeline's corpus
  construction), register the task-9 fixture sim (the run's REAL movement
  sim + counterfactual UNKNOWN abstention on rows 61-62 all cols + rows
  53-62 cols 1-10, on every return path), then run N turns of the NEW
  prompt against the real local LLM. The probe question: at the recorded
  fork the real run masked the key box as "noise" (set_ignore at f22-24,
  right after its own blocked entries made the box flash). With
  abstention inside the model instead, does the model (a) investigate
  before abstaining, (b) write UNKNOWN, (c) read the abstained log
  (check()), (d) record region correlation data in notes?

- ``full-replay`` (task 12, L3b): seed at frame 0 with NO simulate
  registered — the model starts cold and must write its own simulate
  from scratch — then run the full level-1 window offline. The
  ReplayHarness steps the REAL game engine (the class loaded from
  environment_files), so the model's chosen actions get genuine next
  states, not recorded ones; divergence from the recording is expected
  and correct. The run ends on win (levels_completed bump), GAME_OVER
  (step-budget exhaustion — measured: board-reset flash at step 43,
  again at 86, GAME_OVER at 129), or the --turns cap. Per-turn
  measure() metrics land in metrics.csv.

Verdicts are RECORDED, never asserted — they are data for human review
of the transcript. Success bars (plus-touch, abstention ≤ ~30% of
changed cells after turn 15, notes accumulate region hypotheses) are
judgment calls on the recorded data, not assertions.

Offline: no live game server, no scorecard burn. ONE LLM call at a
time, strictly sequential — local LM Studio constraint (AGENTS.md
"Local LLM constraint"). LLM config is env-driven (LLM_BASE_URL /
LLM_MODEL / LLM_API_KEY), same client construction as
scripts/experiment_level_transition.py.

Dry-run mode (--dry-run) exercises the FULL seeding path plus one
scripted turn (no LLM) and writes the same output files, proving the
plumbing end-to-end — in both modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.llm_client import LLMClient
from agents.simulator_agent.check import UNKNOWN
from agents.simulator_agent.conversation import (
    SIM_STATE_HEADER,
    is_sim_state_block,
    strip_old_images,
    trim_messages_for_context,
)
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.prompts import (
    AGENT_PYTHON_TOOL_SCHEMA,
    AGENT_SYSTEM_PROMPT,
    BOARD_RESET_TEXT,
    UPDATE_NOTES_TOOL_SCHEMA,
    build_agent_user_prompt,
)
from agents.simulator_agent.reset_policy import RESET_ACTION, is_reset
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import SET_PHASE_TOOL_SCHEMA, WorkflowController
from agents.simulator_agent.world_model import format_notes
from replay.harness import ReplayHarness
from vision.render import grid_to_image, image_to_base64

# The task-9 fixture: the run's REAL registered sim (llm.jsonl seq 10) plus
# the counterfactual UNKNOWN abstention. The probe registers this EXACT
# source — read from the fixture file so the probe and the golden test can
# never diverge.
FIXTURE_SIM_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "unit"
    / "simulator_agent"
    / "fixtures"
    / "f9e3e301_simulate.py"
)

# The recorded run's world model at the fork (llm.jsonl seq 36, frame 16 —
# the last update_notes before the f22 masking decision). Seeded so the
# probe starts from the model's own prior beliefs, not a blank slate.
SEED_NOTES = (
    "DIRECTIONS: action 1=UP (5 cells), 2=DOWN, 3=LEFT, 4=RIGHT. Player "
    "moves 5 cells per action. Player 5x5. BOX: rows9-15, interior blue "
    "target at rows11-13 cols35-37. Box floor row16 is walkable(3). "
    "Corridor rows17-24 cols34-38. Box interior is 3 wide (cols35-37) but "
    "player is 5 wide (cols34-38) — cols34,38 are walls. Question: does "
    "player enter box (over walls) or stop at row17?"
)
SEED_PLAN = (
    "Take real action 1 several times to see if/where player enters box "
    "and whether win triggers."
)

# One scripted turn for --dry-run: exercises the investigation surface
# (print_region over the abstained region), the abstained log (check()),
# and notes — the exact tool surface the real turns use, without an LLM.
DRY_RUN_CODE = (
    "print_region(current_frame, 53, 63, 0, 12)\n"
    "r = check()\n"
    "print('coverage', r.get('coverage'), "
    "'abstained_changed', r.get('abstained_changed'))\n"
    "update_notes('dry-run: exercised seeding + check abstained log', "
    "'verify outputs')\n"
)

# Context budget mirrors the live agent (agent.py:123).
_CONTEXT_BUDGET_TOKENS = max(1024, 32768 - 4096 - 512)

# Sandbox exec cannot run these fixture lines: the dunder pattern blocks
# `from __future__`, and `agents` is not an allowed import (UNKNOWN is
# already a sandbox global — same constant, check.UNKNOWN).
_STRIP_IMPORT_LINES = ("from __future__", "from agents.simulator_agent.check")


def build_registration_code() -> str:
    """Fixture sim source + set_simulate call, execable in the sandbox.

    Reads the golden-tested fixture file and strips only the two import
    lines the sandbox cannot exec. Everything else — the verbatim recorded
    movement logic and the UNKNOWN wrapper — is byte-identical to what the
    golden test (test_golden_f9e3e301.py) asserts against.
    """
    src = FIXTURE_SIM_PATH.read_text(encoding="utf-8")
    kept = [
        line for line in src.splitlines() if not line.startswith(_STRIP_IMPORT_LINES)
    ]
    return "\n".join(kept) + "\nset_simulate(my_sim)\n"


# ── Seeding ────────────────────────────────────────────────────────────────


def _recording_lines(recording: Path, upto: int) -> list[dict[str, Any]]:
    """Recording lines 0..upto (inclusive), parsed."""
    lines: list[dict[str, Any]] = []
    with open(recording, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > upto:
                break
            if line.strip():
                lines.append(json.loads(line))
    if len(lines) < upto + 1:
        raise RuntimeError(f"recording has {len(lines)} lines, need {upto + 1}")
    return lines


def _frame_layers(frame: Any) -> list[list[list[int]]]:
    """Plain-int layer stack from a raw env FrameData.

    The env returns numpy arrays; ``[list(layer) for layer in ...]`` keeps
    ``np.int8`` cells, which ``frame_layers._is_board`` rejects — flash
    stacks then classify UNKNOWN and ``settled_board`` wrongly picks the
    uniform animation layer. ``.tolist()`` yields plain ints (same
    conversion ReplayHarness._convert_raw_frame_data applies).
    """
    return [arr.tolist() for arr in frame.frame]


def _last_action_result(
    prev: list[list[int]] | None, curr: list[list[int]], frame: Any
) -> dict[str, Any]:
    """last_action_result dict — mirrors agent._step_env_callback's shape."""
    return {
        "board_changed": prev is None or prev != curr,
        "done": frame.state.value in ("GAME_OVER", "WIN"),
        "level_completed": False,  # single-level prefix; levels never bump
        "game_over": frame.state.value == "GAME_OVER",
        "run_complete": frame.state.value == "WIN",
        "reward": 0,
        "valid_actions": list(frame.available_actions)
        if frame.available_actions
        else [],
    }


def _make_step_callback(
    harness: ReplayHarness,
    lines: list[dict[str, Any]],
    *,
    seed_frame: int,
    cursor: dict[str, Any],
) -> Any:
    """Step callback: recorded prefix (parity-checked queue), then the live
    offline env (replay_timeline's post-incident continuation pattern).

    Prefix serving is read-only: frames[i+1] == recording line i. Past the
    seed frame the model's actions step the real offline environment, so a
    model that repeats the recorded run's blocked entries sees the real
    key-box flash — the exact situation the probe measures.
    """

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        m = cursor["next_line"]
        if m <= seed_frame:
            recorded = lines[m]["data"]["action_input"]["id"]
            expected = RESET_ACTION if is_reset(recorded) else int(recorded)
            if expected != action_id:
                raise RuntimeError(
                    f"action-id mismatch at recording line {m}: "
                    f"recorded {expected!r} != executed {action_id!r}"
                )
            cursor["next_line"] = m + 1
            frame = harness.frames[m + 1]
            curr = [list(row) for row in settled_board(frame.frame)]
            prev = cursor["last_grid"]
            cursor["last_grid"] = curr
            last = _last_action_result(prev, curr, frame)
            last["board_changed"] = prev != curr
            return {
                "objects": (),
                "adjacency": frozenset(),
                "grid": curr,
                "valid_actions": last["valid_actions"],
                "last_action_result": last,
                "history": cursor["history"],
                "frame_layers": [list(layer) for layer in frame.frame],
            }
        # Post-seed: live offline env continuation.
        from arcengine import GameAction

        game_action = GameAction.from_id(action_id)
        frame = harness.env.step(game_action)
        if frame is None:
            raise RuntimeError(f"env.step returned None for action {action_id}")
        curr = [list(row) for row in settled_board(_frame_layers(frame))]
        prev = cursor["last_grid"]
        cursor["last_grid"] = curr
        cursor["next_line"] = m + 1
        last = _last_action_result(prev, curr, frame)
        prev_levels = harness.frames[-1].levels_completed or 0
        curr_levels = frame.levels_completed or 0
        last["level_completed"] = curr_levels > prev_levels
        last["reward"] = curr_levels - prev_levels
        return {
            "objects": (),
            "adjacency": frozenset(),
            "grid": curr,
            "valid_actions": last["valid_actions"],
            "last_action_result": last,
            "history": cursor["history"],
            "frame_layers": _frame_layers(frame),
        }

    return step


def seed_sandbox(
    recording: Path,
    *,
    seed_frame: int,
    timeout: float = 30.0,
) -> tuple[SimulatorSandbox, dict[str, Any]]:
    """Build a live-mode sandbox seeded to the recorded state at *seed_frame*.

    Corpus construction mirrors replay_timeline.reconstruct: the virtual
    RESET pair ([B0copy, B0], [RESET]) first, then recording lines 1..N
    replayed through the sandbox's own action() machinery so the corpus
    grows exactly like the live agent's. At seed_frame=20 this yields 22
    corpus entries [0..21] — transition 20 (the key-box flash: 76 cells,
    all abstained under the fixture sim) is the last seeded transition.

    Returns (sandbox, cursor) where cursor carries the mutable step state
    (line counter, history list, last grid) for turn-time inspection.
    """
    lines = _recording_lines(recording, seed_frame)
    harness = ReplayHarness.from_recording(recording, seed=0)
    harness.replay_to(seed_frame + 1)

    b0 = [list(row) for row in settled_board(harness.frames[1].frame)]
    history: list[dict[str, Any]] = [
        {
            "action": RESET_ACTION,
            "frame_index": 0,
            "frame": [row[:] for row in b0],
            "provenance": "iteration 0: env RESET — initial observation",
        }
    ]
    cursor: dict[str, Any] = {
        "next_line": 1,  # line 0 is the RESET (pair-seed)
        "last_grid": [row[:] for row in b0],
        "history": history,
    }
    sandbox = SimulatorSandbox(
        step_env_callback=_make_step_callback(
            harness, lines, seed_frame=seed_frame, cursor=cursor
        ),
        timeout=timeout,
    )
    # Virtual RESET pair — identical to replay_timeline's seeding.
    sandbox._grids = [[row[:] for row in b0], [row[:] for row in b0]]  # noqa: SLF001
    sandbox._actions = [RESET_ACTION]  # noqa: SLF001
    sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
    sandbox.namespace["history"] = history  # noqa: SLF001
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

    # Replay lines 1..seed_frame through action() so the corpus grows via
    # the sandbox's own machinery (predict_and_compare stays silent: no
    # simulate registered yet).
    for m in range(1, seed_frame + 1):
        recorded = lines[m]["data"]["action_input"]["id"]
        action_id = RESET_ACTION if is_reset(recorded) else int(recorded)
        response = sandbox.namespace["action"](action_id)
        history.append(
            {
                "action": action_id,
                "frame_index": m,
                "frame": [row[:] for row in response["grid"]],
            }
        )
        if len(history) > 30:
            del history[:-30]

    # Register the fixture sim through the REAL set_simulate path
    # (linecache source capture + globals freeze).
    output, error, _ = sandbox.run_code(build_registration_code())
    if error:
        raise RuntimeError(f"fixture sim registration failed: {error}\n{output}")
    if sandbox._simulate is None:  # noqa: SLF001
        raise RuntimeError("fixture sim did not register")

    return sandbox, cursor


def seed_full_replay(
    recording: Path,
    *,
    timeout: float = 30.0,
) -> tuple[SimulatorSandbox, dict[str, Any]]:
    """Build a live-mode sandbox at frame 0 with NO simulate registered.

    The cold-start seed for --mode full-replay: the virtual RESET pair
    ([B0copy, B0], [RESET]) only — the model must write its own simulate
    from scratch. The step callback drives the REAL offline env (the
    ReplayHarness's game class), so every action the model takes gets a
    genuine next state; divergence from the recording is expected.

    Returns (sandbox, cursor) — same contract as seed_sandbox.
    """
    harness = ReplayHarness.from_recording(recording, seed=0)
    harness.replay_to(1)

    b0 = [list(row) for row in settled_board(harness.frames[1].frame)]
    history: list[dict[str, Any]] = [
        {
            "action": RESET_ACTION,
            "frame_index": 0,
            "frame": [row[:] for row in b0],
            "provenance": "iteration 0: env RESET — initial observation",
        }
    ]
    cursor: dict[str, Any] = {
        "next_line": 1,
        "last_grid": [row[:] for row in b0],
        "history": history,
        "levels_completed": 0,
        "game_over": False,
    }

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        from arcengine import GameAction

        frame = harness.env.step(GameAction.from_id(action_id))
        if frame is None:
            raise RuntimeError(f"env.step returned None for action {action_id}")
        layers = _frame_layers(frame)
        curr = [list(row) for row in settled_board(layers)]
        prev = cursor["last_grid"]
        cursor["last_grid"] = curr
        cursor["next_line"] += 1
        prev_levels = cursor["levels_completed"]
        curr_levels = frame.levels_completed or 0
        cursor["levels_completed"] = max(prev_levels, curr_levels)
        cursor["game_over"] = frame.state.value in ("GAME_OVER", "WIN")
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
        history.append(
            {
                "action": action_id,
                "frame_index": cursor["next_line"],
                "frame": [row[:] for row in curr],
            }
        )
        if len(history) > 30:
            del history[:-30]
        return {
            "objects": (),
            "adjacency": frozenset(),
            "grid": curr,
            "valid_actions": last["valid_actions"],
            "last_action_result": last,
            "history": history,
            "frame_layers": layers,
        }

    sandbox = SimulatorSandbox(step_env_callback=step, timeout=timeout)
    # Virtual RESET pair — identical to seed_sandbox's seeding.
    sandbox._grids = [[row[:] for row in b0], [row[:] for row in b0]]  # noqa: SLF001
    sandbox._actions = [RESET_ACTION]  # noqa: SLF001
    sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
    sandbox.namespace["history"] = history  # noqa: SLF001
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

    return sandbox, cursor


# ── Prompt assembly (mirrors the live agent's turn shape) ──────────────────


def build_sim_state_block(sandbox: SimulatorSandbox) -> str:
    """[Simulator state] block — the NEW (mask-free) format.

    Mirrors agent._build_sim_state_block: simulate-only presence, full
    registered source. The abstention summary lives in the frame prompt's
    simulate_status (build_simulate_status below), rendered separately.
    """
    source = sandbox._simulate_source or ""  # noqa: SLF001
    lines = [f"{SIM_STATE_HEADER} (authoritative; refreshed every call)"]
    if source and source != "(source unavailable)":
        lines.append(
            "simulate: registered (frozen copy — helpers fixed at registration)"
        )
        lines.append(source)
    else:
        lines.append("simulate: registered (source not captured this session)")
    return "\n".join(lines)


def build_simulate_status(sandbox: SimulatorSandbox) -> str:
    """Simulate status line — mirrors agent._build_simulate_status.

    Renders ONLY from cached sandbox state (no recompute) so identical
    state yields byte-identical text (LM Studio prefix-cache stable).
    Legacy cached checks (no ``schema`` key) degrade silently.
    """
    if sandbox._simulate is None:  # noqa: SLF001
        return ""
    lines = ["Simulator: registered."]
    result = sandbox._last_check_result  # noqa: SLF001
    if result and "error" not in result:
        acc = result.get("overall_accuracy")
        wrong = result.get("wrong_cells", 0)
        correct = result.get("frames_correct", 0)
        total = result.get("frames_total", 0)
        if acc is not None:
            lines.append(
                f"Last check(): {acc:.1f}% accuracy, {wrong} wrong cells, "
                f"{correct}/{total} frames correct."
            )
        else:
            lines.append(f"Last check(): {correct}/{total} frames correct.")
        abst_changed = result.get("abstained_changed", 0)
        abst_stable = result.get("abstained_stable", 0)
        if result.get("schema") == 2 and abst_changed + abst_stable > 0:
            coverage = result.get("coverage")
            if coverage is not None:
                if coverage == 0:
                    lines.append(
                        "Last check(): covered 0% of changed cells — you "
                        "predicted NOTHING of what changed; this is not a pass"
                    )
                else:
                    committed = result.get("total_correct", 0) + result.get(
                        "total_wrong", 0
                    )
                    lines.append(
                        f"Last check(): covered {coverage:.1f}% of changed "
                        f"cells — modeled {committed / total:.1f} cells/frame, "
                        f"abstained {abst_changed + abst_stable} "
                        f"({abst_changed} changed)"
                    )
            clusters = result.get("abstained_clusters") or []
            lines.append(
                f"abstains: {len(clusters)} regions ({abst_changed} changed, "
                f"{abst_stable} stable) — see check() abstained log"
            )
    elif result and "error" in result:
        lines.append("Last check(): error.")
    else:
        lines.append("No check() run yet. Call check() to test it.")
    suppressed = getattr(sandbox, "_suppressed_abstained_diffs", 0)
    if isinstance(suppressed, int) and suppressed > 0:
        lines.append(
            f"NOTE: {suppressed} diffs on abstained cells were suppressed by "
            "predict_and_compare since your last check() — the world moved "
            "where you abstained; run check()"
        )
    return " ".join(lines)


def build_history_summary(history: list[dict[str, Any]]) -> str:
    """Recent history summary — mirrors agent._build_history_summary."""
    if not history:
        return ""
    entries = [
        f"Frame {t.get('frame_index', '?')}: action={t.get('action', '?')}"
        for t in history[-10:]
    ]
    return "Recent history:\n" + "\n".join(entries)


def _frame_user_content(
    sandbox: SimulatorSandbox,
    history: list[dict[str, Any]],
    workflow: WorkflowController,
    *,
    frame_index: int,
) -> list[dict[str, Any]]:
    """Frame user prompt blocks (phase directive + grid image + status)."""
    grid = sandbox._current_frame  # noqa: SLF001
    grid_b64 = image_to_base64(grid_to_image(grid, scale=8)) if grid else None
    return build_agent_user_prompt(
        grid_image_b64=grid_b64,
        world_model_text="",
        available_actions=list(sandbox.namespace.get("valid_actions", [])),
        frame_index=frame_index,
        history_summary=build_history_summary(history),
        simulate_status=build_simulate_status(sandbox),
        phase_directive=workflow.directive(),
    )


def build_initial_messages(
    sandbox: SimulatorSandbox,
    history: list[dict[str, Any]],
    world_model: dict[str, str],
    workflow: WorkflowController,
    *,
    frame_index: int,
) -> list[dict[str, Any]]:
    """Turn-1 message list: system + [Simulator state] + [Current notes] +
    the frame user prompt."""
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_sim_state_block(sandbox)},
        {"role": "user", "content": f"[Current notes]\n{format_notes(world_model)}"},
        *_frame_user_content(sandbox, history, workflow, frame_index=frame_index),
    ]


def refresh_state_messages(
    messages: list[dict[str, Any]],
    sandbox: SimulatorSandbox,
    world_model: dict[str, str],
) -> None:
    """Refresh the [Simulator state] block and [Current notes] message in
    place (agent._inject_sim_state_block + _update_notes_message shapes).

    The sim-state block is replaced at its existing position; when absent
    it is re-inserted at index 1. The notes message is replaced in place;
    when absent it is appended.
    """
    block = build_sim_state_block(sandbox)
    for msg in messages:
        if is_sim_state_block(msg):
            if msg["content"] != block:
                msg["content"] = block
            break
    else:
        messages.insert(1, {"role": "user", "content": block})
    notes_text = f"[Current notes]\n{format_notes(world_model)}"
    for msg in reversed(messages):
        if (
            msg.get("role") == "user"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith("[Current notes]")
        ):
            msg["content"] = notes_text
            return
    messages.append({"role": "user", "content": notes_text})


def append_frame_prompt(
    messages: list[dict[str, Any]],
    sandbox: SimulatorSandbox,
    history: list[dict[str, Any]],
    workflow: WorkflowController,
    *,
    frame_index: int,
) -> None:
    """Append a fresh frame user prompt (used when the grid changed mid-probe
    because the model took actions)."""
    messages.extend(
        _frame_user_content(sandbox, history, workflow, frame_index=frame_index)
    )
    strip_old_images(messages, keep_last_n_user=2)


# ── Verdict extraction (recorded, never asserted) ──────────────────────────


def _evidence_snippet(text: str, needle: str, width: int = 120) -> str:
    """One-line evidence snippet around the first *needle* occurrence."""
    idx = text.find(needle)
    if idx < 0:
        return ""
    start = max(0, idx - width // 2)
    return text[start : start + width].replace("\n", " ⏎ ")


def extract_verdicts(
    code: str,
    output: str,
    notes_text: str,
    response_content: str,
) -> dict[str, Any]:
    """Parse one turn's code/output/notes for the four procedure signals.

    Verdicts are DATA for human review — no expected behavior is asserted
    here. Each verdict carries its evidence snippet so the reviewer can
    judge without re-reading the full transcript.
    """
    investigated = "print_region(" in code or "diff(" in code
    wrote_unknown = "UNKNOWN" in code
    called_check = "check(" in code
    notes_carry_region_data = bool(
        notes_text
        and (
            "region" in notes_text.lower()
            or "abstain" in notes_text.lower()
            or "rows" in notes_text.lower()
        )
    )
    return {
        "investigated_before_abstaining": {
            "verdict": investigated,
            "evidence": _evidence_snippet(code, "print_region(")
            or _evidence_snippet(code, "diff("),
        },
        "wrote_unknown_in_code": {
            "verdict": wrote_unknown,
            "evidence": _evidence_snippet(code, "UNKNOWN"),
        },
        "called_check_read_abstained_log": {
            "verdict": called_check,
            "evidence": _evidence_snippet(code, "check("),
            "abstained_log_in_output": "Abstained regions" in output,
        },
        "notes_carry_region_correlation": {
            "verdict": notes_carry_region_data,
            "evidence": notes_text[:200],
        },
        "response_text_head": response_content[:200],
    }


# One scripted turn for --dry-run in full-replay mode: registers a trivial
# sim (identity + UNKNOWN on the timer rows — the abstention surface), takes
# one real action, and calls check() — proving the cold-seed + live-env +
# registration + scoring plumbing without an LLM.
FULL_REPLAY_DRY_RUN_CODE = (
    "def trivial_sim(g, a):\n"
    "    out = [row[:] for row in g]\n"
    "    for r in (61, 62):\n"
    "        for c in range(64):\n"
    "            out[r][c] = UNKNOWN\n"
    "    return out\n"
    "set_simulate(trivial_sim)\n"
    "action(1)\n"
    "r = check()\n"
    "print('coverage', r.get('coverage'))\n"
    "update_notes('dry-run: cold seed + trivial sim + one action', 'verify outputs')\n"
)

_METRICS_CSV_HEADER = (
    "turn,frame,accuracy,wrong_cells,abstained_changed,abstained_stable,"
    "coverage,latency_s\n"
)


# ── Probe runner ────────────────────────────────────────────────────────────


def run_probe(
    recording: Path,
    *,
    seed_frame: int,
    turns: int,
    out_dir: Path,
    dry_run: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Seed at *seed_frame*, run *turns* LLM turns (or one scripted turn in
    dry-run), write verdicts.json + transcript.jsonl under *out_dir*.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sandbox, cursor = seed_sandbox(recording, seed_frame=seed_frame, timeout=timeout)
    history: list[dict[str, Any]] = cursor["history"]

    # Seed-state verification (the dry-run stopping condition; printed in
    # both modes so the real run's transcript starts from proven state).
    n_corpus = len(sandbox._grids)  # noqa: SLF001
    expected_corpus = seed_frame + 2  # pair seed [B0copy, B0] + lines 1..seed_frame
    sim_registered = sandbox._simulate is not None  # noqa: SLF001
    unknown_in_ns = sandbox.namespace.get("UNKNOWN") == UNKNOWN
    print(f"[seed] corpus entries: {n_corpus} (expected {expected_corpus})")
    print(f"[seed] actions: {len(sandbox._actions)}")  # noqa: SLF001
    print(f"[seed] simulate registered: {sim_registered}")
    print(f"[seed] UNKNOWN in namespace: {unknown_in_ns}")
    print(f"[seed] suppressed_abstained_diffs: {sandbox._suppressed_abstained_diffs}")  # noqa: SLF001
    if n_corpus != expected_corpus:
        raise RuntimeError(
            f"seed corpus shape wrong: {n_corpus} != {expected_corpus} "
            "(pair-seeding contract violated)"
        )
    if not sim_registered:
        raise RuntimeError("no simulate registered at seed")
    if not unknown_in_ns:
        raise RuntimeError("UNKNOWN missing from sandbox namespace")

    workflow = WorkflowController(sandbox)
    workflow.set_phase("MODEL", reason="probe seed: simulate registered")
    world_model: dict[str, str] = {"notes": SEED_NOTES, "plan": SEED_PLAN}

    client: LLMClient | None = None if dry_run else LLMClient()
    verdicts: list[dict[str, Any]] = []
    transcript_path = out_dir / "transcript.jsonl"
    messages = build_initial_messages(
        sandbox,
        history,
        world_model,
        workflow,
        frame_index=n_corpus - 1,
    )

    with open(transcript_path, "w", encoding="utf-8") as transcript:
        for turn in range(1, turns + 1):
            label = "(dry-run scripted)" if dry_run else ""
            print(f"\n--- Turn {turn}/{turns} {label} ---")
            refresh_state_messages(messages, sandbox, world_model)
            outbound = trim_messages_for_context(
                messages, budget_tokens=_CONTEXT_BUDGET_TOKENS
            )

            if dry_run:
                code = DRY_RUN_CODE
                response_content = "(dry-run: scripted turn, no LLM call)"
                latency_s = 0.0
            else:
                assert client is not None  # narrowed by the dry_run flag
                t0 = time.time()
                try:
                    response = client.chat(
                        messages=outbound,
                        tools=[
                            AGENT_PYTHON_TOOL_SCHEMA,
                            UPDATE_NOTES_TOOL_SCHEMA,
                            SET_PHASE_TOOL_SCHEMA,
                        ],
                        tool_choice="auto",
                        max_tokens=4096,
                    )
                except Exception as exc:
                    # Record the failure per turn and continue — matches the
                    # existing experiment infra's error handling.
                    latency_s = time.time() - t0
                    print(f"LLM call failed: {exc}")
                    verdicts.append(
                        {
                            "turn": turn,
                            "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                            "code": "",
                            "output": "",
                            "error": str(exc),
                            "latency_s": round(latency_s, 1),
                            "metrics": None,
                            "verdicts": None,
                        }
                    )
                    transcript.write(
                        json.dumps({"turn": turn, "error": str(exc)}) + "\n"
                    )
                    continue
                latency_s = time.time() - t0
                response_content = response.content or ""
                code = ""
                for tc in response.tool_calls or []:
                    if tc["function"]["name"] == "python":
                        try:
                            code = json.loads(tc["function"]["arguments"]).get(
                                "code", ""
                            )
                        except Exception:
                            code = ""
                        break
                print(f"LLM latency: {latency_s:.1f}s")
                if not code and not response_content:
                    print("no tool call and no text — recording empty turn")

            # Execute the turn's code in the sandbox (both modes).
            grids_before = len(sandbox._grids)  # noqa: SLF001
            output, error, _acted = sandbox.run_code(code)

            # Sync notes recorded inside the snippet (sandbox update_notes)
            # + notes from the update_notes LLM tool call.
            pending = sandbox._pending_notes  # noqa: SLF001
            for key in ("notes", "plan"):
                if pending.get(key):
                    world_model[key] = pending[key]
            if not dry_run:
                for tc in response.tool_calls or []:
                    if tc["function"]["name"] == "update_notes":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}
                        if args.get("notes"):
                            world_model["notes"] = str(args["notes"])
                        if args.get("plan"):
                            world_model["plan"] = str(args["plan"])

            # Append assistant + tool result to the conversation (carried
            # across turns — the model sees its own history).
            tool_text = output if output else "(no output)"
            if error:
                tool_text = (
                    f"{tool_text}\nError: {error}" if tool_text else f"Error: {error}"
                )
            remaining = turns - turn
            tool_text = (
                f"{tool_text}\nTurn {turn}/{turns} completed. "
                f"{remaining} turns remaining."
            )
            if not dry_run:
                python_calls = [
                    tc
                    for tc in response.tool_calls or []
                    if tc["function"]["name"] == "python"
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_content or None,
                        "tool_calls": python_calls or None,
                    }
                )
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]
            for img in sandbox.pending_images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img['b64']}"},
                    }
                )
                content_parts.append({"type": "text", "text": img.get("caption", "")})
            sandbox.pending_images.clear()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"probe_{turn}",
                    "content": content_parts if len(content_parts) > 1 else tool_text,
                }
            )
            # Fresh frame prompt when the model acted (the grid changed).
            if len(sandbox._grids) != grids_before:  # noqa: SLF001
                append_frame_prompt(
                    messages,
                    sandbox,
                    history,
                    workflow,
                    frame_index=len(sandbox._grids) - 1,  # noqa: SLF001
                )

            verdict = extract_verdicts(
                code, output, world_model.get("notes", ""), response_content
            )
            metrics = sandbox.measure()
            verdicts.append(
                {
                    "turn": turn,
                    "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                    "code": code,
                    "output": output[:4000],
                    "error": error,
                    "latency_s": round(latency_s, 1),
                    "metrics": {
                        "accuracy": metrics.get("accuracy"),
                        "wrong_cells": metrics.get("wrong_cells"),
                        "abstained_changed": metrics.get("abstained_changed"),
                        "coverage": metrics.get("coverage"),
                        "frames_correct": metrics.get("frames_correct"),
                        "frames_total": metrics.get("frames_total"),
                    },
                    "verdicts": verdict,
                }
            )
            transcript.write(
                json.dumps(
                    {
                        "turn": turn,
                        "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                        "code": code,
                        "output": output[:4000],
                        "error": error,
                        "latency_s": round(latency_s, 1),
                        "response_content": response_content[:1000],
                        "messages": outbound,
                    },
                    default=str,
                )
                + "\n"
            )
            print(
                f"code chars: {len(code)} | output chars: {len(output)} | "
                f"error: {error}"
            )
            print(
                "verdicts: "
                f"investigate={verdict['investigated_before_abstaining']['verdict']} "
                f"unknown={verdict['wrote_unknown_in_code']['verdict']} "
                f"check={verdict['called_check_read_abstained_log']['verdict']} "
                f"notes={verdict['notes_carry_region_correlation']['verdict']}"
            )

    result = {
        "recording": str(recording),
        "seed_frame": seed_frame,
        "turns": turns,
        "dry_run": dry_run,
        "seed_state": {
            "corpus_entries": n_corpus,
            "actions": len(sandbox._actions),  # noqa: SLF001
            "simulate_registered": sim_registered,
            "unknown_in_namespace": unknown_in_ns,
        },
        "verdicts": verdicts,
    }
    verdicts_path = out_dir / "verdicts.json"
    with open(verdicts_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[out] verdicts: {verdicts_path}")
    print(f"[out] transcript: {transcript_path}")
    return result


# ── Full-replay runner (L3b) ───────────────────────────────────────────────


def run_full_replay(
    recording: Path,
    *,
    turns: int,
    out_dir: Path,
    dry_run: bool = False,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Cold-start at frame 0 (no simulate), play level 1 offline against the
    real game engine until win / GAME_OVER / the *turns* cap; write
    metrics.csv + verdicts.json + transcript.jsonl under *out_dir*.

    End conditions (mirrors the live agent's outer-loop guards):
    - win: the step callback saw levels_completed bump → LEVEL_TRANSITION
      message next turn, run ends (level 1 is the window);
    - GAME_OVER: the engine's step budget is exhausted (measured: reset
      flash at step 43, GAME_OVER at 129);
    - the --turns cap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sandbox, cursor = seed_full_replay(recording, timeout=timeout)
    history: list[dict[str, Any]] = cursor["history"]

    n_corpus = len(sandbox._grids)  # noqa: SLF001
    sim_registered = sandbox._simulate is not None  # noqa: SLF001
    unknown_in_ns = sandbox.namespace.get("UNKNOWN") == UNKNOWN
    print(f"[seed] corpus entries: {n_corpus} (expected 2 — the virtual RESET pair)")
    print(f"[seed] simulate registered: {sim_registered} (expected False — cold start)")
    print(f"[seed] UNKNOWN in namespace: {unknown_in_ns}")
    if n_corpus != 2:
        raise RuntimeError(
            f"full-replay seed corpus shape wrong: {n_corpus} != 2 "
            "(pair-seeding contract violated)"
        )
    if sim_registered:
        raise RuntimeError("simulate registered at cold seed — seeding bug")
    if not unknown_in_ns:
        raise RuntimeError("UNKNOWN missing from sandbox namespace")

    workflow = WorkflowController(sandbox)
    world_model: dict[str, str] = {"notes": "", "plan": ""}

    client: LLMClient | None = None if dry_run else LLMClient()
    verdicts: list[dict[str, Any]] = []
    transcript_path = out_dir / "transcript.jsonl"
    metrics_path = out_dir / "metrics.csv"
    messages = build_initial_messages(
        sandbox,
        history,
        world_model,
        workflow,
        frame_index=0,
    )

    def _metrics_row(
        turn: int, frame: int, metrics: dict[str, Any], latency_s: float
    ) -> str:
        acc = metrics.get("accuracy")
        cov = metrics.get("coverage")
        return (
            f"{turn},{frame},"
            f"{'' if acc is None else round(acc, 1)},"
            f"{metrics.get('wrong_cells', '')},"
            f"{metrics.get('abstained_changed', '')},"
            f"{metrics.get('abstained_stable', '')},"
            f"{'' if cov is None else round(cov, 1)},"
            f"{round(latency_s, 1)}\n"
        )

    end_reason = "turns-cap"
    with (
        open(transcript_path, "w", encoding="utf-8") as transcript,
        open(metrics_path, "w", encoding="utf-8") as metrics_csv,
    ):
        metrics_csv.write(_METRICS_CSV_HEADER)
        for turn in range(1, turns + 1):
            if cursor["game_over"]:
                end_reason = "game-over"
                break
            if cursor["levels_completed"] > 0:
                end_reason = "level-win"
                break

            # Turn-boundary board-reset consumption (live-agent parity,
            # incident 6685d7d2): the flash hard-aborted the previous
            # batch and left _board_reset_pending alive; without consuming
            # it here, every subsequent action() raises BoardReset and the
            # model is permanently stuck. The structured BOARD_RESET_TEXT
            # rides at the head of the frame prompt, exactly like the live
            # agent's turn boundary.
            board_reset_msg = ""
            if sandbox._board_reset_pending:  # noqa: SLF001
                board_reset_msg = BOARD_RESET_TEXT.format(
                    action_id=sandbox._action_taken  # noqa: SLF001
                )
                sandbox._board_reset_pending = False  # noqa: SLF001
                print(f"[turn {turn}] board reset consumed — budget refreshed")

            label = "(dry-run scripted)" if dry_run else ""
            print(f"\n--- Turn {turn}/{turns} {label} ---")
            refresh_state_messages(messages, sandbox, world_model)
            if board_reset_msg:
                messages.append({"role": "user", "content": board_reset_msg})
            outbound = trim_messages_for_context(
                messages, budget_tokens=_CONTEXT_BUDGET_TOKENS
            )

            if dry_run:
                code = FULL_REPLAY_DRY_RUN_CODE
                response_content = "(dry-run: scripted turn, no LLM call)"
                latency_s = 0.0
            else:
                assert client is not None  # narrowed by the dry_run flag
                t0 = time.time()
                try:
                    response = client.chat(
                        messages=outbound,
                        tools=[
                            AGENT_PYTHON_TOOL_SCHEMA,
                            UPDATE_NOTES_TOOL_SCHEMA,
                            SET_PHASE_TOOL_SCHEMA,
                        ],
                        tool_choice="auto",
                        max_tokens=4096,
                    )
                except Exception as exc:
                    latency_s = time.time() - t0
                    print(f"LLM call failed: {exc}")
                    verdicts.append(
                        {
                            "turn": turn,
                            "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                            "code": "",
                            "output": "",
                            "error": str(exc),
                            "latency_s": round(latency_s, 1),
                            "metrics": None,
                            "verdicts": None,
                        }
                    )
                    metrics_csv.write(
                        _metrics_row(turn, len(sandbox._grids) - 1, {}, latency_s)  # noqa: SLF001
                    )
                    transcript.write(
                        json.dumps({"turn": turn, "error": str(exc)}) + "\n"
                    )
                    continue
                latency_s = time.time() - t0
                response_content = response.content or ""
                code = ""
                for tc in response.tool_calls or []:
                    if tc["function"]["name"] == "python":
                        try:
                            code = json.loads(tc["function"]["arguments"]).get(
                                "code", ""
                            )
                        except Exception:
                            code = ""
                        break
                print(f"LLM latency: {latency_s:.1f}s")
                if not code and not response_content:
                    print("no tool call and no text — recording empty turn")

            grids_before = len(sandbox._grids)  # noqa: SLF001
            output, error, _acted = sandbox.run_code(code)

            pending = sandbox._pending_notes  # noqa: SLF001
            for key in ("notes", "plan"):
                if pending.get(key):
                    world_model[key] = pending[key]
            if not dry_run:
                for tc in response.tool_calls or []:
                    if tc["function"]["name"] == "update_notes":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}
                        if args.get("notes"):
                            world_model["notes"] = str(args["notes"])
                        if args.get("plan"):
                            world_model["plan"] = str(args["plan"])

            tool_text = output if output else "(no output)"
            if error:
                tool_text = (
                    f"{tool_text}\nError: {error}" if tool_text else f"Error: {error}"
                )
            remaining = turns - turn
            tool_text = (
                f"{tool_text}\nTurn {turn}/{turns} completed. "
                f"{remaining} turns remaining."
            )
            if not dry_run:
                python_calls = [
                    tc
                    for tc in response.tool_calls or []
                    if tc["function"]["name"] == "python"
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_content or None,
                        "tool_calls": python_calls or None,
                    }
                )
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]
            for img in sandbox.pending_images:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img['b64']}"},
                    }
                )
                content_parts.append({"type": "text", "text": img.get("caption", "")})
            sandbox.pending_images.clear()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"replay_{turn}",
                    "content": content_parts if len(content_parts) > 1 else tool_text,
                }
            )
            if len(sandbox._grids) != grids_before:  # noqa: SLF001
                append_frame_prompt(
                    messages,
                    sandbox,
                    history,
                    workflow,
                    frame_index=len(sandbox._grids) - 1,  # noqa: SLF001
                )

            verdict = extract_verdicts(
                code, output, world_model.get("notes", ""), response_content
            )
            metrics = sandbox.measure()
            verdicts.append(
                {
                    "turn": turn,
                    "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                    "code": code,
                    "output": output[:4000],
                    "error": error,
                    "latency_s": round(latency_s, 1),
                    "metrics": {
                        "accuracy": metrics.get("accuracy"),
                        "wrong_cells": metrics.get("wrong_cells"),
                        "abstained_changed": metrics.get("abstained_changed"),
                        "abstained_stable": metrics.get("abstained_stable"),
                        "coverage": metrics.get("coverage"),
                        "frames_correct": metrics.get("frames_correct"),
                        "frames_total": metrics.get("frames_total"),
                    },
                    "verdicts": verdict,
                }
            )
            metrics_csv.write(
                _metrics_row(
                    turn,
                    len(sandbox._grids) - 1,  # noqa: SLF001
                    metrics,
                    latency_s,
                )
            )
            transcript.write(
                json.dumps(
                    {
                        "turn": turn,
                        "frame": len(sandbox._grids) - 1,  # noqa: SLF001
                        "code": code,
                        "output": output[:4000],
                        "error": error,
                        "latency_s": round(latency_s, 1),
                        "response_content": response_content[:1000],
                        "messages": outbound,
                    },
                    default=str,
                )
                + "\n"
            )
            print(
                f"code chars: {len(code)} | output chars: {len(output)} | "
                f"error: {error}"
            )
            print(
                "verdicts: "
                f"investigate={verdict['investigated_before_abstaining']['verdict']} "
                f"unknown={verdict['wrote_unknown_in_code']['verdict']} "
                f"check={verdict['called_check_read_abstained_log']['verdict']} "
                f"notes={verdict['notes_carry_region_correlation']['verdict']}"
            )
            if dry_run:
                break

    result = {
        "recording": str(recording),
        "mode": "full-replay",
        "turns": turns,
        "dry_run": dry_run,
        "end_reason": end_reason,
        "levels_completed": cursor["levels_completed"],
        "seed_state": {
            "corpus_entries": n_corpus,
            "simulate_registered": sim_registered,
            "unknown_in_namespace": unknown_in_ns,
        },
        "verdicts": verdicts,
    }
    verdicts_path = out_dir / "verdicts.json"
    with open(verdicts_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[out] end reason: {end_reason} | levels: {cursor['levels_completed']}")
    print(f"[out] verdicts: {verdicts_path}")
    print(f"[out] transcript: {transcript_path}")
    print(f"[out] metrics: {metrics_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "L3/L3b harness for the UNKNOWN-abstention rework. Mode 'probe': "
            "seed a SimulatorSandbox at the f9e3e301 recorded f20 state with "
            "the UNKNOWN-abstention fixture sim, then run N turns of the NEW "
            "prompt against the local LLM. Mode 'full-replay': cold-start at "
            "frame 0 with NO simulate, play the full level-1 window offline "
            "against the real game engine. Records per-turn verdicts "
            "(investigate-before-abstain, UNKNOWN-in-code, check()-readership, "
            "notes quality) for human review."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["probe", "full-replay"],
        default="probe",
        help=(
            "probe: seed at --seed-frame with the fixture sim, N LLM turns "
            "(task 11). full-replay: cold-start at frame 0, no simulate, play "
            "level 1 offline to win/game-over/turns-cap (task 12, L3b)."
        ),
    )
    parser.add_argument(
        "--recording",
        type=Path,
        required=True,
        help="Recording .recording.jsonl file (the f9e3e301 run)",
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=20,
        help="Probe mode: recording transition to seed TO (default: 20 — the "
        "key-box flash fork; corpus becomes 22 pair-seeded entries [0..21])",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Turn cap (default: 10 in probe mode, 100 in full-replay mode)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for verdicts.json + transcript.jsonl (+ "
        "metrics.csv in full-replay mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No LLM calls: full seeding + one scripted turn, proving the "
        "plumbing end-to-end",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Sandbox execution timeout in seconds (default: 30)",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Recording not found: {args.recording}")
        sys.exit(1)

    turns = (
        args.turns
        if args.turns is not None
        else (100 if args.mode == "full-replay" else 10)
    )

    if args.mode == "full-replay":
        run_full_replay(
            args.recording,
            turns=turns,
            out_dir=args.out,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    else:
        run_probe(
            args.recording,
            seed_frame=args.seed_frame,
            turns=turns,
            out_dir=args.out,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
