"""Reconstruct the exact agent/sandbox state at a recording frame.

Faithful state reconstruction for replay experiments: given a recording and
its sibling ``.llm.jsonl`` log, rebuild the world (via ``ReplayHarness``),
the sandbox (registered simulate, ignore mask, grid history), the workflow
phase, and the LLM conversation exactly as they stood at the start of the
spiral turn (frame 11 of incident ``1786060d``).

The design principle: **derive everything from recorded artifacts, verify
against recorded state**. Where the recording stores state verbatim
(``simulator_state``, messages), we load it; where it doesn't (sandbox
namespace, ignore mask), we replay the exact ``python()`` calls from the
LLM log and let the sandbox machinery produce the state itself.

Index convention (single owner: ``agents/simulator_agent/reset_policy.py``
— cite it, do not restate the mapping):

- The corpus IS the full harness frames, including the synthetic
  ``env.reset()`` frame at index 0: ``sandbox grid j == harness.frames[j]``
  element-wise, and ``actions == [int(ai["id"]) for ai in
  harness.action_inputs]`` (``"RESET"`` normalized to 0 via
  ``reset_policy.is_reset``). This is byte-for-byte the offline loading
  path (``sandbox.py`` offline branch) and the NEW live shape (the virtual
  RESET pair ``([B0, B0], [0])`` emerges naturally: ``frames[0] ==
  frames[1]`` board-wise and ``actions[0] == 0`` for RESET-first
  recordings).
- ``history[i] ↔ transition i ↔ actions[i]``: transition i maps
  ``grids[i] --actions[i]--> grids[i+1]``; the action that PRODUCED frame
  ``i`` is ``actions[i-1]`` (frame 0 is produced by RESET itself).
- History entries follow the task-6 live format: entry 0 is the env RESET
  (action 0, ``provenance`` marker, env-initiated framing); entry ``i``
  is ``{"action": actions[i], "frame_index": i, "frame": grids[i+1]}``.

Incident anchors (recording 1786060d-b75e-48f7-b0fb-ac224be8c18a):
  - seq 29 (frame 7): ``my_sim`` written and registered via set_simulate()
  - seq 30 (frame 7): ``set_ignore`` for yellow HUD cells (rows 61-62)
  - seq 34 (frame 7): batch ``for _ in range(4): action(1)`` — produced
    frames 8-11; the 4th up-move walked the player into the goal box and
    fired the 123-cell exception flow
  - seq 35 (frame 11): spiral-turn opening conversation (101 messages)

Not reconstructed (deliberately):
  - ``_objects``/``_adjacency`` segmentation caches (agent-side, cosmetic;
    the sandbox does not consume them)
  - trimmed-message internals beyond what the log preserved
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.simulator_agent.agent import ITER0_RESET_PROVENANCE
from agents.simulator_agent.reset_policy import RESET_ACTION, is_reset
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController
from replay.harness import ReplayHarness

# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IncidentMarker:
    """Positions of the replayed incident inside the recording + LLM log.

    ``spiral_frame`` is the recording line index of the spiral turn's
    observation grid (the turn whose conversation is ``spiral_seq``).
    ``simulate_seq``/``ignore_seq`` are the LLM-log seqs of the frame-7
    ``set_simulate``/``set_ignore`` python calls replayed during seeding.
    """

    spiral_frame: int = 11
    simulate_seq: int = 29
    ignore_seq: int = 30
    spiral_seq: int = 35


@dataclass
class ReconstructedState:
    """Everything needed to resume the agent mid-run at the spiral turn."""

    harness: ReplayHarness
    sandbox: SimulatorSandbox
    workflow: WorkflowController
    messages: list[dict[str, Any]]  # conversation prefix (spiral seq, verbatim)
    frame_index: int
    llm_calls: int  # LLM calls consumed before this turn
    simulate_source: str  # the simulate code registered during seeding
    set_ignore_source: str  # the set_ignore code run during seeding
    notes: dict[str, str]  # pending world-model notes at turn start
    history_turns: list[dict[str, Any]]  # agent-owned history entries
    diff_images: list[dict[str, str]]  # batch predict-and-compare diffs
    marker: IncidentMarker
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


def _count_llm_calls(rows: list[dict[str, Any]], upto_seq: int) -> int:
    """Count LLM calls with seq <= upto_seq (inclusive)."""
    return sum(1 for r in rows if r.get("seq", 0) <= upto_seq)


# ── Environment callback ──────────────────────────────────────────────────


def _make_step_env_callback(
    harness: ReplayHarness,
    sandbox: SimulatorSandbox,
    history_turns: list[dict[str, Any]],
    *,
    prev_grid: list[list[int]] | None = None,
) -> Callable[[int, dict[str, Any] | None], dict[str, Any]]:
    """Build a step_env_callback that steps the replayed offline env.

    Mirrors ``agent._step_env_callback``'s state_response shape (grid,
    valid_actions, last_action_result, history) so ``sandbox.action()``
    behaves like the live agent's. ``history_turns`` is owned by the caller
    (the agent owns it in production, not the sandbox). ``prev_grid`` is the
    grid the env was at when seeding finished — the first ``step()`` call
    transitions from it. The raw env step result (not ``harness.frames``,
    which stays frozen after replay) provides the post-action state.
    """

    state: dict[str, Any] = {"prev": prev_grid}

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        from arcengine import GameAction

        game_action = GameAction.from_id(action_id)
        frame = harness.env.step(game_action)
        if frame is None:
            raise RuntimeError(f"env.step returned None for action {action_id}")

        prev = state["prev"]
        curr_grid: list[list[int]] = [list(row) for row in frame.frame[0]]
        prev_levels = harness.frames[-1].levels_completed or 0
        curr_levels = frame.levels_completed or 0
        if prev is not None:
            last = {
                "board_changed": prev != curr_grid,
                "done": frame.state.value == "GAME_OVER"
                or frame.state.value == "WIN",
                "level_completed": curr_levels > prev_levels,
                "game_over": frame.state.value == "GAME_OVER",
                "run_complete": frame.state.value == "WIN",
                "reward": curr_levels - prev_levels,
                "valid_actions": list(frame.available_actions)
                if frame.available_actions
                else [],
            }
        else:
            last = {}
        state["prev"] = curr_grid

        return {
            "objects": (),
            "adjacency": frozenset(),
            "grid": curr_grid,
            "valid_actions": last.get("valid_actions", [1, 2, 3, 4]),
            "last_action_result": last,
            "history": history_turns,
        }

    return step


# ── Sandbox seeding ───────────────────────────────────────────────────────


def _seed_sandbox(
    sandbox: SimulatorSandbox,
    grids: list[list[list[int]]],
    actions: list[int],
    history_turns: list[dict[str, Any]],
) -> None:
    """Seed the sandbox with the replayed corpus so check()/predict work.

    Unified convention (single owner ``reset_policy``): the corpus IS the
    full harness frames — ``grids[j] == harness.frames[j]`` element-wise,
    ``actions == [int(ai["id"]) for ai in harness.action_inputs]``. For
    RESET-first recordings the pair shape emerges naturally:
    ``grids[0] == grids[1]`` and ``actions[0] == 0`` — the [0]/[1] board
    duplicate IS the RESET transition ``(B0, RESET) -> B0``. This is the
    same shape the NEW live agent produces via
    ``update_state(reset_seeded=True)`` (task 5's ``virtual_reset_pair``)
    plus its ``action()`` appends, and byte-for-byte what the offline
    loading path (``sandbox.py`` offline branch) builds.

    Sandbox semantics: ``_grids[i]`` is the grid before ``_actions[i]``;
    ``check()`` pairs grids[i]/actions[i] with grids[i+1]. Live-mode
    ``action()`` appends the next transition as usual because ``_grids``
    is non-empty. ``namespace["history"]`` is pushed here the same way the
    live agent's ``update_state`` push does every turn.
    """
    sandbox._grids = [  # noqa: SLF001
        [[cell for cell in row] for row in g] for g in grids
    ]
    sandbox._actions = list(actions)  # noqa: SLF001
    sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
    sandbox.namespace["history"] = history_turns  # noqa: SLF001
    sandbox._current_frame = [[cell for cell in row] for row in grids[-1]] if grids else None  # noqa: SLF001
    sandbox.namespace["current_frame"] = sandbox._current_frame  # noqa: SLF001
    sandbox.namespace["previous_frame"] = (  # noqa: SLF001
        [[cell for cell in row] for row in grids[-2]] if len(grids) >= 2 else None
    )
    # last_action_result for the final recorded action (frame 11 = action 1)
    sandbox._last_action_result = {  # noqa: SLF001
        "board_changed": True,
        "done": False,
        "level_completed": False,
        "game_over": False,
        "run_complete": False,
        "reward": 0,
        "valid_actions": [1, 2, 3, 4],
    }
    sandbox.namespace["last_action_result"] = sandbox._last_action_result  # noqa: SLF001
    sandbox.namespace["valid_actions"] = [1, 2, 3, 4]  # noqa: SLF001


# ── Main entry ────────────────────────────────────────────────────────────


def reconstruct(
    recording_path: Path,
    *,
    marker: IncidentMarker | None = None,
    seed: int = 0,
) -> ReconstructedState:
    """Rebuild the exact state at the spiral turn from recording + LLM log.

    Steps:
      1. Replay the recording through ``ReplayHarness`` (deterministic env).
      2. Build a live-mode sandbox wired to the replayed env.
      3. Seed grids/actions from the replay (frames 0..frame inclusive).
      4. Re-run the recorded set_simulate() and set_ignore() python calls.
      5. Load the spiral conversation prefix verbatim.
      6. Verify: check() reproduces the recorded 100% accuracy; the
         pending exception flow contains the recorded 123-cell diff.

    Returns a ``ReconstructedState`` with a ``verify`` report.
    """
    m = marker or IncidentMarker()
    rows = load_llm_rows(Path(str(recording_path).replace(".recording.jsonl", ".llm.jsonl")))

    # Harness frames index the reset frame at 0: frames[i] is the result of
    # action_inputs[i-1]. The spiral-turn observation grid is recording line
    # marker.spiral_frame, i.e. harness.frames[marker.spiral_frame + 1].
    grid_index = m.spiral_frame + 1

    harness = ReplayHarness.from_recording(recording_path, seed=seed)
    harness.replay_to(grid_index)
    if len(harness.frames) < grid_index + 1:
        raise RuntimeError(
            f"replay produced {len(harness.frames)} frames, need {grid_index + 1}"
        )

    # Unified convention (single owner reset_policy — see module docstring):
    # the corpus IS the full harness frames, synthetic reset at [0] included.
    # sandbox grid j == harness.frames[j] element-wise; actions carry every
    # recorded action with "RESET" normalized to 0 (is_reset), so the
    # virtual RESET pair ([B0, B0], [0]) occupies slots 0/1 for RESET-first
    # recordings — the same shape the NEW live agent's pair-seeded corpus
    # has, and byte-for-byte the offline loading path's output.
    grids = [
        [[cell for cell in row] for row in fd.frame[0]] for fd in harness.frames
    ]
    sandbox_actions = [
        RESET_ACTION if is_reset(ai["id"]) else int(ai["id"])
        for ai in harness.action_inputs[:grid_index]
    ]

    # Agent-owned history entries, in the order the live agent built them
    # (task-6 format): entry 0 is the env RESET (provenance marker, frame 0
    # produced by RESET itself); entry j is the action that produced frame j
    # (actions[j]) with the grid it produced (grids[j+1] == frames[j+1] ==
    # recording line j's frame).
    history_turns: list[dict[str, Any]] = [
        {
            "action": sandbox_actions[j],
            "frame_index": j,
            "frame": [[cell for cell in row] for row in grids[j + 1]],
            **(
                {"provenance": ITER0_RESET_PROVENANCE}
                if j == 0
                else {}
            ),
        }
        for j in range(len(sandbox_actions))
    ]

    # Live-mode sandbox whose callback closes over the sandbox itself:
    # circular at construction, resolved via a deferred holder.
    holder: dict[str, Any] = {}
    sandbox = SimulatorSandbox(
        step_env_callback=lambda a, d: holder["step"](a, d),
    )
    holder["step"] = _make_step_env_callback(
        harness, sandbox, history_turns, prev_grid=grids[-1]
    )

    # Rebuild turn history in the order the live agent experienced it:
    # (a) seed the sandbox corpus through the frame-7 observation — the
    #     pair-seeded shape the NEW live agent had at that point,
    # (b) run the set_simulate python call → cached check() = 100% (the
    #     degenerate RESET transition adds one scored frame),
    # (c) run the set_ignore python call,
    # (d) append the batch transitions (the up-moves after line FRAME7_LINE) —
    #     predict_and_compare fires per transition; the LAST one leaves the
    #     pending 123-cell exception flow the spiral turn opened with.
    FRAME7_LINE = 7  # recording line of the frame-7 observation grid
    seed_grids = grids[: FRAME7_LINE + 2]
    seed_actions = sandbox_actions[: FRAME7_LINE + 1]
    _seed_sandbox(sandbox, seed_grids, seed_actions, history_turns)
    by_seq = {r["seq"]: r for r in rows}

    simulate_code = _tool_code(by_seq[m.simulate_seq], "python")
    output, error, _ = sandbox.run_code(simulate_code)
    if error or "[set_simulate] registered" not in output:
        raise RuntimeError(f"set_simulate replay failed: output={output!r} error={error!r}")
    if sandbox._simulate is None:  # noqa: SLF001
        raise RuntimeError("simulate not registered after replay")

    ignore_code = _tool_code(by_seq[m.ignore_seq], "python")
    output, error, _ = sandbox.run_code(ignore_code)
    if error:
        raise RuntimeError(f"set_ignore replay failed: {error!r}")
    if not sandbox._ignore_mask:  # noqa: SLF001
        raise RuntimeError("ignore mask empty after replay")

    # (d) Append the batch transitions (recording lines FRAME7_LINE+1..spiral)
    # the same way the live agent's action() would have; collect the
    # predict-and-compare diff images the loop would attach to tool results.
    batch_grids = grids[FRAME7_LINE + 2 :]
    prev_grid = grids[FRAME7_LINE + 1]
    sandbox._pending_exception_flow = None  # noqa: SLF001
    for i, g_next in enumerate(batch_grids):
        sandbox._grids.append([[cell for cell in row] for row in g_next])  # noqa: SLF001
        sandbox._actions.append(1)  # noqa: SLF001
        sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
        sandbox._current_frame = [
            [cell for cell in row] for row in g_next
        ]  # noqa: SLF001
        sandbox.namespace["current_frame"] = sandbox._current_frame  # noqa: SLF001
        sandbox._previous_grid = (  # noqa: SLF001
            [[cell for cell in row] for row in batch_grids[i - 1]] if i > 0 else None
        )
        sandbox.predict_and_compare(prev_grid, g_next, 1)
        prev_grid = g_next
    sandbox.namespace["n_frames"] = len(sandbox._grids)  # noqa: SLF001
    diff_images = list(sandbox.pending_images)
    sandbox.pending_images = []

    # 4. Notes at turn start: last pending notes recorded before seq 35.
    # The agent applies update_notes from the previous turn's last run_code
    # into the prompt; in the sandbox they land in _pending_notes.
    notes = dict(sandbox._pending_notes)  # noqa: SLF001

    # Conversation prefix: spiral seq's messages verbatim (images stripped by
    # the logger as '[image omitted]' — re-rendering is out of scope here).
    messages: list[dict[str, Any]] = by_seq[m.spiral_seq]["messages"]

    # Workflow state: MODEL phase, no prior exception flows. The counter
    # mirrors the live agent's action_counter at the spiral turn: one
    # executed action per corpus action (RESET included — step_env counts it).
    workflow = WorkflowController(sandbox)
    workflow._action_counter = len(sandbox_actions)  # noqa: SLF001
    workflow._phase = Phase.MODEL  # noqa: SLF001

    return ReconstructedState(
        harness=harness,
        sandbox=sandbox,
        workflow=workflow,
        messages=messages,
        frame_index=m.spiral_frame,
        llm_calls=_count_llm_calls(rows, m.spiral_seq - 1),
        simulate_source=simulate_code,
        set_ignore_source=ignore_code,
        notes=notes,
        history_turns=history_turns,
        diff_images=diff_images,
        marker=m,
        verify={},
    )


def verify_reconstruction(state: ReconstructedState) -> dict[str, Any]:
    """Run all fidelity checks against recorded ground truth.

    Checks (incident ground truth in parentheses):
      1. check() accuracy == 100.0 over the pre-divergence transitions
         (recorded: 100.0%, 0 wrong, 7/7 frames; under the unified
         convention the corpus carries the degenerate RESET transition in
         addition, so the recomputed cache reports 8/8 — same accuracy,
         +1 corpus offset from the virtual RESET pair).
      2. predict_and_compare on action 1 from frame 11 reproduces the
         123-cell / 2-region exception flow (rows 53-62 cols 1-10: 76 cells
         5->0; rows 10-19 cols 34-38: 47 cells 3->9, 9->5).
      3. Sandbox has a registered simulate and non-empty ignore mask.
      4. Conversation prefix has 101 messages ending with the exception
         flow user message.
    """
    sandbox = state.sandbox
    checks: dict[str, Any] = {}

    # 1. Stale-check fidelity: the recorded check result shown to the LLM
    # (msg[1] "Last check(): 100.0% ... 7/7 frames correct") is the cached
    # _last_check_result from frame 7 — NOT a live re-check over the batch
    # transitions (which would show HUD drift). Read the cached value BEFORE
    # the live check() call: check() overwrites _last_check_result.
    checks["stale_check_accuracy"] = (  # noqa: SLF001
        (sandbox._last_check_result or {}).get("overall_accuracy")
    )
    live = sandbox.namespace["check"]()
    checks["live_check_wrong"] = live.get("total_wrong")
    checks["check_frames"] = (
        live.get("frames_correct"),
        live.get("frames_total"),
    )

    # 3. registered simulate + ignore mask
    checks["has_simulate"] = sandbox._simulate is not None  # noqa: SLF001
    checks["ignore_cells"] = len(sandbox._ignore_mask)  # noqa: SLF001

    # 2. exception flow reproduction: predict_and_compare on the divergence
    prev = [list(r) for r in state.harness.frames[-2].frame[0]]
    curr = [list(r) for r in state.harness.frames[-1].frame[0]]
    sandbox._pending_exception_flow = None  # noqa: SLF001
    sandbox.predict_and_compare(prev, curr, 1)
    pending = sandbox._pending_exception_flow  # noqa: SLF001
    if pending:
        checks["exception_n_diff"] = pending.get("n_diff")
        checks["exception_n_regions"] = pending.get("n_regions")
        checks["exception_regions"] = [
            (tuple(r["bbox"]), r["n_cells"]) for r in pending.get("regions", [])
        ]
    else:
        checks["exception_n_diff"] = None

    # 4. conversation shape
    checks["n_messages"] = len(state.messages)
    checks["n_diff_images"] = len(state.diff_images)
    last_user = [
        m for m in state.messages if m.get("role") == "user"
    ]
    checks["last_user_has_exception"] = any(
        isinstance(m.get("content"), str)
        and "EXCEPTION FLOW" in m["content"]
        for m in last_user[-2:]
    )

    return checks