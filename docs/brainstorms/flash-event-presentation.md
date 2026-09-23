# Flash-event presentation — stable history + flash as first-class event

> **Status**: Refactor COMPLETE (2026-09-23); experiment NOT yet run.
> Companion to
> [`cosmetic-verification-abstention.md`](cosmetic-verification-abstention.md)
> (approach A validated → insufficient alone; this is the follow-up
> experiment). Built in the same session as the A-validation; run it in a
> NEW session per the plan below.

---

## The hypothesis under test

The abb14eca incident's central signal — the top-box flash (rows 9-15 +
HUD, white overlay) on blocked goal-tile entry — was **ingested as board
state** because `settled_board`'s class-position rule (HIGHLIGHT → layer
0) picks the *transient flashed* layer for this stack family. Verified
against the recording (continuity evidence): the game continues from the
LAST layer on all 11 highlight events (line 17: next-frame diff 52 vs
112), while d91cdde0's highlights continue from layer 0 — the class rule
is wrong in general; **temporal continuity** (argmin diff vs the previous
extraction) is the rule that holds on both recordings.

Two distinct presentation hypotheses, one experiment:

- **H1 (de-noising)**: the flash residue corrupting the corpus (permanent
  70/60-cell wrong frames in every check() — the 88% plateau) is the
  blocker; with a stable corpus the model's own instruments (check,
  exception flow) surface the blocked move cleanly and the investigation
  proceeds.
- **H2 (event data)**: even with a clean corpus, a *vanished* flash is
  worse than a *reported* one; the model needs the flash as explicit,
  correlated EVENT data (FLASH line in the action() result + flash log)
  to treat it as state feedback rather than ignore it.

## What was built (this session)

### `agents/simulator_agent/board_extraction.py` (NEW)

- `BoardPolicy = Literal["class", "continuity"]`.
- `extract_board(stack, prev_board, policy) -> (board, FlashEvent | None)`:
  class → `settled_board` byte-identical, never emits events; continuity →
  argmin layer (tie-break toward the class pick), FLASH_RESET/LEVEL_WIN
  keep the class path (board restoration has its own channel), no-anchor
  → class fallback. Event = overlay diff (cells, regions, transitions)
  when the chosen layer ≠ layer 0.
- `flash_event_text(event)`: the one-line FLASH annotation (echoes the
  exception-flow region format: "FLASH: 60 cells (3->0 (52)) at rows
  9-15 ... — transient animation overlay, NOT board state ... probe what
  triggers them").
- `FlashEvent.as_dict()/from_dict()` round-trip for sandbox responses.
- Tests: `tests/unit/simulator_agent/test_board_extraction.py` — 13
  tests: the full abb14eca walk (all 11 highlight events fire with
  correct overlays: top-box rows 9-15 white on 17/66, timer rows 53-62
  on 36-42/56/57/75/76; FLASH_RESET lines 49/97 keep the class path),
  synthetic d91cdde0-style highlight (continuity stays byte-identical to
  class there), malformed-input totality, purity.

### Wiring (behavior-preserving under the default)

- `agent.py`: `board_policy` ctor kwarg (default "class");
  **extraction single-owner** — `step_env` extracts once per frame into
  `_last_extraction` (+ `_last_extraction_frame` identity tag,
  `_previous_extraction_board` anchor); consumers (`_update_segmentation`,
  `_append_history`, run() prompt build, `_step_env_callback`) read the
  cache via `_cached_board_for(frame)` (identity-checked; direct-call
  fallbacks keep `__new__`-built test agents byte-identical — house
  convention). Sandbox response carries `flash_event` when continuity
  fired.
- `sandbox.py`: `_flash_events[transition_index]` store (parallel to
  `_engine_event_transitions`), FLASH line printed into the action()
  tool result; `reset_for_level_transition` clears it.
- `replay_timeline.py`: `reconstruct(..., board_policy="class")` — the
  **era flag**: default replays old recordings byte-faithfully (f24/f57
  regression tests stay green — verified); "continuity" diverges exactly
  at flash transitions (probe: f57 walk under continuity scores
  transition 17 correct — 19/19 vs embedded 18/25@87.8% — the flashed
  board no longer enters the corpus; embedded-check mismatches there
  are the expected, honest divergence).
- Full suite green (1619 passed), ruff + mypy --strict clean on changed
  files.

### Verified probe results (hold these as baselines)

- f24 (24,106) and f57 (57,164) seeds under **class**: corpus/sim/
  stale-check/flow ALL byte-green.
- f57 under **continuity**: `_flash_events` = transitions {17, 36, 37,
  40, 41, 42, 56, 57} by the f57 marker (the corpus at that point);
  grids[18] == line-17 STABLE layer; flashed board absent from corpus.
- The f24-claim/f57-route scenarios in
  `scripts/experiment_cosmetic_rule.py` remain the A-experiment anchors
  (era-flagged class replays).

## The experiment (NOT run — this is the next session's work)

### Seed: `(13, 51)` — before the first flash

The A-experiment seeds are CONTAMINATED for this test (their 122-message
prefixes carry old-world evidence: the flashed board in the corpus, the
88% check cache, the "decorative" notes). Seed instead at the turn
opening of frame 13 (first_seq=51): `my_sim` registered at 99.6% (seq-45
check), phase MODEL, no flash fired yet (first flash is line 17; lines
0-16 are all SINGLE — **both policies reconstruct byte-identically** to
this marker, provable before any LLM call). The divergence happens live
inside the window: the model walks left/down toward the marker, and the
recorded batch (seq 51: LLL batch; seq 52: DOWN to (30,19)) is exactly
the approach path — under continuity the flash fires when reality
deviates from that script, in-environment.

**Counterfactual honesty**: what the OLD run did after the flash (the
seq-110 cosmetic claim) is not reproducible under continuity — that's
the point. The arms are compared to each other, not to the recording.

### Arms (3)

1. **`control`** — class policy. Flash enters corpus as board state;
   image flickers white; check inherits the residue. Expected to
   reproduce the incident dynamics (the 99.6% → 88% collapse after the
   first flash ingestion).
2. **`continuity`** — continuity policy, NO event surfacing (the
   flash_event is captured but its text is NOT printed — requires one
   harness switch, see "open items"). Isolates H1: is the stable corpus
   alone enough?
3. **`flash-aware`** — continuity + FLASH line in action() results +
   flash log. Isolates H2: does explicit event data change behavior
   beyond de-noising?

### Harness: new script (extract the shared core first)

`experiment_cosmetic_rule.py` carries ~350 lines of generic machinery
(Trace, capped-LLM seam, `_build_agent`, `_wire_action_bridge`,
verdicts). House precedent: `scripts/experiment_simulator.py` is a thin
CLI over `agents/simulator_agent/experiment.py`. Do the same:

1. Extract `agents/simulator_agent/experiment_harness.py` (Trace, capped
   LLM, agent build incl. `board_policy` passthrough, action bridge,
   transcript/verdicts). The cosmetic script slims to scenarios +
   variant patching; verify by re-running its fake-LLM smoke (no real
   LLM needed — outputs must be byte-comparable in structure).
2. New `scripts/experiment_flash_presentation.py`: seed (13, 51),
   scenario forensics (counters from the log timeline before seq 51:
   check_failures, exception_flow_count — derive as in the A-experiment,
   from the recorded logs), the three arms via `board_policy` +
   flash-surfacing toggles.
3. **Fake-LLM mechanical gate FIRST** (no network): scripted LLM drives
   the blocking action (replay the seq-51 LLL batch + seq-52 DOWN, then
   force entry toward the box); assert per arm:
   - control: corpus CONTAINS the flashed board at the flash transition;
     check() residue appears (wrong cells at rows 9-15/53-62).
   - continuity: corpus clean; NO residue; exception flow fires on the
     blocked move with player-cells-only diff; `_flash_events` populated
     but no FLASH line in tool results.
   - flash-aware: corpus clean + FLASH line present in the action()
     tool result; flash log renders (transitions, actions, regions).
   These are the pre-registered mechanical predictions — falsifiable
   before any model call.
4. Real runs: 3 arms × 16-call cap, ONE PROCESS AT A TIME (local LLM
   constraint, 60-130s/call → ~30-45 min/arm). Optionally one repeat of
   the decisive pair for variance; single runs are demos with mechanical
   anchors, not statistics — say so in the writeup.

### Pre-registered behavioral predictions

| Arm | corpus after flash | check() | exception flow on blocked move | predicted model behavior |
|---|---|---|---|---|
| control | flashed board ingested | 99.6% → ~88% residue | overlay cells listed (60-112) | dismisses/investigates HUD flash as unexplained (incident replay) |
| continuity | clean | stays ~100% | clean BLOCKED signal (player cells only) | notices blocked move; may probe; NO flash data to correlate |
| flash-aware | clean | stays ~100% | clean BLOCKED + FLASH event line | references flash in text/notes; probes entry from offsets; possibly infers hidden state |

Success criteria (from the A-postmortem): flash-aware cites the flash
log / probes the top-box entry from offsets within the window;
control reproduces the dismissal. If NEITHER continuity nor flash-aware
investigates even with the event handed over pre-correlated → the
"make evidence available" hypothesis is falsified at this model scale →
escalate to mechanical enforcement (approaches B/C in the cosmetic
brainstorm).

### Verdicts to record (extend the harness verdict extractor)

- existing: cosmetic_claims, unknown_writes, check_calls,
  abstained_log_read, hud_mentions
- new: `flash_references` (text/notes mentioning the FLASH line),
  `blocked_entry_probes` (actions attempting entry from ≥2 offsets),
  `residue_trajectory` (wrong-cell counts per check in each arm)

## Open items (decide before running)

1. **Continuity arm surfacing switch**: the sandbox prints the FLASH
   line unconditionally when `flash_event` is in the response. For the
   `continuity` arm (H1 isolation) it must be suppressible — add a
   sandbox ctor flag (e.g. `surface_flash_events: bool = True`) OR have
   the agent omit `flash_event` from the response under that arm. Prefer
   the ctor flag (agent-side omission changes the response contract).
2. **Flash log**: `_flash_events` is stored but has no LLM-visible
   aggregate surface yet (the per-action FLASH line only). For
   flash-aware, add the correlation footer to check() output or the
   sim-state block: "Flash log: rows 9-15 fired on transitions {17,66} —
   actions {2:2}..." (same shape as the abstained log). Small change in
   sandbox check()/`_build_sim_state_block`.
3. **Timer flash noise**: 9 of the 11 abb14eca flashes are the
   bottom-HUD timer (rows 53-62, every ~5 actions). Decide: does the
   FLASH line fire for every overlay (and risk nudge-fatigue), or does
   flash-aware suppress repeats of an already-logged region (region-key
   dedup, first occurrence + count)? Recommend: log all events in
   `_flash_events`, print the FLASH line only for NEW regions (the
   correlation footer carries the aggregate).
4. **Board-reset double-fire**: FLASH_RESET stacks keep the class path
   and never emit flash events (verified) — no interaction with
   `is_board_reset`. But the walker's continuity anchor now updates
   through reset lines; confirm the f57-era probe covers this (it does —
   the walk passed with the reset at line 49).

## Cost discipline

Local LLM (LM Studio), one request at a time, 60-130s/call (AGENTS.md
constraint). 3 arms + optional repeat pair ≈ 3-5 h wall time. Run arms
in separate processes, sequentially. The fake-LLM gate costs minutes
and must pass before any real run.