# Flash-event presentation — stable history + flash as first-class event

> **Status**: COMPLETE (2026-09-24). 3-arm experiment DONE (see
> "Results"); **live flash-aware run DONE** — config seam validated
> live, the frame-43 flash was cited + investigated (both citation
> criteria met, stronger than the experiment's flash-aware arm), and
> incident 6b32987c (composite-flash dedup suppression) found + fixed.
> See "Live flash-aware run".
> Companion to
> [`cosmetic-verification-abstention.md`](cosmetic-verification-abstention.md)
> (approach A validated → insufficient alone; this is the follow-up
> experiment).

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

## Open items — RESOLVED (2026-09-24 build session)

1. **Continuity arm surfacing switch** ✅ — `surface_flash_events: bool =
   True` ctor flag on `SimulatorSandbox`, `board_policy`-style kwarg on the
   agent. When False: `_flash_events` is still STORED (harness verdicts read
   it — the continuity arm's flashes must be confirmable) but every
   LLM-visible surface is off: per-action FLASH line, check() flash log,
   sim-state block footer (the gate asserts all three).
2. **Flash log aggregate** ✅ — `flash_log_text()` in `board_extraction.py`
   (per region: where, fire count, transitions, actions histogram — same
   shape as the abstained log). Surfaces in check() output (inside the
   truncation-protected zone, per-frame detail still prints last) and in
   the `[Simulator state]` block via `_flash_log()`. check() also caches
   it on `_last_check_result["flash_log"]`.
3. **Timer-flash dedup** ✅ — identity is region OVERLAP
   (`FlashEvent.overlaps`, timer digits shift ±1 cell): the per-action
   FLASH line fires only for NEW regions (first occurrence), repeats stay
   in the store + aggregate. One subtlety the tests caught: the seen-check
   must run BEFORE the store write or the event matches itself and every
   print is suppressed.
4. **Board-reset double-fire** ✅ — unchanged from the original analysis:
   FLASH_RESET/LEVEL_WIN keep the class path (no flash events), and the
   f57-era probe covers the reset crossing at line 49.

## Build-session findings (2026-09-24) — read before the real runs

- **Empirical correction to the seed forensics**: at the (13,51) seed the
  LIVE phase is MODEL (auto-advance at the frame-13 boundary, 16:21:57
  log), NOT the walk-derived EXECUTE — the experiment script seeds the
  live counters explicitly (phase MODEL, check_failures=0, flow_count=1,
  fired_for=1).
- **Empirical correction to the line-17 semantics**: the DOWN at seq 52 is
  **goal entry, not a blocked move** — the stack top reaches (30,19)
  (the white marker's cover position) and the 60-cell flash (top-box rows
  9-15 white overlay + HUD digit block rows 53-62) is the game's feedback
  on reaching the goal. Under continuity the sim residual at that
  transition is 10 cells (HUD timer digits only); under class it is 70
  (24 top-box + 46 HUD ingested as board state). The pre-registered
  "blocked move / player-cells-only" prediction in the table below is
  superseded by these calibrated numbers (the gate asserts them).
- **Production bug found by the gate** (fixed): the walker's live branch
  passed the raw env step's `frame.frame` (numpy-backed layers,
  `np.int8` cells) into `extract_board` — `_is_board` rejects numpy
  cells, silently degrading every post-seed extraction to the class rule
  (no flash events, flashed board ingested). The fix converts to plain
  ints before extraction; `list(row)` is NOT enough (keeps `np.int8`) —
  `int(cell)` per cell is load-bearing.
- **flow_match note**: at this seed `verify_reconstruction`'s
  flow_match/phase_crosscheck are expected-honest-divergent (the final
  transition 13 was a clean UP; the prefix's flow message is frame-12's
  blocked LEFT which the walker replays sim-perfect with the final sim).
  The gate relies on corpus_shape/stale_check/has_simulate only.

## Harness (BUILT)

`agents/simulator_agent/experiment_harness.py` — the shared machinery
(Trace, capped + scripted LLM seams, `build_agent` incl. `board_policy`/
`surface_flash_events` passthrough, action bridge, image substitution,
transcript render, check-trajectory extractor). The cosmetic script is
now a thin CLI over it (fake-LLM smoke green: rule splice verified,
action bridge, abstention surface).

`scripts/experiment_flash_presentation.py` — 3 arms (control /
continuity / flash-aware), seed (13,51), `--gate` (fake-LLM mechanical
gate — **PASSING on all three arms**, no network, ~20 s):

- control: corpus[18] IS the flashed layer-0 board; 70-cell residual
  incl. top-box; no flash surfaces.
- continuity: corpus[18] is the stable layer; 60-cell flash event stored
  at transition 17 (24 top-box cells verified); 10-cell HUD-only
  residual; zero flash text anywhere.
- flash-aware: clean corpus + FLASH line in the DOWN's tool result +
  flash log in check() output.

Real runs (DONE 2026-09-24, artifacts in `out/flash_{control,continuity,aware}.{json,txt}`):

    uv run python scripts/experiment_flash_presentation.py --arm control --out out/flash_control.json --log out/flash_control.txt
    uv run python scripts/experiment_flash_presentation.py --arm continuity --out out/flash_continuity.json --log out/flash_continuity.txt
    uv run python scripts/experiment_flash_presentation.py --arm flash-aware --out out/flash_aware.json --log out/flash_aware.txt

## Results (2026-09-24 real runs)

All three arms ran to the 16-call cap on `qwen/qwen3.8-27b` (the
`.env` LLM_MODEL; gemma was not loaded), seed (13,51), one process at a
time. Same first two moves in every arm (recorded approach: L then
DOWN onto the goal tile → flash at transition 17), then free behavior.

| Metric | control | continuity | flash-aware |
|---|---|---|---|
| wall / calls | 20.4 min / 16 | 17.6 min / 16 | 19.0 min / 16 |
| flash stored @17 | no (class) | yes, 60 cells | yes, 60 cells |
| residual at DOWN | 70 cells | 10 cells | 10 cells |
| attribution | "sim's collision model is slightly off (it blocked a move that reality allowed)" | "only in the decorative corner and the timer bar — NOT the goal region. The move was NOT blocked" | same 10-cell read + FLASH line in the DOWN tool result |
| investigated rows 9-15 cols 33-39 | via full-board prints only (goal hunt) | **never once mentioned in 16 calls** | **call 7 goal hypothesis** ("delivery destination"), calls 9-16 drove the stack toward it (7 action-bearing calls, most of any arm) |
| check() called | no | no | no |
| end | spiral guard → EXPLORE | 3-flow guard → EXPLORE | MODEL, acting |

Verdicts vs the pre-registered criteria:

1. **Control reproduces the dismissal — CONFIRMED.** 70-cell diff →
   sim-bug misattribution; spiral guard at 12 non-action calls.
2. **H1 (de-noising) — CONFIRMED.** Continuity's clean corpus killed
   the blocked-move misattribution (the 10-cell HUD diff was read as
   timer tick, correctly). But the flash itself was NOT investigated:
   call 13 re-saw the overlay pixels in the live frame and normalized
   them as "two black HUD boxes — static decoration".
3. **H2 (event data) — PARTIAL WIN, decisive behaviorally.** The
   strict criterion (textual citation of the flash log) was NOT met —
   flash_references scored 0 in all arms and no arm ever called
   check(), so the flash log was never seen. But the behavioral
   criterion is met by flash-aware alone: with the exception-flow
   injection IDENTICAL between continuity and flash-aware, only the
   flash-aware arm turned the flash region into its goal hypothesis
   and spent its remaining budget driving there. The divergence is
   attributable to the FLASH line.

Confounds held in the writeup:

- **Ground truth**: the recorded agent never completed level 1
  (levels_completed=0 across all 100 lines; timer deaths at 49/97).
  No arm could confirm its goal hypothesis in-window; flash-aware's
  "delivery destination" read stays epistemically open (though the
  flash firing exactly at goal-tile entry is strong evidence it IS
  the goal indicator).
- **First-call runaway**: every arm's call-1 python tool code spun
  into the 30M line-event budget (contained as designed; identical
  across arms, so no arm bias).
- **Exception flow at the DOWN**: fired in both clean arms (10-cell
  HUD diff); guidance text identical; continuity concluded "move NOT
  blocked" from it — the arms differ only in the FLASH line.
- **Mechanical verdict extractor**: `flash_references` scans assistant
  text + tool args but NOT tool results (where the FLASH line lives),
  and `blocked_entry_probes` is a coarse first-6-action-call scan —
  read the transcripts, not just the numbers.

## Live flash-aware run (2026-09-24, recording 6b32987c)

First live game under the continuity policy + flash surfaces, via the
yaml config seam (`SIMULATOR_CONFIG=config_simulator.yaml`,
`main.py --agent=simulatorfirst --game=ls20`):

- **80/80 actions over 2h16m** (13:59-16:16), 162 LLM calls, 0 levels
  completed — parity with the reference recording (no run has
  completed ls20 level 1 yet). One spiral escape at frame 6, one board
  reset at frame 46 (class path, correct).
- **Behavioral verdict — both citation criteria met live.** At frame
  43 a blocked UP fired a 76-cell timer flash; the model CITED it in
  update_notes ("When blocked, a white flash (5->0) appeared at rows
  53-62 cols 1-10 — bottom-left region. This is likely a status
  indicator region") and IMMEDIATELY probed the region (two
  print_region calls). The experiment's flash-aware arm never achieved
  textual citation — the live run did, on the first surfaced flash.
- **The wild confirmed the composite-flash shape**: the goal-entry
  flash at line 56 (block TL reaches (30,19), the dot) is 60 cells —
  top-box rows 9-15 cols 33-39 (24) + HUD timer rows 53-62 (36) —
  exactly the experiment's transition-17 prediction.
- **Surfaces actually consumed**: only the per-action FLASH line. The
  model never registered a simulate (set_simulate count 0), so check()
  and the sim-state-block flash log were never exercised live —
  consistent with the experiment (no check() calls in any arm).
- **Final world model**: the DOT at (32,21) is the goal; the top box
  is UNREACHABLE (block max-up is TL(15,34) — row 15 blocks passage;
  its sparse blue marker doesn't match the block's solid blue). Plan:
  center the block on the dot at TL(30,19) and retest. Reached
  (30,19) twice (lines 56/58) — level did not complete; the actual
  win condition remains open.

### Incident 6b32987c — composite-flash dedup suppression (FIXED)

The line-56 goal-entry flash **never printed**: `_flash_region_seen`'s
event-level ANY-seen check saw the HUD-timer region (already flashed at
42/43) and suppressed the whole event — including the NEW top-box
region, the one flash the presentation effort exists to surface. The
docstring promised region granularity ("fires only for NEW regions")
but the code checked at event granularity.

**Fix**: `_flash_has_new_region` — the FLASH line prints when ANY
region of the event is new; a stored-but-suppressed flash now logs a
DEBUG line (the only live trace — finding this required recording
forensics). Regression tests pin both sides: composite-with-new-region
prints, composite-all-seen stays suppressed. Mechanical gate re-PASS.

## Next step (proposal)

A hybrid before mechanical enforcement (B/C): when a flash fires and
the model has not called check() within N calls, inject a nudge
("FLASH fired at transition T — call check() to see the flash log
before planning"). The gap between flash-aware's behavioral
investigation and its missing textual citation is exactly a
prompt-injection gap, not a perception gap.

## Original plan (pre-build, kept for the record)

## Cost discipline

Local LLM (LM Studio), one request at a time, 60-130s/call (AGENTS.md
constraint). 3 arms + optional repeat pair ≈ 3-5 h wall time. Run arms
in separate processes, sequentially. The fake-LLM gate costs minutes
and must pass before any real run.