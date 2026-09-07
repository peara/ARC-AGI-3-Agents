# Brainstorm: ls20 frame-19 spiral — simulate() convergence failure

> Case study from run `ls20-9607627b.simulatorfirstagent.simulatorfirst.a21a2571`
> (2026-09-04). Agent spent 44 min / 37 tool calls at frame 19 trying to write a
> working `simulate()`, never emitted an action, terminated by the 36-call
> SpiralGuard cap. Cross-checked against 3 recordings (56 transitions total).
> Companion to the P1/P2 guardrail fixes — this doc covers WHY the model
> couldn't converge, independent of the loop-control issues.

## TL;DR

The game IS learnable from the data the agent had: I verified a hand-written
simulate with **0 wrong cells (excl. HUD) across all 3 recordings**. The agent
stalled at 40.8% because of (a) an off-by-one misread of the `history` array
that kept corrupting the action→direction mapping, (b) a non-obvious collision
model (feet-only) it never tested, (c) never calling `diagnose()` despite it
being built for exactly this, and (d) oscillation — rewriting verified notes
instead of building on the best-so-far attempt.

## The true game rule (ls20-9607627b, decoded)

1. **Sprite**: 5×5 block — orange(12) body on top 2 rows, blue(9) feet on
   bottom 3 rows. Moves exactly **5 cells per action**. Mapping:
   `1=up, 2=down, 3=left, 4=right`.
2. **Feet-only collision**: walls are black(5) AND off-black(4), checked ONLY
   on the destination feet rows (rows +2..+4 of the 5×5). The **body may
   overlap walls** — at the top, the sprite visually sits on the goal square's
   black interior (row 15 under the orange body). A whole-block collision test
   makes the winning move impossible.
3. **Per-cell terrain restore**: vacated cells revert to 5 inside the goal
   square interior (rows 9–15, cols 31–41), 3 elsewhere. A single constant
   clear color fails ~half the transitions.
4. **Hidden "pound" rule**: bottom-left box (rows 53–62, cols 1–10) flips to
   white(0) iff the sprite presses UP while pinned at row 15; any other action
   at the top reverts it to black. Verified 77/77 transitions, but only ~2
   flips per recording — hard to discover in one session.
5. **Color 9 is shared** (feet + goal glyph + box glyph): naive
   `find_color(9)` localization breaks; anchor on orange(12) instead.

## What blocked the agent (ranked)

### Trap 1 — history off-by-one poisoned the action mapping (biggest)

`history[i] = {action, frame}` where `frame` is the grid AFTER `action` fired.
To derive a transition you must pair `history[i-1].frame` + `history[i].action`.
The model repeatedly paired `history[i].action` with the *next* transition.

Proof: its "corrected" mapping `{1:down, 2:left, 3:right, 4:up}` is exactly
what the off-by-one pairing yields from this run's first 4 transitions. It had
recorded the CORRECT mapping in notes early (from live observation), then
"corrected" it to the wrong one via this misread, and flip-flopped for the rest
of the spiral. With a wrong mapping, every moving transition predicts the
sprite in two wrong places — ~52 wrong cells/frame regardless of everything
else being right.

### Trap 2 — feet-only collision never discovered

Every attempt used whole-5×5 (or body) collision. That makes the winning move
(up onto the goal square, body overlapping black) impossible, and makes blocked
moves look random. The evidence was visible in check() output (up-onto-goal
transition = 50 wrong cells with the sprite landing exactly on row 15) but
never localized.

### Trap 3 — diagnose() never called (0 of 72 LLM calls)

`diagnose()` prints MISSED / SPURIOUS / WRONG_VALUE clustered by spatial
region — it would have pointed straight at "cluster rows 15–19 cols 34–38"
and "cluster rows 53–62 cols 1–10". The model only saw check()'s aggregate
+ ≤20 sample cells and iterated blindly on constants (step size, clear color,
mapping).

### Trap 4 — oscillation instead of convergence

Each attempt rewrote the mapping from scratch (sometimes right, sometimes the
off-by-one version), overwriting its own verified notes, never building on the
best-so-far version. Accuracy went 1.4% → 34.3% → 0.0% → 37.1% → 40.8% —
regressions from self-inflicted rewrites, not from new information.

## Candidate fixes (deferred / partial)

| Idea | Status | Rationale |
|---|---|---|
| **`transitions()` sandbox helper** — returns `(before_grid, action, after_grid)` triples, removing the need to hand-derive pairing from `history` | Candidate | Kills Trap 1 deterministically (~10 lines in `_build_namespace`). Risk: the model may ignore it, as it ignored `diagnose()`. Cheapest possible Trap-1 fix though. |
| **Document `history` semantics in the sandbox/prompt** — one line: "history[i].frame is the grid AFTER history[i].action fired; pair history[i-1].frame with history[i].action for transitions" | Candidate | Zero-code alternative to `transitions()`. Doc-only; relies on the model reading it. |
| **Preload frame 0 into sandbox (`grids[0]`/`current_frame` at init)** | Candidate | Minor: the sandbox in live mode starts with grids=[]; check() has nothing to test against until 2 actions have fired. Seeding the RESET frame gives check() a usable transition table 1 turn earlier. Low impact on this run (the model had 19 frames by frame 19) but cheap. |
| **Mandate diagnose() after a failed check()** — e.g. when accuracy < X%, inject a directive: "run diagnose() before your next rewrite" | Deferred (user decision) | Kills Trap 3, but adds prompt pressure mid-loop; revisit after P1/P2. |
| **Verified-facts pinning** — append-only section in notes for confirmed mappings; or prompt guidance: "don't rewrite verified mappings unless live evidence contradicts" | Candidate | Kills Trap 4's worst behavior (self-overwrite of correct facts). |
| **Pound-rule discoverability** (more recordings, rule-proposer integration) | Open | The 2-flips-per-recording pattern is inherently hard; maybe fine to leave undiscovered — planning can still win without it. |

## Score context

- Model reached 40.8% check accuracy at termination and was climbing.
- 60/80 actions were still unspent when the game ended.
- With the mapping un-corrupted (Trap 1 fixed alone), a feet-collision +
  terrain-restore simulate reaches 100% on this data (verified by hand).

## Follow-ups

- [x] P1: log ContextVar propagation into sandbox worker thread (observability regression from 24e7dce)
- [x] P2: evaluate workflow guardrails inside the tool loop (MODEL→EXPLORE on 5 check failures)
- [ ] Decide: `transitions()` helper vs history-semantics doc line (or both)
- [ ] Decide: preload frame 0 into live sandbox
- [ ] Later: diagnose() mandate; verified-facts pinning