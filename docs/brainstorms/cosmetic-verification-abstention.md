# Cosmetic verification via abstention — closing the abb14eca failure modes

> **Status**: Approach A validated (2026-09-23) — **result: insufficient
> alone** (see "Validation status"). Follow-up experiment (flash-event
> presentation — the mechanical instrument this doc argues for) is
> planned and its infrastructure is built: see
> [`flash-event-presentation.md`](flash-event-presentation.md).
> Written 2026-09-22 from the
> abb14eca live-run investigation. Builds on the UNKNOWN-abstention rework
> (`8b6391b` abstained-cluster correlation log; `d509464` epistemically
> neutral abstention contract) and the structured-notes ledger discussion
> in this session. Experiment validation uses
> [`scripts/experiment_cosmetic_rule.py`](../../scripts/experiment_cosmetic_rule.py)
> (built for approach A; see "Validation status" below).

---

## Context (what the live run proved)

Live run `recordings/ls20-...abb14eca-e587-4aa0-ad7f-ec5cdb712e53.*`
(Sep 22, 100/100 actions, 198 LLM calls): no spiral (guard rework held),
sim 99% first registration → 100% by f13 → 88% plateau from f25, but
`levels_completed=0` — **level 1 never solved**. Two failure modes,
both confirmed against the recording and the decoded game source
(`environment_files/ls20/9607627b/ls20.py`):

1. **The state-feedback signal was labeled "cosmetic".** Landing on the
   goal tile with a rotation mismatch blocks the move and fires a red
   flash. The agent *wrote the correlation down* — seq 110 (f24):
   *"Errors are all in HUD decorations (rows 9-15, 53-62) that flash
   white on BLOCKED moves … cosmetic"* — and never revisited it. That
   flash was the game's only state-feedback mechanism, presented 4+
   times. Meanwhile the white cross at (19,30) — actually a hidden
   **rotation switcher** (overlap does `rotation=(rotation+1)%4` without
   blocking) — was filed as "STATIC decoration" (seq 182), and the top
   box — containing the goal tile — as "portrait" → "portal" → "= START,
   not goal" (seq 131→138→164→187). At f82–f83 the player's rotation was
   unknowingly goal-correct one move from the winning tile; the wrong
   notes-ledger walked it away and re-flipped the state.

2. **UNKNOWN abstention was never used.** Zero of ~150 code cells
   reference `UNKNOWN`; the final registered sim abstains nowhere;
   0/100 frames show any abstained-changed/stable cells. Coverage was
   always 100% — it predicted *everything*, absorbing the HUD as 12%
   persistent wrong-cells. Contrast with fdb596e0 (previous run), which
   DID abstain — but only *after* a 0%→crisis forced it.
   **The model abstains reactively, never proactively.** With no crisis,
   the procedural abstention teaching never triggered; instead it
   invented a third path the contract doesn't have: declare the region
   "decorative" in prose and move on.

Key insight: these are **one problem**. The abstained-cluster correlation
log (commit `8b6391b`) — per abstained region, which transitions changed
it and which actions produced them — is exactly the "check very carefully
before calling something cosmetic" instrument. It just never fired,
because abstention never happened.

---

## The three approaches

### A. Cosmetic-prohibition rule (system prompt)

"Decorative/cosmetic" is not an allowed free-text claim. Operational
definitions:

- A region is **cosmetic** only if it is *uncorrelated*: changes every
  frame or never, independent of actions and player position.
- A region that changes *conditionally* (on specific actions, specific
  positions, specific outcomes) is a **mechanic** — candidate signal.
- A region you cannot model → **abstain it** (`UNKNOWN`) and read the
  abstained log's correlation table; cite it for any cosmetic claim.
- **Contract violation to call out**: "X is cosmetic" while check()
  reports wrong-cells in X — the sim predicted X static and reality
  disagreed, so X is by definition *not* modeled and *not* proven
  cosmetic.

Location: `agents/simulator_agent/prompts.py` — **both** prompts carry the
abstention contract: `AGENT_SYSTEM_PROMPT` (line ~327, live agent) and
`SYSTEM_PROMPT` (line ~546, offline replay/sandbox). Patch both.

### B. Proactive-abstention trigger (check() output, small code change)

The "born at 96%, never burned" path: a high first-check score means no
crisis ever forces abstention. Convert a *score* observation into a
*signal* observation. When check() reports the **same wrong-cell cluster
persisting across ≥2 consecutive checks** and nothing in the sim models
that region, append one line to the check output:

```
persistent unmodeled cluster at rows 9-15 — consider abstaining (UNKNOWN)
to get its action-correlation via the abstained log
```

Location: `agents/simulator_agent/check.py` (the check-output builder —
where "Frames correct / Modeled / abstained log" lines are produced).
Needs a persistent-cluster detector: diff wrong-cell sets between the
last two check() results (both already cached in sim-state), require
≥N overlap (N≈4) and no UNKNOWN cells covering them.

This piece is what makes A enforceable: the prompt rule says "abstain
before claiming cosmetic", the check output *reminds* at the exact moment
the evidence accumulates.

### C. Ledger unknowns slot (update_notes schema + prompt)

From the structured-notes ledger discussion (same session): extend
`update_notes` (`agents/simulator_agent/prompts.py:377` schema,
`agent.py:543`/`659`/`854` handling, `conversation.py`, `replay_timeline.py`:

```json
"mechanics": [
  {"claim": "...", "status": "CONFIRMED|HYPOTHESIS|FALSIFIED",
   "evidence": "...", "falsified_if": "..."}
],
"unknowns": ["rows 9-15 flash white on blocked box entry — unexplained"]
```

- Sticky FALSIFIED: re-arming a falsified claim requires new evidence.
- Mandatory evidence field — no status without a cited observation.
- `unknowns` is the parking lot for exactly the unexplained signals the
  cosmetic rule would otherwise silently close. The HUD flash becomes a
  tracked open question instead of a prose verdict.
- Render the ledger above prose notes each turn; plans must reference
  status (e.g. "cover marker center" is a HYPOTHESIS re-armed for the
  4th time after two falsifications — visible, blockable).

Related directive from the same discussion (pairs with all three):
**probe rule** — "a move your sim predicts legal but reality blocks is a
mechanic — probe it from offsets; it may change YOUR state."

---

## How the three pieces interact

```text
unexplained region ──B──> abstention happens ──> abstained log: correlation table
      │                                                │
      └──C──> parked as unknowns[] entry        A: cosmetic claim must cite table
                                                     │
                        conditional correlation = mechanic → probe rule
                        uncorrelated = cosmetic (now evidenced)
```

abb14eca counterfactual: HUD abstained at first 88% check → abstained log
reads "rows 9-15: changed on transitions {34,57,76,83}, actions {1:4}"
→ A forbids "decorative" prose → C parks it in unknowns → probe rule
suggests blocked-entry experiments → the state-mismatch mechanic surfaces
as a HYPOTHESIS instead of a dismissed decoration.

**Honest caveat**: nothing mechanical makes the leap from "flash
correlates with blocked entry" to "the player has hidden state checked by
the blocker" — that inference stays with the model + the probe directive.
The instruments make the evidence unavoidable; the model still has to
read it.

---

## Validation plan (before any live run)

1. Extend `scripts/experiment_spiral_guard.py` with a variant covering
   A (+C once its schema lands). The B change is code, not prompt —
   validate via unit tests on synthetic consecutive check results
   (see house conventions in `.agents/skills/test-writing/`).
2. Seeds: replay **f24 and f57 turns of abb14eca** — mid-run, sim at
   88%, the cosmetic claim just made (seq 110) / about to be made.
   Both are pre-spiral, so the replay walker handles them; verify
   seed-fidelity with the md5 technique (strip variant deltas →
   compare to recorded system prompt).
3. Cost discipline: local LLM, one request at a time, 30–100s/call —
   one arm per seed, never parallel (AGENTS.md constraint).
4. Success criterion: the variant arm abstains on the HUD cluster
   within 2 turns of the f24 seed and either retracts the cosmetic
   claim or cites the abstained log; control arm (legacy prompts)
   reproduces the seq-110 mislabel.

## Open questions (defer)

- Does B's reminder need phase-awareness (EXPLORE vs MODEL) or does it
  fire on any consecutive-check persistence?
- Should A ban the *word* cosmetic or require the evidence citation?
  (Ban the claim, keep the word — the label is useful once earned.)
- Ledger scope: does C land in `update_notes` directly or as a separate
  tool call (`update_mechanics`) to keep notes cheap?
- Interaction with the sim-state block: does the abstained-log summary
  in the `[Simulator state]` message need the persistent-cluster line
  too (it currently renders coverage/abstention counts only)?

## Validation status (2026-09-23, approach A)

Harness: `scripts/experiment_cosmetic_rule.py` — same architecture as
`experiment_spiral_guard.py` (timeline-walker seed → production
`_run_tool_loop` → capped LLM seam → traced verdicts), two scenarios ×
two variants, one arm per process.

### Walker fixes required to make the seeds reconstructable

The abb14eca recording exposed two walker gaps, fixed in
`agents/simulator_agent/replay_timeline.py` (+ helpers in
`reconstruction.py`, regression tests in
`tests/unit/simulator_agent/test_reconstruction.py`):

1. **Live-parity errors** — 5 recorded python rows (seqs 48/73/89/96/97)
   error on re-execution because they errored live too (the model's own
   `or=[...]` SyntaxError, blocked `import sys`, a bfs that blew the
   line-event budget). The walker flagged any re-execution error as
   drift. Fix: `error_is_live_parity()` compares the re-executed error
   against the recorded tool result in the next llm-row's prefix
   (`"Error: <error>"` substring — sandbox caps output at 4 KB, log at
   40 KB, so the trailing error line is always preserved). Parity → not
   drift.
2. **Board-reset boundary** — the f57 walk crosses the frame-49 engine
   flash. Live, the tool loop ENDS the turn on the flash (remaining
   calls in the flashing row never run) and the next turn boundary
   consumes `_board_reset_pending` + `workflow.on_board_reset()`. The
   walker now mirrors both: consume the flag right after the flashing
   row's `run_code` returns, skip that row's remaining tool calls.
   Without it, post-reset batches stall silently (`BoardReset` is
   swallowed as `error=None`, the corpus freezes at 51, and every
   per-line embedded check from there on would mismatch).

Seed forensics (workflow counters must be seeded explicitly — the walk
replays `set_phase` rows but not guardrails): f24 → MODEL,
`check_failures=0` (reset by the frame-12 spiral switch),
`exception_flow_count=1`, `fired_for=1`; f57 → MODEL, `check_failures=2`
(seqs 108/110), `exception_flow_count=3`, `fired_for=1`.

### A/B results (14-call cap, 4 arms, all real-LLM, sequential)

| Arm | Calls→actions | check()/UNKNOWN | Cosmetic prose | Top-box (rows 9-15) treatment |
|---|---|---|---|---|
| f24 control | 14→2 | 2 / 1 | 0 explicit | never investigated |
| f24 cosmetic | 14→4 | 0 / 0 | **1 — "isolated/decorative HUD"** | **dismissed in prose (call 4)** |
| f57 control | 14→1 | 1 diagnose / 0 | 0 | investigated (diagnose + frame diffs) |
| f57 cosmetic | 14→1 | 2 diagnose / 0 | 0 | investigated (diagnose + frame diffs) |

**Verdict: A alone failed the success criterion** on f24 — the cosmetic
arm made the *exact* failure the rule bans ("top box (black, rows 9-15)
is isolated/decorative HUD", call 4) with zero check()/abstention
across all 14 calls. The rule text was verified in the system prompt the
model saw (md5 `81491402...` ≠ control `35c88500...`), so this is a
compliance failure, not a plumbing failure. Note the run-to-run variance:
neither arm's f24 control reproduced the seq-110 mislabel verbatim
(LLM sampling), and f57 arms show the rule neither helped nor hurt the
investigation pattern. The brainstorm's "honest caveat" is confirmed at
this model scale (qwen3.8-27b, local): a system-prompt prohibition does
not make the model read the evidence — **the enforcement instrument
must be mechanical, not prose**. This is exactly what B (the check()
persistent-cluster reminder) and C (the unknowns ledger slot) are for:
A's contract text stays as the definition, B fires it at the moment the
evidence accumulates.

A second structural finding (2026-09-23, follow-up investigation): the
flash never *reached* the abstention instrument at all — it was ingested
as **board state**. The blocked-move flash arrives as a HIGHLIGHT-class
frame stack whose game continues from the LAST layer, but
`settled_board` picks layer 0 (the flashed overlay) for HIGHLIGHT — so
the flashed board entered the corpus/history/turn image and became
permanent wrong-cell residue in every later check() (the 88% plateau's
root cause). The corpus-side fix (continuity extraction + flash events)
is built and era-flagged; the presentation experiment is planned in
[`flash-event-presentation.md`](flash-event-presentation.md).

Artifacts: `out/cosmetic_f24_{control,cosmetic}.{json,txt}`,
`out/cosmetic_f57_{control,cosmetic}.{json,txt}` (gitignored run
artifacts; regenerate with the script's `--scenario/--variant` flags).

## Not in scope

- Making the abstention *procedural* (agent-side auto-abstain on wrong
  clusters) — the model must choose to abstain; agent-side automation
  changes the epistemic contract.
- Per-game reverse-engineering of ls20 (house convention:
  game-agnostic mechanisms only).
- The falsified-goal guardrail (separate proposal from the same
  session's ledger discussion).