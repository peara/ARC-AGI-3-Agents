# Directed exploration — kill the classical learner, kill randomness

> Status: **experiment draft**. Not a committed plan. We test before we commit.
> Builds on the discussion in `llm-agent-loop.md` (slice 4) and the
> cold-start phase already implemented in `agents/templates/llm_curiosity_agent.py`.
> Supersedes the classical-learner path described in `perception-agent.md`
> and `llm-curiosity-agent.md`.

## The chicken-and-egg

Three layers that need each other:

| Layer | What | Owner today | When it settles |
|---|---|---|---|
| **A. Entity identity** | which cells = one entity, compounds, color roles, controllable | LLM coldstart + grouping (LLM-confirmed) | late — over many frames |
| **B. Physics rules** | "action X moves controllable by Δ" | classical learner (`effects/learn.py`) | early — random phase |
| **C. Game rules** | carry, score, terminal, targets | LLM rule proposer | ongoing |

The classical learner runs **B during the random phase on a half-formed A**.
When grouping later merges/splits entities, B's rules are keyed on entities
that no longer exist as-is → wasted / wrong rules. We have observed this in
multiple games: the classical learner proposes rules for entities that the LLM
later combines or ignores.

### Verified facts (grounding the design)

- Classical movement rules are keyed on `(entity_id, position_before)` —
  `guard_spec = {"dim": "pos", "of": <entity_id>, "eq": [...]}`,
  `Effect("pos", <entity_id>, "set"/"delta", ...)`. The `Rule` dataclass has
  **no role/color field** — only `entity_id`. Entity merges invalidate rules.
  (`effects/learn.py:167-224`, `effects/rules.py:94-102`)
- `plan_bfs` with zero movement rules is graceful: every action →
  `predict` returns `unknown=True` → collected as `UnknownAction` → BFS
  exhausts → returns `(None, unknowns)`. No crash. (`planning/search.py:49-84`)
- `RuleFirstPolicy` already gates on `len(ctx.movement_rules) > 0`
  (line 160-166) — so if we just stop calling the classical learner, the agent
  stays in random exploration. The fallback we need already exists.
- The LLM rule proposer already fires with zero rules — trigger is
  `if _residual or _observed_transition`, and `observed_transition` is set from
  raw `state_before → state_after` regardless of existing rules.
  (`agents/templates/llm_curiosity_agent.py:431-451`)

## Thesis

Randomness is the wrong tool for convergence. The classical learner needed a
bag of observations; an LLM-directed loop doesn't. Replace the random phase with
**directed, targeted exploration led by the LLM**, and let the LLM rule
proposer be the sole rule source — bootstrapped on a *settled* entity layer.

## Proposed architecture

```
coldstart    : LLM probes → color config (entities to track).        [exists today]
directed     : LLM-directed targeted exploration. Two sub-modes:
                - "settle": repeat an action until grouping + rule proposer
                            are confident on the entities/rules for it.
                - "explore": pick an action that exposes something NEW
                            (untried action, new entity, new region, or
                            anything plausibly tied to the win condition).
              No classical learning. LLM rule proposer is sole rule source.
              Exit when: both grouping and rules have converged enough.
llm_directed : LLM planner drives probes; LLM rule proposer + online
              residual engine refine rules. [exists today, minus learn.py]
```

The "directed" phase replaces today's "random" phase. It is *not* random — it
is a two-mode loop run by the planner:

- **Settle mode** — drive convergence on the current action/entity. Repeat the
  same action until grouping confirms the involved entities and the rule
  proposer confirms the movement rule for it. This is the "repeat action until
  both LLMs are confident" loop.
- **Explore mode** — drive *novelty*. Pick an action that exposes something
  not yet seen: an untried action, an unseen entity, an unexplored region, or
  anything plausibly tied to the win condition (mechanics notepad `objective` /
  `progress_signals`). This is *not* rule-work — the goal is to gather new
  observations so the next settle cycle has new material.

The planner picks the mode each cycle. The default cadence is: settle when
there is an in-flight action with unresolved grouping/rules; explore when the
current front is exhausted or stale.

## What already exists vs what we'd need to add

Verified against the codebase. This is the *real* scope, not a guess.

| Need | Exists? | Notes |
|---|---|---|
| Grouping LLM graded confidence | **NO** | Ternary verdict (`confirm`/`reject`/`split`) only. `ConfirmedGroup.confidence` is a support *count*, not a graded score. (`grouping/llm_engine.py:68-77`, `grouping/engine.py:38-43`) |
| Use grouping as convergence gate | Partial | Can count consecutive identical verdicts via `support`, but it's not an LLM-graded confidence. |
| Rule proposer "confident" signal | Partial | Proposer returns validated proposals; "confident" = "a movement rule for this action was confirmed by the online engine." Queryable via `ctx.movement_rules`. |
| Planner prompt mentions win condition | Indirect | `levels_completed` appears in `recent_actions` and `mechanics_hypothesis` only. No explicit win-condition wording in the system prompt. (`planning/llm_planner.py:35-162`) |
| Planner prompt has visited/frontier/novelty | **NO** | Not in prompt or bundle. `unknowns` and `coverage_gaps` exist but are residual-flavoured, not novelty-flavoured. (`planning/query.py:62-97`) |
| Planner prompt has "settle vs explore" mode | **NO** | Single mode: output one probe goal. |
| `ProbeGoal` has `kind`/`mode` field | **NO** | Uniform shape: `target + action + reason`. (`planning/probe.py:23-31`) |
| Mechanics notepad covers unexplored areas | **NO** | Only `objective`/`mechanics`/`progress`/`roles`/`next_steps`/`confidence`/`status`. (`planning/mechanics_notepad.py:27-41`) |
| `RuleFirstPolicy` graceful with empty rules | **YES** | Phase gate keeps it in exploration until `movement_rules > 0`. (`planning/rule_first.py:160-166`) |
| LLM rule proposer fires with zero rules | **YES** | Trigger is `observed_transition`, not residual. (`llm_curiosity_agent.py:431-451`) |
| Grouping skipped during coldstart | **YES** | `skip_grouping=self._phase == "coldstart"` at agent line 288 suppresses `_apply_compound_grouping` only; tracks + singleton entities still build every frame. (`entity/builder.py:188-196`) |
| Grouping engine replayable over historical frames | **NO** | Stateful: `_states`, `_confirmed`, `_debounce_buffer`, `_mismatch_counters`, `_prev_grid`. Cannot re-call on live registry; must re-ingest through a fresh `PerceptionSession` + `EntityBuilder` with color config pre-set. (`grouping/engine.py:385-397`, `grouping/combined_engine.py:68-83`) |
| Color config changes which entities grouping sees | **YES** | `set_color_config` → `_strip_ignored_entities` removes singletons whose colors have empty `track_dims`. Post-config catalog is smaller than pre-config. (`entity/builder.py:216-220, 302-320`) |
| Raw `_grid_history` enough to rebuild registry | **YES** | `ObjectRegistry.update()` is action-agnostic, only needs grid. `_grid_history` + `session.action_ids` suffice to re-ingest from scratch. (`perception/session.py:82-125`) |

### Sizing

This is the **medium** option, not the small one. Concretely:

1. **`ProbeGoal` + planner prompt** — add a `mode: "settle" | "explore"` field
   to `ProbeGoal` (`planning/probe.py`), and add a "settle vs explore" section
   to the planner system prompt (`planning/llm_planner.py:35-162`). *Small.*
2. **Novelty signal** — the explore mode needs the planner to *reason about*
   novelty. Add a `frontier` / `untried_actions` / `unseen_entities` field to
   `QueryInterface.bundle()` (`planning/query.py:62-97`) and surface it in the
   prompt. *Medium — needs a small tracker in `RuleFirstPolicy` or the agent.*
3. **Convergence gate** — instead of adding a graded confidence to the grouping
   LLM (large schema change), use cheap composite signals:
   - Grouping convergence: "no new `ConfirmedGroup` for N frames" (already
     queryable from `GroupingEngine` output history).
   - Rule convergence: "every action the planner has settled has a confirmed
     movement rule in `ctx.movement_rules`."
   *Small — no LLM schema change.*
4. **Rule proposer bootstrap mode** — with zero existing rules, the proposer
   triggers on raw `observed_transition`, not residual. Need to test whether
   the existing proposer prompt handles zero-rule state or needs a bootstrap
   variant that asks for movement rules from raw transitions. *Test first,
   then decide.*
5. **Delete the classical learner** — `effects/learn.py`, `effects/learn_multi.py`,
   the `learn_effect_context_multi` call in `rule_first.py:146`, re-exports in
   `effects/__init__.py`, offline scripts, and the dedicated tests. *Mechanical,
   once the above work.*

## Experiments to run before committing

We test before we commit. Each experiment is a *small, falsifiable* probe, not
a full implementation. Order matters — earlier experiments de-risk later ones.

### E0 — Grouping replay over coldstart history (head-start on convergence)

**Question:** After coldstart produces the color config, can we re-run the
grouping engine on the historical frames *with the color config applied* to
get a head start on entity convergence — so the directed phase starts with
compound entities instead of raw tracks?

**Background:** Today `skip_grouping=self._phase == "coldstart"` (agent line
288) suppresses compound grouping for ~6 frames. After coldstart, the color
config is known, but those ~6 frames of grouping opportunity are lost. The
grouping engine is stateful (debounce, confidence counters, mismatch history,
prev_grid for vision — `grouping/engine.py:385-397`, `grouping/combined_engine.py:68-83`),
so we cannot re-call it on the live registry. But `_grid_history` + `action_ids`
are enough to re-ingest from scratch through a fresh `PerceptionSession` +
`EntityBuilder` with the color config pre-set (`perception/session.py:82-125`).

**Method:** Take 3 recordings. For each, run coldstart to produce the color
config. Then spin up a fresh `PerceptionSession` + `EntityBuilder` with the
color config pre-set (via `set_color_config`), and re-ingest the cached
`_grid_history` in order. Compare:
- (a) `len(confirmed_groups)` and the set of `member_ids` after replay.
- (b) The live agent's `confirmed_groups` at the same frame index (grouping
  enabled post-coldstart only).

**Pass:** replay produces ≥1 `ConfirmedGroup` from the historical frames in
≥2 of 3 games, AND the replay catalog has fewer raw singletons than the
live agent's catalog at the coldstart boundary (i.e. color-config stripping
+ grouping actually did something). **Fail:** replay produces zero groups
or the same catalog as live — would mean the historical frames don't carry
enough signal for grouping, and the head-start is illusory.

**Why first:** E1 (proposer bootstrap) depends on whether the proposer sees
compound entities or raw tracks. If E0 succeeds, E1 runs against a
compound-aware catalog, which is the realistic target state. If E0 fails,
E1 runs against raw tracks and we know the proposer must handle that case.

**Caveat:** Replay will not produce *identical* confirmed groups to running
those frames live with grouping enabled from frame 0, because the grouping
engine's sequential state (debounce, vision prev_grid, mismatch history)
differs. The test is whether replay produces *useful* convergence, not
bit-identical convergence.

### E0 — Result: SOFT PASS with a blocking perception bug

Ran on `wa30` (`scripts/e0_grouping_replay.py`). Replaying 7 coldstart frames
with color config + grouping enabled produced **6 confirmed groups** (4
containment, 2 same-shape sibling) where today's coldstart produces 0. So
the historical frames *do* carry useful signal — E0 passes on its own terms.

But the **player compound was missed**. Investigation revealed two
compounding bugs:

1. **Registry matcher: exact `shape_key` with no rotation tolerance**
   (`perception/registry.py` Rule A, lines 248-265). The player head is a
   4-cell bar that rotates 90° when changing direction (1×4 ↔ 4×1). Rule A
   requires exact `shape_key` match, so the rotated bar is a "different
   object" — old track dies, new track born with `displacement=None`. This
   kills 2 of 4 directional displacements (the ones where the player turns).

2. **Co-movement: magnitude tolerance too strict for compound parts**
   (`grouping/heuristics.py` `_displacement_close`, tolerance=1). Even on
   frames where displacements *are* recorded, the head moves (7,0) while
   the body moves (3,0) — same direction, same frame, same action, but
   |7−3|=4 > tolerance 1. The heuristic sees only the static (0,0) frames
   as matched, which fail the `nonzero` gate.

The root issue is architectural: **ARC-AGI-3 games are discrete-state
(teleport, rotate, split/merge), but the per-frame matcher and grouping
heuristics assume spatial continuity.** A per-frame greedy matcher can't
reason "the bar rotated and jumped 7 cells — same object." It needs temporal
context. The information is in the frame history; the architecture just
doesn't expose it.

This is a perception bug independent of the directed-exploration work — it
affects the live agent today. But it's also a prerequisite: the settle loop
can't converge on the player compound if co_movement can't propose it.

### E0b — Sequence-aware grouping on coldstart history

**Question:** Given all coldstart frames *at once* (not frame-by-frame), can
a sequence-aware matcher correctly identify the player as one entity across
track-ID changes (head tracks 0→12→14, body tracks 10→13→15) and produce
the co-movement merge?

**Background:** E0 exposed that the per-frame registry matcher
(`perception/registry.py` Rule A) and the per-frame grouping engine both
assume spatial continuity — objects move small distances, shapes stay
roughly the same. In ARC-AGI-3's discrete grid games, objects teleport by
7+ cells, rotate 90° discretely, and split/merge. A per-frame greedy matcher
sees each transition in isolation and can't use temporal context that's
present in the history.

Reframing the grouping engine from "run frame-by-frame" to "run over a
sequence of frames with access to history" should do better — it can see
that the bar at frame 1 (horizontal) and frame 2 (vertical, 7 cells away)
is the same object because no other color-0 object exists and the rotation
is consistent with a direction change.

**Method:** Take the same coldstart recordings as E0 (3 games). For each,
extract all atoms (color, `shape_key`, centroid, bbox) from all N frames.
Run a **global assignment** over the sequence to reconstruct trajectories,
then feed the reconstructed trajectories to grouping. Test three approaches:

#### E0b-C — Classical sequence matcher

A simple recursive/beam-search assignment over the sequence:
- For each color, link atoms across frames into trajectories using
  **rotation-tolerant shape matching** (reuse `_normalize_shape_key` from
  `grouping/heuristics.py:125` which already handles 90° rotation/reflection).
- **No distance cap** — penalize large distances in the cost function but
  don't hard-reject. ARC-AGI-3 teleports are normal; a distance cap would
  re-introduce the continuity assumption.
- **Co-movement with direction-only tolerance**: match on
  `sign(d1) == sign(d2)` instead of `abs(d1-d2) <= tolerance`. "Same
  direction on same frame under same action" is the co-movement signal;
  magnitude difference is expected for compound parts of different sizes.
- The output is reconstructed trajectories (one per entity) with complete
  displacement histories (no `None` gaps from track-ID changes).

**Pass:** the player is identified as one compound entity (head+body merged)
from the 7 coldstart frames, with all 4 directional displacements present.
**Fail:** the player is still split across multiple track IDs, or
displacements are still missing.

#### E0b-L — LLM sequence matcher

Give the grouping LLM all 7 frames at once (grid images + per-frame atom
list with colors, shapes, centroids) and ask it to:
1. Identify distinct entities (grouping atoms into entities across frames).
2. For each entity, output the list of (frame, atom_id) pairs that belong
   to it.
3. For compound entities, output the member entities and the merge
   relationship.

The LLM can reason about rotation and teleportation naturally — it sees
the full visual sequence and can say "the horizontal bar at frame 1 became
the vertical bar at frame 2; same object, it rotated."

**Pass:** same as E0b-C. **Fail:** LLM can't track entities across
rotation+teleport, or produces inconsistent assignments.

#### E0b-H — Hybrid (classical → LLM → LLM adjudication on disagreement)

A tandem pipeline that uses both and asks the LLM to adjudicate:

1. **Classical pass** (E0b-C): produces candidate trajectories + proposed
   groups using rotation-tolerant matching + direction-only co-movement.
2. **LLM pass** (E0b-L): independently produces candidate trajectories +
   proposed groups from the full frame sequence.
3. **Agreement check**: compare the two outputs.
   - **Agree** → accept, done.
   - **Disagree** → run a second LLM call (adjudication) showing both the
     classical and LLM proposals + the evidence (grid images, per-frame
     atoms, displacements, shape rotations). Ask the LLM to pick the
     correct assignment with reasoning. If the LLM can't decide, fall back
     to the classical result (it's deterministic and conservative).

The hypothesis is that the classical pass handles the easy cases (color
uniqueness, obvious rotation) cheaply, the LLM handles the hard cases
(split/merge, ambiguous color sharing), and the adjudication resolves
disagreements with full context. This is more expensive than E0b-C but
should be more robust than either alone.

**Pass:** same as E0b-C/E0b-L, AND the adjudication resolves ≥80% of
disagreements (i.e. the LLM doesn't deadlock on "can't decide").
**Fail:** the hybrid is no better than the better of E0b-C/E0b-L alone,
or the adjudication deadlocks too often.

#### Shared test harness

All three variants use the same input (coldstart frames + atoms) and the
same output metric (player compound discovered? all 4 displacements
present?). The harness:
1. Extracts atoms from all N frames using `perception.objects.segment`.
2. Runs the variant-specific assignment.
3. Feeds reconstructed trajectories to co_movement (with direction-only
   tolerance).
4. Reports: entities identified, track-ID links, displacement completeness,
   confirmed groups, whether the player compound was found.

**Why after E0:** E0 established that the historical frames carry signal
and that the per-frame matcher is the bottleneck. E0b tests whether
sequence-aware matching fixes the bottleneck. If E0b passes, E1 (proposer
bootstrap) runs against a compound-aware catalog with complete
displacements — the realistic target state.

**Caveat:** A batch/sequence matcher is more expensive than per-frame. For
the live agent, the likely deployment is: batch on coldstart (7 frames,
one-time cost), then lightweight per-frame live with periodic
re-optimization. E0b only tests the batch-on-coldstart path; the live
incremental path is a separate concern.

### E1 — Rule proposer in zero-rule state (falsifies the foundation)

**Question:** Does the LLM rule proposer produce useful movement rules from
raw `observed_transition` when `ctx.movement_rules` is empty?

**Method:** Take a recording with a known movement rule (e.g. a simple
grid-step game). Replay the first 6 frames. Stub `ctx` to have zero rules.
Feed the proposer the raw `observed_transition` (no residual). Does it emit a
`kind="movement"` `Rule` that matches the known movement?

**Pass:** proposer emits a correct movement rule for ≥1 action in ≥1 of 3
test games. **Fail:** proposer only emits terminal/relational rules or
hallucinates — would mean we need a dedicated bootstrap prompt variant.

**Why first:** If the proposer can't bootstrap from zero, the whole
"LLM as sole rule source" thesis is dead and we stop.

**Depends on E0:** E0 determines whether the proposer sees compound entities
or raw tracks at bootstrap. If E0 passes, run E1 against a compound-aware
catalog (the realistic target). If E0 fails, run E1 against raw tracks and
note the gap.

### E2 — Grouping convergence under repeated actions (confirms the settle loop)

**Question:** Does `GroupingEngine` actually converge (produce stable
`ConfirmedGroup`s) when the same action is repeated N times, starting from
post-coldstart state?

**Method:** Take 3 recordings. For each, run coldstart, then replay a
fixed action (e.g. action 2) 5 times. Record `len(confirmed_groups)` and the
set of `member_ids` after each frame. Does it stabilize within N ≤ 5?

**Pass:** `confirmed_groups` is identical across 2 consecutive frames after
≤5 repeats, in ≥2 of 3 games. **Fail:** grouping never stabilises under
repetition — would mean the settle loop needs a stronger signal or a
different action strategy.

### E3 — Planner two-mode prompt (validates the planner can mode-switch)

**Question:** Given a modified prompt that explains "settle vs explore" and a
`mode` field on `ProbeGoal`, does the LLM pick `settle` when there is an
in-flight unresolved action and `explore` when the current front is stale?

**Method:** Build 3 synthetic bundle snapshots: (a) one action taken once,
grouping unconfirmed, no rule; (b) same action taken 4×, grouping confirmed,
rule confirmed; (c) all tried actions have rules. Send each through a
prototype two-mode prompt. Check the returned `mode`.

**Pass:** LLM returns `settle` for (a), `explore` for (c), in ≥2 of 3 cases.
**Fail:** LLM can't mode-switch from prompt alone — would need explicit
orchestration logic in the agent instead of LLM-chosen mode.

### E4 — Novelty signal necessity (is explore mode useful without frontier data?)

**Question:** Can the LLM pick a *novel* action in explore mode without an
explicit `frontier`/`untried_actions` field, just from `recent_actions` +
`mechanics_hypothesis`?

**Method:** Build a bundle with `recent_actions` showing actions {1,2,3}
taken, mechanics notepad with `objective`. Ask the LLM (explore mode) to
pick an action. Is it action 4 or 5 (untried) or a new region?

**Pass:** LLM picks an untried action or cites a new region in `reason` in
≥2 of 3 games. **Fail:** LLM re-picks a tried action — would mean we *must*
add the frontier signal before explore mode is useful.

### E5 — Classical-learner removal smoke test (de-risks the deletion)

**Question:** If we simply stop calling `learn_effect_context_multi` in
`RuleFirstPolicy.decide()`, does the agent still function (stay in
exploration, not crash, eventually produce rules via the proposer)?

**Method:** Stub `learn_effect_context_multi` to return `None`. Run the
agent on 2 short games (≤20 frames). Observe: (a) no crash, (b) agent
stays in exploration phase (phase gate works), (c) proposer eventually
emits ≥1 movement rule.

**Pass:** all three in both games. **Fail:** any crash or the agent gets
stuck with zero rules forever — would mean we need a manual bootstrap
before removing the classical learner.

### E6 — Stale-rule invalidation after entity merges (residual risk check)

**Question:** Once the entity layer merges mid-game, do LLM-proposed
entity-keyed rules get invalidated the same way classical ones do? Does
`prune_rules` / dormancy handle this, or do we inherit the volatility?

**Method:** Find a recording where grouping merges an entity mid-game
after an LLM-proposed movement rule was confirmed. Check whether the rule
is pruned/dormanted when the entity merges, or whether it becomes a
zombie rule.

**Pass:** rules for merged entities are moved to dormant or pruned. **Fail:**
zombie rules persist — would mean we need a merge-aware rule cleanup
*regardless* of removing the classical learner. Worth knowing either way.

## What we explicitly are NOT testing yet

- **Graded confidence from the grouping LLM.** Adding a `confidence` field
  to the verdict schema is a larger change; the settle loop can use the
  composite convergence gate (E2 + `ctx.movement_rules`) instead. Defer.
- **Per-game win-condition extraction.** We rely on the mechanics notepad's
  `objective`/`progress_signals` as the win-condition proxy. If E4 shows
  that's insufficient, we revisit — but we don't build a new win-condition
  module for this experiment.
- **Eval-path (Kaggle, no-network) implications.** The LLM rule proposer is
  dev-only today; the eval path is already classical-only. Removing the
  classical learner affects the eval path and needs a separate decision.
  This experiment is about the *dev* agent loop only.

## Decision gate

After E0–E6, we commit to a plan only if:

- E0 passes OR is skipped (grouping replay gives a head start) — **soft**.
  If E0 fails, E1 runs against raw tracks and the plan accounts for that.
- E0b: at least one of {E0b-C, E0b-L, E0b-H} passes (sequence-aware
  grouping fixes the track-ID instability + co-movement miss) — **soft
  but important**. If all three fail, the player compound can't be
  discovered from coldstart frames, and the settle loop has a structural
  gap. We proceed only with a documented workaround (e.g. defer compound
  discovery to the live directed phase).
- E1 passes (proposer can bootstrap from zero) — **mandatory**.
- E2 passes (grouping converges under repetition) — **mandatory**.
- E3 OR E4 passes (planner can mode-switch OR novelty works without frontier)
  — at least one, else the two-mode loop isn't worth it and we fall back to
  the simpler "explore phase feeds grouping, then hand off" design from the
  earlier discussion.
- E5 passes (removal doesn't break the agent) — **mandatory**.
- E6 informs the plan but isn't blocking — zombie rules are a separate fix.

If E1 or E5 fails, we stop and reconsider whether the classical learner is
actually removable at all.

## Related

- `docs/brainstorms/llm-agent-loop.md` — slice 4, the original planner loop.
- `docs/reports/llm-curiosity-agent.md` — current agent design (to be updated
  if we commit).
- `.omo/plans/` — *not* used yet; we save a plan only after the experiments
  pass the decision gate.