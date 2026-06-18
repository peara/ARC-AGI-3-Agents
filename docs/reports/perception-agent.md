# Perception-First Agent — Research Report

> A living design document (not a dated diary entry). Captures our conjectures,
> the reasoning behind them, what we've built, and what we found, so we can
> refer back and revise as evidence comes in.
>
> Last updated: 2026-06-18 (LLM rule proposer + planner loop; see llm-agent-loop.md)

---

## 1. Problem framing

ARC-AGI-3 is an **interactive** benchmark: an agent observes a 64×64 grid of
colour indices (0–15), picks an action (`RESET`, `ACTION1`–`ACTION7`, some with
`(x, y)` coords), and must discover the game's rules from sparse feedback over
long horizons. Scoring is **RHAE** (Relative Human Action Efficiency, roughly
`(human_actions / your_actions)²` per level), with an action budget ~5× human —
so brute force is punished hard.

Key constraints shaping our approach:

- **Kaggle prize track is offline**: no external LLM APIs at evaluation. Only
  bundled weights + classical compute. So our core must run without network.
- **Frontier models score <1%** on the official leaderboard; humans ~100%. The
  gap is by design. This is **hard to overfit** — see §2.
- Raw grid → LLM is a poor interface (4096 cells, no spatial bias). VLMs still
  struggle with fine grid logic.

## 2. Conjectures

These are beliefs we hold with varying confidence; each should be falsifiable.

- **C1 (competition intent).** The 7-month duration + <1% scores mean this is a
  **standing generalization benchmark**, not a sprint to a tunable harness. The
  winning property is *generic adaptation to unseen games*, not per-game
  reverse-engineering. → We should bias toward game-agnostic mechanisms.
  Confidence: medium.
- **C2 (perception interface).** Feeding an LLM a raw JSON grid does not work.
  The fix is **abstraction/compression** (objects + relations + diffs), not a
  different serialization. Confidence: high.
- **C3 (objectness is causal, not visual).** A single frame is *underdetermined*
  — we cannot know whether objects group by colour, adjacency, or shape, nor
  which blob is the agent. **Interaction is the oracle**: things that change
  together under an action are one object; the thing that moves with directional
  actions is the agent. Confidence: high (this is the central bet).
- **C4 (classical first, learning later).** With no labels on day one, a
  supervised CNN has nothing to learn. Classical connected components +
  interaction-driven effect statistics get us far and stay debuggable. A learned
  **action-conditioned forward model** is reserved for dynamics/planning once
  the basics work. Confidence: medium-high.

## 3. Approach — the perception ladder

Build and validate one rung at a time; each rung is independently testable and
reusable, and rungs 1–4 need **no training and no network**.

1. **Static perception** — connected-component candidate objects under multiple
   grouping hypotheses (don't commit to one segmentation). ✅ done
2. **Delta + common-fate binding** — cluster cells that change together across
   steps to merge/split candidates. ✅ done
2.5 **Persistent object registry** — stable ids across an episode; roles &
   entities derived from trajectories. ✅ done
3. **Controllable-object identification** — correlate actions with object motion;
   tag controllable entity + observed motion-by-action. ✅ done (v1 heuristic)
4. **EffectModel + roles** — `effects.predict` + relational rules (terminal, counter);
   classify wall / pickup / hazard / door by *consequence*. ✅ slice 2 (terminal +
   counter; overlap/`exists` deferred). ✅ slice 3 (residual → propose / confirm /
   prune engine; g50t abstains, no latent templates).
5. **Partial-state planning** — snapshot → `effects.predict` → BFS on a
   caller-defined subset of state; verify against recordings. ✅ v1 (movement)
6. **Curiosity-driven live agent** — random cold start until a controllable
   emerges, then BFS toward the *unknown* with a per-step verify→replan loop. ✅ v1

Where the LLM fits (dev only, not Kaggle eval): consume the **compact symbolic
scene** the perception layer emits and propose high-level hypotheses ("looks
like a key/door game"). LLM proposes, the interaction loop disposes.

## 4. Progress

### Rung 1 — static perception (done)

New, dependency-light (`numpy` + `pillow`), Kaggle-portable package:

- `perception/objects.py` — `to_grid`, `infer_background`, `segment`,
  `segment_hypotheses`, `GameObject` (bbox, centroid, size, translation-invariant
  `shape_key`), `Scene`, `scene_summary`.
  - Grouping hypotheses exposed: `color4`, `color8` (same-colour 4-/8-connected)
    and `any8` (colour-agnostic non-background blobs).
- `perception/viz.py` — `render_grid`, `overlay_objects` (labeled bboxes),
  `hstack` for side-by-side hypothesis comparison.
- `scripts/perceive_recording.py` — run on any `*.recording.jsonl`, dump overlay
  PNGs + console object summaries. Offline.

Run example:

```bash
uv run python scripts/perceive_recording.py \
  recordings/ls20-9607627b.random.80.*.recording.jsonl --frames 0,2,5
```

## 5. Results & observations (ls20 random recording)

### Rung 1 (static perception, old recording)

Used initially: `recordings/ls20-9607627b.random.80.b21c2002-*.recording.jsonl`

- **Recording data shape**: each event = a `(frame, action_input)` pair; frames
  are 1 layer of 64×64. Colours present: `{0,1,3,4,5,8,9,11,12}`.
- **Background auto-detected = colour 4** (green, ~2609 cells). The big dark-green
  room is one object (colour 3, size 892).
- **`color4`** yields ~19 clean objects; **`any8`** collapses them into ~4
  rooms/blobs — the segmentation ambiguity (C3) made visible.

### Recording fix + fresh labeled data (2026-06-11)

**Problem:** API-returned `frame.action_input.id` is always `0` (RESET) even after
real steps. Old recordings looked "broken" but the game state *was* updating.

**Fix:** `agents/agent.py` now overwrites `action_input` in recordings with the
action the agent actually sent (`append_frame(frame, action)`).

**New reference recording (legal actions only):**
`recordings/ls20-9607627b.random.80.4778fe67-d8c5-4853-90d8-36aff85bb079.recording.jsonl`
(81 events, actions 1–4 only; random agent now samples from `available_actions`)

Previous recording `dece8d0e-*.recording.jsonl` mixed in illegal ACTION5–7 picks
(~40% of steps); keep for comparison but prefer `4778fe67-*` for Rung 2+.

**Random agent fix:** `agents/templates/random_agent.py` now chooses only from
`latest_frame.available_actions` (falls back to all non-RESET if empty).

**Action→motion on colour 12 (player confirmed):**

| Action | Effect on colour-12 centroid |
|--------|------------------------------|
| ACTION1 (id=1) | dy = −5 (up) |
| ACTION2 (id=2) | dy = +5 (down) |
| ACTION3 (id=3) | dx = −5 (left) |
| ACTION4 (id=4) | dx = +5 (right) |

Colour 9 also shifts with some moves but colour 12 tracks cleanly with
ACTION1–4. On ls20, `available_actions` is always `[1,2,3,4]` — interact/click/undo
are not offered in this game state.

→ Ready for Rung 2 (delta + common-fate) and Rung 3 (controllable-object ID).

### Rung 2 — delta + common-fate (done, exploratory)

Code: `perception/motion.py` (`compute_delta`, `track_objects`, `bind_common_fate`,
`build_transitions`, `aggregate_by_action`), `perception/viz.py:draw_motion`,
`scripts/analyze_motion.py`. Tracking = `shape_key` + nearest-centroid match;
common fate = group matches by shared displacement vector.

Run:

```bash
uv run python scripts/analyze_motion.py \
  recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl --steps 1,2,3,7
```

**Findings (each one a real discovery from the instruments, not assumed):**

1. **The player is a *compound* object bound by common fate.** Colour 9 (15 px)
   and colour 12 (10 px) translate by the *same* vector on every move
   (agreement 1.0). So "the agent" = {c9 ∪ c12}, 25 px — discovered, not
   hardcoded. Confirms C3. `langgraph_thinking`'s "colour 12 = player" is only
   half the object.
2. **Action→displacement map (dr, dc), confirmed via tracking:**
   ACTION1 `(-5,0)` up · ACTION2 `(+5,0)` down · ACTION3 `(0,-5)` left ·
   ACTION4 `(0,+5)` right. Moves are 5-cell steps.
3. **`appeared`/`vanished` never fire on ls20 — all motion is `recolored`.**
   The playfield *floor* is colour 3 (non-background); true background is
   colour 4 (outer border). The player moves over floor, so cells go
   non-bg→non-bg. ⇒ A single global background is the wrong model; we need a
   **local/floor-aware background** or to lean on the `recolored` channel +
   tracking rather than appeared/vanished.
4. **Blocked move = a crisp signature: `changed≈2`, `moving_objs=0`.** When the
   player walks into a wall it doesn't move; only ~2 cells change (likely an
   energy/step HUD counter). Cheap, reliable "I hit a wall / nothing happened"
   detector — useful for search pruning later.
5. **Whole-screen flash, invisible to metadata (steps 42–43).** Frame 42 is a
   *single colour* (all 4096 cells = colour 11), then frame 43 repaints the
   level. Throughout: `state=NOT_FINISHED`, `levels_completed=0`,
   `full_reset=False` — so this is **not** a death/level event and the event
   metadata never flags it. Death (`state=GAME_OVER`) and level-complete
   (`levels_completed` increments) *are* in metadata, so we don't need a
   transition detector for those; but unknown perceptual events like this flash
   exist. ⇒ Need a cheap **degenerate-frame guard** (`n_unique==1` or
   near-total delta) so flashes don't corrupt tracking (the all-colour-11 frame
   would otherwise become one giant bogus object). These outliers also inflate
   naive per-action means (action 2 mean_changed 281 vs action 1's 44).
6. **Tracking noise is visible and rare.** Floor fragments occasionally form a
   matching `shape_key` and produce a spurious large-displacement match (n=1,
   colour 3). Low frequency; filterable by size/agreement/Δ magnitude.

Visual confirmation: `motion_out/motion_003.png` shows both c9 and c12 with
parallel `(0,+5)` arrows.

### Emerging insight: object *kinds* need different identity mechanisms

"Object" is not one thing. Each kind is re-identified across frames by a
*different* correspondence cue, and a single matcher can't cover all of them
(a mover keeps its shape but changes position; an in-place HUD counter keeps its
position but changes shape). This motivates a persistent object registry with
multiple matchers and per-object property trajectories.

| Object kind | Re-identification cue | Defining property |
|-------------|----------------------|-------------------|
| Player / movers | common fate (shared displacement) | translation |
| HUD counter | in-place positional/support overlap | size/length over time (monotonic) |
| Walls / floor / structure | persistence (unchanged) | stable position |
| Key ↔ door (the **goal**) | shape match under linear transform (scale/rotate) | compound-shape signature |
| Flash / transition frames | perceptual anomaly (1 colour / huge delta) | absent from metadata |

The key↔door relation is special: it is (likely) the **win condition** — a
compound shape (inner pattern inside a box) that must be matched between the
bottom-left key and the top-middle door under scale (and possibly rotation),
where even the box sizes differ. A dedicated transform-invariant matcher, built
*on top of* stable object identities.

### Rung 2.5 — persistent object registry (done, lean v1)

Code: `perception/registry.py` (`ObjectRegistry`, `is_degenerate`, `derive_roles`,
`derive_entities`), `perception/viz.py:overlay_tracks`,
`scripts/track_recording.py`.

Design (agreed): atoms = colour-pure CCs; **action-agnostic** matching cascade
(A rigid shape+colour+nearest-centroid, B cell-IoU+colour for in-place mutators,
C containment); 1-to-1 with appear/disappear; degenerate-frame guard carries ids
across flashes; floor handled by tagging huge atoms `structural`. Roles & entities
are a separate derived pass so they emerge from trajectories.

Run:

```bash
uv run python scripts/track_recording.py \
  recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl --frames 0,10,41,43
```

**Findings (ls20, 81 frames):**

1. **Stable ids held for the whole episode.** 23 tracks, almost all `n_obs=80`,
   `lifespan=81` — including across the colour-11 flash at frame 42. The
   degenerate guard fired exactly once (frame 42, `n_unique==1`) and ids carried
   across the gap. ✓
2. **Player recovered as a common-fate entity:** tracks `#14 (c9)` + `#18 (c12)`,
   both `n_move=56`, `centroid_span≈40` — discovered, not assumed. ✓
3. **Compound key/door surfaced by containment** (visual: `track_out/track_010.png`):
   - `#7` (gray box, bottom-left) = **key**, contains pattern `#15` (c9, size 20).
   - `#6` (gray box, top) = **door**, contains pattern `#12` (c9, size 5).
   - `#15` (size 20) vs `#12` (size 5) = the **same-shape / different-scale**
     relation — the goal signal for Rung "key↔door".
4. **Two known v1 rough edges (keep simple, refine later):**
   - **Floor false-positive "mover".** `#3` (floor, c3, ~890 px) tagged `mover`
     because the flash induced `n_move=2`. Mover criterion is too loose
     (`n_move>=2`); should require motion as a fraction of lifespan or
     action-correlation (Rung 3), and/or exclude `structural`.
   - **Containment over-fires via huge bbox.** The floor's bbox spans the room,
     so "everything inside #3" (21 containment hits, most spurious). Fix: use
     **cell-containment** not bbox, and/or skip `structural`/oversized outers.
     The real compound relations (#15⊂#7, #12⊂#6) are present but buried.
5. **No `counter` role detected yet.** Expected an energy/HUD counter (the
   "2 cells change on blocked move" from Rung 2). Gray bars `#5`/`#8` stayed
   constant in this run — needs a recording where energy visibly depletes, or a
   finer in-place-change detector. Open.

### Rung 3 — entity layer + controllable detection (done, v1)

Code: `perception/entities.py` (`Entity`, `EntityCatalog`, `build_entities`),
`perception/roles.py` (`detect_controllable`, `HeuristicRoleAssignerV1`,
`assign_roles`), updated `scripts/track_recording.py`.

Design:

- **Three layers:** Track (atom, action-agnostic) → Entity (grouped tracks) →
  role/affordance labels (action-aware derivation only).
- **`build_entities`:** common-fate compounds + singleton leftovers. Container
  grouping (key/door) deferred — see what we get first without it.
- **`assign_roles`:** pluggable assigner; v1 runs `detect_controllable` only.
  Detectors emit `RolePatch`es; failure returns an unchanged catalog (no crash).
- **Safe accessors:** `catalog.controllable()`, `catalog.observed_motion_by_action()`
  — callers must handle `None` when detection fails on other games.
- **Naming:** role `"controllable"` + affordance `controllable=True`; observed
  stats in meta as `motion_by_action` (not ground-truth physics).

Run:

```bash
uv run python scripts/track_recording.py \
  recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl --frames 0,10
```

**Findings (ls20, 81 frames):**

1. **Controllable entity recovered:** entity `#0` compound `{#14, #18}` (c9∪c12),
   `motion_agreement=1.0`. Matches Rung 2 common-fate player. ✓
2. **Observed motion-by-action:** `{1:(-5,0), 2:(+5,0), 3:(0,-5), 4:(0,+5)}` —
   same map as motion analysis; now attached to the entity catalog. ✓
3. **Structural floor excluded from controllable detection.** Track `#3` still
   reads as `mover` in per-track `derive_roles`, but is not tagged controllable
   (structural filter in detector). Partial fix for the false-positive mover. ✓
4. **Failure path is safe.** Forcing impossible agreement → `controllable()` is
   `None`, catalog otherwise intact (22 singleton/compound entities, no labels).
5. **Heuristic is game-shape-specific.** Action→displacement correlation works on
   ls20 but may fail on click-only or non-translational control — by design the
   detector returns no patch rather than guessing.

### Rung 5 — partial-state planning (done, v1 movement)

Code: `effects/` (`SceneState`, `MovementModel`, `learn_movement_model`, `predict`,
`predict_move`), `planning/` (`PlanSpec`, `snapshot`, `plan_bfs`, `recording_eval`),
`scripts/plan_recording.py`, `tests/reference_recordings.json`,
`tests/unit/test_planning.py`.

Design:

- **`SceneState` is ephemeral** — built per BFS call from a caller `PlanSpec`
  (which entity ids, which dims, goal predicate). Agent/LLM chooses the spec;
  BFS does not own persistent state.
- **Goals use entity ids**, not roles — supports multi-controllable games; roles
  are only a discovery aid (`catalog.controllables()`).
- **Partial state:** only `relevant` dims are hashed and planned over; volatile
  dims (e.g. stamina) can be recorded but excluded from dedup.
- **No wall ontology:** movement model learns observed `(pos, action) → next_pos`
  and `(pos, action) → block` from the episode; unseen pairs extrapolate via
  `motion_by_action` only. Open-loop verify checks each step against recording
  where observed (`matched` / `extrapolated` / `diverged`).
- **Dev loop:** `tests/reference_recordings.json` lists plan cases per game;
  pytest parametrizes over them. Add a recording + entity id + frame pair to
  test a new game. Run with `uv` — see `tests/README.md`.

Run:

```bash
uv sync --group dev
uv run pytest tests/unit/test_planning.py -v

uv run python scripts/plan_recording.py --manifest-case ls20-random-legal-e0-f0-g10 \
  --verify-segments
```

**Findings (ls20, manifest cases):**

1. **BFS finds paths between frame positions.** Frame 0→10: plan `[4, 1]` (2 steps);
   frame 0→40: 6 steps; predict replay reaches goal. ✓
2. **Segment verify: all steps `matched` on 0→10 case** — plan steps align with
   observed `(pos, action)` transitions from the recording (no extrapolation needed
   on that short path). ✓
3. **Movement model: 30 known transitions, 9 known blocks** — all from observation,
   not assumed solids. ✓
4. **Live verify/replan deferred** — recording validates observed physics; live
   agent loop (execute → re-snapshot → replan) is the next increment for unseen
   `(pos, action)` pairs. → built in Rung 6.

### Rung 6 — curiosity-driven live agent (done, v1)

Code: `perception/session/` (`PerceptionSession`, `SceneSnapshot`),
`planning/` (`ExplorationConfig`, `ExplorationPolicy`, `Planner` protocol),
`agents/templates/curiosity_agent.py` (`Curiosity` agent),
`tests/unit/test_exploration.py`.

The idea (from the design discussion): confidence drives behaviour. At the start
*nothing is confirmed* — we don't know which blob we control — so the agent acts
randomly to generate action→effect evidence. Once `detect_controllable` fires, it
switches to using the movement model + BFS to steer the controllable entity
toward the **unknown**, closing the live loop Rung 5 could only check offline.

Design:

- **Three layers.** `PerceptionSession` owns registry + catalog and emits
  `SceneSnapshot` after each ingest. `ExplorationPolicy` (a `Planner`) reads
  snapshots only — no perception state. The `Curiosity` agent orchestrates:
  `session.ingest()` → `policy.on_observed()` → `policy.decide()`. An LLM planner
  swaps in at the policy slot without touching the session.
- **Phase 1 — cold start (curiosity = ignorance).** Until a controllable entity
  is confirmed *and* `min_random_steps` probes have run, pick a random legal
  action. The registry/roles pipeline watches passively (it is action-agnostic by
  design), so the random phase *is* the data-collection phase.
- **Phase 2 — BFS toward the unknown.** With a controllable entity in hand, learn
  the movement model and pick a curiosity target in two tiers: (1) the nearest
  **unconfirmed, non-structural entity** (likely interactive — a thing to bump
  into), else (2) the nearest **unvisited frontier cell** (`goal = pos ∉ visited`).
  BFS plans to it; the agent executes one step at a time.
- **Verify → replan loop (the new bit).** Before sending action `a`, the planner
  records its expectation `(pos_before, a, predicted_after)`. On the next
  observation it compares `predicted_after` to the actual position. A mismatch —
  an extrapolated move that was actually **blocked**, lost tracking, or an
  unexpected jump — drops the stale plan. No separate wall ontology is needed:
  the live transition/block is already in the registry, so the next
  `learn_movement_model()` absorbs it and the replan routes around it.

**Findings (offline `GridWorld` sim + reference recording):**

1. **The whole loop runs without game knowledge.** In a boxed 30×30 room the
   planner discovers which 3×3 blob is the player, learns the *exact*
   action→displacement map (matches simulator truth), then leaves the random
   phase for `frontier` BFS. ✓
2. **Verify loop fires on a genuine surprise.** When BFS extrapolates a move
   through an unseen wall, the step `diverged`, the plan was dropped, and the
   block was absorbed into `model.known_blocks` (≥1) so later plans avoid it. ✓
3. **Curiosity spreads the agent out.** Visited-cell count grows well beyond
   idling — the frontier goal keeps pulling the controllable into unexplored
   lattice. ✓
4. **Real-data wiring confirmed.** Replaying the ls20 reference recording through
   `PerceptionSession.from_recording()` recovers the controllable and a 5-cell-step
   motion model — same result as the dedicated Rung 3 detector. ✓
5. **Open edges (v1).** Rebuilds entities+roles every frame (fine at ls20 scale,
   may need incremental update on busy games). Tier-1 entity targeting re-issues
   BFS for unreachable entities each replan (wasted budget, not wrong). Action
   budget (RHAE) not yet optimised — this is a research driver, `MAX_ACTIONS=200`.

### Rung 7 — LLM rule proposer (done, dev-only)

Code: `planning/llm_rule_proposer.py` (`NULL_RULE_PROPOSER`, `SYSTEM_PROMPT`,
`parse_proposals`, `validate_proposal`, `make_rule_proposer`),
`planning/llm_planner.py` (`call_rule_proposer`, `_build_rule_proposer_messages`,
`_extract_scene_entities`, `_extract_engine_rules`).

The rule proposer closes the loop between observation and hypothesis. After
`engine_step` runs and produces **unexplained residuals** (prediction mismatches
that no confirmed rule accounts for), the proposer is invoked to generate new
`Rule` hypotheses from those residuals. The same propose → confirm → prune
lifecycle applies: the engine's existing machinery confirms or prunes LLM-proposed
rules just like classical ones.

**What it does:**

1. Takes a scene bundle + residual list + optional failure context.
2. Sends a structured prompt to the LLM describing guard/effect DSL syntax.
3. Parses the LLM's JSON response into proposal dicts.
4. Validates each proposal: kind check, guard parse, entity ID existence, effect
   structure, DSL→Rule conversion.
5. Deduplicates against confirmed engine rules and within the batch.
6. Returns a list of `Rule` objects ready for engine confirmation.

**Eval path:** `NULL_RULE_PROPOSER` is a zero-argument stub that always returns
`[]`. It satisfies the `RuleProposerFn` signature on the Kaggle offline path
where no network is available. The classical engine continues to propose from
templates; only the LLM channel is silenced.

**Trigger:** In the `llm_directed` phase, after `engine_step` fires and the
residual is non-empty. The `make_rule_proposer` factory wraps an LLM callable
with a cooldown circuit breaker (default 5 seconds between calls) to avoid
hammering the API on fast game loops.

**Key design choices:**

- **Network-free module.** `llm_rule_proposer.py` and `llm_planner.py` take an
  `llm_call` callable, not an API client. The agent injects the actual LLM
  connection. This keeps `planning/` free of API dependencies and makes testing
  straightforward with mock callables.
- **DSL wire format.** The LLM emits rules in a structured JSON format
  (`kind`, `guard`, `effect`, `support`). `validate_proposal` checks entity IDs
  against the live scene, parses guard clauses through `guard_parse.py`, and
  converts via `dsl_to_rule`. Invalid proposals silently return `None`.
- **No separate rule hypothesis phase.** The proposer reuses the same engine
  lifecycle. LLM proposals enter the rule store as `proposed`, then get confirmed
  or pruned by the same residual-driven engine that handles classical proposals.

### Phase-1 wrap-up — `summary()` boundary contract (done)

Code: `perception/objects.py` (`frame_stack`, `n_subframes`, settled `to_grid`),
`perception/roles.py` (`detect_counter`), `perception/session/snapshot.py`
(`SceneSnapshot.summary()`, `StepObservation`), `tests/unit/test_perception_contract.py`,
`tests/reference_recordings.json` (ls20 + g50t).

**Boundary rule:** perception emits observations and events; it never predicts
and never assigns game semantics. Downstream EffectModel and LLM planners consume
`SceneSnapshot.summary()` — a JSON-serializable dict with entities, events
(animation, delta, registry), globals (counters), and a determinism beacon
(same settled state + action → different outcome = non-Markovian handoff).

**g50t validation** (`recordings/g50t-5849a774.curiosity.200.*.recording.jsonl`):

1. **API frames are temporal animation stacks, not spatial layers.** Sub-frame
   count varies per step (1–45); `last_subframe(t) == first_subframe(t+1)`. The
   settled post-action state is the **last** sub-frame; `to_grid` now defaults
   to it. In-frame sub-frames are replays (action 5 = memory playback).
2. **Controllable player + sequence-memory overlay.** The player is a compound
   `{color-9 ring + color-5 dot}` (entity #0) that translates ±6 cells:
   action 1=up, 2=down, 3=left, 4=right. Action 5 (spacebar) replays the move
   history as a ghost and resets the player to the top — so the *same* settled
   state + action can yield *different* outcomes, which the determinism beacon
   flags as non-Markovian (hidden memory is EffectModel/LLM scope). A bottom-row
   tally bar (color 1) grows monotonically — `detect_counter` fires.
3. **Two detector bugs fixed by g50t (would have hidden the player):**
   - `detect_controllable` required *every* entity member to independently pass
     the agreement test (`members <= controllable`); the player's co-moving
     color-5 dot fails the threshold (action-5 reset noise), so the whole entity
     was discarded. Fixed: an entity is controllable when it *contains*
     controllable track(s) and no structural member (co-moving members belong to
     the compound).
   - `_controllable_tracks` mapped *every* action's dominant displacement,
     letting RESET (0) and the noisy replay (5) pollute `motion_by_action`. Fixed:
     skip RESET and keep only actions whose per-action agreement clears the
     threshold, so the map is the clean `{1:up, 2:down, 3:left, 4:right}`.
4. **Second game in reference manifest (C1).** ls20 controllable + movement tests
   still pass; g50t contract tests assert controllable #0, counter detected,
   animation events, and non-Markovian beacon.

## 6. Open questions / next steps

- [x] Rung 2.5: persistent object registry + derived roles/entities.
- [x] Rung 2: frame-delta + common-fate clustering on labeled recording.
- [x] Rung 3: entity layer + controllable detection (v1 heuristic).
- [x] Rung 5 (v1): partial-state snapshot + empirical movement model + BFS +
      recording-based verification loop.
- [x] Rung 6 (v1): curiosity-driven live agent — random cold start → controllable
      detection → BFS toward unknown → per-step verify/replan.
- [x] Live planner agent: execute plan → re-snapshot → detect divergence → replan
      (`ExplorationPolicy`; absorbs new blocks into the movement model).
- [x] Split perception session from planner (`perception/session/`, `planning/`).
- [x] Degenerate-frame guard (in registry).
- [x] Merge multi-colour movers into one entity (compound via common fate).
- [x] Refine `derive_roles` mover criterion (fraction-of-life, exclude structural).
- [ ] Container entity grouping (cell-containment, skip oversized outers) for
      key/door compounds — deferred from v1 `build_entities`.
- [ ] **Key↔door transform-invariant matcher**: canonicalize compound shapes
      (extract inner pattern, normalize scale/rotation) to detect the goal relation.
- [ ] **Floor-aware background**: model per-region background so appeared/vanished
      become meaningful (needed for pickups/doors, not just movement).
- [x] Additional role detectors + richer `effects.predict` (terminal + counter rules).
- [x] Curiosity v2: confirm/refute rules by *consequence* (Markovian residuals via
      `effects/engine.py`; wired in `ExplorationPolicy`, optional `log_engine`).
- [x] Effects slice 3: rule engine (propose / confirm / prune on visible dims;
      abstain + flag on non-Markovian). See `scripts/run_effect_engine.py`.
- [x] Effects slice 4: LLM planner + rule proposer + query interface + ProbeGoal DSL;
      classical verify loop (`docs/brainstorms/llm-agent-loop.md`). LLM rule proposer
      ships with same propose/confirm/prune lifecycle as classical. Eval path uses
      `NULL_RULE_PROPOSER`. Closes: random → kinematics → directed probes → hypothesis
      rules → engine confirm.
- [x] Add non-ls20 entries to `tests/reference_recordings.json` (C1: g50t).
- [x] Multi-sub-frame API frames: temporal animation stacks; use last sub-frame
      as settled state (`to_grid`, `n_subframes`, animation events in `summary()`).
- [x] Phase-1 perception boundary: `SceneSnapshot.summary()` contract + counter
      detector + determinism beacon.

> The predictive layer that consumes this perception output is scoped in
> `docs/brainstorms/effect-model.md` (`effects/` package).

## 7. Artifacts

- Code: `effects/` (incl. `residual.py`, `engine.py`, `engine_log.py`), `planning/`,
   `perception/objects.py`, `perception/motion.py`, `perception/registry.py`,
   `perception/entities.py`, `perception/roles.py`, `perception/session/`,
   `perception/viz.py`, `agents/templates/curiosity_agent.py`,
   `agents/templates/llm_curiosity_agent.py`,
   `planning/llm_rule_proposer.py`, `planning/llm_planner.py`,
  `scripts/perceive_recording.py`, `scripts/analyze_motion.py`,
  `scripts/track_recording.py`, `scripts/plan_recording.py`,
  `scripts/run_effect_engine.py`,
  `tests/reference_recordings.json`, `tests/unit/test_planning.py`,
  `tests/unit/test_exploration.py`, `tests/unit/test_effects.py`,
  `tests/unit/test_effects_engine.py`, `tests/unit/test_perception_contract.py`,
  recording fix in `agents/agent.py`
- Reference recordings: `recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl`,
  `recordings/g50t-5849a774.curiosity.200.*.recording.jsonl`
- Sample outputs: `perception_out/frame_*.png`, `motion_out/motion_*.png`,
  `track_out/track_*.png`
- Related: `docs/diary/2026-06-09.md` (background research, leaderboard notes)
