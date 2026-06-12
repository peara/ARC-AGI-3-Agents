# Perception-First Agent — Research Report

> A living design document (not a dated diary entry). Captures our conjectures,
> the reasoning behind them, what we've built, and what we found, so we can
> refer back and revise as evidence comes in.
>
> Last updated: 2026-06-11

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
3. **Controllable-object identification** — correlate `ACTION1`–`ACTION4` with
   object motion to tag "the agent". ⬜
4. **EffectModel + roles** — running action→effect statistics per object;
   classify wall / pickup / hazard / door by *consequence*. ⬜
5. **Active disambiguation + planning** — information-gain probes, then
   BFS/greedy on the abstracted state. Optional learned forward model here. ⬜

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
PYTHONPATH=. python3 scripts/perceive_recording.py \
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
PYTHONPATH=. python3 scripts/analyze_motion.py \
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
PYTHONPATH=. python3 scripts/track_recording.py \
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

## 6. Open questions / next steps

- [x] Rung 2.5: persistent object registry + derived roles/entities.
- [ ] Refine `derive_roles` mover criterion (fraction-of-life / action-correlated,
      exclude structural) so the floor stops reading as a mover.
- [ ] Refine `derive_entities` containment (cell-containment, skip oversized
      outers) to drop floor-bbox noise.
- [x] Rung 2: frame-delta + common-fate clustering on labeled recording.
- [ ] **Persistent object registry** (next): stable IDs across an episode via
      multiple cues (displacement / positional overlap / persistence) + per-object
      property trajectories. Lets roles emerge from data; resolves "are these two
      blobs the same object across frames?".
- [ ] **Degenerate-frame guard**: skip/flag frames with `n_unique==1` or
      near-total delta so flashes/transitions don't corrupt tracking.
- [ ] **Key↔door transform-invariant matcher**: canonicalize compound shapes
      (extract inner pattern, normalize scale/rotation) to detect the goal relation.
- [ ] **Floor-aware background**: model per-region background so appeared/vanished
      become meaningful (needed for pickups/doors, not just movement).
- [ ] Rung 3: promote common-fate result to an explicit "controllable object"
      tag using the action→displacement consistency (already 1.0 agreement here).
- [ ] Merge multi-colour movers into one tracked entity (compound-object id).
- [ ] How to merge multi-layer frames when games have >1 layer.
- [ ] Validate perception on a non-ls20 game to test game-agnosticism (C1).

## 7. Artifacts

- Code: `perception/objects.py`, `perception/motion.py`, `perception/registry.py`,
  `perception/viz.py`, `scripts/perceive_recording.py`, `scripts/analyze_motion.py`,
  `scripts/track_recording.py`, recording fix in `agents/agent.py`
- Reference recording: `recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl`
- Sample outputs: `perception_out/frame_*.png`, `motion_out/motion_*.png`,
  `track_out/track_*.png`
- Related: `docs/diary/2026-06-09.md` (background research, leaderboard notes)
