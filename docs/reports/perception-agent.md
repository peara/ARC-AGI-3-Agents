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
   steps to merge/split candidates. ⬜ next
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

## 6. Open questions / next steps

- [ ] Rung 2: implement frame-delta extraction + common-fate clustering on the
      new labeled recording.
- [ ] Rung 3: auto-identify controllable object via ACTION1–4 motion correlation.
- [ ] Decide object identity/tracking across frames (use `shape_key` + centroid
      nearest-match).
- [ ] How to merge multi-layer frames when games have >1 layer.
- [ ] Validate perception on a non-ls20 game to test game-agnosticism (C1).

## 7. Artifacts

- Code: `perception/`, `scripts/perceive_recording.py`, recording fix in `agents/agent.py`
- Reference recording: `recordings/ls20-9607627b.random.80.4778fe67-*.recording.jsonl`
- Sample outputs: `perception_out/frame_*.png`
- Related: `docs/diary/2026-06-09.md` (background research, leaderboard notes)
