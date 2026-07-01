# Rule engine v2 — stable identity, richer rules, proactive lifecycle

> Status: **direction**. Builds on `rule-first-v2.md` (implemented), `unified-rules.md`
> (implemented), and `rules-only-predict.md` (implemented). This doc defines the
> next round of improvements motivated by the first v2 recording analysis.

## Evidence

Recording `wa30-ee6fef47.llmcuriosityv2.15d2eb33-b63b-487a-bce8-dd4a6b453bb2.recording.jsonl`
(61 frames, `RuleFirstPolicy`, `policy_version=v2`):

1. **Entity IDs are not stable.** Entity 0 starts as the black hole (color 0,
   4 cells), then becomes the central static block (track 1, size 20), then
   becomes the compound avatar. Entity 10 starts as the red block (color 14,
   12 cells), then becomes a fragment, then becomes the right-side counter
   (static). The same ID maps to different physical objects across the game.

2. **The compound never forms.** The black hole and red block are always
   adjacent and co-moving, but the `EntityBuilder` only compounds same-color
   tracks that merge/split. It never groups adjacent different-color objects.
   So we get two parallel rule sets (for entity 0 and entity 10) for one
   physical avatar, and they become stale when IDs swap.

3. **Rules only model `pos` and `size`.** The avatar has orientation (the black
   hole faces a direction relative to the red block). Moving and rotating are
   independent — some actions rotate without moving. The current rule engine
   is blind to orientation: a rotation-only transition produces no `pos` or
   `size` residual, so no rule is proposed.

4. **Rule confirmation is slow and reactive.** A rule needs `support >= 2`
   matching transitions, accumulated one frame at a time. There's no
   retroactive testing against stored history, no active confirmation, no
   stale-rule cleanup. 114 proposed rules remain unconfirmed at the end of the
   run — many are noise from entity fragmentation, but some are legitimate
   rules that just didn't accumulate enough support.

## Proposal

Four workstreams, ordered by dependency:

### 1. Stable entity identity — IMPLEMENTED (2026-06-30)

**Problem**: Entity IDs are reassigned. Rules learned for "entity 0 = black
hole" are later applied to "entity 0 = central static block."

**Root cause**: The `EntityCatalog` assigns IDs from a fresh slot pool each
frame. When a track dies, the slot is recycled. The `LogicalRegistry` +
`reconciler` links dead tracks to born tracks, but the entity ID doesn't
follow the link — the new track gets a fresh ID.

**Implemented design**:

- **Monotonic ID counter.** New entities always get a fresh ID from a
  counter (`_next_entity_id`) that never decreases. IDs are never recycled.
  No `retired_ids` set needed — the monotonic counter alone prevents reuse.

- **Entity lifecycle states:**

  | State | Meaning | Rules |
  |-------|---------|-------|
  | `active` | Object exists as a singleton in the current catalog | Learned normally |
  | `merged` | Object is a member of a compound | Sub-track rules frozen; compound gets its own rules |
  | `dormant` | Track vanished, no successor linked yet | Rules kept but inactive; reactivated if reconciler links a new track |
  | `dead` | TTL exceeded with no successor | Rules pruned |

- **Reconciler-driven inheritance.** `EntityBuilder` consumes the
  reconciler's merge map (`dead_track → born_track`) and maps the new track
  to the old entity's ID via `_track_to_entity`. The link exists in
  `reconciler.py` (unchanged); the ID assignment now follows it in
  `entity/builder.py`.

- **Compound ID stability.** `_compound_members` stores entity IDs (not
  track IDs) so compound membership is stable across reconciler track
  rotations. `_merge_into_compound(reuse_id=True)` reuses the compound
  entity ID when the member set is unchanged across frames. `_common_fate_groups`
  was removed from `build_entities` — compound creation is now entirely via
  `co_movement` in `EntityBuilder._apply_compound_grouping`.

**What this fixes**: Rules for "entity 5 = black hole" persist even when the
black hole merges into a compound. If the compound later splits, the rules
reactivate. Entity 5 is never accidentally reassigned to the central static
block.

**Actual scope** (differs from original plan): `entity/builder.py` (all
stable-ID logic), `perception/entities.py` (`build_entities` stripped to
singletons-only, `_common_fate_groups` removed). `entity/reconciler.py` and
`entity/logical_registry.py` were NOT modified — the reconciler already
produced the right merge map; only the consumer needed changing.

**Verification** (2026-07-01): compound IDs stable across frames (id=13 held
24 frames, id=15 held 9 frames); only ID change was legitimate membership
expansion. Zero spurious `CONTROLLABLE ID CHANGED` warnings. 428 tests pass.

**Resolved open questions**:
- ~~How large can the ID space grow?~~ Monotonic counter means IDs grow
  linearly with entity births. For 64×64 grids with sparse objects, this is
  bounded and not a practical concern. Dormant TTL=3 ensures dead entities
  are pruned promptly.
- Should `dormant` rules be visible to the LLM proposer? Still open —
  deferred to workstream 3.

---

### 2. Orientation dimension

**Problem**: Rules only model `pos` and `size`. The avatar in wa30 has
orientation (the black hole faces a direction relative to the red block).
Rotations are invisible to the rule engine.

**Design**:

- **New `SceneState` dimension: `orientation`.**

  ```
  entity_id -> {
      "pos": (row, col),
      "size": int,
      "orientation": int  # 0=N, 1=E, 2=S, 3=W (or game-specific enum)
  }
  ```

- **Perception: extract orientation from compounds.** For a compound entity,
  compute orientation from the relative offset of members. E.g., if member A
  (black hole) is north of member B (red block), orientation = 0 (facing
  north). This is a new function in the perception/entity layer — one that
  computes the "facing" of a composite from its parts' relative positions.

  For singletons without inherent facing, `orientation` is `None` and not
  tracked.

- **Rule effects:**

  ```python
  Effect(dim="orientation", of=entity_id, op="set", value=0)   # face north
  Effect(dim="orientation", of=entity_id, op="delta", value=1)  # rotate 90° CW
  ```

  A single rule can have multiple effects:

  ```python
  Rule(
      guard_spec={"action": 1},
      effects=(
          Effect("pos", compound_id, "delta", (0, -4)),      # move up 4
          Effect("orientation", compound_id, "set", 0),       # face north
      ),
      kind="movement",
  )
  ```

  The `Effect` dataclass already supports arbitrary `dim` strings — no schema
  change needed. The `Rule.apply()` method handles `delta` and `set` for any
  dim. The new work is in `SceneState` (store orientation) and `predict()`
  (apply orientation effects).

- **Classical learner: `learn_orientation_rules`.** Analogous to
  `learn_movement_rules` — for each compound entity, check if orientation
  changed between frames and correlate with actions. Emit rules like
  `guard={action: N} → orientation set V`.

- **LLM proposer: include orientation in the bundle** (see workstream 3).

**What this fixes**: The engine can distinguish "move up" from "move up +
rotate" — they're different rules with different effects. Rotation-only
transitions (which currently produce no residual) become visible and
learnable.

**Scope**: `effects/state.py` (SceneState), `effects/predict.py`,
`effects/learn.py` (new `learn_orientation_rules`), `perception/` (new
`extract_orientation`), `effects/residual.py` (track orientation in
residuals).

**Open questions**:
- Orientation encoding: is 4-directional (N/E/S/W) sufficient, or do some
  games have 8-directional or continuous orientation? Start with 4-dir and
  generalize if needed.
- Should orientation be a property of the compound only, or also of
  singletons? Start with compound-only.
- How to compute orientation for compounds with >2 members or complex
  shapes? Need a heuristic (e.g., "head" = smallest member, "body" = largest,
  orientation = direction from body to head).

---

### 3. LLM proposer context — show the full picture

**Problem**: The LLM rule proposer sees entities as independent objects with
positions and sizes. It doesn't know that entity 0 and entity 10 are parts of
the same avatar, or that the avatar has a facing direction.

**Design**: Enrich `QueryInterface.bundle()` with four new sections:

**a) Group/composition info**

```
COMPOUND ENTITIES:
  Entity 20 (compound, members=[5, 12]):
    Member 5: color=0 (black), size=4, offset=(-3, 0) from centroid  # "head"
    Member 12: color=14 (red), size=12, offset=(+1, 0) from centroid  # "body"
    Orientation: north (head is above body)
    Position: [48, 34]
    Movement rules: action 1 → pos delta [0,-4], orientation set north
```

This requires the grouping engine (`GroupingEngine`) to be active and
connected to the policy. If `RuleFirstPolicy` isn't wired to grouping, that's
a prerequisite.

**b) Co-movement evidence**

```
CO-MOVEMENT OBSERVED:
  Entities [5, 12] have moved together on 8/8 transitions
  They maintain a constant relative offset
  Hypothesis: they form one object with a body (entity 12) and head (entity 5)
```

This is computed from the transition history (see workstream 4a). For each
pair of entities, count how many transitions they co-moved (same delta) and
whether their relative offset is constant.

**c) Rule coverage gaps**

```
RULE GAPS:
  Entity 20 has movement rules for actions 1,2,3,4
  Entity 20 has NO orientation rules — 3 transitions showed orientation
  change without position change
  These transitions are unexplained by current rules
```

This surfaces "what the engine can't explain" to the LLM, guiding it toward
the missing rules.

**d) Recent transition history** (for retroactive inference)

```
RECENT TRANSITIONS (last 5):
  Frame 15: action=3, entity 20 pos [54,0]→[54,0] (no move), orientation N→W
  Frame 16: action=1, entity 20 pos [54,0]→[54,7] (moved down), orientation W→W
```

The LLM can propose rules like "action 3 rotates the avatar 90° clockwise"
because it can see that action 3 sometimes changes orientation without
changing position.

**Bundle size control**: Keep the existing caps (`unknowns[:5]`,
`proposed_rules[:20]`). Add caps for the new sections: top N=3 compound
entities, last K=5 transitions, top M=5 co-movement pairs. This keeps the
bundle under the existing token budget.

**Scope**: `planning/query.py` (QueryInterface.bundle),
`grouping/engine.py` (connect to policy), `planning/rule_first.py` (wire
GroupingEngine).

**Open questions**:
- Should the LLM see `dormant` rules? Probably yes, labeled as
  "hypothesized but inactive" — the LLM can reason about reactivation.
- How to present orientation changes compactly? Use directional letters
  (N/E/S/W) rather than integers in the bundle.

---

### 4. Automated rule lifecycle — proactive, not reactive

**Problem**: The engine is purely reactive — it proposes rules from residuals
and confirms them one frame at a time. No retroactive testing, no active
confirmation, no stale cleanup.

**Design**: Four mechanisms:

#### 4a. Retroactive testing on proposal

When a new rule is proposed (by the classical learner or the LLM), test it
against the stored transition history:

```python
def retroactive_test(rule, history: list[Transition]) -> int:
    """Test rule against all stored transitions. Return support count."""
    support = 0
    for t in history:
        if rule.guard(t.state_before, t.action):
            predicted = rule.apply(t.state_before, t.action)
            if matches(predicted, t.state_observed, rule.effects):
                support += 1
    return support
```

If a rule matches 5 historical transitions, it starts with `support=5` and is
confirmed immediately. This eliminates the slow frame-by-frame wait.

**Requirement**: Store `(state_before, action, state_observed)` tuples in a
ring buffer. Currently, `RuleFirstPolicy` only keeps the last one
(`_engine_state_before`). Add a `TransitionHistory` with a cap (e.g., 100
frames).

**Integration point**: `engine_step()` in `effects/engine.py` — after
`propose_rules()` adds a candidate, call `retroactive_test()` and set initial
support.

#### 4b. Active confirmation

When the planner picks an action, prefer actions that would confirm or refute
uncertain rules:

```python
def score_action(action, ctx, current_state):
    score = 0
    for rule in ctx.proposed_rules:
        if rule.support < ctx.confirm_threshold:
            if rule.guard(current_state, action):
                score += 1  # action triggers the rule → we learn something
    return score
```

This is a lightweight curiosity signal on top of the existing novelty-based
exploration. It doesn't replace the BFS planner but biases action selection
when multiple actions are equally novel.

**Integration point**: `RuleFirstPolicy._plan_toward_unknown()` — when BFS
returns multiple plans of equal length, pick the one whose first action has
the highest `score_action`.

#### 4c. Stale rule pruning

Rules for entities that haven't been seen recently should be deactivated:

```python
def prune_stale(ctx, active_entity_ids, frame_idx, ttl=15):
    """Move rules for dormant entities to dormant_rules bucket."""
    dormant = []
    active = []
    for rule in ctx.movement_rules:
        entity_ids = {eff.of for eff in rule.effects}
        if entity_ids & active_entity_ids:
            active.append(rule)
        else:
            dormant.append(rule)
    return replace(ctx, movement_rules=tuple(active), dormant_rules=tuple(dormant))
```

If a dormant entity reappears (reconciler links it), its rules move back to
active. This depends on workstream 1 (stable IDs + reconciler-driven
inheritance).

**Integration point**: `RuleFirstPolicy.on_observed()` — after updating the
catalog, call `prune_stale()` with the active entity set.

#### 4d. Rule conflict detection

If two confirmed rules have the same guard but different effects, one is
wrong:

```python
def detect_conflicts(ctx):
    by_guard = defaultdict(list)
    for rule in ctx.movement_rules + ctx.proposed_rules:
        by_guard[(rule.kind, canonical_guard(rule.guard_spec))].append(rule)
    conflicts = []
    for key, rules in by_guard.items():
        effect_sets = {tuple(r.effects) for r in rules}
        if len(effect_sets) > 1:
            conflicts.append(rules)
    return conflicts
```

Conflicts trigger immediate active testing — the agent should try the
conflicting action to see which rule wins. This integrates with 4b: conflict
actions get a high `score_action`.

**Integration point**: `engine_step()` — after `confirm_rules()`, run
`detect_conflicts()` and log conflicts. The policy reads them for action
scoring.

---

## Dependency graph

```
1. Entity ID stability (foundation) — DONE
   ├── Without this, rules are attached to wrong objects
   ├── Implemented in: EntityBuilder + build_entities (reconciler/logical_registry untouched)
   └── Unblocks: 2, 3, 4c

2. Orientation dimension (rule expressiveness)
   ├── Depends on: stable entity IDs (so orientation is per-object) ✓ unblocked
   ├── New perception function: extract_orientation(compound)
   └── New SceneState field + Effect dim

3. LLM bundle enrichment (proposer quality)
   ├── Depends on: stable IDs + orientation (so bundle is meaningful)
   ├── Requires: GroupingEngine connected to RuleFirstPolicy
   └── New QueryInterface section: groups + orientation + history

4. Rule lifecycle (engine quality)
   ├── 4a Retroactive testing: needs transition history buffer
   ├── 4b Active confirmation: needs planner integration
   ├── 4c Stale pruning: needs entity TTL tracking (depends on 1) ✓ unblocked
   └── 4d Conflict detection: needs rule indexing by guard
```

**Recommended order**: ~~1~~ → 4a (quick win) → 2 → 3 → 4b/4c/4d

Workstream 1 is complete. Next up is 4a (retroactive testing) — lowest-effort,
highest-impact. It doesn't require perception changes — just a transition
history buffer and a test function. It gives immediate value: faster rule
confirmation, less noise from unconfirmed proposals.

## Total effort (rough)

| Piece | New/Modified | Est. lines |
|-------|-------------|-----------|
| 1. Stable entity IDs | EntityBuilder, LogicalRegistry, reconciler, EntityCatalog | ~200 |
| 2. Orientation dimension | SceneState, predict, learn, perception | ~150 |
| 3. LLM bundle enrichment | QueryInterface, GroupingEngine wiring | ~100 |
| 4a. Retroactive testing | TransitionHistory, retroactive_test, engine_step | ~80 |
| 4b. Active confirmation | RuleFirstPolicy action scoring | ~50 |
| 4c. Stale pruning | EffectContext dormant_rules, prune_stale | ~60 |
| 4d. Conflict detection | detect_conflicts, policy integration | ~50 |
| **Total** | | **~690** |

## Open questions (left for observation)

1. **Compound orientation heuristic.** How to compute orientation for
   compounds with >2 members or complex shapes? Start with "head = smallest
   member, body = largest, orientation = direction from body to head."
   Other games may need different heuristics.

2. **Dormant rule reactivation.** When a dormant entity reappears, should its
   rules be immediately active or start with reduced support? Probably
   immediately active — the rules were confirmed before, and the physical
   object is the same.

3. **Transition history size.** 100 frames may be too small for long games or
   too large for short ones. Consider adaptive sizing based on game length,
   or a rolling window that prioritizes recent transitions.

4. **Active confirmation vs. exploration.** If the planner always picks
   actions that test uncertain rules, it may neglect novel-state exploration.
   Need a balance — maybe a weighted score: `novelty * 0.7 +
   rule_test_score * 0.3`.

5. **GroupingEngine connection.** Is `GroupingEngine` already wired to
   `RuleFirstPolicy`, or does it need integration? If not, that's a
   prerequisite for workstream 3.