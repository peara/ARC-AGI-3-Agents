# Collision rules — solidity and overlap validation

> Brainstorm for extending the rule system with collision/solidity so
> `predict()` can reject moves into walls without hardcoding grid assumptions.

## Problem

Current `predict_move` only checks `known_blocks` (positions the player has
**actually visited and failed at**). For any unvisited position, it blindly
applies `motion_by_action` — walking through walls.

Empirical test on ls20 (frame 40):
- Symbolic BFS (rules only): 8 entities "reachable" — **6 false positives**
- Grid BFS (player removed, bg=walkable): 2 entities reachable — correct

The symbolic rules have no collision model. `known_blocks` grows one entry at
a time as the player hits walls, but can't generalize to unvisited positions.

## Goal

Add a **validate** step to `predict()`:

```
predict(state, action, ctx) -> SceneState | None

  state' = kinematics(state, action, ctx.movement)   # extrapolate
  if state' is None: return None

  → validate(state', rules)                           # NEW
  → if invalid: return old_state (blocked)
  → if valid: apply terminal + relational rules, return state'

  (rest of pipeline unchanged)
```

No assumptions about walls or floors baked in. The rules determine what blocks.

## Rule form: entity-pair collision

```json
{
  "kind": "collision",
  "guard": {"overlaps": {"entity_a": 0, "entity_b": 5}},
  "effect": {"dim": "valid", "op": "set", "value": false}
}
```

"A rule that says: if entity A's cells overlap with entity B's cells, the state is invalid."

### Why entity-pair, not entity-solidity or color-based

| Approach | Example | Pro | Con |
|---|---|---|---|
| Entity solidity | "entity 5 is solid" | Simple, one rule per wall | Assumes a single "blocker" role; can't express "entity 3 blocks entity 5 but not entity 7" |
| Color-based | "color 5 blocks movement" | Very compact (~9 rules) | Needs raw grid in predict; colors get overwritten on overlap |
| **Entity-pair collision** | "entity 0 can't overlap entity 5" | General — any pair, any direction; no player assumption | More rules (N² worst case) |

Entity-pair is the most general. It handles:
- Player blocked by wall: `{entity_a: 0, entity_b: 5}`
- NPC blocked by wall: `{entity_a: 3, entity_b: 5}`
- Two objects can't overlap: `{entity_a: 3, entity_b: 7}`
- Player can walk through key-forger: **no rule proposed** → allowed

### The generality question

For unknown games, generality is always better. We don't know:
- Which entity is the player (could be 2 players, or none)
- Which entities are walls vs walkable items
- Whether collisions are symmetric (A blocks B but B doesn't block A?)

Entity-pair collision rules let the LLM discover these from observation,
without us baking in assumptions.

### Can the LLM handle it?

The LLM sees:
- Entity bboxes (size, position)
- Entity roles (structure, counter, controllable, mover)
- Movement blocks (known_blocks — where the player was blocked)

From this it can reason:
- "Entity 4 is 42×40 and static → probably a wall → propose collision with player"
- "Entity 1 is 2×2 and the player walked through it → no collision rule"
- "The player was blocked at (27,51) going up, and entity 5 is at that position → entity 5 blocks the player"

The propose/confirm/prune lifecycle handles uncertainty:
1. **Propose**: LLM sees a large static entity → proposes collision rule
2. **Confirm**: player tries to move into entity, is blocked → support increases
3. **Prune**: player walks through entity → collision rule removed

## Validation mechanics

### What predict needs

To validate, `predict()` needs to check cell overlap between entities at the
candidate position. This requires:

1. **Entity cells** — already tracked in `Observation.cells` (per-track, per-frame)
2. **Entity footprint at candidate position** — compute from bbox + motion delta
3. **Collision rules** — list of entity-pair rules in `EffectContext`

### The overlap check

Given candidate state `state'` where entity 0 moved to new position:

1. Compute entity 0's cells at the new position (translate by delta)
2. For each collision rule `{entity_a: 0, entity_b: N}`:
   - Get entity N's cells at current frame
   - Check if any cells overlap
   - If overlap → state is invalid → return old state

### Cell translation

The player's cells at the new position = old cells + motion delta.
We don't need to re-segment the grid — just translate the known cell set.

```python
def translate_cells(cells: frozenset[Pos], delta: Pos) -> frozenset[Pos]:
    dr, dc = delta
    return frozenset((r + dr, c + dc) for r, c in cells)
```

### Occlusion

When the player walks over the key-forger:
- Grid shows player's color replacing key-forger's cells
- But entity tracking persists — we know key-forger is still there
- No collision rule → validation passes → move accepted
- On next frame, perception may show key-forger "disappeared" → residual
- LLM proposes a rule to explain (e.g., "overlap → entity disappears" = exists rule)

This is the natural separation: collision rules say what blocks, exists rules
say what happens on overlap. Both are rule-driven, not hardcoded.

## EffectContext changes

```python
@dataclass(frozen=True)
class EffectContext:
    movement: MovementModel
    terminal_rules: tuple[Rule, ...] = ()
    relational_rules: tuple[Rule, ...] = ()
    proposed_rules: tuple[Rule, ...] = ()
    collision_rules: tuple[Rule, ...] = ()     # NEW
    non_markovian: bool = False
    confirm_threshold: int = 2
    # entity_cells: frozenset[Pos] per entity at current frame
    # (passed in from perception, not stored on context)
```

Or: collision rules are just another kind in the existing `relational_rules`
list, with `kind="collision"` and `effect.dim="valid"`. No new tuple needed.

## predict() changes

```python
def predict(state, action, ctx, entity_cells):
    """entity_cells: dict[int, frozenset[Pos]] — current cells per entity."""
    
    # 1. Kinematics (unchanged)
    state' = kinematics(state, action, ctx.movement)
    if state' is None:
        return None
    
    # 2. Validate (NEW)
    if not validate(state', state, action, ctx, entity_cells):
        return state  # blocked — return old state
    
    # 3. Terminal rules (unchanged)
    for rule in ctx.terminal_rules:
        if rule.guard(state', action):
            state' = rule.apply(state', action)
    
    # 4. Relational rules (unchanged)
    for rule in ctx.relational_rules:
        if rule.guard(state', action):
            state' = rule.apply(state', action)
    
    return state'
```

The validate function:

```python
def validate(state', state, action, ctx, entity_cells) -> bool:
    """Check collision rules against candidate state."""
    for rule in ctx.collision_rules:
        if not rule.guard(state', action):
            continue
        # Rule says: entity_a can't overlap entity_b
        a_id = rule.guard_spec["overlaps"]["entity_a"]
        b_id = rule.guard_spec["overlaps"]["entity_b"]
        
        # Get cells for entity_a at new position
        old_pos = state.pos(a_id)
        new_pos = state'.pos(a_id)
        delta = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        a_cells = translate_cells(entity_cells[a_id], delta)
        b_cells = entity_cells[b_id]
        
        if a_cells & b_cells:  # overlap
            return False  # invalid
    
    return True
```

## Rule proposer prompt changes

Add to the system prompt:

```
## Collision rules

You can propose that two entities cannot overlap:

{"kind": "collision", "guard": {"overlaps": {"entity_a": 0, "entity_b": 5}}, "effect": {"dim": "valid", "op": "set", "value": false}}

This means: when entity 0's cells overlap with entity 5's cells, the state is
invalid (movement is blocked).

Propose collision rules when:
- A large entity appears to block movement (walls, borders)
- The player was blocked at a position near an entity (check known_blocks)
- An entity has role "structure" and is large

Do NOT propose collision rules for:
- Small entities the player might walk through (items, keys)
- Entities that change size (counters, HUD elements)
- The controllable entity itself
```

## Open questions

1. **Symmetric vs asymmetric**: If entity A can't overlap entity B, can B overlap A?
   - For walls: yes (symmetric)
   - For items: maybe not (player picks up item, doesn't collide)
   - Default: make rules directional; LLM proposes both directions if needed

2. **Static vs dynamic cells**: Entity cells change over time (objects move).
   - `entity_cells` should be from the current frame (perception provides this)
   - For BFS: cells are from the BFS start frame; moving entities' cells are
     predicted by their own movement model (if any)
   - For now: assume cells are static during BFS (walls don't move)
   - Later: extend predict to also predict entity movements

3. **Performance**: N² pairwise checks per BFS node.
   - Prune by bbox distance (skip pairs whose bboxes don't overlap)
   - Most entities don't have collision rules (only proposed ones)
   - Typical: 2-5 collision rules, not N²

4. **Confirmation**: How to confirm a collision rule?
   - Player tries to move into entity, is blocked → support +1
   - Player walks through entity → prune
   - But we only know the player was blocked from `known_blocks` (position + action)
   - Need to correlate: which entity was at the blocked position?

5. **LLM capacity**: Can the LLM propose collision rules effectively?
   - It needs to reason: "entity 4 is large, at this position, and the player
     was blocked here → entity 4 blocks the player"
   - The scene bundle now includes bbox, so the LLM can see entity extent
   - `known_blocks` is in the movement model (also in the bundle)
   - This is a reasonable reasoning task for the LLM

## Implementation steps

1. **Add collision kind to DSL** — `effects/dsl.py` and `effects/rules.py`
2. **Add validate step to predict** — `effects/predict.py`
3. **Pass entity_cells through** — from `SceneSnapshot` to `predict()`
4. **Update rule proposer prompt** — `planning/llm_rule_proposer.py`
5. **Update validate_proposal** — accept collision kind
6. **Wire into BFS** — `planning/search.py` passes entity_cells
7. **Test on ls20** — verify player can't walk through walls, can walk through key-forger