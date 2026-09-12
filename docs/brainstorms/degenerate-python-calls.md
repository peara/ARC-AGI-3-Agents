# Degenerate `# previous code omitted` calls — close the last-mile failure

> Brainstorm for eliminating the model's comment-only python calls: it emits
> `python("# previous code omitted")` instead of real code, burning a 55s LLM
> round-trip for `(no output)`. Observed in both the incident run (5681a14a,
> killed at the final call) and the follow-up run (bf6acc0c: 7 occurrences,
> chains of 2–3). Proposed: cheap harness-side detection + nudge, and an
> EXECUTE-phase scaffold. Motivated by run
> `bf6acc0c` (2026-09-11) — see §2 for the evidence.

## 1. The failure mode

The model ends a turn-internal reasoning chain with a one-line python call that
is *only a comment*:

```python
"# previous code omitted"
```

Properties:

- **Valid tool call, zero content.** The sandbox executes it, gets `(no output)`,
  appends the tool result, and the loop calls the LLM again — a full 40–160s
  round-trip spent on nothing.
- **It is a lazy summarization habit.** The model "remembers" that relevant code
  exists in the conversation, treats it as elided by context trimming, and
  re-references it instead of rewriting it. Two aggravators:
  1. **Trimming really does eat prior code** (`keep_last_n=3` for tool results),
     so the model's laziness is sometimes *justified* from its point of view —
     the code it wants to reuse is no longer in the transcript.
  2. **The namespace survives even when the transcript doesn't.** Sandbox state
     (`path`, helper functions) is still callable — the model under-uses this.
- **It chains.** A degenerate call returns `(no output)`; the next call
  references "previous code" again; chains of 2–3 are common (bf6acc0c f44:
  3 in a row). The loop only stops the chain when some other mechanism breaks
  the pattern (e.g. the model finally prints something).

## 2. Evidence

**Incident run 5681a14a** (2026-09-10): the final call before the process was
killed was exactly this — `python("# previous code omitted")`, 34 chars, after
`set_phase('EXECUTE')` had succeeded with a validated 7-action path in hand.
The path existed in the *persistent sandbox namespace* (`path` variable from
its own BFS two calls earlier); the model never wrote `for a in path: action(a)`.

**Follow-up run bf6acc0c** (2026-09-11, after the sim-state block landed):
7 degenerate calls across 195, in chains:

| Frame | Chain | Preceding call | What it wanted |
|---|---|---|---|
| 6 | ×2 | `python: # Look at history...` (itself a stub) | re-run earlier explore code |
| 44 | ×3 | `python: def player_tl(grid): ...` (real locator) | reuse the player-locator code |
| 63 | ×2 | `python: # locate white plus...` (stub) | re-run BFS/locator |

Pattern: the degenerate call follows **real code it just wrote within the same
turn** — i.e. the code is often *still in recent context* (untrimmed) — yet the
model prefers referencing over repeating. At f44 the three chained calls each
cost ~90s: ~4.5 min of pure waste in one turn. It recovers only when it
abandons reference-mode and writes fresh code (`print("hello", len(current_frame))`
broke the chain).

## 3. Why current mitigations don't catch it

- **The SpiralGuard counts non-action calls**, but a degenerate call counts as
  one ordinary tool call — indistinguishable from a real one in the counter.
  Chains of 2–3 never approach the cap.
- **The nudge after non-action** says "call action()" — wrong medicine when the
  model is mid-reconstruction; it nudges *toward acting*, not *toward writing
  real code*.
- **The sim-state block** (shipped) made sandbox state visible, which helps the
  model *know* what exists — but does not stop the summarization reflex when
  the model wants to re-run its own recent code. Visibility ≠ anti-laziness.
- **No output ≠ failure.** The loop treats `(no output)` as a normal tool
  result. There is no signal that a call produced literally nothing.

## 4. Proposed solution

Three parts, ordered by cost. Parts A+B are cheap, game-agnostic, and
testable offline; C is a prompt addition only.

### A. Detect comment-only calls and say so in the tool result

In `_handle_python_call` (or the sandbox's `run_code` wrapper): if the code
contains **no executable statements** (comment-only / blank lines), short-circuit:

- Skip execution (no worker dispatch — saves the sandbox exec too).
- Tool result: `[No executable code — comment-only call. Rewrite the code you
  intended to run; references to "previous code" are not visible to you.
  The sandbox namespace persists: variables and helpers from earlier calls
  are still available.]`

Rationale: the model must be told, at the point of failure, that (a) its call
was empty and (b) the reference is unusable. One message kills the whole chain
— the model can't repeat "previous code omitted" when the result says the
reference is void.

Detection detail: comment-only code parses to `ast.Module(body=[])` — comments
never reach the AST. So the check is: `ast.parse(code)` → body empty =
degenerate. One line, no false positives (code with any statement has a
non-empty body).

### B. Count degenerate calls in the SpiralGuard

A degenerate call increments a per-turn `degenerate_call_count`. At ≥2
consecutive degenerate calls, append to the tool result:
`[Second consecutive empty call — you are in a reference loop. Write the
actual code now, or call action() if you have a path.]`. This gives the
guardrail visibility without changing its escape semantics (still no
MODEL-phase forcing — that mechanism is orthogonal).

### C. EXECUTE-phase scaffold (the deferred idea, now justified)

When the workflow accepts `set_phase('EXECUTE')` and the sandbox holds a BFS
path, inject one user message after the tool result:

```
EXECUTE: your registered bfs() path is available in the namespace as `path`
(from your last bfs() call). Your next python call should be:
    for a in path: action(a)
```

This directly targets the terminal failure of 5681a14a (path in namespace,
model emitted a placeholder instead of executing). Implementation: the hook
that refreshes the sim-state block can co-locate this — but keep it a
separate user message, not part of the sim-state block (different lifetime:
state is persistent, the EXECUTE scaffold is one-shot per set_phase).

Guardrails:
- Only when a path actually exists (sandbox `_last_bfs_result` non-empty) and
  phase is EXECUTE.
- One-shot: cleared when an `action()` is observed or phase changes.
- No path content duplication if the model already acted this turn.

## 5. Test sketch

- Comment-only call → tool result contains the rewrite nudge; sandbox not
  exec'd (no side effects); chain-breaking message at 2nd consecutive call.
- AST check unit tests: `# only comment` → degenerate; `x = 1` → not;
  `print("x")` → not; empty string → degenerate (also caught).
- EXECUTE scaffold: set_phase('EXECUTE') with `_last_bfs_result` → injected
  message present exactly once; cleared after action.
- Incident-anchored: bf6acc0c f44 chain (3 degenerate calls) → with fix, chain
  length ≤ 1.

## 6. Non-goals

- Not changing the SpiralGuard's non-action counter (orthogonal mechanism).
- Not caching/model-side summarization control (can't change the model).
- Not blocking comment-only calls outright — they're valid python; we only
  respond to them. If a future model emits meaningful comment-then-code in
  one call, the AST check (executable statements present?) does not trigger.

## 7. Success metrics (next live run)

- Zero degenerate chains of length ≥ 2 (bf6acc0c: two chains ≥ 2).
- Total comment-only calls ≤ 1 per 100 frames (bf6acc0c: 7/100).
- No regression in set_simulate count, check accuracy trajectory, or actions
  per level.