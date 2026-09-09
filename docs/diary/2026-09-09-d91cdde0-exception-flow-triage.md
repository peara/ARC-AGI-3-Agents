# Exception-flow triage — d91cdde0 (ls20, 100 frames)

Date: 2026-09-09. Recording:
`recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst.d91cdde0-b45a-41b4-9da3-d50985102d94`
(sidecars `.logs.log`, `.llm.jsonl`). Run: 100 frames, level 1 won at frame 70,
3h40m wall-clock, 208 LLM calls.

`grep -c "exception_flow #"` on the `.logs.log` = **17**. The table below has
exactly 17 rows — one per workflow `exception_flow #N` line
(`agents.simulator_agent.workflow` channel, produced by
`workflow.py:211-218 on_exception_flow()`). The 3 sandbox WARNING lines
(`exception_flow: simulate crashed during predict-and-compare: ...`) carry no
`#` and are crash *evidence* for rows 41 and 49, not separate events.

## Cause classes

| Class | Meaning | Discriminator |
|---|---|---|
| `flash-caused` | The flow's diff is the budget-exhaustion board flash (6-layer animation stack, 5 uniform single-color layers + settled restored board). The live agent read layer 0 (all-yellow) as the board → phantom diff. | Flow frame ∈ {48, 92} (stack sweep), or the flow's diff payload is the uniform phantom board (4014 cells 11→base) on the frame right after a flash. |
| `highlight-transition` | The flow's diff is the 6-layer highlight overlay (5 identical base layers + highlighted last) or its one-frame landing lag — a real board change the simulate didn't model, but cosmetic (bar/box highlight), not a movement error. | Flow frame ∈ {19, 20, 21, 35, 38, 61} (stack sweep) or diff dominated by 5→0 / 0→5 highlight repaints. |
| `simulate-crash` | `predict_and_compare` caught an exception in the LLM's simulate (`error` key set in `_pending_exception_flow`; sandbox.py:809-822). Message says "Your simulate() CRASHED". | "simulate() CRASHED" in the llm.jsonl EXCEPTION FLOW payload + matching sandbox WARNING. |
| `real-divergence` | Genuine prediction miss on a real board change (blocked move, level-2 transition, HUD/step-timer ticks). Legitimate modeling signal. | Diff payload matches the real grid diff; no crash; no flash/highlight frame involved. |

## Triage table (17 rows = grep count)

| Frame | Timestamp (log) | Action | Phase transition | Cause class | Evidence |
|---|---|---|---|---|---|
| 31 | 2026-09-08 13:47:16,894 | 3 (left) | EXECUTE→MODEL (#1) | real-divergence | 1-layer frame; EXCEPTION FLOW payload: 50 cells, rows 20-24 cols 29-38 (9→4, 3→9) — player/box interaction, not flash/highlight (first flash is 16:00:25). |
| 34 | 2026-09-08 14:12:03,842 | 4 (right) | MODEL→MODEL (#2) | real-divergence | 1-layer; payload: 50 cells, rows 15-19 cols 34-43 (3→9, 9→4) — box interaction. |
| 36 | 2026-09-08 14:14:29,060 | 2 (down) | MODEL→MODEL (#3) | highlight-transition | Frame 35 is a highlight frame (stack sweep); real diff L0(35)→L0(36) = 128 cells (0→5 ×76 = highlight collapse, 9↔3 ×30) — the highlight landing lag simulate didn't model. |
| 38 | 2026-09-08 14:57:08,199 | 1 (up) | EXPLORE→MODEL (#1) | highlight-transition | Frame 38 IS a highlight frame (6 layers); payload: 56 cells rows 53-60 cols 1-10 (5→0) — the bar highlight repaint; settled (last layer) == L0(37), diff 0. |
| 41 | 2026-09-08 15:19:21,724 | 2 (down) | PLAN→MODEL (#2) | simulate-crash | 3× sandbox WARNING 15:19:21,671-717 "simulate crashed during predict-and-compare: list index out of range"; payload: "simulate() CRASHED … File \"<sandbox>\", line 28, in my_sim — IndexError: list index out of range". Pre-flash (first flash 16:00:25) → independent, not flash-cascaded. |
| 43 | 2026-09-08 15:29:33,972 | 1 (up) | MODEL→MODEL (#3) | real-divergence | 1-layer; payload: 15 cells rows 32-34 cols 34-38 (12→9) — small object-position miss. |
| 48 | 2026-09-08 16:00:25,077 | 3 (left) | EXECUTE→MODEL (#1) | flash-caused | Frame 48 = FLASH (6 layers, stack sweep); payload: 3892 cells, rows 0-63 cols 0-63 (4→11, 3→11) — the uniform yellow phantom; LLM reply: "The board turned entirely yellow (color 11) — a full-screen flash". First flash fired 16:00:25. |
| 49 | 2026-09-08 16:02:23,949 | 4 (right) | MODEL→MODEL (#2) | simulate-crash (flash-cascaded) | Sandbox WARNING 16:02:23,941 "min() iterable argument is empty"; payload: CRASHED in detect_player (line 5) via my_sim line 21. Frame 49's layer-0 board is the uniform phantom (4014 cells 11→base vs frame 48) — detect_player's min() over a color finds nothing on the all-11 board → crash is CAUSED by the flash misread. |
| 52 | 2026-09-08 16:10:54,991 | 1 (up) | MODEL→MODEL (#3) | real-divergence | 1-layer; payload: 25 cells rows 35-39 cols 39-43 (9→3, 12→3) — player footprint miss. |
| 56 | 2026-09-08 16:12:35,838 | 3 (left) | MODEL→MODEL (#1) | real-divergence | 1-layer; payload: 25 cells rows 30-34 cols 29-33 (9→4, 12→4). |
| 57 | 2026-09-08 16:19:27,321 | 1 (up) | MODEL→MODEL (#2) | real-divergence | 1-layer; payload: 25 cells rows 30-34 cols 34-38 (9→3, 12→3). |
| 61 | 2026-09-08 16:22:18,703 | 2 (down) | MODEL→MODEL (#3) | highlight-transition | Frame 61 IS a highlight frame (6 layers); payload: 49 cells in 2 regions — rows 9-15 cols 33-39 (5→0 ×24, box highlight) + rows 25-29 cols 19-23 (9→3, 12→3). |
| 67 | 2026-09-08 16:30:29,838 | 4 (right) | MODEL→MODEL (#1) | real-divergence | 1-layer; payload: 25 cells rows 25-29 cols 29-33 (9→3, 12→3). |
| 71 | 2026-09-08 16:33:53,889 | 1 (up) | MODEL→MODEL (#1) | real-divergence | Frame 70 = LEVEL_WIN (2 layers); real diff L0(70)→L0(71) = 1459 cells (4↔3 board repaint = level-2 board) — simulate had no level-2 model. Legitimate unmodeled transition, not flash/highlight. |
| 72 | 2026-09-08 16:36:32,439 | 2 (down) | MODEL→MODEL (#2) | real-divergence | 1-layer; payload: 29 cells in 2 regions — player footprint (25 cells) + bar rows 61-62 (11→3 ×4, step-timer tick). |
| 74 | 2026-09-08 16:37:18,976 | 4 (right) | MODEL→MODEL (#3) | real-divergence | 1-layer; payload: 29 cells — player footprint + bar tick (11→3 ×4). |
| 75 | 2026-09-08 16:38:37,142 | 1 (up) | EXPLORE→MODEL (#1) | real-divergence | 1-layer; payload: 29 cells — player footprint + bar tick (11→3 ×4). |

### Class totals

| Class | Frames | Count |
|---|---|---|
| flash-caused | 48 | 1 |
| simulate-crash (flash-cascaded) | 49 | 1 |
| simulate-crash (independent) | 41 | 1 |
| highlight-transition | 36, 38, 61 | 3 |
| real-divergence | 31, 34, 43, 52, 56, 57, 67, 71, 72, 74, 75 | 11 |

Note on frames 49-50: the task brief asks to flag simulate-crash rows at
"frames 49-50" as flash-cascaded vs independent. The log has no exception-flow
event at frame 50 (grep count 17 = rows above); the crash payload at frame 49
is the flash-cascaded one (uniform all-11 board breaks `detect_player`'s
`min()`), while frame 41's IndexError crash predates the first flash and is
independent. Frame 50 itself produced no exception flow.

## Fixed-by mapping (plan `.omo/plans/simulator-board-reset.md`)

| Cause class | Fixed by | Mechanism |
|---|---|---|
| flash-caused (48) | Tasks 3/6/7/8 | Task 3: settled-board extraction (no phantom layer-0 board in `_current_grid`/history). Task 6: `is_board_reset` detection + `BoardReset` abort before `predict_and_compare` — the flash transition never reaches the diff. Task 7: check()/diagnose() skip-set (no failure-counter churn post-flash). Task 8: `on_board_reset()` (path invalidation only) + BOARD_RESET_TEXT (model told the board was restored; no MODEL churn). |
| simulate-crash, flash-cascaded (49) | Tasks 3/6 (cascaded) | Eliminated transitively: with Task 3 the sandbox never sees the all-11 board, so `detect_player`'s min() cannot go empty. The crash-class itself (LLM simulate bugs) is NOT in this plan's scope. |
| simulate-crash, independent (41) | NOT fixed by this plan | LLM-authored simulate bug (IndexError in my_sim line 28). Mitigated only by the crash message's remedy hint (re-register via set_simulate). |
| highlight-transition (36, 38, 61) | NOT fixed by this plan | Real board changes (highlight overlay landing); Task 3's settled-board rule (layer 0 for HIGHLIGHT) keeps the base board correct, but the diff vs simulate remains a legitimate modeling signal. No false-positive risk from the detector (C1-C4 rejects highlights — validated: fires exactly {48, 92}). |
| real-divergence (11 rows) | NOT fixed by this plan (by design) | These are the exception-flow mechanism working as intended: real prediction misses the model should learn from. |

## Success metric (conservative)

- **Flash-caused exception flows → 0** after Tasks 3/6/7/8 (the flash
  transition is detected and handled at RESET-parity before
  `predict_and_compare` runs; the settled board replaces the phantom).
- **Flash-cascaded simulate-crash flows → 0** transitively (no phantom board
  → no empty-min() crash).
- **Conservative wall-clock target: ~30% reduction on ls20-class runs.**
  This run spent 3h40m wall-clock, 219 of 220 min = LLM latency; the two
  flash frames (48, 92) each triggered multi-minute MODEL churn spirals, but
  most of the latency is inherent LLM call time. This plan does NOT claim to
  "eliminate the 219-min problem" — it removes the flash-induced churn share
  only. The 11 real-divergence + 3 highlight-transition + 1 independent-crash
  flows remain and are legitimate.

## Evidence

- Log: `recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst.d91cdde0-b45a-41b4-9da3-d50985102d94.logs.log`
  - `grep -c "exception_flow #"` → 17
  - Crash WARNINGs: 15:19:21,671/.696/.717 (list index out of range → frame 41), 16:02:23,941 (min() iterable argument is empty → frame 49)
  - First flash: `frame=48 exception_flow #1 EXECUTE→MODEL` at 16:00:25,077
- LLM sidecar: `recordings/ls20-...d91cdde0....llm.jsonl` — EXCEPTION FLOW
  payloads per frame (action_id, n_diff, regions / crash traceback), first
  occurrence per unique payload matched 1:1 to the 17 flow frames.
- Stack sweep anchors (prior analysis, plan Context): flash frames {48, 92};
  highlight frames {19, 20, 21, 35, 38, 61}; win frame 70.
- Code paths: `agents/simulator_agent/workflow.py:211-218`
  (`on_exception_flow` — every `#N` log line), `agents/simulator_agent/sandbox.py:683-822`
  (`predict_and_compare` — diff path populates `regions`, crash path sets
  `error` + WARNING), `agents/simulator_agent/agent.py:663-676`
  (injection + `on_exception_flow()` call),
  `agents/simulator_agent/exception_flow.py:30-88` (message builder).