# 2026-09-09 — Frame-stack sweep (sampled corpus scan)

Task 5 of the simulator-board-reset plan: run `classify_stack` over a
per-game sample of `recordings/`, pin the d91cdde0 output as a regression
fixture, and triage the duckharness 553ef211 9-consecutive multi-layer
streak (frames 95-103).

Script: `scripts/scan_frame_stacks.py` (single path / `--all DIR` /
`--sample DIR`). Evidence: `.omo/evidence/task-5-*.txt|json`.

## Sample composition

`--sample recordings/` groups recordings by `<game>.<agent>` (first two
dot-components) and picks up to 2 per group, prioritised by multi-layer
frame count, plus one zero-hit recording per group so every prefix is
covered. The incident anchors d91cdde0 and 553ef211 are force-included.
Result: **19 recordings, 13 groups, all covered**:

| Group | Picked | Why |
|---|---|---|
| ls20.simulatorfirstagent | 11fe642e (23 ML), b3a8ef1f (20), **d91cdde0** (9, anchor) | 2 most multi-layer-heavy + mandatory anchor |
| ls20.duckharnessagent | **553ef211** (13, anchor), 74246995 (9) | anchor + next-heaviest |
| ls20.curiosity | befa1f2f (6), 8426cb9f (5) | heaviest two |
| ls20.llmcuriosity | ec7fc720 (50), 9b0411e1 (13) | heaviest two |
| ls20.random | d9c77dd1 (2), 3b408ada (1) | heaviest two |
| ls20.probe | 9dc3036f (0) | prefix coverage |
| wa30.human | 697d5e3e (1) | only recording |
| wa30-ee6fef47.{duckharnessagent, langgraphunifiedagent, langgraphvisionagent, llmcuriosity, llmcuriosityv2, probe} | 1 each (all 0 ML) | prefix coverage — wa30-class runs show no animation stacks |

## Sampled-sweep results

Totals: **18 flash_reset / 131 highlight / 3 level_win** across 143
multi-layer frames in 19 recordings. Every FLASH_RESET satisfied C3
(`diff(settled, level_start) ≤ 128`); **zero C3 failures, zero UNKNOWN
classifications**. Per-recording breakdown is in
`.omo/evidence/task-5-sweep-summary.json`.

Notable: `ls20.llmcuriosity.ec7fc720` alone contributes 50 highlight
frames (a long highlight-heavy run); `wa30.human.697d5e3e` frame 29 is a
2-layer LEVEL_WIN (human run, action ACTION5).

## d91cdde0 pin (regression fixture)

Single-path run reproduces the incident numbers exactly (pinned in
`tests/unit/simulator_agent/test_frame_layers.py::TestD91cdde0SweepPin`,
loaded by path since the recording is deliberately not in
`tests/reference_recordings.json`):

| frame | class | layers | C3 |
|---|---|---|---|
| 19, 20, 21, 35, 38, 61 | highlight | 6 | n/a |
| 48 | flash_reset | 6 | diff=4 ≤ 128 OK |
| 70 | level_win | 2 | n/a |
| 92 | flash_reset | 6 | diff=4 ≤ 128 OK |

Note on f92's level-start reference: the sweep's level-start tracker keys
on `levels_completed` increments, so level 2's start board is the settled
board of frame 70 (the LEVEL_WIN stack's last layer), not frame 71's
single-layer board. `diff(f92 settled, f70 settled) = 4`; the 58-cell
figure quoted in learnings.md is the diff against f71's board (one action
of drift after the win transition). Both are ≤ TOL; the tracker's choice
is the tighter, more principled reference.

## 553ef211 streak triage (frames 95-103)

Ground truth: the agent pressed ACTION_1 nine times in a row with the
board frozen — `diff(f95 L5, f103 L5) = 0`, `levels_completed = 0`
throughout, and the first board change after the streak is a 2-cell
player step at f104 (action 3). Each response frame is a 6-layer stack:
layer 5 = the live board (identical to f94's board), layers 0-4 = the
same board with a 76-cell colour-5 block (rows 53-62, cols 1-10)
repainted colour 0.

Per-frame classification with `level_start = settled_board(f0)`
(`levels_completed` never increments in this recording; TOL = 128):

| frame | class | layers | anim_uniform | diff(settled, level_start) | is_board_reset fires |
|---|---|---|---|---|---|
| 95 | highlight | 6 | False | 148 | **False** |
| 96 | highlight | 6 | False | 148 | **False** |
| 97 | highlight | 6 | False | 148 | **False** |
| 98 | highlight | 6 | False | 148 | **False** |
| 99 | highlight | 6 | False | 148 | **False** |
| 100 | highlight | 6 | False | 148 | **False** |
| 101 | highlight | 6 | False | 148 | **False** |
| 102 | highlight | 6 | False | 148 | **False** |
| 103 | highlight | 6 | False | 148 | **False** |
| 43 (real flash) | flash_reset | 6 | True | 4 | True |
| 87 (real flash) | flash_reset | 6 | True | 8 | True |

**Verdict: option (b) — a different animation class (HIGHLIGHT), and the
detector correctly does not fire on any streak frame.** The streak is
NOT repeated budget resets: the animation layers are non-uniform
(identical repainted boards, not flat colour flashes), so C2 fails, and
the settled layer is the frozen live board, 148 cells from the level
start, so C3 would fail too. The two genuine flashes (43, 87) fire
exactly as expected. No re-entry-guard concern arises from this
recording; Task 6's guard is still valuable for the repeated-flash case
seen in other runs (e.g. 74246995 with 5 flashes).

## Anomalies

None requiring issues.md: zero C3 failures, zero UNKNOWN frames, zero
malformed lines beyond the known scorecard-only trailing lines (skipped
and noted by the script). The 553ef211 streak resolved to a documented
class (HIGHLIGHT) rather than an unanticipated one.