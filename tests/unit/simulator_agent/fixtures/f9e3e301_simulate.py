"""The f9e3e301 run's REAL registered simulate — verbatim, plus a
counterfactual UNKNOWN abstention addition.

Source: recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst.
f9e3e301-7f2f-4703-ae19-85f051412259.llm.jsonl seq 10 (frame 4) — the
``set_simulate(my_sim)`` python tool call.  The movement logic
(``_my_sim_verbatim``, originally named ``my_sim``) is byte-identical to
what the agent registered in that run: a pure player-movement sim
(orange-12/blue-9 5×5 player, DIRS 1-4 by 5 cells, walls 4/5 block).

The run masked the key box + timer EXTERNALLY via ``set_ignore`` (the
then-available masking tool).  This fixture is the COUNTERFACTUAL: the
same sim abstains INSIDE the model — ``my_sim`` writes ``UNKNOWN (-1)``
into the masked regions (rows 61-62 all cols; rows 53-62 cols 1-10) on
every return path, including the blocked-move early returns (the
key-box flash transitions 10/20/22 are all blocked moves — an
end-of-body-only addition would silently skip them).

Importable surface: ``my_sim(grid, action)`` and ``_my_sim_verbatim``
(the pristine movement sim, for the negative control).
"""

# ruff: noqa: E701, E702, I001
# The verbatim block below is RECORDED SOURCE (llm.jsonl seq 10) committed
# byte-identical; its compressed style (multi-statement lines, mid-file
# import) is historical data, not code to reformat.

from __future__ import annotations

from agents.simulator_agent.check import UNKNOWN

# ── Verbatim sim source (llm.jsonl seq 10) — DO NOT EDIT ───────────────────
# fmt: off
import copy
def find_player(g):
    oc=[(r,c) for r in range(64) for c in range(64) if g[r][c]==12]
    if not oc: return None
    r0=min(r for r,c in oc)
    c0=min(c for r,c in oc); c1=max(c for r,c in oc)
    return (r0, c0, r0+4, c1)

DIRS={1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}
WALL={4,5}

def _my_sim_verbatim(g, a):
    g=copy.deepcopy(g)
    if a not in DIRS: return g
    p=find_player(g)
    if not p: return g
    r0,c0,r1,c1=p
    dr,dc=DIRS[a]
    nr0,nc0=r0+dr,c0+dc
    # check all destination cells within bounds and not wall
    ok=True
    for rr in range(nr0,nr0+5):
        for cc in range(nc0,nc0+5):
            if rr<0 or rr>=64 or cc<0 or cc>=64: ok=False; break
            if g[rr][cc] in WALL: ok=False; break
        if not ok: break
    if not ok: return g
    # clear old
    for rr in range(r0,r0+5):
        for cc in range(c0,c0+5):
            g[rr][cc]=3
    # draw new: top2 orange, bottom3 blue
    for rr in range(nr0,nr0+5):
        for cc in range(nc0,nc0+5):
            g[rr][cc]=12 if rr<nr0+2 else 9
    return g
# fmt: on
# ── End verbatim source ────────────────────────────────────────────────────


def my_sim(g, a):
    """Verbatim movement sim + counterfactual UNKNOWN abstention.

    The UNKNOWN writes below are the test fixture's counterfactual of the
    run's external ``set_ignore`` masking, applied inside the model: rows
    61-62 all cols, and rows 53-62 cols 1-10.  They run on EVERY return
    path (the flash transitions 10/20/22 are blocked moves that early-
    return inside ``_my_sim_verbatim``).
    """
    g = _my_sim_verbatim(g, a)
    for _r in (61, 62):
        for _c in range(64):
            g[_r][_c] = UNKNOWN
    for _r in range(53, 63):
        for _c in range(1, 11):
            g[_r][_c] = UNKNOWN
    return g
