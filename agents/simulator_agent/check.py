"""Check and diagnostic functions for the simulator agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.simulator_agent.reset_policy import check_includes_reset, is_reset
from agents.simulator_agent.tools import grid_diff

UNKNOWN = -1
"""Sentinel a simulate writes into a cell it cannot predict ("I don't know").

check() scores committed cells only; abstained cells are counted per bucket
(abstained-changed / abstained-stable) and never judged wrong or spurious.
"""

__all__ = ["UNKNOWN", "cluster_cells", "diagnose", "run_check"]


def run_check(
    simulate_fn: Callable[[list[list[int]], int], list[list[int]]],
    grids: list[list[list[int]]],
    actions: list[int],
    *,
    verbose: bool = True,
    skip_transitions: set[int] | None = None,
) -> dict[str, Any]:
    """Test *simulate_fn* against all recorded frame transitions.

    For each frame *i*, runs ``simulate_fn(grids[i], actions[i])`` and compares
    the result to ``grids[i + 1]`` (the actual next grid).

    Per-cell classification, UNKNOWN-first: a cell the model abstained on
    (``predicted == UNKNOWN``) is never judged — it lands in
    **abstained_changed** (reality changed it) or **abstained_stable**
    (reality left it alone).  Committed cells keep the classic buckets:

      - **wrong**: committed cells where predicted != actual
      - **changed**: cells where grid_before != grid_after (what the game changed)
      - **correct**: of changed cells, how many simulate predicted right
      - **spurious**: cells simulate changed that shouldn't have changed

    ``accuracy = correct / (correct + wrong)`` over committed cells only;
    a zero denominator (nothing committed) is defined as 100%.  ``coverage``
    is the committed share of actually-changed cells (``None`` when nothing
    changed).

    If *skip_transitions* is provided, those transition indices are excluded
    from scoring entirely: no per-frame entry, no simulate call, and no
    contribution to any aggregate (totals, accuracy denominators,
    frames_correct/frames_total).  Use this for engine-initiated board resets
    whose transition no simulate can model (the level-budget flash).  RESET
    *actions* remain scored per ``check_includes_reset()`` — only
    engine-event transitions are skipped.  Out-of-range indices are ignored.

    Returns aggregate + per-frame metrics.  Prints a summary when *verbose*.
    """
    results: list[dict[str, Any]] = []
    total_wrong = 0
    total_changed = 0
    total_correct = 0
    total_spurious = 0
    total_abstained_changed = 0
    total_abstained_stable = 0
    frames_correct = 0
    abstained_changed_by_transition: dict[int, set[tuple[int, int]]] = {}
    abstained_stable_cells: set[tuple[int, int]] = set()
    frame_lines: list[str] = []
    skip = skip_transitions or set()
    all_transitions = max(len(grids) - 1, 0)
    scored_transitions = [i for i in range(all_transitions) if i not in skip]
    frames_total = len(scored_transitions)
    skipped_transitions = sorted(i for i in skip if 0 <= i < all_transitions)
    degenerate_reset_frames = 0
    score_reset = check_includes_reset()

    for i in scored_transitions:
        grid_before = grids[i]
        action = actions[i]
        grid_after = grids[i + 1]

        try:
            predicted = simulate_fn(grid_before, action)
        except Exception as exc:
            if verbose:
                print(f"Frame {i}: ERROR — {type(exc).__name__}: {exc}")
            results.append({"frame": i, "error": str(exc), "wrong": 4096})
            total_wrong += 4096
            continue

        if predicted is None:
            if verbose:
                print(f"Frame {i}: simulate returned None")
            total_wrong += 4096
            results.append({"frame": i, "error": "returned None", "wrong": 4096})
            continue

        # grid_diff doubles as shape validation (raises on mismatch).
        wrong = grid_diff(predicted, grid_after)
        changed = grid_diff(grid_before, grid_after)

        # UNKNOWN-first: an abstained cell is never judged.  wrong/pred_changed
        # carry (r, c, predicted_val, actual_val) / (r, c, before, predicted).
        wrong = [d for d in wrong if d[2] != UNKNOWN]
        changed_set = {(r, c) for r, c, _, _ in changed}
        wrong_set = {(r, c) for r, c, _, _ in wrong}
        abstained_changed_set = {
            (r, c) for r, c in changed_set if predicted[r][c] == UNKNOWN
        }
        n_abstained = sum(row.count(UNKNOWN) for row in predicted)
        abstained_stable = n_abstained - len(abstained_changed_set)

        correct = len(changed_set - wrong_set - abstained_changed_set)

        # Spurious: simulate changed a committed cell that didn't actually change
        pred_changed = grid_diff(grid_before, predicted)
        pred_changed = [d for d in pred_changed if d[3] != UNKNOWN]
        pred_changed_set = {(r, c) for r, c, _, _ in pred_changed}
        spurious = len(pred_changed_set - changed_set)

        n_wrong = len(wrong)
        n_changed = len(changed)
        n_spurious = spurious
        n_abstained_changed = len(abstained_changed_set)
        committed = correct + n_wrong
        accuracy = correct / committed * 100 if committed else 100.0

        if abstained_changed_set:
            abstained_changed_by_transition[i] = abstained_changed_set
        # Stable-CELL-SET for the cluster line: abstained in this transition
        # AND abstained-changed in NO transition (a cell that changed in one
        # transition and was stable in another is NOT "no changes observed").
        # The cross-transition exclusion is applied after the loop, when the
        # abstained-changed union is complete.
        abstained_stable_cells.update(
            (r, c)
            for r in range(len(predicted))
            for c in range(len(predicted[r]))
            if predicted[r][c] == UNKNOWN and (r, c) not in abstained_changed_set
        )

        total_wrong += n_wrong
        total_changed += n_changed
        total_correct += correct
        total_spurious += n_spurious
        total_abstained_changed += n_abstained_changed
        total_abstained_stable += abstained_stable

        if n_wrong == 0:
            frames_correct += 1

        is_degenerate_reset = score_reset and is_reset(action) and n_changed == 0
        if is_degenerate_reset:
            degenerate_reset_frames += 1

        if verbose:
            if n_wrong == 0 and n_abstained_changed == 0:
                frame_lines.append(f"Frame {i}: OK")
            else:
                frame_lines.append(
                    f"Frame {i}: {n_wrong} wrong — "
                    f"{n_abstained_changed} abstained-changed"
                )
            if 0 < n_wrong <= 20:
                for r, c, pv, av in wrong:
                    frame_lines.append(f"  ({r},{c}): predicted={pv}, actual={av}")

        results.append(
            {
                "frame": i,
                "wrong": n_wrong,
                "changed": n_changed,
                "correct": correct,
                "spurious": n_spurious,
                "abstained_changed": n_abstained_changed,
                "abstained_stable": abstained_stable,
                "accuracy": round(accuracy, 1),
                "degenerate_reset": is_degenerate_reset,
            }
        )

    committed_total = total_correct + total_wrong
    overall_acc = total_correct / committed_total * 100 if committed_total else 100.0
    coverage = (
        (total_changed - total_abstained_changed) / total_changed * 100
        if total_changed
        else None
    )

    abstained_clusters = _build_abstained_clusters(
        abstained_changed_by_transition, actions
    )
    abstained_changed_union: set[tuple[int, int]] = (
        set().union(*abstained_changed_by_transition.values())
        if abstained_changed_by_transition
        else set()
    )
    abstained_stable_cells -= abstained_changed_union

    if verbose:
        print()
        print(
            f"Overall: {total_wrong} wrong, {total_changed} changed, "
            f"{total_correct} correct ({overall_acc:.1f}%), "
            f"{total_spurious} spurious"
        )
        summary = f"Frames correct: {frames_correct}/{frames_total}"
        if degenerate_reset_frames:
            summary += f" ({degenerate_reset_frames} degenerate RESET)"
        print(summary)
        coverage_str = f"{coverage:.1f}" if coverage is not None else "n/a"
        print(
            f"Modeled: {total_correct} correct, {total_wrong} wrong, "
            f"{total_abstained_changed} abstained-changed "
            f"({total_abstained_stable} abstained-stable), "
            f"coverage {coverage_str}%"
        )
        if abstained_clusters or abstained_stable_cells:
            print("Abstained regions (you did not predict these):")
            for cluster in sorted(
                abstained_clusters,
                key=lambda cl: len(cl["changed_on_transitions"]),
                reverse=True,
            )[:5]:
                changed_on = cluster["changed_on_transitions"]
                actions_hist = cluster["actions"]
                actions_str = ", ".join(
                    f"{action_id}: {count}"
                    for action_id, count in sorted(actions_hist.items())
                )
                print(
                    f"  rows {cluster['bbox'][0]}-{cluster['bbox'][2]}, "
                    f"cols {cluster['bbox'][1]}-{cluster['bbox'][3]}: "
                    f"{cluster['cells_per_frame_max']} cells/frame-max — "
                    f"changed on {len(changed_on)}/{frames_total} transitions "
                    f"— actions {{{actions_str}}}"
                )
            if abstained_stable_cells:
                stable_cluster_count = len(
                    cluster_cells(sorted(abstained_stable_cells))
                )
                print(
                    f"  +{stable_cluster_count} regions abstained-stable "
                    f"(no changes observed)"
                )
        # Per-frame detail goes LAST so sandbox print truncation eats it
        # first; the summary and abstained log always survive intact.
        for line in frame_lines:
            print(line)

    return {
        "schema": 2,
        "total_wrong": total_wrong,
        "total_changed": total_changed,
        "total_correct": total_correct,
        "total_spurious": total_spurious,
        "abstained_changed": total_abstained_changed,
        "abstained_stable": total_abstained_stable,
        "coverage": round(coverage, 1) if coverage is not None else None,
        "overall_accuracy": round(overall_acc, 1),
        "frames_correct": frames_correct,
        "frames_total": frames_total,
        "degenerate_reset_frames": degenerate_reset_frames,
        "skipped_transitions": skipped_transitions,
        "wrong_cells": total_wrong,
        "abstained_clusters": abstained_clusters,
        "per_frame": results,
    }


def _build_abstained_clusters(
    abstained_changed_by_transition: dict[int, set[tuple[int, int]]],
    actions: list[int],
) -> list[dict[str, Any]]:
    """Cluster abstained-changed cells and measure their change/action history.

    Pure measurement over recorded history: for each spatial cluster (union
    of abstained-changed cells across all scored transitions, grouped by
    ``cluster_cells``), report where it is, how many of its cells changed in
    the busiest transition, which transitions changed it, and which actions
    produced those transitions (an action counts once per transition, not
    once per cell).
    """
    union_cells = sorted(
        cell for cells in abstained_changed_by_transition.values() for cell in cells
    )
    clusters = cluster_cells(union_cells)
    tables: list[dict[str, Any]] = []
    for cluster in clusters:
        changed_on: list[int] = []
        actions_hist: dict[int, int] = {}
        cells_per_frame_max = 0
        for transition, cells in abstained_changed_by_transition.items():
            overlap = cluster & cells
            if not overlap:
                continue
            changed_on.append(transition)
            actions_hist[actions[transition]] = (
                actions_hist.get(actions[transition], 0) + 1
            )
            cells_per_frame_max = max(cells_per_frame_max, len(overlap))
        rows = [r for r, _ in cluster]
        cols = [c for _, c in cluster]
        tables.append(
            {
                "bbox": (min(rows), min(cols), max(rows), max(cols)),
                "cells_per_frame_max": cells_per_frame_max,
                "changed_on_transitions": sorted(changed_on),
                "actions": dict(sorted(actions_hist.items())),
            }
        )
    return tables


def diagnose(
    simulate_fn: Callable[[list[list[int]], int], list[list[int]]],
    grids: list[list[list[int]]],
    actions: list[int],
    skip_transitions: set[int] | None = None,
) -> dict[str, Any]:
    """Run simulate on all frames and print semantic error analysis.

    Classifies each wrong cell as MISSED, SPURIOUS, or WRONG_VALUE,
    then clusters errors spatially to identify which object region
    is causing problems.

    If *skip_transitions* is provided, those transition indices are excluded
    from scoring and from every aggregate total (same semantics as
    :func:`run_check`).  Out-of-range indices are ignored.
    """
    print("=== Diagnosis ===")
    total_wrong = 0
    total_missed = 0
    total_spurious = 0
    total_wrong_val = 0
    total_abstained = 0
    skip = skip_transitions or set()
    all_transitions = max(len(grids) - 1, 0)
    scored_transitions = [i for i in range(all_transitions) if i not in skip]
    frames_total = len(scored_transitions)
    skipped_transitions = sorted(i for i in skip if 0 <= i < all_transitions)

    for i in scored_transitions:
        grid_before = grids[i]
        action = actions[i]
        grid_after = grids[i + 1]

        try:
            predicted = simulate_fn(grid_before, action)
        except Exception as exc:
            print(f"Frame {i} Action {action}: ERROR — {type(exc).__name__}: {exc}")
            continue
        if predicted is None:
            print(f"Frame {i} Action {action}: returned None")
            continue

        actual_diff = grid_diff(grid_before, grid_after)
        pred_diff = grid_diff(grid_before, predicted)

        # UNKNOWN-first (mirrors run_check): a cell the model abstained on
        # (predicted == UNKNOWN) is never judged — it is neither MISSED nor
        # SPURIOUS. Without this filter every abstained cell would land in
        # pred_set and count as SPURIOUS (phantom errors).
        abstained_set = {
            (r, c)
            for r in range(len(predicted))
            for c in range(len(predicted[r]))
            if predicted[r][c] == UNKNOWN
        }
        actual_set = {(r, c) for r, c, _, _ in actual_diff}
        pred_set = {(r, c) for r, c, _, _ in pred_diff if (r, c) not in abstained_set}

        missed = actual_set - pred_set - abstained_set
        spurious = pred_set - actual_set
        wrong_val = {
            (r, c)
            for r, c, _, _ in actual_diff
            if (r, c) in pred_set and predicted[r][c] != grid_after[r][c]
        }
        n_abstained = len(abstained_set)

        n_missed = len(missed)
        n_spurious = len(spurious)
        n_wrong_val = len(wrong_val)
        n_wrong = n_missed + n_spurious + n_wrong_val

        total_wrong += n_wrong
        total_missed += n_missed
        total_spurious += n_spurious
        total_wrong_val += n_wrong_val
        total_abstained += n_abstained

        if n_wrong == 0:
            continue

        print(f"\nFrame {i} Action {action}: {n_wrong} errors")
        print(
            f"  MISSED ({n_missed}): cells that changed in reality but your simulate didn't change"
        )
        print(
            f"  SPURIOUS ({n_spurious}): cells your simulate changed but shouldn't have"
        )
        print(
            f"  WRONG_VALUE ({n_wrong_val}): cells your simulate changed but to the wrong color"
        )

        all_errors = (
            [(r, c, "MISSED", grid_after[r][c], grid_before[r][c]) for r, c in missed]
            + [
                (r, c, "SPURIOUS", grid_before[r][c], predicted[r][c])
                for r, c in spurious
            ]
            + [
                (r, c, "WRONG_VAL", grid_after[r][c], predicted[r][c])
                for r, c in wrong_val
            ]
        )

        # Cluster errors by spatial proximity (group nearby cells)
        clusters = cluster_cells([(r, c) for r, c, *_ in all_errors])
        for ci, cluster in enumerate(clusters):
            rows = [r for r, _ in cluster]
            cols = [c for _, c in cluster]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            cluster_errors = [e for e in all_errors if (e[0], e[1]) in cluster]
            types = set(e[2] for e in cluster_errors)
            # What colors are at this region in before/after/predicted
            before_colors = set(grid_before[r][c] for r, c in cluster)
            after_colors = set(grid_after[r][c] for r, c in cluster)
            pred_colors = set(predicted[r][c] for r, c in cluster)
            print(
                f"  Cluster {ci + 1}: rows {r_min}-{r_max}, cols {c_min}-{c_max} ({len(cluster)} cells)"
            )
            print(f"    Error types: {types}")
            print(f"    Grid before colors: {before_colors}")
            print(f"    Grid after colors:  {after_colors}")
            print(f"    Predicted colors:   {pred_colors}")
            if len(cluster) <= 10:
                for r, c, etype, v1, v2 in cluster_errors:
                    print(
                        f"    ({r},{c}): {etype} before={grid_before[r][c]} after={grid_after[r][c]} pred={predicted[r][c]}"
                    )

    print("\n=== Diagnosis Summary ===")
    print(
        f"Total errors: {total_wrong} (missed={total_missed}, spurious={total_spurious}, wrong_val={total_wrong_val})"
    )
    if total_abstained:
        print(f"Abstained cells (not judged): {total_abstained}")
    return {
        "total_wrong": total_wrong,
        "total_missed": total_missed,
        "total_spurious": total_spurious,
        "total_wrong_val": total_wrong_val,
        "total_abstained": total_abstained,
        "frames_total": frames_total,
        "skipped_transitions": skipped_transitions,
    }


def cluster_cells(
    cells: list[tuple[int, int]], threshold: int = 3
) -> list[set[tuple[int, int]]]:
    """Group cells that are within *threshold* rows/cols of each other."""
    if not cells:
        return []
    clusters: list[set[tuple[int, int]]] = []
    for cell in cells:
        merged = False
        for cluster in clusters:
            if any(
                abs(cell[0] - other[0]) <= threshold
                and abs(cell[1] - other[1]) <= threshold
                for other in cluster
            ):
                cluster.add(cell)
                merged = True
                break
        if not merged:
            clusters.append({cell})
    return clusters
