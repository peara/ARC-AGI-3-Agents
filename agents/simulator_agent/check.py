"""Check and diagnostic functions for the simulator agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.simulator_agent.tools import grid_diff


def run_check(
    simulate_fn: Callable[[int, int], list[list[int]]],
    grids: list[list[list[int]]],
    actions: list[int],
    *,
    verbose: bool = True,
    ignore_mask: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Test *simulate_fn* against all recorded frame transitions.

    For each frame *i*, runs ``simulate_fn(i, actions[i])`` and compares the
    result to ``grids[i + 1]`` (the actual next grid).

    Metrics:
      - **wrong**: cells where predicted != actual
      - **changed**: cells where grid_before != grid_after (what the game changed)
      - **correct**: of changed cells, how many simulate predicted right
      - **spurious**: cells simulate changed that shouldn't have changed

    If *ignore_mask* is provided, those cells are excluded from wrong/changed/
    spurious counts — use ``set_ignore()`` to skip HUD cells that change every
    frame regardless of the action.

    Returns aggregate + per-frame metrics.  Prints a summary when *verbose*.
    """
    results: list[dict[str, Any]] = []
    total_wrong = 0
    total_changed = 0
    total_correct = 0
    total_spurious = 0
    frames_correct = 0
    frames_total = len(grids) - 1

    for i in range(frames_total):
        grid_before = grids[i]
        action = actions[i]
        grid_after = grids[i + 1]

        try:
            predicted = simulate_fn(i, action)
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

        wrong = grid_diff(predicted, grid_after)
        changed = grid_diff(grid_before, grid_after)

        if ignore_mask:
            wrong = [d for d in wrong if (d[0], d[1]) not in ignore_mask]
            changed = [d for d in changed if (d[0], d[1]) not in ignore_mask]

        changed_set = {(r, c) for r, c, _, _ in changed}
        wrong_set = {(r, c) for r, c, _, _ in wrong}
        correct = len(changed_set - wrong_set)

        # Spurious: simulate changed a cell that didn't actually change
        pred_changed = grid_diff(grid_before, predicted)
        if ignore_mask:
            pred_changed = [
                d for d in pred_changed if (d[0], d[1]) not in ignore_mask
            ]
        pred_changed_set = {(r, c) for r, c, _, _ in pred_changed}
        spurious = len(pred_changed_set - changed_set)

        n_wrong = len(wrong)
        n_changed = len(changed)
        n_spurious = spurious
        accuracy = correct / max(n_changed, 1) * 100

        total_wrong += n_wrong
        total_changed += n_changed
        total_correct += correct
        total_spurious += n_spurious

        if n_wrong == 0:
            frames_correct += 1

        if verbose:
            print(
                f"Frame {i}: {n_wrong} wrong, {n_changed} changed, "
                f"{correct} correct ({accuracy:.0f}%), {n_spurious} spurious"
            )
            if 0 < n_wrong <= 20:
                for r, c, pv, av in wrong:
                    print(f"  ({r},{c}): predicted={pv}, actual={av}")

        results.append({
            "frame": i,
            "wrong": n_wrong,
            "changed": n_changed,
            "correct": correct,
            "spurious": n_spurious,
            "accuracy": round(accuracy, 1),
        })

    overall_acc = total_correct / max(total_changed, 1) * 100

    if verbose:
        print()
        print(
            f"Overall: {total_wrong} wrong, {total_changed} changed, "
            f"{total_correct} correct ({overall_acc:.1f}%), "
            f"{total_spurious} spurious"
        )
        print(f"Frames correct: {frames_correct}/{frames_total}")

    return {
        "total_wrong": total_wrong,
        "total_changed": total_changed,
        "total_correct": total_correct,
        "total_spurious": total_spurious,
        "overall_accuracy": round(overall_acc, 1),
        "frames_correct": frames_correct,
        "frames_total": frames_total,
        "per_frame": results,
    }


def diagnose(
    simulate_fn: Callable[[int, int], list[list[int]]],
    grids: list[list[list[int]]],
    actions: list[int],
    ignore_mask: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Run simulate on all frames and print semantic error analysis.

    Classifies each wrong cell as MISSED, SPURIOUS, or WRONG_VALUE,
    then clusters errors spatially to identify which object region
    is causing problems.

    If *ignore_mask* is provided, those cells are excluded from the analysis.
    """
    print("=== Diagnosis ===")
    total_wrong = 0
    total_missed = 0
    total_spurious = 0
    total_wrong_val = 0
    frames_total = len(grids) - 1

    for i in range(frames_total):
        grid_before = grids[i]
        action = actions[i]
        grid_after = grids[i + 1]

        try:
            predicted = simulate_fn(i, action)
        except Exception as exc:
            print(f"Frame {i} Action {action}: ERROR — {type(exc).__name__}: {exc}")
            continue
        if predicted is None:
            print(f"Frame {i} Action {action}: returned None")
            continue

        actual_diff = grid_diff(grid_before, grid_after)
        pred_diff = grid_diff(grid_before, predicted)

        if ignore_mask:
            actual_diff = [d for d in actual_diff if (d[0], d[1]) not in ignore_mask]
            pred_diff = [d for d in pred_diff if (d[0], d[1]) not in ignore_mask]

        actual_set = {(r, c) for r, c, _, _ in actual_diff}
        pred_set = {(r, c) for r, c, _, _ in pred_diff}

        missed = actual_set - pred_set
        spurious = pred_set - actual_set
        wrong_val = {
            (r, c) for r, c, _, _ in actual_diff
            if (r, c) in pred_set and predicted[r][c] != grid_after[r][c]
        }

        n_missed = len(missed)
        n_spurious = len(spurious)
        n_wrong_val = len(wrong_val)
        n_wrong = n_missed + n_spurious + n_wrong_val

        total_wrong += n_wrong
        total_missed += n_missed
        total_spurious += n_spurious
        total_wrong_val += n_wrong_val

        if n_wrong == 0:
            continue

        print(f"\nFrame {i} Action {action}: {n_wrong} errors")
        print(f"  MISSED ({n_missed}): cells that changed in reality but your simulate didn't change")
        print(f"  SPURIOUS ({n_spurious}): cells your simulate changed but shouldn't have")
        print(f"  WRONG_VALUE ({n_wrong_val}): cells your simulate changed but to the wrong color")

        all_errors = (
            [(r, c, "MISSED", grid_after[r][c], grid_before[r][c]) for r, c in missed]
            + [(r, c, "SPURIOUS", grid_before[r][c], predicted[r][c]) for r, c in spurious]
            + [(r, c, "WRONG_VAL", grid_after[r][c], predicted[r][c]) for r, c in wrong_val]
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
            print(f"  Cluster {ci+1}: rows {r_min}-{r_max}, cols {c_min}-{c_max} ({len(cluster)} cells)")
            print(f"    Error types: {types}")
            print(f"    Grid before colors: {before_colors}")
            print(f"    Grid after colors:  {after_colors}")
            print(f"    Predicted colors:   {pred_colors}")
            if len(cluster) <= 10:
                for r, c, etype, v1, v2 in cluster_errors:
                    print(f"    ({r},{c}): {etype} before={grid_before[r][c]} after={grid_after[r][c]} pred={predicted[r][c]}")

    print("\n=== Diagnosis Summary ===")
    print(f"Total errors: {total_wrong} (missed={total_missed}, spurious={total_spurious}, wrong_val={total_wrong_val})")
    return {
        "total_wrong": total_wrong,
        "total_missed": total_missed,
        "total_spurious": total_spurious,
        "total_wrong_val": total_wrong_val,
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
