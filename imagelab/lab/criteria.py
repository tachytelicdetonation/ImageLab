"""
What counts as "the result" of a run — read from the run itself, not from a central table.

Tasks declare their own editorial choices as class attributes (see imagelab/task.py):

    class MyTask(Task):
        overfit_gate = ("val/loss", "<=", 0.01)     # --tier overfit kill criterion
        key_metrics = ["val/fid", "val/loss"]       # ledger/leaderboard columns
        higher_is_better = frozenset({"val/acc"})

The trainer copies these into every ledger row, so `lab runs/board/compare/gallery`
render any model family without importing its code — and keep working if the task's
source moves or disappears. The heuristics below only serve rows written before these
fields existed (or by tasks that declare nothing).
"""

from __future__ import annotations


def gate_verdict(gate: tuple | None, metrics: dict) -> tuple[str, str]:
    """('pass'|'fail'|'unknown', human explanation) for an overfit-tier run.
    `gate` is the task's (metric, "<="|" >=", threshold) declaration."""
    if not gate:
        return "unknown", ("task declares no overfit_gate — set "
                           "`overfit_gate = (\"metric\", \"<=\", threshold)` on the class")
    try:
        metric, op, threshold = gate
    except (TypeError, ValueError):
        return "unknown", f"malformed overfit_gate {gate!r} (want (metric, op, threshold))"
    if op not in ("<=", ">="):
        return "unknown", f"overfit_gate op {op!r} must be '<=' or '>='"
    v = metrics.get(metric)
    if v is None:
        return "unknown", f"gate metric {metric!r} missing from final metrics"
    ok = (v <= threshold) if op == "<=" else (v >= threshold)
    shown_op = op if ok else (">" if op == "<=" else "<")
    note = (f"{metric} {v:.4f} {shown_op} {threshold} — "
            + ("memorized; the gradient path works" if ok else
               "could NOT memorize a single sample; suspect gradient flow / loss wiring / "
               "data pipeline before burning a real run"))
    return ("pass" if ok else "fail"), note


def key_metrics_of(row: dict, n: int = 3) -> dict:
    """The first n of the run's declared key metrics present in its final metrics.
    Rows without a declaration fall back to val/ then eval/ prefixed metrics."""
    fm = row.get("final_metrics") or {}
    keys = row.get("key_metrics") or (
        [k for k in fm if k.startswith("val/")] + [k for k in fm if k.startswith("eval/")])
    out = {}
    for k in keys:
        if k in fm:
            out[k] = fm[k]
            if len(out) >= n:
                break
    return out


# Fallback name hints for rows that never declared higher_is_better.
_HIB_HINTS = ("acc", "psnr", "ssim", "usage", "score", "precision", "recall", "iou")


def is_higher_better(metric: str, *rows: dict) -> bool:
    """Whether bigger is better for `metric`, per the first row that declares it;
    otherwise by name heuristic (accuracy-ish names are higher-is-better)."""
    for row in rows:
        declared = row.get("higher_is_better")
        if declared is not None:
            return metric in declared
    m = metric.lower()
    return any(h in m for h in _HIB_HINTS)
