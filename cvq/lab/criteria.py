"""
What counts as "the result" of a run — the lab's editorial choices, in one file.

YOURS TO TUNE: these constants encode research taste, not engineering necessity.
KEY_METRICS decides which columns `lab runs`/`lab board` show and what `lab compare`
leads with; OVERFIT_* decides when `--tier overfit` says a change is broken. When your
eye starts going to a different number, change it here and every view follows.
"""

from __future__ import annotations

# Shown (when present) by `lab runs` / `lab board` / `lab compare`, in this order.
# First entry that exists is the default leaderboard sort (lower_is_better below).
KEY_METRICS = [
    "val/rFID",
    "val/recon_l2_full",
    "val/lpips",
    "eval/combined/token_acc",
    "eval/all/token_acc",
    "eval/combined/recon_l2",
]

# Metrics where bigger is better (everything else is treated as lower-is-better).
HIGHER_IS_BETTER = {"eval/combined/token_acc", "eval/all/token_acc",
                    "eval/combined/bit_acc", "eval/all/bit_acc",
                    "val/PSNR", "val/SSIM", "codebook/usage"}

# --- the overfit kill gate (--tier overfit) --------------------------------- #
# "Can it memorize ONE image?" — per-task: (metric to read, must-be-below threshold).
# A healthy tokenizer drives recon L2 on a single image to ~1e-3 within ~300 steps;
# an AR head should hit >90% token accuracy on one image's fixed token sequence.
OVERFIT_GATES = {
    "tokenizer": ("val/recon_l2_full", 0.01),
    "e2e": ("val/recon_l2_full", 0.01),
    "car": ("car/token_acc", 0.90),     # higher-is-better, handled below
}
_OVERFIT_HIGHER = {"car/token_acc"}


def overfit_verdict(task: str, metrics: dict) -> tuple[str, str]:
    """('pass'|'fail'|'unknown', human explanation) for an overfit-tier run."""
    gate = OVERFIT_GATES.get(task)
    if gate is None:
        return "unknown", f"no overfit gate defined for task '{task}' (see lab/criteria.py)"
    metric, threshold = gate
    v = metrics.get(metric)
    if v is None:
        return "unknown", f"gate metric '{metric}' missing from final metrics"
    if metric in _OVERFIT_HIGHER:
        ok = v >= threshold
        cmp = f"{v:.4f} {'>=' if ok else '<'} {threshold}"
    else:
        ok = v <= threshold
        cmp = f"{v:.4f} {'<=' if ok else '>'} {threshold}"
    verdict = "pass" if ok else "fail"
    note = (f"{metric} {cmp} — "
            + ("memorized; the gradient path works" if ok else
               "could NOT memorize a single image; suspect gradient flow / dead codes / "
               "loss wiring before burning a real run"))
    return verdict, note


def key_metrics_of(row_metrics: dict, n: int = 3) -> dict:
    """The first n KEY_METRICS present in a run's final metrics (ledger display)."""
    out = {}
    for k in KEY_METRICS:
        if k in row_metrics:
            out[k] = row_metrics[k]
            if len(out) >= n:
                break
    return out
