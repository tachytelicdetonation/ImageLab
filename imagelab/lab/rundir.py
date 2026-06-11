"""
RunDir — one experiment, one self-describing folder.

    runs/0610-1432_tok_inet_64/
      config.yaml      # RESOLVED config: every default materialized at run time
      meta.json        # provenance: git sha, seed, device, params, wall-clock, status
      metrics.jsonl    # scalar time series (appended by RunLogger every log_every)
      metrics.json     # final eval metrics
      samples/         # image grids
      checkpoints/     # resumable + latest/best checkpoints

plus a row in runs/ledger.jsonl (status running -> done|crashed). The folder is the
source of truth; wandb/tensorboard are optional mirrors. Readers take the LAST ledger
row per run_id, so appending an updated row "overwrites" without rewriting the file.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Resolved relative to the process cwd by convention (run `lab` from your project root);
# set IMAGELAB_RUNS_ROOT to pin an absolute location instead.
RUNS_ROOT = Path(os.environ.get("IMAGELAB_RUNS_ROOT", "runs"))


def _git_info() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return {"sha": sha or None, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


class RunDir:
    def __init__(self, dir: Path):
        self.dir = Path(dir)
        self.run_id = self.dir.name
        self._t0 = time.time()

    @classmethod
    def create(cls, name: str, root: Path | None = None) -> "RunDir":
        root = Path(root) if root else RUNS_ROOT
        stamp = datetime.now().strftime("%m%d-%H%M%S")
        d = root / f"{stamp}_{name}"
        n = 1
        while d.exists():  # same-second launches (sweeps)
            n += 1
            d = root / f"{stamp}_{name}-{n}"
        (d / "samples").mkdir(parents=True)
        (d / "checkpoints").mkdir()
        return cls(d)

    # ---- provenance ------------------------------------------------------- #
    def write_config(self, resolved: dict):
        """`resolved` is the config as ACTUALLY used, defaults materialized
        (Task.resolved_config builds it — tasks with their own schema override that)."""
        import yaml
        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False))

    def write_meta(self, **fields):
        """Merge fields into meta.json (start writes identity, end writes outcome)."""
        path = self.dir / "meta.json"
        meta = json.loads(path.read_text()) if path.exists() else {}
        meta.update(fields)
        path.write_text(json.dumps(meta, indent=2, default=str))
        return meta

    @property
    def root(self) -> Path:
        return self.dir.parent

    def start(self, *, name: str, task: str, config_path: str, tier: str,
              deltas: dict, resolved: dict, seed, device: str, amp: str,
              key_metrics: list | None = None, higher_is_better: list | None = None,
              argv=None):
        import torch
        self.write_config(resolved)
        self.write_meta(
            run_id=self.run_id, name=name, task=task, tier=tier,
            config_path=str(config_path), deltas=deltas,
            seed=seed, device=device, amp=amp,
            # The task's editorial declarations travel WITH the run, so the CLI can
            # render any model family's results without importing its code.
            key_metrics=key_metrics or [], higher_is_better=higher_is_better or [],
            torch=torch.__version__, git=_git_info(), argv=argv,
            started=datetime.now().isoformat(timespec="seconds"), status="running",
        )
        ledger_append(self._ledger_row({"status": "running"}), root=self.root)

    def finish(self, *, status: str, steps: int, final_metrics: dict | None = None,
               params: dict | None = None, error: str | None = None,
               verdict: str | None = None):
        wall = time.time() - self._t0
        fields = {
            "status": status, "steps": steps, "wall_min": round(wall / 60, 2),
            "steps_per_sec": round(steps / wall, 3) if steps else 0.0,
        }
        if params:
            fields["params"] = params
        if error:
            fields["error"] = error
        if verdict:
            fields["verdict"] = verdict
        if final_metrics:
            fields["final_metrics"] = _floats(final_metrics)
            (self.dir / "metrics.json").write_text(
                json.dumps(_floats(final_metrics), indent=2))
        self.write_meta(**fields)
        ledger_append(self._ledger_row(fields), root=self.root)

    def _ledger_row(self, extra: dict) -> dict:
        meta = json.loads((self.dir / "meta.json").read_text())
        keep = ("run_id", "name", "task", "tier", "deltas", "seed", "started", "status",
                "steps", "wall_min", "steps_per_sec", "params", "final_metrics",
                "verdict", "parent", "git", "key_metrics", "higher_is_better")
        row = {k: meta[k] for k in keep if k in meta}
        row.update(extra)
        return row


# --------------------------------------------------------------------------- #
# Ledger — append-only JSONL; the last row per run_id is the current state
# --------------------------------------------------------------------------- #
def ledger_path(root: Path | None = None) -> Path:
    return (Path(root) if root else RUNS_ROOT) / "ledger.jsonl"


def ledger_append(row: dict, root: Path | None = None):
    p = ledger_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def ledger_rows(root: Path | None = None) -> list[dict]:
    """Current state of every run, oldest first (last JSONL row per run_id wins)."""
    p = ledger_path(root)
    if not p.exists():
        return []
    by_id = {}
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue       # half-written row from a crash mid-append; skip, don't crash
        by_id[row.get("run_id")] = row
    return list(by_id.values())


def _floats(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        try:
            out[k] = round(float(v), 6)
        except (TypeError, ValueError):
            out[k] = v
    return out
