"""
Checkpoint I/O — one adapter, three scripts.

Before this module, train.py / train_car.py / train_e2e.py each open-coded:
  * save format (`{tokenizer, disc, opt_g, opt_d, step, epoch, config}` etc.),
  * a stable `latest.pt` (model-only) pointer,
  * keep_last pruning of `*_step*.pt`,
  * (train.py only) best-tracking + wandb artifact upload.
That's three copies that drifted (train_e2e.py silently lost best-tracking).

`CheckpointStore` is the seam: callers say WHAT modules go in, the store handles HOW.

Conventions kept identical to the old scripts so existing checkpoints load:
  * save() writes `{prefix}_step{N:06d}.pt` with optimizer state (resumable),
    plus a model-only `{latest_name}.pt` pointer.
  * save_best() writes `best.pt` (model-only) when score improves; tracks best in memory
    OR re-reads from disk on construction so resumed runs don't lose their best.
  * keep_last prunes oldest `{prefix}_step*.pt`.
"""

from __future__ import annotations

from pathlib import Path

import torch


class CheckpointStore:
    """File-system-backed checkpoint store. One instance per training run.

    Args:
        ckpt_dir: directory to write into (created if missing).
        prefix: filename prefix for resumable checkpoints (e.g. "cvq", "car", "e2e").
        latest_name: name of the model-only pointer file (e.g. "latest.pt", "car_latest.pt").
        keep_last: keep this many of the most recent `{prefix}_step*.pt`. 0 = keep all.
    """

    def __init__(self, ckpt_dir: str | Path, prefix: str = "cvq",
                 latest_name: str = "latest.pt", keep_last: int = 5):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.latest_name = latest_name
        self.keep_last = keep_last
        # Resume-aware: pick up the prior best score from disk if it exists.
        self.best_score = float("inf")
        best_path = self.dir / "best.pt"
        if best_path.exists():
            try:
                prev = torch.load(best_path, map_location="cpu")
                if "score" in prev and isinstance(prev["score"], (int, float)):
                    self.best_score = float(prev["score"])
            except Exception:
                pass

    # --- save -------------------------------------------------------------- #
    def save(self, step: int, epoch: int, cfg: dict,
             model_state: dict, opt_state: dict | None = None,
             latest_model_keys: list[str] | None = None) -> Path:
        """Write a resumable `{prefix}_step{N}.pt` + a model-only `{latest_name}.pt` pointer.

        Args:
            model_state: dict of {key: state_dict} for every module to persist
                         (e.g. {"tokenizer": tok.state_dict(), "disc": disc.state_dict()}).
            opt_state:   dict of {key: state_dict} for optimizers / schedulers (resumable only).
            latest_model_keys: subset of model_state keys to mirror into `latest.pt`.
                               None => all model_state keys.
        """
        path = self.dir / f"{self.prefix}_step{step:06d}.pt"
        full = dict(model_state)
        if opt_state:
            full.update(opt_state)
        full.update({"step": step, "epoch": epoch, "config": cfg})
        torch.save(full, path)

        # model-only "latest" pointer (used by reconstruct/generate; not resumable)
        keys = latest_model_keys if latest_model_keys is not None else list(model_state.keys())
        latest = {k: model_state[k] for k in keys if k in model_state}
        latest.update({"config": cfg, "step": step})
        torch.save(latest, self.dir / self.latest_name)

        self._prune()
        return path

    def save_best(self, score: float, step: int, cfg: dict, model_state: dict,
                  lower_is_better: bool = True) -> bool:
        """Write `best.pt` when score improves. Returns True if a new best was written."""
        better = score < self.best_score if lower_is_better else score > self.best_score
        if not better:
            return False
        self.best_score = score
        payload = dict(model_state)
        payload.update({"config": cfg, "step": step, "score": score})
        torch.save(payload, self.dir / "best.pt")
        return True

    # --- load -------------------------------------------------------------- #
    @staticmethod
    def load(path: str | Path, map_location="cpu") -> dict:
        return torch.load(str(path), map_location=map_location)

    # --- helpers ----------------------------------------------------------- #
    def _prune(self):
        if not self.keep_last:
            return
        cks = sorted(self.dir.glob(f"{self.prefix}_step*.pt"))
        for old in cks[: -self.keep_last]:
            old.unlink(missing_ok=True)

    def best_path(self) -> Path:
        return self.dir / "best.pt"

    def latest_path(self) -> Path:
        return self.dir / self.latest_name
