"""
Core config — the ONLY sections the framework interprets: train (loop mechanics),
out (artifact locations), wandb (mirror logging; read as a raw dict by RunLogger).

Everything else in the YAML belongs to the task. The framework never warns about keys
it doesn't know — they're someone else's schema. Tasks that want strict validation of
their own sections do it in `Task.validate_config(raw)` — build your typed config
there and fail at load time, not at step 4000.

Convention keys (interpreted by tiers/tasks, not the loop):
    train.overfit_n       >0: tasks clamp their dataset to the first N samples
    train.warmup_steps    tasks build their own schedulers but tiers clamp this
    data.hflip/augment    tiers disable these for --tier overfit
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


class ConfigError(ValueError):
    """A config that cannot produce a meaningful run. Raised at load time."""


@dataclass
class TrainCore:
    batch_size: int = 16
    epochs: int = 1
    device: str = "auto"
    amp: str = "none"                # none | bf16
    grad_accum: int = 1
    grad_clip: float | None = 1.0
    seed: int = 0
    overfit_n: int = 0               # >0: tasks clamp data to first N samples (tiers set 1)
    num_workers: int = 0
    disc_start_step: int = 0         # GAN tasks: delay the disc optimizer
    # ---- cadences ----
    log_every: int = 50
    sample_every: int = 500
    val_every: int = 1000
    ckpt_every: int = 1000


@dataclass
class OutCore:
    ckpt_dir: str = "checkpoints"    # trainer redirects into the run dir unless YAML-pinned
    sample_dir: str = "samples"
    run_dir: str | None = None       # tensorboard mirror location
    keep_last: int = 5


@dataclass
class CoreConfig:
    train: TrainCore
    out: OutCore
    raw: dict = field(default_factory=dict, repr=False)


def core_from_dict(cfg: dict) -> CoreConfig:
    """Extract + validate the framework sections. Unknown keys are SILENTLY left to the
    task (the opposite of a closed schema — this is what makes the trainer generic)."""
    core = CoreConfig(train=_pick(TrainCore, cfg.get("train")),
                      out=_pick(OutCore, cfg.get("out")), raw=cfg)
    t = core.train
    if t.amp not in ("none", "bf16"):
        raise ConfigError(f"train.amp={t.amp!r} — must be 'none' or 'bf16'")
    for k in ("batch_size", "epochs", "grad_accum"):
        if getattr(t, k) < 1:
            raise ConfigError(f"train.{k}={getattr(t, k)} must be >= 1")
    if t.overfit_n < 0:
        raise ConfigError(f"train.overfit_n={t.overfit_n} must be >= 0")
    for k in ("log_every", "sample_every", "val_every", "ckpt_every"):
        if getattr(t, k) < 1:
            raise ConfigError(f"train.{k} must be >= 1 (use a huge value to disable)")
    return core


def _pick(cls, d: dict | None):
    """Build a dataclass from the keys it knows, ignoring the rest."""
    d = d or {}
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})
