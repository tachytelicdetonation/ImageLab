"""
Task — what varies between training runs. The trainer (cvq/trainer.py) owns the spine
(config, seed, device, checkpoint store, logger, resume, loop, final save); a Task
declares the models, the per-step losses, and the eval callbacks.

To add a new training recipe: subclass Task, decorate with @register("task", "myname"),
import it from cvq/tasks/__init__.py, and run `python -m cvq.trainer --task myname`.
"""

from __future__ import annotations

import torch

from cvq.config import RunConfig
from cvq.training_loop import StepOutput  # noqa: F401  (re-export for task modules)


class Task:
    gan: bool = False              # True -> trainer wraps generator/discriminator_fn in GANStep
    ckpt_prefix: str = "run"
    latest_name: str = "latest.pt"
    has_val: bool = False          # True -> trainer wires val_fn into the loop cadence

    def __init__(self, rc: RunConfig, args, device: str, rng: torch.Generator):
        self.rc = rc
        self.args = args
        self.device = device
        self.rng = rng
        # Injected by the trainer after setup(), before the loop:
        self.store = None          # CheckpointStore
        self.logger = None         # RunLogger
        # Tasks that run evals overwrite this each time; the trainer ships the final
        # value to the run dir + ledger (the run's "result" row).
        self.last_val_metrics: dict = {}
        # Populated by setup():
        self.dataloader = None
        self.optimizers: list = []
        self.schedulers: list = []
        self.gen_params: list = []
        self.disc_params = None    # GAN tasks set this
        self.disc = None           # GAN tasks set this

    # ------------------------------------------------------------------ #
    def setup(self):
        """Build datasets, models, losses, optimizers. Construction ORDER inside setup
        must be stable: it determines the global-RNG stream, and a fixed seed should keep
        producing the same initialization run over run."""
        raise NotImplementedError

    # ---- per-step work (gan=False tasks implement step_fn; gan=True the other two) ----
    def step_fn(self, batch, step) -> StepOutput:
        raise NotImplementedError

    def generator_fn(self, batch, step) -> StepOutput:
        raise NotImplementedError

    def discriminator_fn(self, batch, step, extras):
        raise NotImplementedError

    # ---- cadenced callbacks ----
    def sample_fn(self, step):
        return None

    def val_fn(self, step):
        return {}, {}

    def grad_clip(self) -> float | None:
        return self.rc.train.grad_clip

    def param_counts(self) -> dict:
        """Per-component parameter counts (millions) for the run's fairness bookkeeping —
        an architecture swap that silently doubles the parameter budget should be visible
        right next to its metrics. Components are found by their conventional attribute
        names; subclasses with exotic structure can override."""
        import torch.nn as nn
        counts = {}
        for attr in ("tok", "car", "disc", "dino"):
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Module):
                n = sum(p.numel() for p in mod.parameters())
                counts[attr] = round(n / 1e6, 3)
        if isinstance(getattr(self, "tok", None), nn.Module):
            for sub in ("encoder", "quantizer", "decoder"):
                m = getattr(self.tok, sub, None)
                if isinstance(m, nn.Module):
                    counts[f"tok.{sub}"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)
        return counts

    # ---- checkpointing ----
    def checkpoint_state(self) -> tuple[dict, dict, list[str]]:
        """(model_state, opt_state, latest_model_keys) for CheckpointStore.save."""
        raise NotImplementedError

    def ckpt_fn(self, step, epoch):
        model_state, opt_state, latest_keys = self.checkpoint_state()
        path = self.store.save(step, epoch, self.rc.raw, model_state=model_state,
                               opt_state=opt_state, latest_model_keys=latest_keys)
        print(f"  saved {path.name}")

    def load_resume(self, ck: dict) -> tuple[int, int]:
        """Restore model/optimizer state from a resumable checkpoint.
        Returns (start_step, start_epoch)."""
        raise NotImplementedError

    def finalize(self, final_step: int):
        """Optional end-of-run work (e.g. a final full validation)."""
