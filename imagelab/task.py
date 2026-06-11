"""
Task — THE extension seam. One class is the whole contract between your research code
and the lab: you own the model, the data, the loss, and the metrics; the trainer owns
the run lifecycle (run dir, ledger, tiers, cadences, checkpoints, crash bookkeeping).

A minimal project is two files in any directory — no packaging, no fork:

    # task.py
    from imagelab import Task, StepOutput

    class MyIdea(Task):
        name = "my_idea"
        overfit_gate = ("train/loss", "<=", 0.05)   # --tier overfit pass/fail — yours
        key_metrics = ["val/loss"]                  # what lab runs/board lead with

        def setup(self):            # build everything; read self.raw for your sections
            ...
        def step_fn(self, batch, step):
            return StepOutput(loss=..., logs={"train/loss": ...})

    # config.yaml
    task: task.py:MyIdea
    train: {batch_size: 32, epochs: 10, log_every: 50}
    # ... any sections you want; the framework only interprets train/out/wandb

then `lab run config.yaml --tier overfit`. Registered tasks (`@register("task", "name")`
+ `task: name` in YAML) work identically — that's the installed-package route.
"""

from __future__ import annotations

import torch

from imagelab.config import CoreConfig
from imagelab.loop import StepOutput  # noqa: F401  (re-export for task modules)


class Task:
    # ---- identity / wiring -------------------------------------------------- #
    name: str = ""                 # ledger display name (defaults to the config's task spec)
    paper: str | None = None       # method source (e.g. "arXiv:2006.11239") for `lab cite`
    gan: bool = False              # True -> trainer wraps generator/discriminator_fn in GANStep
    ckpt_prefix: str = "run"
    latest_name: str = "latest.pt"
    has_val: bool = False          # True -> trainer wires val_fn into the loop cadence

    # ---- editorial declarations: what counts as "the result" ----------------- #
    # YOURS TO TUNE — these encode research taste, not engineering necessity, and the
    # trainer copies them into every ledger row (so `lab runs/board/compare` can render
    # results without importing your model code).
    #
    # overfit_gate: ("metric", "<=" | ">=", threshold) — the `--tier overfit` kill
    # criterion ("can it memorize ONE sample?"). None -> verdict is "unknown".
    overfit_gate: tuple | None = None
    # key_metrics: ledger/leaderboard columns, in display order.
    key_metrics: list = []
    # metric names where bigger is better (everything else: lower is better).
    higher_is_better: frozenset = frozenset()

    def __init__(self, core: CoreConfig, raw: dict, args, device: str,
                 rng: torch.Generator):
        self.core = core           # validated framework sections (train/out)
        self.raw = raw             # the FULL config dict — your sections live here
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

    # ---- class-level hooks the trainer calls BEFORE instantiation ------------ #
    @classmethod
    def validate_config(cls, raw: dict) -> None:
        """Fail at config load, not at step 4000. Raise on invalid combinations."""

    @classmethod
    def apply_tier(cls, cfg: dict, tier: str) -> int | None:
        """Task-specific tier transforms, applied AFTER the generic ones (lab/tiers.py).
        Return a max_steps override for the tier, or None to keep the default."""
        return None

    @classmethod
    def add_args(cls, parser) -> None:
        """Declare task-specific CLI flags (e.g. cvq's --tokenizer_ckpt)."""

    @classmethod
    def cite_components(cls, cfg: dict) -> list:
        """(kind, name) pairs whose registered `paper=` references `lab cite` should
        print for this config. Override to map your config schema to your components."""
        return [("task", cls.name or str(cfg.get("task", "")))]

    @classmethod
    def resolved_config(cls, core: CoreConfig, raw: dict) -> dict:
        """What gets written to the run's config.yaml: the config as ACTUALLY used,
        defaults materialized. The default resolves the framework sections and passes
        your sections through verbatim; override if your task has its own schema."""
        import dataclasses
        out = {"task": raw.get("task")}
        out["train"] = {**dataclasses.asdict(core.train), **(raw.get("train") or {})}
        out["out"] = dataclasses.asdict(core.out)
        for k, v in raw.items():
            if k not in out:
                out[k] = v
        return out

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
        return self.core.train.grad_clip

    def param_counts(self) -> dict:
        """Per-component parameter counts (millions) for the run's fairness bookkeeping —
        an architecture swap that silently doubles the parameter budget should be visible
        right next to its metrics. The default reports every nn.Module attribute;
        override for sub-component detail."""
        import torch.nn as nn
        counts = {}
        for attr, mod in vars(self).items():
            if isinstance(mod, nn.Module):
                counts[attr] = round(sum(p.numel() for p in mod.parameters()) / 1e6, 3)
        return counts

    # ---- checkpointing ----
    def checkpoint_state(self) -> tuple[dict, dict, list[str]]:
        """(model_state, opt_state, latest_model_keys) for CheckpointStore.save."""
        raise NotImplementedError

    def ckpt_fn(self, step, epoch):
        model_state, opt_state, latest_keys = self.checkpoint_state()
        path = self.store.save(step, epoch, self.raw, model_state=model_state,
                               opt_state=opt_state, latest_model_keys=latest_keys)
        print(f"  saved {path.name}")

    def load_resume(self, ck: dict) -> tuple[int, int]:
        """Restore model/optimizer state from a resumable checkpoint.
        Returns (start_step, start_epoch)."""
        raise NotImplementedError

    def finalize(self, final_step: int):
        """Optional end-of-run work (e.g. a final full validation)."""
