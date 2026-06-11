"""
`lab new <kind> <name>` — idea to runnable experiment in one command.

The framework ships the "task" scaffold: a self-contained project directory
(<name>/task.py + <name>/config.yaml) whose WORKING toy objective passes
`--tier smoke` and `--tier overfit` as-is — replace the model/data/loss, keep the seams.

Projects register scaffolds for their own component kinds (cvq registers "quantizer"
and "ar_head") via modules listed under [tool.imagelab] imports:

    from imagelab.lab.scaffold import register_scaffold

    @register_scaffold("quantizer")
    def new_quantizer(name: str) -> int: ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

SCAFFOLDS: dict[str, Callable[[str], int]] = {}


def register_scaffold(kind: str):
    def deco(fn):
        SCAFFOLDS[kind] = fn
        return fn
    return deco


def scaffold(kind: str, name: str) -> int:
    fn = SCAFFOLDS.get(kind)
    if fn is None:
        print(f"no scaffold for kind {kind!r} (have: {sorted(SCAFFOLDS)})")
        return 1
    if not name.isidentifier():
        print(f"{name!r} is not a valid python identifier")
        return 1
    return fn(name)


# --------------------------------------------------------------------------- #
TASK_T = '''"""{name}: an imagelab task — you own model/data/loss/metrics, the trainer
owns the run lifecycle (run dir, ledger, tiers, cadences, checkpoints).

This scaffold is a WORKING toy: a small conv autoencoder memorizing seeded random
images, so `lab run {name}/config.yaml --tier smoke` and `--tier overfit` pass before
you write a line. Replace the data and the objective; keep the seams.

Construction ORDER in setup() is RNG-load-bearing — keep it stable so seeds reproduce.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from imagelab import StepOutput, Task
from imagelab.loop import split_decay_groups, warmup_lr_lambda


class ToyImages(Dataset):
    """Stand-in data: N fixed random images. Swap for your dataset
    (imagelab.data.ManifestImageDataset handles the manifest layout)."""

    def __init__(self, n: int = 64, size: int = 32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, size, size, generator=g) * 2 - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {{"image": self.x[i]}}


class {cls}(Task):
    name = "{name}"
    ckpt_prefix = "{name}"
    # YOURS TO TUNE — the --tier overfit kill criterion and ledger columns:
    overfit_gate = ("train/loss", "<=", 0.02)
    key_metrics = ["train/loss"]

    def setup(self):
        t, m = self.core.train, self.raw.get("model", {{}})
        ds = ToyImages(size=int(m.get("size", 32)), seed=t.seed)
        if t.overfit_n:                       # --tier overfit clamps to 1 sample
            ds.x = ds.x[: t.overfit_n].repeat(256 // t.overfit_n + 1, 1, 1, 1)[:256]
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True)
        ch = int(m.get("ch", 32))
        self.model = nn.Sequential(            # TODO: your model here
            nn.Conv2d(3, ch, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Tanh(),
        ).to(self.device)
        lr = float(self.raw.get("train", {{}}).get("lr", 1e-3))
        groups = split_decay_groups(list(self.model.parameters()), lr, 1e-4)
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        warmup = int(self.raw.get("train", {{}}).get("warmup_steps", 0))
        self.optimizers = [opt]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_lr_lambda(s, warmup))]
        self.gen_params = [p for g in groups for p in g["params"]]

    def step_fn(self, batch, step) -> StepOutput:
        x = batch["image"].to(self.device)
        loss = F.mse_loss(self.model(x), x)    # TODO: your objective here
        return StepOutput(loss=loss, logs={{"train/loss": loss.item()}})

    def checkpoint_state(self):
        return ({{"model": self.model.state_dict()}},
                {{"opt": self.optimizers[0].state_dict()}}, ["model"])

    def load_resume(self, ck):
        self.model.load_state_dict(ck["model"])
        self.optimizers[0].load_state_dict(ck["opt"])
        return ck["step"], 0
'''

CONFIG_T = """task: task.py:{cls}

model:
  ch: 32
  size: 32

train:
  batch_size: 16
  epochs: 1000          # max_steps from the tier is the real stop for probe runs
  lr: 1e-3
  warmup_steps: 0
  log_every: 10
  seed: 0

wandb:
  enabled: false
"""


@register_scaffold("task")
def _new_task(name: str) -> int:
    dest = Path(name)
    if dest.exists():
        print(f"{dest}/ already exists — refusing to overwrite")
        return 1
    dest.mkdir(parents=True)
    cls = "".join(w.capitalize() for w in name.split("_")) or name.upper()
    (dest / "task.py").write_text(TASK_T.format(name=name, cls=cls))
    (dest / "config.yaml").write_text(CONFIG_T.format(cls=cls))
    print(f"wrote   {dest}/task.py")
    print(f"wrote   {dest}/config.yaml")
    print("\nnext:")
    print(f"  1. lab run {name}/config.yaml --tier smoke      # executes? (~30s)")
    print(f"  2. lab run {name}/config.yaml --tier overfit    # learns? (pass/fail)")
    print(f"  3. make it yours: model/data/loss in {name}/task.py, "
          f"gate + key_metrics on the class")
    return 0
