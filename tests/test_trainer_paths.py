"""Trainer-path tests the seam suite doesn't reach: the GAN step choreography,
resume-from-checkpoint, grad-accum window continuity, the lab CLI's rendering
commands, and the dotted task-resolution form. CPU-only, no network.

    uv run pytest tests/test_trainer_paths.py -q
"""

from __future__ import annotations

import pytest
import torch

# A complete gan=True project: conv generator + conv discriminator. The point is
# executing GANStep's freeze/backward dance and the disc-last optimizer convention,
# not learning anything.
GAN_TASK_PY = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from imagelab import StepOutput, Task


class Pixels(Dataset):
    def __init__(self, n=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 8, 8, generator=g) * 2 - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i]}


class TinyGANTask(Task):
    name = "tiny_gan"
    gan = True
    ckpt_prefix = "tgan"
    overfit_gate = ("train/g_loss", "<=", 0.1)
    key_metrics = ["train/g_loss"]

    def setup(self):
        t = self.core.train
        self.dataloader = DataLoader(Pixels(seed=t.seed), batch_size=t.batch_size,
                                     drop_last=True)
        self.gen = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(8, 3, 3, padding=1), nn.Tanh())
        self.disc = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(),
                                  nn.Conv2d(8, 1, 3, padding=1))
        g_opt = torch.optim.AdamW(self.gen.parameters(), lr=1e-3)
        d_opt = torch.optim.AdamW(self.disc.parameters(), lr=1e-3)
        self.optimizers = [g_opt, d_opt]               # disc opt LAST by convention
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lambda s: 1.0)
                           for o in self.optimizers]
        self.gen_params = list(self.gen.parameters())
        self.disc_params = list(self.disc.parameters())

    def generator_fn(self, batch, step):
        x = batch["image"]
        recon = self.gen(x)
        g_loss = F.mse_loss(recon, x) - self.disc(recon).mean() * 0.01
        return StepOutput(loss=g_loss, logs={"train/g_loss": g_loss.item()},
                          extras={"recon": recon})

    def discriminator_fn(self, batch, step, extras):
        real = self.disc(batch["image"]).mean()
        fake = self.disc(extras["recon"].detach()).mean()
        d_loss = fake - real
        return d_loss, {"train/d_loss": d_loss.item()}

    def checkpoint_state(self):
        return ({"gen": self.gen.state_dict(), "disc": self.disc.state_dict()},
                {"g_opt": self.optimizers[0].state_dict()}, ["gen"])

    def load_resume(self, ck):
        self.gen.load_state_dict(ck["gen"])
        self.disc.load_state_dict(ck["disc"])
        return ck["step"], 0
'''

GAN_CONFIG_YAML = """task: task.py:TinyGANTask
train:
  batch_size: 8
  epochs: 1000
  log_every: 4
  seed: 0
  disc_start_step: 2
wandb:
  enabled: false
"""

# Same TinyTask as test_framework.py, used here for resume + CLI fixtures.
TASK_PY = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from imagelab import StepOutput, Task


class Pixels(Dataset):
    def __init__(self, n=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 3, 8, 8, generator=g) * 2 - 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"image": self.x[i]}


class TinyTask(Task):
    name = "tiny"
    ckpt_prefix = "tiny"
    overfit_gate = ("train/loss", "<=", 0.05)
    key_metrics = ["train/loss"]

    def setup(self):
        t = self.core.train
        self.dataloader = DataLoader(Pixels(seed=t.seed), batch_size=t.batch_size,
                                     drop_last=True)
        self.model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
                                   nn.Conv2d(16, 3, 3, padding=1), nn.Tanh())
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-3)
        self.optimizers = [opt]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)]
        self.gen_params = list(self.model.parameters())

    def step_fn(self, batch, step):
        x = batch["image"]
        loss = F.mse_loss(self.model(x), x)
        return StepOutput(loss=loss, logs={"train/loss": loss.item()})

    def checkpoint_state(self):
        return ({"model": self.model.state_dict()},
                {"opt": self.optimizers[0].state_dict()}, ["model"])

    def load_resume(self, ck):
        self.model.load_state_dict(ck["model"])
        self.optimizers[0].load_state_dict(ck["opt"])
        return ck["step"], 0
'''

CONFIG_YAML = """task: task.py:TinyTask
train:
  batch_size: 8
  epochs: 1000
  log_every: 4
  ckpt_every: 6
  seed: 0
wandb:
  enabled: false
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    (tmp_path / "task.py").write_text(TASK_PY)
    (tmp_path / "config.yaml").write_text(CONFIG_YAML)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ledger(tmp_path):
    from imagelab.lab.rundir import ledger_rows
    return ledger_rows(tmp_path / "runs")


# --------------------------------------------------------------------------- #
# GAN path
# --------------------------------------------------------------------------- #
def test_gan_task_runs_smoke_end_to_end(tmp_path, monkeypatch):
    """gan=True must exercise GANStep: disc-frozen generator backward, fresh-graph disc
    backward, disc-opt-last stepping gated by disc_start_step — none of which the
    NoGANStep tests touch."""
    from imagelab.trainer import run
    (tmp_path / "task.py").write_text(GAN_TASK_PY)
    (tmp_path / "config.yaml").write_text(GAN_CONFIG_YAML)
    monkeypatch.chdir(tmp_path)
    final = run(argv=["--config", str(tmp_path / "config.yaml"), "--tier", "smoke"])
    assert final == 8
    row = _ledger(tmp_path)[0]
    assert row["status"] == "done" and row["task"] == "tiny_gan"
    # both networks were counted (param bookkeeping saw gen AND disc)
    assert {"gen", "disc"} <= set(row["params"])


# --------------------------------------------------------------------------- #
# resume round-trip
# --------------------------------------------------------------------------- #
def test_resume_round_trip(project):
    """Save at step 6 -> resume -> continue to 12. Also proves every object the store
    writes survives torch.load(weights_only=True) — the safe-load default."""
    from imagelab.trainer import run
    run(argv=["--config", str(project / "config.yaml"), "--tier", "smoke",
              "--name", "first"])
    first = [r for r in _ledger(project) if "first" in r["run_id"]][0]
    cks = sorted((project / "runs" / first["run_id"] / "checkpoints").glob("tiny_step*.pt"))
    assert cks, "smoke run wrote no resumable checkpoint"

    final = run(argv=["--config", str(project / "config.yaml"),
                      "--resume", str(cks[-1]), "--max-steps", "12", "--name", "resumed"])
    assert final == 12
    resumed = [r for r in _ledger(project) if "resumed" in r["run_id"]][0]
    assert resumed["status"] == "done"


def test_checkpoint_store_load_is_weights_only(tmp_path):
    """A checkpoint smuggling a pickled object must be REFUSED by the default load."""
    import argparse
    from imagelab.checkpoint import CheckpointStore

    p = tmp_path / "evil.pt"
    # argparse.Namespace pickles fine but is NOT on torch's weights_only allowlist —
    # a stand-in for any pickled object that could carry a __reduce__ payload.
    torch.save({"model": {}, "obj": argparse.Namespace(x=1)}, p)
    with pytest.raises(Exception):
        CheckpointStore.load(p)
    # explicit opt-out still available for trusted files
    assert "obj" in CheckpointStore.load(p, weights_only=False)


# --------------------------------------------------------------------------- #
# grad-accum window continuity
# --------------------------------------------------------------------------- #
def test_accum_windows_do_not_reset_at_epoch_boundaries(tmp_path):
    """3 batches/epoch x 2 epochs with accum=2 = 6 micro-batches = 3 optimizer steps.
    The old per-epoch counter silently dropped the trailing micro-batch of every epoch
    (2 steps, 2 discarded gradients)."""
    from imagelab.loop import Cadence, NoGANStep, RunLogger, StepOutput, TrainLoop

    model = torch.nn.Linear(4, 4)
    steps = {"n": 0}

    class CountingSGD(torch.optim.SGD):
        def step(self, *a, **kw):
            steps["n"] += 1
            return super().step(*a, **kw)

    opt = CountingSGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    def step_fn(batch, step):
        loss = model(batch).pow(2).mean()
        return StepOutput(loss=loss, logs={"train/loss": loss.item()})

    loop = TrainLoop(device="cpu", amp="none", accum=2,
                     cadence=Cadence(log_every=100, sample_every=10**9,
                                     val_every=10**9, ckpt_every=10**9),
                     logger=RunLogger({"wandb": {"enabled": False}}),
                     batch_size=2)
    data = [torch.randn(2, 4) for _ in range(3)]      # 3 batches: NOT divisible by accum
    final = loop.run(dataloader=data, epochs=2, start_step=0, start_epoch=0,
                     step_runner=NoGANStep(step_fn, accum=2, device="cpu", amp="none"),
                     optimizers=[opt], schedulers=[sched],
                     gen_params=list(model.parameters()))
    assert final == 6
    assert steps["n"] == 3


# --------------------------------------------------------------------------- #
# lab CLI rendering commands (against a real ledger written by real runs)
# --------------------------------------------------------------------------- #
def test_cli_runs_board_compare_report(project, capsys):
    from imagelab.lab.cli import main
    from imagelab.trainer import run
    run(argv=["--config", str(project / "config.yaml"), "--tier", "smoke",
              "--name", "runa"])
    run(argv=["--config", str(project / "config.yaml"), "--tier", "smoke",
              "--name", "runb", "--set", "train.batch_size=4"])
    rows = _ledger(project)
    ida, idb = rows[0]["run_id"], rows[1]["run_id"]
    capsys.readouterr()

    assert main(["runs"]) == 0
    out = capsys.readouterr().out
    assert ida in out and idb in out and "tiny" in out

    assert main(["board"]) == 0
    assert "leaderboard by train/loss" in capsys.readouterr().out

    assert main(["compare", ida, idb]) == 0
    out = capsys.readouterr().out
    assert "config delta" in out and "batch_size" in out    # the --set shows up

    assert main(["report", "--format", "md"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("| run | steps") and "loss" in out.splitlines()[0]


def test_cli_run_is_documented_and_forwards(project, capsys):
    """`lab run` must appear in --help (it used to be invisible) and forward to the
    trainer with the bare-config shorthand."""
    from imagelab.lab.cli import main
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "run" in capsys.readouterr().out
    assert main(["run", str(project / "config.yaml"), "--tier", "smoke"]) == 0
    assert _ledger(project)[-1]["status"] == "done"


# --------------------------------------------------------------------------- #
# task resolution: the dotted module:Class form
# --------------------------------------------------------------------------- #
def test_resolve_task_dotted_module_form():
    from imagelab.task import Task
    from imagelab.trainer import resolve_task
    assert resolve_task("imagelab.task:Task") is Task
    with pytest.raises(SystemExit, match="no attribute"):
        resolve_task("imagelab.task:Nope")
    with pytest.raises(SystemExit, match="not importable"):
        resolve_task("no.such.module:Cls")


# --------------------------------------------------------------------------- #
# the store/logger footgun has a helpful error
# --------------------------------------------------------------------------- #
def test_store_access_during_setup_raises_helpfully():
    from imagelab.task import Task

    t = Task.__new__(Task)
    Task.__init__(t, core=None, raw={}, args=None, device="cpu",
                  rng=torch.Generator())
    with pytest.raises(RuntimeError, match="setup\\(\\)"):
        _ = t.store
    with pytest.raises(RuntimeError, match="setup\\(\\)"):
        _ = t.logger
