"""Framework seam tests: the generic trainer must run a task it has never heard of —
defined in a plain directory, no registry, no packaging — through the full lifecycle
(tiers, run dir, ledger, overfit gate). No network, no datasets, CPU-only.

    uv run pytest tests/test_framework.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# A complete user project: one task file + one config. The task memorizes a fixed
# random image with a single conv — small enough to pass the overfit gate in ~100 steps.
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
        ds = Pixels(seed=t.seed)
        if t.overfit_n:
            ds.x = ds.x[: t.overfit_n].repeat(64, 1, 1, 1)[:64]
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, drop_last=True)
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
        return ck["step"], 0
'''

CONFIG_YAML = """task: task.py:TinyTask
train:
  batch_size: 8
  epochs: 1000
  log_every: 20
  seed: 0
wandb:
  enabled: false
"""


@pytest.fixture()
def project(tmp_path, monkeypatch):
    (tmp_path / "task.py").write_text(TASK_PY)
    (tmp_path / "config.yaml").write_text(CONFIG_YAML)
    monkeypatch.chdir(tmp_path)        # runs/ + the ledger land in the tmp project
    return tmp_path


def _ledger(tmp_path) -> list[dict]:
    from imagelab.lab.rundir import ledger_rows
    return ledger_rows(tmp_path / "runs")


def test_unknown_model_family_runs_end_to_end(project):
    """File-path task spec -> smoke tier -> self-describing run folder + ledger row."""
    from imagelab.trainer import run
    final = run(argv=["--config", str(project / "config.yaml"), "--tier", "smoke"])
    assert final == 8
    rows = _ledger(project)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "done" and row["task"] == "tiny"
    assert row["key_metrics"] == ["train/loss"]      # declaration traveled with the run
    rd = project / "runs" / row["run_id"]
    assert (rd / "config.yaml").exists() and (rd / "metrics.jsonl").exists()
    assert json.loads((rd / "meta.json").read_text())["tier"] == "smoke"
    assert list((rd / "checkpoints").glob("tiny_*.pt"))


def test_overfit_tier_gates_pass_and_fail(project):
    """The kill gate reads the TASK'S declaration: pass when it memorizes, fail when it
    can't (here: a probe too short to converge) — never a silent 'complete'."""
    from imagelab.trainer import run
    run(argv=["--config", str(project / "config.yaml"), "--tier", "overfit",
              "--max-steps", "150"])
    assert _ledger(project)[-1]["verdict"] == "pass"

    run(argv=["--config", str(project / "config.yaml"), "--tier", "overfit",
              "--max-steps", "5", "--name", "sabotaged"])
    sab = [r for r in _ledger(project) if "sabotaged" in r["run_id"]][0]
    assert sab["verdict"] == "fail"


def test_scaffolded_project_passes_smoke_as_written(project):
    """`lab new task x` must produce a WORKING project — scaffold-rot is a CI failure."""
    from imagelab.lab.scaffold import scaffold
    from imagelab.trainer import run
    assert scaffold("task", "my_idea") == 0
    final = run(argv=["--config", str(project / "my_idea" / "config.yaml"),
                      "--tier", "smoke"])
    assert final == 8
    assert _ledger(project)[-1]["status"] == "done"


def test_core_config_ignores_task_sections_but_validates_its_own():
    from imagelab.config import ConfigError, core_from_dict
    core = core_from_dict({"train": {"batch_size": 4, "epochs": 2, "lr": 1e-3,
                                     "my_custom_knob": 7},
                           "diffusion": {"timesteps": 1000}})
    assert core.train.batch_size == 4                # known keys picked up
    assert core.raw["diffusion"]["timesteps"] == 1000  # unknown sections untouched
    with pytest.raises(ConfigError, match="amp"):
        core_from_dict({"train": {"amp": "fp8"}})


def test_resolve_task_error_messages(tmp_path):
    from imagelab.trainer import resolve_task
    with pytest.raises(SystemExit, match="not found"):
        resolve_task("missing.py:Nope", tmp_path)
    with pytest.raises(SystemExit, match="unknown task"):
        resolve_task("never_registered_task_name", tmp_path)


def test_key_metrics_fallback_for_legacy_rows():
    """Rows written before tasks declared key_metrics still render sensibly."""
    from imagelab.lab.criteria import is_higher_better, key_metrics_of
    legacy = {"final_metrics": {"train/loss": 0.1, "val/rFID": 33.0, "eval/acc": 0.5}}
    assert list(key_metrics_of(legacy)) == ["val/rFID", "eval/acc"]   # val/ then eval/
    declared = {"final_metrics": {"a": 1, "b": 2}, "key_metrics": ["b", "a"]}
    assert list(key_metrics_of(declared)) == ["b", "a"]
    assert is_higher_better("val/acc", legacy)                  # name heuristic
    assert not is_higher_better("val/acc", {"higher_is_better": ["val/other"]})
