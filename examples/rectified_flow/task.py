"""Rectified flow (Liu et al. 2022, arXiv:2209.03003) — the SD3-era formulation, and
deliberately the SIMPLEST possible generative baseline: no beta schedule, no posterior
algebra. Interpolate x_t = (1-t)*x0 + t*eps, regress the constant velocity v = eps - x0,
sample by integrating dx/dt = -v_hat from t=1 (noise) to t=0 (data) with Euler steps.

Diff this file against examples/ddpm/task.py: same UNet, same data, same val protocol —
the model families differ by ~40 lines of objective + sampler. `lab compare` a run of
each to see two families side by side in one ledger.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from imagelab import StepOutput, Task
from imagelab.data import ManifestImageDataset, OverfitDataset
from imagelab.loop import split_decay_groups, warmup_lr_lambda
from imagelab.utils import denorm

# Reuse the DDPM example's backbone — in your own project this is just your model import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ddpm"))
from unet import TinyUNet  # noqa: E402


class RectifiedFlowTask(Task):
    name = "rectified_flow"
    paper = "arXiv:2209.03003"
    ckpt_prefix = "rf"
    latest_name = "rf_latest.pt"
    has_val = True
    # YOURS TO TUNE — same protocol as ddpm's gate (fixed (t, eps) pairs, online weights).
    overfit_gate = ("val/v_mse", "<=", 0.05)
    key_metrics = ["val/v_mse", "val/v_mse_ema"]

    @classmethod
    def apply_tier(cls, cfg, tier):
        if tier != "overfit":
            return None
        t = cfg.setdefault("train", {})              # same reasoning as ddpm's hook
        t["batch_size"] = min(t.get("batch_size", 16), 16)
        return 600

    def setup(self):
        t, d = self.core.train, self.raw.get("data", {})
        m = self.raw.get("model", {})
        root, size = d.get("root", "data/imagenette"), int(d.get("size", 64))
        val_fraction = float(d.get("val_fraction", 0.1))

        ds = ManifestImageDataset(root, size=size, hflip=bool(d.get("hflip", True)),
                                  split="train", val_fraction=val_fraction)
        if t.overfit_n:
            ds = OverfitDataset(ds, t.overfit_n)
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True)
        val_ds = ds if t.overfit_n else ManifestImageDataset(
            root, size=size, hflip=False, split="val", val_fraction=val_fraction)
        n_val = min(8, len(val_ds))
        self.x_val = torch.stack([val_ds[i]["image"] for i in range(n_val)]).to(self.device)

        self.sample_steps = int(m.get("sample_steps", 50))
        self.model = TinyUNet(ch=int(m.get("ch", 64)),
                              ch_mult=tuple(m.get("ch_mult", [1, 2, 2])),
                              num_res_blocks=int(m.get("num_res_blocks", 2))).to(self.device)
        self.ema_decay = float(m.get("ema_decay", 0.999))
        self.ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        tr = self.raw.get("train", {})
        groups = split_decay_groups(list(self.model.parameters()),
                                    float(tr.get("lr", 2e-4)),
                                    float(tr.get("weight_decay", 0.0)))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
        self.optimizers = [opt]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_lr_lambda(s, int(tr.get("warmup_steps", 0))))]
        self.gen_params = [p for g in groups for p in g["params"]]

        g = torch.Generator().manual_seed(1234)          # constant on purpose: val must
        k = 4                                            # mean the same thing every run
        n = self.x_val.shape[0] * k
        self.t_val = torch.rand(n, generator=g).to(self.device)
        self.eps_val = torch.randn(n, *self.x_val.shape[1:], generator=g).to(self.device)
        self.noise_sample = torch.randn(n_val, *self.x_val.shape[1:], generator=g
                                        ).to(self.device)

    # ------------------------------------------------------------------ #
    # The entire method: straight-line interpolant, constant-velocity target.
    def _v_loss(self, x0, t, eps):
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * eps
        v_hat = self.model(xt, t * 1000.0)   # scale t into the embedding range ddpm uses
        return F.mse_loss(v_hat, eps - x0)

    def step_fn(self, batch, step) -> StepOutput:
        self._ema_update()
        x0 = batch["image"].to(self.device)
        t = torch.rand(x0.shape[0], device=self.device)
        loss = self._v_loss(x0, t, torch.randn_like(x0))
        return StepOutput(loss=loss, logs={"train/loss": loss.item()})

    @torch.no_grad()
    def _ema_update(self):
        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
            else:
                self.ema[k].copy_(v)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _v_mse(self) -> float:
        k = self.t_val.shape[0] // self.x_val.shape[0]
        x0 = self.x_val.repeat_interleave(k, dim=0)
        t = self.t_val
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * self.eps_val
        return F.mse_loss(self.model(xt, t * 1000.0), self.eps_val - x0).item()

    def val_fn(self, step):
        self.model.eval()
        online = self._v_mse()
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema)
        ema = self._v_mse()
        self.model.load_state_dict(live)
        self.model.train()
        metrics = {"val/v_mse": online, "val/v_mse_ema": ema}
        self.last_val_metrics = metrics
        return metrics, {}

    @torch.no_grad()
    def _euler_sample(self, n_steps: int) -> torch.Tensor:
        """Integrate dx/dt = v_hat from t=1 (noise) to t=0 (data). That's the sampler."""
        self.model.eval()
        x = self.noise_sample.clone()
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), 1.0 - i * dt, device=self.device)
            x = x - self.model(x, t * 1000.0) * dt
        self.model.train()
        return x.clamp(-1, 1)

    def sample_fn(self, step):
        x = self._euler_sample(self.sample_steps)
        path = Path(self.core.out.sample_dir) / f"gen_{step:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(make_grid(denorm(x), nrow=4), path)
        return {"gen": make_grid(denorm(x), nrow=4)}

    def finalize(self, final_step: int):
        metrics, _ = self.val_fn(final_step)
        if self.logger:
            self.logger.log(metrics, final_step)
        print("final: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

    # ------------------------------------------------------------------ #
    def checkpoint_state(self):
        return ({"model": self.model.state_dict(), "ema": self.ema},
                {"opt": self.optimizers[0].state_dict()}, ["model", "ema"])

    def load_resume(self, ck):
        self.model.load_state_dict(ck["model"])
        self.ema = {k: v.to(self.device) for k, v in ck["ema"].items()}
        self.optimizers[0].load_state_dict(ck["opt"])
        return ck["step"], 0
