"""DDPM (Ho et al. 2020, arXiv:2006.11239) — the "very early and easy" diffusion
baseline, as an imagelab task: ~150 lines for schedule + training + sampling.

The full task contract in one read:
  * setup()      — data (ImageNette via the manifest battery), model, EMA, optimizer
  * step_fn()    — the whole objective: MSE(eps_hat, eps) at a random timestep
  * val_fn()     — deterministic held-out noise-MSE (FIXED (t, eps) pairs, so the number
                   is comparable across runs and seeds — not a new dice roll each eval)
  * sample_fn()  — 50-step DDIM grid into the run dir
  * overfit_gate — the `--tier overfit` kill criterion, declared on the class

Run it:    lab run examples/ddpm/config.yaml --tier smoke|overfit|fast
Compare:   lab compare <this> <rectified-flow-run>   # same ledger, same val protocol
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from imagelab import StepOutput, Task
from imagelab.data import ManifestImageDataset, OverfitDataset
from imagelab.loop import split_decay_groups, warmup_lr_lambda
from imagelab.utils import denorm

from unet import TinyUNet  # sibling file in this directory — any nn.Module works


class DDPMTask(Task):
    name = "ddpm"
    paper = "arXiv:2006.11239"
    ckpt_prefix = "ddpm"
    latest_name = "ddpm_latest.pt"
    has_val = True
    # YOURS TO TUNE — a healthy UNet memorizes one image's noise field fast; if this
    # gate fails, suspect the time conditioning or the schedule before the architecture.
    overfit_gate = ("val/noise_mse", "<=", 0.05)
    key_metrics = ["val/noise_mse", "val/noise_mse_ema"]

    @classmethod
    def apply_tier(cls, cfg, tier):
        if tier != "overfit":
            return None
        # Diffusion fits one image slower than a reconstruction objective (each step
        # sees fresh (t, eps) draws), so 600 probe steps instead of the default 300 —
        # and a clamped batch: 16 copies of one image per step probe just as well.
        t = cfg.setdefault("train", {})
        t["batch_size"] = min(t.get("batch_size", 16), 16)
        return 600

    def setup(self):
        t, d = self.core.train, self.raw.get("data", {})
        m = self.raw.get("model", {})
        root, size = d.get("root", "data/imagenette"), int(d.get("size", 64))
        hflip = bool(d.get("hflip", True))
        val_fraction = float(d.get("val_fraction", 0.1))

        ds = ManifestImageDataset(root, size=size, hflip=hflip, split="train",
                                  val_fraction=val_fraction)
        if t.overfit_n:
            ds = OverfitDataset(ds, t.overfit_n)
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True)
        # Fixed eval batch: first N HELD-OUT images in manifest order, so grids and
        # val numbers line up across runs. In overfit mode "val" IS the clamped image.
        val_ds = ds if t.overfit_n else ManifestImageDataset(
            root, size=size, hflip=False, split="val", val_fraction=val_fraction)
        n_val = min(8, len(val_ds))
        self.x_val = torch.stack([val_ds[i]["image"] for i in range(n_val)]).to(self.device)

        # ---- the DDPM schedule (linear betas, eps-prediction) ----
        self.T = int(m.get("timesteps", 1000))
        betas = torch.linspace(float(m.get("beta_start", 1e-4)),
                               float(m.get("beta_end", 0.02)), self.T)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(self.device)
        self.sample_steps = int(m.get("sample_steps", 50))

        self.model = TinyUNet(ch=int(m.get("ch", 64)),
                              ch_mult=tuple(m.get("ch_mult", [1, 2, 2])),
                              num_res_blocks=int(m.get("num_res_blocks", 2))).to(self.device)
        # EMA weights — what diffusion papers actually evaluate/sample from.
        self.ema_decay = float(m.get("ema_decay", 0.999))
        self.ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        tr = self.raw.get("train", {})
        lr = float(tr.get("lr", 2e-4))
        groups = split_decay_groups(list(self.model.parameters()), lr,
                                    float(tr.get("weight_decay", 0.0)))
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.99))
        warmup = int(tr.get("warmup_steps", 0))
        self.optimizers = [opt]
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_lr_lambda(s, warmup))]
        self.gen_params = [p for g in groups for p in g["params"]]

        # FIXED val noise: the same (t, eps) pairs every eval, every run (constant seed
        # on purpose — train.seed must not change what "val loss" means).
        g = torch.Generator().manual_seed(1234)
        k = 4                                            # (t, eps) pairs per val image
        n = self.x_val.shape[0] * k
        self.t_val = torch.randint(0, self.T, (n,), generator=g).to(self.device)
        self.eps_val = torch.randn(n, *self.x_val.shape[1:], generator=g).to(self.device)
        self.noise_sample = torch.randn(n_val, *self.x_val.shape[1:], generator=g
                                        ).to(self.device)

    # ------------------------------------------------------------------ #
    def _q_sample(self, x0, t, eps):
        ac = self.alphas_cumprod[t][:, None, None, None]
        return ac.sqrt() * x0 + (1 - ac).sqrt() * eps

    def step_fn(self, batch, step) -> StepOutput:
        self._ema_update()           # tracks weights as of the previous optimizer step
        x0 = batch["image"].to(self.device)
        t = torch.randint(0, self.T, (x0.shape[0],), device=self.device)
        eps = torch.randn_like(x0)
        loss = F.mse_loss(self.model(self._q_sample(x0, t, eps), t), eps)
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
    def _noise_mse(self, model) -> float:
        k = self.t_val.shape[0] // self.x_val.shape[0]
        x0 = self.x_val.repeat_interleave(k, dim=0)
        xt = self._q_sample(x0, self.t_val, self.eps_val)
        return F.mse_loss(model(xt, self.t_val), self.eps_val).item()

    def val_fn(self, step):
        self.model.eval()
        online = self._noise_mse(self.model)
        live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema)
        ema = self._noise_mse(self.model)
        self.model.load_state_dict(live)
        self.model.train()
        # Gate on the ONLINE metric: at probe scale (a few hundred steps) EMA@0.999
        # still lags far behind — it would test the averaging horizon, not the model.
        metrics = {"val/noise_mse": online, "val/noise_mse_ema": ema}
        self.last_val_metrics = metrics
        return metrics, {}

    @torch.no_grad()
    def _ddim_sample(self, n_steps: int) -> torch.Tensor:
        """Deterministic DDIM (eta=0) over an evenly strided subset of the T steps."""
        self.model.eval()
        x = self.noise_sample.clone()
        taus = torch.linspace(self.T - 1, 0, n_steps).long().to(self.device)
        for i, t in enumerate(taus):
            tb = t.repeat(x.shape[0])
            eps = self.model(x, tb)
            ac = self.alphas_cumprod[t]
            x0 = ((x - (1 - ac).sqrt() * eps) / ac.sqrt()).clamp(-1, 1)
            ac_prev = self.alphas_cumprod[taus[i + 1]] if i + 1 < n_steps \
                else torch.tensor(1.0, device=self.device)
            x = ac_prev.sqrt() * x0 + (1 - ac_prev).sqrt() * eps
        self.model.train()
        return x

    def sample_fn(self, step):
        x = self._ddim_sample(self.sample_steps)
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
