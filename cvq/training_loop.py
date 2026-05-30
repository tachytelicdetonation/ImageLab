"""
TrainLoop — the scaffolding three training scripts used to copy-paste.

Owns:
  * AMP autocast context, gradient accumulation boundary,
  * the `for p in disc.parameters(): p.requires_grad_(False/True)` GAN dance,
  * periodic-cadence dispatch (log_every / sample_every / val_every / ckpt_every),
  * wandb init + scalar/image logging (with TensorBoard mirror),
  * resume + final-step bookkeeping.

The variable part of each script is `step_fn(batch, step) -> StepOutput`. Recon-only,
NTP-only, joint-EOSTok all express their per-step work behind the same seam.

Two adapters at the GAN seam: GANStep wraps the generator/discriminator choreography so
recon-only and joint-E2E share it; NoGANStep is the same loop without the disc dance for
train_car (frozen tokenizer + CAR-only).
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except Exception:
    _HAS_TB = False


# --------------------------------------------------------------------------- #
# Optimizer helpers (moved from train.py / train_e2e.py — were duplicated)
# --------------------------------------------------------------------------- #
def split_decay_groups(params, lr: float, weight_decay: float):
    """tensors with ndim>=2 (conv/linear/embeddings) get WD; ndim<2 (norm γβ + biases) don't.

    Standard rule (GPT/ViT/timm): decaying a normalization's learnable scale fights the
    normalization, so norms+biases are excluded. Returns 1-2 optimizer groups.
    """
    params = [p for p in params if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def autocast_ctx(device: str, amp: str):
    """bf16 autocast on CUDA when amp=='bf16'; else fp32 no-op."""
    if amp == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def warmup_lr_lambda(step: int, warmup: int) -> float:
    return min(1.0, (step + 1) / max(1, warmup))


# --------------------------------------------------------------------------- #
# Per-step output: the contract between step_fn and the loop
# --------------------------------------------------------------------------- #
@dataclass
class StepOutput:
    """What a step_fn returns to the loop.

    loss: the scalar to backprop (generator-side loss in a GAN setup).
    logs: scalars to log every log_every steps.
    extras: pass-through dict (e.g. the disc loss for GAN logging; never backpropped here).
    """
    loss: torch.Tensor
    logs: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# wandb + tensorboard logger
# --------------------------------------------------------------------------- #
class RunLogger:
    """Thin wrapper around wandb + TB. Same call sites as the inline `wlog` closures."""

    def __init__(self, cfg: dict, run_dir: str | Path | None = None):
        wcfg = cfg.get("wandb", {}) or {}
        self.use_wandb = _HAS_WANDB and wcfg.get("enabled", False)
        if self.use_wandb:
            mode = wcfg.get("mode", "online")
            if mode == "online" and not os.environ.get("WANDB_API_KEY"):
                print("WANDB_API_KEY not set -> falling back to offline mode")
                mode = "offline"
            wandb.init(project=wcfg.get("project", "cvq-pokemon"),
                       entity=wcfg.get("entity"), name=wcfg.get("name"),
                       mode=mode, config=cfg)
            print(f"wandb: logging to project '{wcfg.get('project', 'cvq-pokemon')}' (mode={mode})")
        self.wcfg = wcfg
        self.writer = None
        if _HAS_TB and run_dir:
            self.writer = SummaryWriter(str(run_dir))

    def log(self, d: dict, step: int):
        if self.writer is not None:
            for k, v in d.items():
                try:
                    self.writer.add_scalar(k, float(v), step)
                except Exception:
                    pass
        if self.use_wandb:
            wandb.log(d, step=step)

    def log_images(self, imgs: dict, step: int):
        if self.use_wandb:
            wandb.log({f"images/{k}": wandb.Image(v) for k, v in imgs.items()}, step=step)

    def log_artifact(self, path: str | Path, name: str, kind: str, step: int, aliases: list[str]):
        if self.use_wandb and self.wcfg.get("log_checkpoints", True):
            art = wandb.Artifact(name, type=kind, metadata={"step": step})
            art.add_file(str(path))
            wandb.log_artifact(art, aliases=aliases)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            wandb.finish()


# --------------------------------------------------------------------------- #
# GAN step adapter — wraps the (generator-fwd, disc-freeze dance, disc-fwd) choreography
# --------------------------------------------------------------------------- #
class GANStep:
    """The generator+discriminator backward choreography, behind a single `__call__`.

    The caller provides:
      * generator_fn(batch, step) -> StepOutput  (the loss to backprop into the generator)
      * discriminator_fn(batch, step, extras) -> (d_loss: Tensor, d_logs: dict)
    The loop handles:
      * freezing disc params during generator backward (so the GAN gradient flows THROUGH the
        disc to the decoder but doesn't deposit gradient INTO the disc),
      * dividing by `accum` for grad-accum scaling,
      * fresh-graph disc backward (recon is detached inside discriminator_fn).
    """

    def __init__(self, discriminator: torch.nn.Module, generator_fn, discriminator_fn,
                 accum: int, device: str, amp: str):
        self.disc = discriminator
        self.gen_fn = generator_fn
        self.disc_fn = discriminator_fn
        self.accum = accum
        self.device = device
        self.amp = amp

    def __call__(self, batch, step):
        # ---- generator forward+backward (disc frozen) ----
        for p in self.disc.parameters():
            p.requires_grad_(False)
        with autocast_ctx(self.device, self.amp):
            out = self.gen_fn(batch, step)
        (out.loss / self.accum).backward()
        for p in self.disc.parameters():
            p.requires_grad_(True)

        # ---- discriminator backward (fresh graph, detached recon inside disc_fn) ----
        with autocast_ctx(self.device, self.amp):
            d_loss, d_logs = self.disc_fn(batch, step, out.extras)
        if torch.is_tensor(d_loss) and d_loss.requires_grad:
            (d_loss / self.accum).backward()
        out.logs.update(d_logs)
        out.extras["d_loss"] = d_loss
        return out


class NoGANStep:
    """Same shape as GANStep but no disc. For train_car (frozen tokenizer + CAR only)."""

    def __init__(self, step_fn, accum: int, device: str, amp: str):
        self.step_fn = step_fn
        self.accum = accum
        self.device = device
        self.amp = amp

    def __call__(self, batch, step):
        with autocast_ctx(self.device, self.amp):
            out = self.step_fn(batch, step)
        (out.loss / self.accum).backward()
        return out


# --------------------------------------------------------------------------- #
# TrainLoop — the loop itself
# --------------------------------------------------------------------------- #
@dataclass
class Cadence:
    log_every: int = 50
    sample_every: int = 500
    val_every: int = 1000
    ckpt_every: int = 1000


class TrainLoop:
    """Drives a step_fn over a DataLoader for `epochs` epochs.

    Cadences:
      * every log_every: call step_fn's logs + injected `extra_logs_fn(step)` + print.
      * every sample_every: call sample_fn(step) -> dict[name -> CHW tensor] (optional).
      * every val_every: call val_fn(step) -> (metrics_dict, images_dict) (optional).
      * every ckpt_every: call ckpt_fn(step, epoch) -> None (optional).

    Optimizer stepping is at the grad-accum boundary; gradient norms are computed there too.
    """

    def __init__(self, *, device: str, amp: str, accum: int, cadence: Cadence,
                 logger: RunLogger, batch_size: int,
                 disc_start_step: int = 0):
        self.device = device
        self.amp = amp
        self.accum = accum
        self.cad = cadence
        self.log = logger
        self.batch_size = batch_size
        self.disc_start_step = disc_start_step

    def run(self, *, dataloader, epochs: int, start_step: int, start_epoch: int,
            step_runner, optimizers: list, schedulers: list,
            gen_params, disc_params=None, grad_clip: float | None = None,
            sample_fn=None, val_fn=None, ckpt_fn=None,
            extra_logs_fn=None, max_steps: int = 0, print_prefix: str = "") -> int:
        """Returns the final global step.

        `step_runner(batch, step) -> StepOutput`: e.g. a GANStep or NoGANStep instance.
        `optimizers` / `schedulers` are stepped together at the accum boundary, except the
        disc optimizer (the LAST entry, if disc_params is given) which is gated by
        disc_start_step — matches the original behaviour in train.py / train_e2e.py.
        """
        from cvq.metrics import grad_norm

        step = start_step
        t0 = time.time()
        for epoch in range(start_epoch, epochs):
            for i, batch in enumerate(dataloader):
                if i % self.accum == 0:
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)

                out = step_runner(batch, step)

                gn_g = gn_d = 0.0
                if (i + 1) % self.accum == 0:
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(gen_params, grad_clip)
                    gn_g = grad_norm(gen_params)
                    if disc_params is not None:
                        gn_d = grad_norm(disc_params)
                    # Step generator opts always; disc opt (last) only after disc_start.
                    if disc_params is not None and len(optimizers) >= 2:
                        for opt, sch in zip(optimizers[:-1], schedulers[:-1]):
                            opt.step(); sch.step()
                        if step >= self.disc_start_step:
                            optimizers[-1].step(); schedulers[-1].step()
                    else:
                        for opt, sch in zip(optimizers, schedulers):
                            opt.step(); sch.step()

                # ---- cadenced logging ----
                if step % self.cad.log_every == 0:
                    ips = (step - start_step + 1) * self.batch_size / max(1e-9, time.time() - t0)
                    logs = dict(out.logs)
                    logs.update({
                        "opt/grad_norm_g": gn_g,
                        "train/img_per_s": ips, "train/epoch": epoch,
                    })
                    if disc_params is not None:
                        logs["opt/grad_norm_d"] = gn_d
                    for j, sch in enumerate(schedulers):
                        try:
                            logs[f"opt/lr_{j}"] = sch.get_last_lr()[0]
                        except Exception:
                            pass
                    if self.device == "cuda":
                        logs["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    if extra_logs_fn is not None:
                        logs.update(extra_logs_fn(step, out) or {})
                    self.log.log(logs, step)
                    print(f"{print_prefix}e{epoch} s{step} | "
                          + " ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in out.logs.items()
                                     if isinstance(v, (int, float)))[:200]
                          + f" | {ips:.1f} img/s")

                # ---- cadenced sampling ----
                if sample_fn is not None and step > 0 and step % self.cad.sample_every == 0:
                    imgs = sample_fn(step) or {}
                    if imgs:
                        self.log.log_images(imgs, step)

                # ---- cadenced validation ----
                if val_fn is not None and step > 0 and step % self.cad.val_every == 0:
                    metrics, images = val_fn(step)
                    if metrics:
                        self.log.log(metrics, step)
                    if images:
                        self.log.log_images(images, step)

                # ---- cadenced checkpoint ----
                if ckpt_fn is not None and step > 0 and step % self.cad.ckpt_every == 0:
                    ckpt_fn(step, epoch)

                step += 1
                if max_steps and step >= max_steps:
                    return step
            if max_steps and step >= max_steps:
                return step
        return step
