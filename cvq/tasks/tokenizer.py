"""
Tokenizer task — CVQ tokenizer training (recon + GAN), formerly cvq/train.py.

Faithful pieces kept: AdamW(beta1=0.5, beta2=0.9), LPIPS + PatchGAN, nested channel
dropout with the channel-count-aware GAN weight, index-backprop IBQ codebook. Batch is
scaled via gradient accumulation instead of 8 GPUs.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.data.dataset import ManifestImageDataset
from cvq.eval.metrics import validate
from cvq.factory import build_discriminator, build_tokenizer
from cvq.losses.losses import CVQLoss
from cvq.nested_dropout import HybridUniformPolicy
from cvq.registry import register
from cvq.tasks.base import StepOutput, Task
from cvq.training_loop import split_decay_groups, warmup_lr_lambda
from cvq.utils import denorm


@register("task", "tokenizer")
class TokenizerTask(Task):
    gan = True
    ckpt_prefix = "cvq"
    latest_name = "latest.pt"
    has_val = True

    def setup(self):
        rc, t, l, device = self.rc, self.rc.train, self.rc.loss, self.device

        # ---- data ----
        ds = ManifestImageDataset(rc.data.root, size=rc.data.size, hflip=rc.data.hflip)
        self.ds = ds
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True,
                                     pin_memory=False)
        print(f"dataset: {len(ds)} images | {len(self.dataloader)} batches/epoch")

        # ---- models ----
        self.tok, _ = build_tokenizer(rc, device)
        self.disc = build_discriminator(device)
        self.crit = CVQLoss(
            disc_start=t.disc_start_step, recon_loss_type=l.recon_loss_type,
            perceptual_weight=l.perceptual_weight, codebook_weight=l.codebook_weight,
            disc_weight=l.disc_weight, disc_loss=l.disc_loss,
            lpips_net=l.lpips_net, gan_eta=l.gan_eta,
        ).to(device)

        # ---- nested channel dropout policy ----
        self.total_channels = rc.model.latent_channels
        self.nested = HybridUniformPolicy(self.total_channels, t.nested_dropout_prob,
                                          generator=self.rng)

        # ---- optimizers (Muon/Pion experimental swap kept) ----
        betas = (t.beta1, t.beta2)
        if t.optimizer in ("muon", "pion"):
            from cvq.muon import MuonAdamW, build_muon_groups
            g_groups = build_muon_groups(
                list(self.tok.named_parameters()), method=t.optimizer,
                muon_lr=t.muon_lr, adamw_lr=t.lr, weight_decay=t.weight_decay,
                momentum=t.muon_momentum, ns_steps=t.muon_ns_steps,
                promotion_steps=t.pion_promotion_steps,
            )
            opt_g = MuonAdamW(g_groups)
            print(f"optimizer: {t.optimizer} | muon_lr={t.muon_lr} adamw_lr={t.lr} "
                  f"| {len(g_groups)} param groups")
        else:
            g_groups = split_decay_groups(self.tok.trainable_parameters(), t.lr, t.weight_decay)
            opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=t.weight_decay)
        self.gen_params = [p for grp in g_groups for p in grp["params"]]
        opt_d = torch.optim.AdamW(
            split_decay_groups(list(self.disc.parameters()), t.lr, t.weight_decay),
            betas=betas, weight_decay=t.weight_decay)
        self.optimizers = [opt_g, opt_d]
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: warmup_lr_lambda(s, t.warmup_steps)),
            torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: warmup_lr_lambda(s, t.warmup_steps)),
        ]
        self.disc_params = list(self.disc.parameters())

        self.sample_dir = self._sample_dir()
        self.fixed_batch = next(iter(self.dataloader))["image"][:8].to(device)

    def _sample_dir(self):
        from pathlib import Path
        p = Path(self.rc.out.sample_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def grad_clip(self):
        # Historical: train.py never passed grad_clip to the loop, so the tokenizer task
        # trains UNCLIPPED (unlike car/e2e). Kept for parity with prior runs.
        return None

    # ------------------------------------------------------------------ #
    def generator_fn(self, batch, step):
        x = batch["image"].to(self.device)
        c_keep = self.nested.sample(step)
        out = self.tok(x, c_keep=c_keep)
        recon, vq_loss = out["recon"], out["vq_loss"]
        last_layer = self.tok.decoder.conv_out.weight
        g_total, g_logs = self.crit.generator_step(
            target=x, recon=recon, vq_loss=vq_loss, discriminator=self.disc,
            last_layer=last_layer, global_step=step, c_keep=c_keep,
            total_channels=self.total_channels,
        )
        logs = dict(g_logs)
        logs.update({
            "codebook/usage_batch": out["stats"]["usage"],
            "codebook/perplexity": out["stats"]["perplexity"],
            "codebook/quant_error": out["stats"]["quant_error"],
            "train/c_keep": c_keep if c_keep is not None else self.total_channels,
        })
        if "entropy_loss" in out["stats"]:
            logs.update({
                "codebook/entropy_loss": out["stats"]["entropy_loss"],
                "codebook/entropy_per_sample": out["stats"]["entropy_per_sample"],
                "codebook/entropy_marginal": out["stats"]["entropy_marginal"],
            })
        return StepOutput(loss=g_total, logs=logs, extras={"x": x, "recon": recon})

    def discriminator_fn(self, batch, step, extras):
        return self.crit.discriminator_step(extras["x"], extras["recon"], self.disc, step)

    # ------------------------------------------------------------------ #
    def sample_fn(self, step):
        self.tok.eval()
        with torch.no_grad():
            r = self.tok(self.fixed_batch)["recon"]
        grid = make_grid(denorm(torch.cat([self.fixed_batch, r], 0)), nrow=8)
        save_image(grid, self.sample_dir / f"recon_{step:06d}.png")
        self.tok.train()
        return {"reconstructions": grid}

    def val_fn(self, step):
        t = self.rc.train
        metrics, images = validate(self.tok, self.ds, self.device, batch_size=t.batch_size,
                                   compute_fid=t.val_fid, lpips_fn=self.crit.perceptual)
        print("  val:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
        score = metrics.get("val/rFID", metrics.get("val/recon_l2_full", float("inf")))
        if self.store.save_best(score, step, self.rc.raw, {"tokenizer": self.tok.state_dict()}):
            self.logger.log_artifact(self.store.best_path(), "cvq-tokenizer", "model",
                                     step, aliases=["best"])
            print(f"  new best ({score:.4f}) -> best.pt")
        return metrics, images

    # ------------------------------------------------------------------ #
    def checkpoint_state(self):
        return ({"tokenizer": self.tok.state_dict(), "disc": self.disc.state_dict()},
                {"opt_g": self.optimizers[0].state_dict(),
                 "opt_d": self.optimizers[1].state_dict()},
                ["tokenizer"])

    def load_resume(self, ck):
        self.tok.load_state_dict(ck["tokenizer"], strict=False)
        self.disc.load_state_dict(ck["disc"])
        self.optimizers[0].load_state_dict(ck["opt_g"])
        self.optimizers[1].load_state_dict(ck["opt_d"])
        return ck["step"], ck["epoch"]

    def finalize(self, final_step):
        t = self.rc.train
        metrics, images = validate(self.tok, self.ds, self.device, batch_size=t.batch_size,
                                   compute_fid=t.val_fid, lpips_fn=self.crit.perceptual)
        self.logger.log(metrics, final_step)
        self.logger.log_images(images, final_step)
        print("final val:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
        final_score = metrics.get("val/rFID", metrics.get("val/recon_l2_full", float("inf")))
        aliases = ["latest", "best"] if final_score <= self.store.best_score else ["latest"]
        self.logger.log_artifact(self.store.latest_path(), "cvq-tokenizer", "model",
                                 final_step, aliases=aliases)
