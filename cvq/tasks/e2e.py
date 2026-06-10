"""
E2E task — joint training of the CVQ/EOSTok tokenizer AND the CAR text-to-image model,
formerly cvq/train_e2e.py.

Faithful EOSTok objective (arXiv:2605.00503), mapped onto our channel-wise stack:

    L_E2E = L_VQVAE  +  lambda_NTP * L_NTP  +  lambda_APR * L_APR  (+ lambda_sem * L_implicit)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cvq.data.car_dataset import CaptionedImageDataset, CARCollate
from cvq.eval.evaluator import GroupedEvaluator
from cvq.factory import (build_car, build_conditioning, build_discriminator,
                         build_text_tokenizer, build_tokenizer)
from cvq.losses.losses import CVQLoss
from cvq.nested_dropout import HybridUniformPolicy, channel_weights
from cvq.registry import register
from cvq.tasks.base import StepOutput, Task
from cvq.training_loop import split_decay_groups, warmup_lr_lambda


@register("task", "e2e")
class E2ETask(Task):
    gan = True
    ckpt_prefix = "e2e"
    latest_name = "e2e_latest.pt"

    def setup(self):
        rc, t, m, l, device = self.rc, self.rc.train, self.rc.model, self.rc.loss, self.device

        # ---- tokenizer (TRAINABLE, joint; optional warm start) ----
        self.tok, _ = build_tokenizer(rc, device,
                                      ckpt=self.args.tokenizer_ckpt or None)
        if self.args.tokenizer_ckpt:
            print(f"tokenizer warm-started from {self.args.tokenizer_ckpt}")
        K = self.tok.quantizer.codebook_size
        self.C = self.tok.latent_channels
        print(f"tokenizer: TRAINABLE | K={K} | channels={self.C}")

        self.disc = build_discriminator(device)
        self.crit = CVQLoss(
            disc_start=t.disc_start_step, recon_loss_type=l.recon_loss_type,
            perceptual_weight=l.perceptual_weight, codebook_weight=l.codebook_weight,
            disc_weight=l.disc_weight, disc_loss=l.disc_loss,
            lpips_net=l.lpips_net, gan_eta=l.gan_eta,
        ).to(device)

        # ---- CAR (LLM backbone + image vocab) + conditioning ----
        text_tok = build_text_tokenizer(m)
        self.car = build_car(m, K, self.C, device)
        self.cond = build_conditioning(m, text_tok, p_uncond=t.cond_dropout_prob,
                                       generator=self.rng, device=device)

        # ---- DINOv2 semantic alignment (EOSTok L_implicit, optional) ----
        self.lam_sem = t.lambda_sem
        self.dino = None
        if self.lam_sem > 0:
            from cvq.models.dino_align import DINOAlign
            self.dino = DINOAlign(latent_channels=self.C, grid=self.tok.grid,
                                  dino_name=m.dino_name).to(device)
            print(f"DINOv2 alignment: ON | lambda_sem={self.lam_sem} | {m.dino_name}")

        # ---- nested channel dropout + channel-weight schedule ----
        self.nested = HybridUniformPolicy(self.C, t.nested_dropout_prob, generator=self.rng)
        self.cw_schedule = t.channel_weight_schedule
        self.chan_w = channel_weights(self.C, t.channel_weight_schedule,
                                      t.channel_weight_alpha,
                                      device=device, dtype=torch.float32)
        if self.cw_schedule != "uniform":
            print(f"channel-weight schedule: {self.cw_schedule}"
                  + (f" (alpha={t.channel_weight_alpha})" if self.cw_schedule in ("linear", "exp") else "")
                  + f" | early/late ratio = {self.chan_w[0].item() / self.chan_w[-1].item():.2f}")

        # ---- data ----
        ds = CaptionedImageDataset(rc.data.root, size=rc.data.size, hflip=rc.data.hflip,
                                   augment=rc.data.augment)
        collate = CARCollate(text_tok, max_len=m.max_text_len)
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True,
                                     collate_fn=collate)
        print(f"dataset: {len(ds)} images | {len(self.dataloader)} batches/epoch")

        # ---- optimizers ----
        betas = (t.beta1, t.beta2)
        wd = t.weight_decay
        tok_groups = split_decay_groups(self.tok.trainable_parameters(), t.lr, wd)
        car_groups = split_decay_groups(self.car.trainable_parameters(), t.resolved_car_lr(), wd)
        g_groups = tok_groups + car_groups
        if self.dino is not None:
            g_groups = g_groups + split_decay_groups(list(self.dino.proj.parameters()), t.lr, wd)
        opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=wd)
        opt_d = torch.optim.AdamW(split_decay_groups(list(self.disc.parameters()), t.lr, wd),
                                  betas=betas, weight_decay=wd)
        self.gen_params = [p for grp in g_groups for p in grp["params"]]
        self.optimizers = [opt_g, opt_d]
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: warmup_lr_lambda(s, t.warmup_steps)),
            torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: warmup_lr_lambda(s, t.warmup_steps)),
        ]
        self.disc_params = list(self.disc.parameters())

        # ---- EOSTok loss weights / gates ----
        self.lam_ntp = t.lambda_ntp
        self.lam_apr = t.lambda_apr
        self.ar_start = t.ar_start_step
        self.apr_lpips_w = t.apr_lpips_weight
        # Floor for the APR c_keep: with c_keep=1 the AR is asked to decode the full RGB
        # image from a single channel-token prediction, which burns gradient on impossible
        # decodes. Default 0.25 -> APR never decodes with fewer than C/4 channels; 0 disables.
        frac = t.apr_min_c_keep_frac
        self.apr_min_c_keep = max(1, int(round(frac * self.C))) if frac > 0 else 1

        self.has_codebook = hasattr(self.tok.quantizer, "embed")  # IBQ yes; FSQ no
        self.cb = self.tok.quantizer.embed.weight if self.has_codebook else None

        # ---- per-dataset eval split (easy-vs-hard) ----
        self.sample_dir = Path(rc.out.sample_dir)
        self.evaluator = GroupedEvaluator(ds, collate=collate, device=device,
                                          sample_dir=self.sample_dir)
        self.prompts_by_ds = m.sample_prompts_by_dataset or {"all": m.sample_prompts}
        print(f"eval split: {len(self.evaluator.eval_batches)} group(s) -> "
              f"{self.evaluator.group_names()}")

    # ------------------------------------------------------------------ #
    def generator_fn(self, batch, step):
        x = batch["image"].to(self.device)
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        text_ids, text_mask = self.cond.maybe_drop(text_ids, text_mask)
        c_keep = self.nested.sample(step)
        ar_on = step >= self.ar_start

        out = self.tok(x, c_keep=c_keep)
        recon, vq_loss = out["recon"], out["vq_loss"]
        idxs = out["indices"]                                  # (B, C) tokenizer's own
        last_layer = self.tok.decoder.conv_out.weight
        g_total, g_logs = self.crit.generator_step(
            target=x, recon=recon, vq_loss=vq_loss, discriminator=self.disc,
            last_layer=last_layer, global_step=step, c_keep=c_keep,
            total_channels=self.C,
        )
        ntp_loss = recon.new_zeros(())
        apr_loss = recon.new_zeros(())
        ar_logs = {}
        if ar_on:
            # AR objective via the head-agnostic seam: softmax NTP (Fork A) or MBM bit-CE
            # (Fork B). channel_weights only apply to the softmax head; flat for MBM.
            use_cw = self.chan_w if self.cw_schedule != "uniform" else None
            ntp_loss, ar_logs0, ar_aux = self.car.ar_loss(text_ids, text_mask, idxs.detach(),
                                                          channel_weights=use_cw)
            ar_logs.update(ar_logs0)

            # --- APR (EOSTok): soft-decode the AR prediction to pixels, prefix-truncated to
            # the step's c_keep. Softmax head + a real codebook only; the MBM/FSQ bit head
            # has no soft-codebook decode, so Fork B runs with lambda_apr=0.
            if self.has_codebook and self.lam_apr > 0 and "logits" in ar_aux:
                logits = ar_aux["logits"]
                p_hat = logits.softmax(-1)
                z_q_apr = torch.einsum("bck,kd->bcd", p_hat.float(), self.cb.float())
                side = int(round((z_q_apr.shape[-1]) ** 0.5))
                z_q_apr = z_q_apr.reshape(z_q_apr.shape[0], self.C, side, side).to(recon.dtype)
                c_keep_apr = max(c_keep, self.apr_min_c_keep) if c_keep is not None else None
                z_q_apr = self.nested.apply(z_q_apr, c_keep_apr)
                recon_apr = self.tok.decoder(z_q_apr)
                apr_loss = torch.nn.functional.mse_loss(recon_apr, x)
                if self.apr_lpips_w > 0:
                    apr_loss = apr_loss + self.apr_lpips_w * self.crit.perceptual(recon_apr, x).mean()
                ar_logs["car/apr_loss"] = apr_loss.item()
                ar_logs["car/apr_c_keep"] = (c_keep_apr if c_keep_apr is not None else self.C)
        sem_loss = recon.new_zeros(())
        if self.dino is not None:
            sem_loss = self.dino(out["z"], x)
            ar_logs["car/sem_loss"] = sem_loss.item()

        total = g_total + self.lam_ntp * ntp_loss + self.lam_apr * apr_loss + self.lam_sem * sem_loss

        logs = dict(g_logs)
        logs.update(ar_logs)
        logs.update({
            "codebook/usage_batch": out["stats"]["usage"],
            "codebook/perplexity": out["stats"]["perplexity"],
            "codebook/entropy_loss": out["stats"].get("entropy_loss", 0.0),
            "train/ar_on": float(ar_on),
            "train/c_keep": c_keep if c_keep is not None else self.C,
        })
        return StepOutput(loss=total, logs=logs, extras={"x": x, "recon": recon})

    def discriminator_fn(self, batch, step, extras):
        return self.crit.discriminator_step(extras["x"], extras["recon"], self.disc, step)

    # ------------------------------------------------------------------ #
    def sample_fn(self, step):
        m, t = self.rc.model, self.rc.train
        self.tok.eval(); self.car.eval()
        ar_on = step >= self.ar_start
        metrics, images = self.evaluator.eval_recon(
            self.tok, perceptual=self.crit.perceptual,
            car=self.car if ar_on else None, step=step)
        if ar_on:
            images.update(self.evaluator.eval_generation(
                self.car, self.tok, self.cond, self.prompts_by_ds,
                amp=t.amp, step=step, temperature=m.temperature,
                top_k=m.top_k, cfg_scale=m.cfg_scale))
        if metrics:
            self.logger.log(metrics, step)
        self.tok.train(); self.car.train()
        return images

    # ------------------------------------------------------------------ #
    def checkpoint_state(self):
        return ({"tokenizer": self.tok.state_dict(), "car": self.car.state_dict(),
                 "disc": self.disc.state_dict()},
                {"opt_g": self.optimizers[0].state_dict(),
                 "opt_d": self.optimizers[1].state_dict()},
                ["tokenizer", "car"])

    def load_resume(self, ck):
        self.tok.load_state_dict(ck["tokenizer"], strict=False)
        self.car.load_state_dict(ck["car"])
        self.disc.load_state_dict(ck["disc"])
        self.optimizers[0].load_state_dict(ck["opt_g"])
        self.optimizers[1].load_state_dict(ck["opt_d"])
        return ck["step"], 0
