"""
CAR task — text-to-image AR training on a FROZEN tokenizer, formerly cvq/train_car.py.

The tokenizer is loaded from a checkpoint and frozen; the CAR learns EOSTok's NTP
cross-entropy over channel-tokens. Joint E2E (APR + tokenizer unfreeze) is the e2e task.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cvq.data.car_dataset import CaptionedImageDataset, CARCollate
from imagelab.data.dataset import OverfitDataset
from cvq.eval.evaluator import sample_generations
from cvq.factory import (build_car, build_conditioning, build_text_tokenizer,
                         build_tokenizer)
from cvq.nested_dropout import channel_weights
from imagelab.registry import register
from cvq.tasks.base import StepOutput, Task


def cosine_lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog))) * (1 - 1e-3) + 1e-3


@register("task", "car", paper="arXiv:2605.26089")
class CARTask(Task):
    name = "car"
    gan = False
    ckpt_prefix = "car"
    latest_name = "car_latest.pt"
    # YOURS TO TUNE — the overfit kill gate + what lab runs/board/compare lead with.
    # An AR head should hit >90% token accuracy on one image's fixed token sequence.
    overfit_gate = ("car/token_acc", ">=", 0.90)
    key_metrics = ["eval/combined/token_acc", "eval/all/token_acc", "car/token_acc"]
    higher_is_better = frozenset({"eval/combined/token_acc", "eval/all/token_acc",
                                  "car/token_acc", "eval/combined/bit_acc",
                                  "eval/all/bit_acc"})

    def setup(self):
        rc, t, m, device = self.rc, self.rc.train, self.rc.model, self.device

        # ---- frozen tokenizer (config embedded in ckpt) ----
        tok_ckpt = self.args.tokenizer_ckpt or "checkpoints/best.pt"
        self.tok, tok_cfg = build_tokenizer({}, device, ckpt=tok_ckpt)
        self.tok.eval()
        for p in self.tok.parameters():
            p.requires_grad_(False)
        K = self.tok.quantizer.codebook_size
        self.C = self.tok.latent_channels
        print(f"tokenizer: frozen | K={K} | channels={self.C} | from {tok_ckpt}")

        # ---- text tokenizer + conditioning + CAR ----
        text_tok = build_text_tokenizer(m)
        # NB: historical behavior — the frozen-tokenizer recipe never passed the seeded
        # generator to Conditioning, so caption-drop decisions ride the global RNG here.
        self.cond = build_conditioning(m, text_tok, p_uncond=t.cond_dropout_prob,
                                       device=device)
        self.car = build_car(m, K, self.C, device)

        # ---- data (root/size come from the tokenizer's own training config) ----
        ds = CaptionedImageDataset(tok_cfg["data"]["root"], size=tok_cfg["data"]["size"],
                                   hflip=t.hflip,
                                   split="train", val_fraction=rc.data.val_fraction)
        if t.overfit_n:
            ds = OverfitDataset(ds, t.overfit_n)
            print(f"OVERFIT MODE: dataset clamped to {ds.n} image(s)")
        collate = CARCollate(text_tok, max_len=m.max_text_len)
        self.dataloader = DataLoader(ds, batch_size=t.batch_size, shuffle=True,
                                     num_workers=t.num_workers, drop_last=True,
                                     collate_fn=collate)
        print(f"dataset: {len(ds)} images | {len(self.dataloader)} batches/epoch")

        # ---- optimizer (historical: betas fixed at (0.9, 0.95) for the LM, cosine LR) ----
        total_steps = (len(self.dataloader) // t.grad_accum) * t.epochs
        opt = torch.optim.AdamW(self.car.trainable_parameters(), lr=t.lr,
                                betas=(0.9, 0.95), weight_decay=t.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: cosine_lr_lambda(s, t.warmup_steps, total_steps))
        self.optimizers = [opt]
        self.schedulers = [sched]
        self.gen_params = list(self.car.trainable_parameters())

        # ---- channel-weight schedule (couples CVQ ordering to NTP) ----
        self.chan_w = channel_weights(self.C, t.channel_weight_schedule,
                                      t.channel_weight_alpha, device=device)
        if t.channel_weight_schedule != "uniform":
            print(f"channel-weight schedule: {t.channel_weight_schedule} | early/late ratio = "
                  f"{self.chan_w[0].item() / self.chan_w[-1].item():.2f}")

        self.sample_dir = Path(rc.out.sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def step_fn(self, batch, step):
        t = self.rc.train
        x = batch["image"].to(self.device)
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)
        # caption dropout for CFG -- applied via the same Conditioning instance sampling uses
        text_ids, text_mask = self.cond.maybe_drop(text_ids, text_mask)
        with torch.no_grad():
            idxs = self.tok(x)["indices"]                  # (B, C)
        use_cw = self.chan_w if t.channel_weight_schedule != "uniform" else None
        loss, logs = self.car.loss(text_ids, text_mask, idxs, channel_weights=use_cw)
        return StepOutput(loss=loss, logs=logs, extras={})

    def sample_fn(self, step):
        m = self.rc.model
        self.car.eval()
        grid = sample_generations(self.car, self.tok, self.cond, m.sample_prompts,
                                  self.device, self.rc.train.amp, self.sample_dir, step,
                                  cfg_scale=m.cfg_scale, temperature=m.temperature,
                                  top_k=m.top_k)
        self.car.train()
        return {"generations": grid}

    # ------------------------------------------------------------------ #
    def checkpoint_state(self):
        return ({"car": self.car.state_dict()},
                {"opt": self.optimizers[0].state_dict()},
                ["car"])

    def load_resume(self, ck):
        self.car.load_state_dict(ck["car"])
        self.optimizers[0].load_state_dict(ck["opt"])
        return ck["step"], 0
