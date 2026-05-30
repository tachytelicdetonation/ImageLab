"""
Train the CAR (channel-wise autoregressive) text-to-image model on top of a trained
CVQ/EOSTok tokenizer.

Baseline: the tokenizer is FROZEN (loaded from a checkpoint) and we train the CAR with
EOSTok's NTP cross-entropy over channel-tokens. Joint E2E (APR + tokenizer unfreeze)
lives in train_e2e.py.

    python -m cvq.train_car --config configs/car_pokemon_qwen.yaml --tokenizer_ckpt checkpoints/best.pt

Scaffolding (AMP, accum, cadenced sample/ckpt, wandb) is in cvq.training_loop; checkpoint
I/O in cvq.checkpoint; conditioning (prompts + caption-dropout + CFG uncond) in cvq.conditioning.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.checkpoint import CheckpointStore
from cvq.conditioning import Conditioning
from cvq.data.car_dataset import CARPokemonDataset, CARCollate
from cvq.models.car import CAR
from cvq.nested_dropout import channel_weights
from cvq.tokenizer_factory import build_tokenizer
from cvq.training_loop import (
    Cadence, NoGANStep, RunLogger, StepOutput, TrainLoop, autocast_ctx, split_decay_groups,
)
from cvq.utils import describe_device, resolve_device


def cosine_lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog))) * (1 - 1e-3) + 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/car_pokemon_qwen.yaml")
    ap.add_argument("--tokenizer_ckpt", default="checkpoints/best.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg, mcfg, ocfg = cfg["train"], cfg["model"], cfg["out"]

    device = resolve_device(tcfg["device"])
    amp = tcfg.get("amp", "none")
    print(f"device: {describe_device(device)} | amp: {amp}")
    torch.manual_seed(tcfg["seed"])

    # ---- frozen tokenizer (config embedded in ckpt) ----
    tok, tok_cfg = build_tokenizer({}, device, ckpt=args.tokenizer_ckpt)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    K = tok.quantizer.codebook_size
    C = tok.latent_channels
    print(f"tokenizer: frozen | K={K} | channels={C} | from {args.tokenizer_ckpt}")

    # ---- Qwen text tokenizer + conditioning + CAR ----
    from transformers import AutoTokenizer
    qwen_name = mcfg.get("qwen_name", "Qwen/Qwen3-0.6B-Base")
    text_tok = AutoTokenizer.from_pretrained(qwen_name)
    cond = Conditioning(text_tok, max_len=mcfg.get("max_text_len", 16),
                        p_uncond=tcfg.get("cond_dropout_prob", 0.0),
                        device=device)
    car = CAR(
        codebook_size=K, num_channels=C, qwen_name=qwen_name,
        freeze_backbone=mcfg.get("freeze_backbone", False),
        attn_impl=mcfg.get("attn_impl", "sdpa"),
    ).to(device)
    n_train = sum(p.numel() for p in car.trainable_parameters())
    print(f"CAR: {qwen_name} | trainable params {n_train/1e6:.1f}M | "
          f"freeze_backbone={mcfg.get('freeze_backbone', False)}")

    # ---- data ----
    ds = CARPokemonDataset(tok_cfg["data"]["root"], size=tok_cfg["data"]["size"],
                           hflip=tcfg.get("hflip", True))
    collate = CARCollate(text_tok, max_len=mcfg.get("max_text_len", 16))
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                    num_workers=tcfg["num_workers"], drop_last=True, collate_fn=collate)
    print(f"dataset: {len(ds)} images | {len(dl)} batches/epoch")

    # ---- optimizer ----
    accum = tcfg.get("grad_accum", 1)
    total_steps = (len(dl) // accum) * tcfg["epochs"]
    opt = torch.optim.AdamW(car.trainable_parameters(), lr=tcfg["lr"],
                            betas=(0.9, 0.95), weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr_lambda(s, tcfg["warmup_steps"], total_steps))

    store = CheckpointStore(ocfg["ckpt_dir"], prefix="car", latest_name="car_latest.pt",
                            keep_last=ocfg.get("keep_last", 5))
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(cfg, run_dir=ocfg.get("run_dir"))

    # ---- resume ----
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = CheckpointStore.load(args.resume, map_location=device)
        car.load_state_dict(ck["car"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} @ step {start_step}")

    sample_prompts = mcfg.get("sample_prompts",
                              ["pikachu", "charizard", "bulbasaur", "mewtwo",
                               "rayquaza mega", "gengar", "eevee", "snorlax"])
    denorm = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5)

    # ---- channel-weight schedule (couples CVQ ordering to NTP) ----
    cw_schedule = tcfg.get("channel_weight_schedule", "uniform")
    cw_alpha = tcfg.get("channel_weight_alpha", 1.0)
    chan_w = channel_weights(C, cw_schedule, cw_alpha, device=device)
    if cw_schedule != "uniform":
        print(f"channel-weight schedule: {cw_schedule} | early/late ratio = "
              f"{chan_w[0].item()/chan_w[-1].item():.2f}")

    # ---- per-step work ----
    def step_fn(batch, step):
        x = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        # caption dropout for CFG -- applied via the same Conditioning instance sampling uses
        text_ids, text_mask = cond.maybe_drop(text_ids, text_mask)
        with torch.no_grad():
            idxs = tok(x)["indices"]                       # (B, C)
        loss, logs = car.loss(text_ids, text_mask, idxs,
                              channel_weights=chan_w if cw_schedule != "uniform" else None)
        return StepOutput(loss=loss, logs=logs, extras={})

    def sample_fn(step):
        return sample_images(car, tok, cond, sample_prompts, device, amp,
                             sample_dir, step, denorm, logger,
                             cfg_scale=mcfg.get("cfg_scale", 1.0),
                             temperature=mcfg.get("temperature", 1.0),
                             top_k=mcfg.get("top_k", 0))

    def ckpt_fn(step, epoch):
        path = store.save(step, epoch, cfg,
                          model_state={"car": car.state_dict()},
                          opt_state={"opt": opt.state_dict()},
                          latest_model_keys=["car"])
        print(f"  saved {path.name}")

    # ---- loop ----
    cadence = Cadence(log_every=tcfg["log_every"], sample_every=tcfg["sample_every"],
                      val_every=tcfg.get("val_every", 10_000_000),  # no rFID for CAR
                      ckpt_every=tcfg["ckpt_every"])
    loop = TrainLoop(device=device, amp=amp, accum=accum, cadence=cadence,
                     logger=logger, batch_size=tcfg["batch_size"])
    step_runner = NoGANStep(step_fn, accum=accum, device=device, amp=amp)
    final_step = loop.run(
        dataloader=dl, epochs=tcfg["epochs"], start_step=start_step, start_epoch=0,
        step_runner=step_runner,
        optimizers=[opt], schedulers=[sched],
        gen_params=list(car.trainable_parameters()),
        grad_clip=tcfg.get("grad_clip", 1.0),
        sample_fn=sample_fn, ckpt_fn=ckpt_fn,
        max_steps=args.max_steps, print_prefix="",
    )

    ckpt_fn(final_step, tcfg["epochs"] - 1)
    logger.close()
    print("CAR training complete.")


@torch.no_grad()
def sample_images(car, tok, cond: Conditioning, prompts, device, amp, sample_dir, step,
                  denorm, logger, cfg_scale=1.0, temperature=1.0, top_k=0):
    car.eval()
    text_ids, text_mask = cond.encode_batch(prompts)
    text_ids = text_ids.to(device); text_mask = text_mask.to(device)
    uncond_ids = uncond_mask = None
    if cfg_scale != 1.0:
        uncond_ids, uncond_mask = cond.unconditional(len(prompts), L=text_ids.shape[1], device=device)
    with autocast_ctx(device, amp):
        idxs = car.generate(text_ids, text_mask, temperature=temperature, top_k=top_k,
                            cfg_scale=cfg_scale, uncond_text_ids=uncond_ids,
                            uncond_text_mask=uncond_mask)
        z_q = tok.quantizer.lookup(idxs)
        imgs = tok.decode(z_q)
    grid = make_grid(denorm(imgs).float().cpu(), nrow=len(prompts))
    path = sample_dir / f"car_gen_{step:06d}.png"
    save_image(grid, path)
    print(f"  sampled {len(prompts)} prompts -> {path.name}")
    car.train()
    return {"generations": grid}


if __name__ == "__main__":
    main()
