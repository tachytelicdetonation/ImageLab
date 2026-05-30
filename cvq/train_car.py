"""
Train the CAR (channel-wise autoregressive) text-to-image model on top of a trained
CVQ/EOSTok tokenizer.

Baseline (this script): the tokenizer is FROZEN (loaded from a checkpoint) and we train the
CAR with EOSTok's next-token-prediction (NTP) cross-entropy over channel-tokens. This is the
debuggable first stage of the end-to-end system; the APR loss + tokenizer unfreeze (true
joint E2E) is the next step and is stubbed below.

    python -m cvq.train_car --config configs/car_pokemon_qwen.yaml --tokenizer_ckpt checkpoints/best.pt

Logs NTP loss / token accuracy, periodically samples images from held-out prompts, and
checkpoints the CAR (tokenizer is not re-saved — it's reproducible from its own ckpt).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.data.car_dataset import CARPokemonDataset, CARCollate, prettify_name
from cvq.models.car import CAR
from cvq.reconstruct import build_from_ckpt
from cvq.utils import resolve_device, describe_device

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def autocast_ctx(device, amp):
    from contextlib import nullcontext
    if amp == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def lr_lambda(step, warmup, total):
    import math
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

    # ---- frozen tokenizer ----
    tok, tok_cfg = build_from_ckpt(args.tokenizer_ckpt, device)
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    K = tok.quantizer.codebook_size
    C = tok.latent_channels
    print(f"tokenizer: frozen | K={K} | channels={C} | from {args.tokenizer_ckpt}")

    # ---- Qwen tokenizer + CAR model ----
    from transformers import AutoTokenizer
    qwen_name = mcfg.get("qwen_name", "Qwen/Qwen3-0.6B-Base")
    text_tok = AutoTokenizer.from_pretrained(qwen_name)
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
        opt, lambda s: lr_lambda(s, tcfg["warmup_steps"], total_steps))

    ckpt_dir = Path(ocfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)

    # ---- wandb ----
    wcfg = cfg.get("wandb", {}) or {}
    use_wandb = _HAS_WANDB and wcfg.get("enabled", False)
    if use_wandb:
        mode = wcfg.get("mode", "online")
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            mode = "offline"
        wandb.init(project=wcfg.get("project", "cvq-pokemon"), entity=wcfg.get("entity"),
                   name=wcfg.get("name"), mode=mode, config=cfg)

    def wlog(d, s):
        if use_wandb:
            wandb.log(d, step=s)

    # fixed prompts for sampling progress
    sample_prompts = mcfg.get("sample_prompts",
                              ["pikachu", "charizard", "bulbasaur", "mewtwo",
                               "rayquaza mega", "gengar", "eevee", "snorlax"])
    denorm = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        car.load_state_dict(ck["car"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"resumed from {args.resume} @ step {start_step}")

    step = start_step
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        car.train()
        for i, batch in enumerate(dl):
            x = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)

            # frozen tokenizer -> channel indices (no grad)
            with torch.no_grad():
                idxs = tok(x)["indices"]                       # (B, C)

            if i % accum == 0:
                opt.zero_grad(set_to_none=True)
            with autocast_ctx(device, amp):
                loss, logs = car.loss(text_ids, text_mask, idxs)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(car.trainable_parameters(),
                                                    tcfg.get("grad_clip", 1.0))
                opt.step(); sched.step()
                logs["car/grad_norm"] = float(gn)

            if step % tcfg["log_every"] == 0:
                ips = (step - start_step + 1) * tcfg["batch_size"] / (time.time() - t0)
                print(f"e{epoch} s{step} | ntp {logs['car/ntp_loss']:.4f} "
                      f"acc {logs['car/token_acc']:.3f} | {ips:.1f} img/s")
                logs.update({"opt/lr": sched.get_last_lr()[0], "train/epoch": epoch,
                             "train/img_per_s": ips})
                if device == "cuda":
                    logs["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                wlog(logs, step)

            if step > 0 and step % tcfg["sample_every"] == 0:
                sample_images(car, tok, text_tok, sample_prompts, device, amp,
                              sample_dir, step, denorm, wlog,
                              mcfg.get("max_text_len", 16),
                              cfg_scale=mcfg.get("cfg_scale", 1.0),
                              temperature=mcfg.get("temperature", 1.0),
                              top_k=mcfg.get("top_k", 0))
                car.train()

            if step > 0 and step % tcfg["ckpt_every"] == 0:
                save_ckpt(ckpt_dir, car, opt, step, epoch, cfg)

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    save_ckpt(ckpt_dir, car, opt, step, epoch, cfg)
    if use_wandb:
        wandb.finish()
    print("CAR training complete.")


@torch.no_grad()
def sample_images(car, tok, text_tok, prompts, device, amp, sample_dir, step,
                  denorm, wlog, max_len, cfg_scale=1.0, temperature=1.0, top_k=0):
    car.eval()
    enc = text_tok(prompts, padding="longest", truncation=True, max_length=max_len,
                   return_tensors="pt")
    text_ids = enc["input_ids"].to(device)
    text_mask = enc["attention_mask"].to(device)
    uncond_ids = uncond_mask = None
    if cfg_scale != 1.0:
        unc = text_tok([""] * len(prompts), padding="max_length", max_length=text_ids.shape[1],
                       return_tensors="pt")
        uncond_ids = unc["input_ids"].to(device)
        uncond_mask = unc["attention_mask"].to(device)
    with autocast_ctx(device, amp):
        idxs = car.generate(text_ids, text_mask, temperature=temperature, top_k=top_k,
                            cfg_scale=cfg_scale, uncond_text_ids=uncond_ids,
                            uncond_text_mask=uncond_mask)
        z_q = tok.quantizer.lookup(idxs)                      # (B, C, side, side)
        imgs = tok.decode(z_q)
    grid = make_grid(denorm(imgs), nrow=len(prompts))
    path = sample_dir / f"car_gen_{step:06d}.png"
    save_image(grid, path)
    print(f"  sampled {len(prompts)} prompts -> {path.name}")
    if wlog and _HAS_WANDB:
        wandb.log({"images/generations": wandb.Image(grid, caption=" | ".join(prompts))},
                  step=step)


def save_ckpt(ckpt_dir, car, opt, step, epoch, cfg):
    path = ckpt_dir / f"car_step{step:06d}.pt"
    torch.save({"car": car.state_dict(), "opt": opt.state_dict(),
                "step": step, "epoch": epoch, "config": cfg}, path)
    torch.save({"car": car.state_dict(), "config": cfg, "step": step},
               ckpt_dir / "car_latest.pt")
    print(f"  saved {path.name}")


if __name__ == "__main__":
    main()
