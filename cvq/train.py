"""
Train the CVQ tokenizer on the Pokemon dataset.

Single-device (MPS/CPU) adaptation of the paper's multi-node torchrun recipe. Keeps
the faithful pieces — AdamW(beta1=0.5, beta2=0.9), LPIPS + PatchGAN, nested channel
dropout with the channel-count-aware GAN weight, EMA codebook — and scales batch via
gradient accumulation instead of 8 GPUs.

Run:
    python -m cvq.train --config configs/cvq_pokemon.yaml
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from cvq.data.dataset import PokemonDataset
from cvq.losses.losses import CVQLoss
from cvq.models.discriminator import NLayerDiscriminator
from cvq.models.tokenizer import CVQTokenizer
from cvq.utils import describe_device, resolve_device


# --------------------------------------------------------------------------- #
# Nested channel dropout policy
# --------------------------------------------------------------------------- #
def sample_c_keep(total_channels: int, prob: float, generator: torch.Generator) -> int | None:
    """Decide how many leading channels to keep this step (nested channel dropout).

    Returns an int in [1, total_channels] to truncate to, or None to use all channels.

    Policy (the paper's hybrid scheme): with probability `prob` (alpha) we apply
    truncation and sample c_keep ~ Uniform[1, total_channels]; otherwise we keep the
    full set. Sampling the *cut point* uniformly trains the decoder to reconstruct at
    every level of detail, which is what makes the channel ordering coarse-to-fine.

    This is a tunable design choice — e.g. you could bias c_keep toward larger values
    (favoring fine detail) with a non-uniform distribution, or anneal `prob` over
    training. Uniform is the simple, faithful default.
    """
    if torch.rand((), generator=generator).item() >= prob:
        return None
    return int(torch.randint(1, total_channels + 1, (), generator=generator).item())


def lr_lambda(step: int, warmup: int):
    return min(1.0, (step + 1) / max(1, warmup))


def autocast_ctx(device: str, amp: str):
    """bf16 autocast on CUDA when amp=='bf16'; otherwise a no-op (fp32)."""
    if amp == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cvq_pokemon.yaml")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0=full)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    tcfg, mcfg, lcfg, ocfg = cfg["train"], cfg["model"], cfg["loss"], cfg["out"]
    device = resolve_device(tcfg["device"])
    amp = tcfg.get("amp", "none")
    print(f"device: {describe_device(device)} | amp: {amp}")
    torch.manual_seed(tcfg["seed"])
    gen = torch.Generator().manual_seed(tcfg["seed"])

    # ---- data ----
    ds = PokemonDataset(cfg["data"]["root"], size=cfg["data"]["size"], hflip=cfg["data"]["hflip"])
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                    num_workers=tcfg["num_workers"], drop_last=True, pin_memory=False)
    print(f"dataset: {len(ds)} images | {len(dl)} batches/epoch")

    # ---- models ----
    tok = CVQTokenizer(
        siglip_name=mcfg["siglip_name"], siglip_layers=mcfg["siglip_layers"],
        freeze_encoder=mcfg["freeze_encoder"], latent_channels=mcfg["latent_channels"],
        codebook_size=mcfg["codebook_size"], codebook_decay=mcfg["codebook_decay"],
        commitment_beta=mcfg["commitment_beta"], decoder_ch=mcfg["decoder_ch"],
        decoder_ch_mult=tuple(mcfg["decoder_ch_mult"]), decoder_res_blocks=mcfg["decoder_res_blocks"],
    ).to(device)
    disc = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)

    crit = CVQLoss(
        disc_start=tcfg["disc_start_step"], recon_loss_type=lcfg["recon_loss_type"],
        perceptual_weight=lcfg["perceptual_weight"], semantic_weight=lcfg["semantic_weight"],
        codebook_weight=lcfg["codebook_weight"], disc_weight=lcfg["disc_weight"],
        disc_loss=lcfg["disc_loss"], lpips_net=lcfg["lpips_net"], gan_eta=lcfg["gan_eta"],
    ).to(device)

    betas = (tcfg["beta1"], tcfg["beta2"])
    if mcfg["freeze_encoder"]:
        # Stage I: only the adapter + decoder train.
        g_groups = [{"params": tok.trainable_parameters(), "lr": tcfg["lr"]}]
    else:
        # Stage II (end-to-end): finetune SigLIP at a lower LR than the fresh head
        # (paper Stage-II lr 2e-5 vs Stage-I 1e-4) to avoid wrecking its features.
        head = list(tok.adapter.parameters()) + list(tok.decoder.parameters())
        enc = [p for p in tok.encoder.parameters() if p.requires_grad]
        g_groups = [
            {"params": head, "lr": tcfg["lr"]},
            {"params": enc, "lr": tcfg.get("encoder_lr", tcfg["lr"] * 0.2)},
        ]
        print(f"Stage II: finetuning encoder at lr={g_groups[1]['lr']:.1e}")
    opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=tcfg["weight_decay"])
    opt_d = torch.optim.AdamW(disc.parameters(), lr=tcfg["lr"], betas=betas,
                              weight_decay=tcfg["weight_decay"])
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: lr_lambda(s, tcfg["warmup_steps"]))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: lr_lambda(s, tcfg["warmup_steps"]))

    ckpt_dir = Path(ocfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(ocfg["run_dir"])

    start_step, start_epoch = 0, 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        # strict=False: the frozen encoder is loaded from HF in __init__, not the ckpt.
        tok.load_state_dict(ck["tokenizer"], strict=False); disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start_step, start_epoch = ck["step"], ck["epoch"]
        print(f"resumed from {args.resume} @ step {start_step}")

    total_channels = mcfg["latent_channels"]
    accum = tcfg["grad_accum"]
    fixed_batch = next(iter(dl))["image"][:8].to(device)  # for sample grids
    step = start_step
    t0 = time.time()

    for epoch in range(start_epoch, tcfg["epochs"]):
        tok.train(); disc.train()
        # Run SigLIP on the reconstruction WITH grad (param-frozen) for the semantic loss.
        sem_encode = partial(tok.encoder, with_grad=True)
        for i, batch in enumerate(dl):
            x = batch["image"].to(device)
            c_keep = sample_c_keep(total_channels, tcfg["nested_dropout_prob"], gen)
            if i % accum == 0:
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)

            # ---- generator forward + backward ----
            # Freeze disc params so the generator's GAN term doesn't deposit gradients
            # into the discriminator (grad still flows *through* it to the decoder).
            for p in disc.parameters():
                p.requires_grad_(False)
            with autocast_ctx(device, amp):
                out = tok(x, c_keep=c_keep)
                recon, vq_loss = out["recon"], out["vq_loss"]
                last_layer = tok.decoder.conv_out.weight
                g_total, g_logs = crit.generator_step(
                    target=x, recon=recon, vq_loss=vq_loss, discriminator=disc,
                    last_layer=last_layer, global_step=step, c_keep=c_keep,
                    total_channels=total_channels, siglip_real=out["siglip_feat"],
                    siglip_recon_fn=sem_encode,
                )
            (g_total / accum).backward()
            for p in disc.parameters():
                p.requires_grad_(True)

            # ---- discriminator backward (fresh graph from detached recon) ----
            with autocast_ctx(device, amp):
                d_loss, d_logs = crit.discriminator_step(x, recon, disc, step)
            if torch.is_tensor(d_loss) and d_loss.requires_grad:
                (d_loss / accum).backward()

            # ---- optimizer step at accumulation boundary ----
            if (i + 1) % accum == 0:
                opt_g.step(); sched_g.step()
                if step >= tcfg["disc_start_step"]:
                    opt_d.step(); sched_d.step()

            # ---- logging ----
            if step % tcfg["log_every"] == 0:
                ips = (step - start_step + 1) * tcfg["batch_size"] / (time.time() - t0)
                ck_str = "full" if c_keep is None else str(c_keep)
                print(f"e{epoch} s{step} | tot {g_logs['loss/total']:.3f} "
                      f"rec {g_logs['loss/recon']:.3f} lpips {g_logs['loss/lpips']:.3f} "
                      f"sem {g_logs['loss/semantic']:.3f} vq {g_logs['loss/vq']:.4f} "
                      f"d {d_logs['loss/disc']:.3f} | use {out['stats']['usage']:.3f} "
                      f"ppl {out['stats']['perplexity']:.0f} c_keep {ck_str} | {ips:.1f} img/s")
                for k, v in {**g_logs, **d_logs}.items():
                    writer.add_scalar(k, v, step)
                writer.add_scalar("codebook/usage", out["stats"]["usage"], step)
                writer.add_scalar("codebook/perplexity", out["stats"]["perplexity"], step)

            # ---- sample reconstructions ----
            if step % tcfg["sample_every"] == 0:
                tok.eval()
                with torch.no_grad():
                    r = tok(fixed_batch)["recon"]
                grid = make_grid(torch.cat([fixed_batch, r], 0).clamp(-1, 1) * 0.5 + 0.5,
                                 nrow=8)
                save_image(grid, sample_dir / f"recon_{step:06d}.png")
                tok.train()

            # ---- checkpoint ----
            if step > 0 and step % tcfg["ckpt_every"] == 0:
                save_ckpt(ckpt_dir, tok, disc, opt_g, opt_d, step, epoch, cfg, ocfg["keep_last"])

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    save_ckpt(ckpt_dir, tok, disc, opt_g, opt_d, step, epoch, cfg, ocfg["keep_last"])
    writer.close()
    print("training complete.")


def _tok_state_no_encoder(tok):
    # The frozen SigLIP backbone is reproducible from HF — don't bloat checkpoints
    # (~370MB) with it. Persist only the trainable adapter/decoder + EMA codebook.
    return {k: v for k, v in tok.state_dict().items() if not k.startswith("encoder.")}


def save_ckpt(ckpt_dir: Path, tok, disc, opt_g, opt_d, step, epoch, cfg, keep_last):
    path = ckpt_dir / f"cvq_step{step:06d}.pt"
    tok_state = _tok_state_no_encoder(tok)
    torch.save({
        "tokenizer": tok_state, "disc": disc.state_dict(),
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "step": step, "epoch": epoch, "config": cfg,
    }, path)
    # also a stable 'latest' pointer (model-only)
    torch.save({"tokenizer": tok_state, "config": cfg, "step": step},
               ckpt_dir / "latest.pt")
    cks = sorted(ckpt_dir.glob("cvq_step*.pt"))
    for old in cks[:-keep_last]:
        old.unlink(missing_ok=True)
    print(f"  saved {path.name}")


if __name__ == "__main__":
    main()
