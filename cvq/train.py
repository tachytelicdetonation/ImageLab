"""
Train the CVQ tokenizer on the Pokemon dataset.

Single-device (MPS/CPU/CUDA) adaptation of the paper's multi-node torchrun recipe. Keeps
the faithful pieces -- AdamW(beta1=0.5, beta2=0.9), LPIPS + PatchGAN, nested channel
dropout with the channel-count-aware GAN weight, index-backprop IBQ codebook -- and
scales batch via gradient accumulation instead of 8 GPUs.

The scaffolding (AMP / accum boundary / GAN disc-freeze dance / wandb / cadenced sample
+ val + ckpt) lives in `cvq.training_loop`; this script just declares the per-step work.

Run:
    python -m cvq.train --config configs/cvq_pokemon.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.checkpoint import CheckpointStore
from cvq.data.dataset import PokemonDataset
from cvq.losses.losses import CVQLoss
from cvq.metrics import validate
from cvq.models.discriminator import NLayerDiscriminator
from cvq.nested_dropout import HybridUniformPolicy
from cvq.tokenizer_factory import build_tokenizer
from cvq.training_loop import (
    Cadence, GANStep, RunLogger, StepOutput, TrainLoop,
    autocast_ctx, split_decay_groups, warmup_lr_lambda,
)
from cvq.utils import describe_device, resolve_device


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
    tok, _ = build_tokenizer(cfg, device)
    disc = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)

    crit = CVQLoss(
        disc_start=tcfg["disc_start_step"], recon_loss_type=lcfg["recon_loss_type"],
        perceptual_weight=lcfg["perceptual_weight"],
        codebook_weight=lcfg["codebook_weight"], disc_weight=lcfg["disc_weight"],
        disc_loss=lcfg["disc_loss"], lpips_net=lcfg["lpips_net"], gan_eta=lcfg["gan_eta"],
    ).to(device)

    # ---- nested channel dropout policy ----
    total_channels = mcfg["latent_channels"]
    nested = HybridUniformPolicy(total_channels, tcfg["nested_dropout_prob"], generator=gen)

    # ---- optimizers (Muon/Pion experimental swap kept) ----
    betas = (tcfg["beta1"], tcfg["beta2"])
    wd = tcfg["weight_decay"]
    optim_name = tcfg.get("optimizer", "adamw").lower()
    if optim_name in ("muon", "pion"):
        from cvq.muon import MuonAdamW, build_muon_groups
        muon_lr = tcfg.get("muon_lr", 0.02)
        g_groups = build_muon_groups(
            list(tok.named_parameters()), method=optim_name,
            muon_lr=muon_lr, adamw_lr=tcfg["lr"], weight_decay=wd,
            momentum=tcfg.get("muon_momentum", 0.95),
            ns_steps=tcfg.get("muon_ns_steps", 5),
            promotion_steps=tcfg.get("pion_promotion_steps", 0),
        )
        opt_g = MuonAdamW(g_groups)
        print(f"optimizer: {optim_name} | muon_lr={muon_lr} adamw_lr={tcfg['lr']} | {len(g_groups)} param groups")
    else:
        g_groups = split_decay_groups(tok.trainable_parameters(), tcfg["lr"], wd)
        opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=wd)
    gen_params = [p for grp in g_groups for p in grp["params"]]
    opt_d = torch.optim.AdamW(split_decay_groups(list(disc.parameters()), tcfg["lr"], wd),
                              betas=betas, weight_decay=wd)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: warmup_lr_lambda(s, tcfg["warmup_steps"]))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: warmup_lr_lambda(s, tcfg["warmup_steps"]))

    # ---- checkpoint + logger ----
    store = CheckpointStore(ocfg["ckpt_dir"], prefix="cvq", latest_name="latest.pt",
                            keep_last=ocfg["keep_last"])
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(cfg, run_dir=ocfg["run_dir"])

    # ---- resume ----
    start_step, start_epoch = 0, 0
    if args.resume and Path(args.resume).exists():
        ck = CheckpointStore.load(args.resume, map_location=device)
        tok.load_state_dict(ck["tokenizer"], strict=False)
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        start_step, start_epoch = ck["step"], ck["epoch"]
        print(f"resumed from {args.resume} @ step {start_step}")

    fixed_batch = next(iter(dl))["image"][:8].to(device)

    # ---- per-step work: generator + discriminator halves ----
    def generator_fn(batch, step):
        x = batch["image"].to(device)
        c_keep = nested.sample(step)
        out = tok(x, c_keep=c_keep)
        recon, vq_loss = out["recon"], out["vq_loss"]
        last_layer = tok.decoder.conv_out.weight
        g_total, g_logs = crit.generator_step(
            target=x, recon=recon, vq_loss=vq_loss, discriminator=disc,
            last_layer=last_layer, global_step=step, c_keep=c_keep,
            total_channels=total_channels,
        )
        logs = dict(g_logs)
        logs.update({
            "codebook/usage_batch": out["stats"]["usage"],
            "codebook/perplexity": out["stats"]["perplexity"],
            "codebook/quant_error": out["stats"]["quant_error"],
            "train/c_keep": c_keep if c_keep is not None else total_channels,
        })
        if "entropy_loss" in out["stats"]:
            logs.update({
                "codebook/entropy_loss": out["stats"]["entropy_loss"],
                "codebook/entropy_per_sample": out["stats"]["entropy_per_sample"],
                "codebook/entropy_marginal": out["stats"]["entropy_marginal"],
            })
        extras = {"x": x, "recon": recon}
        return StepOutput(loss=g_total, logs=logs, extras=extras)

    def discriminator_fn(batch, step, extras):
        return crit.discriminator_step(extras["x"], extras["recon"], disc, step)

    # ---- callbacks ----
    def sample_fn(step):
        tok.eval()
        with torch.no_grad():
            r = tok(fixed_batch)["recon"]
        grid = make_grid(torch.cat([fixed_batch, r], 0).clamp(-1, 1) * 0.5 + 0.5, nrow=8)
        save_image(grid, sample_dir / f"recon_{step:06d}.png")
        tok.train()
        return {"reconstructions": grid}

    def val_fn(step):
        metrics, images = validate(tok, ds, device, batch_size=tcfg["batch_size"],
                                   compute_fid=tcfg.get("val_fid", True),
                                   lpips_fn=crit.perceptual)
        print("  val:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
        score = metrics.get("val/rFID", metrics.get("val/recon_l2_full", float("inf")))
        if store.save_best(score, step, cfg, {"tokenizer": tok.state_dict()}):
            logger.log_artifact(store.best_path(), "cvq-tokenizer", "model", step, aliases=["best"])
            print(f"  new best ({score:.4f}) -> best.pt")
        return metrics, images

    def ckpt_fn(step, epoch):
        path = store.save(step, epoch, cfg,
                          model_state={"tokenizer": tok.state_dict(), "disc": disc.state_dict()},
                          opt_state={"opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()},
                          latest_model_keys=["tokenizer"])
        print(f"  saved {path.name}")

    # ---- loop ----
    cadence = Cadence(log_every=tcfg["log_every"], sample_every=tcfg["sample_every"],
                      val_every=tcfg.get("val_every", 1000), ckpt_every=tcfg["ckpt_every"])
    loop = TrainLoop(device=device, amp=amp, accum=tcfg["grad_accum"], cadence=cadence,
                     logger=logger, batch_size=tcfg["batch_size"],
                     disc_start_step=tcfg["disc_start_step"])
    step_runner = GANStep(disc, generator_fn, discriminator_fn,
                          accum=tcfg["grad_accum"], device=device, amp=amp)

    final_step = loop.run(
        dataloader=dl, epochs=tcfg["epochs"], start_step=start_step, start_epoch=start_epoch,
        step_runner=step_runner,
        optimizers=[opt_g, opt_d], schedulers=[sched_g, sched_d],
        gen_params=gen_params, disc_params=list(disc.parameters()),
        sample_fn=sample_fn, val_fn=val_fn, ckpt_fn=ckpt_fn,
        max_steps=args.max_steps,
    )

    # ---- final save + validation + artifact ----
    ckpt_fn(final_step, tcfg["epochs"] - 1)
    metrics, images = validate(tok, ds, device, batch_size=tcfg["batch_size"],
                               compute_fid=tcfg.get("val_fid", True),
                               lpips_fn=crit.perceptual)
    logger.log(metrics, final_step); logger.log_images(images, final_step)
    print("final val:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    final_score = metrics.get("val/rFID", metrics.get("val/recon_l2_full", float("inf")))
    aliases = ["latest", "best"] if final_score <= store.best_score else ["latest"]
    logger.log_artifact(store.latest_path(), "cvq-tokenizer", "model", final_step, aliases=aliases)
    logger.close()
    print("training complete.")


if __name__ == "__main__":
    main()
