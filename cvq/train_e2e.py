"""
End-to-end joint training of the CVQ/EOSTok tokenizer AND the CAR text-to-image model.

Faithful EOSTok objective (arXiv:2605.00503), mapped onto our channel-wise stack:

    L_E2E = L_VQVAE  +  lambda_NTP * L_NTP  +  lambda_APR * L_APR  (+ lambda_sem * L_implicit)

The training scaffolding (AMP / accum / GAN disc-freeze dance / cadenced sample+ckpt /
wandb) lives in cvq.training_loop. The conditioning protocol (prompts + caption-dropout
+ CFG uncond, with the matched-empty-prompt invariant) lives in cvq.conditioning. This
script only declares per-step work.

    python -m cvq.train_e2e --config configs/car_e2e_pokemon.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.checkpoint import CheckpointStore
from cvq.conditioning import Conditioning
from cvq.data.car_dataset import CARPokemonDataset, CARCollate
from cvq.losses.losses import CVQLoss
from cvq.models.car import CAR
from cvq.models.discriminator import NLayerDiscriminator
from cvq.nested_dropout import HybridUniformPolicy, channel_weights
from cvq.tokenizer_factory import build_tokenizer
from cvq.training_loop import (
    Cadence, GANStep, RunLogger, StepOutput, TrainLoop,
    autocast_ctx, split_decay_groups, warmup_lr_lambda,
)
from cvq.utils import describe_device, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/car_e2e_pokemon.yaml")
    ap.add_argument("--tokenizer_ckpt", default="", help="optional warm-start for the tokenizer")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    tcfg, mcfg, lcfg, ocfg = cfg["train"], cfg["model"], cfg["loss"], cfg["out"]

    device = resolve_device(tcfg["device"])
    amp = tcfg.get("amp", "none")
    print(f"device: {describe_device(device)} | amp: {amp}")
    torch.manual_seed(tcfg["seed"])
    gen = torch.Generator().manual_seed(tcfg["seed"])

    # ---- tokenizer (TRAINABLE, joint) ----
    tok, _ = build_tokenizer(cfg, device,
                             ckpt=args.tokenizer_ckpt if args.tokenizer_ckpt else None)
    if args.tokenizer_ckpt:
        print(f"tokenizer warm-started from {args.tokenizer_ckpt}")
    K = tok.quantizer.codebook_size
    C = tok.latent_channels
    print(f"tokenizer: TRAINABLE | K={K} | channels={C}")

    disc = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)
    crit = CVQLoss(
        disc_start=tcfg["disc_start_step"], recon_loss_type=lcfg["recon_loss_type"],
        perceptual_weight=lcfg["perceptual_weight"], codebook_weight=lcfg["codebook_weight"],
        disc_weight=lcfg["disc_weight"], disc_loss=lcfg["disc_loss"],
        lpips_net=lcfg["lpips_net"], gan_eta=lcfg["gan_eta"],
    ).to(device)

    # ---- CAR (Qwen backbone + image vocab) ----
    from transformers import AutoTokenizer
    qwen_name = mcfg.get("qwen_name", "Qwen/Qwen3-0.6B-Base")
    text_tok = AutoTokenizer.from_pretrained(qwen_name)
    car = CAR(codebook_size=K, num_channels=C, qwen_name=qwen_name,
              freeze_backbone=mcfg.get("freeze_backbone", False),
              attn_impl=mcfg.get("attn_impl", "sdpa")).to(device)
    print(f"CAR: {qwen_name} | trainable {sum(p.numel() for p in car.trainable_parameters())/1e6:.1f}M "
          f"| freeze_backbone={mcfg.get('freeze_backbone', False)}")

    # ---- conditioning (single seam for caption-dropout + CFG uncond) ----
    cond = Conditioning(text_tok, max_len=mcfg.get("max_text_len", 16),
                        p_uncond=tcfg.get("cond_dropout_prob", 0.0),
                        generator=gen, device=device)
    if cond.p_uncond > 0:
        print(f"caption dropout: ON | p={cond.p_uncond} (CFG-enabled)")

    # ---- DINOv2 semantic alignment (EOSTok L_implicit, optional) ----
    lam_sem = tcfg.get("lambda_sem", 0.0)
    dino = None
    if lam_sem > 0:
        from cvq.models.dino_align import DINOAlign
        dino = DINOAlign(latent_channels=C, grid=tok.grid,
                         dino_name=mcfg.get("dino_name", "facebook/dinov2-large")).to(device)
        print(f"DINOv2 alignment: ON | lambda_sem={lam_sem} | {mcfg.get('dino_name', 'facebook/dinov2-large')}")

    # ---- nested channel dropout policy ----
    total_channels = C
    nested = HybridUniformPolicy(total_channels, tcfg["nested_dropout_prob"], generator=gen)

    # ---- channel-weight schedule for NTP / APR (couples CVQ ordering to EOSTok losses) ----
    # schedule="uniform" reproduces the old behaviour exactly.
    cw_schedule = tcfg.get("channel_weight_schedule", "linear")
    cw_alpha = tcfg.get("channel_weight_alpha", 1.0)
    chan_w = channel_weights(total_channels, cw_schedule, cw_alpha,
                             device=device, dtype=torch.float32)
    # Per-pixel weight for the APR L2/LPIPS terms: broadcast each channel's weight across
    # spatial dims so wrong pixels in coarse-channel regions cost more. (APR decodes to RGB,
    # so we keep this 1-D over channels of the decoder INPUT, not the RGB output.)
    if cw_schedule != "uniform":
        print(f"channel-weight schedule: {cw_schedule}"
              + (f" (alpha={cw_alpha})" if cw_schedule in ("linear", "exp") else "")
              + f" | early/late ratio = {chan_w[0].item()/chan_w[-1].item():.2f}")

    # ---- data ----
    ds = CARPokemonDataset(cfg["data"]["root"], size=cfg["data"]["size"], hflip=cfg["data"]["hflip"],
                           augment=cfg["data"].get("augment", False))
    collate = CARCollate(text_tok, max_len=mcfg.get("max_text_len", 16))
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True,
                    num_workers=tcfg["num_workers"], drop_last=True, collate_fn=collate)
    print(f"dataset: {len(ds)} images | {len(dl)} batches/epoch")

    # ---- optimizers ----
    betas = (tcfg["beta1"], tcfg["beta2"])
    wd = tcfg["weight_decay"]
    tok_groups = split_decay_groups(tok.trainable_parameters(), tcfg["lr"], wd)
    car_groups = split_decay_groups(car.trainable_parameters(), tcfg.get("car_lr", tcfg["lr"]), wd)
    g_groups = tok_groups + car_groups
    if dino is not None:
        g_groups = g_groups + split_decay_groups(list(dino.proj.parameters()), tcfg["lr"], wd)
    opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=wd)
    opt_d = torch.optim.AdamW(split_decay_groups(list(disc.parameters()), tcfg["lr"], wd),
                              betas=betas, weight_decay=wd)
    gen_params = [p for grp in g_groups for p in grp["params"]]
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: warmup_lr_lambda(s, tcfg["warmup_steps"]))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: warmup_lr_lambda(s, tcfg["warmup_steps"]))

    lam_ntp = tcfg["lambda_ntp"]
    lam_apr = tcfg["lambda_apr"]
    ar_start = tcfg["ar_start_step"]
    apr_lpips_w = tcfg.get("apr_lpips_weight", 0.0)
    # Floor for the APR c_keep: with c_keep=1 the AR is asked to decode the full RGB image
    # from a single channel-token prediction, which burns gradient on impossible decodes. The
    # tokenizer's own recon-loop accepts this regime (it has all C real channels); the AR's
    # APR does not. Default 0.25 -> APR never decodes with fewer than C/4 channels; 0 disables.
    apr_min_c_keep_frac = tcfg.get("apr_min_c_keep_frac", 0.25)
    apr_min_c_keep = max(1, int(round(apr_min_c_keep_frac * total_channels))) if apr_min_c_keep_frac > 0 else 1

    store = CheckpointStore(ocfg["ckpt_dir"], prefix="e2e", latest_name="e2e_latest.pt",
                            keep_last=ocfg.get("keep_last", 5))
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(cfg, run_dir=ocfg.get("run_dir"))

    sample_prompts = mcfg.get("sample_prompts",
                              ["pikachu", "charizard", "bulbasaur", "mewtwo",
                               "rayquaza mega", "gengar", "eevee", "snorlax"])
    denorm = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5)
    accum = tcfg.get("grad_accum", 1)
    fixed_batch = next(iter(DataLoader(ds, batch_size=8)))["image"][:8].to(device)

    # ---- resume ----
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = CheckpointStore.load(args.resume, map_location=device)
        tok.load_state_dict(ck["tokenizer"], strict=False); car.load_state_dict(ck["car"])
        disc.load_state_dict(ck["disc"]); opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"]); start_step = ck["step"]
        print(f"resumed from {args.resume} @ step {start_step}")

    cb = tok.quantizer.embed.weight  # (K, D) — live codebook for APR

    # ---- per-step work ----
    def generator_fn(batch, step):
        x = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        text_ids, text_mask = cond.maybe_drop(text_ids, text_mask)
        c_keep = nested.sample(step)
        ar_on = step >= ar_start

        out = tok(x, c_keep=c_keep)
        recon, vq_loss = out["recon"], out["vq_loss"]
        idxs = out["indices"]                                  # (B, C) tokenizer's own
        last_layer = tok.decoder.conv_out.weight
        g_total, g_logs = crit.generator_step(
            target=x, recon=recon, vq_loss=vq_loss, discriminator=disc,
            last_layer=last_layer, global_step=step, c_keep=c_keep,
            total_channels=total_channels,
        )
        ntp_loss = recon.new_zeros(())
        apr_loss = recon.new_zeros(())
        ar_logs = {}
        if ar_on:
            logits = car(text_ids, text_mask, idxs.detach())  # (B, C, K)
            # --- channel-weighted NTP (couples CVQ ordering to EOSTok's NTP) ---
            ce_pt = torch.nn.functional.cross_entropy(
                logits.reshape(-1, K), idxs.detach().reshape(-1),
                reduction="none",
            ).reshape(logits.shape[0], C)              # (B, C)
            ntp_loss = (ce_pt * chan_w[None, :]).mean()  # mean(w)=1, scale preserved
            with torch.no_grad():
                acc = (logits.argmax(-1) == idxs).float().mean()
                # Diagnostic: prefix accuracy on the first 25% of channels.
                cprefix = max(1, C // 4)
                acc_prefix = (logits[:, :cprefix].argmax(-1) ==
                              idxs[:, :cprefix]).float().mean()

            # --- APR with prefix-fidelity supervision ---
            # Decode CAR's soft prediction back to pixels, but truncate to the SAME c_keep
            # nested dropout drew this step. This forces the AR's prefix (early channels)
            # to be sufficient for a coherent low-frequency reconstruction -- exactly what
            # CVQ's nested dropout taught the tokenizer's decoder. Without this, APR only
            # ever supervises the full 256-channel decode and never rewards prefix fidelity.
            p_hat = logits.softmax(-1)
            z_q_apr = torch.einsum("bck,kd->bcd", p_hat.float(), cb.float())
            side = int(round((z_q_apr.shape[-1]) ** 0.5))
            z_q_apr = z_q_apr.reshape(z_q_apr.shape[0], C, side, side).to(recon.dtype)
            # Prefix-fidelity coupling. Floor c_keep so APR doesn't supervise on impossibly-
            # short prefixes (see apr_min_c_keep_frac above).
            c_keep_apr = max(c_keep, apr_min_c_keep) if c_keep is not None else None
            z_q_apr = nested.apply(z_q_apr, c_keep_apr)
            recon_apr = tok.decoder(z_q_apr)
            apr_loss = torch.nn.functional.mse_loss(recon_apr, x)
            if apr_lpips_w > 0:
                apr_loss = apr_loss + apr_lpips_w * crit.perceptual(recon_apr, x).mean()
            ar_logs = {"car/ntp_loss": ntp_loss.item(), "car/token_acc": acc.item(),
                       "car/token_acc_prefix": acc_prefix.item(),
                       "car/apr_loss": apr_loss.item(),
                       "car/apr_c_keep": (c_keep_apr if c_keep_apr is not None else C)}
        sem_loss = recon.new_zeros(())
        if dino is not None:
            sem_loss = dino(out["z"], x)
            ar_logs["car/sem_loss"] = sem_loss.item()

        total = g_total + lam_ntp * ntp_loss + lam_apr * apr_loss + lam_sem * sem_loss

        logs = dict(g_logs)
        logs.update(ar_logs)
        logs.update({
            "codebook/usage_batch": out["stats"]["usage"],
            "codebook/perplexity": out["stats"]["perplexity"],
            "codebook/entropy_loss": out["stats"].get("entropy_loss", 0.0),
            "train/ar_on": float(ar_on),
            "train/c_keep": c_keep if c_keep is not None else total_channels,
        })
        extras = {"x": x, "recon": recon}
        return StepOutput(loss=total, logs=logs, extras=extras)

    def discriminator_fn(batch, step, extras):
        return crit.discriminator_step(extras["x"], extras["recon"], disc, step)

    # ---- callbacks ----
    def sample_fn(step):
        tok.eval(); car.eval()
        imgs_out = {}
        with torch.no_grad():
            r = tok(fixed_batch)["recon"]
        recon_grid = make_grid(torch.cat([denorm(fixed_batch), denorm(r)], 0), nrow=8)
        save_image(recon_grid, sample_dir / f"recon_{step:06d}.png")
        imgs_out["reconstructions"] = recon_grid
        if step >= ar_start:
            gen_grid = sample_generations(
                car, tok, cond, sample_prompts, device, amp, sample_dir, step, denorm,
                mcfg.get("cfg_scale", 1.0), mcfg.get("temperature", 1.0), mcfg.get("top_k", 0),
            )
            if gen_grid is not None:
                imgs_out["generations"] = gen_grid
        tok.train(); car.train()
        return imgs_out

    def ckpt_fn(step, epoch):
        path = store.save(step, epoch, cfg,
                          model_state={"tokenizer": tok.state_dict(),
                                       "car": car.state_dict(),
                                       "disc": disc.state_dict()},
                          opt_state={"opt_g": opt_g.state_dict(),
                                     "opt_d": opt_d.state_dict()},
                          latest_model_keys=["tokenizer", "car"])
        print(f"  saved {path.name}")

    # ---- loop ----
    cadence = Cadence(log_every=tcfg["log_every"], sample_every=tcfg["sample_every"],
                      val_every=tcfg.get("val_every", 10_000_000),
                      ckpt_every=tcfg["ckpt_every"])
    loop = TrainLoop(device=device, amp=amp, accum=accum, cadence=cadence,
                     logger=logger, batch_size=tcfg["batch_size"],
                     disc_start_step=tcfg["disc_start_step"])
    step_runner = GANStep(disc, generator_fn, discriminator_fn,
                          accum=accum, device=device, amp=amp)
    final_step = loop.run(
        dataloader=dl, epochs=tcfg["epochs"], start_step=start_step, start_epoch=0,
        step_runner=step_runner,
        optimizers=[opt_g, opt_d], schedulers=[sched_g, sched_d],
        gen_params=gen_params, disc_params=list(disc.parameters()),
        grad_clip=tcfg.get("grad_clip", 1.0),
        sample_fn=sample_fn, ckpt_fn=ckpt_fn,
        max_steps=args.max_steps,
    )

    ckpt_fn(final_step, tcfg["epochs"] - 1)
    logger.close()
    print("E2E training complete.")


@torch.no_grad()
def sample_generations(car, tok, cond: Conditioning, prompts, device, amp, sample_dir, step,
                       denorm, cfg_scale, temperature, top_k):
    text_ids, text_mask = cond.encode_batch(prompts)
    text_ids = text_ids.to(device); text_mask = text_mask.to(device)
    uncond_ids = uncond_mask = None
    if cfg_scale != 1.0:
        uncond_ids, uncond_mask = cond.unconditional(len(prompts), L=text_ids.shape[1], device=device)
    with autocast_ctx(device, amp):
        idxs = car.generate(text_ids, text_mask, temperature=temperature, top_k=top_k,
                            cfg_scale=cfg_scale, uncond_text_ids=uncond_ids,
                            uncond_text_mask=uncond_mask)
        imgs = tok.decode(tok.quantizer.lookup(idxs))
    grid = make_grid(denorm(imgs).float().cpu(), nrow=len(prompts))
    save_image(grid, sample_dir / f"gen_{step:06d}.png")
    print(f"  sampled {len(prompts)} prompts -> gen_{step:06d}.png")
    return grid


if __name__ == "__main__":
    main()
