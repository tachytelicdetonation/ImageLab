"""
End-to-end joint training of the CVQ/EOSTok tokenizer AND the CAR text-to-image model.

This is the faithful EOSTok objective (arXiv:2605.00503), mapped onto our channel-wise stack:

    L_E2E = L_VQVAE  +  lambda_NTP * L_NTP  +  lambda_APR * L_APR

  * L_VQVAE : image -> encoder -> IBQ -> decoder -> recon. Pixel l2 + LPIPS + PatchGAN +
              codebook/commitment/entropy (the tokenizer's own loss). Trains the tokenizer.
  * L_NTP   : CAR predicts each channel-code from [name][BOI][prefix]; cross-entropy against
              the tokenizer's own indices (labels detached). Trains the CAR.
  * L_APR   : Autoregressive Prediction Reconstruction. Take the CAR's teacher-forced SOFT
              prediction p_hat = softmax(logits), form z_q_apr = p_hat @ codebook, decode to
              pixels, and take a reconstruction loss vs the image. Differentiable, so generation
              feedback flows into BOTH the CAR and the tokenizer (codebook + decoder). This is
              EOSTok's keystone -- "direct supervision from generation to the tokenizer" -- and
              what prevents the NTP-only latent from collapsing into an unpredictable code soup.

Why SOFT codes for APR: it mirrors IBQ's own soft straight-through path (p @ C), so the APR
gradient reaches the codebook through the exact geometry the quantizer already uses.

Unlike the staged baseline (train_car.py, frozen tokenizer), here the tokenizer is TRAINABLE and
trained jointly from scratch, exactly as EOSTok argues for. The AR losses are ramped on only
after `ar_start_step` so the tokenizer is non-garbage before generation feedback kicks in
(analogous to the GAN's disc_start delay).

    python -m cvq.train_e2e --config configs/car_e2e_pokemon.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.data.car_dataset import CARPokemonDataset, CARCollate
from cvq.losses.losses import CVQLoss
from cvq.metrics import grad_norm
from cvq.models.car import CAR
from cvq.models.discriminator import NLayerDiscriminator
from cvq.models.tokenizer import CVQTokenizer
from cvq.utils import describe_device, resolve_device

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


def autocast_ctx(device, amp):
    if amp == "bf16" and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def lr_lambda(step, warmup):
    return min(1.0, (step + 1) / max(1, warmup))


def split_decay_groups(params, lr, weight_decay):
    params = [p for p in params if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def sample_c_keep(total_channels, prob, generator):
    if torch.rand((), generator=generator).item() >= prob:
        return None
    return int(torch.randint(1, total_channels + 1, (), generator=generator).item())


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
    tok = CVQTokenizer(
        resolution=cfg["data"]["size"],
        latent_channels=mcfg["latent_channels"],
        codebook_size=mcfg["codebook_size"], commitment_beta=mcfg["commitment_beta"],
        quantizer_kwargs=mcfg.get("quantizer_kwargs", None),
        enc_ch=mcfg.get("enc_ch", 128), enc_ch_mult=tuple(mcfg.get("enc_ch_mult", [1, 1, 2, 2, 4])),
        decoder_ch=mcfg["decoder_ch"], decoder_ch_mult=tuple(mcfg["decoder_ch_mult"]),
        decoder_res_blocks=mcfg["decoder_res_blocks"],
    ).to(device)
    if args.tokenizer_ckpt and Path(args.tokenizer_ckpt).exists():
        ck = torch.load(args.tokenizer_ckpt, map_location=device)
        tok.load_state_dict(ck["tokenizer"], strict=False)
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

    # ---- DINOv2 semantic alignment (EOSTok L_implicit, optional) ----
    lam_sem = tcfg.get("lambda_sem", 0.0)
    dino = None
    if lam_sem > 0:
        from cvq.models.dino_align import DINOAlign
        dino = DINOAlign(latent_channels=C, grid=tok.grid,
                         dino_name=mcfg.get("dino_name", "facebook/dinov2-large")).to(device)
        print(f"DINOv2 alignment: ON | lambda_sem={lam_sem} | {mcfg.get('dino_name', 'facebook/dinov2-large')}")

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
        # h_omega (the alignment projector) is the only trainable part of DINOAlign.
        g_groups = g_groups + split_decay_groups(list(dino.proj.parameters()), tcfg["lr"], wd)
    opt_g = torch.optim.AdamW(g_groups, betas=betas, weight_decay=wd)
    opt_d = torch.optim.AdamW(split_decay_groups(list(disc.parameters()), tcfg["lr"], wd),
                              betas=betas, weight_decay=wd)
    gen_params = [p for grp in g_groups for p in grp["params"]]
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: lr_lambda(s, tcfg["warmup_steps"]))
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lambda s: lr_lambda(s, tcfg["warmup_steps"]))

    lam_ntp = tcfg["lambda_ntp"]
    lam_apr = tcfg["lambda_apr"]
    ar_start = tcfg["ar_start_step"]
    apr_lpips_w = tcfg.get("apr_lpips_weight", 0.0)

    # ---- classifier-free guidance: caption dropout ----
    # With prob cond_drop, replace a sample's caption with the EMPTY string so the CAR learns
    # p(image | "") — the unconditional distribution CFG interpolates against at sampling time.
    # Without this, generate()'s uncond branch is untrained and CFG (cfg_scale>1) is meaningless.
    # The empty tokenization here MUST match sample_generations()'s `text_tok([""]*n, ...)`.
    cond_drop = tcfg.get("cond_dropout_prob", 0.0)
    if cond_drop > 0:
        _emp = text_tok([""], padding="max_length", truncation=True,
                        max_length=mcfg.get("max_text_len", 16), return_tensors="pt")
        empty_ids = _emp["input_ids"][0].to(device)      # (L,)
        empty_mask = _emp["attention_mask"][0].to(device)  # (L,)
        print(f"caption dropout: ON | p={cond_drop} (CFG-enabled)")

    ckpt_dir = Path(ocfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = Path(ocfg["sample_dir"]); sample_dir.mkdir(parents=True, exist_ok=True)

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

    sample_prompts = mcfg.get("sample_prompts",
                              ["pikachu", "charizard", "bulbasaur", "mewtwo",
                               "rayquaza mega", "gengar", "eevee", "snorlax"])
    denorm = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5)
    accum = tcfg.get("grad_accum", 1)
    total_channels = C
    fixed_batch = next(iter(DataLoader(ds, batch_size=8)))["image"][:8].to(device)

    step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        tok.load_state_dict(ck["tokenizer"], strict=False); car.load_state_dict(ck["car"])
        disc.load_state_dict(ck["disc"]); opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"]); step = ck["step"]
        print(f"resumed from {args.resume} @ step {step}")

    cb = tok.quantizer.embed.weight  # (K, D) — the live codebook for APR
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        tok.train(); car.train(); disc.train()
        for i, batch in enumerate(dl):
            x = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            c_keep = sample_c_keep(total_channels, tcfg["nested_dropout_prob"], gen)
            ar_on = step >= ar_start

            if i % accum == 0:
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)

            for p in disc.parameters():
                p.requires_grad_(False)
            with autocast_ctx(device, amp):
                out = tok(x, c_keep=c_keep)
                recon, vq_loss = out["recon"], out["vq_loss"]
                idxs = out["indices"]                                  # (B, C), tokenizer's own
                last_layer = tok.decoder.conv_out.weight
                g_total, g_logs = crit.generator_step(
                    target=x, recon=recon, vq_loss=vq_loss, discriminator=disc,
                    last_layer=last_layer, global_step=step, c_keep=c_keep,
                    total_channels=total_channels,
                )
                ntp_loss = recon.new_zeros(())
                apr_loss = recon.new_zeros(())
                ntp_logs = {}
                if ar_on:
                    # ---- NTP: CAR predicts the tokenizer's indices (labels detached) ----
                    logits = car(text_ids, text_mask, idxs.detach())  # (B, C, K)
                    ntp_loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, K), idxs.detach().reshape(-1))
                    with torch.no_grad():
                        acc = (logits.argmax(-1) == idxs).float().mean()
                    # ---- APR: decode CAR's soft prediction back to pixels ----
                    p_hat = logits.softmax(-1)                         # (B, C, K)
                    z_q_apr = torch.einsum("bck,kd->bcd", p_hat.float(), cb.float())
                    side = int(round((z_q_apr.shape[-1]) ** 0.5))
                    z_q_apr = z_q_apr.reshape(z_q_apr.shape[0], C, side, side).to(recon.dtype)
                    recon_apr = tok.decoder(z_q_apr)
                    apr_loss = torch.nn.functional.mse_loss(recon_apr, x)
                    if apr_lpips_w > 0:
                        apr_loss = apr_loss + apr_lpips_w * crit.perceptual(recon_apr, x).mean()
                    ntp_logs = {"car/ntp_loss": ntp_loss.item(), "car/token_acc": acc.item(),
                                "car/apr_loss": apr_loss.item()}
                sem_loss = recon.new_zeros(())
                if dino is not None:
                    sem_loss = dino(out["z"], x)                   # EOSTok L_implicit (cosine)
                    ntp_logs["car/sem_loss"] = sem_loss.item()
                total = g_total + lam_ntp * ntp_loss + lam_apr * apr_loss + lam_sem * sem_loss
            (total / accum).backward()
            for p in disc.parameters():
                p.requires_grad_(True)

            with autocast_ctx(device, amp):
                d_loss, d_logs = crit.discriminator_step(x, recon, disc, step)
            if torch.is_tensor(d_loss) and d_loss.requires_grad:
                (d_loss / accum).backward()

            gn_g = 0.0
            if (i + 1) % accum == 0:
                gn_g = grad_norm(gen_params)
                torch.nn.utils.clip_grad_norm_(gen_params, tcfg.get("grad_clip", 1.0))
                opt_g.step(); sched_g.step()
                if step >= tcfg["disc_start_step"]:
                    opt_d.step(); sched_d.step()

            if step % tcfg["log_every"] == 0:
                ips = (step + 1) * tcfg["batch_size"] / (time.time() - t0)
                ck_str = "full" if c_keep is None else str(c_keep)
                ar_str = (f"ntp {ntp_logs['car/ntp_loss']:.3f} acc {ntp_logs['car/token_acc']:.3f} "
                          f"apr {ntp_logs['car/apr_loss']:.3f} " if ntp_logs else "ar:off ")
                print(f"e{epoch} s{step} | tot {g_logs['loss/total']:.3f} rec {g_logs['loss/recon']:.3f} "
                      f"lpips {g_logs['loss/lpips']:.3f} vq {g_logs['loss/vq']:.4f} d {d_logs['loss/disc']:.3f} "
                      f"| {ar_str}| use {out['stats']['usage']:.3f} ppl {out['stats']['perplexity']:.0f} "
                      f"ck {ck_str} | {ips:.1f} img/s")
                scalars = {**g_logs, **d_logs, **ntp_logs}
                scalars.update({
                    "codebook/usage_batch": out["stats"]["usage"],
                    "codebook/perplexity": out["stats"]["perplexity"],
                    "codebook/entropy_loss": out["stats"].get("entropy_loss", 0.0),
                    "opt/lr_g": sched_g.get_last_lr()[0], "opt/grad_norm_g": gn_g,
                    "train/ar_on": float(ar_on), "train/epoch": epoch, "train/img_per_s": ips,
                })
                if device == "cuda":
                    scalars["sys/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                wlog(scalars, step)

            if step > 0 and step % tcfg["sample_every"] == 0:
                # reconstruction grid + (if AR on) text->image generations
                tok.eval(); car.eval()
                with torch.no_grad():
                    r = tok(fixed_batch)["recon"]
                save_image(make_grid(torch.cat([denorm(fixed_batch), denorm(r)], 0), nrow=8),
                           sample_dir / f"recon_{step:06d}.png")
                if ar_on:
                    sample_generations(car, tok, text_tok, sample_prompts, device, amp,
                                       sample_dir, step, denorm, use_wandb,
                                       mcfg.get("max_text_len", 16),
                                       mcfg.get("cfg_scale", 1.0), mcfg.get("temperature", 1.0),
                                       mcfg.get("top_k", 0))
                tok.train(); car.train()

            if step > 0 and step % tcfg["ckpt_every"] == 0:
                save_ckpt(ckpt_dir, tok, car, disc, opt_g, opt_d, step, epoch, cfg)

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    save_ckpt(ckpt_dir, tok, car, disc, opt_g, opt_d, step, epoch, cfg)
    if use_wandb:
        wandb.finish()
    print("E2E training complete.")


@torch.no_grad()
def sample_generations(car, tok, text_tok, prompts, device, amp, sample_dir, step, denorm,
                       use_wandb, max_len, cfg_scale, temperature, top_k):
    enc = text_tok(prompts, padding="longest", truncation=True, max_length=max_len,
                   return_tensors="pt")
    text_ids, text_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    uncond_ids = uncond_mask = None
    if cfg_scale != 1.0:
        unc = text_tok([""] * len(prompts), padding="max_length", max_length=text_ids.shape[1],
                       return_tensors="pt")
        uncond_ids, uncond_mask = unc["input_ids"].to(device), unc["attention_mask"].to(device)
    with autocast_ctx(device, amp):
        idxs = car.generate(text_ids, text_mask, temperature=temperature, top_k=top_k,
                            cfg_scale=cfg_scale, uncond_text_ids=uncond_ids,
                            uncond_text_mask=uncond_mask)
        imgs = tok.decode(tok.quantizer.lookup(idxs))
    # .float().cpu(): grids are bf16 under autocast; wandb.Image -> numpy() has no bfloat16 dtype.
    grid = make_grid(denorm(imgs).float().cpu(), nrow=len(prompts))
    save_image(grid, sample_dir / f"gen_{step:06d}.png")
    print(f"  sampled {len(prompts)} prompts -> gen_{step:06d}.png")
    if use_wandb and _HAS_WANDB:
        wandb.log({"images/generations": wandb.Image(grid, caption=" | ".join(prompts))}, step=step)


def save_ckpt(ckpt_dir, tok, car, disc, opt_g, opt_d, step, epoch, cfg):
    path = ckpt_dir / f"e2e_step{step:06d}.pt"
    torch.save({"tokenizer": tok.state_dict(), "car": car.state_dict(), "disc": disc.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                "step": step, "epoch": epoch, "config": cfg}, path)
    torch.save({"tokenizer": tok.state_dict(), "car": car.state_dict(), "config": cfg, "step": step},
               ckpt_dir / "e2e_latest.pt")
    print(f"  saved {path.name}")


if __name__ == "__main__":
    main()
