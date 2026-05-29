"""
Reconstruct Pokemon with a trained CVQ tokenizer and report codebook utilization.

Also renders the coarse-to-fine progression: the same image decoded while keeping an
increasing number of leading channels (nested dropout). If the channel ordering worked,
early channels give a blurry global sketch and later channels add fine detail — the
visual signature of Channel-wise VQ.

    python -m cvq.reconstruct --ckpt checkpoints/latest.pt --n 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from cvq.data.dataset import PokemonDataset
from cvq.models.tokenizer import CVQTokenizer
from cvq.utils import resolve_device


def build_from_ckpt(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    m = cfg["model"]
    tok = CVQTokenizer(
        encoder_type=m.get("encoder_type", "siglip"),
        siglip_name=m["siglip_name"], siglip_layers=m["siglip_layers"],
        freeze_encoder=m["freeze_encoder"], resolution=cfg["data"]["size"],
        latent_channels=m["latent_channels"],
        codebook_size=m["codebook_size"], commitment_beta=m["commitment_beta"],
        entropy_weight=m.get("entropy_weight", 0.0),
        entropy_temperature=m.get("entropy_temperature", 1.0),
        enc_ch=m.get("enc_ch", 128), enc_ch_mult=tuple(m.get("enc_ch_mult", [1, 1, 2, 2, 4])),
        decoder_ch=m["decoder_ch"], decoder_ch_mult=tuple(m["decoder_ch_mult"]),
        decoder_res_blocks=m["decoder_res_blocks"],
    ).to(device)
    tok.load_state_dict(ck["tokenizer"], strict=False)  # encoder comes from HF
    tok.eval()
    return tok, cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--n", type=int, default=8, help="images to reconstruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="samples/eval")
    args = ap.parse_args()
    device = resolve_device(args.device)

    tok, cfg = build_from_ckpt(args.ckpt, device)
    ds = PokemonDataset(cfg["data"]["root"], size=cfg["data"]["size"], hflip=False)
    dl = DataLoader(ds, batch_size=args.n, shuffle=True)
    batch = next(iter(dl))
    x = batch["image"].to(device)

    out = tok(x)
    recon = out["recon"]
    denorm = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(torch.cat([denorm(x), denorm(recon)], 0), nrow=args.n)
    save_image(grid, outdir / "originals_vs_recon.png")

    # ---- codebook utilization over the whole dataset ----
    used = torch.zeros(tok.quantizer.codebook_size, dtype=torch.bool, device=device)
    full = DataLoader(ds, batch_size=cfg["train"]["batch_size"])
    for b in full:
        idxs = tok(b["image"].to(device))["indices"]
        used[idxs.reshape(-1).unique()] = True
    util = used.float().mean().item()
    print(f"codebook utilization over dataset: {util*100:.1f}% "
          f"({int(used.sum())}/{tok.quantizer.codebook_size})")

    # ---- coarse-to-fine progression for the first image ----
    C = tok.latent_channels
    keeps = sorted(set([max(1, C // 32), C // 8, C // 4, C // 2, C]))
    prog = [denorm(tok(x[:1], c_keep=k)["recon"]) for k in keeps]
    save_image(make_grid(torch.cat([denorm(x[:1])] + prog, 0), nrow=len(prog) + 1),
               outdir / "coarse_to_fine.png")
    print(f"coarse-to-fine c_keep steps: {['orig'] + keeps}")
    print(f"saved visualizations to {outdir}/")


if __name__ == "__main__":
    main()
