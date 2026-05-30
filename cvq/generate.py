"""
Generate Pokemon from text prompts with a trained CAR + tokenizer.

    python -m cvq.generate --car_ckpt checkpoints_car/car_latest.pt \
        --tokenizer_ckpt checkpoints/best.pt --prompts "pikachu" "charizard" "mega rayquaza" \
        --cfg 3.0 --top_k 256 --out samples_car/gen.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from cvq.checkpoint import CheckpointStore
from cvq.conditioning import Conditioning
from cvq.models.car import CAR
from cvq.tokenizer_factory import build_tokenizer
from cvq.utils import resolve_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car_ckpt", default="checkpoints_car/car_latest.pt")
    ap.add_argument("--tokenizer_ckpt", default="checkpoints/best.pt")
    ap.add_argument("--prompts", nargs="+", default=["pikachu", "charizard", "bulbasaur"])
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="samples_car/gen.png")
    args = ap.parse_args()
    device = resolve_device(args.device)

    tok, _ = build_tokenizer({}, device, ckpt=args.tokenizer_ckpt)
    tok.eval()
    ck = CheckpointStore.load(args.car_ckpt, map_location=device)
    cfg = ck["config"]["model"]
    from transformers import AutoTokenizer
    qwen_name = cfg.get("qwen_name", "Qwen/Qwen3-0.6B-Base")
    text_tok = AutoTokenizer.from_pretrained(qwen_name)
    car = CAR(codebook_size=tok.quantizer.codebook_size, num_channels=tok.latent_channels,
              qwen_name=qwen_name).to(device)
    car.load_state_dict(ck["car"]); car.eval()

    cond = Conditioning(text_tok, max_len=cfg.get("max_text_len", 16), device=device)
    prompts = [p.replace("-", " ") for p in args.prompts]
    text_ids, text_mask = cond.encode_batch(prompts)
    text_ids = text_ids.to(device); text_mask = text_mask.to(device)
    uncond_ids = uncond_mask = None
    if args.cfg != 1.0:
        uncond_ids, uncond_mask = cond.unconditional(len(prompts), L=text_ids.shape[1], device=device)

    idxs = car.generate(text_ids, text_mask, temperature=args.temperature, top_k=args.top_k,
                        cfg_scale=args.cfg, uncond_text_ids=uncond_ids,
                        uncond_text_mask=uncond_mask)
    imgs = tok.decode(tok.quantizer.lookup(idxs))
    grid = make_grid((imgs.clamp(-1, 1) * 0.5 + 0.5), nrow=len(prompts))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out)
    print(f"saved {len(prompts)} generations -> {out}")


if __name__ == "__main__":
    main()
