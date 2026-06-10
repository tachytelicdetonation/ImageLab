"""
Standalone evaluation: the full metric suite for any checkpoint, one command.

    # tokenizer-only checkpoint (latest.pt / best.pt) or e2e checkpoint (e2e_latest.pt):
    python -m cvq.evaluate --ckpt checkpoints/latest.pt
    python -m cvq.evaluate --ckpt checkpoints_e2e/e2e_latest.pt --generate

Reports rFID / PSNR / SSIM / LPIPS / recon L2 / codebook utilization / per-c_keep recon
over the checkpoint's own dataset (per `dataset` tag when the manifest is multi-source),
plus generation grids when the checkpoint contains a CAR and --generate is set.
Writes a JSON next to the grids so runs are comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cvq.checkpoint import CheckpointStore
from cvq.config import from_dict
from cvq.data.car_dataset import CARCollate, CaptionedImageDataset
from cvq.data.dataset import ManifestImageDataset
from cvq.eval import GroupedEvaluator, validate
from cvq.factory import (build_car, build_conditioning, build_text_tokenizer,
                         build_tokenizer)
from cvq.utils import describe_device, resolve_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="tokenizer or e2e checkpoint (.pt)")
    ap.add_argument("--data-root", default="", help="override the checkpoint's data root")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-fid", action="store_true")
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="which split to evaluate on (default: the held-out val split)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap the full-dataset pass (0 = all; useful for quick checks)")
    ap.add_argument("--generate", action="store_true", help="also sample generations (needs a CAR)")
    ap.add_argument("--out", default="samples/eval")
    args = ap.parse_args()
    device = resolve_device(args.device)
    print(f"device: {describe_device(device)}")

    ck = CheckpointStore.load(args.ckpt, map_location=device)
    raw = ck["config"]
    if args.data_root:
        raw["data"]["root"] = args.data_root
    rc = from_dict(raw)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    tok, _ = build_tokenizer(rc, device, ckpt=args.ckpt)
    tok.eval()

    import lpips
    lpips_fn = lpips.LPIPS(net=rc.loss.lpips_net).eval().to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    # ---- full-split tokenizer metrics ----
    split = None if args.split == "all" else args.split
    ds = ManifestImageDataset(rc.data.root, size=rc.data.size, hflip=False,
                              split=split, val_fraction=rc.data.val_fraction)
    print(f"evaluating on split={args.split} ({len(ds)} images)")
    metrics, images = validate(tok, ds, device, batch_size=rc.train.batch_size,
                               compute_fid=not args.no_fid, lpips_fn=lpips_fn,
                               max_images=args.max_images)

    # ---- per-source split (multi-dataset manifests) ----
    has_car = "car" in ck
    if has_car:
        text_tok = build_text_tokenizer(rc.model)
        car = build_car(rc.model, tok.quantizer.codebook_size, tok.latent_channels, device)
        car.load_state_dict(ck["car"]); car.eval()
        cds = CaptionedImageDataset(rc.data.root, size=rc.data.size, hflip=False,
                                    split=split, val_fraction=rc.data.val_fraction)
        ev = GroupedEvaluator(cds, collate=CARCollate(text_tok, max_len=rc.model.max_text_len),
                              device=device, sample_dir=out_dir)
        gm, gi = ev.eval_recon(tok, perceptual=lpips_fn, car=car, step=ck.get("step", 0))
        metrics.update(gm); images.update(gi)
        if args.generate:
            cond = build_conditioning(rc.model, text_tok, device=device)
            prompts = rc.model.sample_prompts_by_dataset or {"all": rc.model.sample_prompts}
            ev.eval_generation(car, tok, cond, prompts, amp=rc.train.amp,
                               step=ck.get("step", 0), temperature=rc.model.temperature,
                               top_k=rc.model.top_k, cfg_scale=rc.model.cfg_scale)
    else:
        ev = GroupedEvaluator(ds, device=device, sample_dir=out_dir)
        gm, gi = ev.eval_recon(tok, perceptual=lpips_fn, step=ck.get("step", 0))
        metrics.update(gm)

    # ---- report ----
    from torchvision.utils import save_image
    for name, grid in images.items():
        save_image(grid, out_dir / (name.replace("/", "_") + ".png"))
    print(f"\n=== {args.ckpt} @ step {ck.get('step', '?')} ===")
    for k in sorted(metrics):
        v = metrics[k]
        print(f"  {k:42s} {v:.4f}" if isinstance(v, float) else f"  {k:42s} {v}")
    report = {"ckpt": str(args.ckpt), "step": ck.get("step"), "split": args.split,
              "metrics": metrics}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir}/metrics.json + image grids")


if __name__ == "__main__":
    main()
