"""
Grouped evaluation — the same metric suite for every run, split by dataset source.

A manifest can tag each record with a `dataset` source (e.g. imagenette|imagewoof). The
evaluator holds one fixed batch per source and reports reconstruction + AR metrics per
group AND combined, so a single run gives an easy-vs-hard breakdown. Single-source sets
collapse to one "all" group.

This was previously inlined in train_e2e.py's sample_fn; it now serves every task and the
standalone `python -m cvq.evaluate` CLI.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

from cvq.conditioning import Conditioning
from cvq.training_loop import autocast_ctx
from cvq.utils import denorm


class GroupedEvaluator:
    """Fixed per-source eval batches + the recon/AR/generation metric passes.

    Args:
        ds: dataset (optionally wrapping a `.base` ManifestImageDataset).
        collate: optional collate_fn (the CAR text-tokenizing collate). Default stacking
                 is used when None (tokenizer-only eval).
        device: where eval batches live during a pass.
        sample_dir: where image grids are written (created if missing).
        n_eval: images per group in the fixed recon batch.
    """

    def __init__(self, ds, collate=None, device="cpu", sample_dir="samples",
                 n_eval: int = 8):
        self.device = device
        self.sample_dir = Path(sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        base_records = ds.base.records if hasattr(ds, "base") else ds.records
        tag_idxs = defaultdict(list)
        for i, r in enumerate(base_records):
            tag_idxs[r.get("dataset", "all")].append(i)
        self.eval_batches = OrderedDict()
        for tag in sorted(tag_idxs):
            idxs = tag_idxs[tag][:n_eval]
            sub = Subset(ds, idxs)
            self.eval_batches[tag] = next(iter(
                DataLoader(sub, batch_size=len(idxs), collate_fn=collate)))
        self.multi = len(self.eval_batches) > 1

    def group_names(self) -> list[str]:
        return list(self.eval_batches.keys())

    @torch.no_grad()
    def eval_recon(self, tok, perceptual=None, car=None, step: int = 0,
                   save: bool = True):
        """Per-group reconstruction (+ optional AR token accuracy) metrics and grids.

        Returns (metrics: dict, images: dict). Pass `car` only when the AR objective is
        active — it adds token_acc (and bit_acc for the MBM head) per group.
        """
        metrics, images = {}, {}
        agg = defaultdict(list)
        for tag, b in self.eval_batches.items():
            x = b["image"].to(self.device)
            out = tok(x)
            r = out["recon"]
            gtag = tag if self.multi else "all"
            grid = make_grid(torch.cat([denorm(x), denorm(r)], 0), nrow=x.shape[0])
            if save:
                save_image(grid, self.sample_dir / f"recon_{gtag}_{step:06d}.png")
            images[f"reconstructions/{gtag}"] = grid
            rl2 = torch.nn.functional.mse_loss(r, x).item()
            metrics[f"eval/{gtag}/recon_l2"] = rl2
            agg["recon_l2"].append(rl2)
            if perceptual is not None:
                lp = perceptual(r, x).mean().item()
                metrics[f"eval/{gtag}/lpips"] = lp
                agg["lpips"].append(lp)
            if car is not None and "text_ids" in b:
                _, logs, _ = car.ar_loss(b["text_ids"].to(self.device),
                                         b["text_mask"].to(self.device), out["indices"])
                metrics[f"eval/{gtag}/token_acc"] = logs.get("car/token_acc", 0.0)
                agg["token_acc"].append(logs.get("car/token_acc", 0.0))
                if "car/bit_acc" in logs:
                    metrics[f"eval/{gtag}/bit_acc"] = logs["car/bit_acc"]
                    agg["bit_acc"].append(logs["car/bit_acc"])
        # Combined roll-up across groups (only meaningful when >1 group).
        if self.multi:
            for k, vals in agg.items():
                if vals:
                    metrics[f"eval/combined/{k}"] = sum(vals) / len(vals)
        return metrics, images

    @torch.no_grad()
    def eval_generation(self, car, tok, cond: Conditioning, prompts_by_group: dict,
                        *, amp: str = "none", step: int = 0, temperature: float = 1.0,
                        top_k: int = 0, cfg_scale: float = 1.0):
        """One generation grid per prompt group. Returns {f'generations/{tag}': grid}."""
        images = {}
        for tag, prompts in prompts_by_group.items():
            gtag = tag if (self.multi or tag != "all") else "all"
            grid = sample_generations(
                car, tok, cond, prompts, self.device, amp, self.sample_dir, step,
                cfg_scale=cfg_scale, temperature=temperature, top_k=top_k, tag=gtag,
            )
            if grid is not None:
                images[f"generations/{gtag}"] = grid
        return images


@torch.no_grad()
def sample_generations(car, tok, cond: Conditioning, prompts, device, amp, sample_dir,
                       step, *, cfg_scale=1.0, temperature=1.0, top_k=0, tag="all"):
    """Sample images for `prompts`, save a grid, return it (CHW in [0,1])."""
    text_ids, text_mask = cond.encode_batch(prompts)
    text_ids = text_ids.to(device); text_mask = text_mask.to(device)
    uncond_ids = uncond_mask = None
    if cfg_scale != 1.0:
        uncond_ids, uncond_mask = cond.unconditional(len(prompts), L=text_ids.shape[1],
                                                     device=device)
    with autocast_ctx(device, amp):
        idxs = car.generate(text_ids, text_mask, temperature=temperature, top_k=top_k,
                            cfg_scale=cfg_scale, uncond_text_ids=uncond_ids,
                            uncond_text_mask=uncond_mask)
        imgs = tok.decode(tok.quantizer.lookup(idxs))
    grid = make_grid(denorm(imgs).float().cpu(), nrow=len(prompts))
    save_image(grid, Path(sample_dir) / f"gen_{tag}_{step:06d}.png")
    print(f"  sampled {len(prompts)} '{tag}' prompts -> gen_{tag}_{step:06d}.png")
    return grid
