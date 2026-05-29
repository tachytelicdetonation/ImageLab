"""
Validation metrics for the CVQ tokenizer.

Paper-style reconstruction quality (rFID, PSNR, SSIM, LPIPS) plus the two CVQ-specific
diagnostics: full-dataset codebook utilization (the ~100% claim) and per-c_keep
reconstruction error (a *quantitative* check that the channel ordering is coarse-to-fine
— error should decrease monotonically as more channels are kept).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

# torchmetrics is optional at import time so the rest of the code runs without it.
try:
    from torchmetrics.functional import (
        peak_signal_noise_ratio as _psnr,
        structural_similarity_index_measure as _ssim,
    )
    from torchmetrics.image.fid import FrechetInceptionDistance
    _HAS_TM = True
except Exception:  # pragma: no cover
    _HAS_TM = False


def grad_norm(params) -> float:
    """L2 norm of gradients over an iterable of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return total ** 0.5


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1] for metric/image use."""
    return (x.clamp(-1, 1) * 0.5 + 0.5)


@torch.no_grad()
def validate(tok, dataset, device, batch_size=32, compute_fid=True, lpips_fn=None,
             c_keep_levels=(8, 32, 64, 128, 256), max_images=0):
    """Run a full validation pass. Returns (metrics_dict, images_dict).

    metrics_dict: scalars to log. images_dict: {'reconstructions','coarse_to_fine'}
    as [0,1] CHW grids (caller wraps for wandb). Pass lpips_fn (the trained LPIPS
    module) to get full-dataset perceptual distance.
    """
    from torchvision.utils import make_grid

    tok.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fid = None
    if compute_fid and _HAS_TM:
        try:
            fid = FrechetInceptionDistance(normalize=True).to(device)
        except Exception:
            fid = None

    n = 0
    psnr_sum = ssim_sum = l2_sum = lpips_sum = 0.0
    used = torch.zeros(tok.quantizer.codebook_size, dtype=torch.bool, device=device)
    first_batch = None

    for batch in dl:
        x = batch["image"].to(device)
        out = tok(x)
        recon = out["recon"]
        used[out["indices"].reshape(-1).unique()] = True

        xr, rr = _denorm(x), _denorm(recon)
        l2_sum += float(torch.mean((x - recon) ** 2)) * x.size(0)
        if lpips_fn is not None:
            lpips_sum += float(lpips_fn(recon, x).mean()) * x.size(0)  # expects [-1,1]
        if _HAS_TM:
            psnr_sum += float(_psnr(rr, xr, data_range=1.0)) * x.size(0)
            ssim_sum += float(_ssim(rr, xr, data_range=1.0)) * x.size(0)
        if fid is not None:
            fid.update(xr, real=True)
            fid.update(rr.clamp(0, 1), real=False)

        if first_batch is None:
            first_batch = (x[:8].clone(), recon[:8].clone())
        n += x.size(0)
        if max_images and n >= max_images:
            break

    metrics = {
        "val/recon_l2_full": l2_sum / max(n, 1),
        "val/codebook_utilization_full": float(used.float().mean()),
        "val/codebook_codes_used": int(used.sum()),
    }
    if lpips_fn is not None:
        metrics["val/lpips_full"] = lpips_sum / max(n, 1)
    if _HAS_TM:
        metrics["val/psnr"] = psnr_sum / max(n, 1)
        metrics["val/ssim"] = ssim_sum / max(n, 1)
    if fid is not None:
        try:
            metrics["val/rFID"] = float(fid.compute())
        except Exception:
            pass

    # ---- per-c_keep reconstruction error: quantifies coarse-to-fine ordering ----
    xc = first_batch[0]
    C = tok.latent_channels
    for k in c_keep_levels:
        if k <= C:
            rk = tok(xc, c_keep=k)["recon"]
            metrics[f"val/recon_l2_c{k}"] = float(torch.mean((xc - rk) ** 2))

    # ---- image grids ----
    images = {}
    xo, ro = first_batch
    images["reconstructions"] = make_grid(
        torch.cat([_denorm(xo), _denorm(ro)], 0), nrow=xo.size(0))
    keeps = [k for k in c_keep_levels if k <= C]
    prog = [_denorm(tok(xc[:1], c_keep=k)["recon"]) for k in keeps]
    images["coarse_to_fine"] = make_grid(
        torch.cat([_denorm(xc[:1])] + prog, 0), nrow=len(prog) + 1)

    tok.train()
    return metrics, images
