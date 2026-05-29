"""
Channel-wise Vector Quantization (CVQ) — 100% LITERAL to arXiv:2605.26089.

Standard VQ quantizes each *spatial location* of a feature map. Channel-wise VQ
transposes the axis: each *channel* z^(k) in (h, w) is flattened to an (h*w)-dim vector
and matched against a shared codebook (entry dim = h*w). At 256x256 with f=16 the grid
is 16x16, so token dim = 256 and channels c = 256 (the paper's "256 version").

    z_q^(k) = argmin_{e_n in C}  || z^(k) - e_n ||_2^2
    forward (straight-through): z_q^(k) = z^(k) + sg[ e_n - z^(k) ]
    L_vq = || sg[z] - e ||^2  +  beta * || z - sg[e] ||^2     (codebook + commitment)

This is PLAIN VQ exactly as the paper describes: a single gradient-updated codebook,
**no EMA, no dead-code restart, no L2-normalization, no factorization**. The paper claims
high codebook utilization from the channel-wise formulation *alone* ("without any bells
and whistles"). Whether that holds at small (non-ImageNet) scale is an open question this
implementation lets us measure directly — including the failure mode (codebook collapse),
which is itself a valid data point.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def _no_autocast(t: torch.Tensor):
    """Keep codebook distance math in fp32 even under bf16/fp16 autocast."""
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class ChannelwiseVQ(nn.Module):
    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25, **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta

        # Single learnable codebook, standard VQGAN init.
        self.embed = nn.Embedding(codebook_size, token_dim)
        self.embed.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        """z: (B, C, h, w) -> (z_q, idxs (B,C), vq_loss, stats)."""
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, f"token_dim mismatch: expected {self.token_dim}, got {D}."

        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()

        # Nearest code by squared L2 (fp32, autocast-safe). argmin needs no gradient.
        with _no_autocast(flat), torch.no_grad():
            cb = self.embed.weight
            dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * (flat @ cb.t()) + cb.pow(2).sum(1)
            idxs = dist.argmin(dim=1)

        quant = self.embed(idxs)                          # differentiable lookup -> codebook grad

        codebook_loss = F.mse_loss(quant, flat.detach())  # move codes toward features
        commit = F.mse_loss(flat, quant.detach())         # move features toward codes
        loss = codebook_loss + self.commitment_beta * commit

        quant = flat + (quant - flat).detach()            # straight-through estimator
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)

        stats = self._stats(idxs)
        stats["quant_error"] = commit.item()
        return z_q, idxs, loss, stats

    @torch.no_grad()
    def _stats(self, idxs: torch.Tensor) -> dict:
        counts = torch.bincount(idxs.reshape(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        return {"perplexity": perplexity.item(), "usage": (counts > 0).float().mean().item()}

    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep: int) -> torch.Tensor:
        """Nested channel dropout: keep first c_keep channels, mask the rest to zero."""
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """(B, C) indices -> (B, C, sqrt(D), sqrt(D)) feature map (for CAR / eval)."""
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        return self.embed(idxs).reshape(B, C, side, side)
