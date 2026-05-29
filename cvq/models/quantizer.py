"""
Channel-wise Vector Quantization (CVQ) — the core contribution of arXiv:2605.26089.

Standard VQ (VQ-VAE / VQGAN) quantizes each *spatial location* of a feature map:
Z in (B, C, h, w) is treated as h*w tokens, each a C-dim vector. Channel-wise VQ
transposes the axis: each *channel* z^(k) in (h, w) is flattened to an (h*w)-dim
vector and matched against a shared codebook whose entries live in R^(h*w). So an
image becomes C tokens (one per channel), each a "level of visual detail". At 256x256
with a patch-16 / f=16 encoder the grid is 16x16, so token dim = 256 and channels
c = 256 — the paper's "256 version".

    z_q^(k) = argmin_{e_n in C}  || z^(k) - e_n ||_2^2
    forward (straight-through): z_q^(k) = z^(k) + sg[ e_n - z^(k) ]

PLAIN VQ (faithful to the paper). The paper states CVQ "achieves 100% codebook
utilization with a 16K+ codebook size *without any bells and whistles*." We therefore
use a plain, gradient-updated codebook with the standard VQ-VAE objective:

    L_vq = || sg[z] - e ||^2  +  beta * || z - sg[e] ||^2
           (codebook loss: move codes->z)  (commitment: move z->codes)

— NO EMA updates and NO dead-code restart. (An earlier version used VILA-U's EMA +
restart; those are exactly the "bells and whistles" the paper says are unnecessary,
so they were removed to test/use the paper's actual mechanism.) A non-gradient
usage-EMA buffer is kept ONLY for the n_dead diagnostic; it never changes the codebook.

Nested channel dropout (Matryoshka-style) is supported via `truncate()`: keeping only
the first c_keep channels (masking the rest to zero, in latent space before decoding)
forces channel 0 to carry coarse global structure and later channels to add fine detail.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def _no_autocast(t: torch.Tensor):
    """Disable autocast for CUDA tensors so codebook matmuls stay fp32."""
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class ChannelwiseVQ(nn.Module):
    """Channel-wise vector quantizer with a plain, gradient-updated codebook.

    Args:
        codebook_size: number of codebook entries (N). Paper uses 16384.
        token_dim: dimensionality of each channel token (D = h*w of the latent grid).
        commitment_beta: weight of the commitment term (paper: "standard VQGAN").
        usage_decay: EMA decay for the diagnostic-only usage tracker.
        dead_threshold: a code counts as "dead" if its usage EMA falls below this.
    """

    def __init__(
        self,
        codebook_size: int = 16384,
        token_dim: int = 256,
        commitment_beta: float = 0.25,
        usage_decay: float = 0.99,
        dead_threshold: float = 1e-3,
        **_ignore,  # tolerate legacy kwargs (e.g. decay) from old configs/checkpoints
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold

        # Learnable codebook (gradient-updated via the codebook loss) — plain VQ.
        self.embed = nn.Embedding(codebook_size, token_dim)
        self.embed.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        # Diagnostic-only: EMA of how often each code is selected. NOT used to update
        # the codebook (which is gradient-trained); only powers the n_dead metric.
        self.register_buffer("usage_ema", torch.zeros(codebook_size))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance to every code (constant ||z||^2 dropped; argmin-safe)."""
        with _no_autocast(flat):
            cb = self.embed.weight  # (N, D)
            e_sq = cb.pow(2).sum(dim=1)  # (N,)
            return e_sq.unsqueeze(0) - 2.0 * (flat @ cb.t())

    @torch.no_grad()
    def _update_usage(self, idxs: torch.Tensor) -> None:
        counts = torch.bincount(idxs, minlength=self.codebook_size).float()
        self.usage_ema.mul_(self.usage_decay).add_(counts, alpha=1 - self.usage_decay)

    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor):
        """Quantize a feature map channel-wise.

        Args:
            z: (B, C, h, w) — encoder feature map. C channel-tokens of dim h*w.
        Returns:
            z_q:   (B, C, h, w) straight-through quantized feature map
            idxs:  (B, C) codebook index per channel
            loss:  scalar VQ loss (codebook loss + beta * commitment)
            stats: dict with perplexity / usage / quant_error
        """
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, (
            f"token_dim mismatch: quantizer expects {self.token_dim}, got h*w={D}."
        )

        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()               # (M, D), M = B*C tokens
        idxs = self._distances(flat).argmin(dim=1)        # (M,)
        quant = self.embed(idxs)                          # (M, D), differentiable wrt codebook

        # Standard VQ-VAE objective: codebook loss moves codes toward encoder outputs;
        # commitment moves the encoder toward its assigned code.
        codebook_loss = F.mse_loss(quant, flat.detach())
        commit_loss = F.mse_loss(flat, quant.detach())
        loss = codebook_loss + self.commitment_beta * commit_loss

        if self.training:
            self._update_usage(idxs)

        # Straight-through estimator: forward uses quant, backward flows to z.
        quant = flat + (quant - flat).detach()

        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)

        stats = self._codebook_stats(idxs)
        stats["quant_error"] = commit_loss.item()
        return z_q, idxs, loss, stats

    @torch.no_grad()
    def _codebook_stats(self, idxs: torch.Tensor) -> dict:
        flat = idxs.reshape(-1)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        usage = (counts > 0).float().mean()
        return {"perplexity": perplexity.item(), "usage": usage.item()}

    @torch.no_grad()
    def codebook_health(self) -> dict:
        """Diagnostic health from the usage-EMA buffer (logging only)."""
        u = self.usage_ema
        return {
            "n_dead": int((u < self.dead_threshold).sum()),
            "cluster_size_min": float(u.min()),
            "cluster_size_mean": float(u.mean()),
            "cluster_size_max": float(u.max()),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep: int) -> torch.Tensor:
        """Keep the first c_keep channels, mask the rest to zero (nested dropout)."""
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """Map (B, C) indices to a (B, C, sqrt(D), sqrt(D)) feature map (for CAR)."""
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        quant = self.embed(idxs)  # (B, C, D)
        return quant.reshape(B, C, side, side)
