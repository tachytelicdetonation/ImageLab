"""
Channel-wise Vector Quantization (CVQ) — the core contribution of arXiv:2605.26089.

Standard VQ (VQ-VAE / VQGAN, e.g. VILA-U) quantizes each *spatial location* of a
feature map: Z in (B, C, h, w) is treated as h*w tokens, each a C-dim vector, and
matched to a codebook of C-dim entries.

Channel-wise VQ transposes the axis being quantized: each *channel* z^(k) in (h, w)
is flattened to an (h*w)-dim vector and matched against a shared codebook whose
entries live in R^(h*w). So an image becomes C tokens (one per channel), each one a
"level of visual detail". At 256x256 with a patch-16 SigLIP encoder the grid is
16x16, giving a token dim of 256 — exactly the paper's "256 token dim" setting.

The codebook itself is updated by exponential moving average (EMA) with dead-code
restart, faithfully ported from VILA-U's VQEmbedding (which itself derives from the
kakaobrain RQ-VAE). CVQ credits its ~100% codebook utilization to the channel-wise
formulation + nested channel dropout rather than to these tricks, but we keep EMA +
restart because that is what the lineage it builds on uses.

Nested channel dropout (Matryoshka-style) is supported via `truncate()`: keeping only
the first c_keep channels (zeroing the rest) forces channel 0 to carry the coarsest
global structure and later channels to add fine detail. This ordering is what makes
"next-channel prediction" (the CAR model, phase 2) meaningful.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _no_autocast(t: torch.Tensor):
    """Disable autocast for CUDA tensors so codebook matmuls stay fp32."""
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class ChannelwiseVQ(nn.Module):
    """Channel-wise vector quantizer with EMA codebook + dead-code restart.

    Args:
        codebook_size: number of codebook entries (N). Paper uses 16384.
        token_dim: dimensionality of each channel token (D = h*w of the latent grid).
        decay: EMA decay for codebook updates (paper/VILA-U: 0.99).
        commitment_beta: weight of the encoder commitment loss term.
        eps: Laplace-smoothing epsilon for cluster sizes.
        restart_unused_codes: reseed dead codes from the current batch.
        restart_threshold: a code is "dead" if its EMA cluster size falls below this.
    """

    def __init__(
        self,
        codebook_size: int = 16384,
        token_dim: int = 256,
        decay: float = 0.99,
        commitment_beta: float = 0.25,
        eps: float = 1e-5,
        restart_unused_codes: bool = True,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.decay = decay
        self.commitment_beta = commitment_beta
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.restart_threshold = restart_threshold

        # Codebook + EMA accumulators are buffers (no autograd): EMA, not SGD, moves them.
        embed = torch.randn(codebook_size, token_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size_ema", torch.zeros(codebook_size))
        self.register_buffer("embed_ema", embed.clone())

    # ------------------------------------------------------------------ #
    # Nearest-neighbour lookup
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance from each token to every codebook entry.

        ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2 ; the constant ||z||^2 term does not
        affect the argmin so we drop it and compute -2 z.e + ||e||^2.
        flat: (M, D) -> returns (M, N).
        """
        with _no_autocast(flat):
            codebook = self.embed  # (N, D)
            e_sq = codebook.pow(2).sum(dim=1)  # (N,)
            # -2 * flat @ codebook.T + ||e||^2
            return e_sq.unsqueeze(0) - 2.0 * (flat @ codebook.t())

    # ------------------------------------------------------------------ #
    # EMA codebook update + dead-code restart
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, idxs: torch.Tensor) -> None:
        ctx = _no_autocast(flat)
        with ctx:
            self._ema_update_impl(flat, idxs)

    def _ema_update_impl(self, flat: torch.Tensor, idxs: torch.Tensor) -> None:
        n_embed, dim = self.codebook_size, self.token_dim
        onehot = F.one_hot(idxs, n_embed).type_as(flat)  # (M, N)
        cluster_size = onehot.sum(dim=0)                 # (N,)
        embed_sum = onehot.t() @ flat                    # (N, D)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        # EMA accumulate
        self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_ema.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Laplace-smoothed normalisation so empty clusters don't divide by zero
        n = self.cluster_size_ema.sum()
        smoothed = (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps) * n
        self.embed.copy_(self.embed_ema / smoothed.unsqueeze(1))

        # Dead-code restart: reseed under-used entries with random live tokens.
        if self.restart_unused_codes:
            dead = self.cluster_size_ema < self.restart_threshold  # (N,)
            n_dead = int(dead.sum())
            if n_dead > 0:
                m = flat.shape[0]
                if m < n_dead:  # tile with small noise if batch has too few tokens
                    reps = (n_dead + m - 1) // m
                    pool = flat.repeat(reps, 1)
                    pool = pool + torch.randn_like(pool) * 0.01 / (dim ** 0.5)
                else:
                    pool = flat
                perm = torch.randperm(pool.shape[0], device=pool.device)[:n_dead]
                if dist.is_available() and dist.is_initialized():
                    dist.broadcast(perm, 0)
                self.embed[dead] = pool[perm]
                self.embed_ema[dead] = pool[perm]
                self.cluster_size_ema[dead] = self.restart_threshold

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor):
        """Quantize a feature map channel-wise.

        Args:
            z: (B, C, h, w) — encoder feature map. C channel-tokens of dim h*w.
        Returns:
            z_q:   (B, C, h, w) straight-through quantized feature map
            idxs:  (B, C) codebook index per channel
            loss:  scalar commitment loss
            stats: dict with 'perplexity' and 'usage' (fraction of codebook used)
        """
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, (
            f"token_dim mismatch: quantizer expects {self.token_dim}, got h*w={D}. "
            f"Set token_dim to match the encoder's latent grid."
        )

        # Codebook math runs in fp32 for numerical stability and to be safe under
        # bf16/fp16 autocast (nearest-neighbour + EMA are sensitive to precision).
        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()               # (M, D), M = B*C tokens
        idxs = self._distances(flat).argmin(dim=1)       # (M,) nearest entry
        quant = F.embedding(idxs, self.embed)            # (M, D) fp32

        if self.training:
            self._ema_update(flat, idxs)

        # Commitment loss: pull the encoder output toward its assigned code.
        # (The code itself is moved by EMA, not gradient, so only the encoder side
        # contributes a gradient term here.)
        commit = F.mse_loss(quant.detach(), flat)
        loss = self.commitment_beta * commit

        # Straight-through estimator: forward uses quant, backward flows to z.
        quant = flat + (quant - flat).detach()

        z_q = quant.reshape(B, C, h, w).to(in_dtype)     # back to autocast dtype
        idxs = idxs.reshape(B, C)

        stats = self._codebook_stats(idxs)
        stats["quant_error"] = commit.item()             # mean sq. dist to nearest code
        return z_q, idxs, loss, stats

    @torch.no_grad()
    def _codebook_stats(self, idxs: torch.Tensor) -> dict:
        """Perplexity and fraction of the codebook touched this batch."""
        flat = idxs.reshape(-1)
        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        usage = (counts > 0).float().mean()
        return {"perplexity": perplexity.item(), "usage": usage.item()}

    @torch.no_grad()
    def codebook_health(self) -> dict:
        """EMA-based health of the codebook (independent of the current batch).

        n_dead: codes below the restart threshold (candidates for being reseeded).
        cluster_size {min/mean/max}: how evenly tokens are spread across codes.
        """
        cs = self.cluster_size_ema
        return {
            "n_dead": int((cs < self.restart_threshold).sum()),
            "cluster_size_min": float(cs.min()),
            "cluster_size_mean": float(cs.mean()),
            "cluster_size_max": float(cs.max()),
        }

    # ------------------------------------------------------------------ #
    # Nested channel dropout
    # ------------------------------------------------------------------ #
    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep: int) -> torch.Tensor:
        """Keep the first `c_keep` channels, zero the rest (nested dropout).

        Zeroing — rather than slicing — preserves the (B, C, h, w) shape so the
        decoder input dimensionality is constant; the decoder learns that 'absent'
        detail channels are zero. This is what enforces the coarse-to-fine ordering.
        """
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """Map (B, C) indices back to a (B, C, sqrt(D), sqrt(D)) feature map.

        Used by the CAR generator (phase 2) to decode sampled channel tokens.
        """
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        quant = F.embedding(idxs, self.embed)  # (B, C, D)
        return quant.reshape(B, C, side, side)
