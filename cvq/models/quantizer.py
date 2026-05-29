"""
Channel-wise Vector Quantization (CVQ) — core of arXiv:2605.26089.

Standard VQ quantizes each *spatial location* of a feature map. Channel-wise VQ
transposes the axis: each *channel* z^(k) in (h, w) is flattened to an (h*w)-dim
vector and matched against a shared codebook (entry dim = h*w). At 256x256 with f=16
the grid is 16x16, so token dim = 256 and channels c = 256 (the paper's "256 version").

    z_q^(k) = argmin_{e_n in C}  || z^(k) - e_n ||_2^2
    forward (straight-through): z_q^(k) = z^(k) + sg[ e_n - z^(k) ]

CODEBOOK — two modes (config `codebook_ema`):
  * use_ema=False (PLAIN): gradient-updated codebook with the VQ-VAE objective
    L = ||sg[z]-e||^2 + beta*||z-sg[e]||^2. This is the paper's literal "no bells and
    whistles" claim — but empirically it COLLAPSES at our scale (1.3k images): a few
    codes near the data mean win every token and the rest die with no gradient.
  * use_ema=True (EMA + dead-code restart): the standard VQ-VAE-EMA stabilization that
    VILA-U (the repo's shipped code) actually uses. Codebook is an EMA of assigned
    features; dead codes are reseeded from the live batch. This trains stably and is
    what we use for real runs. Loss is then just the commitment term.

Nested channel dropout via `truncate()`: keep the first c_keep channels (mask the rest
to zero in latent space before decoding) -> coarse-to-fine channel ordering.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class ChannelwiseVQ(nn.Module):
    def __init__(
        self,
        codebook_size: int = 16384,
        token_dim: int = 256,
        commitment_beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_unused_codes: bool = True,
        restart_threshold: float = 1.0,
        usage_decay: float = 0.99,
        dead_threshold: float = 1e-3,
        **_ignore,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps
        self.restart_unused_codes = restart_unused_codes
        self.restart_threshold = restart_threshold
        self.usage_decay = usage_decay
        self.dead_threshold = dead_threshold

        if use_ema:
            # Codebook is a buffer moved by EMA (not by the optimizer).
            embed = torch.randn(codebook_size, token_dim)
            self.register_buffer("embed", embed)
            self.register_buffer("cluster_size_ema", torch.zeros(codebook_size))
            self.register_buffer("embed_ema", embed.clone())
        else:
            # Plain VQ: codebook is a learnable parameter (gradient-updated).
            self.embed = nn.Parameter(torch.empty(codebook_size, token_dim).uniform_(
                -1.0 / codebook_size, 1.0 / codebook_size))
            self.register_buffer("usage_ema", torch.zeros(codebook_size))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        with _no_autocast(flat):
            cb = self.embed
            e_sq = cb.pow(2).sum(dim=1)
            return e_sq.unsqueeze(0) - 2.0 * (flat @ cb.t())

    # ---- EMA update + dead-code restart (use_ema mode) ----
    @torch.no_grad()
    def _ema_update(self, flat, idxs):
        with _no_autocast(flat):
            n_embed, dim = self.codebook_size, self.token_dim
            onehot = F.one_hot(idxs, n_embed).type_as(flat)
            cluster_size = onehot.sum(dim=0)
            embed_sum = onehot.t() @ flat
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)
            self.cluster_size_ema.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_ema.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size_ema.sum()
            smoothed = (self.cluster_size_ema + self.eps) / (n + n_embed * self.eps) * n
            self.embed.copy_(self.embed_ema / smoothed.unsqueeze(1))
            if self.restart_unused_codes:
                dead = self.cluster_size_ema < self.restart_threshold
                n_dead = int(dead.sum())
                if n_dead > 0:
                    m = flat.shape[0]
                    if m < n_dead:
                        reps = (n_dead + m - 1) // m
                        pool = flat.repeat(reps, 1) + torch.randn(reps * m, dim, device=flat.device) * 0.01 / (dim ** 0.5)
                    else:
                        pool = flat
                    perm = torch.randperm(pool.shape[0], device=pool.device)[:n_dead]
                    if dist.is_available() and dist.is_initialized():
                        dist.broadcast(perm, 0)
                    self.embed[dead] = pool[perm]
                    self.embed_ema[dead] = pool[perm]
                    self.cluster_size_ema[dead] = self.restart_threshold

    @torch.no_grad()
    def _update_usage(self, idxs):
        counts = torch.bincount(idxs, minlength=self.codebook_size).float()
        self.usage_ema.mul_(self.usage_decay).add_(counts, alpha=1 - self.usage_decay)

    # ------------------------------------------------------------------ #
    def forward(self, z: torch.Tensor):
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, f"token_dim mismatch: expected {self.token_dim}, got {D}."

        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()
        idxs = self._distances(flat).argmin(dim=1)
        quant = F.embedding(idxs, self.embed)

        if self.use_ema:
            if self.training:
                self._ema_update(flat, idxs)
            commit = F.mse_loss(quant.detach(), flat)
            loss = self.commitment_beta * commit
        else:
            codebook_loss = F.mse_loss(quant, flat.detach())
            commit = F.mse_loss(flat, quant.detach())
            loss = codebook_loss + self.commitment_beta * commit
            if self.training:
                self._update_usage(idxs)

        quant = flat + (quant - flat).detach()           # straight-through
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)

        stats = self._codebook_stats(idxs)
        stats["quant_error"] = commit.item()
        return z_q, idxs, loss, stats

    @torch.no_grad()
    def _codebook_stats(self, idxs):
        counts = torch.bincount(idxs.reshape(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        return {"perplexity": perplexity.item(), "usage": (counts > 0).float().mean().item()}

    @torch.no_grad()
    def codebook_health(self):
        u = self.cluster_size_ema if self.use_ema else self.usage_ema
        thr = self.restart_threshold if self.use_ema else self.dead_threshold
        return {
            "n_dead": int((u < thr).sum()),
            "cluster_size_min": float(u.min()),
            "cluster_size_mean": float(u.mean()),
            "cluster_size_max": float(u.max()),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def truncate(z_q, c_keep):
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs):
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        return F.embedding(idxs, self.embed).reshape(B, C, side, side)
