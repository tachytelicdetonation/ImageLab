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
                 commitment_beta: float = 0.25,
                 entropy_weight: float = 0.0, entropy_temperature: float = 1.0,
                 **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        # Entropy regularization (MAGVIT-v2 / VQGAN lineage; arXiv:2310.05737). A
        # MODIFICATION on top of the literal paper: weight 0.0 -> exactly plain VQ.
        # > 0 -> add L_ent = E[H(q)] - H(E[q]) to push usage toward uniform (anti-collapse).
        self.entropy_weight = entropy_weight
        self.entropy_temperature = entropy_temperature

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

        # Squared-L2 distance to every code (fp32, autocast-safe). Kept WITH grad so the
        # optional entropy term below can backprop into both encoder and codebook; argmin
        # itself is non-differentiable, so we detach for it.
        with _no_autocast(flat):
            cb = self.embed.weight
            dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * (flat @ cb.t()) + cb.pow(2).sum(1)
            idxs = dist.detach().argmin(dim=1)

        quant = self.embed(idxs)                          # differentiable lookup -> codebook grad

        codebook_loss = F.mse_loss(quant, flat.detach())  # move codes toward features
        commit = F.mse_loss(flat, quant.detach())         # move features toward codes
        loss = codebook_loss + self.commitment_beta * commit

        stats = self._stats(idxs.reshape(B, C))
        stats["quant_error"] = commit.item()

        # ---- optional entropy regularization (anti-collapse modification) ----
        if self.entropy_weight > 0.0:
            ent_loss, ent_logs = self._entropy_loss(dist)
            loss = loss + self.entropy_weight * ent_loss
            stats.update(ent_logs)

        quant = flat + (quant - flat).detach()            # straight-through estimator
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)
        return z_q, idxs, loss, stats

    def _entropy_loss(self, dist: torch.Tensor):
        """Entropy regularization on the soft code assignment (MAGVIT-v2 eq.).

            q(z|x) = softmax(-dist / tau)                      # (B*C, N) over the codebook
            L_ent  = E_x[H(q(z|x))]  -  H(E_x[q(z|x)])

        Term 1 (per-token entropy, MINIMIZED) sharpens each assignment toward a single
        code so quantization stays decisive. Term 2 (marginal entropy, MAXIMIZED via the
        minus sign) flattens the *aggregate* usage histogram toward uniform -> every code
        gets pulled into use. This is the VQ twin of the MoE load-balancing loss.
        """
        with _no_autocast(dist):
            logits = -dist / self.entropy_temperature
            log_q = F.log_softmax(logits, dim=1)
            q = log_q.exp()
            per_sample = -(q * log_q).sum(dim=1).mean()       # E[H(q)]
            avg = q.mean(dim=0)                               # E[q] marginal usage
            marginal = -(avg * (avg + 1e-12).log()).sum()     # H(E[q])
            ent_loss = per_sample - marginal
        logs = {
            "entropy_loss": ent_loss.item(),
            "entropy_per_sample": per_sample.item(),
            "entropy_marginal": marginal.item(),
        }
        return ent_loss, logs

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
