"""
Channel-wise Lookup-Free Quantization (LFQ) — the repo's WORKED EXAMPLE of adding a
quantizer (README "Trying an architecture change"). It is also a real method: LFQ is
MAGVIT-v2's discretizer (arXiv:2310.05737), applied here on the CVQ channel axis the
same way ChannelFSQ lifts FSQ.

Each channel-token (the h*w-dim map of one channel) is projected to k = log2(|C|) dims
and binarized by SIGN with a straight-through estimator; the implicit codebook is
{-1,+1}^k. Two auxiliary terms shape the code distribution (both from the paper, in the
factorized per-bit form lucidrains' implementation popularized, since full-codebook
entropy is intractable for k beyond ~16):

  * per-sample entropy (minimize): each bit should be CONFIDENT, not sit near 0;
  * marginal entropy (maximize): across the batch, each bit should use both signs —
    the anti-collapse pressure a learned codebook gets from its embedding gradient.

Like FSQ it is codebook-free, so Fork B's MBM bit head applies directly (the index IS
the bit-string), and `lambda_apr` must stay 0 (no embedding matrix to soft-decode).
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from cvq.registry import register


def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


@register("quantizer", "lfq", paper="arXiv:2310.05737")
class ChannelLFQ(nn.Module):
    """Channel-wise LFQ: project token_dim -> k bits, sign-quantize (STE), project back.

    Args:
        token_dim: dim of one channel-token (= grid**2, supplied by the factory).
        codebook_size: must be a power of two; k = log2(codebook_size) bits.
        entropy_weight / commit_weight: aux loss weights (paper defaults are ~0.1/0.25
            at scale; tune per run via model.quantizer_kwargs).
        temperature: softness of the per-bit probabilities inside the entropy terms.
    """

    def __init__(self, token_dim: int, codebook_size: int, entropy_weight: float = 0.1,
                 commit_weight: float = 0.25, temperature: float = 1.0, **_ignore):
        super().__init__()
        k = round(math.log2(codebook_size))
        if 2 ** k != codebook_size:
            raise ValueError(f"lfq needs a power-of-two codebook_size, got {codebook_size}")
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.n_bits = k
        self.entropy_weight = entropy_weight
        self.commit_weight = commit_weight
        self.temperature = temperature
        self.project_in = nn.Linear(token_dim, k)
        self.project_out = nn.Linear(k, token_dim)
        self.register_buffer("_pow2", 2 ** torch.arange(k, dtype=torch.int64),
                             persistent=False)

    def forward(self, z: torch.Tensor):
        """z (B,C,h,w) -> (z_q (B,C,h,w), idxs (B,C), aux_loss, stats)."""
        B, C, h, w = z.shape
        assert h * w == self.token_dim, \
            f"token_dim mismatch: expected {self.token_dim}, got {h * w}"
        in_dtype = z.dtype
        with _no_autocast(z):
            tok = self.project_in(z.reshape(B, C, h * w).float())   # (B,C,k) bit logits
            bits = torch.where(tok > 0, 1.0, -1.0)
            q = tok + (bits - tok).detach()                          # sign with STE
            idxs = ((bits > 0).long() * self._pow2).sum(-1)          # (B,C) bit-packed

            # --- factorized entropy objective (confident bits, balanced usage) ---
            p = torch.sigmoid(2 * tok / self.temperature)            # P(bit=+1)
            per_sample = _binary_entropy(p).mean()
            marginal = _binary_entropy(p.mean(dim=(0, 1))).mean()
            commit = F.mse_loss(tok, bits.detach())
            aux = self.entropy_weight * (per_sample - marginal) + self.commit_weight * commit

            z_q = self.project_out(q).reshape(B, C, h, w)
        with torch.no_grad():
            counts = torch.bincount(idxs.reshape(-1), minlength=self.codebook_size).float()
            probs = counts / counts.sum().clamp_min(1)
            nz = probs[probs > 0]
        stats = {
            "usage": (counts > 0).float().mean().item(),   # same convention as ibq/fsq
            "perplexity": torch.exp(-(nz * nz.log()).sum()).item(),
            "quant_error": commit.item(),
            "entropy_loss": (per_sample - marginal).item(),
            "entropy_per_sample": per_sample.item(),
            "entropy_marginal": marginal.item(),
        }
        return z_q.to(in_dtype), idxs, aux.to(in_dtype), stats

    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep) -> torch.Tensor:
        if c_keep is None or c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """(B,C) bit-packed indices -> the same (B,C,side,side) features forward produced."""
        B, C = idxs.shape
        bits = ((idxs.unsqueeze(-1) & self._pow2) > 0).float() * 2 - 1   # (B,C,k) in ±1
        side = int(round(self.token_dim ** 0.5))
        return self.project_out(bits).reshape(B, C, side, side)


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())
