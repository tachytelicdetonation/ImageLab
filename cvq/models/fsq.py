"""
Channel-wise Finite Scalar Quantization (FSQ) — Fork B (channel-FSQ-AR / "Bit-CAR").

Lifts BAR's FSQ discretizer (arXiv:2602.09024; FSQ from Mentzer et al. 2309.15505) and
applies it on the CVQ *channel* axis: each channel-token (the h*w-dim spatial map of one
channel) is projected to `len(levels)` dims, tanh-bounded, and round-quantized (STE) to a
fixed integer grid, then projected back. The implicit codebook is `prod(levels)` — no
learned codebook, no commitment/entropy losses, ~100% utilization by construction. The
per-channel integer index has a meaningful mixed-radix/bit expansion, which is exactly what
the MBM head (`cvq/models/mbm_head.py`) predicts instead of a flat softmax over |C|.

Drop-in for `IBQChannelVQ`: identical `(z_q, idxs, loss, stats)` forward plus
`truncate`/`lookup`, so `CVQTokenizer` and `train_e2e` treat it the same. There is no `embed`
codebook (FSQ is parameter-free apart from the in/out projections); the aux loss is 0.

Fidelity note vs the downloaded BAR fsq.py: we use the *canonical normalized* FSQ —
`quantize` divides by the half-width so `project_out` receives [-1,1]-scaled codes at BOTH
train (forward) and decode (lookup/indices_to_codes). Pick `prod(levels)` to be a power of
two so `k=log2|C|` bits cover the index space exactly (no out-of-range codes at sampling).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


def _no_autocast(t: torch.Tensor):
    """Keep the bound/round/index math in fp32 even under bf16/fp16 autocast."""
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class ChannelFSQ(nn.Module):
    """Channel-wise FSQ. Per channel-token (dim D=h*w), project to len(levels) scalars,
    scalar-quantize each to its level count, project back. |C| = prod(levels)."""

    def __init__(self, levels, token_dim: int = 256, eps: float = 1e-3, **_ignore):
        super().__init__()
        levels = [int(x) for x in levels]
        self.levels = levels
        self.token_dim = token_dim
        self.n_dims = len(levels)
        self.eps = eps
        cb = 1
        for l in levels:
            cb *= l
        self.codebook_size = cb
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.int64), persistent=False)
        self.register_buffer(
            "_basis",
            torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int64), dim=0),
            persistent=False,
        )
        # Per-channel projections (shared across the C channel-tokens), mirroring the shared
        # codebook of IBQ — one FSQ grid for all channels.
        self.project_in = nn.Linear(token_dim, self.n_dims)
        self.project_out = nn.Linear(self.n_dims, token_dim)

    # ----------------------------- FSQ core ----------------------------- #
    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """tanh bound so the rounded grid is symmetric; half-level offset for even L."""
        levels = self._levels.to(z.dtype)
        half_l = (levels - 1) * (1 + self.eps) / 2
        offset = (self._levels % 2 == 0).to(z.dtype) * 0.5
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """z (..., n_dims) -> normalized quantized codes in ~[-1, 1] with straight-through."""
        q = self._bound(z)
        q = q + (torch.round(q) - q).detach()        # STE round to integer-centered levels
        half_width = (self._levels // 2).to(z.dtype)
        return q / half_width                          # renormalize to [-1, 1]

    def _codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """normalized codes (..., n_dims) -> (...) int64 index in [0, |C|)."""
        half_width = (self._levels // 2).to(zhat.dtype)
        z = zhat * half_width + half_width             # back to integer levels [0, L)
        return (z * self._basis.to(zhat.dtype)).sum(dim=-1).round().to(torch.int64)

    def _indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """(...) index -> (..., n_dims) normalized codes (for decode / lookup)."""
        idx = indices.unsqueeze(-1)
        codes_non_centered = (idx // self._basis) % self._levels
        half_width = self._levels // 2
        return (codes_non_centered - half_width).float() / half_width.float()

    # --------------------- IBQ-compatible interface --------------------- #
    def forward(self, z: torch.Tensor):
        """z (B, C, h, w) -> (z_q (B,C,h,w), idxs (B,C), loss=0, stats)."""
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, f"token_dim mismatch: expected {self.token_dim}, got {D}."
        in_dtype = z.dtype
        with _no_autocast(z):
            tok = z.reshape(B, C, D).float()
            codes = self._quantize(self.project_in(tok))   # (B,C,n_dims) normalized
            idxs = self._codes_to_indices(codes)           # (B,C)
            z_q = self.project_out(codes)                  # (B,C,D)
        z_q = z_q.reshape(B, C, h, w).to(in_dtype)
        loss = z.new_zeros(())
        stats = self._stats(idxs)
        return z_q, idxs.reshape(B, C), loss, stats

    @torch.no_grad()
    def _stats(self, idxs: torch.Tensor) -> dict:
        counts = torch.bincount(idxs.reshape(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        return {"perplexity": perplexity.item(),
                "usage": (counts > 0).float().mean().item(),
                "entropy_loss": 0.0}

    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep) -> torch.Tensor:
        """Nested channel dropout: keep first c_keep channels, zero the rest."""
        if c_keep is None or c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """(B, C) indices -> (B, C, sqrt(D), sqrt(D)) feature map (for CAR / eval)."""
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        codes = self._indices_to_codes(idxs)               # (B,C,n_dims) normalized
        z = self.project_out(codes.float())                # (B,C,token_dim)
        return z.reshape(B, C, side, side)
