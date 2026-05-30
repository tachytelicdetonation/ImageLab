"""
Masked Bit Modeling (MBM) head — faithful port of amazon-far/BAR
(`modeling/mbm_head.py`, arXiv:2602.09024), adapted to our channel-AR.

BAR replaces the AR transformer's flat softmax-over-|C| head with a tiny DiT-style MLP that
predicts a token's k = log2|C| *bits*. Given the AR hidden state for a position (here a CVQ
channel) it:
  * training  — masks a random subset of the k bits (a learned [MASK] id), embeds the bit
                ids, conditions a stack of AdaLN ResBlocks on (mask-ratio timestep + the AR
                latent), and predicts every bit with CE; masked bits weighted 1.0, unmasked 0.1.
  * inference — iterative confidence-based unmasking (gumbel noise + temperature annealing)
                over a `tokens_allocation` schedule; CFG mixes cond/uncond bit logits.

Cost is O(k)=O(log|C|) per token vs O(|C|) for a flat softmax — what makes a large (FSQ)
codebook learnable for the AR with little data. Needs a *structured* (binary FSQ) index, so it
pairs with ChannelFSQ, not the unstructured IBQ index.

The DiT blocks (RMSNorm / SwiGLU / AdaLN modulate / FinalLayer / Fourier timestep) are vendored
here with standard definitions so this module is self-contained (BAR imports them from
`modeling/modules/blocks.py`). The `MBMHead` wrapper adapts BAR's (target, conditions) API to
our (ar_hidden, flat_index) call sites in cvq/models/car.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# bit <-> int (LSB-first; matches ChannelFSQ binary-level index ordering)
# --------------------------------------------------------------------------- #
def _int_to_bits(idx: torch.Tensor, k: int) -> torch.Tensor:
    bits = (idx.unsqueeze(-1) >> torch.arange(k, device=idx.device)) & 1
    return bits.long()


def _bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    k = bits.shape[-1]
    weights = (2 ** torch.arange(k, device=bits.device)).to(torch.int64)
    return (bits.long() * weights).sum(-1)


# --------------------------------------------------------------------------- #
# Vendored DiT blocks (standard defs; BAR uses the same shapes)
# --------------------------------------------------------------------------- #
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class FinalLayer(nn.Module):
    def __init__(self, width, norm_layer):
        super().__init__()
        self.norm = norm_layer(width)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class ResBlock(nn.Module):
    """AdaLN-modulated SwiGLU residual block (BAR/DiT)."""

    def __init__(self, channels, norm_layer=RMSNorm):
        super().__init__()
        self.in_ln = norm_layer(channels)
        self.mlp = SwiGLUFFN(in_features=channels,
                             hidden_features=int(2 / 3 * int(channels * 4.0)))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp.unsqueeze(1) * h


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.normal(0, scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        t = t[:, None] * self.W[None, :] * 2 * math.pi
        return self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


# --------------------------------------------------------------------------- #
# BAR MaskBitModelingHead (binary), verbatim logic over our vendored blocks
# --------------------------------------------------------------------------- #
class MaskBitModelingHead(nn.Module):
    def __init__(self, num_layers=3, width=1024, seq_len=14):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.seq_len = seq_len
        self.target_codebook_size = 2          # binary FSQ
        self.mask_token_id = 2
        norm_layer = RMSNorm

        per = math.ceil(width / seq_len)
        self.input_embed = nn.Embedding(self.target_codebook_size + 1, per)
        self.input_proj = nn.Linear(per * seq_len, width)
        self.ln_pre = norm_layer(width)
        self.transformer = nn.ModuleList([ResBlock(width, norm_layer) for _ in range(num_layers)])
        self.output_embed = nn.Linear(width, seq_len * self.target_codebook_size)
        self.t_embedder = GaussianFourierEmbedding(width)
        self.adaln_before_head = FinalLayer(width, norm_layer)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def _mask(self, target):
        B, S = target.shape
        z = torch.randn(B, device=target.device) * 0.8
        mask_ratio = torch.clamp(1.0 - torch.sigmoid(z), min=1.0 / S, max=1.0)
        n_mask = (S * mask_ratio).round().clamp(min=1)
        rp = torch.rand(B, S, device=target.device).argsort(dim=-1)
        masks = rp < n_mask[:, None]
        masked = torch.where(masks, torch.full_like(target, self.mask_token_id), target)
        return masked, masks, mask_ratio

    def forward_fn(self, ids, conditions, mask_ratio):
        x = self.input_embed(ids).reshape(ids.shape[0], -1)      # (N, seq*per)
        x = self.input_proj(x)                                   # (N, width)
        s = F.silu(self.t_embedder(mask_ratio).unsqueeze(1) + conditions).squeeze(1)  # (N, width)
        x = self.ln_pre(x).unsqueeze(1)                          # (N, 1, width) — 1 "token"
        for blk in self.transformer:
            x = blk(x, s)
        x = self.adaln_before_head(x, s).squeeze(1)              # (N, width)
        return self.output_embed(x).reshape(-1, self.seq_len, self.target_codebook_size)

    def forward(self, target, conditions):
        """target (N, seq) {0,1}; conditions (N, 1, width) -> (loss, bit_acc)."""
        masked, masks, mask_ratio = self._mask(target)
        pred = self.forward_fn(masked, conditions, mask_ratio)   # (N, seq, 2)
        loss = self.loss_fn(pred.float().transpose(1, 2), target)  # (N, seq)
        w = (1.0 - masks.float()) * 0.1 + masks.float()
        loss = (loss * w).sum() / (w.sum() + 1e-8)
        with torch.no_grad():
            pred_ids = pred.argmax(-1)
            m = masks.float()
            bit_acc = ((pred_ids == target).float() * m).sum() / (m.sum() + 1e-8)
        return loss, bit_acc

    @torch.no_grad()
    def predict_greedy(self, conditions):
        """All-masked single pass -> (N, seq) greedy bits (cheap eval diagnostic)."""
        B = conditions.shape[0]
        ids = torch.full((B, self.seq_len), self.mask_token_id, device=conditions.device)
        mr = torch.ones(B, device=conditions.device)
        return self.forward_fn(ids, conditions, mr).argmax(-1)

    @torch.no_grad()
    def sample(self, conditions, tokens_allocation, randomize_temperature=1.0,
               guidance_scale=1.0, use_cfg=False):
        """Iterative masked-bit decode. conditions (N,1,width); if use_cfg, conditions is
        (2N,1,width) = [cond; uncond]. Returns (N, seq) bits."""
        device = conditions.device
        B = conditions.shape[0] // 2 if use_cfg else conditions.shape[0]

        def gumbel(t):
            n = torch.zeros_like(t).uniform_(0, 1)
            return -torch.log(-torch.log(n.clamp(min=1e-20)).clamp(min=1e-20))

        steps = len(tokens_allocation)
        cum = [0] + [sum(tokens_allocation[: i + 1]) for i in range(steps)]
        ids = torch.full((B, self.seq_len), self.mask_token_id, device=device)
        for step in range(steps):
            nxt_ratio = 1.0 - cum[step + 1] / self.seq_len
            temp = randomize_temperature * (1.0 - step / steps)
            mr = torch.full((B,), 1.0 - cum[step] / self.seq_len, device=device)
            if use_cfg:
                mr2 = mr.repeat(2)
                logits = self.forward_fn(torch.cat([ids, ids], 0), conditions, mr2)
                c, u = logits.split(B, dim=0)
                logits = u + guidance_scale * (c - u)
            else:
                logits = self.forward_fn(ids, conditions, mr)
            noisy = logits + temp * gumbel(logits)
            sampled = noisy.argmax(-1)                            # (B, seq) in {0,1}
            conf = torch.gather(logits, -1, sampled.unsqueeze(-1)).squeeze(-1)
            is_mask = ids == self.mask_token_id
            sampled = torch.where(is_mask, sampled, ids)
            conf = torch.where(is_mask, conf, torch.full_like(conf, float("inf")))
            if step == steps - 1:
                ids = sampled
            else:
                mask_len = max(1, int(round(self.seq_len * nxt_ratio)))
                mask_len = min(mask_len, int(is_mask.sum(-1).min().item()) - 1) if is_mask.sum() else mask_len
                mask_len = max(0, mask_len)
                cutoff = conf.sort(dim=-1).values[:, mask_len:mask_len + 1]
                remask = conf < cutoff
                ids = torch.where(remask, torch.full_like(sampled, self.mask_token_id), sampled)
        return ids


# --------------------------------------------------------------------------- #
# Wrapper adapting BAR's head to our (ar_hidden, flat_index) call sites
# --------------------------------------------------------------------------- #
class MBMHead(nn.Module):
    """Adapts MaskBitModelingHead to channel-AR: projects the Qwen hidden state to the head
    width as the AdaLN condition, and converts flat FSQ indices <-> bits (LSB-first)."""

    def __init__(self, dim: int, codebook_size: int, depth: int = 3, n_heads: int = 8,
                 n_infer_steps: int = 4, width: int | None = None):
        super().__init__()
        self.k = math.ceil(math.log2(codebook_size))
        self.codebook_size = codebook_size
        width = width or dim
        self.cond_proj = nn.Linear(dim, width)
        self.head = MaskBitModelingHead(num_layers=depth, width=width, seq_len=self.k)
        self.alloc = self._default_alloc(self.k, n_infer_steps)

    @staticmethod
    def _default_alloc(n_bits, steps):
        steps = max(1, min(steps, n_bits))
        base, rem = divmod(n_bits, steps)
        alloc = [base] * steps
        for i in range(rem):                      # add the remainder to the tail (non-decreasing)
            alloc[steps - 1 - i] += 1
        return alloc

    def forward(self, ctx, target_idx):
        """ctx (N, dim) AR hidden; target_idx (N,) -> (loss, {'bit_acc'})."""
        bits = _int_to_bits(target_idx, self.k)                  # (N, k) {0,1}
        cond = self.cond_proj(ctx).unsqueeze(1)                  # (N, 1, width)
        loss, bit_acc = self.head(bits, cond)
        return loss, {"bit_acc": bit_acc.item()}

    @torch.no_grad()
    def exact_match(self, ctx, target_idx):
        cond = self.cond_proj(ctx).unsqueeze(1)
        bits = self.head.predict_greedy(cond)
        pred_idx = _bits_to_int(bits).clamp_(0, self.codebook_size - 1)
        return (pred_idx == target_idx).float().mean()

    @torch.no_grad()
    def generate(self, ctx, temperature: float = 1.0):
        cond = self.cond_proj(ctx).unsqueeze(1)
        bits = self.head.sample(cond, self.alloc, randomize_temperature=temperature)
        return _bits_to_int(bits).clamp_(0, self.codebook_size - 1)

    @torch.no_grad()
    def generate_cfg(self, ctx, ctx_u, cfg_scale: float, temperature: float = 1.0):
        cond = torch.cat([self.cond_proj(ctx), self.cond_proj(ctx_u)], dim=0).unsqueeze(1)
        bits = self.head.sample(cond, self.alloc, randomize_temperature=temperature,
                                guidance_scale=cfg_scale, use_cfg=True)
        return _bits_to_int(bits).clamp_(0, self.codebook_size - 1)
