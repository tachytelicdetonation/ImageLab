"""
AR output heads — the pluggable seam between the CAR backbone and the token space.

The backbone produces one hidden state per image position; a head turns hidden states
into (a) a training loss against the target token indices and (b) a sampled token at
generation time. Fork A is a flat K-way softmax (EOSTok NTP); Fork B is masked-bit
modeling over FSQ's bit-structured indices (BAR). A new head (e.g. a diffusion head,
a hierarchical softmax) registers under "ar_head" and is selected by `model.head_type`
in the YAML — CAR itself never changes.

Contract:
    loss(img_hidden (B,C,H), targets (B,C), channel_weights (C,)|None)
        -> (loss, logs: dict, aux: dict)
        aux carries head-specific tensors downstream losses need (the softmax head
        exposes aux["logits"] for EOSTok's APR soft-decode; MBM has no soft decode).
    sample(last_h (B,H), u_last_h (B,H)|None, *, temperature, top_k, cfg_scale, do_cfg)
        -> (B, 1) int64 token indices
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cvq.registry import register

from .mbm_head import MBMHead


@register("ar_head", "softmax", paper="arXiv:2605.00503")
class SoftmaxARHead(nn.Module):
    """Flat K-way classification per channel-token (EOSTok / Fork A)."""

    def __init__(self, hidden: int, codebook_size: int,
                 backbone_dtype: torch.dtype = torch.bfloat16, **_ignore):
        super().__init__()
        self.hidden = hidden
        self.codebook_size = codebook_size
        self.proj = nn.Linear(hidden, codebook_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.proj.to(backbone_dtype)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Head in fp32 for stable cross-entropy / softmax."""
        return self.proj(hidden).float()

    def loss(self, img_hidden, targets, channel_weights=None):
        B, C = targets.shape
        logits = self.logits(img_hidden)                          # (B, C, K) fp32
        K = logits.shape[-1]
        ce_pt = F.cross_entropy(logits.reshape(-1, K), targets.reshape(-1),
                                reduction="none").reshape(B, C)
        if channel_weights is None:
            loss = ce_pt.mean()
        else:
            loss = (ce_pt * channel_weights.to(ce_pt.device)[None, :]).mean()
        with torch.no_grad():
            acc = (logits.argmax(-1) == targets).float().mean()
            cprefix = max(1, C // 4)
            acc_prefix = (logits[:, :cprefix].argmax(-1) ==
                          targets[:, :cprefix]).float().mean()
        logs = {"car/ntp_loss": loss.item(), "car/token_acc": acc.item(),
                "car/token_acc_prefix": acc_prefix.item()}
        return loss, logs, {"logits": logits}

    def sample(self, last_h, u_last_h=None, *, temperature=1.0, top_k=0,
               cfg_scale=1.0, do_cfg=False):
        logits = self.logits(last_h)
        if do_cfg:
            ulogits = self.logits(u_last_h)
            logits = ulogits + cfg_scale * (logits - ulogits)
        logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        return torch.multinomial(logits.softmax(-1), 1)


@register("ar_head", "mbm", paper="arXiv:2602.09024")
class MBMARHead(nn.Module):
    """Masked-bit modeling over FSQ bit indices (BAR / Fork B).

    No soft-codebook decode exists for bit prediction, so aux is empty and the EOSTok
    APR loss must be off (config validation enforces lambda_apr=0)."""

    def __init__(self, hidden: int, codebook_size: int, mbm_depth: int = 3,
                 mbm_heads: int = 8, mbm_infer_steps: int = 4, **_ignore):
        super().__init__()
        self.hidden = hidden
        self.codebook_size = codebook_size
        self.mbm = MBMHead(hidden, codebook_size, depth=mbm_depth,
                           n_heads=mbm_heads, n_infer_steps=mbm_infer_steps)

    def loss(self, img_hidden, targets, channel_weights=None):
        B, C = targets.shape
        ctx = img_hidden.reshape(B * C, self.hidden).float()
        tgt = targets.reshape(B * C)
        loss, hlogs = self.mbm(ctx, tgt)
        with torch.no_grad():
            tok_acc = self.mbm.exact_match(ctx, tgt)
        logs = {"car/ntp_loss": loss.item(), "car/bit_acc": hlogs["bit_acc"],
                "car/token_acc": tok_acc.item()}
        return loss, logs, {}

    def sample(self, last_h, u_last_h=None, *, temperature=1.0, top_k=0,
               cfg_scale=1.0, do_cfg=False):
        if do_cfg:
            idx = self.mbm.generate_cfg(last_h.float(), u_last_h.float(),
                                        cfg_scale, temperature)
        else:
            idx = self.mbm.generate(last_h.float(), temperature)
        return idx.unsqueeze(1)
