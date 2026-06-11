"""A small time-conditioned UNet — the shared backbone for the diffusion examples.

Standard DDPM-style architecture (Ho et al. 2020, App. B) scaled down for 64px/single
GPU (the rectified_flow example imports it too): sinusoidal time embedding -> MLP,
residual blocks with time injection, one
self-attention at the lowest resolution, skip connections. ~12M params at ch=64.

Nothing here is framework-specific — any nn.Module works in a task; this file exists so
the examples have a credible model without an external dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of (possibly fractional) timesteps, transformer-style."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, temb_ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = F.scaled_dot_product_attention(  # (B, HW, C) single-head
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(attn.transpose(1, 2).reshape(B, C, H, W))


class TinyUNet(nn.Module):
    """forward(x (B,3,H,W), t (B,)) -> (B,3,H,W). `t` may be float (rectified flow
    passes t*1000 so both examples share one embedding scale)."""

    def __init__(self, ch: int = 64, ch_mult=(1, 2, 2), num_res_blocks: int = 2,
                 in_ch: int = 3):
        super().__init__()
        temb_ch = ch * 4
        self.ch = ch
        self.time_mlp = nn.Sequential(nn.Linear(ch, temb_ch), nn.SiLU(),
                                      nn.Linear(temb_ch, temb_ch))
        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)

        chans = [ch * m for m in ch_mult]
        self.downs = nn.ModuleList()
        skip_chs, cur = [ch], ch
        for i, c in enumerate(chans):
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(cur, c, temb_ch))
                skip_chs.append(c)
                cur = c
            if i < len(chans) - 1:
                self.downs.append(nn.Conv2d(cur, cur, 3, stride=2, padding=1))
                skip_chs.append(cur)

        self.mid = nn.ModuleList([ResBlock(cur, cur, temb_ch), Attention(cur),
                                  ResBlock(cur, cur, temb_ch)])

        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResBlock(cur + skip_chs.pop(), c, temb_ch))
                cur = c
            if i > 0:
                self.ups.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.out_norm = nn.GroupNorm(8, cur)
        self.out_conv = nn.Conv2d(cur, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)    # predict ~0 at init: stable early training
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.ch))
        h = self.stem(x)
        skips = [h]
        for layer in self.downs:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, temb) if isinstance(layer, ResBlock) else layer(h)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], dim=1), temb)
            else:
                h = layer(h)
        return self.out_conv(F.silu(self.out_norm(h)))
