"""
VQGAN-style convolutional decoder, ported from taming-transformers (the exact decoder
lineage VILA-U / CVQ build on; see VILA-U rqvaesiglip/modules.py).

Changes vs. the original:
  * The original hardcodes bfloat16 in the Upsample op (an NVIDIA-ism). We keep
    everything in the module's native dtype so it runs cleanly on Apple MPS.
  * Exposed as a self-contained module that maps a latent grid (B, z_channels, h, w)
    back to an image (B, 3, H, W) in [-1, 1].
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)  # b, n, c
        k = k.reshape(b, c, hh * ww)                    # b, c, n
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=2)
        v = v.reshape(b, c, hh * ww)
        out = torch.bmm(v, attn.permute(0, 2, 1)).reshape(b, c, hh, ww)
        return x + self.proj_out(out)


class Decoder(nn.Module):
    """Latent grid -> image. Faithful taming-transformers decoder structure."""

    def __init__(
        self,
        *,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        z_channels: int = 256,
        resolution: int = 256,
        tanh_out: bool = True,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to keep order consistent with i_level

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h
