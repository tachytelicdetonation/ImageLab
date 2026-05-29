"""
VQGAN-style convolutional encoder, trained from scratch — the paper's actual tokenizer
setup ("standard VQGAN approach"). Mirrors the decoder in decoder.py: ResNet blocks +
self-attention at the bottleneck + strided downsampling.

Maps an image (B, 3, H, W) to a latent grid (B, z_channels, H/f, W/f). For channel-wise
VQ we set z_channels = number of channel-tokens (e.g. 256) and downsample f=16 so the
grid is 16x16 -> token dim = 256. Encoder and codebook then co-adapt during training,
which is what lets plain VQ avoid codebook collapse (no EMA / restart needed).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import AttnBlock, Normalize, ResnetBlock, nonlinearity


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # asymmetric pad (0,1,0,1) then stride-2 conv — taming convention
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int = 128,
        ch_mult=(1, 1, 2, 2, 4),   # 256 -> 16 (4 downsamples)
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int = 3,
        resolution: int = 256,
        z_channels: int = 256,     # = number of channel-tokens (latent_channels)
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h  # (B, z_channels, grid, grid)
