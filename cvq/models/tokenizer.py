"""
CVQ tokenizer: encoder -> channel-wise quantizer -> decoder.

  image (B,3,H,W) in [-1,1]
     │  encoder (registry "encoder", default: from-scratch VQGAN CNN) — f = 2^(len(enc_ch_mult)-1)
     ▼
  z    (B, Ctok, g, g)          Ctok channel-tokens, each of dim g*g
     │  channel-wise quantizer (registry "quantizer": ibq | fsq | yours)
     ▼
  z_q  (B, Ctok, g, g)
     │  decoder (registry "decoder", default: VQGAN)
     ▼
  recon (B,3,H,W) in [-1,1]

The tokenizer is pure composition: components are built by `cvq.factory.build_tokenizer`
from the config's registry names and injected here. To try a new encoder/quantizer/decoder,
register it (see cvq/registry.py) and name it in the YAML — this file does not change.
"""

from __future__ import annotations

import torch
from torch import nn


class CVQTokenizer(nn.Module):
    def __init__(self, encoder: nn.Module, quantizer: nn.Module, decoder: nn.Module,
                 latent_channels: int, grid: int):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.latent_channels = latent_channels
        self.grid = grid

    # ---- encode / decode halves (used by CAR in phase 2) ----
    def encode(self, x):
        z = self.encoder(x)
        z_q, idxs, vq_loss, stats = self.quantizer(z)
        return z, z_q, idxs, vq_loss, stats

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x, c_keep: int | None = None):
        """Full reconstruction.

        Args:
            x: (B,3,H,W) in [-1,1]
            c_keep: if given, apply nested channel dropout (keep first c_keep channels).
        Returns dict with recon, vq_loss, indices, and codebook stats.
        """
        z = self.encoder(x)
        z_q, idxs, vq_loss, stats = self.quantizer(z)
        if c_keep is not None:
            z_q = self.quantizer.truncate(z_q, c_keep)
        recon = self.decoder(z_q)
        return {
            "recon": recon,
            "vq_loss": vq_loss,
            "indices": idxs,
            "stats": stats,
            "z": z,            # pre-quant latent (for DINOv2 semantic alignment in E2E)
        }

    def trainable_parameters(self):
        """Params optimized by the generator optimizer: the whole tokenizer (encoder is
        trained from scratch, codebook is a plain gradient-updated embedding)."""
        return list(self.parameters())
