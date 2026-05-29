"""
CVQ tokenizer: trainable CNN encoder -> channel-wise IBQ -> trainable VQGAN decoder.

  image (B,3,256,256) in [-1,1]
     │  CNN encoder (VQGAN, from scratch) — f = 2^(len(enc_ch_mult)-1) = 16
     ▼
  z    (B, Ctok, 16, 16)          Ctok channel-tokens, each of dim 16*16=256
     │  channel-wise IBQ (softmax-over-all-codes + index-backprop STE + nested dropout)
     ▼
  z_q  (B, Ctok, 16, 16)
     │  decoder (trainable, VQGAN)
     ▼
  recon (B,3,256,256) in [-1,1]

The encoder is a single swappable component. Today it is the from-scratch CNN (the paper's
VQGAN setup, which beat the frozen-SigLIP semantic encoder on reconstruction — see RESULTS.md).
A future semantic encoder (e.g. a LeJEPA-pretrained ViT, for the Phase-2 CAR) plugs in here.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoder import Decoder
from .encoder_cnn import Encoder as CNNEncoder
from .quantizer import IBQChannelVQ


class CVQTokenizer(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        latent_channels: int = 256,     # number of channel-tokens (CAR sequence length)
        codebook_size: int = 16384,
        commitment_beta: float = 0.25,
        quantizer_kwargs: dict | None = None,
        enc_ch: int = 128,
        enc_ch_mult=(1, 1, 2, 2, 4),
        decoder_ch: int = 128,
        decoder_ch_mult=(1, 1, 2, 2, 4),
        decoder_res_blocks: int = 2,
    ):
        super().__init__()
        # Trainable CNN encoder from scratch (paper's VQGAN setup). f = 2^(len-1).
        g = resolution // (2 ** (len(enc_ch_mult) - 1))
        self.encoder = CNNEncoder(
            ch=enc_ch, ch_mult=tuple(enc_ch_mult), num_res_blocks=decoder_res_blocks,
            z_channels=latent_channels, resolution=resolution, attn_resolutions=(g,),
        )
        token_dim = g * g

        self.quantizer = IBQChannelVQ(
            codebook_size=codebook_size,
            token_dim=token_dim,
            commitment_beta=commitment_beta,
            **(quantizer_kwargs or {}),
        )
        self.decoder = Decoder(
            ch=decoder_ch,
            out_ch=3,
            ch_mult=decoder_ch_mult,
            num_res_blocks=decoder_res_blocks,
            z_channels=latent_channels,
            resolution=resolution,
            attn_resolutions=(g,),
        )
        self.latent_channels = latent_channels
        self.grid = g

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
        }

    def trainable_parameters(self):
        """Params optimized by the generator optimizer: the whole tokenizer (encoder is
        trained from scratch, codebook is a plain gradient-updated embedding)."""
        return list(self.parameters())
