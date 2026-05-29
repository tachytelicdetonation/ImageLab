"""
CVQ tokenizer: frozen SigLIP encoder -> trainable channel adapter -> channel-wise VQ
-> trainable VQGAN decoder.

  image (B,3,256,256) in [-1,1]
     │  SigLIP (frozen, first-n layers)
     ▼
  feat (B, 768, 16, 16)
     │  adapter (trainable conv head) — re-bases features into channel-token space
     ▼
  z    (B, Ctok, 16, 16)          Ctok channel-tokens, each of dim 16*16=256
     │  channel-wise VQ (EMA codebook + dead-code restart + nested dropout)
     ▼
  z_q  (B, Ctok, 16, 16)
     │  decoder (trainable, VQGAN)
     ▼
  recon (B,3,256,256) in [-1,1]

Stage I trains {adapter, decoder, discriminator} with the codebook moved by EMA; the
SigLIP backbone is frozen. Stage II can unfreeze SigLIP for end-to-end finetuning.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoder import Decoder, Normalize, ResnetBlock, nonlinearity
from .quantizer import ChannelwiseVQ
from .siglip_encoder import SiglipEncoder


class ChannelAdapter(nn.Module):
    """Trainable head mapping frozen SigLIP features (B,768,g,g) -> (B,Ctok,g,g).

    A couple of residual blocks give it enough capacity to learn a coarse-to-fine
    channel ordering under nested dropout, while staying tiny relative to SigLIP.
    """

    def __init__(self, in_ch: int, out_ch: int, hidden: int = 512):
        super().__init__()
        self.block1 = ResnetBlock(in_channels=in_ch, out_channels=hidden)
        self.block2 = ResnetBlock(in_channels=hidden, out_channels=hidden)
        self.norm = Normalize(hidden)
        self.proj = nn.Conv2d(hidden, out_ch, kernel_size=1)

    def forward(self, x):
        h = self.block2(self.block1(x))
        return self.proj(nonlinearity(self.norm(h)))


class CVQTokenizer(nn.Module):
    def __init__(
        self,
        siglip_name: str = "google/siglip-base-patch16-256",
        siglip_layers: int | None = None,
        freeze_encoder: bool = True,
        latent_channels: int = 256,     # number of channel-tokens (CAR sequence length)
        codebook_size: int = 16384,
        codebook_decay: float = 0.99,
        commitment_beta: float = 0.25,
        decoder_ch: int = 128,
        decoder_ch_mult=(1, 1, 2, 2, 4),
        decoder_res_blocks: int = 2,
    ):
        super().__init__()
        self.encoder = SiglipEncoder(siglip_name, n_layers=siglip_layers, freeze=freeze_encoder)
        g = self.encoder.grid
        token_dim = self.encoder.token_dim  # g*g

        self.adapter = ChannelAdapter(self.encoder.channels, latent_channels)
        self.quantizer = ChannelwiseVQ(
            codebook_size=codebook_size,
            token_dim=token_dim,
            decay=codebook_decay,
            commitment_beta=commitment_beta,
        )
        self.decoder = Decoder(
            ch=decoder_ch,
            out_ch=3,
            ch_mult=decoder_ch_mult,
            num_res_blocks=decoder_res_blocks,
            z_channels=latent_channels,
            resolution=self.encoder.image_size,
            attn_resolutions=(g,),
        )
        self.latent_channels = latent_channels
        self.grid = g

    # ---- encode / decode halves (used by CAR in phase 2) ----
    def encode(self, x):
        feat = self.encoder(x)
        z = self.adapter(feat)
        z_q, idxs, vq_loss, stats = self.quantizer(z)
        return z, z_q, idxs, vq_loss, stats

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x, c_keep: int | None = None):
        """Full reconstruction.

        Args:
            x: (B,3,H,W) in [-1,1]
            c_keep: if given, apply nested channel dropout (keep first c_keep channels).
        Returns dict with recon, vq_loss, indices, codebook stats, and the encoder
        feature (for the semantic loss).
        """
        feat = self.encoder(x)
        z = self.adapter(feat)
        z_q, idxs, vq_loss, stats = self.quantizer(z)
        if c_keep is not None:
            z_q = self.quantizer.truncate(z_q, c_keep)
        recon = self.decoder(z_q)
        return {
            "recon": recon,
            "vq_loss": vq_loss,
            "indices": idxs,
            "stats": stats,
            "siglip_feat": feat,
        }

    def trainable_parameters(self):
        """Params optimized by the generator optimizer.

        Now includes the quantizer: the codebook is a plain gradient-updated
        embedding (paper's no-EMA VQ), so it must be in the optimizer. Excludes the
        frozen SigLIP encoder.
        """
        params = (list(self.adapter.parameters())
                  + list(self.quantizer.parameters())
                  + list(self.decoder.parameters()))
        if not self.encoder.frozen:
            params += [p for p in self.encoder.parameters() if p.requires_grad]
        return params
