"""
SigLIP vision encoder wrapper.

The CVQ README states: "we directly inherit the SigLIP architecture, using only its
first n layers." We therefore load a pretrained SiglipVisionModel, optionally truncate
its transformer to the first `n_layers`, and expose the patch features reshaped into a
2D latent grid (B, C, h, w) that the channel-wise quantizer consumes.

SigLIP has no CLS token (it pools with a MAP head), so every output token is a patch:
num_patches = (image_size / patch_size)^2, which reshapes cleanly to a square grid.

SigLIP's image normalization is mean=std=0.5, i.e. it expects pixel values in [-1, 1] —
the exact same range our decoder produces via tanh and our reconstruction target uses.
So one normalized tensor feeds both the encoder and the loss.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import SiglipVisionModel


class SiglipEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-256",
        n_layers: int | None = None,   # None -> use all layers
        freeze: bool = True,
        apply_post_layernorm: bool = True,
    ):
        super().__init__()
        model = SiglipVisionModel.from_pretrained(model_name)
        self.vision = model.vision_model
        cfg = model.config.vision_config if hasattr(model.config, "vision_config") else model.config

        # Truncate to the first n transformer layers (saves compute; matches "first n layers").
        total = len(self.vision.encoder.layers)
        self.n_layers = total if n_layers is None else min(n_layers, total)
        self.vision.encoder.layers = self.vision.encoder.layers[: self.n_layers]
        self.apply_post_layernorm = apply_post_layernorm

        self.hidden_size = cfg.hidden_size              # C, e.g. 768
        self.patch_size = cfg.patch_size                # 16
        self.image_size = cfg.image_size                # 256
        self.grid = self.image_size // self.patch_size  # 16
        self.token_dim = self.grid * self.grid          # 256  (== quantizer token_dim)

        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    @property
    def channels(self) -> int:
        return self.hidden_size

    def train(self, mode: bool = True):
        # Keep a frozen backbone in eval mode regardless of parent .train() calls,
        # so BatchNorm/dropout-free SigLIP stays deterministic.
        super().train(mode)
        if self.frozen:
            # Put the backbone in eval directly; this recurses into the backbone's
            # children only (not back through this module), avoiding infinite recursion.
            self.vision.eval()
        return self

    def forward(self, pixel_values: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        """pixel_values: (B, 3, H, W) in [-1, 1] -> latent grid (B, C, grid, grid).

        with_grad=True builds the autograd graph through the (still param-frozen)
        backbone, which the semantic loss needs to push gradients into the decoder
        via the reconstruction. For encoding the *input*, leave it False (no_grad,
        memory-efficient — the trainable adapter starts a fresh graph from feat).
        """
        use_grad = with_grad or (not self.frozen)
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            h = self.vision.embeddings(pixel_values)            # (B, num_patches, C)
            # encoder.layers were truncated; SiglipEncoder layer returns a tuple
            attn_mask = None
            for layer in self.vision.encoder.layers:
                h = layer(h, attention_mask=attn_mask, output_attentions=False)[0]
            if self.apply_post_layernorm:
                h = self.vision.post_layernorm(h)
        B, N, C = h.shape
        assert N == self.grid * self.grid, f"expected {self.grid**2} patches, got {N}"
        # (B, N, C) -> (B, C, grid, grid)
        return h.permute(0, 2, 1).reshape(B, C, self.grid, self.grid)
