"""
DINOv2 semantic alignment — EOSTok's third contribution (arXiv:2605.00503 §3.3, Eq. 6).

EOSTok regularizes the tokenizer's latent toward a vision-foundation-model representation so
the 1D/channel latent carries global semantics (REPA-style, yu2025repa). The implicit-alignment
loss is:

    L_implicit(omega, phi) = -(1/N) sum_n  sim( h_omega(h_Enc[n]), y[n] )

where h_Enc are the encoder's hidden patch embeddings, y are frozen DINOv2 patch features, h_omega
is a small LEARNED MLP projector (its params omega are optimized jointly), and sim is cosine.

Implementation notes / faithful adaptations for our channel-wise CNN stack:
  * h_Enc: we align the pre-quant latent z (B, Cch, 16, 16). EOSTok aligns a pre-latent encoder
    hidden so z itself isn't forced to match; with a CNN encoder the pre-quant feature map is the
    natural REPA hook and keeps the patch grid (16x16) aligned to DINOv2's. Documented deviation.
  * y: DINOv2-ViT-L/14 patch tokens. At 224x224 the grid is 224/14 = 16x16 = 256 patches, which
    EXACTLY matches our 16x16 latent grid, so alignment is per-patch with no interpolation.
  * EOSTok also has a decoder-alignment term on masked-decoder tokens. Our VQGAN decoder has no
    mask tokens, so that term is omitted (would require a masked-decoder rebuild). The implicit
    term is EOSTok's main contributor (Table 2).

DINOv2 is frozen (no grad); only h_omega and (through the cosine target) the encoder/tokenizer
receive gradient. Image is renormalized from [-1,1] (our pixel range) to ImageNet stats for DINOv2.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOAlign(nn.Module):
    def __init__(self, latent_channels: int, grid: int,
                 dino_name: str = "facebook/dinov2-large", dino_dim: int = 1024,
                 proj_hidden: int = 2048, dino_res: int = 224):
        super().__init__()
        from transformers import AutoModel

        self.grid = grid
        self.dino_res = dino_res
        self.dino = AutoModel.from_pretrained(dino_name)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # h_omega: learned MLP projecting the latent's per-patch channels -> DINOv2 dim.
        self.proj = nn.Sequential(
            nn.Linear(latent_channels, proj_hidden), nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden), nn.GELU(),
            nn.Linear(proj_hidden, dino_dim),
        )
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def _dino_features(self, x):
        """x in [-1,1] (B,3,H,W) -> frozen DINOv2 patch features (B, grid*grid, dino_dim)."""
        x01 = (x.clamp(-1, 1) * 0.5 + 0.5)                         # -> [0,1]
        x01 = F.interpolate(x01, size=(self.dino_res, self.dino_res),
                            mode="bicubic", align_corners=False)
        x01 = (x01 - self.mean) / self.std
        out = self.dino(pixel_values=x01)
        feat = out.last_hidden_state[:, 1:, :]                     # drop CLS -> (B, P, dino_dim)
        return feat

    def forward(self, z, x):
        """z: (B, Cch, g, g) pre-quant latent. x: (B,3,H,W) source image in [-1,1].
        Returns scalar L_implicit (lower is better)."""
        B, Cch, g, _ = z.shape
        with torch.no_grad():
            y = self._dino_features(x).float()                    # (B, P, D)
        P = y.shape[1]
        zt = z.permute(0, 2, 3, 1).reshape(B, g * g, Cch)         # (B, g*g, Cch)
        if g * g != P:
            # grids differ -> bilinearly resize DINOv2 patch grid to (g,g)
            side = int(round(P ** 0.5))
            yg = y.transpose(1, 2).reshape(B, -1, side, side)
            yg = F.interpolate(yg, size=(g, g), mode="bilinear", align_corners=False)
            y = yg.reshape(B, -1, g * g).transpose(1, 2)
        h = self.proj(zt.float())                                 # (B, g*g, D)
        sim = F.cosine_similarity(h, y, dim=-1)                   # (B, g*g)
        return (1.0 - sim).mean()                                 # = -sim + const, minimized
