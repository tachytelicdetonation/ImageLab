# Papers & Attribution

Quick reference for the research this project is built on, and exactly what we used
from each. This project is a faithful, scaled reimplementation — the primary paper's
official repo ships no training code, so the method was reconstructed from the paper
plus the prior works it credits.

---

## Primary paper (the method we implement)

### Channel-wise Vector Quantization (CVQ)
- **arXiv:** [2605.26089](https://arxiv.org/abs/2605.26089) (2026)
- **Authors:** Wei Song, Tianhang Wang, Yitong Chen, Tong Zhang, Zuxuan Wu, Ming Li, Jiaqi Wang, Kaicheng Yu
- **Repo:** https://github.com/songweii/CVQ  *(paper + figures only — no code released)*

**What we used from it:**
- **Channel-wise quantization** — quantize each *channel* `z⁽ᵏ⁾∈ℝ^{h·w}` against a shared
  codebook (entry dim = `h·w`), instead of each spatial patch. `z_q⁽ᵏ⁾ = argminₙ‖z⁽ᵏ⁾−eₙ‖²`,
  straight-through `z_q = z + sg[e − z]`. → `cvq/models/quantizer.py`
- **Plain VQ, no tricks** — the paper's claim of ~100% codebook utilization "without any
  bells and whistles": gradient-updated codebook, **no EMA, no dead-code restart**.
- **Nested channel dropout** — `c_keep ~ U(1,c)`, ratio α=0.25, mask remaining channels
  to zero after quantization → coarse-to-fine ordering. → `quantizer.truncate`, `train.sample_c_keep`
- **Loss stack** — pixel-wise ℓ2 + commitment + LPIPS + PatchGAN. → `cvq/losses/losses.py`
- **Channel-count-aware GAN weight** — `λ_GAN(c_keep)=λ₀/(1+e^{−η(c_keep−c/2)})`, η=0.05, λ₀=1.
- **Hyperparameters** — codebook 16,384; token dim 256 (16×16 grid at 256²); Adam(β=0.5,0.9),
  lr 1e-4, wd 1e-4, 100 epochs. (Batch scaled 256→32 for our ~1.3k-image dataset.)
- **Channel-wise Autoregressive (CAR)** next-channel prediction — *planned, phase 2.*

---

## Prior works we drew architecture / code from (acknowledged by CVQ)

| Work | Ref | What we used |
|------|-----|--------------|
| **VILA-U / DualToken** | [arXiv:2409.04429](https://arxiv.org/abs/2409.04429) | SigLIP-encoder + VQ + decoder tokenizer lineage; codebook/quantizer structure reference (`rqvaesiglip`). The earlier EMA-codebook version was ported from here (since removed to match CVQ's plain-VQ). |
| **VQGAN / taming-transformers** | [arXiv:2012.09841](https://arxiv.org/abs/2012.09841) (Esser et al., 2021) | Convolutional decoder (ResNet + attention + nearest-up); PatchGAN adversarial training; last-layer adaptive GAN weight. → `cvq/models/decoder.py` |
| **pix2pix (PatchGAN)** | Isola et al., CVPR 2017 | `NLayerDiscriminator`. → `cvq/models/discriminator.py` |
| **LPIPS** | [arXiv:1801.03924](https://arxiv.org/abs/1801.03924) (Zhang et al., 2018) | Perceptual reconstruction loss (`lpips`, VGG). |
| **SigLIP** | [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) (Zhai et al., 2023) | Frozen ViT image encoder (`google/siglip-base-patch16-256`). → `cvq/models/siglip_encoder.py` |

---

## Notes on faithfulness
- Where the paper is explicit, we match it (channel-wise quant, plain VQ, ℓ2, nested
  dropout α=0.25, λ₀=1, optimizer/lr). Deviations are intentional scale adaptations for a
  ~1,300-image dataset on a single GPU and are documented inline in the configs.
- The semantic loss (`sem_weight`) seen in the repo's `run.sh` belongs to the repo's SigLIP
  *unified* variant, **not** the paper's tokenizer objective — so it is disabled here.
