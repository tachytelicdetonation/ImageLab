# CVQ-Pokémon

A faithful, scaled-down reimplementation of **Channel-wise Vector Quantization (CVQ)**
([arXiv:2605.26089](https://arxiv.org/abs/2605.26089)) trained on a Pokémon image
dataset (all ~1,300 species *and variant forms*, official artwork, name as caption).

> **Why a reimplementation?** The official [`songweii/CVQ`](https://github.com/songweii/CVQ)
> repo currently ships only the paper, a README, and figures — **no training code**.
> This project reconstructs the method from the paper + README hyperparameters, grounded
> in the lineage it credits (VILA-U / DualToken, OpenCLIP, taming-transformers/VQGAN).

## What CVQ does

Standard VQ-VAE/VQGAN quantizes each **spatial patch** of a feature map. CVQ instead
quantizes each **channel**: a feature map `Z ∈ (B, C, h, w)` becomes `C` tokens, each a
flattened `h·w`-dim vector matched to a shared codebook. An image is then represented as
*levels of visual detail* rather than a grid of patches. **Nested channel dropout**
(keep the first `c_keep` channels, zero the rest) forces a coarse-to-fine ordering — the
basis for the paper's "next-channel prediction" generative model (CAR, phase 2 here).

```
image 256x256 ─► SigLIP ViT (frozen, first-n layers) ─► (B,768,16,16)
   ─► trainable channel adapter ─► (B, 256, 16, 16)            256 channel-tokens, dim 16*16=256
   ─► channel-wise VQ (EMA codebook + dead-code restart + nested dropout)
   ─► VQGAN conv decoder ─► reconstruction 256x256
Losses: L1/L2 + LPIPS + PatchGAN (adaptive λ, channel-count-aware) + SigLIP semantic + VQ commitment
```

## Project layout

```
cvq/
  data/download_pokemon.py   # PokéAPI -> images_256/ + manifest.jsonl  (edit make_caption!)
  data/dataset.py            # loads images in [-1,1]
  models/siglip_encoder.py   # frozen SigLIP, first-n layers -> latent grid
  models/quantizer.py        # ChannelwiseVQ: EMA codebook, restart, nested dropout
  models/decoder.py          # taming-transformers VQGAN decoder (MPS/CUDA clean)
  models/discriminator.py    # PatchGAN
  models/tokenizer.py        # ties it together (encode/decode)
  losses/losses.py           # full CVQ loss stack
  train.py                   # training loop (auto device, bf16 AMP, Stage I/II)
  reconstruct.py             # eval: recon grid, codebook utilization, coarse-to-fine
configs/
  cvq_pokemon.yaml           # local Mac / MPS (batch 4, codebook 4096)
  cvq_pokemon_cuda.yaml      # A100/H100 (batch 32, codebook 16384, bf16)
scripts/
  setup_server.sh            # install deps on a CUDA box (keeps preinstalled torch)
  run_server.sh              # download data + train
```

## Run on a CUDA server (A100/H100) — recommended

```bash
git clone <your-repo-url> cvq-pokemon && cd cvq-pokemon
bash scripts/setup_server.sh          # installs deps (keeps the image's CUDA torch)
bash scripts/run_server.sh            # downloads data, then trains cvq_pokemon_cuda.yaml

# monitor
tensorboard --logdir runs --port 6006
# evaluate
python -m cvq.reconstruct --ckpt checkpoints/latest.pt --n 8
```

Long runs: `nohup bash scripts/run_server.sh > train.log 2>&1 &` then `tail -f train.log`.
Training is **resumable** from a full checkpoint (which includes optimizer state):
`python -m cvq.train --config configs/cvq_pokemon_cuda.yaml --resume checkpoints/cvq_step010000.pt`
(`latest.pt` is model-only for eval; the `cvq_step*.pt` files are the resumable ones.)

### Stage I vs Stage II
- **Stage I** (default): SigLIP frozen; only the adapter + decoder + codebook learn.
- **Stage II** (end-to-end): set `model.freeze_encoder: false` in the CUDA config. SigLIP
  is finetuned at `train.encoder_lr` (2e-5) while the head stays at `lr` (1e-4).

## Run locally (Mac / Apple Silicon)

```bash
uv venv --python 3.12 && uv pip install -e .
python -m cvq.data.download_pokemon --size 256       # ~1,300 images
python -m cvq.train --config configs/cvq_pokemon.yaml
```
The Mac config freezes SigLIP and uses batch 4 + the 16×16 grid; expect ~3 img/s on an
M-series GPU (so the full 100-epoch run is ~12h — the CUDA path is far faster).

## Your knobs
- `cvq/data/download_pokemon.py :: make_caption` — how Pokémon names become captions
  (keep variant forms vs. collapse to base name vs. templated). Shapes phase-2 generation.
- `cvq/train.py :: sample_c_keep` — the nested-dropout policy (how many channels to keep).
- `configs/*.yaml` — codebook size, losses, batch, AMP, stages.

## Faithfulness notes
- Hyperparameters mirror the README (`lr 1e-4`, `β=(0.5,0.9)`, `wd 1e-4`, codebook 16384,
  100 epochs, `sem_weight 1`, `gan_eta 0.05`); warmup/disc-start are scaled to the smaller
  dataset. Deviations are commented in the YAML.
- The codebook EMA + dead-code restart are ported from VILA-U's `VQEmbedding`.
- The decoder is the taming-transformers VQGAN decoder (hardcoded bf16 removed for portability).

## Status
- [x] Channel-wise VQ tokenizer (encode → quantize → decode) — trains & validated
- [ ] CAR next-channel autoregressive text-to-image model (phase 2)
