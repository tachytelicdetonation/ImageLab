# Experiment Log — CVQ Tokenizer

Living record of every tokenizer training run: what we changed, why, and what
happened. The point is **memory** — when we propose the next modification, we read
this first so we never re-run a dead end or forget a result after a server is wiped.

Pair this with [Papers.md](Papers.md) (what the method *should* be) — this file is
what the method *actually did* on our data.

## Setup (constant unless a run says otherwise)
- **Data:** ~1.3k Pokémon official-artwork images, 256×256, hflip aug. (ImageNet-scale
  the paper assumes — we are 3–4 orders of magnitude smaller. This is the central
  tension behind every collapse below.)
- **Encoder:** CNN VQGAN from scratch (paper's method), f=16 → 16×16 grid, z=256 ch.
- **Quantizer:** channel-wise VQ, token dim 256 (=16·16), `commitment_beta=0.25`.
- **Losses:** ℓ2 + LPIPS(vgg, w=1) + commitment; PatchGAN with `disc_start_step=2000`.
- **Optim:** AdamW(0.5, 0.9), lr 1e-4, wd 1e-4, bf16, batch 16 × grad_accum 2 = 32 eff.
- **Hardware:** Vast.ai RTX 6000 Ada (49 GB).

> **Short-run caveat:** the 1k-step screening runs below stop at step 1000, but
> `disc_start_step=2000`, so **the GAN never turns on** in them. They measure the pure
> autoencoder + VQ behavior (recon + LPIPS + codebook dynamics). That's exactly what we
> want for diagnosing codebook collapse — the GAN is downstream of a healthy codebook.

## What to watch (the collapse signature)
| Metric (wandb key) | Healthy | Collapsing |
|---|---|---|
| `codebook/usage_batch` | rises toward 1.0 | falls toward ~0 (few codes win) |
| `codebook/perplexity` | high (≫ 1) | → 1 (one code dominates) |
| `loss/vq` | small, stable | diverges upward |
| `val/codebook_utilization_full` | high | tiny (e.g. <1%) |
| `val/rFID` | falls | high / NaN |

---

## Methodology for screening runs
We cap experimental runs at **1000 steps** (~20 min) instead of 100 epochs. Codebook
collapse, when it happens, is visible within a few hundred steps, so the *trajectory*
of `usage_batch` / `perplexity` / `loss/vq` over 1k steps is enough to decide go/no-go.
Only promising trajectories graduate to a full run.

---

## Runs

### Legend
`status`: 🟡 queued · 🔵 running · ✅ done · ❌ collapsed · ⭐ promising

| # | Date | Change vs literal | Steps | Codebook | Final usage | Final ppl | rFID | Status |
|---|------|-------------------|-------|----------|-------------|-----------|------|--------|
| 1 | 2026-05-29 | none (100% literal baseline) | 1000 | 16384 | _tbd_ | _tbd_ | _tbd_ | 🟡 |
| 2 | 2026-05-29 | codebook 16384→4096 | 1000 | 4096 | _tbd_ | _tbd_ | _tbd_ | 🟡 |
| 3 | 2026-05-29 | + entropy loss (w=0.1, τ=1) | 1000 | 16384 | _tbd_ | _tbd_ | _tbd_ | 🟡 |

---

### Run 1 — Literal baseline (`cvq-cnn-literal-cb16384-1k`)
- **Config:** `configs/cvq_pokemon_cnn.yaml`
- **Hypothesis:** the paper's "no bells and whistles" plain VQ collapses at our scale —
  16384 codes ≫ what ~1.3k images of 256 channels can populate. Establishes the
  reference everything else is measured against.
- **wandb:** _link tbd_
- **Result:** _tbd_
- **Reconstructions:** _tbd_
- **Takeaway:** _tbd_

### Run 2 — Smaller codebook (`cvq-cnn-literal-cb4096-1k`)
- **Config:** `configs/cvq_pokemon_cnn_cb4096.yaml`
- **Hypothesis:** shrinking the codebook 4× sizes capacity closer to the data. If
  collapse is fundamentally a "too many codes for too little data" problem, usage and
  perplexity should hold up noticeably better than Run 1 — *without* any VQ trick, so
  it stays a clean, paper-adjacent change.
- **wandb:** _link tbd_
- **Result:** _tbd_
- **Reconstructions:** _tbd_
- **Takeaway:** _tbd_

### Run 3 — Entropy loss (`cvq-cnn-entropy0.1-cb16384-1k`)
- **Config:** `configs/cvq_pokemon_cnn_entropy.yaml` (`entropy_weight=0.1`, `entropy_temperature=1.0`)
- **Hypothesis:** an explicit usage-pressure loss can fix collapse without EMA/restart. Adds
  the MAGVIT-v2 / VQGAN entropy regularizer on the soft assignment
  `q = softmax(-dist/τ)`:  `L_ent = E[H(q)] − H(E[q])`. Term 1 (min) keeps assignments
  sharp; term 2 (max marginal entropy) flattens aggregate usage → all codes pulled in.
- **Why this loss:** it is the **VQ analogue of the LLM MoE load-balancing loss**
  (Switch Transformer: `N·Σ fᵢ·Pᵢ`) — codebook collapse and MoE routing collapse are the
  same winner-take-all failure. Cheapest principled fix; no architecture change.
- **Implementation:** `cvq/models/quantizer.ChannelwiseVQ._entropy_loss`. Gated by
  `entropy_weight` (0.0 ⇒ exactly the literal baseline). Soft `dist` kept with-grad so it
  backprops to encoder + codebook; argmin detached. Memory: `(B·C, N)` softmax — fine at
  16k, scales linearly (MAGVIT-v2's caveat for huge codebooks).
- **wandb keys added:** `codebook/entropy_loss`, `entropy_per_sample`, `entropy_marginal`.
- **Result:** _tbd_
- **Takeaway:** _tbd_

#### Research backing (2026 arXiv search, 2026-05-29)
- **Entropy loss** — MAGVIT-v2 / "Language Model Beats Diffusion" ([arXiv:2310.05737](https://arxiv.org/html/2310.05737v2)); formula `E[H(q)] − H(E[q])`, "inspired by image VQGAN." ← what we implemented.
- **Beyond Stationarity** ([arXiv:2602.18896](https://arxiv.org/abs/2602.18896), Feb 2026) — root cause = encoder drift stranding unselected codes; fixes are *architectural* (NSVQ kernel rule, TransVQ codebook transform), near-100% utilization. Candidate if a loss alone underperforms.
- **Scalable Training, 100% Codebook Utilization** ([arXiv:2509.10140](https://arxiv.org/pdf/2509.10140), 2025).
- **Distributional/Wasserstein matching** ([arXiv:2506.15078](https://arxiv.org/pdf/2506.15078), 2025) — align feature & code distributions.
- **VQGAN-LC / one linear layer** ([arXiv:2411.02038](https://arxiv.org/html/2411.02038v1)) — frozen pretrained codebook + projection, ~99% at 100k codes.
- **LLM analogue** — MoE load-balancing aux loss (Switch Transformer): `N·Σ fᵢ·Pᵢ`, same objective as entropy term 2.

---

## Backlog (candidate modifications, in rough order of principled-ness)
Each is an explicit, documented deviation on top of the literal baseline. Pick based on
what the screening runs above reveal.
0. **Entropy / load-balancing loss** — ✅ implemented, queued as Run 3 above.
1. **Codebook init** — data-dependent / unit-scale init (attacks cold-start; barely a trick).
2. **More data** — game sprites + augmentation, codebook sized to data (attacks root cause).
3. **Cosine codebook** (ViT-VQGAN L2-norm) — proven high utilization, no EMA.
4. **EMA + dead-code restart** — heaviest hammer, known to work (our earlier v3 run).

## History before this log (from prior sessions, pre-1k-step methodology)
- Plain VQ collapse confirmed **3×** at full-length runs: usage → ~0.1%, `loss/vq`
  diverged (to ~38 with the CNN encoder), with **both** frozen-SigLIP and trained-CNN
  encoders. This is what motivated going 100% literal first and logging it as a data point.
- EMA + restart (VILA-U lineage) **did** stabilize utilization in an earlier run, but is
  not in the paper, so it was stripped to establish the literal baseline. Kept as backlog #4.
