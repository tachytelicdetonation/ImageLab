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

## ⚠️ Two different utilization metrics — don't confuse them
- **`codebook/usage_batch`** (wandb line panel): codes used in **one batch** (16 imgs × 256 ch =
  4096 tokens). **Ceiling = 4096/16384 = 25%** — it physically cannot exceed that at K=16384.
- **`val/codebook_utilization_full`** (logged at val steps; `metrics.py:92`): unique codes over the
  **entire dataset** (~332,800 tokens). This is the standard "codebook utilization" papers report,
  and the one used to rank variants in the table below. A run can show usage_batch ~0.1 yet
  full-utilization ~0.8 — both correct, different scope.

## What to watch (the collapse signature)
| Metric (wandb key) | Healthy | Collapsing |
|---|---|---|
| `codebook/usage_batch` | rises (ceiling 25% at K=16384) | falls toward ~0 (few codes win) |
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
Sorted best→worst by utilization. PSNR↑ / LPIPS↓ / recon_L2↓ are full-channel reconstruction.

| Run | Method | Utilization | PSNR | LPIPS | recon_L2 | Status |
|---|--------|-------------|------|-------|----------|--------|
| 5 | **TransVQ** (frozen cb + transformer) | **94.2%** | 16.28 | 0.306 | 0.095 | ⭐ best util |
| 4 | SimVQ (frozen cb + linear) | 83.5% | 14.97 | 0.325 | 0.128 | ✅ |
| 8 | **IBQ channel-wise** (EOSTok synth) | 81.5% | **16.53** | **0.306** | **0.090** | ⭐ best recon |
| 7 | Wasserstein matching (γ=0.5) | 79.3% | 14.86 | 0.361 | 0.132 | ✅ |
| 3 | entropy loss (w=0.1, τ=1) | 56.6% | 14.46 | 0.368 | 0.145 | ✅ |
| 6 | FVQ / VQBridge (p=16, 140M) | 47.9% | 15.52 | 0.335 | 0.114 | ✅ (overfit?) |
| 2 | plain VQ, cb4096 | 0.93% | 14.01 | 0.370 | 0.161 | ❌ collapsed |
| 1 | plain VQ (literal baseline) | 0.15% | 14.09 | 0.367 | 0.158 | ❌ collapsed |

All runs: 1000 steps, K=16384 (except Run 2), CNN encoder, GAN off (disc_start 2000 > 1000).
**Headline:** every anti-collapse method beats the baseline by 50–600×; the two best are
**TransVQ** (utilization) and **IBQ** (reconstruction), nearly tied and clearly ahead.

---

### Run 1 — Literal baseline (`cvq-cnn-literal-cb16384-1k`)
- **Config:** `configs/cvq_pokemon_cnn.yaml`
- **Hypothesis:** the paper's "no bells and whistles" plain VQ collapses at our scale —
  16384 codes ≫ what ~1.3k images of 256 channels can populate. Establishes the
  reference everything else is measured against.
- **wandb:** [k3czdb5i](https://wandb.ai/tanmayd/cvq-pokemon/runs/k3czdb5i)
- **Result:** ❌ **collapsed.** `vq_loss` 0.166 → **33.95** (diverged ~200×); `usage_batch`
  0.140 → 0.002; perplexity ~23; `val/codebook_utilization_full` = **0.15%** (~24 of 16384).
  Recon stayed low (recon_l2 0.158, psnr 14.09, ssim 0.687, lpips 0.367) — decoder + ~20
  surviving codes memorize 1.3k images.
- **Takeaway:** the paper's "no bells and whistles" plain VQ does **not** reproduce at our
  scale; collapse within ~100 steps. Low recon masks a dead codebook — useless for a CAR
  tokenizer. This is the reference all variants are measured against.

### Run 2 — Smaller codebook (`cvq-cnn-literal-cb4096-1k`)
- **Config:** `configs/cvq_pokemon_cnn_cb4096.yaml`
- **Hypothesis:** shrinking the codebook 4× sizes capacity closer to the data.
- **wandb:** [pi0m17bi](https://wandb.ai/tanmayd/cvq-pokemon/runs/pi0m17bi)
- **Result:** ❌ **also collapsed.** `val/codebook_utilization_full` = **0.93%** (~38 of 4096)
  — better fraction than Run 1 but still ~38 live codes; `vq_loss` still diverged to ~32.5,
  perplexity ~30. recon/psnr/ssim essentially identical to Run 1.
- **Takeaway:** **collapse is NOT merely a capacity problem.** A 4× smaller codebook gives only
  marginally more live codes and does not stop `vq_loss` divergence. The fix must change the
  codebook *dynamics* (→ the structural variants below), not just its size.

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
- **Result:** ✅ util **56.6%** (PSNR 14.46, LPIPS 0.368, recon_L2 0.145) — huge jump from baseline's
  0.15%, but the **weakest of the working methods**; `vq_loss` still rose (w=0.1 entropy pressure
  alone didn't fully halt divergence).
- **Takeaway:** a pure *loss* helps but is weaker than a *mechanism* — the softmax-over-all /
  shared-codebook-function methods (IBQ, TransVQ) clearly dominate it.

#### Research backing (2026 arXiv search, 2026-05-29)
- **Entropy loss** — MAGVIT-v2 / "Language Model Beats Diffusion" ([arXiv:2310.05737](https://arxiv.org/html/2310.05737v2)); formula `E[H(q)] − H(E[q])`, "inspired by image VQGAN." ← what we implemented.
- **Beyond Stationarity** ([arXiv:2602.18896](https://arxiv.org/abs/2602.18896), Feb 2026) — root cause = encoder drift stranding unselected codes; fixes are *architectural* (NSVQ kernel rule, TransVQ codebook transform), near-100% utilization. Candidate if a loss alone underperforms.
- **Scalable Training, 100% Codebook Utilization** ([arXiv:2509.10140](https://arxiv.org/pdf/2509.10140), 2025).
- **Distributional/Wasserstein matching** ([arXiv:2506.15078](https://arxiv.org/pdf/2506.15078), 2025) — align feature & code distributions.
- **VQGAN-LC / one linear layer** ([arXiv:2411.02038](https://arxiv.org/html/2411.02038v1)) — frozen pretrained codebook + projection, ~99% at 100k codes.
- **LLM analogue** — MoE load-balancing aux loss (Switch Transformer): `N·Σ fᵢ·Pᵢ`, same objective as entropy term 2.

---

## Paper-variant experiments (Runs 4–7)
Four published collapse fixes, faithfully adapted to our channel-token setting, each over
the **16k baseline with no entropy loss** so each paper's mechanism is isolated. Implemented
in `cvq/models/vq_variants.py`, selected via `model.quantizer_type`. The literal
`ChannelwiseVQ` in `quantizer.py` is left untouched.

**Structural insight:** SimVQ / TransVQ / FVQ all share one idea — *the codebook you quantize
against is a function of a base codebook, so a gradient on one selected code flows through
shared weights into all codes, so dead codes can't form.* They differ only in the function:
linear (SimVQ), transformer (TransVQ), bridge-net (FVQ). Wasserstein is the odd one out: a
pure loss added to plain VQ. Run 2 showed collapse is a *dynamics* problem, which is exactly
what the three structural methods target.

### Run 4 — SimVQ (`cvq-cnn-simvq-cb16384-1k`)
- **Paper:** [arXiv:2411.02038](https://arxiv.org/html/2411.02038v1) ("…with One Linear Layer").
  **Note:** this ID is **SimVQ**, *not* VQGAN-LC (that's [2406.11837](https://arxiv.org/html/2406.11837v1)).
- **Mechanism (faithful):** frozen **random** codebook `C` (buffer) + one trainable linear
  layer `W` (256×256, no bias, **identity-init** so step 0 = plain frozen-codebook VQ).
  Quantize against `C·W`; the standard codebook-loss term trains `W` (not `C`).
- **Why this adaptation:** SimVQ's own ablation shows a frozen *random* codebook + `W` matches
  a pretrained (CLIP/k-means) init — and CLIP/ImageNet init does **not** transfer to our
  *channel*-tokens (channels aren't a shared semantic space like spatial patches). So the
  random-frozen form is both faithful and the right call here. Config: `cvq_pokemon_cnn_simvq.yaml`.
- **Cost:** +0.07M params. **Result:** ✅ util **83.5%**, PSNR 14.97, LPIPS 0.325, recon_L2 0.128.
  Big win for ~70KB of extra params — the cheapest fix by far.

### Run 5 — TransVQ (`cvq-cnn-transvq-cb16384-1k`)
- **Paper:** [arXiv:2602.18896](https://arxiv.org/abs/2602.18896) (Beyond Stationarity, Feb 2026).
  This paper proposes two methods (NSVQ + TransVQ); we implement **TransVQ** as the faithful
  representative (NSVQ's published equations are inconsistent with its released code, per the
  authors' repo; TransVQ is the cleaner, self-contained method — flagged for a possible later run).
- **Mechanism (faithful):** frozen codebook `C` + `C' = P_φ(C)` where `P_φ` is a 1-layer
  **linear-attention** transformer over codes-as-tokens (linear attention so K=16384 is O(K),
  not O(K²)). Standard VQ runs against `C'`. depth 1, model_dim 256, 1 head, MLP ratio 2.
- **Cost:** +1.7M params. Config: `cvq_pokemon_cnn_transvq.yaml`. **Result:** ⭐ **best utilization
  (94.2%)** and excellent recon (PSNR 16.28, LPIPS 0.306, recon_L2 0.095). The transformer remap is
  the most expressive shared codebook function — spreads usage *and* lifts fidelity together.

### Run 6 — FVQ / VQBridge (`cvq-cnn-fvq-cb16384-1k`)
- **Paper:** [arXiv:2509.10140](https://arxiv.org/pdf/2509.10140) (Scalable, 100% utilization).
- **Mechanism (faithful):** **trainable** codebook `C` remapped by **VQBridge** =
  compress(group-flatten) → 2 ViT blocks → recover, producing `Ĉ`. Quantize against `Ĉ`;
  recomputed every step. groups `p=16` (the paper's value for K=16384). No dead-code resets —
  utilization is meant to be structural. At inference `Ĉ` is baked and the bridge dropped.
- **Caveat:** the paper's `W_comp`/`W_exp` dims are internally inconsistent; we use the
  dimensionally-consistent group-flatten reading (compress `(K/p·D)→d'`). With p=16 this is
  **~140M params** (dwarfs the 29M encoder) — a real risk of overfitting on 1.3k images; the
  behavior is itself a data point. Config: `cvq_pokemon_cnn_fvq.yaml`. **Result:** ✅ but **weakest
  of the structural methods**: util 47.9% (PSNR 15.52, LPIPS 0.335, recon_L2 0.114). The 140M bridge
  on 1.3k images is over-parameterized (the flagged overfit risk) — likely needs more data or a
  larger `p`/smaller bridge to shine; its strength is ImageNet-scale.

### Run 7 — Wasserstein matching (`cvq-cnn-wasserstein-cb16384-1k`)
- **Paper:** [arXiv:2506.15078](https://arxiv.org/pdf/2506.15078) (Distributional Matching).
- **Mechanism (faithful):** plain channel-wise VQ + a **Bures-Wasserstein** term (closed-form
  W₂ between Gaussian fits of the encoder feature tokens and the codebook vectors; matrix sqrt
  via `eigh`), weight **γ=0.5**, **no stop-gradient** (flows to encoder + all codes). Not
  sliced-Wasserstein and not Sinkhorn — the paper uses the Gaussian/FID form.
- **Caveat:** Gaussian assumption + 256-D covariance from B·C≈4k tokens; ε-ridge for stability.
- **Cost:** +1.05M params (codebook). Config: `cvq_pokemon_cnn_wasserstein.yaml`. **Result:** ✅ util
  **79.3%**, PSNR 14.86, LPIPS 0.361, recon_L2 0.132. Strong utilization (the global distribution
  match revives codes well) but LPIPS lags the reparam methods — fidelity slightly behind.

### Run 8 — IBQ channel-wise — the EOSTok synthesis (`cvq-cnn-ibq-cb16384-1k`) ⭐ highest-potential bet
- **Papers:** [EOSTok arXiv:2605.00503](https://arxiv.org/abs/2605.00503) (ICML 2026) + its quantizer
  [IBQ arXiv:2412.02692](https://arxiv.org/abs/2412.02692). **2605.00503 = EOSTok** (End-to-End AR
  Image Gen w/ 1D Semantic Tokenizer); its discretizer is **IBQ off-the-shelf**.
- **Synthesis (the requested combination):** apply IBQ **per channel-token** (the CVQ axis)
  instead of per spatial/query token. Per the directive, IBQ **replaces** plain VQ where they
  conflict. IBQ = softmax over ALL K codes (cosine-ℓ2 logits, τ=1) + straight-through on the
  full distribution → every code gets gradient every step (anti-collapse, ~100% util in EOSTok)
  + double-quant loss (Eq.12, β=0.25) + entropy penalty (w=0.05, bumped from 0.01 for small data).
  K=16384 × dim 256 = IBQ's native default → no dimensional clash.
- **What we did NOT adopt (and why):** EOSTok's 1D ViT tokenizer + "drop the 2D prior" are about
  *raster-order spatial AR*, orthogonal to our channel-wise quantizer — kept our VQGAN CNN enc/dec.
  **APR loss + NTP** are EOSTok's standout anti-collapse glue but require the autoregressive model
  → **deferred to phase 2 (CAR)**. APR is the top phase-2 candidate: it's what stops the
  channel-AR loss from collapsing the codebook (EOSTok Table 1: usage 51.8% → 99.7% with APR).
- **Implementation:** `IBQChannelVQ` in `cvq/models/vq_variants.py`. Index-backprop STE
  `Ind = onehot + (p − sg[p])`, `z_q = Ind @ C` (no additive STE — grad to encoder flows through
  the softmax, IBQ's design). Config: `cvq_pokemon_cnn_ibq.yaml`. **Cost:** +4.2M (codebook).
- **Logits fix (rechecked vs source):** uses IBQ-canonical **unnormalized dot product** `zᵀC`
  (2412.02692 Eq.3-4), NOT EOSTok's cosine/τ. Verified: cosine ∈ [−1,1] over K=16384 with τ=1
  makes the softmax permanently ~uniform (maxprob ≈ 1/K), crippling both the index-backprop
  gradient and the entropy penalty — degenerate. Dot product is unbounded so the assignment
  sharpens as the encoder learns (smoke test: maxprob 0.002→0.285 as feature scale 1→4). Cosine
  remains available via `ibq_l2_norm=true` but needs a CLIP-style temp (τ≈0.07).
- **Hypothesis:** this is the most principled small-scale collapse fix of the set — softmax-over-all
  is a strictly stronger anti-collapse signal than the frozen-codebook reparam tricks, and it's
  validated at ~100% util in the source paper. Expect the best utilization + non-diverging vq_loss.
- **Result:** ✅✅✅ **best reconstruction of all 8 runs.** util **81.5%**, PSNR **16.53**, LPIPS 0.306,
  recon_L2 **0.090**; `vq_loss` stayed ~0.009 the entire run (baseline diverged to 33.95 — never
  diverged here). Coarse-to-fine intact (recon_L2 c8 0.160 → c256 0.113). Confirms the hypothesis:
  softmax-over-all-codes is the strongest mechanism; ties TransVQ on quality, slightly lower util.

### Run 9 — IBQ × TransVQ synthesis (`cvq-cnn-ibqtransvq-cb16384-1k`) 🟡 queued
- **Goal:** combine the two winners — TransVQ's best utilization (94.2%) + IBQ's best reconstruction
  (PSNR 16.53) — into one quantizer.
- **Design:** frozen base codebook `C` (buffer) → `C' = P_φ(C)` (TransVQ's 1-layer linear-attention
  transformer) → IBQ's softmax-over-all-codes + index-backprop STE + double-quant loss + entropy,
  all against `C'`. Dot-product logits. **Only IBQ's STE** (TransVQ's additive STE dropped — it would
  detach `C'` from the recon path). `C` frozen so *all* codebook learning routes through shared φ.
- **Why it should compose (not fight):** IBQ's dense softmax gives φ a **K-term gradient per step**
  (vs TransVQ's single-code STE — sparser, noisier), and freezing `C` makes "every code gets gradient"
  a structural invariant (proof: `∂L/∂φ = Σ_k (∂L/∂C'_k)(∂C'_k/∂φ)`, all K terms nonzero).
- **Implementation:** `IBQTransVQ` in `vq_variants.py` (4.85M trainable φ + 4.19M frozen buffer).
  Config: `cvq_pokemon_cnn_ibqtransvq.yaml`. Smoke test: avg_maxprob 0.67 at init (sharp — φ remap
  gives peaked logits, so the over-smoothing risk looks mild).
- **Prediction (from design agent):** beats both parents on utilization (→high-90s%), ties/slightly
  beats IBQ on recon (PSNR ≈16.5–16.7); ~30% risk triple spread-pressure over-smooths → watch
  `avg_maxprob` (should rise); fix = `ibq_entropy_weight` 0.05→0.02.
- **Result:** _running/tbd_

### Run 9b — IBQ × TransVQ with LEARNABLE base codebook (`cvq-cnn-ibqtransvq-learnable-cb16384-1k`) 🟡 queued
- **Question:** is freezing the base codebook actually better, or is a *learnable* base (more
  flexible) better given IBQ's dense softmax should keep it alive even with φ on top? Tests the
  "learnable is more flexible/better" intuition head-to-head with frozen Run 9.
- **Design:** identical to Run 9 but `base_learnable=true` → base `C` is an `nn.Parameter`
  (9.05M trainable: 4.85M φ + 4.19M base C) instead of a frozen buffer.
- **Why it might win:** more capacity/flexibility; IBQ's per-step dense gradient may prevent the
  learnable-`C` collapse that killed plain VQ. **Why it might lose:** reintroduces per-code freedom
  (collapse risk) + adds variance; φ may become partly redundant (design agent's "Alt A", which it
  predicted *strictly worse* for max-utilization). Config: `cvq_pokemon_cnn_ibqtransvq_learnable.yaml`.
- **Result:** _queued/tbd_

---

## Phase-2 (CAR) toolbox — evaluated, parked until the autoregressive model exists
Methods that are about *generation*, not the tokenizer — they don't move utilization now,
but are directly applicable once the channel-wise CAR lands. Recorded so we don't re-litigate.
- **APR loss** (EOSTok, [2605.00503](https://arxiv.org/abs/2605.00503)): decode the CAR's predicted
  channel-tokens back to pixels; the antidote to NTP-induced codebook collapse during *joint*
  tokenizer+CAR training (EOSTok Table 1: usage 51.8%→99.7%). **Top phase-2 candidate.**
- **BAR / Masked Bit Modeling** (Amazon FAR, [2602.09024](https://arxiv.org/abs/2602.09024)):
  replace the CAR's K-way softmax head (16384-way per channel-token — compute-heavy + hard to
  learn on 1.3k imgs) with a masked-bit head predicting `log₂K=14` bits via a 3-layer
  SwiGLU+adaLN MaskGIT-style model. Verdict: **NOT a tokenizer run** (its FSQ tokenizer abandons
  channel-wise VQ); valuable only as a scalable generation head. Caveat: our learned-codebook
  indices have no bit semantics, so the 14-bit decomposition is arbitrary (MBM models bit
  dependencies jointly, so still works). Sampling: progressive bit-unmask, e.g. allocation
  [5,5,4], guidance 3.0, temp 4.5→0.
- **FSQ as a non-CVQ baseline** (optional): lookup-free, structurally collapse-free, but per-scalar
  not per-channel-vector — would be a *different tokenizer*, not a CVQ variant.

## Evaluated and DEPRIORITIZED (don't re-litigate)
- **LeJEPA / SIGReg** ([2511.08544](https://arxiv.org/abs/2511.08544), Balestriero & LeCun, Nov 2025):
  an SSL method, NOT a tokenizer. Transferable nugget = SIGReg: regularize encoder embeddings to an
  isotropic Gaussian via random 1D projections + Epps-Pulley characteristic-function test (cheap,
  O(N), no matrix-sqrt; gradient to encoder). It's an *encoder-side* anti-collapse lever — a cheaper
  cousin of our Wasserstein variant (Run 7), which matched features→codes. **Deprioritized because:**
  (a) feature-side matching already *lost* to codebook-side mechanisms here (Wasserstein 79% util <
  IBQ/TransVQ 81–94%); (b) its isotropic-Gaussian target is optimal for linear-probing, and may
  *fight reconstruction* (we have a decoder; SSL doesn't). Only worth it as a tiny-λ auxiliary
  regularizer alongside a codebook-side winner if utilization ever stalls — never standalone.

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
