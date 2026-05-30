# High-Fidelity Image-Gen Ideas — ranked for THIS project

Context: channel-wise IBQ tokenizer (K=16384, 16×16 grid, 256 channel-tokens) + channel-AR
(CAR) text→image on Qwen3-0.6B, EOSTok end-to-end joint training (NTP + APR + DINOv2 align),
~1.3k Pokémon images, single 48GB GPU. Standing goal: anything that makes high-fidelity gen
possible. Ranked by (impact × feasibility-here) / cost. Compiled from 2024–2026 literature +
the project's actual bottlenecks.

The honest framing: with ~1.3k images, **reconstruction fidelity is achievable; text→image
generation is data-starved**. So the ideas split into (A) cheap wins that raise what we already
have, (B) tokenizer/decoder fidelity, (C) attacking the data bottleneck, (D) bigger bets.

---

## A. Cheap, high-ROI wins (do these first — hours, not days)

### A1. Classifier-free guidance at sampling  ★ highest ROI
- **Finding:** CFG is reported as *the single largest quality lever* for conditional AR image
  gen (LlamaGen, VAR, MUSE, MaskGIT). `logits = uncond + s·(cond − uncond)`, scale **1.5–7.5**.
- **Our state:** `cfg_scale: 1.0` in both configs = no guidance. We're leaving the biggest free
  win on the table. `car.generate()` already has the uncond path wired.
- **PREREQUISITE TO VERIFY:** CFG only works if the unconditional distribution was *trained* —
  i.e. ~10% caption dropout during training (replace text with empty/null ids). **Check
  `train_e2e.py` for caption dropout; if absent, add `p_uncond≈0.1` (drop text_ids→empty per
  sample).** Without it, the uncond logits are garbage and CFG can hurt.
- **Action:** (1) add caption dropout if missing → retrain; (2) sweep `cfg_scale ∈ {1.5, 3, 5, 7.5}`
  at sampling on a checkpoint. Pure inference sweep, no retrain needed for the sweep itself.

### A2. Sampling hyperparameters
- **CORRECTION (EOSTok-faithful):** EOSTok (2605.00503) samples with *"temperature of 1.0
  without top-k or top-p"* and reports its headline FID **1.48 *without* guidance**. The
  top_k~100–1000 advice is from VQGAN/LlamaGen — a *different* tokenizer family. EOSTok can
  skip top-k because its **APR loss shapes the token distribution to be generation-aware**
  during joint training, so the softmax has no noisy 16k tail to truncate. Top-k here is
  redundant-with / unfaithful-to EOSTok's core contribution.
- **Faithful protocol:** primary = `top_k=0`, `temperature=1.0`, `cfg_scale=1.0` (no guidance).
  CFG is a SECONDARY sweep only (paper uses it for small models as `l_g=l_u+s(l_c−l_u)`).
- **Caption dropout 0.1 IS faithful** — it's EOSTok Table 9 ("Class dropout ratio 0.1"), kept.

### A3. Heavier data augmentation
- **Our state:** `hflip: true` only. On 1.3k images this is the cheapest effective-data multiplier.
- **Action:** add random scale/crop, mild rotation (±15°), color jitter, maybe CutMix/MixUp on
  the pixel side. Pokémon are centered on white → be careful crops keep the subject. Could 5–10×
  effective data and directly fights overfitting in both tokenizer and CAR.

---

## B. Tokenizer / decoder fidelity (caps everything downstream)

### B1. FlowMo-style flow-matching decoder  (arXiv:2503.11056)
- **Finding:** Replace the deterministic VQGAN (L2+LPIPS+GAN) decoder with a small *flow-matching
  generative* decoder in latent space → SOTA reconstruction FID at 256² **without GAN**, recovering
  high-frequency detail that L2/LPIPS/GAN blur. Two-stage: train w/ flow decoder, then distill to
  1-step for fast inference.
- **Fit:** Keep our channel-wise IBQ encoder + codebook; swap only the decoder. The recon ceiling
  is what bounds CAR sample quality, so this raises the whole stack. Medium implementation cost.
- **Caveat:** stochastic decoder + EOSTok's APR term (which decodes the AR prediction to pixels)
  need to be reconciled — APR assumes a deterministic D(·). Doable (use the distilled 1-step D for APR).

### B2. Fix/measure codebook utilization
- **Our state:** cosine logits at `tau=1`, K=16384 → documented near-flat softmax degeneracy;
  observed batch usage ~0.08–0.10, perplexity ~1000–1500 (out of 16384) → **most of the codebook
  is idle**, which directly limits reconstruction richness.
- **Action:** A/B the dot-product (non-cosine) IBQ path, or lower `tau` (e.g. 0.1) to sharpen the
  softmax, or shrink K to 4096 (EOSTok's actual value) so usage density rises. Measure recon vs
  usage. This is a fidelity lever AND a fair-ablation question already flagged in the config.

### B3. GigaTok insight (arXiv:2504.08736) — tokenizer scaling stabilizes downstream AR
- Larger/decoder-heavier tokenizers give *more* stable, better downstream AR. Suggests our
  from-scratch tokenizer may be under-capacity for the AR to latch onto. Cheap version: widen
  `enc_ch`/`decoder_ch` or add res-blocks; watch whether NTP/token-acc improves.

---

## C. Attack the data bottleneck (the real limiter for *generation*)

### C1. Expand the dataset beyond official artwork
- 1.3k → 10k+ by adding game sprites (multiple gens), shiny variants, back-sprites, Pokémon-style
  fan-art / 3D-render datasets on HF. More views of the same concepts = the difference between
  "abstract color fields" and recognizable creatures. **Likely the single biggest real-world lever.**

### C2. Pretrained-tokenizer transfer
- Instead of training the tokenizer from scratch on 1.3k images, init the encoder/codebook from a
  pretrained tokenizer (ImageNet VQGAN, or a released IBQ/LlamaGen tokenizer) and fine-tune. A
  latent space learned on millions of images transfers; fidelity jumps. Tension with the "from
  scratch, closest-to-paper" preference — frame as a separate transfer-learning ablation.

---

## D. Bigger bets (rearchitecture; park unless A–C plateau)

### D1. MAR — continuous tokens + diffusion loss (arXiv:2406.11838, NeurIPS'24)
- Drops VQ entirely; per-token continuous distribution via a small diffusion-MLP head. Eliminates
  codebook collapse AND the discrete recon ceiling → strong FID. Cost: per-token diffusion sampling,
  and it abandons the discrete-channel-VQ thesis of this project. A clean "what if no VQ" comparison.

### D2. VAR-style next-scale prediction (coarse→fine)
- The dominant AR-image win of 2024–25: predict token *scales* (1×1→2×2→…→16×16) instead of a flat
  sequence. Maps surprisingly well onto our *nested channel dropout* coarse-to-fine ordering — the
  channel order already is a coarse→fine curriculum. Could reframe CAR as next-scale over channel
  groups. Research-grade rearchitecture.

### D3. CVQ × FSQ × BAR — channel-FSQ tokenizer + masked-bit CAR head  ★ the real generation lever
*(arXiv:2602.09024 BAR; LFQ/MAGVIT-v2; FSQ 2309.15505; BSQ/Infinity. Supersedes the old "raw-LFQ
big-codebook" framing in `[[bar-lfq-more-codes]]`.)*

**Why this exists.** On 1.3k images reconstruction is fine but *generation* is data-starved, and the
root cause is mechanical: the CAR head predicts **one 1-of-16384 index per channel**. With ~1.3k
images that categorical is unlearnable (`token_acc≈0.012`, near-random). BAR's finding is that you
don't fix this with *more tokens per channel* (RVQ lengthens the sequence) — you **factorize the
prediction target**. Keep one token per channel, make that token a **bit code**, and predict its
bits. The per-step problem drops from 1-of-16384 to k≈14 cheap low-cardinality choices, and the head
goes from O(K) to **O(log₂K)**. This is the only idea here that structurally attacks the gap rather
than nibbling at it.

**What changes vs. the faithful EOSTok×CVQ baseline (Fork A):**

| Component | Fork A (faithful) | Fork B (this) |
|---|---|---|
| Quantizer | channel-vector cosine **IBQ**, K=16384 codebook | channel-wise **FSQ/BSQ**, parameter-free, k bits |
| Quant losses | commitment + codebook + entropy (`L_Q`,`L_E`) | **none** — FSQ needs no aux losses (recon only) |
| CAR head | `image_head: Linear(hidden, K)` + K-way softmax | **MBM**: bit embedding + masked-bit-modeling head |
| AR loss | (flat) NTP cross-entropy over K | **bit-wise CE** with random bit masking |
| Codebook collapse | the whole reason IBQ exists | gone by construction (FSQ ~100% usage) |

**Preserved (still CVQ, still EOSTok-shaped):** channel-wise tokenization (the token is a channel),
coarse-to-fine **nested dropout**, next-**channel** AR over the Qwen backbone, text conditioning + CFG
dropout, the **APR** loss, optional DINOv2 alignment. Only the *discretizer* and the *prediction head*
change.

**Architecture sketch.**
1. **Channel-wise FSQ tokenizer** (`cvq/models/fsq.py`, analog of `quantizer.py`):
   - Encoder → `z ∈ (B, C=256, h, w)`; per channel the content is its `h·w=256`-dim spatial map.
   - Shared (weight-tied across channels) down-projection `256 → d_fsq` (`d_fsq≈5–14`).
   - **FSQ:** bound each of the `d_fsq` scalars (`tanh`+`round`-STE) to `L` levels → code = `∏ levels`.
     Default to match K=16384: `levels=[8,8,8,8,4]` (5 dims) **or** pure-binary **BSQ** `[2]×14`
     (sign-quantize, k=14 independent bits — the MBM-friendliest start).
   - Up-projection `d_fsq → 256`, reshape to `(h,w)`, feed the **same** VQGAN decoder.
   - Nested dropout still zeroes the tail channels of `z_q` before decode → coarse-to-fine ordering
     unchanged; FSQ only changes *how each channel is discretized*.
   - The index ↔ bit string is now **meaningful** (FSQ digit expansion) — exactly the structure IBQ
     lacks and what makes bit-prediction pay off.
2. **MBM CAR head** (`cvq/models/mbm_head.py`, used by a head-gated `car.py`):
   - **Input embedding:** embed the channel's bit code as a sum of per-bit/per-digit embeddings →
     `hidden`, replacing `image_embed: Embedding(K, hidden)`.
   - **Output (BAR Eq.6–7):** the Qwen hidden state `h_i` (context `[text][BOI][ch_1..ch_{i-1}]`)
     conditions a small masked-bit head that predicts channel `i`'s k bits by **iterative unmasking**
     (mask a subset, predict, reveal, repeat). Train with random mask ratio + **bit-wise CE** over
     masked positions; sample with a schedule (`[4,4,4,4]` for 14–16 bits). MBM beats a plain
     parallel-Bernoulli bit head — the masking is a regularizer (BAR §3.4).
3. **Loss mapping** (`train_e2e.py`):
   - `L_NTP → L_bit` (bit-wise CE, O(k) not O(K)).
   - `L_APR` **kept** and gets *cleaner*: soft-decode predicted bit-probabilities → expected FSQ value
     → decoder → L2+LPIPS. Still the key low-data lever.
   - `L_VQVAE` shrinks to **recon + GAN only** (no commitment/codebook/entropy).
   - `L_implicit` (DINOv2) unchanged, optional.

**Files:** new `cvq/models/fsq.py`, `cvq/models/mbm_head.py`, `configs/car_e2e_pokemon_64_bar.yaml`;
gate `quantizer_type: fsq` in `tokenizer_factory.py` and `head_type: mbm` in `car.py`; add the bit-CE
+ soft-bit-APR path in `train_e2e.py` (drop codebook/entropy logs when FSQ).

**Prototype plan & success criteria.**
1. *Tokenizer alone, recon-only:* channel-FSQ should match IBQ recon (`recon ≲ 0.04`, `lpips ≲ 0.15`)
   with ~100% usage by construction. If recon regresses, raise `d_fsq` / level budget. **Go/no-go.**
2. *+ MBM head:* log **bit-accuracy** and channel **exact-match**; compare gen to Fork A at matched
   steps. Success = generations become recognizable, or exact-match ≫ Fork A's `token_acc`.
3. *Scale test:* head is O(log K), so push `levels` up (bigger effective K) and check recon improves
   **without** the AR getting harder — the property single-index IBQ can't give us.

**Risks / open questions.** (a) Channel-wise FSQ is non-standard (FSQ is usually on spatial tokens) —
the per-channel down/up projection is the novel, unproven part; verify it keeps coarse-to-fine channel
semantics under nested dropout. (b) The `256→d_fsq` bottleneck may cap recon; tune `d_fsq`. (c)
Inference is 256 channel-steps × a short inner unmask loop — confirm it stays cheap. (d) This is
**explicitly a new method, not EOSTok** — keep Fork A as the faithful baseline and run B as a labeled
ablation (working name: **Bit-CAR / channel-FSQ-AR**). Tension with `[[cvq-fidelity-preference]]` is
intentional and bounded: B is the "make generation work" bet, A stays the paper-faithful control.

---

## Suggested execution order
1. **A2 + A1 sweep** on the current 64px checkpoint (verify caption dropout first; add + short
   retrain if missing). Cheapest path to "do the gens actually look like Pokémon."
2. **A3** augmentation → fold into the next training run.
3. **B2** codebook-utilization A/B (cheap, answers an open config question).
4. **C1** dataset expansion (biggest real lever for generation).
5. **B1 FlowMo decoder** if recon fidelity is the visible bottleneck after the above.
6. Park **C2 / D1 / D2** as labeled ablations.
7. **D3 (Bit-CAR / channel-FSQ + MBM)** — the structural fix for generation; prototype after A–C if the
   gap persists, or sooner if generation (not recon) is the priority. Tokenizer recon is the go/no-go.

## References
- FlowMo: Variational Flow Matching for Image Tokenization — arXiv:2503.11056
- GigaTok: Scaling Visual Tokenizers to 3B — arXiv:2504.08736
- MAR: Autoregressive Image Generation without Vector Quantization — arXiv:2406.11838
- CFG for AR visual gen: LlamaGen / VAR / MUSE / MaskGIT (standard practice, scale 1.5–7.5, ~10% cond-drop)
- BAR: Autoregressive Image Generation with Masked Bit Modeling — arXiv:2602.09024 (D3 head)
- FSQ: Finite Scalar Quantization (VQ-VAE made simple) — arXiv:2309.15505 (D3 quantizer)
- LFQ / MAGVIT-v2 — arXiv:2310.05737; BSQ / Infinity — bit-token tokenizers (D3 lineage)
- RQ-VAE / RQ-Transformer: residual quant, multi-token-per-unit — arXiv:2203.01941 (D3 alternative)
