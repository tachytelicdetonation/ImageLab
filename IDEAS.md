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

---

## Suggested execution order
1. **A2 + A1 sweep** on the current 64px checkpoint (verify caption dropout first; add + short
   retrain if missing). Cheapest path to "do the gens actually look like Pokémon."
2. **A3** augmentation → fold into the next training run.
3. **B2** codebook-utilization A/B (cheap, answers an open config question).
4. **C1** dataset expansion (biggest real lever for generation).
5. **B1 FlowMo decoder** if recon fidelity is the visible bottleneck after the above.
6. Park **C2 / D1 / D2** as labeled ablations.

## References
- FlowMo: Variational Flow Matching for Image Tokenization — arXiv:2503.11056
- GigaTok: Scaling Visual Tokenizers to 3B — arXiv:2504.08736
- MAR: Autoregressive Image Generation without Vector Quantization — arXiv:2406.11838
- CFG for AR visual gen: LlamaGen / VAR / MUSE / MaskGIT (standard practice, scale 1.5–7.5, ~10% cond-drop)
