# Archived configs — preserved experiment provenance (NOT in the active pipeline)

We committed to **channel-wise IBQ** as the project's single quantizer (see `../../RESULTS.md`).
IBQ was the robust long-schedule winner; every other quantizer was either screened out or
collapsed over a longer run. The quantizer code for those alternatives was deleted from
`cvq/models/` (the old `vq_variants.py` and the plain-VQ + entropy paths in `quantizer.py`).

These YAMLs are kept **only as a record of what was run** — full results live in `RESULTS.md`.
They are out of the main `configs/` dir on purpose so the active config set is just the IBQ
recipe. To actually re-run any of them, check out the matching commit before the IBQ-only
cleanup (the quantizer classes they reference no longer exist on `main`).

## ⚠️ Two runnability tiers

**Tier 1 — NOT runnable on current `main`** (their `quantizer_type` code was deleted):

| file | `quantizer_type` | what it was | RESULTS.md |
|---|---|---|---|
| `cvq_pokemon_cnn_simvq.yaml` | `simvq` | SimVQ — frozen codebook + 1 linear layer (arXiv:2411.02038) | Run 4 |
| `cvq_pokemon_cnn_transvq.yaml` | `transvq` | TransVQ — frozen codebook + linear-attn transformer remap (2602.18896) | Run 5 |
| `cvq_pokemon_cnn_fvq.yaml` | `fvq` | FVQ / VQBridge — trainable codebook + compress→ViT→recover (2509.10140) | Run 6 |
| `cvq_pokemon_cnn_wasserstein.yaml` | `wasserstein` | plain VQ + Bures-Wasserstein feature/code matching (2506.15078) | Run 7 |
| `cvq_pokemon_cnn_ibqtransvq.yaml` | `ibqtransvq` | IBQ × TransVQ synthesis, frozen base (1k) | Run 9 |
| `cvq_pokemon_cnn_ibqtransvq_4k.yaml` | `ibqtransvq` | same, 4k — exposed the transient-peak collapse | Run 9 (4k) |
| `cvq_pokemon_cnn_ibqtransvq_learnable.yaml` | `ibqtransvq` (`base_learnable: true`) | learnable-base variant | Run 9b |
| `cvq_pokemon_cnn_entropy.yaml` | plain + `entropy_weight: 0.1` | MAGVIT-v2 entropy loss on plain VQ; the entropy path was removed | Run 3 |

**Tier 2 — plain-VQ baselines, `quantizer_type` absent/`plain`** (the plain `ChannelwiseVQ`
quantizer was removed). These still parse, but ⚠️ **the `quantizer_type`/`entropy_*` keys are now
ignored and the tokenizer always builds IBQ** — so running one of these on current `main` would
*not* reproduce the original plain-VQ result; it would just be IBQ with that config's hyperparams.

| file | what it was | RESULTS.md |
|---|---|---|
| `cvq_pokemon_cnn.yaml` | CNN literal plain-VQ baseline, K=16384 (the collapse reference) | Run 1 |
| `cvq_pokemon_cnn_cb4096.yaml` | plain-VQ baseline, K=4096 | Run 2 |
| `cvq_pokemon.yaml` | original SigLIP-encoder baseline (pre-CNN) | pre-log |
| `cvq_pokemon_cuda.yaml` | early CUDA baseline | pre-log |

## Active configs (in `../`, not here)
- `cvq_pokemon_cnn_ibq.yaml` — the committed recipe: channel-wise IBQ + AdamW.
- `cvq_pokemon_cnn_ibq_pion.yaml` — same IBQ quantizer, Pion optimizer (Run 10; lost to AdamW
  but still fully runnable since `cvq/muon.py` is kept — left active as the optimizer-ablation entry point).
