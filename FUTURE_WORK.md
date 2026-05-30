# Future work / deferred experiments

Things we want to try but deferred for a working baseline first.

## Backbone: Qwen3.5-0.8B (deferred — speed)
- User wants the CAR backbone to be `Qwen/Qwen3.5-0.8B` (hybrid linear-attention multimodal LM).
- **Blocker:** its Gated-DeltaNet linear attention needs the `flash-linear-attention` + `causal-conv1d`
  CUDA fast-path kernels. On the Blackwell RTX PRO 5000 (sm_120 / CUDA 13) the fast path is NOT
  available (`fla` installed but transformers still reports "fast path not available"; `causal-conv1d`
  has no sm_120 wheel and is a risky source build). Without it a single E2E step is **~23 min**
  (benchmarked 1413 s) vs ~4 s with plain Qwen3-0.6B.
- **To retry:** get `causal-conv1d` (and the fla fast path) working on sm_120, OR run on an
  Ampere/Hopper box where prebuilt kernels exist. Then set `qwen_name: Qwen/Qwen3.5-0.8B` in
  `configs/car_e2e_pokemon.yaml`. `cvq/models/car.py` already supports it (bf16, `get_decoder()`,
  text_config hidden) — verified loading + one bf16 forward/backward on GPU.

## VFM alignment: DINOv3 ViT-L/16 (deferred — grid fix)
- User wants `facebook/dinov3-vitl16-pretrain-lvd1689m` for the EOSTok semantic-alignment target.
- HF access is unlocked (gated=manual, license accepted, HF_TOKEN on box works).
- **Blocker:** DINOv3 patch-16 at 224px yields 14x14=196 patch tokens + CLS + 4 register tokens
  = 201 total. `cvq/models/dino_align.py` takes the last `grid*grid` tokens and the interp branch
  assumed a clean square — crashes (`reshape 201 -> 14x14`). Needs: take exactly the last
  `(dino_res/patch)^2` patch tokens (196), reshape to 14x14, interpolate to our 16x16. (DINOv2
  patch-14 @ 224 lands on 16x16 = 256 exactly, which is why DINOv2 works as-is.)
- **To retry:** fix the patch-count math in `DINOAlign._dino_features`/`forward`, then set
  `dino_name: facebook/dinov3-vitl16-pretrain-lvd1689m`. `configs/car_e2e_pokemon_dinov3.yaml`
  exists for this.

## Current baseline (what's actually running)
- `configs/car_e2e_pokemon.yaml`: Qwen3-0.6B (normal/instruct) + DINOv2-large + channel-wise IBQ
  tokenizer (cosine, K=16384, tau=1), pure-EOSTok joint E2E (ar_start_step=0),
  lambda_ntp=0.1 / lambda_apr=1.0 / lambda_sem=1.0.
