# ImageLab

A small lab for **casual image-model experimentation**: swap architecture components by
editing a YAML, train on a basic dataset in minutes-to-hours, and get the same metric
suite out of every run so results are comparable.

The lab grew out of a faithful reimplementation of **Channel-wise VQ**
([CVQ, arXiv:2605.26089](https://arxiv.org/abs/2605.26089)) +
**EOSTok** ([arXiv:2605.00503](https://arxiv.org/abs/2605.00503)) text-to-image, and that
remains the reference stack (see `RESULTS.md` for the experimental log and `IDEAS.md` for
the fork roadmap). But every layer is now pluggable.

## The stack

```
image ─► encoder ─► channel-wise quantizer ─► decoder ─► reconstruction   (tokenizer)
text  ─► LLM backbone ─► AR head ─► channel-tokens ─► decode              (CAR, phase 2)
```

An image becomes `C` channel-tokens ordered coarse-to-fine (nested channel dropout);
generation is next-channel prediction with a Qwen backbone. Two reference forks:

| | quantizer | AR head | notes |
|---|---|---|---|
| **Fork A** (EOSTok-faithful) | `ibq` (index-backprop VQ) | `softmax` | supports the APR soft-decode loss |
| **Fork B** (BAR-style) | `fsq` (binary FSQ) | `mbm` (masked-bit) | parameter-free codebook, bit-structured indices |

## Running things

```bash
uv venv --python 3.12 && uv pip install -e .
python -m cvq.data.download_pokemon --size 256          # ~1.3k images + manifest.jsonl

# train (task comes from the config's `task:` key)
python -m cvq.trainer --config configs/cvq_pokemon_cnn_ibq.yaml      # tokenizer only
python -m cvq.trainer --config configs/car_e2e_inet_64.yaml          # joint E2E (Fork B)
python -m cvq.trainer --config configs/car_pokemon_qwen.yaml \
    --tokenizer_ckpt checkpoints/best.pt                             # CAR on frozen tok

# metrics for any checkpoint (rFID/PSNR/SSIM/LPIPS/codebook util/per-c_keep recon,
# split per dataset source, + generation grids if the ckpt has a CAR):
python -m cvq.evaluate --ckpt checkpoints/latest.pt
python -m cvq.evaluate --ckpt checkpoints_e2e/e2e_latest.pt --generate

# inference / visualization
python -m cvq.generate --car_ckpt checkpoints_car/car_latest.pt --prompts "pikachu" --cfg 3
python -m cvq.reconstruct --ckpt checkpoints/latest.pt --n 8

# tests (fast, CPU, no downloads)
uv run pytest tests/ -q
```

Legacy entry points (`python -m cvq.train`, `cvq.train_car`, `cvq.train_e2e`) still work —
they are shims onto the trainer, so existing launch scripts don't break. Training is
resumable: `--resume checkpoints/cvq_step010000.pt` (the `*_step*.pt` files carry optimizer
state; `latest.pt`/`best.pt` are model-only).

## Trying an architecture change

Components register under a `(kind, name)` pair and are selected by name in YAML.
Kinds: `quantizer`, `ar_head`, `encoder`, `decoder`, `task`.

```python
# cvq/models/my_quantizer.py
from cvq.registry import register

@register("quantizer", "lfq")
class ChannelLFQ(nn.Module):
    def __init__(self, token_dim, codebook_size, temperature=0.1, **_ignore): ...
    def forward(self, z):          # (B,C,h,w) -> (z_q, idxs (B,C), aux_loss, stats)
    def truncate(self, z_q, c_keep): ...   # nested-dropout mask
    def lookup(self, idxs): ...            # indices -> feature map (for generation)
```

Then: add `from . import my_quantizer` to `cvq/models/__init__.py`, set
`quant_type: lfq` (+ `quantizer_kwargs: {temperature: 0.2}`) in a config, train. No other
file changes. The full constructor conventions per kind are documented in `cvq/factory.py`;
the AR-head contract in `cvq/models/heads.py`; new training recipes subclass
`cvq/tasks/base.py`.

Invalid combinations fail **at config load**, not at step 4000 — cross-field rules live in
`cvq/config.py::validate` (e.g. the `mbm` head requires `fsq`'s bit-structured indices;
EOSTok's APR needs a real codebook). When an experiment wastes a GPU-day on a bad combo,
encode the lesson there.

## Layout

```
cvq/
  registry.py        # @register("quantizer", "lfq") — the extension seam
  config.py          # typed config schema + defaults + cross-field validation
  factory.py         # composition root: config names -> built models
  trainer.py         # ONE training spine (seed/data/optim/resume/loop/ckpt)
  tasks/             # recipes: tokenizer | car | e2e  (register "task" kinds)
  models/            # components: encoder_cnn, decoder, quantizer (ibq), fsq,
                     #   heads (softmax|mbm), car, mbm_head, discriminator, dino_align
  losses/            # VQGAN/CVQ loss stack (LPIPS + PatchGAN + λ_gan(c_keep))
  eval/              # metrics.py (full-suite validate) + evaluator.py (per-source splits)
  data/              # ManifestImageDataset (+ captioned wrapper), download_pokemon
  evaluate.py        # `python -m cvq.evaluate` — metrics CLI for any checkpoint
  generate.py        # text -> image inference
  reconstruct.py     # recon grids, codebook utilization, coarse-to-fine progression
  conditioning.py    # caption-dropout + CFG uncond invariant
  nested_dropout.py  # c_keep policies + channel-weight schedules
configs/             # one YAML per experiment (task: key selects the recipe)
tests/               # fast CPU regression tests for the seams
```

## Datasets

Any dataset is `<root>/images_<size>/` + `<root>/manifest.jsonl` with
`{"file", "name", "caption", "dataset"?}` records. The optional `dataset` tag splits eval
metrics/grids per source (e.g. ImageNette = easy vs ImageWoof = hard from one run).
Built-ins: Pokemon (~1.3k official artworks via `cvq.data.download_pokemon`), and the
ImageNette+Woof 64px set used by `configs/car_e2e_inet_64.yaml` (built on the box; see
`.setup_box.sh`).

## Faithfulness notes

- IBQ follows the official SEED-Voken code (commitment/codebook weight ordering, sharpened
  entropy temperature) — see the docstring in `cvq/models/quantizer.py`.
- The tokenizer GAN recipe keeps taming-transformers' adaptive weight and the paper's
  channel-count-aware λ_gan(c_keep); AdamW β=(0.5, 0.9).
- Historical quirks are preserved deliberately and commented where they live: the tokenizer
  task trains without grad clipping (train.py never clipped); the frozen-tokenizer CAR task
  uses β=(0.9, 0.95) + cosine LR.
- The refactor to this structure was verified bit-for-bit against the old trainers: same
  seed → identical model init, identical loss values, identical sampled tokens.
