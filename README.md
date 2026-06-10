# ImageLab

A small lab for **casual image-model experimentation**: swap architecture components by
name, kill bad ideas in seconds-to-minutes instead of GPU-days, and get every run as a
self-describing folder with the same held-out metric suite — so results stay comparable
forever.

```bash
uv venv --python 3.12 && uv pip install -e .
python -m cvq.data.download_imagenette --size 64     # blessed dataset (~100MB, Apache-2.0)

lab run configs/tok_inet_64.yaml --tier smoke        # does it execute?        ~30s
lab run configs/tok_inet_64.yaml --tier overfit      # can it learn AT ALL?    ~min, pass/fail
lab run configs/tok_inet_64.yaml --tier fast         # better than baseline?   2k steps
lab run configs/tok_inet_64.yaml                     # the full, faithful run
```

The lab grew out of faithful reimplementations of **Channel-wise VQ**
([arXiv:2605.26089](https://arxiv.org/abs/2605.26089)) + **EOSTok**
([arXiv:2605.00503](https://arxiv.org/abs/2605.00503)) text-to-image, which remain the
reference stack (`RESULTS.md` is the lab notebook, `IDEAS.md` the fork roadmap).

## The idea: a kill funnel, then a ledger

Prototyping speed is how cheaply you can kill a bad idea. Each tier answers one
question, costs ~10× the previous, and has a kill criterion:

```
lab check ──► --tier smoke ──► --tier overfit ──► --tier fast ──► full
 wired right?   executes?       memorizes 1 image?   beats baseline?   paper numbers
 ~5s            ~30s            ~min (auto verdict)  ~20min            hours
```

Everything that survives lands in the **ledger** — every run is a folder:

```
runs/0610-1432_tok_inet_64_quant_type-lfq/
  config.yaml      # RESOLVED config (every default materialized)
  meta.json        # git sha, seed, device, params/component, steps/sec, status
  metrics.jsonl    # scalar time series (offline source of truth; wandb is a mirror)
  metrics.json     # final held-out metrics
  samples/  checkpoints/
```

```bash
lab runs                                   # table of every run, newest first
lab compare 0610-1432 0609-1820            # config diff + metric diff, side by side
lab board --metric val/rFID                # leaderboard
lab gallery                                # static HTML: grids + learning curves
lab report --format latex                  # publication-ready table from the ledger
lab sweep configs/tok_inet_64.yaml --grid train.lr=1e-4,3e-4 --tier fast
```

Sample grids are comparable **by eye** across runs: the fixed eval batch is the same
held-out images for every run regardless of seed, and generation prompts are pinned.

## Trying an architecture change

Components register under a `(kind, name)` pair (`quantizer`, `ar_head`, `encoder`,
`decoder`, `task`) and configs select them by name. The full loop:

```bash
lab new quantizer my_idea        # scaffolds cvq/models/my_idea.py — a WORKING
                                 # contract implementation, registered + importable
lab check quantizer my_idea      # shapes / truncate / lookup==z_q / STE gradient probes
lab run configs/tok_inet_64.yaml --set model.quant_type=my_idea --tier overfit
lab run configs/tok_inet_64.yaml --set model.quant_type=my_idea --tier fast
lab compare <your-run> <baseline-run>
```

`--set` deltas are recorded in the run's metadata and encoded into its name — the
experiment is stated as data, not remembered. The worked example is
**`cvq/models/lfq.py`** (channel-wise MAGVIT-v2 LFQ, arXiv:2310.05737): a real method in
~100 lines that shows the whole contract, registered as `quant_type: lfq`.

Invalid combinations fail **at config load** — cross-field rules live in
`cvq/config.py::validate` ("EXPERIMENT GUARDRAILS"). When a bad combo wastes a GPU-day,
encode the lesson there. What counts as "the result" of a run (ledger columns,
leaderboard sort, overfit pass thresholds) lives in `cvq/lab/criteria.py` — tune it to
your taste.

## The reference stack

```
image ─► encoder ─► channel-wise quantizer ─► decoder ─► reconstruction   (tokenizer)
text  ─► LLM backbone ─► AR head ─► channel-tokens ─► decode              (CAR, phase 2)
```

An image becomes `C` channel-tokens ordered coarse-to-fine (nested channel dropout);
generation is next-channel prediction with a Qwen3 backbone.

| | quantizer | AR head | notes |
|---|---|---|---|
| **Fork A** (EOSTok-faithful) | `ibq` (arXiv:2412.02692) | `softmax` | supports the APR soft-decode loss |
| **Fork B** (BAR-style) | `fsq` (arXiv:2309.15505) | `mbm` (masked-bit) | parameter-free codebook, bit indices |
| example | `lfq` (arXiv:2310.05737) | — | the "add a quantizer" tutorial |

`lab cite <config>` prints the papers behind any config. Tasks: `tokenizer` (recon+GAN),
`car` (AR on a frozen tokenizer), `e2e` (joint EOSTok objective).

```bash
python -m cvq.evaluate --ckpt runs/<id>/checkpoints/best.pt    # full metric suite, any ckpt
python -m cvq.generate --car_ckpt ... --prompts "a photo of a church" --cfg 3
python -m cvq.reconstruct --ckpt ...
```

REPL/notebook poking without config assembly:

```python
from cvq.dev import quick
tok = quick.tokenizer(quant="lfq")        # tiny built tokenizer, cpu, seeded
x = quick.batch(8)                        # real held-out images in [-1,1]
quick.show(tok(x)["recon"])               # grid -> opens on macOS
```

`cvq.dev.tiny_qwen()` fabricates a 2-layer Qwen3 (~1MB) so the full text→image stack
prototypes on CPU in seconds.

## Datasets

Any dataset is `<root>/images_<size>/` + `<root>/manifest.jsonl` with
`{"file", "name", "caption", "dataset"?, "split"?}` records.

- **Held-out split**: records with an explicit `split` keep it; the rest are assigned by
  a stable filename hash (`data.val_fraction`, default 10%). Per-item stable — adding
  images never reassigns old ones — so metrics are comparable across your whole ledger.
  All reported metrics are held-out; `data.val_fraction: 0` restores train-set eval if
  you need parity with old runs.
- **Blessed benchmark**: ImageNette/ImageWoof 64px (`python -m cvq.data.download_imagenette
  --which both`) — Apache-2.0, ~26k images, official val split preserved, and the
  `dataset` tag gives an easy-vs-hard axis that eval splits automatically.
- Pokemon (`cvq.data.download_pokemon`) remains for fun/local use; note the images are
  Nintendo IP — don't publish results/checkpoints built on it.

## Faithfulness notes

- IBQ follows the official SEED-Voken code; deviations are commented where they live.
  The tokenizer GAN recipe keeps taming-transformers' adaptive weight and the paper's
  channel-count-aware λ_gan(c_keep). Historical quirks are preserved deliberately
  (tokenizer trains unclipped; frozen-tok CAR uses β=(0.9,0.95) + cosine).
- The move to this structure was verified **bit-for-bit** against the original trainers
  (same seed → identical init, losses, sampled tokens).
- **FID protocol**: rFID via `torchmetrics` (InceptionV3, 299px bilinear resize) over the
  val split. FID is implementation-sensitive (clean-fid, CVPR'22) — compare numbers only
  within this protocol.

## Not goals

Single GPU (or Apple Silicon), 64–256px, faithfulness over speed. No distributed
training, no production serving, no Lightning/Hydra/MLflow — files + JSONL + a small
CLI. The package is `cvq`; the project is `imagelab`.

## Layout

```
cvq/
  registry.py     # @register("quantizer", "lfq") — the extension seam
  config.py       # typed schema + defaults + EXPERIMENT GUARDRAILS
  factory.py      # composition root: config names -> built models (+ the contracts)
  trainer.py      # ONE training spine: tiers, --set, run dirs, ledger, overfit gate
  tasks/          # recipes: tokenizer | car | e2e
  models/         # components: encoder_cnn, decoder, quantizer(ibq), fsq, lfq, heads, car
  lab/            # the lab layer: rundir/ledger, tiers, criteria, cli, checkers,
                  #   scaffold, gallery
  eval/           # metric suite + per-source GroupedEvaluator
  data/           # manifest datasets + download_imagenette / download_pokemon
  dev.py          # `quick` REPL helpers + tiny_qwen fixture
configs/          # blessed bases; experiments are --set deltas, not copies
tests/            # fast CPU seam tests (no network, no datasets)
```

MIT licensed. PRs: see `CONTRIBUTING.md` — new components need `lab check` + a
fast-tier comparison vs. baseline.
