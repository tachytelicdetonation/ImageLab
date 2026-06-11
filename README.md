# ImageLab

A small lab for **single-GPU model experimentation**. You bring the model, data, loss,
and metrics — any family: diffusion, autoregressive, tokenizers, something else entirely.
The lab brings the part every research codebase reinvents badly: **runs as data** (every
run a self-describing folder + ledger row), a **kill funnel** that murders bad ideas at
the cheapest possible tier, and a CLI that keeps results comparable forever.

```bash
uv venv --python 3.12 && uv pip install -e .
python -m imagelab.data.download_imagenette --size 64    # blessed dataset (~100MB, Apache-2.0)

lab run examples/ddpm/config.yaml --tier smoke           # does it execute?      ~30s
lab run examples/ddpm/config.yaml --tier overfit         # can it learn AT ALL?  pass/fail
lab run examples/ddpm/config.yaml --tier fast            # better than baseline? 2k steps
lab run examples/ddpm/config.yaml                        # the real run
```

## Your whole project is one class

```python
# task.py
from imagelab import Task, StepOutput

class MyIdea(Task):
    name = "my_idea"
    overfit_gate = ("train/loss", "<=", 0.05)    # the --tier overfit kill criterion
    key_metrics = ["val/loss"]                   # what lab runs/board lead with

    def setup(self):                             # model/data/optimizers — all yours
        ...
    def step_fn(self, batch, step):              # the objective — yours
        return StepOutput(loss=loss, logs={"train/loss": loss.item()})
```

```yaml
# config.yaml
task: task.py:MyIdea        # resolved relative to this file — no packaging, no fork
train: {batch_size: 32, epochs: 10}
my_section: {anything: you want}   # the framework only interprets train/out/wandb
```

`lab run config.yaml --tier overfit` — that's it. `lab new task my_idea` scaffolds this
as a WORKING project (its toy objective passes smoke + overfit before you edit a line).
The trainer owns seeding, run dirs, cadenced logging/sampling/validation/checkpoints,
resume, crash bookkeeping, and the ledger row; `Task` declares everything scientific,
including optional hooks (`validate_config`, `apply_tier`, `add_args`) when your project
needs its own config schema, tier tweaks, or CLI flags.

## The idea: a kill funnel, then a ledger

Prototyping speed is how cheaply you can kill a bad idea. Each tier answers one
question, costs ~10× the previous, and has a kill criterion:

```
lab check ──► --tier smoke ──► --tier overfit ──► --tier fast ──► full
 wired right?   executes?       memorizes 1 sample?  beats baseline?  real numbers
 ~5s            ~30s            ~min (auto verdict)  ~20min           hours
```

The overfit verdict comes from YOUR task's `overfit_gate` declaration — if a change
can't memorize one image, it's broken (gradient path, loss wiring, data pipeline), and
that's learnable in minutes, not after a GPU-day.

Everything that survives lands in the **ledger** — every run is a folder:

```
runs/0610-1432_ddpm_lr-3e-4/
  config.yaml      # RESOLVED config (every default materialized)
  meta.json        # git sha, seed, device, params/component, steps/sec, status,
                   #   the task's key_metrics declaration (runs stay readable forever)
  metrics.jsonl    # scalar time series (offline source of truth; wandb is a mirror)
  metrics.json     # final held-out metrics
  samples/  checkpoints/
```

```bash
lab runs                                   # table of every run, newest first
lab compare 0610-1432 0609-1820            # config diff + metric diff (any two runs —
                                           #   even different model families)
lab board --metric val/noise_mse           # leaderboard
lab gallery                                # static HTML: grids + learning curves
lab report --format latex                  # publication-ready table from the ledger
lab sweep examples/ddpm/config.yaml --grid train.lr=1e-4,3e-4 --tier fast
```

`--set` deltas are recorded in the run's metadata and encoded into its name — the
experiment is stated as data, not remembered.

## The examples: two model families, ~40 lines apart

| | objective | sampler | gate |
|---|---|---|---|
| `examples/ddpm/` | `MSE(ε̂, ε)` over a 1000-step schedule (arXiv:2006.11239) | 50-step DDIM | `val/noise_mse` |
| `examples/rectified_flow/` | `MSE(v̂, ε − x₀)` on straight-line interpolants (arXiv:2209.03003) | 50-step Euler | `val/v_mse` |

Same tiny UNet, same data, same fixed-noise val protocol — diff the two `task.py` files
to see exactly what "a model family" costs here. Both write deterministic val metrics
(fixed `(t, ε)` pairs, constant seed) so numbers are comparable across runs *and seeds*.

## The resident research project: cvq

The lab grew out of faithful reimplementations of **Channel-wise VQ**
([arXiv:2605.26089](https://arxiv.org/abs/2605.26089)) + **EOSTok**
([arXiv:2605.00503](https://arxiv.org/abs/2605.00503)) text-to-image. The `cvq` package
is that research, rebuilt as an imagelab consumer — and the reference for every "how do
I do X in my own project?" question:

- own typed config schema + load-time guardrails → `cvq/config.py`, hooked in via
  `Task.validate_config`
- own component kinds (quantizer/encoder/decoder/ar_head) with registry + `paper=`
  citations → `cvq/models/`, `lab cite <config>`
- own contract checkers + scaffolds for those kinds → `cvq/lab/`, `lab check quantizer
  lfq`, `lab new quantizer my_idea`
- registered task names (`task: tokenizer|car|e2e`) via `[tool.imagelab] imports` in
  pyproject.toml — the installed-package alternative to `task.py:Class` paths

Faithfulness notes for cvq (IBQ per official SEED-Voken code, bit-for-bit refactor
verification, FID protocol) live in `RESULTS.md` and inline where they apply.

## Datasets

Any dataset is `<root>/images_<size>/` + `<root>/manifest.jsonl` with
`{"file", "name", "caption", "dataset"?, "split"?}` records — `imagelab.data` provides
the loader, but tasks are free to ignore it entirely.

- **Held-out split**: records with an explicit `split` keep it; the rest are assigned by
  a stable filename hash (`data.val_fraction`, default 10%). Per-item stable — adding
  images never reassigns old ones — so metrics are comparable across your whole ledger.
  `data.val_fraction: 0` restores train-set eval if you need parity with old runs.
- **Blessed benchmark**: ImageNette/ImageWoof 64px
  (`python -m imagelab.data.download_imagenette --which both`) — Apache-2.0, ~26k
  images, official val split preserved.
- Pokemon (`cvq.data.download_pokemon`) remains for fun/local use; the images are
  Nintendo IP — don't publish results/checkpoints built on it.

## Not goals

Single GPU (or Apple Silicon), small-to-mid resolutions, trustworthy comparisons over
raw scale. No distributed training, no production serving, no Lightning/Hydra/MLflow —
files + JSONL + a small CLI.

## Layout

```
imagelab/           # the framework — model-family agnostic
  task.py           #   THE seam: Task (+ overfit_gate/key_metrics declarations)
  trainer.py        #   one spine: tiers, --set, run dirs, ledger, gates
  loop.py           #   TrainLoop / RunLogger / StepOutput (+ GAN adapter)
  config.py         #   core schema (train/out/wandb); task sections are opaque
  registry.py       #   open (kind, name) registry with paper metadata
  lab/              #   ledger, tiers, criteria, CLI, gallery, probes, scaffolds
  data/             #   optional batteries: manifest datasets, stable split, ImageNette
examples/
  ddpm/             # blessed baseline (self-contained: task + UNet + config)
  rectified_flow/   # second family — diff against ddpm to see the seam
cvq/                # the resident research project (CVQ/EOSTok, forks A & B)
configs/            # cvq's blessed bases; experiments are --set deltas, not copies
tests/              # fast CPU seam tests (framework + cvq; no network, no datasets)
```

MIT licensed. PRs: see `CONTRIBUTING.md`.
