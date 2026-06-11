# ImageLab

[![ci](https://github.com/tachytelicdetonation/ImageLab/actions/workflows/ci.yml/badge.svg)](https://github.com/tachytelicdetonation/ImageLab/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

A small lab for **single-GPU model experimentation**. You bring the model, data, loss,
and metrics — any family: diffusion, autoregressive, tokenizers, something else entirely.
The lab brings the part every research codebase reinvents badly: **runs as data** (every
run a self-describing folder + ledger row), a **kill funnel** that murders bad ideas at
the cheapest possible tier, and a CLI that keeps results comparable forever.

```bash
git clone https://github.com/tachytelicdetonation/ImageLab && cd ImageLab
uv sync                                                          # or: pip install -e .
uv run python -m imagelab.data.download_imagenette --size 64    # blessed dataset (~100MB, Apache-2.0)

uv run lab run examples/ddpm/config.yaml --tier smoke           # does it execute?      ~30s
uv run lab run examples/ddpm/config.yaml --tier overfit         # can it learn AT ALL?  pass/fail
uv run lab run examples/ddpm/config.yaml --tier fast            # better than baseline? 2k steps
uv run lab run examples/ddpm/config.yaml                        # the real run
```

(`source .venv/bin/activate` once and you can drop every `uv run` prefix. Source
install only: the PyPI name `imagelab` belongs to an unrelated package — don't
`pip install imagelab`.)

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

Three ways to point a config at a task, no packaging required for the first:

```yaml
task: task.py:MyIdea          # a file next to the config — clone-free experimentation
task: mypkg.tasks:MyIdea      # a dotted module path
task: my_idea                 # a registered name ([tool.imagelab] imports in pyproject.toml)
```

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

and one `lab runs` away:

```
$ lab runs
run                        task   tier     status  steps  min   seed  result               deltas
-------------------------  -----  -------  ------  -----  ----  ----  -------------------  ----------
0610-1845_ddpm_lr-0.0003   ddpm   fast     done    2000   18.2  0     noise_mse 0.0214     lr=0.0003
0610-1432_ddpm_overfit     ddpm   overfit  done    300    2.1   0     overfit:pass         
0610-1430_rf_overfit       rf     overfit  done    300    2.0   0     overfit:pass         
0610-1102_ddpm_smoke       ddpm   smoke    done    8      0.1   0     loss 0.9871          
```

```bash
lab runs                                   # the table above
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

## Working with AI agents

The repo ships **[AGENTS.md](AGENTS.md)**: the operating manual for coding agents
(Claude Code, Codex, ...) doing research on this framework — how to implement a paper
faithfully behind the Task seam, walk the tier ladder instead of jumping to full runs,
and treat the ledger as the lab notebook. If you point an agent at this repo, it reads
that file first; the discipline it encodes (kill cheap, change one variable, never tune
a gate to pass) is good advice for humans too.

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
tests/              # fast CPU seam tests (no network, no datasets)
AGENTS.md           # operating manual for AI coding agents (and disciplined humans)
```

MIT licensed. PRs: see `CONTRIBUTING.md`. Releases: see `CHANGELOG.md`.
