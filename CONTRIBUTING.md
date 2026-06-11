# Contributing

ImageLab is a small lab for single-GPU model experimentation: the framework owns the
run lifecycle, users own everything scientific. Contributions that fit best: framework
seams that make bad runs die cheaper, new example model families, and datasets in the
manifest format.

## The seams are the public API

- **`imagelab.Task`** (imagelab/task.py) is THE contract: `setup` / `step_fn` (or
  `generator_fn`+`discriminator_fn`) / `val_fn` / `sample_fn` / `checkpoint_state`,
  plus the editorial declarations (`overfit_gate`, `key_metrics`, `higher_is_better`)
  and the class hooks (`validate_config`, `apply_tier`, `add_args`, `cite_components`).
- **Runs as data**: `runs/<id>/` folders + `runs/ledger.jsonl` (append-only, last row
  per run_id wins). Ledger rows must stay renderable WITHOUT importing model code.
- **`(kind, name)` registry** with open kinds; projects register via
  `[tool.imagelab] imports` in pyproject.toml.

Changes to these seams are breaking changes and need a strong reason.

## Adding an example model family

An example is a self-contained directory: `task.py` + `config.yaml` (+ model code) that
runs via `lab run examples/<name>/config.yaml --tier smoke|overfit|fast`. The bar:

1. `lab check task examples/<name>/task.py:<Class>` passes (CI runs this).
2. A declared `overfit_gate` whose threshold you actually calibrated with a real
   `--tier overfit` run — report the number in the PR.
3. Deterministic val metrics (fixed eval inputs/noise, constant seed) so runs are
   comparable across seeds.
4. A `paper=` reference (or arXiv id in the docstring) for the method.

## House rules

- **Fail at config load, not at step 4000.** Framework-level validation lives in
  `imagelab/config.py`; project schemas hook in via `Task.validate_config`.
- **Offline-first.** `runs/<id>/` + the ledger are the source of truth; wandb/TB are
  optional mirrors (the `[logging]` extra). Tests must run with no network and no
  datasets.
- **Faithfulness over convenience** in reference implementations — match the official
  code/paper and comment deviations where they live (AGENTS.md has the full
  discipline). Construction order in `setup()` is RNG-load-bearing; don't break seeded
  reproducibility casually.
- **No silent science changes.** Anything that alters what a metric means (splits, eval
  batches, protocols) gets a loud README note.

## Out of scope (by design)

Multi-GPU/distributed training, production serving, framework adoption
(Lightning/Hydra/MLflow), and web dashboards. The niche is single-GPU research with
trustworthy comparisons.

## Dev setup

```bash
uv sync                                            # locked install incl. dev tools
uv run pytest -q                                   # CPU-only, seconds
uv run ruff check imagelab examples tests          # CI gates on this
```

CI (`.github/workflows/ci.yml`) runs lint + tests + a scaffold smoke run + the example
contract probes on Python 3.11 and 3.12, installed from `uv.lock` — if you change
dependencies, commit the regenerated lockfile (`uv lock`).
