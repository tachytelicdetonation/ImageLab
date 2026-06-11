# Changelog

## 0.4.0 — 2026-06-10

The repo is now the framework. Breaking release; also a **history rewrite** (media that
should never have been committed was purged from every commit) — re-clone rather than
pull if you have an old checkout.

- **Barebones split**: the public repo is `imagelab/` + `examples/` + `tests/` only.
  The research project the framework grew out of (CVQ/EOSTok) moved out of the tree.
- **Safety/correctness fixes** from an external-style review:
  - gradient-accumulation windows no longer reset at epoch boundaries (trailing
    micro-batches were silently dropped when `len(loader) % accum != 0`)
  - discriminator gradients are clipped under `train.grad_clip` (previously only
    logged)
  - `torch.load` defaults to `weights_only=True` everywhere — checkpoints can't
    execute code on load
  - file-path task imports: hash-based module names (no cache collisions), clear
    error for unloadable files
  - `self.store`/`self.logger` access inside `setup()` raises a helpful lifecycle
    error instead of `AttributeError`
  - a half-written ledger row no longer crashes every `lab` command;
    `IMAGELAB_RUNS_ROOT` env var can pin the runs root
  - `lab run` appears in `lab --help`; `--set key=` (empty value) warns about YAML
    null
- **Leaner install**: dependencies are torch/torchvision/pillow/numpy/tqdm/pyyaml;
  wandb + tensorboard moved behind the `[logging]` extra (the filesystem ledger is the
  source of truth without them).
- **CI that actually runs**: lock-pinned installs (`uv sync --locked`, CPU torch),
  ruff lint gate, Python 3.11 + 3.12 matrix, scaffold-rot smoke run, example contract
  probes. The previous workflow had never executed past its first step.
- **New tests**: GAN-path smoke, resume round-trip (incl. `weights_only=True`
  compatibility), accum-window regression, `lab runs/board/compare/report` rendering,
  dotted `module:Class` resolution.
- **AGENTS.md**: operating manual for AI coding agents (tier ladder, paper-fidelity
  rules, ledger discipline). README quickstart is now copy-pasteable on a clean
  machine (`uv sync` + `uv run`).

## 0.3.0 — 2026-06-10

- `imagelab/` extracted as a model-family-agnostic framework: `Task` seam, generic
  trainer spine, tasks declare `overfit_gate`/`key_metrics`/`higher_is_better`,
  declarations stamped into ledger rows so the CLI renders any family without
  importing model code.
- `examples/ddpm` + `examples/rectified_flow`: two complete model families ~40 lines
  apart, shared TinyUNet, deterministic fixed-noise validation.
- Open registries for project-defined checkers (`lab check`) and scaffolds
  (`lab new`); project discovery via `[tool.imagelab] imports`.

## 0.2.0 — 2026-06-09

- The lab layer: `lab` CLI (run/runs/compare/board/gallery/report/sweep/check/new/
  cite), execution tiers (smoke/overfit/fast/full), `--set` deltas, `runs/<id>/`
  self-describing run dirs + `runs/ledger.jsonl`.
- Held-out val split by stable filename hash (`data.val_fraction`).

## 0.1.0 and earlier

- Research codebase (vector-quantized tokenizers + autoregressive image generation)
  from which the framework was extracted.
