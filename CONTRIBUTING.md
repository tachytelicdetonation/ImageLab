# Contributing

ImageLab is a small lab for casual image-model experimentation. Contributions that fit
best: new registered components (quantizers, AR heads, encoders, decoders, tasks),
datasets in the manifest format, and guardrails/diagnostics that make bad runs die
cheaper.

## The component contracts are the public API

Configs select components by registry name; the constructor/forward conventions are
documented in `cvq/factory.py` (quantizer/encoder/decoder), `cvq/models/heads.py`
(ar_head), and `cvq/tasks/base.py` (task). Changes to these seams are breaking changes
and need a strong reason.

## Adding a component

```bash
lab new quantizer my_idea       # scaffolds a WORKING contract implementation
lab check quantizer my_idea     # contract probes (CI runs these too)
lab run configs/tok_inet_64.yaml --set model.quant_type=my_idea --tier overfit
lab run configs/tok_inet_64.yaml --set model.quant_type=my_idea --tier fast
```

The bar for a PR that adds a component:

1. `lab check` passes (CI enforces this for builtins — add yours to `.github/workflows/ci.yml`).
2. A contract test in `tests/test_lab.py` (add your name to the parametrized list).
3. A `--tier fast` result vs. the unmodified base config on ImageNette-64, reported in
   the PR description (`lab compare <yours> <baseline>` output is ideal).
4. A `paper=` reference on the `@register` line when the method has a source.

## House rules

- **Faithfulness over convenience.** Reference reimplementations match the official
  code/paper, with deviations commented where they live. The refactor history here was
  verified bit-for-bit; don't break seeded reproducibility casually (construction order
  in `setup()` is RNG-load-bearing).
- **Fail at config load, not at step 4000.** Invalid combinations belong in
  `cvq/config.py::validate` (the "EXPERIMENT GUARDRAILS" section).
- **Offline-first.** `runs/<id>/` + the ledger are the source of truth; wandb/TB are
  mirrors. Tests must run with no network and no datasets.
- **The package is `cvq`, the project is `imagelab`** (like scikit-learn/sklearn) —
  don't rename imports.

## Out of scope (by design)

Multi-GPU/distributed training, production serving, >256px scale, framework adoption
(Lightning/Hydra/MLflow), and web dashboards. The niche is single-GPU casual research
with trustworthy comparisons.

## Dev setup

```bash
uv venv --python 3.12 && uv pip install -e . --group dev
uv run pytest tests/ -q
```
