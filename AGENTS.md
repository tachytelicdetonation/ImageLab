# AGENTS.md — operating manual for coding agents

You are working in (or on top of) **imagelab**: a single-GPU experimentation framework
where the user owns everything scientific and the framework owns the run lifecycle.
This file is the contract for how to do research here. Humans: it's good advice for
you too.

## The one-seam rule

Everything scientific lives on ONE class — an `imagelab.Task` subclass: model, data,
loss, metrics, sampler, checkpoint schema. The framework owns run dirs, the ledger,
tiers, cadences, seeding, resume, and crash bookkeeping.

- **Never modify `imagelab/` to make a task work.** If a task needs something the seam
  doesn't offer, that's either a task-side hook you missed (`validate_config`,
  `apply_tier`, `add_args`, `resolved_config`, `cite_components`) or a framework
  feature request — say so instead of patching around it.
- A new experiment is a new directory: `task.py` + `config.yaml`, pointed at by
  `task: task.py:MyClass`. Start from `lab new task <name>` — it scaffolds a WORKING
  project (toy objective passes smoke + overfit before you edit a line). Replace the
  data and loss; keep the seams.
- Don't write outputs anywhere except the run dir the trainer gives you
  (`self.core.out.*`). Never write into another run's folder, never delete ledger rows.

## The tier ladder is not optional

Walk it in order. Each tier exists to kill the idea at the cheapest possible price:

```
lab check task task.py:MyClass    # wired right?               ~5s, no training
lab run cfg.yaml --tier smoke     # executes end to end?       8 steps
lab run cfg.yaml --tier overfit   # can it memorize 1 sample?  ~300 steps, auto verdict
lab run cfg.yaml --tier fast      # better than the baseline?  2k steps
lab run cfg.yaml                  # real numbers               hours
```

- **Never start with a full run.** If you haven't seen the overfit gate pass, you have
  no evidence the gradient path works.
- **Never tune `overfit_gate` to make a failing run pass.** The gate encodes the user's
  research taste. A marginal failure is a finding — investigate the loss wiring, the
  data pipeline, the LR — and report it.
- An experiment is a `--set` delta on a base config (`--set train.lr=3e-4
  model.width=128`), not a copied-and-edited YAML. Deltas land in the run name and the
  ledger; copies lose provenance.
- The ledger is the lab notebook. Negative results stay in it. Don't re-run something
  to "get a cleaner row"; run it with a different seed and compare.

## Implementing a paper faithfully

This is the most common task and the one with the strictest rules:

1. **Faithful first, clever later.** The first implementation matches the paper (and
   the official code, when it exists — papers under-specify; the code is the spec).
   Match: the objective (exact equation), schedules and their endpoints, init scheme,
   optimizer + betas + weight decay + EMA decay, data normalization, and the sampler.
   Only after the faithful version passes its gates do you ablate your own changes —
   one variable at a time, each as a `--set` delta or a clearly-named task subclass.
2. **Cite at the point of use.** Every non-obvious constant or formula gets a comment
   naming its source: `# Ho et al. 2020, Eq. 11` / `# official repo: beta swap, see
   quantize.py#L87`. Set the class attribute `paper = "arXiv:XXXX.XXXXX"` so
   `lab cite` can answer "what method is this run?".
3. **Every deviation is declared, justified, and isolated.** If you must deviate
   (memory, single-GPU budget, dataset size), write it down where it lives:
   `# DEVIATION from paper: batch 64 -> 16 (single-GPU); LR scaled linearly per
   Goyal et al.` A deviation the reader can't find is a bug with good intentions.
4. **Know your init-time invariants and check them at smoke tier.** Examples: an
   ε-prediction MSE with zero-initialized output conv starts at ≈1.0 (unit-normal
   target); a rectified-flow v-MSE starts at ≈1.24; a uniform softmax over K classes
   starts at ln(K). If step-0 loss is far off, the wiring is wrong — stop, don't train
   through it.
5. **Deterministic validation.** Fix the val inputs and the noise (constant seed,
   generated once in `setup()`), so the metric is comparable across runs AND seeds.
   See `examples/ddpm/task.py` for the pattern.
6. **Construction order in `setup()` is RNG-load-bearing.** A fixed seed must produce
   the same init run over run; don't reorder model/data construction casually, and say
   so in the diff when you must.
7. **Write the gate from the paper's expected behavior**, then calibrate: run
   `--tier overfit` once, see what a healthy run achieves, set the threshold with
   margin, and record the calibration number in the commit/PR message.

## Reporting results

- Lead with the ledger: run ids, the gate verdict, and the declared `key_metrics` —
  `lab runs` / `lab compare A B` output beats prose.
- Compare like with like: same steps, same seed policy, same val protocol. `lab
  compare` prints NOTEs when step counts/seeds/tasks differ — repeat those caveats in
  your summary, don't drop them.
- If a metric changed meaning (split change, eval-batch change, protocol change), every
  number before the change is incomparable — flag it loudly.

## Mechanics you'll need

- `self.raw` = the full config dict (your sections live there); `self.core` = the
  validated framework sections (`train`/`out`). `wandb:` is read by the logger.
- `self.store`/`self.logger` are injected AFTER `setup()` — using them inside `setup()`
  raises with an explanation. Warm-start by loading files directly.
- `step_fn` returns `StepOutput(loss=..., logs={...})`. GAN tasks set `gan = True` and
  implement `generator_fn`/`discriminator_fn` instead; the disc optimizer goes LAST in
  `self.optimizers`.
- `has_val = True` + `val_fn(step) -> (metrics_dict, images_dict)`; the last value of
  `self.last_val_metrics` becomes the run's result row.
- `checkpoint_state()` returns `(model_state, opt_state, latest_model_keys)`;
  checkpoints load with `weights_only=True` — keep the schema to tensors/dicts/
  primitives.
- Tests: `uv run pytest -q` (CPU, seconds, no network). Lint: `uv run ruff check
  imagelab examples tests`. Both must pass before you call work done.

## Don'ts, compressed

- Don't skip tiers; don't train through a failed smoke/overfit.
- Don't weaken a gate, a seed, or a val protocol to make a number look better.
- Don't copy configs when a `--set` delta states the experiment.
- Don't put non-distributable data (licensed images, scraped IP) in the repo or in
  samples that get committed/published.
- Don't claim "implemented the paper" while a known deviation is undocumented.
- Don't leave a finished run un-summarized: verdict, key metrics, and what you'd run
  next.
