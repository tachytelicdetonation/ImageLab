# Examples — complete model families, one class each

Each directory is a full imagelab project: `task.py` + `config.yaml` (+ model code),
runnable from the repo root with no packaging:

```bash
python -m imagelab.data.download_imagenette --size 64       # once

lab run examples/ddpm/config.yaml --tier smoke              # executes?
lab run examples/ddpm/config.yaml --tier overfit            # learns? (auto verdict)
lab run examples/rectified_flow/config.yaml --tier overfit
lab compare <ddpm-run> <rf-run>                             # two families, one ledger
```

| example | method | objective | sampler |
|---|---|---|---|
| `ddpm/` | DDPM (Ho et al. 2020, arXiv:2006.11239) | ε-prediction MSE | DDIM (50 steps) |
| `rectified_flow/` | Rectified flow (Liu et al. 2022, arXiv:2209.03003) | velocity MSE | Euler (50 steps) |

They share the UNet in `ddpm/unet.py` and the same deterministic val protocol (fixed
`(t, ε)` pairs, constant seed 1234), so their numbers are comparable run-to-run and
seed-to-seed. Diff the two `task.py` files — the model families differ by ~40 lines of
objective + sampler. That diff is the framework's whole pitch.

Starting your own project? `lab new task my_idea` scaffolds this exact shape with a
working toy objective; replace the data and the loss, keep the seams.

Note on the overfit gates: the shipped thresholds (`0.05` on the fixed-noise MSE at 600
probe steps) are provisional — calibrate against your hardware with one
`--tier overfit` run and tighten them to what a healthy run actually achieves. The gate
is a declaration on the task class; it's meant to encode YOUR taste.
