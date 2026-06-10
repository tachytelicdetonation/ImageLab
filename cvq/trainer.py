"""
The trainer — one spine for every task.

    python -m cvq.trainer --config configs/tok_inet_64.yaml                # full run
    python -m cvq.trainer --config configs/tok_inet_64.yaml --tier smoke   # does it run?
    python -m cvq.trainer --config configs/tok_inet_64.yaml --tier overfit # can it learn?
    python -m cvq.trainer --config configs/tok_inet_64.yaml \
        --set model.quant_type=lfq train.lr=2e-4 --tier fast               # the experiment

(`lab run ...` is the same thing; the legacy entry points `python -m cvq.train{,_car,_e2e}`
still work as thin shims.)

The spine: load config -> tier transform -> --set overrides -> validate -> run dir +
ledger row -> seed -> build task (cvq/tasks/) -> resume -> TrainLoop -> final checkpoint
+ eval -> ledger row (done|crashed, final metrics, params, steps/sec). Every run lands in
runs/<id>/ with its resolved config, provenance, and metrics.jsonl — see cvq/lab/.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import yaml

import cvq.tasks  # noqa: F401  — imports register all built-in tasks
from cvq.checkpoint import CheckpointStore
from cvq.config import from_dict
from cvq.lab.criteria import overfit_verdict
from cvq.lab.rundir import RunDir
from cvq.lab.tiers import TIERS, apply_overrides, apply_tier, auto_name
from cvq.registry import available, build
from cvq.training_loop import Cadence, GANStep, NoGANStep, RunLogger, TrainLoop
from cvq.utils import describe_device, resolve_device


def run(default_task: str | None = None, default_config: str | None = None, argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config, required=default_config is None)
    ap.add_argument("--task", default=None,
                    help=f"training task ({', '.join(available('task'))}); "
                         f"defaults to the config's `task:` key")
    ap.add_argument("--tier", default="full", choices=list(TIERS),
                    help="smoke=does it run, overfit=can it memorize 1 image, "
                         "fast=2k-step budget, full=the config as written")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set model.quant_type=lfq train.lr=2e-4")
    ap.add_argument("--name", default="", help="run name (default: config stem + deltas)")
    ap.add_argument("--tokenizer_ckpt", default="",
                    help="tokenizer checkpoint (car: required source; e2e: optional warm start)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0=full)")
    args = ap.parse_args(argv)

    # ---- config: parse -> tier -> overrides (overrides win) -> validate ----
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{args.config} did not parse to a mapping")
    tier_steps = apply_tier(cfg, args.tier)
    deltas = apply_overrides(cfg, args.overrides)
    task_name = args.task or cfg.get("task") or default_task
    if not task_name:
        raise SystemExit(f"no task given: pass --task or set `task:` in the config "
                         f"(available: {available('task')})")

    # ---- run dir + ledger ----
    name = auto_name(Path(args.config).stem, args.tier, deltas, explicit=args.name)
    rd = RunDir.create(name)
    wcfg = cfg.setdefault("wandb", {})
    if wcfg.get("enabled") and not wcfg.get("name"):
        wcfg["name"] = rd.run_id        # mirror runs share the ledger's identity
    rc = from_dict(cfg)
    out_raw = cfg.get("out") or {}
    if "ckpt_dir" not in out_raw:       # YAML-pinned out dirs (box scripts) stay honored
        rc.out.ckpt_dir = str(rd.dir / "checkpoints")
    if "sample_dir" not in out_raw:
        rc.out.sample_dir = str(rd.dir / "samples")
    rd.start(name=name, task=task_name, config_path=args.config, tier=args.tier,
             deltas=deltas, rc=rc, argv=argv if argv is not None else sys.argv[1:])

    device = resolve_device(rc.train.device)
    print(f"run: {rd.run_id}\ntask: {task_name} | tier: {args.tier} | "
          f"device: {describe_device(device)} | amp: {rc.train.amp}")
    torch.manual_seed(rc.train.seed)
    rng = torch.Generator().manual_seed(rc.train.seed)

    logger = None
    try:
        task = build("task", task_name, rc=rc, args=args, device=device, rng=rng)
        task.setup()

        store = CheckpointStore(rc.out.ckpt_dir, prefix=task.ckpt_prefix,
                                latest_name=task.latest_name, keep_last=rc.out.keep_last)
        logger = RunLogger(rc.raw, run_dir=rc.out.run_dir or rd.dir / "tb",
                           jsonl_path=rd.dir / "metrics.jsonl")
        task.store, task.logger = store, logger

        start_step, start_epoch = 0, 0
        if args.resume and Path(args.resume).exists():
            ck = CheckpointStore.load(args.resume, map_location=device)
            start_step, start_epoch = task.load_resume(ck)
            print(f"resumed from {args.resume} @ step {start_step}")

        cadence = Cadence(log_every=rc.train.log_every, sample_every=rc.train.sample_every,
                          val_every=rc.train.val_every, ckpt_every=rc.train.ckpt_every)
        loop = TrainLoop(device=device, amp=rc.train.amp, accum=rc.train.grad_accum,
                         cadence=cadence, logger=logger, batch_size=rc.train.batch_size,
                         disc_start_step=rc.train.disc_start_step)
        if task.gan:
            runner = GANStep(task.disc, task.generator_fn, task.discriminator_fn,
                             accum=rc.train.grad_accum, device=device, amp=rc.train.amp)
        else:
            runner = NoGANStep(task.step_fn, accum=rc.train.grad_accum, device=device,
                               amp=rc.train.amp)

        final_step = loop.run(
            dataloader=task.dataloader, epochs=rc.train.epochs,
            start_step=start_step, start_epoch=start_epoch,
            step_runner=runner,
            optimizers=task.optimizers, schedulers=task.schedulers,
            gen_params=task.gen_params, disc_params=task.disc_params,
            grad_clip=task.grad_clip(),
            sample_fn=task.sample_fn, val_fn=task.val_fn if task.has_val else None,
            ckpt_fn=task.ckpt_fn, max_steps=args.max_steps or tier_steps,
        )

        task.ckpt_fn(final_step, rc.train.epochs - 1)
        task.finalize(final_step)

        # train-side logs as fallback so every run has a result row; val metrics win.
        final_metrics = {**loop.last_logs, **task.last_val_metrics}
        verdict = None
        if rc.train.overfit_n:
            verdict, note = overfit_verdict(task_name, final_metrics)
            print(f"\noverfit gate: {verdict.upper()} — {note}")
        rd.finish(status="done", steps=final_step, final_metrics=final_metrics,
                  params=task.param_counts(), verdict=verdict)
        print(f"{task_name} training complete. run dir: {rd.dir}")
        return final_step
    except KeyboardInterrupt:
        rd.finish(status="killed", steps=0)
        raise
    except (Exception, SystemExit) as e:
        rd.finish(status="crashed", steps=0, error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    finally:
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    run()
