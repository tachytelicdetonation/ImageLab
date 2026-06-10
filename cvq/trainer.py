"""
The trainer — one spine for every task.

    python -m cvq.trainer --task e2e --config configs/car_e2e_inet_64.yaml
    python -m cvq.trainer --config configs/my.yaml          # task: from the YAML

(The legacy entry points `python -m cvq.train{,_car,_e2e}` still work; they are thin
shims that call run() with their historical default task + config.)

The spine: load+validate config -> seed -> build task (models/data/optimizers, see
cvq/tasks/) -> checkpoint store + logger -> resume -> TrainLoop -> final checkpoint.
Everything experiment-specific lives in the task; everything bookkeeping lives here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import cvq.tasks  # noqa: F401  — imports register all built-in tasks
from cvq.checkpoint import CheckpointStore
from cvq.config import load_config
from cvq.registry import available, build
from cvq.training_loop import Cadence, GANStep, NoGANStep, RunLogger, TrainLoop
from cvq.utils import describe_device, resolve_device


def run(default_task: str | None = None, default_config: str | None = None, argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_config, required=default_config is None)
    ap.add_argument("--task", default=None,
                    help=f"training task ({', '.join(available('task'))}); "
                         f"defaults to the config's `task:` key")
    ap.add_argument("--tokenizer_ckpt", default="",
                    help="tokenizer checkpoint (car: required source; e2e: optional warm start)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0=full)")
    args = ap.parse_args(argv)

    rc = load_config(args.config)
    task_name = args.task or rc.task or default_task
    if not task_name:
        raise SystemExit(f"no task given: pass --task or set `task:` in the config "
                         f"(available: {available('task')})")

    device = resolve_device(rc.train.device)
    amp = rc.train.amp
    print(f"task: {task_name} | device: {describe_device(device)} | amp: {amp}")
    torch.manual_seed(rc.train.seed)
    rng = torch.Generator().manual_seed(rc.train.seed)

    task = build("task", task_name, rc=rc, args=args, device=device, rng=rng)
    task.setup()

    store = CheckpointStore(rc.out.ckpt_dir, prefix=task.ckpt_prefix,
                            latest_name=task.latest_name, keep_last=rc.out.keep_last)
    logger = RunLogger(rc.raw, run_dir=rc.out.run_dir)
    task.store, task.logger = store, logger

    start_step, start_epoch = 0, 0
    if args.resume and Path(args.resume).exists():
        ck = CheckpointStore.load(args.resume, map_location=device)
        start_step, start_epoch = task.load_resume(ck)
        print(f"resumed from {args.resume} @ step {start_step}")

    cadence = Cadence(log_every=rc.train.log_every, sample_every=rc.train.sample_every,
                      val_every=rc.train.val_every, ckpt_every=rc.train.ckpt_every)
    loop = TrainLoop(device=device, amp=amp, accum=rc.train.grad_accum, cadence=cadence,
                     logger=logger, batch_size=rc.train.batch_size,
                     disc_start_step=rc.train.disc_start_step)
    if task.gan:
        runner = GANStep(task.disc, task.generator_fn, task.discriminator_fn,
                         accum=rc.train.grad_accum, device=device, amp=amp)
    else:
        runner = NoGANStep(task.step_fn, accum=rc.train.grad_accum, device=device, amp=amp)

    final_step = loop.run(
        dataloader=task.dataloader, epochs=rc.train.epochs,
        start_step=start_step, start_epoch=start_epoch,
        step_runner=runner,
        optimizers=task.optimizers, schedulers=task.schedulers,
        gen_params=task.gen_params, disc_params=task.disc_params,
        grad_clip=task.grad_clip(),
        sample_fn=task.sample_fn, val_fn=task.val_fn if task.has_val else None,
        ckpt_fn=task.ckpt_fn, max_steps=args.max_steps,
    )

    task.ckpt_fn(final_step, rc.train.epochs - 1)
    task.finalize(final_step)
    logger.close()
    print(f"{task_name} training complete.")
    return final_step


if __name__ == "__main__":
    run()
