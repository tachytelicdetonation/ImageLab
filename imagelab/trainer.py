"""
The trainer — one spine for every task, whatever the model family.

    lab run config.yaml                          # full run
    lab run config.yaml --tier smoke             # does it run?
    lab run config.yaml --tier overfit           # can it learn? (1 sample, auto verdict)
    lab run config.yaml --set train.lr=2e-4 --tier fast    # the experiment

(`python -m imagelab.trainer --config ...` is the same thing.)

The spine: load config -> resolve task -> tier transform -> --set overrides -> validate
-> run dir + ledger row -> seed -> task.setup() -> TrainLoop -> final checkpoint + eval
-> ledger row (done|crashed, final metrics, params, steps/sec). Every run lands in
runs/<id>/ with its resolved config, provenance, and metrics.jsonl — see imagelab/lab/.

Task resolution (`task:` in the YAML or --task):
    task: my_idea                    # a name registered via @register("task", "my_idea")
    task: task.py:MyIdea             # a file next to the config — no packaging needed
    task: mypkg.tasks:MyIdea         # a dotted module path
Registered names come from the modules listed in pyproject.toml [tool.imagelab] imports,
which this module imports automatically (see project_imports).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
import traceback
from pathlib import Path

import torch
import yaml

from imagelab.checkpoint import CheckpointStore
from imagelab.config import core_from_dict
from imagelab.lab.criteria import gate_verdict
from imagelab.lab.rundir import RunDir
from imagelab.lab.tiers import TIERS, apply_overrides, apply_tier, auto_name
from imagelab.loop import Cadence, GANStep, NoGANStep, RunLogger, TrainLoop
from imagelab.registry import available, get as registry_get
from imagelab.utils import describe_device, resolve_device


# --------------------------------------------------------------------------- #
# Project discovery — how user code gets imported (and thus registered)
# --------------------------------------------------------------------------- #
def project_imports(start: Path | None = None) -> list[str]:
    """Walk up from `start` (default cwd) to the nearest pyproject.toml and import the
    modules listed under [tool.imagelab] imports. The project root is also put on
    sys.path so those modules resolve from any subdirectory. Failures warn, not crash —
    a half-broken module shouldn't take down `lab runs`."""
    import tomllib
    d = (start or Path.cwd()).resolve()
    for root in (d, *d.parents):
        pp = root / "pyproject.toml"
        if not pp.exists():
            continue
        try:
            mods = tomllib.loads(pp.read_text()).get("tool", {}).get(
                "imagelab", {}).get("imports", [])
        except Exception as e:
            print(f"[lab] WARNING: could not parse {pp}: {e}")
            return []
        if mods and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        loaded = []
        for m in mods:
            try:
                importlib.import_module(m)
                loaded.append(m)
            except Exception as e:
                print(f"[lab] WARNING: [tool.imagelab] import {m!r} failed: "
                      f"{type(e).__name__}: {e}")
        return loaded
    return []


def _import_file(path: Path):
    """Import a .py file under a stable synthetic module name. The file's directory goes
    on sys.path so sibling imports (`from unet import UNet`) work — a project can be a
    plain directory of files."""
    path = path.resolve()
    name = "_labmod_" + re.sub(r"[^A-Za-z0-9]+", "_", str(path)).strip("_")[-80:]
    if name in sys.modules:
        return sys.modules[name]
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_task(spec: str, config_dir: Path | None = None) -> type:
    """`task:` value -> Task class. See module docstring for the three forms."""
    from imagelab.task import Task
    if ":" in spec:
        mod_part, _, attr = spec.rpartition(":")
        if mod_part.endswith(".py"):
            cand = Path(mod_part)
            if not cand.is_absolute():
                rel = (config_dir or Path.cwd()) / cand
                cand = rel if rel.exists() else cand
            if not cand.exists():
                raise SystemExit(f"task file {mod_part!r} not found "
                                 f"(looked at {cand})")
            module = _import_file(cand)
        else:
            try:
                module = importlib.import_module(mod_part)
            except ImportError as e:
                raise SystemExit(f"task module {mod_part!r} not importable: {e}")
        cls = getattr(module, attr, None)
        if cls is None:
            raise SystemExit(f"{mod_part} has no attribute {attr!r}")
    else:
        try:
            cls = registry_get("task", spec)
        except KeyError:
            raise SystemExit(
                f"unknown task {spec!r}. Registered: {available('task') or '(none)'}.\n"
                f"Either register it ([tool.imagelab] imports in pyproject.toml) or "
                f"point at it directly: `task: path/to/task.py:ClassName`")
    if not (isinstance(cls, type) and issubclass(cls, Task)):
        raise SystemExit(f"{spec!r} resolved to {cls!r}, which is not an "
                         f"imagelab.Task subclass")
    return cls


# --------------------------------------------------------------------------- #
def run(argv=None) -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None,
                    help="task spec (registered name | file.py:Class | module:Class); "
                         "defaults to the config's `task:` key")
    ap.add_argument("--tier", default="full", choices=list(TIERS),
                    help="smoke=does it run, overfit=can it memorize 1 sample, "
                         "fast=2k-step budget, full=the config as written")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VAL",
                    help="config overrides, e.g. --set model.ch=128 train.lr=2e-4")
    ap.add_argument("--name", default="", help="run name (default: config stem + deltas)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0=full)")
    args, extra = ap.parse_known_args(argv)

    # ---- config: parse -> resolve task -> tier -> overrides (overrides win) ----
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"{args.config} did not parse to a mapping")
    project_imports(Path(args.config).resolve().parent)
    task_spec = args.task or cfg.get("task")
    if not task_spec:
        raise SystemExit(f"no task given: pass --task or set `task:` in the config "
                         f"(registered: {available('task') or '(none)'})")
    task_cls = resolve_task(task_spec, Path(args.config).resolve().parent)

    tier_steps = apply_tier(cfg, args.tier)
    hook_steps = task_cls.apply_tier(cfg, args.tier)
    if hook_steps is not None:
        tier_steps = hook_steps
    deltas = apply_overrides(cfg, args.overrides)
    task_cls.validate_config(cfg)            # fail at load, not at step 4000

    # ---- task-declared CLI flags (e.g. cvq's --tokenizer_ckpt) ----
    tp = argparse.ArgumentParser(prog=f"task {task_spec!r} flags", allow_abbrev=False)
    task_cls.add_args(tp)
    for k, v in vars(tp.parse_args(extra)).items():
        setattr(args, k, v)

    # ---- run dir + ledger ----
    stem = Path(args.config).stem
    if stem in ("config", "cfg"):       # examples/ddpm/config.yaml -> "ddpm", not "config"
        stem = Path(args.config).resolve().parent.name
    name = auto_name(stem, args.tier, deltas, explicit=args.name)
    rd = RunDir.create(name)
    wcfg = cfg.setdefault("wandb", {})
    if wcfg.get("enabled") and not wcfg.get("name"):
        wcfg["name"] = rd.run_id        # mirror runs share the ledger's identity
    core = core_from_dict(cfg)
    out_raw = cfg.get("out") or {}
    if "ckpt_dir" not in out_raw:       # YAML-pinned out dirs stay honored
        core.out.ckpt_dir = str(rd.dir / "checkpoints")
    if "sample_dir" not in out_raw:
        core.out.sample_dir = str(rd.dir / "samples")
    task_label = task_cls.name or task_spec
    rd.start(name=name, task=task_label, config_path=args.config, tier=args.tier,
             deltas=deltas, resolved=task_cls.resolved_config(core, cfg),
             seed=core.train.seed, device=core.train.device, amp=core.train.amp,
             key_metrics=list(task_cls.key_metrics),
             higher_is_better=sorted(task_cls.higher_is_better),
             argv=argv if argv is not None else sys.argv[1:])

    device = resolve_device(core.train.device)
    print(f"run: {rd.run_id}\ntask: {task_label} | tier: {args.tier} | "
          f"device: {describe_device(device)} | amp: {core.train.amp}")
    torch.manual_seed(core.train.seed)
    rng = torch.Generator().manual_seed(core.train.seed)

    logger = None
    try:
        task = task_cls(core=core, raw=cfg, args=args, device=device, rng=rng)
        task.setup()

        store = CheckpointStore(core.out.ckpt_dir, prefix=task.ckpt_prefix,
                                latest_name=task.latest_name, keep_last=core.out.keep_last)
        logger = RunLogger(cfg, run_dir=core.out.run_dir or rd.dir / "tb",
                           jsonl_path=rd.dir / "metrics.jsonl")
        task.store, task.logger = store, logger

        start_step, start_epoch = 0, 0
        if args.resume and Path(args.resume).exists():
            ck = CheckpointStore.load(args.resume, map_location=device)
            start_step, start_epoch = task.load_resume(ck)
            print(f"resumed from {args.resume} @ step {start_step}")

        cadence = Cadence(log_every=core.train.log_every,
                          sample_every=core.train.sample_every,
                          val_every=core.train.val_every, ckpt_every=core.train.ckpt_every)
        loop = TrainLoop(device=device, amp=core.train.amp, accum=core.train.grad_accum,
                         cadence=cadence, logger=logger, batch_size=core.train.batch_size,
                         disc_start_step=core.train.disc_start_step)
        if task.gan:
            runner = GANStep(task.disc, task.generator_fn, task.discriminator_fn,
                             accum=core.train.grad_accum, device=device, amp=core.train.amp)
        else:
            runner = NoGANStep(task.step_fn, accum=core.train.grad_accum, device=device,
                               amp=core.train.amp)

        final_step = loop.run(
            dataloader=task.dataloader, epochs=core.train.epochs,
            start_step=start_step, start_epoch=start_epoch,
            step_runner=runner,
            optimizers=task.optimizers, schedulers=task.schedulers,
            gen_params=task.gen_params, disc_params=task.disc_params,
            grad_clip=task.grad_clip(),
            sample_fn=task.sample_fn, val_fn=task.val_fn if task.has_val else None,
            ckpt_fn=task.ckpt_fn, max_steps=args.max_steps or tier_steps,
        )

        task.ckpt_fn(final_step, core.train.epochs - 1)
        task.finalize(final_step)

        # train-side logs as fallback so every run has a result row; val metrics win.
        final_metrics = {**loop.last_logs, **task.last_val_metrics}
        verdict = None
        if core.train.overfit_n:
            verdict, note = gate_verdict(task_cls.overfit_gate, final_metrics)
            print(f"\noverfit gate: {verdict.upper()} — {note}")
        rd.finish(status="done", steps=final_step, final_metrics=final_metrics,
                  params=task.param_counts(), verdict=verdict)
        print(f"{task_label} training complete. run dir: {rd.dir}")
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
