"""
Execution tiers + --set overrides: iteration speed as a feature, not a hand-made config.

Every base config gets four tiers for free; each answers ONE question, costs ~10x the
previous, and has a kill criterion. Bad ideas should die at the cheapest tier that can
kill them:

    smoke    ~30s   does it execute?       (8 steps, no eval, no wandb)
    overfit  ~min   can it learn AT ALL?   (1 sample, 300 steps, pass/fail against the
                                            task's `overfit_gate` declaration)
    fast     ~20min better than baseline?  (2k-step budget, normal eval cadence)
    full     hours  paper-faithful numbers (config untouched)

Tiers are dict transforms applied to the parsed YAML BEFORE validation, and they land in
the run's resolved config.yaml + checkpoint-embedded config — a tiered run is fully
honest about what it actually ran. Explicit --set overrides are applied after the tier,
so they always win.

The generic tiers only touch framework keys plus two documented CONVENTION keys
(train.warmup_steps, data.hflip/augment — harmless when a task ignores them). Tasks add
their own transforms (and can override the step cap) via `Task.apply_tier(cfg, tier)`.
"""

from __future__ import annotations

import re

import yaml

TIERS = ("full", "fast", "overfit", "smoke")


def apply_tier(cfg: dict, tier: str) -> int:
    """Mutate `cfg` for the tier; returns the tier's max_steps cap (0 = no cap)."""
    if tier not in TIERS:
        raise SystemExit(f"unknown tier {tier!r} (choose from {TIERS})")
    if tier == "full":
        return 0
    t = cfg.setdefault("train", {})
    cfg.setdefault("wandb", {})

    if tier == "smoke":
        t.update(log_every=1, sample_every=4, val_every=10**9, ckpt_every=10**9,
                 num_workers=0)
        cfg["wandb"]["enabled"] = False
        return 8

    if tier == "overfit":
        d = cfg.setdefault("data", {})
        d["hflip"] = False           # memorize THE sample, not it and its mirror
        d["augment"] = False
        t.update(overfit_n=1,
                 log_every=10, sample_every=100, val_every=10**9, ckpt_every=10**9,
                 epochs=10000)                 # max_steps is the real stop
        # A full-run warmup inside a 300-step probe would test the LR schedule, not the
        # architecture — the probe must reach full LR almost immediately.
        t["warmup_steps"] = min(t.get("warmup_steps", 0), 20)
        cfg["wandb"]["enabled"] = False
        return 300

    if tier == "fast":
        t.setdefault("val_every", 500)
        t["val_every"] = min(t.get("val_every", 500), 500)
        t["sample_every"] = min(t.get("sample_every", 500), 500)
        t["ckpt_every"] = min(t.get("ckpt_every", 1000), 1000)
        t["warmup_steps"] = min(t.get("warmup_steps", 0), 200)
        return 2000
    return 0


def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """Apply `--set dotted.key=value` overrides (YAML-typed) to the parsed config.
    Returns the {key: value} delta dict — the experiment, stated as data."""
    deltas = {}
    for pair in pairs or []:
        key, sep, val = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects dotted.key=value, got {pair!r}")
        parsed = yaml.safe_load(val)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.setdefault(p, {})
            if not isinstance(nxt, dict):
                raise SystemExit(f"--set {key}: '{p}' is not a config section")
            node = nxt
        node[parts[-1]] = parsed
        deltas[key] = parsed
    return deltas


def auto_name(config_stem: str, tier: str, deltas: dict, explicit: str = "") -> str:
    """Run names encode the experiment: <config>[_tier][_key-value...]."""
    if explicit:
        return _slug(explicit)
    name = config_stem
    if tier != "full":
        name += f"_{tier}"
    for k, v in deltas.items():
        name += f"_{k.split('.')[-1]}-{_slug(str(v))}"
    return name[:80]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
