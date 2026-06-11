"""
cvq's Task base — the bridge between the generic imagelab seam and cvq's typed config.

imagelab's trainer hands every task (core, raw, args, device, rng); this base turns
`raw` into cvq's fully-typed RunConfig (cvq/config.py — defaults + EXPERIMENT
GUARDRAILS) so the recipes keep their `self.rc.model.codebook_size`-style access.
cvq is deliberately a *consumer* of the framework: everything here is something any
project can do for itself (own config schema, own CLI flags, own tier tweaks).

To add a cvq training recipe: subclass Task, decorate with @register("task", "myname"),
import it from cvq/tasks/__init__.py, and set `task: myname` in a config.
"""

from __future__ import annotations

import dataclasses

import torch

import imagelab.task
from cvq.config import RunConfig, from_dict
from imagelab.loop import StepOutput  # noqa: F401  (re-export for task modules)


class Task(imagelab.task.Task):
    def __init__(self, core, raw: dict, args, device: str, rng: torch.Generator):
        super().__init__(core, raw, args, device, rng)
        # validate_config already ran (loudly) at load time; build quietly here.
        self.rc: RunConfig = from_dict(raw, quiet=True)
        # The trainer resolves artifact dirs (run dir unless YAML-pinned) into core.out.
        self.rc.out.ckpt_dir = core.out.ckpt_dir
        self.rc.out.sample_dir = core.out.sample_dir
        self.rc.out.run_dir = core.out.run_dir
        self.rc.out.keep_last = core.out.keep_last

    # ---- class-level hooks ---------------------------------------------- #
    @classmethod
    def validate_config(cls, raw: dict) -> None:
        from_dict(raw)              # cvq's full schema: warns on unknowns, raises on invalid

    @classmethod
    def apply_tier(cls, cfg: dict, tier: str):
        # cvq-specific tier hygiene on top of the generic transforms:
        # FID is meaningless at probe scale (and undefined over one repeated image).
        if tier in ("smoke", "overfit"):
            cfg.setdefault("train", {})["val_fid"] = False
        return None

    @classmethod
    def add_args(cls, parser) -> None:
        parser.add_argument("--tokenizer_ckpt", default="",
                            help="tokenizer checkpoint (car: required source; "
                                 "e2e: optional warm start)")

    @classmethod
    def resolved_config(cls, core, raw: dict) -> dict:
        """Every cvq default materialized — what the run ACTUALLY used."""
        rc = from_dict(raw, quiet=True)
        rc.out.ckpt_dir, rc.out.sample_dir = core.out.ckpt_dir, core.out.sample_dir
        rc.out.run_dir, rc.out.keep_last = core.out.run_dir, core.out.keep_last
        out = {"task": rc.task}
        for f in dataclasses.fields(rc):
            if f.name not in ("task", "raw"):
                out[f.name] = dataclasses.asdict(getattr(rc, f.name))
        return out

    @classmethod
    def cite_components(cls, cfg: dict) -> list:
        m = cfg.get("model", {}) or {}
        return [("task", cfg.get("task", cls.name)),
                ("encoder", m.get("encoder_type", "cnn")),
                ("quantizer", m.get("quant_type", "ibq")),
                ("decoder", m.get("decoder_type", "vqgan")),
                ("ar_head", m.get("head_type", "softmax"))]

    # ---- instance helpers ------------------------------------------------ #
    def grad_clip(self) -> float | None:
        return self.rc.train.grad_clip

    def param_counts(self) -> dict:
        """cvq detail on top of the generic per-module counts: tokenizer swaps should
        show their encoder/quantizer/decoder budgets right next to the metrics."""
        import torch.nn as nn
        counts = {}
        for attr in ("tok", "car", "disc", "dino"):
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Module):
                n = sum(p.numel() for p in mod.parameters())
                counts[attr] = round(n / 1e6, 3)
        if isinstance(getattr(self, "tok", None), nn.Module):
            for sub in ("encoder", "quantizer", "decoder"):
                m = getattr(self.tok, sub, None)
                if isinstance(m, nn.Module):
                    counts[f"tok.{sub}"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 3)
        return counts
