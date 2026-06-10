"""
Typed run configuration — one place for defaults, one place for validation.

Before this module every default lived at its call site (`tcfg.get("car_lr", tcfg["lr"])`
scattered across three trainers, sometimes with *different* fallbacks for the same key),
and invalid combinations (an MBM bit head on a non-bit IBQ codebook) crashed mid-training
instead of at load time.

The YAML layout is unchanged — same sections, same keys — so every existing config keeps
working. `load_config(path)` parses, applies defaults, validates, and returns a
`RunConfig` whose `.raw` dict is byte-identical to the YAML (that raw dict is what gets
embedded in checkpoints and sent to wandb, exactly as before).

Unknown keys WARN rather than error: several older configs carry stale keys
(`encoder_lr`, `quantizer_type`) and they should keep loading — but a typo like
`lamda_ntp` should be visible the second you launch, not after an 8-epoch run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_SAMPLE_PROMPTS = [
    "pikachu", "charizard", "bulbasaur", "mewtwo",
    "rayquaza mega", "gengar", "eevee", "snorlax",
]


class ConfigError(ValueError):
    """A config that cannot produce a meaningful run. Raised at load time."""


# --------------------------------------------------------------------------- #
# Sections — field names match the YAML keys 1:1
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    root: str = ""        # empty is allowed for the car task (root comes from the tokenizer ckpt)
    size: int = 256
    hflip: bool = True
    augment: bool = False


@dataclass
class ModelConfig:
    # ---- tokenizer: encoder -> quantizer -> decoder (all registry names) ----
    latent_channels: int = 256
    encoder_type: str = "cnn"
    enc_ch: int = 128
    enc_ch_mult: list = field(default_factory=lambda: [1, 1, 2, 2, 4])
    quant_type: str = "ibq"
    codebook_size: int = 16384
    commitment_beta: float = 0.25
    quantizer_kwargs: dict = field(default_factory=dict)
    fsq_levels: list | None = None
    fsq_bits: int | None = None
    decoder_type: str = "vqgan"
    decoder_ch: int = 128
    decoder_ch_mult: list = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_res_blocks: int = 2
    # ---- CAR: LLM backbone + image head ----
    qwen_name: str = "Qwen/Qwen3-0.6B-Base"
    freeze_backbone: bool = False
    attn_impl: str = "sdpa"
    head_type: str = "softmax"
    mbm_depth: int = 3
    mbm_heads: int = 8
    mbm_infer_steps: int = 4
    dino_name: str = "facebook/dinov2-large"
    max_text_len: int = 16
    # ---- sampling ----
    sample_prompts: list = field(default_factory=lambda: list(DEFAULT_SAMPLE_PROMPTS))
    sample_prompts_by_dataset: dict | None = None
    temperature: float = 1.0
    top_k: int = 0
    cfg_scale: float = 1.0

    def downsample_factor(self) -> int:
        return 2 ** (len(self.enc_ch_mult) - 1)


@dataclass
class LossConfig:
    recon_loss_type: str = "l1"
    perceptual_weight: float = 1.0
    codebook_weight: float = 1.0
    disc_weight: float = 0.8
    disc_loss: str = "hinge"
    lpips_net: str = "vgg"
    gan_eta: float = 0.05


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    device: str = "auto"
    amp: str = "none"
    grad_accum: int = 1
    car_lr: float | None = None            # None -> lr
    beta1: float = 0.5
    beta2: float = 0.9
    weight_decay: float = 1e-4
    warmup_steps: int = 0
    grad_clip: float = 1.0
    optimizer: str = "adamw"               # adamw | muon | pion (tokenizer task only)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    pion_promotion_steps: int = 0
    # ---- schedule gates ----
    disc_start_step: int = 0
    ar_start_step: int = 0
    # ---- EOSTok loss weights ----
    lambda_ntp: float = 1.0
    lambda_apr: float = 0.0
    apr_lpips_weight: float = 0.0
    apr_min_c_keep_frac: float = 0.25
    lambda_sem: float = 0.0
    # ---- CVQ nested dropout + channel weighting ----
    nested_dropout_prob: float = 0.0
    channel_weight_schedule: str = "uniform"
    channel_weight_alpha: float = 1.0
    cond_dropout_prob: float = 0.0
    # ---- cadence / bookkeeping ----
    log_every: int = 50
    sample_every: int = 500
    val_every: int = 1000
    ckpt_every: int = 1000
    val_fid: bool = True
    num_workers: int = 0
    seed: int = 0
    hflip: bool = True                     # used by the car task (data root comes from ckpt)
    encoder_lr: float | None = None        # stale (SigLIP era); accepted, unused

    def resolved_car_lr(self) -> float:
        return self.car_lr if self.car_lr is not None else self.lr


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "cvq-pokemon"
    entity: str | None = None
    name: str | None = None
    mode: str = "online"
    log_checkpoints: bool = True


@dataclass
class OutConfig:
    ckpt_dir: str = "checkpoints"
    sample_dir: str = "samples"
    run_dir: str | None = None
    keep_last: int = 5


@dataclass
class RunConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    wandb: WandbConfig
    out: OutConfig
    task: str | None = None      # which task this config is meant for (tokenizer|car|e2e)
    raw: dict = field(default_factory=dict, repr=False)  # the YAML verbatim (ckpt/wandb)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
_SECTIONS = {
    "data": DataConfig, "model": ModelConfig, "loss": LossConfig,
    "train": TrainConfig, "wandb": WandbConfig, "out": OutConfig,
}


def _section(cls, name: str, d: dict | None, warnings: list[str]):
    d = dict(d or {})
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - known)
    for k in unknown:
        warnings.append(f"config: unknown key {name}.{k} (ignored)")
        d.pop(k)
    required = [f.name for f in dataclasses.fields(cls)
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
                and f.name not in d]
    if required:
        raise ConfigError(f"config section '{name}' is missing required key(s): {required}")
    return cls(**d)


def from_dict(cfg: dict, strict: bool = False) -> RunConfig:
    """Build a RunConfig from a parsed YAML dict. `cfg` is kept verbatim as `.raw`."""
    warnings: list[str] = []
    sections = {key: _section(cls, key, cfg.get(key), warnings)
                for key, cls in _SECTIONS.items()}
    top_unknown = sorted(set(cfg) - set(_SECTIONS) - {"task"})
    for k in top_unknown:
        warnings.append(f"config: unknown top-level section '{k}' (ignored)")
    rc = RunConfig(task=cfg.get("task"), raw=cfg, **sections)
    warnings += validate(rc)
    for w in warnings:
        print(f"[config] WARNING: {w}")
    if strict and warnings:
        raise ConfigError("strict mode: warnings above are fatal")
    return rc


def load_config(path: str | Path, strict: bool = False) -> RunConfig:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} did not parse to a mapping")
    return from_dict(cfg, strict=strict)


# --------------------------------------------------------------------------- #
# Validation — fail at load time, not at step 4000
# --------------------------------------------------------------------------- #
def validate(rc: RunConfig) -> list[str]:
    """Cross-field checks. Hard violations raise ConfigError; returns soft warnings.

    EXPERIMENT GUARDRAILS: when an architecture combination wastes a GPU-day, encode the
    lesson here as a rule. This function is the single gate every run passes through.
    """
    m, t, l, d = rc.model, rc.train, rc.loss, rc.data
    warnings: list[str] = []

    # ---- enums ----
    _enum("loss.recon_loss_type", l.recon_loss_type, ("l1", "l2"))
    _enum("loss.disc_loss", l.disc_loss, ("hinge", "vanilla"))
    _enum("train.amp", t.amp, ("none", "bf16"))
    _enum("train.channel_weight_schedule", t.channel_weight_schedule,
          ("uniform", "linear", "sqrt", "exp"))
    _enum("train.optimizer", t.optimizer, ("adamw", "muon", "pion"))
    _enum("model.head_type", m.head_type, ("softmax", "mbm"))

    # ---- geometry: image size must survive the encoder's downsampling ----
    f = m.downsample_factor()
    if d.size % f != 0:
        raise ConfigError(
            f"data.size={d.size} is not divisible by the encoder downsample factor "
            f"f={f} (= 2^(len(enc_ch_mult)-1)); adjust enc_ch_mult or size")
    if len(m.enc_ch_mult) != len(m.decoder_ch_mult):
        warnings.append(
            f"enc_ch_mult ({len(m.enc_ch_mult)} levels) != decoder_ch_mult "
            f"({len(m.decoder_ch_mult)} levels): encoder f={f} but the decoder upsamples "
            f"2^{len(m.decoder_ch_mult) - 1}x — recon resolution will not match input")

    # ---- quantizer requirements ----
    if m.quant_type == "fsq":
        if not m.fsq_levels and not m.fsq_bits:
            raise ConfigError("quant_type=fsq requires model.fsq_bits or model.fsq_levels")
        if m.codebook_size != ModelConfig.codebook_size and not m.fsq_levels:
            # fsq derives |C| from levels; a hand-set codebook_size is ignored
            warnings.append("quant_type=fsq: model.codebook_size is ignored "
                            "(|C| = prod(levels) = 2^fsq_bits)")
    elif m.quant_type == "ibq":
        if m.codebook_size <= 0:
            raise ConfigError("quant_type=ibq requires model.codebook_size > 0")
        if m.fsq_bits or m.fsq_levels:
            warnings.append("quant_type=ibq: fsq_bits/fsq_levels are ignored")

    # ---- head <-> quantizer coupling ----
    if m.head_type == "mbm" and m.quant_type != "fsq":
        raise ConfigError(
            "head_type=mbm predicts the BITS of each token index, which is only meaningful "
            "for FSQ's bit-structured indices (quant_type=fsq). IBQ indices are arbitrary "
            "codebook slots — bit prediction over them is noise.")
    if t.lambda_apr > 0 and (m.head_type != "softmax" or m.quant_type != "ibq"):
        raise ConfigError(
            "lambda_apr > 0 needs the EOSTok soft-codebook decode: a softmax head over a "
            "learned codebook (head_type=softmax + quant_type=ibq). For Fork B (fsq/mbm) "
            "set lambda_apr: 0.")
    if m.head_type == "mbm" and t.channel_weight_schedule != "uniform":
        warnings.append("channel_weight_schedule is ignored by the MBM head "
                        "(weights only apply to softmax NTP)")

    # ---- CFG sanity ----
    if m.cfg_scale != 1.0 and t.cond_dropout_prob <= 0:
        warnings.append(
            f"cfg_scale={m.cfg_scale} but cond_dropout_prob=0: the model never saw the "
            "empty prompt during training, so CFG will amplify noise, not signal")

    return warnings


def _enum(name: str, value, allowed: tuple):
    if value not in allowed:
        raise ConfigError(f"{name}={value!r} — must be one of {allowed}")
