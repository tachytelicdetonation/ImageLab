"""
Factory — the composition root. Configs (registry names + kwargs) go in, built models
come out. This is the ONLY place that maps config fields onto component constructors;
trainers, eval, and inference scripts all build through here.

Conventions a new component must satisfy (see cvq/registry.py for registration):
  * quantizer: __init__(token_dim=..., codebook_size=..., **model.quantizer_kwargs);
               forward(z (B,C,h,w)) -> (z_q, idxs (B,C), aux_loss, stats dict);
               truncate(z_q, c_keep); lookup(idxs) -> z_q. Accept **_ignore for kwargs
               you don't use.
  * encoder:   __init__(ch, ch_mult, num_res_blocks, z_channels, resolution,
               attn_resolutions); forward(x) -> (B, z_channels, g, g).
  * decoder:   same kwargs + out_ch; forward(z_q) -> (B, 3, H, W) in [-1,1].
  * ar_head:   see cvq/models/heads.py.
"""

from __future__ import annotations

from pathlib import Path

import torch

import cvq.models  # noqa: F401  — imports register all built-in components
from cvq.config import DataConfig, ModelConfig, RunConfig, _section
from cvq.conditioning import Conditioning
from cvq.models.car import CAR
from cvq.models.discriminator import NLayerDiscriminator
from cvq.models.tokenizer import CVQTokenizer
from imagelab.registry import build


def _model_data(cfg) -> tuple[ModelConfig, DataConfig, dict]:
    """Normalize a RunConfig OR a raw YAML/checkpoint dict to typed (model, data) sections."""
    if isinstance(cfg, RunConfig):
        return cfg.model, cfg.data, cfg.raw
    warnings: list[str] = []  # silently tolerate stale keys in old checkpoint-embedded configs
    m = _section(ModelConfig, "model", cfg.get("model"), warnings)
    d = _section(DataConfig, "data", cfg.get("data"), warnings)
    return m, d, cfg


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def build_tokenizer(cfg, device: str | torch.device,
                    ckpt: str | Path | None = None,
                    strict_load: bool = False) -> tuple[CVQTokenizer, dict]:
    """Build (and optionally warm-start) the CVQ tokenizer.

    Args:
        cfg: RunConfig, or the raw config dict, or {} when loading purely from a checkpoint
             (the checkpoint's embedded config is used — reconstruct/generate path).
        ckpt: optional checkpoint path; loads `ckpt["tokenizer"]` into the model.
        strict_load: forwarded to load_state_dict (False matches historical behavior).

    Returns:
        (tokenizer, cfg_dict) where cfg_dict is the raw dict actually used (the input's,
        or the checkpoint's embedded one).
    """
    if ckpt is not None and Path(ckpt).exists():
        ck = torch.load(str(ckpt), map_location=device)
        if not cfg:
            cfg = ck["config"]
        tok = _construct_tokenizer(cfg, device)
        tok.load_state_dict(ck["tokenizer"], strict=strict_load)
        m, d, raw = _model_data(cfg)
        return tok, raw

    tok = _construct_tokenizer(cfg, device)
    _, _, raw = _model_data(cfg)
    return tok, raw


def _construct_tokenizer(cfg, device) -> CVQTokenizer:
    m, d, _ = _model_data(cfg)
    g = d.size // m.downsample_factor()
    token_dim = g * g

    # Construction order (encoder -> quantizer -> decoder) is load-bearing: it preserves
    # the RNG consumption sequence of the original CVQTokenizer.__init__, so a fixed seed
    # yields bit-identical initialization before and after the refactor.
    encoder = build("encoder", m.encoder_type,
                    ch=m.enc_ch, ch_mult=tuple(m.enc_ch_mult),
                    num_res_blocks=m.decoder_res_blocks,    # historical: encoder reuses this
                    z_channels=m.latent_channels, resolution=d.size,
                    attn_resolutions=(g,))

    qkw = dict(m.quantizer_kwargs or {})
    if m.quant_type == "fsq":
        qkw.setdefault("levels", m.fsq_levels if m.fsq_levels else [2] * int(m.fsq_bits))
    else:
        qkw.setdefault("commitment_beta", m.commitment_beta)
    quantizer = build("quantizer", m.quant_type,
                      token_dim=token_dim, codebook_size=m.codebook_size, **qkw)

    decoder = build("decoder", m.decoder_type,
                    ch=m.decoder_ch, out_ch=3, ch_mult=tuple(m.decoder_ch_mult),
                    num_res_blocks=m.decoder_res_blocks,
                    z_channels=m.latent_channels, resolution=d.size,
                    attn_resolutions=(g,))

    return CVQTokenizer(encoder, quantizer, decoder,
                        latent_channels=m.latent_channels, grid=g).to(device)


# --------------------------------------------------------------------------- #
# CAR + text stack
# --------------------------------------------------------------------------- #
def build_car(m: ModelConfig, codebook_size: int, num_channels: int, device) -> CAR:
    car = CAR(codebook_size=codebook_size, num_channels=num_channels,
              qwen_name=m.qwen_name, freeze_backbone=m.freeze_backbone,
              attn_impl=m.attn_impl, head_type=m.head_type,
              mbm_depth=m.mbm_depth, mbm_heads=m.mbm_heads,
              mbm_infer_steps=m.mbm_infer_steps).to(device)
    n_train = sum(p.numel() for p in car.trainable_parameters())
    print(f"CAR: {m.qwen_name} | head={m.head_type} | trainable {n_train / 1e6:.1f}M | "
          f"freeze_backbone={m.freeze_backbone}")
    return car


def build_text_tokenizer(m: ModelConfig):
    """The HF text tokenizer for the CAR backbone — one loading site instead of three."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(m.qwen_name)


def build_conditioning(m: ModelConfig, text_tok, p_uncond: float = 0.0,
                       generator: torch.Generator | None = None,
                       device=None) -> Conditioning:
    cond = Conditioning(text_tok, max_len=m.max_text_len, p_uncond=p_uncond,
                        generator=generator, device=device)
    if cond.p_uncond > 0:
        print(f"caption dropout: ON | p={cond.p_uncond} (CFG-enabled)")
    return cond


def build_discriminator(device) -> NLayerDiscriminator:
    return NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)
