"""
Tokenizer factory — one place that knows how to build a CVQTokenizer from a config.

Before this module, the constructor's 10-kwarg call was open-coded in train.py,
train_e2e.py, reconstruct.py (build_from_ckpt), and would have grown to anywhere else
the tokenizer was instantiated. Adding a new kwarg meant editing every site — or the
new kwarg silently fell back to the constructor default at the call site that forgot.

`build_tokenizer(cfg, device, ckpt=None)` is the seam. Both fresh-build and resume go
through the same kwarg-from-cfg mapping; a future encoder swap (the docstring already
mentions "a LeJEPA-pretrained ViT plugs in here") becomes an arg on this factory rather
than a new tokenizer class.
"""

from __future__ import annotations

from pathlib import Path

import torch

from cvq.models.tokenizer import CVQTokenizer


def build_tokenizer(cfg: dict, device: str | torch.device,
                    ckpt: str | Path | None = None,
                    strict_load: bool = False) -> tuple[CVQTokenizer, dict]:
    """Build (and optionally warm-start) the CVQ tokenizer from a full run config.

    Args:
        cfg: full run config dict (the YAML, parsed). Reads `cfg["model"]` and `cfg["data"]`.
        device: torch device (or string).
        ckpt: optional checkpoint path. If given, loads `ckpt["tokenizer"]` into the model.
        strict_load: forwarded to load_state_dict. Default False because the frozen encoder
                     historically wasn't in the checkpoint (it was loaded from HF in __init__);
                     keeping False for parity with the old code paths.

    Returns:
        (tokenizer, cfg). cfg is the input cfg when building fresh, or the checkpoint's
        embedded config when loading (matches reconstruct.build_from_ckpt's old semantics).
    """
    if ckpt is not None and Path(ckpt).exists():
        ck = torch.load(str(ckpt), map_location=device)
        # When the caller passes only a ckpt and an empty/placeholder cfg, fall back to the
        # config embedded in the checkpoint (this is what reconstruct.py needs).
        if not cfg:
            cfg = ck["config"]
        tok = _construct(cfg, device)
        tok.load_state_dict(ck["tokenizer"], strict=strict_load)
        return tok, cfg

    tok = _construct(cfg, device)
    return tok, cfg


def _construct(cfg: dict, device) -> CVQTokenizer:
    m = cfg["model"]
    return CVQTokenizer(
        resolution=cfg["data"]["size"],
        latent_channels=m["latent_channels"],
        codebook_size=m.get("codebook_size", 16384),
        commitment_beta=m.get("commitment_beta", 0.25),
        quantizer_kwargs=m.get("quantizer_kwargs", None),
        quant_type=m.get("quant_type", "ibq"),
        fsq_levels=m.get("fsq_levels", None),
        fsq_bits=m.get("fsq_bits", None),
        enc_ch=m.get("enc_ch", 128),
        enc_ch_mult=tuple(m.get("enc_ch_mult", [1, 1, 2, 2, 4])),
        decoder_ch=m["decoder_ch"],
        decoder_ch_mult=tuple(m["decoder_ch_mult"]),
        decoder_res_blocks=m["decoder_res_blocks"],
    ).to(device)
