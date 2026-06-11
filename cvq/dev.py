"""
quick — REPL/notebook one-liners for poking at components. Thinking happens by poking;
none of this should require assembling a full config first.

    from cvq.dev import quick
    q   = quick.quantizer("lfq", temperature=0.2)   # tiny built component, cpu
    tok = quick.tokenizer(quant="fsq", fsq_bits=6)  # tiny full tokenizer
    x   = quick.batch(8)                            # real local images, [-1,1]
    quick.show(tok(x)["recon"])                     # grid -> /tmp + `open` on macOS
    tok2 = quick.load("runs/0610_lfq/checkpoints/best.pt")   # any checkpoint

Everything is CPU-sized and seeds torch.manual_seed(0) before construction so two REPL
sessions build identical weights.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

TINY_QWEN_DIR = Path.home() / ".cache" / "imagelab" / "tiny_qwen3"


class quick:
    """Namespace, not an instance — call the staticmethods directly."""

    @staticmethod
    def cfg(**overrides) -> dict:
        """A tiny VALID raw config dict (cpu, 32px, 16 channels). Section overrides merge:
        quick.cfg(model={"quant_type": "fsq", "fsq_bits": 6})."""
        base = {
            "data": {"root": "data", "size": 32, "hflip": False},
            "model": {"latent_channels": 16, "codebook_size": 64,
                      "enc_ch": 32, "enc_ch_mult": [1, 2],          # GroupNorm needs ch%32==0
                      "decoder_ch": 32, "decoder_ch_mult": [1, 2],
                      "decoder_res_blocks": 1},
            "train": {"batch_size": 2, "epochs": 1, "lr": 1e-4, "device": "cpu"},
            "wandb": {"enabled": False},
        }
        for sec, d in overrides.items():
            base.setdefault(sec, {}).update(d if isinstance(d, dict) else {sec: d})
        return base

    @staticmethod
    def quantizer(name: str, token_dim: int = 16, codebook_size: int = 64, **kwargs):
        import cvq.models  # noqa: F401
        from imagelab.registry import build
        torch.manual_seed(0)
        return build("quantizer", name, token_dim=token_dim,
                     codebook_size=codebook_size, **kwargs)

    @staticmethod
    def ar_head(name: str, hidden: int = 32, codebook_size: int = 64, **kwargs):
        import cvq.models  # noqa: F401
        from imagelab.registry import build
        torch.manual_seed(0)
        kw = {"backbone_dtype": torch.float32, "mbm_depth": 1, "mbm_heads": 2,
              "mbm_infer_steps": 2, **kwargs}
        return build("ar_head", name, hidden=hidden, codebook_size=codebook_size, **kw)

    @staticmethod
    def tokenizer(quant: str = "ibq", size: int = 32, **model_overrides):
        """A tiny full encoder->quantizer->decoder tokenizer on cpu."""
        from cvq.factory import build_tokenizer
        model = dict(model_overrides)
        model["quant_type"] = quant
        if quant == "fsq":
            model.setdefault("fsq_bits", 6)
        c = quick.cfg(model=model)
        c["data"]["size"] = size
        torch.manual_seed(0)
        tok, _ = build_tokenizer(c, "cpu")
        return tok

    @staticmethod
    def batch(n: int = 8, size: int = 64, root: str = "data") -> torch.Tensor:
        """(n,3,size,size) in [-1,1]: first n val images from the local dataset, or random
        tensors (with a notice) when no dataset is downloaded."""
        try:
            from imagelab.data.dataset import ManifestImageDataset
            ds = ManifestImageDataset(root, size=size, hflip=False, split="val")
            return torch.stack([ds[i]["image"] for i in range(min(n, len(ds)))], 0)
        except (FileNotFoundError, RuntimeError):
            print(f"(no dataset at {root}/ — returning random tensors; "
                  f"`python -m imagelab.data.download_imagenette` for real ones)")
            torch.manual_seed(0)
            return torch.rand(n, 3, size, size) * 2 - 1

    @staticmethod
    def load(ckpt: str, device: str = "cpu"):
        """Any tokenizer-bearing checkpoint -> built + loaded CVQTokenizer."""
        from cvq.factory import build_tokenizer
        tok, _ = build_tokenizer({}, device, ckpt=ckpt)
        return tok.eval()

    @staticmethod
    def load_car(ckpt: str, device: str = "cpu"):
        """A car/e2e checkpoint -> (car, tok), both eval()."""
        from cvq.config import ModelConfig, _section
        from cvq.factory import build_car, build_tokenizer
        ck = torch.load(ckpt, map_location=device)
        tok, raw = build_tokenizer({}, device, ckpt=ckpt)
        m = _section(ModelConfig, "model", raw.get("model"), [])
        car = build_car(m, tok.quantizer.codebook_size, tok.latent_channels, device)
        car.load_state_dict(ck["car"])
        return car.eval(), tok.eval()

    @staticmethod
    def show(x: torch.Tensor, path: str = "/tmp/quick_show.png", denormalize: bool = True):
        """Save a grid of (B,3,H,W) [-1,1] images and `open` it (macOS)."""
        from torchvision.utils import make_grid, save_image

        from imagelab.utils import denorm
        if x.dim() == 3:
            x = x[None]
        grid = make_grid(denorm(x.detach().float().cpu()) if denormalize else x,
                         nrow=min(8, x.shape[0]))
        save_image(grid, path)
        print(f"saved {path}")
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        return path


def tiny_qwen(dir: Path | str | None = None) -> str:
    """A 2-layer/64-dim Qwen3 (+ the real Qwen tokenizer) for CAR/e2e prototyping —
    fabricated deterministically on first use (~1MB), cached under ~/.cache/imagelab.
    Point `model.qwen_name` at the returned path and the full text->image stack runs on
    CPU in seconds. First call needs network once for the tokenizer files (or a warm HF
    cache); everything after is offline.
    """
    d = Path(dir) if dir else TINY_QWEN_DIR
    if (d / "config.json").exists() and (d / "tokenizer_config.json").exists():
        return str(d)
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    d.mkdir(parents=True, exist_ok=True)
    print(f"fabricating tiny qwen3 -> {d}")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    cfg = Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      vocab_size=tok.vocab_size + 1024,  # room for added/special tokens
                      max_position_embeddings=512)
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(cfg)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)
