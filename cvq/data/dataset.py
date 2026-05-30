"""Pokemon image dataset backed by the manifest.jsonl produced by download_pokemon.py."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    """Loads images and captions; returns pixels normalized to [-1, 1].

    [-1, 1] is the range SigLIP expects (mean=std=0.5) and the range our tanh decoder
    produces, so the same tensor serves as encoder input and reconstruction target.
    """

    def __init__(self, root: str | Path, size: int = 256, hflip: bool = True,
                 augment: bool = False):
        self.root = Path(root)
        self.size = size
        self.hflip = hflip
        self.augment = augment
        # Heavy augmentation for tiny datasets (~1.3k imgs): mild because Pokemon are centered
        # subjects on light backgrounds. White fill on rotate/crop so we never teach dark borders.
        self._aug = None
        if augment:
            from torchvision import transforms as T
            self._aug = T.Compose([
                T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR, fill=255),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ])
        self.img_dir = self.root / f"images_{size}"
        manifest = self.root / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} not found — run `python -m cvq.data.download_pokemon` first."
            )
        self.records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        # Keep only records whose image actually exists at this resolution.
        self.records = [r for r in self.records if (self.img_dir / r["file"]).exists()]
        if not self.records:
            raise RuntimeError(f"No images found in {self.img_dir}.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(self.img_dir / rec["file"]).convert("RGB")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.LANCZOS)
        if self._aug is not None:
            img = self._aug(img)                           # PIL->PIL, keeps (size,size)
        x = torch.from_numpy(_to_float_chw(img))           # (3,H,W) in [0,1]
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        x = x * 2.0 - 1.0                                   # -> [-1,1]
        return {"image": x, "caption": rec["caption"], "name": rec["name"]}


def _to_float_chw(img: Image.Image):
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0          # (H,W,3)
    return arr.transpose(2, 0, 1).copy()
