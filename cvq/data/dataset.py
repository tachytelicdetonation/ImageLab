"""Generic manifest-backed image dataset.

Any dataset becomes usable here by writing a directory of the form

    <root>/images_<size>/<file>.png
    <root>/manifest.jsonl      # one record per line:
                               # {"file": ..., "name": ..., "caption": ...,
                               #  "dataset": <optional source tag>}

(`cvq/data/download_pokemon.py` produces this layout for Pokemon; an ImageNette+Woof
builder produces it with `dataset` tags for per-source eval splits.)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    """Loads images and captions; returns pixels normalized to [-1, 1].

    [-1, 1] is the range the encoder expects (mean=std=0.5) and the range our tanh decoder
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
                f"{manifest} not found — build the dataset first "
                f"(e.g. `python -m cvq.data.download_pokemon`)."
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
        # `dataset` tags which source a record came from (e.g. imagenette|imagewoof) so eval
        # can split easy-vs-hard grids/metrics. Absent for single-source sets (Pokemon) -> "all".
        return {"image": x, "caption": rec["caption"], "name": rec["name"],
                "dataset": rec.get("dataset", "all")}


# Back-compat alias (the class predates multi-dataset support).
PokemonDataset = ManifestImageDataset


def _to_float_chw(img: Image.Image):
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0          # (H,W,3)
    return arr.transpose(2, 0, 1).copy()
