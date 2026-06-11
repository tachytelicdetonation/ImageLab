"""Generic manifest-backed image dataset.

Any dataset becomes usable here by writing a directory of the form

    <root>/images_<size>/<file>.png
    <root>/manifest.jsonl      # one record per line:
                               # {"file": ..., "name": ..., "caption": ...,
                               #  "dataset": <optional source tag>,
                               #  "split": <optional "train"|"val">}

(`imagelab/data/download_imagenette.py` produces this layout with `dataset` + `split`
tags; any script that writes the two files above makes a dataset.)

Train/val splitting: records with an explicit `split` field keep it (ImageNette ships an
official split). Records without one are assigned by a stable hash of their filename, so
the assignment never changes across runs/machines AND never reshuffles when the dataset
grows — adding images can't silently move old ones between train and val.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def record_split(rec: dict, val_fraction: float) -> str:
    """'train' or 'val' for a manifest record. Explicit manifest `split` wins; otherwise
    a filename-hash bucket (deterministic, per-item stable, library-independent)."""
    if rec.get("split") in ("train", "val"):
        return rec["split"]
    if val_fraction <= 0:
        return "train"
    bucket = zlib.crc32(rec["file"].encode("utf-8")) % 1000
    return "val" if bucket < int(round(val_fraction * 1000)) else "train"


class ManifestImageDataset(Dataset):
    """Loads images and captions; returns pixels normalized to [-1, 1].

    [-1, 1] is the range the encoder expects (mean=std=0.5) and the range our tanh decoder
    produces, so the same tensor serves as encoder input and reconstruction target.
    """

    def __init__(self, root: str | Path, size: int = 256, hflip: bool = True,
                 augment: bool = False, split: str | None = None,
                 val_fraction: float = 0.1):
        self.root = Path(root)
        self.size = size
        self.hflip = hflip
        self.augment = augment
        self.split = split
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
                f"(e.g. `python -m imagelab.data.download_imagenette --size 64`)."
            )
        self.records = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        # Keep only records whose image actually exists at this resolution.
        self.records = [r for r in self.records if (self.img_dir / r["file"]).exists()]
        if not self.records:
            raise RuntimeError(f"No images found in {self.img_dir}.")
        if split is not None:
            if split not in ("train", "val"):
                raise ValueError(f"split must be 'train', 'val' or None, got {split!r}")
            chosen = [r for r in self.records if record_split(r, val_fraction) == split]
            if not chosen:
                # Tiny datasets (smoke tests, toy manifests) can hash entirely into one
                # bucket. Falling back to everything beats crashing for a sanity run, but
                # the metrics are then train-set metrics — say so loudly.
                print(f"[data] WARNING: split='{split}' matched 0/{len(self.records)} "
                      f"records (val_fraction={val_fraction}) — using the FULL dataset; "
                      f"metrics will not be held-out")
            else:
                self.records = chosen

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


class OverfitDataset(Dataset):
    """The overfit sanity check: clamp a dataset to its first `n` records.

    If a change can't memorize ONE image, it's broken — wrong gradient path, dead
    codebook, capacity bug — and that's learnable in minutes, not after a full run
    (`--tier overfit` uses n=1). The n records are repeated up to `repeat_to` items so
    a "epoch" has enough batches for the loop's cadences to fire at sane intervals.
    """

    def __init__(self, ds: Dataset, n: int, repeat_to: int = 256):
        self.base = ds
        self.n = min(n, len(ds))
        self.repeat_to = max(repeat_to, self.n)
        base_records = ds.base.records if hasattr(ds, "base") else ds.records
        # Mirror `records` so GroupedEvaluator's tag-grouping sees the clamped view.
        self.records = [base_records[i % self.n] for i in range(self.repeat_to)]

    def __len__(self):
        return self.repeat_to

    def __getitem__(self, i):
        return self.base[i % self.n]


def _to_float_chw(img: Image.Image):
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0          # (H,W,3)
    return arr.transpose(2, 0, 1).copy()
