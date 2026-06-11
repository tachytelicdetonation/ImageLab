"""
Captioned dataset for AR training: pairs each image with a text prompt, tokenized by the
backbone's text tokenizer at collate time.

The image channel-token indices are NOT precomputed here — they are produced on the fly
by the CVQ tokenizer inside the training loop (so a re-trained tokenizer needs no data
regeneration). This dataset only handles pixels + text.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from imagelab.data.dataset import ManifestImageDataset


def prettify_name(name: str) -> str:
    """`rayquaza-mega` -> `rayquaza mega`; `raichu-mega-x` -> `raichu mega x`."""
    return name.replace("-", " ").strip()


class CaptionedImageDataset(Dataset):
    """Wraps ManifestImageDataset; returns image + raw prompt string. Tokenization happens
    in collate so we can batch-pad with the backbone's tokenizer."""

    def __init__(self, root, size=256, hflip=True, augment=False, split=None,
                 val_fraction=0.1):
        self.base = ManifestImageDataset(root, size=size, hflip=hflip, augment=augment,
                                         split=split, val_fraction=val_fraction)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base[idx]
        # Prefer an explicit caption (e.g. "a photo of a golden retriever") when the manifest
        # provides one richer than the bare name; fall back to the prettified name. For Pokemon
        # caption == name so behaviour is unchanged; for ImageNette/woof this gives the AR a
        # CLIP-style template with grounded class words instead of a lone token.
        cap = rec.get("caption", "").strip()
        name = prettify_name(rec["name"])
        prompt = cap if (cap and cap != rec["name"]) else name
        return {"image": rec["image"], "prompt": prompt,
                "dataset": rec.get("dataset", "all")}


# Back-compat alias.
CARPokemonDataset = CaptionedImageDataset


class CARCollate:
    """Collate that tokenizes the batch of prompts with the text tokenizer (right-padded)."""

    def __init__(self, tokenizer, max_len: int = 16):
        self.tok = tokenizer
        self.max_len = max_len
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch], 0)
        prompts = [b["prompt"] for b in batch]
        enc = self.tok(
            prompts, padding="longest", truncation=True, max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "image": images,
            "text_ids": enc["input_ids"],
            "text_mask": enc["attention_mask"],
            "prompts": prompts,
            "datasets": [b.get("dataset", "all") for b in batch],
        }
