"""
CAR dataset: pairs each Pokemon image with its name, tokenized by the Qwen tokenizer.

The image channel-token indices are NOT precomputed here — they are produced on the fly
by the frozen CVQ tokenizer inside the training loop (so a re-trained tokenizer needs no
data regeneration). This dataset only handles pixels + text.

Captions are the Pokemon name verbatim (e.g. "pikachu", "rayquaza-mega"). We prettify the
hyphenated form into a natural prompt ("rayquaza mega") so the Qwen tokenizer sees words.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .dataset import PokemonDataset


def prettify_name(name: str) -> str:
    """`rayquaza-mega` -> `rayquaza mega`; `raichu-mega-x` -> `raichu mega x`."""
    return name.replace("-", " ").strip()


class CARPokemonDataset(Dataset):
    """Wraps PokemonDataset; returns image + raw prompt string. Tokenization happens in
    collate so we can batch-pad with the Qwen tokenizer."""

    def __init__(self, root, size=256, hflip=True):
        self.base = PokemonDataset(root, size=size, hflip=hflip)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base[idx]
        return {"image": rec["image"], "prompt": prettify_name(rec["name"])}


class CARCollate:
    """Collate that tokenizes the batch of prompts with the Qwen tokenizer (right-padded)."""

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
        }
