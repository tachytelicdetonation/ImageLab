"""
Text conditioning — the protocol that makes CFG actually work.

Classifier-free guidance has one load-bearing invariant: the empty prompt the CAR sees
at SAMPLING time must tokenize to the exact same ids/mask the CAR saw at TRAINING time
when caption-dropout replaced a real prompt with "". Before this module the invariant
was kept by a code comment ("MUST match sample_generations()'s text_tok([""]*n, ...)")
spread across train_e2e.py + train_car.py + generate.py. One module now owns it, so the
training drop and the sampling uncond come from the same call.

Use:
    cond = Conditioning(text_tok, max_len=16, p_uncond=0.1, generator=gen, device=dev)
    ids, mask = cond.encode_batch(prompts)                # for both train & sample
    ids, mask = cond.maybe_drop(ids, mask)                # in train loop, per step
    u_ids, u_mask = cond.unconditional(batch_size, L=ids.shape[1])  # for CFG sampling
"""

from __future__ import annotations

import torch


class Conditioning:
    """Owns prompt tokenization, caption-dropout, and unconditional construction.

    Args:
        text_tok: HuggingFace tokenizer (Qwen). pad_token is auto-set if missing.
        max_len: max prompt length (truncates / max-length-pads).
        p_uncond: per-sample probability of replacing a real prompt with "" during training.
            0 disables caption dropout (and CFG at sampling is then untrained).
        generator: torch RNG for reproducible drop decisions. Optional.
        device: device the empty-prompt ids/mask live on. If None, lives on CPU until used.
    """

    def __init__(self, text_tok, max_len: int = 16, p_uncond: float = 0.0,
                 generator: torch.Generator | None = None, device: str | torch.device | None = None):
        self.tok = text_tok
        self.max_len = max_len
        self.p_uncond = p_uncond
        self.gen = generator
        self.device = device
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # Pre-tokenize "" at max-length — this is the canonical empty prompt. Training drop
        # and sampling uncond both come from THIS tensor, so they cannot drift.
        enc = self.tok([""], padding="max_length", truncation=True, max_length=max_len,
                       return_tensors="pt")
        self._empty_ids = enc["input_ids"][0]                 # (max_len,)
        self._empty_mask = enc["attention_mask"][0]           # (max_len,)
        if device is not None:
            self._empty_ids = self._empty_ids.to(device)
            self._empty_mask = self._empty_mask.to(device)

    def encode_batch(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of prompts (right-padded to the batch's longest, capped at max_len)."""
        enc = self.tok(prompts, padding="longest", truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def maybe_drop(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample caption dropout: with prob p_uncond, replace a row with the empty prompt.

        Empty-prompt rows are RIGHT-PADDED to the batch's text length so the resulting
        (B, L) tensor is rectangular. mask=0 on those pad positions, exactly as the encoder
        would emit if you tokenized "" at this length.
        """
        if self.p_uncond <= 0:
            return text_ids, text_mask
        B, L = text_ids.shape
        device = text_ids.device
        # Truncate-or-pad the cached empty ids to this batch's L.
        emp_ids = self._empty_ids.to(device)
        emp_mask = self._empty_mask.to(device)
        if emp_ids.shape[0] < L:
            pad_id = self.tok.pad_token_id
            extra = torch.full((L - emp_ids.shape[0],), pad_id, dtype=emp_ids.dtype, device=device)
            emp_ids = torch.cat([emp_ids, extra], 0)
            emp_mask = torch.cat([emp_mask, torch.zeros_like(extra)], 0)
        else:
            emp_ids = emp_ids[:L]
            emp_mask = emp_mask[:L]

        drop = (torch.rand(B, generator=self.gen, device="cpu") < self.p_uncond).to(device)
        if not drop.any():
            return text_ids, text_mask
        out_ids = text_ids.clone()
        out_mask = text_mask.clone()
        out_ids[drop] = emp_ids
        out_mask[drop] = emp_mask
        return out_ids, out_mask

    def unconditional(self, batch_size: int, L: int | None = None,
                      device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (B, L) unconditional ids/mask for CFG sampling.

        Pads/truncates the cached empty prompt to L (defaults to max_len) so the result
        matches the shape of a conditional batch encoded with `encode_batch`.
        """
        L = L if L is not None else self.max_len
        device = device if device is not None else self.device
        emp_ids = self._empty_ids
        emp_mask = self._empty_mask
        if emp_ids.shape[0] < L:
            pad_id = self.tok.pad_token_id
            extra = torch.full((L - emp_ids.shape[0],), pad_id, dtype=emp_ids.dtype)
            emp_ids = torch.cat([emp_ids.cpu(), extra], 0)
            emp_mask = torch.cat([emp_mask.cpu(), torch.zeros_like(extra)], 0)
        else:
            emp_ids = emp_ids[:L]
            emp_mask = emp_mask[:L]
        ids = emp_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        mask = emp_mask.unsqueeze(0).expand(batch_size, -1).contiguous()
        if device is not None:
            ids = ids.to(device)
            mask = mask.to(device)
        return ids, mask
