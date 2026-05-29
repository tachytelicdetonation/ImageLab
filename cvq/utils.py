"""Small shared helpers (device selection)."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> str:
    """Pick a device. 'auto' -> cuda > mps > cpu. Falls back gracefully."""
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to auto")
        return resolve_device("auto")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable, falling back to CPU")
        return "cpu"
    return name


def describe_device(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"cuda: {p.name}, {p.total_memory / 1e9:.0f}GB"
    return device
