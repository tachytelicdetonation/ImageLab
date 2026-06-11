"""Small shared helpers: device selection, image denorm, grad norms."""

from __future__ import annotations

import torch


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] model space -> [0,1] image space (for saving/metrics). Was copy-pasted as a
    lambda in every trainer; this is the one definition."""
    return x.clamp(-1, 1) * 0.5 + 0.5


def grad_norm(params) -> float:
    """L2 norm of gradients over an iterable of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().norm() ** 2)
    return total ** 0.5


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
