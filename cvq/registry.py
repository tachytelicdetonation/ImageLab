"""
Component registry — the extension seam for casual architecture experiments.

Every swappable component (quantizer, AR head, encoder, decoder, task, ...) registers
itself under a (kind, name) pair. Configs then select architectures by name:

    # cvq/models/my_quantizer.py
    from cvq.registry import register

    @register("quantizer", "lfq")
    class ChannelLFQ(nn.Module):
        def __init__(self, token_dim, **kwargs): ...

    # configs/my_experiment.yaml
    model:
      quant_type: lfq

Nothing else changes: the factory builds whatever name the config asks for. Registration
happens at import time, so built-in components are imported (and thus registered) by
`cvq/models/__init__.py`. A third-party module just needs to be imported once before the
factory runs — drop it in cvq/models/ and add it to that __init__, or import it from your
config-loading script.

Kinds in use today: "quantizer", "ar_head", "encoder", "decoder", "task".
New kinds need no declaration — the first register() call creates them.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    """Class decorator: make `cls` buildable as build(kind, name, **kwargs)."""

    def deco(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket and bucket[name] is not cls:
            raise ValueError(
                f"duplicate registration: {kind}/{name} already bound to "
                f"{bucket[name].__module__}.{bucket[name].__qualname__}"
            )
        bucket[name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> type:
    """Look up a registered class. Raises with the known names on a miss, so a typo or a
    never-imported module fails loudly instead of falling back to a default."""
    bucket = _REGISTRY.get(kind, {})
    if name not in bucket:
        known = ", ".join(sorted(bucket)) or "<none — is the defining module imported?>"
        raise KeyError(f"unknown {kind} '{name}'. Registered: {known}")
    return bucket[name]


def build(kind: str, name: str, **kwargs):
    """Instantiate a registered component: build('quantizer', 'ibq', token_dim=256, ...)."""
    return get(kind, name)(**kwargs)


def available(kind: str) -> list[str]:
    """Names registered under a kind (for error messages, docs, --help listings)."""
    return sorted(_REGISTRY.get(kind, {}))
