"""
Component registry — the extension seam for casual architecture experiments.

Every swappable component (quantizer, AR head, encoder, decoder, task, ...) registers
itself under a (kind, name) pair. Configs then select architectures by name:

    # my_quantizer.py (anywhere that gets imported)
    from imagelab.registry import register

    @register("quantizer", "lfq")
    class ChannelLFQ(nn.Module):
        def __init__(self, token_dim, **kwargs): ...

    # configs/my_experiment.yaml
    model:
      quant_type: lfq

Registration happens at import time, so a module just needs to be imported once before
anything builds by name. Projects declare their modules in pyproject.toml:

    [tool.imagelab]
    imports = ["myproject.models", "myproject.tasks"]

and the trainer/CLI import them automatically. Kinds are open strings — "task",
"quantizer", "ar_head", whatever your project's anatomy needs; the first register()
call creates a kind. The framework itself only ever looks up "task".
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}
_META: dict[tuple[str, str], dict] = {}


def register(kind: str, name: str, paper: str | None = None) -> Callable[[T], T]:
    """Class decorator: make `cls` buildable as build(kind, name, **kwargs).

    `paper` (e.g. "arXiv:2309.15505") credits the method's source — `lab cite <config>`
    collects these for every component a run uses.
    """

    def deco(cls: T) -> T:
        bucket = _REGISTRY.setdefault(kind, {})
        if name in bucket and bucket[name] is not cls:
            raise ValueError(
                f"duplicate registration: {kind}/{name} already bound to "
                f"{bucket[name].__module__}.{bucket[name].__qualname__}"
            )
        bucket[name] = cls
        if paper:
            _META[(kind, name)] = {"paper": paper}
        return cls

    return deco


def meta(kind: str, name: str) -> dict:
    """Registration metadata (paper reference etc.) for a component; {} if none."""
    return _META.get((kind, name), {})


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
