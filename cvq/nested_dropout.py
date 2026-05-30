"""
Nested channel dropout — the paper's coarse-to-fine training signal.

CVQ (arXiv:2605.26089) trains the decoder to reconstruct from a *prefix* of the channel-
tokens (`z_q[:, :c_keep]`, rest zeroed). Sampling the cut point during training is what
forces the channel ordering to be coarse-to-fine: early channels carry global structure
because they have to *alone* produce a recognizable reconstruction, later channels carry
detail because they're only ever supervised on top of the earlier ones.

Three things were spread across train.py / train_e2e.py / quantizer.py / losses.py:
  * sampling c_keep per step (duplicated literally in train.py and train_e2e.py)
  * applying the mask to z_q (`IBQChannelVQ.truncate`)
  * channel-count-aware GAN weight (`CVQLoss.lambda_gan`)
They are one concept; this module owns it. (The GAN weight stays on the loss — it's a
loss-time decision that *consumes* c_keep rather than producing it.)

Policies are pluggable: today's hybrid "with prob `prob` truncate uniformly, else full"
is `HybridUniformPolicy`. A bias-toward-fine policy or an annealed `prob` schedule swaps
in as a new policy without touching the train loops.
"""

from __future__ import annotations

from typing import Protocol

import torch


# --------------------------------------------------------------------------- #
# Channel-weight schedules — the bridge from "channels are ordered coarse->fine"
# (CVQ's nested-dropout claim) to "penalize early channels more" (EOSTok's NTP/APR).
#
# Without this, EOSTok's NTP cross-entropy is uniform across channels: a wrong c_0 (global
# structure) costs as much as a wrong c_255 (a texture detail). Weighting NTP/APR by a
# decaying schedule formally couples the two papers: the AR model is now told that the
# CVQ-ordered prefix matters more, which matches what nested dropout already taught the
# tokenizer's decoder.
# --------------------------------------------------------------------------- #
def channel_weights(total_channels: int, schedule: str = "linear",
                    alpha: float = 1.0, device=None, dtype=torch.float32) -> torch.Tensor:
    """Per-channel loss weight vector of shape (C,), normalized so mean(w) = 1.

    Holding mean(w) = 1 means changing the schedule does NOT change the overall NTP/APR
    scale -- only its distribution across channels. So `lambda_ntp` / `lambda_apr` in the
    config keep their old meaning when this is turned on.

    Schedules (all parameterized by a single knob `alpha` = early/late ratio):
      * "uniform":  w_c = 1.  (the old behavior; baseline.)
      * "linear":   w decays linearly from alpha * mean to (2 - alpha) * mean over c=0..C-1.
                    alpha=2 => first channel weighted 2x the average, last channel = 0
                    (degenerate, silences the tail). alpha=1.5 is the recommended default:
                    first channel 3x last channel, neither extreme. alpha=1 => uniform.
      * "sqrt":     w_c proportional to sqrt(C - c). Fixed ratio C^0.5:1 between first and
                    last (~16:1 at C=256). Old-style schedule kept for back-compat.
      * "exp":      w_c proportional to exp(-alpha * c / C). alpha=1 => first/last = e (~2.7x),
                    alpha=2 => ~7.4x. Smoothest decay.

    Sanity reference at C=256:
      linear  alpha=1.5  -> first 1.50x mean, last 0.50x mean   (3:1)
      linear  alpha=2.0  -> first 2.00x mean, last 0.00x mean   (DEGENERATE)
      exp     alpha=1.0  -> first 1.58x mean, last 0.58x mean   (~2.7:1)
      sqrt              -> first 1.50x mean, last 0.09x mean   (~16:1)
    """
    c = torch.arange(total_channels, device=device, dtype=dtype)
    if schedule == "uniform":
        w = torch.ones_like(c)
    elif schedule == "linear":
        # w_c = alpha - (2*alpha - 2) * c / (C - 1)  in [2-alpha, alpha]; mean = 1 already.
        # alpha=1 -> uniform; alpha=2 -> last is 0; alpha=1.5 -> first/last = 3.
        if total_channels == 1:
            w = torch.ones_like(c)
        else:
            w = alpha - (2.0 * alpha - 2.0) * c / (total_channels - 1)
    elif schedule == "sqrt":
        w = (total_channels - c).clamp_min(1.0).sqrt()
    elif schedule == "exp":
        w = torch.exp(-alpha * c / max(1, total_channels))
    else:
        raise ValueError(f"unknown channel-weight schedule: {schedule}")
    return w / w.mean()


class NestedDropoutPolicy(Protocol):
    """Decides how many leading channels to keep this step and applies the mask.

    `sample(step)` returns an int in [1, total_channels] meaning "truncate to this many",
    or None meaning "keep all channels (full reconstruction)".
    """

    total_channels: int

    def sample(self, step: int) -> int | None: ...
    def apply(self, z_q: torch.Tensor, c_keep: int | None) -> torch.Tensor: ...


class HybridUniformPolicy:
    """The paper's hybrid scheme — current default.

    With probability `prob` (alpha): truncate, sample c_keep ~ Uniform[1, total_channels].
    Otherwise: keep all channels. Sampling the cut point uniformly trains the decoder at
    every level of detail.

    `prob` is constant in `step` here; subclass or wrap to anneal it.
    """

    def __init__(self, total_channels: int, prob: float, generator: torch.Generator | None = None):
        self.total_channels = total_channels
        self.prob = prob
        self.gen = generator

    def sample(self, step: int) -> int | None:
        if torch.rand((), generator=self.gen).item() >= self.prob:
            return None
        return int(torch.randint(1, self.total_channels + 1, (), generator=self.gen).item())

    def apply(self, z_q: torch.Tensor, c_keep: int | None) -> torch.Tensor:
        """Keep the first c_keep channels, zero the rest. c_keep=None => no-op."""
        if c_keep is None or c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out
