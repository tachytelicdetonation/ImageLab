"""
Muon / Pion optimizer (single-GPU), for the image tokenizer — an EXPERIMENTAL optimizer
swap, not part of the faithful CVQ recipe.

Muon (Jordan et al.): for 2D weight matrices, orthogonalize the momentum-smoothed gradient
via a Newton-Schulz quintic (≈ msign), then take a spectrally-normalized step. Embeddings,
biases, and norm γ/β do NOT go through Muon — they use AdamW (this hybrid is the standard
Muon usage). Conv weights (4D) are reshaped to 2D (out, in·kh·kw) before orthogonalization.

Pion (arXiv:2605.19282, repo github.com/OPTML-Group/Pion): replaces Muon's msign polynomial
with a two-stage "high-pass" filter — a promotion polynomial then a suppression polynomial
that drives SMALL singular values to 0 (keeps the informative head, kills the noisy tail).
The paper motivates it for VLA / RLVR (low-rank / low-SNR gradients) and makes NO claim it
helps dense supervised training — so for our reconstruction tokenizer, base Muon is the
faithful default and Pion is a flagged ablation.

Math transcribed verbatim from the repo (coefficients, EMA+Nesterov momentum, decoupled WD,
scale = scale_factor·√(max(out,in))). Newton-Schulz runs in bf16.
"""

from __future__ import annotations

import torch
from torch import Tensor


def muon_ns(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Base Muon Newton-Schulz orthogonalization (quintic). 2D input only."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    if transposed:
        X = X.mT
    return X


def high_pass_ns(G: Tensor, promotion_steps: int, suppression_steps: int,
                 eps: float = 1e-7) -> Tensor:
    """Pion two-stage high-pass filter on the singular values.
    Promotion f_p(s)=1.875s−1.25s³+0.375s⁵ (monotone amplify); then
    suppression f_s(s)=2.5s³−1.5s⁵ (no linear term → small s driven to 0)."""
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    for _ in range(promotion_steps):
        A = X @ X.mT
        B = -1.25 * A + 0.375 * (A @ A)
        X = 1.875 * X + B @ X
    for _ in range(suppression_steps):
        A = X @ X.mT
        B = 2.5 * A - 1.5 * (A @ A)
        X = B @ X
    if transposed:
        X = X.mT
    return X


class MuonAdamW(torch.optim.Optimizer):
    """Hybrid optimizer. Each param group carries `method` in {muon, pion, adamw}:
      muon/pion -> orthogonalized matrix update (2D weights, conv reshaped)
      adamw     -> standard AdamW (embeddings, heads, norm γ/β, biases)
    Build groups with `build_muon_groups`."""

    def __init__(self, param_groups):
        defaults = dict(method="adamw", lr=1e-3, weight_decay=0.0,
                        momentum=0.95, nesterov=True, ns_steps=5, promotion_steps=0,
                        scale_factor=2.0, betas=(0.9, 0.95), eps=1e-8)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["method"] == "adamw":
                self._adamw(group)
            else:
                self._muon_like(group)

    def _muon_like(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        mom, nesterov = group["momentum"], group["nesterov"]
        k, kp = group["ns_steps"], group["promotion_steps"]
        ks = max(0, k - kp)
        sf, method = group["scale_factor"], group["method"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(g)
            buf = st["momentum_buffer"]
            buf.lerp_(g, 1 - mom)                       # EMA: M = mom·M + (1-mom)·g
            g = g.lerp(buf, mom) if nesterov else buf   # Nesterov look-ahead (out-of-place)

            orig = g.shape
            if g.ndim == 4:                             # conv -> (out, in·kh·kw)
                g = g.reshape(g.size(0), -1)
            o = high_pass_ns(g, kp, ks) if method == "pion" else muon_ns(g, steps=k)
            o = o.to(p.dtype).reshape(orig)

            scale = sf * (max(o.size(-2), o.size(-1)) ** 0.5) if o.ndim >= 2 \
                else sf * (o.numel() ** 0.5)
            p.mul_(1 - lr * wd)                         # decoupled weight decay
            p.add_(o, alpha=-lr * scale)

    def _adamw(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        b1, b2 = group["betas"]
        eps = group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[p]
            if "step" not in st:
                st["step"] = 0
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            st["step"] += 1
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.lerp_(g, 1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            bc1 = 1 - b1 ** st["step"]
            bc2 = 1 - b2 ** st["step"]
            p.mul_(1 - lr * wd)
            p.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(eps), value=-lr)


def build_muon_groups(named_params, method: str, muon_lr: float, adamw_lr: float,
                      weight_decay: float, momentum: float = 0.95,
                      ns_steps: int = 5, promotion_steps: int = 0, scale_factor: float = 2.0):
    """Partition (name, param) pairs into Muon/Pion (2D non-embedding weights + conv) and
    AdamW (embeddings, norm γ/β, biases). AdamW further splits decay (embeddings) vs
    no-decay (ndim<2). Returns optimizer param groups for MuonAdamW."""
    muon, adamw_decay, adamw_nodecay = [], [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        is_embed = ("embed" in name.lower() or ".pos" in name.lower()
                    or name.lower().endswith("pos.weight"))
        if p.ndim >= 2 and not is_embed:
            muon.append(p)
        elif p.ndim >= 2:                              # embeddings/codebook -> AdamW (decayed)
            adamw_decay.append(p)
        else:                                          # norm γ/β, biases -> AdamW (no decay)
            adamw_nodecay.append(p)
    groups = []
    if muon:
        groups.append(dict(params=muon, method=method, lr=muon_lr, weight_decay=weight_decay,
                           momentum=momentum, nesterov=True, ns_steps=ns_steps,
                           promotion_steps=promotion_steps, scale_factor=scale_factor))
    if adamw_decay:
        groups.append(dict(params=adamw_decay, method="adamw", lr=adamw_lr,
                           weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8))
    if adamw_nodecay:
        groups.append(dict(params=adamw_nodecay, method="adamw", lr=adamw_lr,
                           weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8))
    return groups
