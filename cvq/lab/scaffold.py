"""
cvq's `lab new` scaffolds — idea to runnable component in one command.

`lab new quantizer <name>` / `lab new ar_head <name>` write cvq/models/<name>.py with a
WORKING minimal implementation of the contract (it passes `lab check` immediately —
replace the math, keep the seams) and register it in cvq/models/__init__.py.

(`lab new task <name>` is the framework's own scaffold — a standalone project directory;
see imagelab/lab/scaffold.py.)
"""

from __future__ import annotations

from pathlib import Path

from imagelab.lab.scaffold import register_scaffold

QUANTIZER_T = '''"""{name}: a channel-wise quantizer experiment.

Contract (see cvq/factory.py): forward (B,C,h,w) -> (z_q, idxs (B,C), aux_loss, stats);
truncate(z_q, c_keep) zeroes trailing channels; lookup(idxs) must reproduce forward's
z_q (generation decodes from indices — if these diverge, sampling outputs garbage).

This scaffold is a WORKING nearest-neighbour VQ with a straight-through estimator —
`lab check quantizer {name}` passes as-is. Replace the math, keep the contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imagelab.registry import register


@register("quantizer", "{name}", paper=None)  # cite your source when there is one
class {cls}(nn.Module):
    def __init__(self, token_dim: int, codebook_size: int, commitment_beta: float = 0.25,
                 **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.beta = commitment_beta
        self.embed = nn.Embedding(codebook_size, token_dim)
        self.embed.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        B, C, h, w = z.shape
        flat = z.reshape(B, C, h * w)                              # (B,C,D) channel-tokens
        d = torch.cdist(flat.reshape(-1, h * w), self.embed.weight)  # TODO: your math here
        idxs = d.argmin(-1).reshape(B, C)
        z_q = self.embed(idxs).reshape(B, C, h, w)
        aux = F.mse_loss(z_q, flat.reshape(B, C, h, w).detach()) \\
            + self.beta * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()                               # straight-through
        used = idxs.unique().numel()
        onehot = F.one_hot(idxs.reshape(-1), self.codebook_size).float().mean(0)
        stats = {{"usage": used / self.codebook_size,
                 "perplexity": torch.exp(-(onehot * (onehot + 1e-10).log()).sum()).item(),
                 "quant_error": F.mse_loss(z_q.detach(), z.detach()).item()}}
        return z_q, idxs, aux, stats

    def truncate(self, z_q: torch.Tensor, c_keep: int | None):
        if c_keep is None or c_keep >= z_q.shape[1]:
            return z_q
        mask = torch.zeros_like(z_q)
        mask[:, :c_keep] = 1.0
        return z_q * mask

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor):
        B, C = idxs.shape
        side = int(self.token_dim ** 0.5)
        return self.embed(idxs).reshape(B, C, side, side)
'''

AR_HEAD_T = '''"""{name}: an AR head experiment (consumes backbone hidden states).

Contract (see cvq/models/heads.py): loss(img_hidden (B,C,H), targets (B,C)) ->
(loss, logs with car/ntp_loss + car/token_acc, aux); sample(last_h, u_last_h, *,
temperature, top_k, cfg_scale, do_cfg) -> (B,1) indices.

This scaffold is a working linear softmax head — `lab check ar_head {name}` passes
as-is. Replace the head, keep the contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from imagelab.registry import register


@register("ar_head", "{name}", paper=None)
class {cls}(nn.Module):
    def __init__(self, hidden: int, codebook_size: int, backbone_dtype=torch.float32,
                 **_ignore):
        super().__init__()
        self.K = codebook_size
        self.proj = nn.Linear(hidden, codebook_size)               # TODO: your head here
        self.proj.weight.data.normal_(0, 0.02); self.proj.bias.data.zero_()
        self.proj.to(backbone_dtype)

    def logits(self, img_hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(img_hidden).float()

    def loss(self, img_hidden, targets, channel_weights=None):
        logits = self.logits(img_hidden)
        ce = F.cross_entropy(logits.reshape(-1, self.K), targets.reshape(-1),
                             reduction="none").reshape(targets.shape)
        if channel_weights is not None:
            ce = ce * channel_weights[None, :]
        loss = ce.mean()
        acc = (logits.argmax(-1) == targets).float().mean().item()
        return loss, {{"car/ntp_loss": loss.item(), "car/token_acc": acc}}, \\
            {{"logits": logits}}

    @torch.no_grad()
    def sample(self, last_h, u_last_h, *, temperature=1.0, top_k=0, cfg_scale=1.0,
               do_cfg=False):
        logits = self.logits(last_h[:, None, :])[:, -1]
        if do_cfg:
            u = self.logits(u_last_h[:, None, :])[:, -1]
            logits = u + cfg_scale * (logits - u)
        if temperature > 0:
            probs = (logits / temperature).softmax(-1)
            return torch.multinomial(probs, 1)
        return logits.argmax(-1, keepdim=True)
'''

YAML_KEY = {"quantizer": "model.quant_type", "ar_head": "model.head_type"}


def _write_component(kind: str, template: str, name: str) -> int:
    pkg = Path("cvq/models")
    dest = pkg / f"{name}.py"
    if dest.exists():
        print(f"{dest} already exists — refusing to overwrite")
        return 1
    cls = "".join(w.capitalize() for w in name.split("_")) or name.upper()
    dest.write_text(template.format(name=name, cls=cls))

    init = pkg / "__init__.py"
    lines = init.read_text().splitlines()
    imp = f"from . import {name}  # noqa: F401  registers {kind}/{name}"
    anchor = max((i for i, l in enumerate(lines) if l.startswith("from . import")),
                 default=len(lines) - 1)
    lines.insert(anchor + 1, imp)
    init.write_text("\n".join(lines) + "\n")

    print(f"wrote   {dest}")
    print(f"updated {init} (+ import)")
    print("\nnext:")
    print(f"  1. lab check {kind} {name}            # passes already — now make it yours")
    print(f"  2. set `{YAML_KEY[kind]}: {name}` in a config")
    print(f"  3. lab run <config> --tier overfit    # can it memorize one image?")
    print(f"  4. lab run <config> --tier fast       # is it better than baseline?")
    return 0


@register_scaffold("quantizer")
def new_quantizer(name: str) -> int:
    return _write_component("quantizer", QUANTIZER_T, name)


@register_scaffold("ar_head")
def new_ar_head(name: str) -> int:
    return _write_component("ar_head", AR_HEAD_T, name)
