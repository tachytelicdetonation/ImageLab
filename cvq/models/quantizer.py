"""
Channel-wise IBQ — the project's single quantizer.

Channel-wise VQ (CVQ, arXiv:2605.26089) transposes the usual VQ axis: instead of quantizing
each *spatial location*, each *channel* z^(k) in (h, w) is flattened to an (h*w)-dim vector and
matched against a shared codebook (entry dim = h*w). At 256x256 with f=16 the grid is 16x16, so
token dim = 256 and channels c = 256 (the paper's "256 version").

The assignment/STE here is IBQ — Index Backpropagation Quantization (arXiv:2412.02692, the
discretizer inside EOSTok arXiv:2605.00503). Plain argmin-NN VQ collapsed at this scale (only
the selected code gets gradient -> rich-get-richer dead codes). IBQ instead takes a SOFTMAX over
ALL K codes and straight-throughs the full categorical distribution, so *every* code gets
gradient every step. That is the anti-collapse mechanism that reaches ~100% utilization, which
plain VQ / SimVQ / TransVQ / FVQ / Wasserstein / IBQxTransVQ were all screened against (see
RESULTS.md). Plain IBQ was the robust long-schedule winner, so it is the only quantizer kept.

    logits = z . C                                  # (B,C,K) unnormalized dot product
    p      = softmax(logits / tau)                  # tau=1 reproduces the official quant softmax
    Ind    = onehot(argmax p) + (p - sg[p])         # hard forward, soft gradient (STE)
    z_q    = Ind @ C                                # index-backprop: grad -> encoder + ALL codes
    # Double-quant loss in the OFFICIAL SEED-Voken IBQ form. NB: the released code puts weight 1.0
    # on the COMMITMENT term and beta on the CODEBOOK term -- the transpose of the paper's printed
    # Eq.12. We follow the code, since that is what produced the reported ImageNet numbers.
    L_Q    = ||z_q - z||^2 + ||sg[z_q'] - z||^2 + beta||z_q' - sg[z]||^2          (z_q' = hard@C)
    L_E    = E[H(p_s)] - H(E[p_s]),  p_s = softmax(logits / tau_E)   # tau_E=0.01, sharpened

APR + NTP losses are EOSTok's other contributions and are deferred to phase 2 (they need the CAR).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def _no_autocast(t: torch.Tensor):
    """Keep codebook distance/softmax math in fp32 even under bf16/fp16 autocast."""
    return torch.autocast(device_type="cuda", enabled=False) if t.is_cuda else nullcontext()


class IBQChannelVQ(nn.Module):
    """Channel-wise IBQ. Per CHANNEL-token (dim D=h*w), quantized against a shared codebook
    (K, D) -- K=16384, D=256 is IBQ's native default.

    Logits are the IBQ-canonical UNNORMALIZED dot product z.C (arXiv:2412.02692 Eq.3-4); the
    unbounded magnitude lets the softmax sharpen as the encoder learns, so the index-backprop
    gradient + entropy penalty can actually do their job. `ibq_l2_norm=True` switches to
    EOSTok's cosine variant (Eq.7), which is DEGENERATE at tau=1 with a large K (cosine in
    [-1,1] => softmax stays ~uniform over 16384 codes); if you use it, pass a CLIP-style
    sharpening temperature (e.g. ibq_tau~=0.07).
    """

    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25,
                 ibq_entropy_weight: float = 0.05, ibq_tau: float = 1.0,
                 ibq_l2_norm: bool = False, ibq_reg_weight: float = 1.0,
                 ibq_entropy_temperature: float = 0.01, **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        self.entropy_weight = ibq_entropy_weight
        self.tau = ibq_tau
        self.l2_norm = ibq_l2_norm
        self.reg_weight = ibq_reg_weight
        self.entropy_temperature = ibq_entropy_temperature
        self.embed = nn.Embedding(codebook_size, token_dim)
        nn.init.normal_(self.embed.weight, std=token_dim ** -0.5)

    def forward(self, z: torch.Tensor):
        """z: (B, C, h, w) -> (z_q, idxs (B,C), vq_loss, stats)."""
        B, C, h, w = z.shape
        D = h * w
        assert D == self.token_dim, f"token_dim mismatch: expected {self.token_dim}, got {D}."
        in_dtype = z.dtype
        tok = z.reshape(B, C, D)
        with _no_autocast(tok):
            tok = tok.float()
            cb = self.embed.weight.float()                       # (K, D)
            tn = F.normalize(tok, dim=-1) if self.l2_norm else tok
            cn = F.normalize(cb, dim=-1) if self.l2_norm else cb
            raw_logits = torch.einsum("bcd,kd->bck", tn, cn)     # (B,C,K) unnormalized dot product
            p = (raw_logits / self.tau).softmax(dim=-1)
            idxs = p.argmax(dim=-1)                              # (B,C)
            hard = F.one_hot(idxs, self.codebook_size).type_as(p)
            Ind = hard + (p - p.detach())                        # STE: hard forward, soft grad

            z_q_soft = torch.einsum("bck,kd->bcd", Ind, cb)      # decoder/recon grad path
            z_q_hard = torch.einsum("bck,kd->bcd", hard, cb)     # hard quant (no grad to encoder)

            # Double-quant loss in the OFFICIAL SEED-Voken IBQ ordering: the released code weights
            # the COMMITMENT (encoder) term at 1.0 and the CODEBOOK term at beta -- the transpose of
            # the paper's printed Eq.12. See module docstring.
            term_soft   = F.mse_loss(z_q_soft, tok)              # ||z_q - z||^2 (index-backprop path)
            term_commit = F.mse_loss(z_q_hard.detach(), tok)     # ||sg[z_q'] - z||^2 -> encoder (w=1)
            term_code   = F.mse_loss(z_q_hard, tok.detach())     # ||z_q' - sg[z]||^2 -> codebook (w=beta)
            l_q = term_soft + term_commit + self.commitment_beta * term_code

            # Entropy penalty on a SEPARATELY sharpened softmax (IBQ entropy_temperature, MAGVIT-v2).
            # At tau=1 the K-way softmax is ~uniform and the penalty is inert; the low temperature is
            # what makes "confident per token, uniform across the book" actually exert force.
            e_logits = raw_logits / self.entropy_temperature
            e_logp = e_logits.log_softmax(dim=-1)                # log-space for stability
            e_p = e_logp.exp()
            ent_token = -(e_p * e_logp).sum(-1).mean()           # E[H(p)] (sample entropy)
            e_bar = e_p.mean(dim=(0, 1))                         # marginal usage (K,)
            ent_batch = -(e_bar * (e_bar + 1e-9).log()).sum()    # H(E[p]) (batch entropy)
            l_e = ent_token - ent_batch
            loss = self.reg_weight * l_q + self.entropy_weight * l_e

        z_q = z_q_soft.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)
        stats = self._stats(idxs)
        stats["quant_error"] = term_commit.item()
        stats["entropy_loss"] = l_e.item()
        stats["entropy_per_sample"] = ent_token.item()
        stats["entropy_marginal"] = ent_batch.item()
        stats["avg_maxprob"] = p.max(-1).values.mean().item()
        return z_q, idxs, loss, stats

    @torch.no_grad()
    def _stats(self, idxs: torch.Tensor) -> dict:
        counts = torch.bincount(idxs.reshape(-1), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1)
        nz = probs[probs > 0]
        perplexity = torch.exp(-(nz * nz.log()).sum())
        return {"perplexity": perplexity.item(), "usage": (counts > 0).float().mean().item()}

    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep: int) -> torch.Tensor:
        """Nested channel dropout: keep first c_keep channels, mask the rest to zero."""
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        """(B, C) indices -> (B, C, sqrt(D), sqrt(D)) feature map (for CAR / eval)."""
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        return self.embed(idxs).reshape(B, C, side, side)
