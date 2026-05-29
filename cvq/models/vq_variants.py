"""
Experimental channel-wise VQ variants — anti-collapse MODIFICATIONS on top of the
100% literal plain VQ in `quantizer.py`. Each reimplements a published collapse fix,
adapted to our channel-token setting (each of C channels is a token of dim D=h*w=256,
quantized against a codebook of `codebook_size` entries of dim D).

All variants share the literal baseline's contract:
    forward(z: (B,C,h,w)) -> (z_q (B,C,h,w), idxs (B,C), vq_loss scalar, stats dict)
so they drop into `CVQTokenizer` unchanged via `build_quantizer(quantizer_type=...)`.
`quantizer_type="plain"` returns the untouched literal `ChannelwiseVQ`.

Implemented variants (faithful adaptations; see RESULTS.md for the source papers):
  - "simvq"       SimVQ            (arXiv:2411.02038): frozen random codebook + 1 linear layer
  - "transvq"     TransVQ          (arXiv:2602.18896): frozen codebook + small transformer map
  - "fvq"         FVQ / VQBridge   (arXiv:2509.10140): trainable codebook remapped by a bridge net
  - "wasserstein" Distributional   (arXiv:2506.15078): plain VQ + Bures-Wasserstein feature/code match

The shared mechanism behind simvq/transvq/fvq: the *effective* codebook is a function of a
base codebook, so a gradient on one selected code flows through shared weights into ALL
codes — dead codes can't form. Wasserstein instead adds a global distribution-matching loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .quantizer import ChannelwiseVQ, _no_autocast


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _usage_perplexity(idxs: torch.Tensor, codebook_size: int) -> dict:
    counts = torch.bincount(idxs.reshape(-1), minlength=codebook_size).float()
    probs = counts / counts.sum().clamp_min(1)
    nz = probs[probs > 0]
    perplexity = torch.exp(-(nz * nz.log()).sum())
    return {"perplexity": perplexity.item(), "usage": (counts > 0).float().mean().item()}


class _VariantBase(nn.Module):
    """Provides the nested-dropout truncate + lookup that the tokenizer/CAR expect."""

    codebook_size: int
    token_dim: int

    @staticmethod
    def truncate(z_q: torch.Tensor, c_keep: int) -> torch.Tensor:
        if c_keep >= z_q.shape[1]:
            return z_q
        out = z_q.clone()
        out[:, c_keep:].zero_()
        return out

    def _effective_codebook(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def lookup(self, idxs: torch.Tensor) -> torch.Tensor:
        B, C = idxs.shape
        side = int(round(self.token_dim ** 0.5))
        cb = self._effective_codebook()
        return cb[idxs.reshape(-1)].reshape(B, C, side, side)


# --------------------------------------------------------------------------- #
# SimVQ (arXiv:2411.02038) — frozen codebook + one trainable linear layer
# --------------------------------------------------------------------------- #
class SimVQ(_VariantBase):
    """Effective codebook = W(C) with C a FROZEN random matrix and W a single trainable
    linear layer (no bias). Quantize against W(C); the standard "codebook" loss term then
    trains W (not C). W is identity-initialized so step 0 == plain frozen-codebook VQ.

        k    = argmin_j || z - (C W)_j ||^2
        z_q  = z + sg[(C W)_k - z]
        L    = || sg[z] - (C W)_k ||^2  +  beta || z - sg[(C W)_k] ||^2   (2nd trains encoder)
    """

    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25, **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        # Frozen random codebook (buffer -> never optimized, moves with .to(device)).
        cb = torch.randn(codebook_size, token_dim) * (token_dim ** -0.5)
        self.register_buffer("frozen_codebook", cb)
        # The single trainable linear layer, init = identity so C@W == C at start.
        self.proj = nn.Linear(token_dim, token_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def _effective_codebook(self) -> torch.Tensor:
        return self.proj(self.frozen_codebook)

    def forward(self, z: torch.Tensor):
        B, C, h, w = z.shape
        D = h * w
        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()
        with _no_autocast(flat):
            cb = self._effective_codebook().float()            # (K, D), grad -> proj only
            dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * (flat @ cb.t()) + cb.pow(2).sum(1)
            idxs = dist.detach().argmin(dim=1)
            quant = cb[idxs]                                    # (B*C, D)
            codebook_loss = F.mse_loss(quant, flat.detach())    # trains proj (C frozen)
            commit = F.mse_loss(flat, quant.detach())           # trains encoder
            loss = codebook_loss + self.commitment_beta * commit
            quant = flat + (quant - flat).detach()              # STE
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)
        stats = _usage_perplexity(idxs, self.codebook_size)
        stats["quant_error"] = commit.item()
        return z_q, idxs, loss, stats


# --------------------------------------------------------------------------- #
# TransVQ (arXiv:2602.18896) — frozen codebook + small transformer remap
# --------------------------------------------------------------------------- #
class _LinearAttention(nn.Module):
    """O(K) linear attention (Performer elu+1 feature map) so it scales to K=16384 codes
    as a single token sequence (softmax attention would be O(K^2))."""

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):                                       # x: (1, K, dim)
        b, n, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bnd,bne->bde", k, v)                 # (1, d, d)
        z = 1.0 / (torch.einsum("bnd,bd->bn", q, k.sum(dim=1)) + 1e-6)
        out = torch.einsum("bnd,bde,bn->bne", q, kv, z)
        return self.to_out(out)


class _TxBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 1, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _LinearAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _CodebookTransformer(nn.Module):
    def __init__(self, frozen_dim: int, out_dim: int, codebook_size: int,
                 depth: int = 1, model_dim: int = 256, heads: int = 1, mlp_ratio: float = 2.0):
        super().__init__()
        self.in_proj = nn.Linear(frozen_dim, model_dim, bias=False)
        self.pos = nn.Embedding(codebook_size, model_dim)
        self.blocks = nn.ModuleList([_TxBlock(model_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, out_dim, bias=False)

    def forward(self, C):                                       # C: (K, frozen_dim)
        K = C.shape[0]
        x = self.in_proj(C).unsqueeze(0)                        # (1, K, model_dim)
        x = x + self.pos(torch.arange(K, device=C.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.out_proj(self.norm(x)).squeeze(0)           # (K, out_dim)


class TransVQ(_VariantBase):
    """Effective codebook C' = P_phi(C) with C frozen and P_phi a 1-layer linear-attention
    transformer (codes-as-tokens). Standard VQ machinery runs against C'. Because every row
    of C' shares phi, one step moves the whole codebook -> no dead codes (preserves k-means
    expressivity since P_phi is a universal map of C)."""

    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25,
                 transvq_depth: int = 1, transvq_model_dim: int = 256,
                 transvq_frozen_dim: int = 256, **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        cb = torch.randn(codebook_size, transvq_frozen_dim) * (transvq_frozen_dim ** -0.5)
        self.register_buffer("frozen_codebook", cb)
        self.transform = _CodebookTransformer(
            transvq_frozen_dim, token_dim, codebook_size,
            depth=transvq_depth, model_dim=transvq_model_dim,
        )

    def _effective_codebook(self) -> torch.Tensor:
        return self.transform(self.frozen_codebook)

    def forward(self, z: torch.Tensor):
        B, C, h, w = z.shape
        D = h * w
        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()
        with _no_autocast(flat):
            cb = self._effective_codebook().float()             # (K, D) recomputed each step
            dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * (flat @ cb.t()) + cb.pow(2).sum(1)
            idxs = dist.detach().argmin(dim=1)
            quant = cb[idxs]
            codebook_loss = F.mse_loss(quant, flat.detach())     # trains phi
            commit = F.mse_loss(flat, quant.detach())            # trains encoder
            loss = codebook_loss + self.commitment_beta * commit
            quant = flat + (quant - flat).detach()
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)
        stats = _usage_perplexity(idxs, self.codebook_size)
        stats["quant_error"] = commit.item()
        return z_q, idxs, loss, stats


# --------------------------------------------------------------------------- #
# FVQ / VQBridge (arXiv:2509.10140) — trainable codebook remapped by a bridge net
# --------------------------------------------------------------------------- #
class _ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):                                       # x: (1, p, dim)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _VQBridge(nn.Module):
    """Compress -> ViT -> recover on the WHOLE codebook. C (K,D) -> Chat (K,D).
    p groups of K/p codes; each group flattened to (K/p*D) and projected to d'."""

    def __init__(self, codebook_size: int, dim: int, groups: int = 16,
                 d_prime: int | None = None, n_blocks: int = 2, heads: int = 4):
        super().__init__()
        assert codebook_size % groups == 0, "codebook_size must be divisible by groups"
        self.K, self.D, self.p = codebook_size, dim, groups
        self.g = codebook_size // groups
        d_prime = d_prime or dim
        self.compress = nn.Linear(self.g * dim, d_prime)
        self.ln_in = nn.LayerNorm(d_prime)
        self.blocks = nn.ModuleList([_ViTBlock(d_prime, heads) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d_prime)
        self.expand = nn.Linear(d_prime, self.g * dim)

    def forward(self, C):                                       # C: (K, D)
        x = C.reshape(self.p, self.g * self.D)                  # (p, g*D) group-flatten
        H = self.ln_in(self.compress(x)).unsqueeze(0)           # (1, p, d')
        for blk in self.blocks:
            H = blk(H)
        H = self.expand(self.ln_out(H.squeeze(0)))              # (p, g*D)
        return H.reshape(self.K, self.D)                        # (K, D) = Chat


class FVQ(_VariantBase):
    """Trainable codebook C remapped by VQBridge: Chat = bridge(C), quantize against Chat.
    The bridge is a shared net over all codes, so the per-step codebook gradient reaches
    every code -> 100% utilization is structural (no dead-code resets). Bridge is dropped
    at inference (bake Chat)."""

    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25,
                 fvq_groups: int = 16, fvq_blocks: int = 2, **_ignore):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_beta = commitment_beta
        self.embed = nn.Embedding(codebook_size, token_dim)
        self.embed.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.bridge = _VQBridge(codebook_size, token_dim, groups=fvq_groups, n_blocks=fvq_blocks)

    def _effective_codebook(self) -> torch.Tensor:
        return self.bridge(self.embed.weight)

    def forward(self, z: torch.Tensor):
        B, C, h, w = z.shape
        D = h * w
        in_dtype = z.dtype
        flat = z.reshape(B * C, D).float()
        with _no_autocast(flat):
            cb = self._effective_codebook().float()             # (K, D) grad -> C + bridge
            dist = flat.pow(2).sum(1, keepdim=True) - 2.0 * (flat @ cb.t()) + cb.pow(2).sum(1)
            idxs = dist.detach().argmin(dim=1)
            quant = cb[idxs]
            codebook_loss = F.mse_loss(quant, flat.detach())
            commit = F.mse_loss(flat, quant.detach())
            loss = codebook_loss + self.commitment_beta * commit
            quant = flat + (quant - flat).detach()
        z_q = quant.reshape(B, C, h, w).to(in_dtype)
        idxs = idxs.reshape(B, C)
        stats = _usage_perplexity(idxs, self.codebook_size)
        stats["quant_error"] = commit.item()
        return z_q, idxs, loss, stats


# --------------------------------------------------------------------------- #
# Distributional matching (arXiv:2506.15078) — plain VQ + Bures-Wasserstein loss
# --------------------------------------------------------------------------- #
def _sym_sqrt(S: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Differentiable symmetric matrix square root via eigendecomposition (autograd-stable
    for SPD matrices)."""
    S = 0.5 * (S + S.transpose(-1, -2))
    w, V = torch.linalg.eigh(S)
    w = torch.clamp(w, min=0.0) + eps
    return (V * w.sqrt().unsqueeze(-2)) @ V.transpose(-1, -2)


def _mean_cov(X: torch.Tensor, eps_cov: float = 1e-4):
    N, D = X.shape
    mu = X.mean(dim=0)
    Xc = X - mu
    cov = (Xc.t() @ Xc) / max(N - 1, 1)
    cov = cov + eps_cov * torch.eye(D, device=X.device, dtype=X.dtype)
    return mu, cov


def _bures_w2(mu1, S1, mu2, S2, eps: float = 1e-5):
    """Squared 2-Wasserstein between Gaussians N(mu1,S1), N(mu2,S2) (Bures / FID form)."""
    mean_term = ((mu1 - mu2) ** 2).sum()
    s1 = _sym_sqrt(S1, eps)
    cross = _sym_sqrt(s1 @ S2 @ s1, eps)
    cov_term = torch.trace(S1) + torch.trace(S2) - 2.0 * torch.trace(cross)
    return torch.clamp(mean_term + cov_term, min=0.0)


class WassersteinVQ(ChannelwiseVQ):
    """Plain channel-wise VQ + a global Bures-Wasserstein term that matches the Gaussian
    fit of the encoder feature tokens to that of the codebook vectors (gradient flows to
    BOTH encoder and all codes -> revives dead codes). L_W has NO stop-gradient (per paper).
        L = L_vq + gamma * sqrt(W2^2(features, codes))
    """

    def __init__(self, codebook_size: int = 16384, token_dim: int = 256,
                 commitment_beta: float = 0.25,
                 wasserstein_weight: float = 0.5, w_eps_cov: float = 1e-4,
                 w_eps_sqrt: float = 1e-5, **_ignore):
        super().__init__(codebook_size=codebook_size, token_dim=token_dim,
                         commitment_beta=commitment_beta)
        self.wasserstein_weight = wasserstein_weight
        self.w_eps_cov = w_eps_cov
        self.w_eps_sqrt = w_eps_sqrt

    def forward(self, z: torch.Tensor):
        z_q, idxs, loss, stats = super().forward(z)             # plain VQ (entropy off)
        B, C, h, w = z.shape
        D = h * w
        flat = z.reshape(B * C, D).float()
        with _no_autocast(flat):
            mu_f, cov_f = _mean_cov(flat, self.w_eps_cov)        # P_A: features
            mu_c, cov_c = _mean_cov(self.embed.weight.float(), self.w_eps_cov)  # P_B: codes
            w2 = _bures_w2(mu_f, cov_f, mu_c, cov_c, self.w_eps_sqrt)
            lw = torch.sqrt(w2 + 1e-12)
        loss = loss + self.wasserstein_weight * lw
        stats["wasserstein"] = lw.item()
        return z_q, idxs, loss, stats


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "simvq": SimVQ,
    "transvq": TransVQ,
    "fvq": FVQ,
    "wasserstein": WassersteinVQ,
}


def build_quantizer(quantizer_type: str = "plain", **kwargs):
    """Return the literal ChannelwiseVQ for 'plain'/'channelwise', else an experimental
    variant. All classes accept **_ignore, so a superset of kwargs may be passed."""
    qt = (quantizer_type or "plain").lower()
    if qt in ("plain", "channelwise", "literal"):
        return ChannelwiseVQ(**kwargs)
    if qt not in _REGISTRY:
        raise ValueError(f"unknown quantizer_type '{quantizer_type}'. "
                         f"options: plain, {', '.join(_REGISTRY)}")
    return _REGISTRY[qt](**kwargs)
