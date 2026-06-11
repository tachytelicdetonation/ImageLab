"""
cvq's `lab check` suites — the component contracts as instant probes.

The moment you save a new component, these answer "is it wired right?" in seconds:
shapes, index ranges, the truncate/lookup contract, and — the classic silent killer —
whether decode-from-indices reproduces the train-time z_q and whether the straight-
through gradient actually reaches the encoder. Every probe failure here is a bug class
that otherwise survives until a full run produces noise.

    lab check quantizer lfq
    lab check quantizer fsq --kwargs "levels: [2,2,2,2,2,2]"
    lab check ar_head mbm

These register into imagelab's generic harness (imagelab/lab/probes.py) — they're the
worked example of a project defining contracts for its OWN component kinds.
"""

from __future__ import annotations

import torch

from imagelab.lab.probes import Probe, assert_ as _assert, register_checker


def _registered(kind: str, name: str):
    import cvq.models, cvq.tasks  # noqa: F401  — register builtins + user modules
    from imagelab.registry import get
    try:
        return get(kind, name)
    except KeyError as e:
        print(e.args[0])
        print("(new module? add `from . import yourmodule` to cvq/models/__init__.py)")
        return None


# --------------------------------------------------------------------------- #
@register_checker("quantizer")
def check_quantizer(name: str, kwargs: dict) -> int:
    from imagelab.registry import build
    if _registered("quantizer", name) is None:
        return 1
    p = Probe()
    kw = {"token_dim": 16, "codebook_size": 64, **kwargs}
    B, C, s = 2, 8, 4  # token_dim = s*s = 16
    box = {}

    def construct():
        try:
            box["q"] = build("quantizer", name, **kw)
        except TypeError as e:
            raise TypeError(f"{e} — constructors take (token_dim, codebook_size, "
                            f"**kwargs); pass extra args via --kwargs 'lvl: ...'") from e
    p.check(f"constructs with {kw}", construct)
    if "q" not in box:
        return p.done("quantizer", name)
    q = box["q"]
    z = torch.randn(B, C, s, s, requires_grad=True)
    out = {}

    def forward():
        z_q, idxs, loss, stats = q(z)
        assert z_q.shape == z.shape, f"z_q {tuple(z_q.shape)} != z {tuple(z.shape)}"
        out.update(z_q=z_q, idxs=idxs, loss=loss, stats=stats)
    p.check("forward(z (B,C,h,w)) -> (z_q, idxs, aux_loss, stats)", forward)
    if not out:
        return p.done("quantizer", name)

    p.check("indices are (B,C) ints within [0, codebook_size)", lambda: (
        _assert(out["idxs"].shape == (B, C), f"idxs shape {tuple(out['idxs'].shape)}"),
        _assert(not out["idxs"].dtype.is_floating_point, "idxs must be integer"),
        _assert(int(out["idxs"].max()) < q.codebook_size, "index out of range")))
    p.check("stats has usage + perplexity", lambda: _assert(
        {"usage", "perplexity"} <= set(out["stats"]), f"got {sorted(out['stats'])}"))
    p.check("truncate(z_q, 3) zeroes tail channels, keeps head", lambda: (
        _assert(q.truncate(out["z_q"], 3)[:, 3:].abs().sum() == 0, "tail not zeroed"),
        _assert(q.truncate(out["z_q"], 3)[:, :3].equal(out["z_q"][:, :3]),
                "head channels changed")))
    p.check("lookup(idxs) -> (B,C,h,w) feature map", lambda: _assert(
        q.lookup(out["idxs"]).shape == (B, C, s, s),
        f"got {tuple(q.lookup(out['idxs']).shape)}"))
    p.check("lookup(idxs) == forward z_q (decode must reproduce training)", lambda: (
        _assert(torch.allclose(q.lookup(out["idxs"]), out["z_q"].detach(), atol=1e-4),
                f"max err {(q.lookup(out['idxs']) - out['z_q']).abs().max():.4f} — "
                f"generation would decode DIFFERENT features than training produced")))

    def ste():
        z2 = torch.randn(B, C, s, s, requires_grad=True)
        z_q2, *_ = q(z2)
        z_q2.sum().backward()
        _assert(z2.grad is not None and z2.grad.abs().sum() > 0,
                "no gradient reached the quantizer input — the encoder would never train")
    p.check("straight-through gradient flows to the input", ste)
    return p.done("quantizer", name)


@register_checker("ar_head")
def check_ar_head(name: str, kwargs: dict) -> int:
    from imagelab.registry import build
    if _registered("ar_head", name) is None:
        return 1
    p = Probe()
    kw = {"hidden": 32, "codebook_size": 64, "backbone_dtype": torch.float32,
          "mbm_depth": 1, "mbm_heads": 2, "mbm_infer_steps": 2, **kwargs}
    B, C = 2, 8
    box = {}
    p.check("constructs with {hidden, codebook_size, backbone_dtype, ...}",
            lambda: box.update(h=build("ar_head", name, **kw)))
    if "h" not in box:
        return p.done("ar_head", name)
    h = box["h"]
    hid = torch.randn(B, C, 32)
    tgt = torch.randint(0, 64, (B, C))
    out = {}

    def loss():
        l, logs, aux = h.loss(hid, tgt)
        assert l.requires_grad, "loss tensor does not require grad"
        out.update(l=l, logs=logs, aux=aux)
    p.check("loss(img_hidden, targets) -> (loss, logs, aux)", loss)
    if out:
        p.check("logs include car/ntp_loss + car/token_acc", lambda: _assert(
            {"car/ntp_loss", "car/token_acc"} <= set(out["logs"]),
            f"got {sorted(out['logs'])}"))

    def sample():
        idx = h.sample(torch.randn(B, 32), torch.randn(B, 32),
                       temperature=1.0, top_k=0, cfg_scale=2.0, do_cfg=True)
        _assert(idx.shape == (B, 1), f"sample shape {tuple(idx.shape)}, want (B,1)")
        _assert(int(idx.max()) < 64, "sampled index out of codebook range")
    p.check("sample(last_h, u_last_h, cfg) -> (B,1) valid indices", sample)
    return p.done("ar_head", name)


@register_checker("encoder")
def check_encoder(name: str, kwargs: dict) -> int:
    from imagelab.registry import build
    if _registered("encoder", name) is None:
        return 1
    p = Probe()
    kw = {"ch": 32, "ch_mult": (1, 2), "num_res_blocks": 1, "z_channels": 8,
          "resolution": 32, "attn_resolutions": (16,), **kwargs}
    box = {}
    p.check("constructs with factory kwargs", lambda: box.update(
        e=build("encoder", name, **kw)))
    if "e" in box:
        p.check("forward (B,3,32,32) -> (B, z_channels, 16, 16)", lambda: _assert(
            box["e"](torch.randn(2, 3, 32, 32)).shape == (2, 8, 16, 16),
            f"got {tuple(box['e'](torch.randn(2, 3, 32, 32)).shape)}"))
    return p.done("encoder", name)


@register_checker("decoder")
def check_decoder(name: str, kwargs: dict) -> int:
    from imagelab.registry import build
    if _registered("decoder", name) is None:
        return 1
    p = Probe()
    kw = {"ch": 32, "out_ch": 3, "ch_mult": (1, 2), "num_res_blocks": 1,
          "z_channels": 8, "resolution": 32, "attn_resolutions": (16,), **kwargs}
    box = {}
    p.check("constructs with factory kwargs", lambda: box.update(
        d=build("decoder", name, **kw)))
    if "d" in box:
        out = {}

        def forward():
            out["y"] = box["d"](torch.randn(2, 8, 16, 16))
            _assert(out["y"].shape == (2, 3, 32, 32), f"got {tuple(out['y'].shape)}")
        p.check("forward (B,z,16,16) -> (B,3,32,32)", forward)
        if "y" in out:
            p.check("output in [-1,1] (the recon target range)", lambda: _assert(
                out["y"].abs().max() <= 1.001, f"max |out| = {out['y'].abs().max():.3f}"))
    return p.done("decoder", name)
