"""Fast CPU regression tests for the experimentation seams: registry, config validation,
quantizers, AR heads, tokenizer composition, eval grouping. No network, no datasets.

    uv run pytest tests/ -q
"""

from __future__ import annotations

import pytest
import torch

from cvq.config import ConfigError, from_dict
from cvq.registry import available, build, get, register


def base_cfg(**model_overrides):
    model = {
        "latent_channels": 16, "codebook_size": 64,
        "enc_ch": 32, "enc_ch_mult": [1, 2], "decoder_ch": 32,   # GroupNorm needs ch % 32 == 0
        "decoder_ch_mult": [1, 2], "decoder_res_blocks": 1,
    }
    model.update(model_overrides)
    return {
        "data": {"root": "", "size": 32, "hflip": False},
        "model": model,
        "train": {"batch_size": 2, "epochs": 1, "lr": 1e-4},
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_has_builtins():
    import cvq.models  # noqa: F401 — triggers registration
    assert set(available("quantizer")) >= {"ibq", "fsq"}
    assert set(available("ar_head")) >= {"softmax", "mbm"}


def test_registry_unknown_name_lists_known():
    import cvq.models  # noqa: F401
    with pytest.raises(KeyError, match="ibq"):
        get("quantizer", "does-not-exist")


def test_registry_register_and_build():
    @register("quantizer", "_test_dummy")
    class Dummy:
        def __init__(self, token_dim, **_ignore):
            self.token_dim = token_dim

    assert build("quantizer", "_test_dummy", token_dim=7, codebook_size=4).token_dim == 7


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_config_loads_with_defaults():
    rc = from_dict(base_cfg())
    assert rc.model.quant_type == "ibq"
    assert rc.train.resolved_car_lr() == rc.train.lr


def test_config_rejects_mbm_on_ibq():
    with pytest.raises(ConfigError, match="mbm"):
        from_dict(base_cfg(head_type="mbm"))  # quant_type defaults to ibq


def test_config_rejects_fsq_without_levels():
    with pytest.raises(ConfigError, match="fsq"):
        from_dict(base_cfg(quant_type="fsq"))


def test_config_rejects_apr_on_forkB():
    cfg = base_cfg(quant_type="fsq", fsq_bits=6, head_type="mbm")
    cfg["train"]["lambda_apr"] = 0.1
    with pytest.raises(ConfigError, match="lambda_apr"):
        from_dict(cfg)


def test_config_rejects_bad_geometry():
    cfg = base_cfg()
    cfg["data"]["size"] = 33  # not divisible by f=2
    with pytest.raises(ConfigError, match="divisible"):
        from_dict(cfg)


def test_config_warns_on_unknown_key(capsys):
    cfg = base_cfg()
    cfg["train"]["lamda_ntp"] = 2.0  # typo
    from_dict(cfg)
    assert "lamda_ntp" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# quantizers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kwargs", [
    ("ibq", {"codebook_size": 64, "commitment_beta": 0.25}),
    ("fsq", {"codebook_size": 64, "levels": [2] * 6}),
])
def test_quantizer_contract(name, kwargs):
    import cvq.models  # noqa: F401
    torch.manual_seed(0)
    q = build("quantizer", name, token_dim=16, **kwargs)
    z = torch.randn(2, 8, 4, 4)
    z_q, idxs, loss, stats = q(z)
    assert z_q.shape == z.shape
    assert idxs.shape == (2, 8)
    assert idxs.max() < q.codebook_size
    assert "usage" in stats and "perplexity" in stats
    # truncate zeroes the tail channels
    t = q.truncate(z_q, 3)
    assert t[:, 3:].abs().sum() == 0 and t[:, :3].equal(z_q[:, :3])
    # lookup maps indices back to a (B, C, side, side) map
    assert q.lookup(idxs).shape == (2, 8, 4, 4)


def test_fsq_lookup_matches_forward():
    """FSQ decode-from-indices must reproduce the train-time z_q (the BAR fidelity note)."""
    import cvq.models  # noqa: F401
    torch.manual_seed(0)
    q = build("quantizer", "fsq", token_dim=16, codebook_size=0, levels=[2] * 6)
    z = torch.randn(2, 8, 4, 4)
    z_q, idxs, _, _ = q(z)
    assert torch.allclose(q.lookup(idxs), z_q, atol=1e-5)


# --------------------------------------------------------------------------- #
# tokenizer composition via factory
# --------------------------------------------------------------------------- #
def test_factory_builds_tokenizer_and_forward():
    from cvq.factory import build_tokenizer
    torch.manual_seed(0)
    tok, _ = build_tokenizer(base_cfg(), "cpu")
    x = torch.rand(2, 3, 32, 32) * 2 - 1
    out = tok(x, c_keep=4)
    assert out["recon"].shape == x.shape
    assert out["indices"].shape == (2, 16)
    assert out["z"].shape == (2, 16, 16, 16)  # grid = 32 / 2


# --------------------------------------------------------------------------- #
# AR heads (no backbone needed — they consume hidden states directly)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["softmax", "mbm"])
def test_ar_head_contract(name):
    import cvq.models  # noqa: F401
    torch.manual_seed(0)
    head = build("ar_head", name, hidden=32, codebook_size=64,
                 backbone_dtype=torch.float32, mbm_depth=1, mbm_heads=2,
                 mbm_infer_steps=2)
    B, C = 2, 8
    hid = torch.randn(B, C, 32)
    tgt = torch.randint(0, 64, (B, C))
    loss, logs, aux = head.loss(hid, tgt)
    assert loss.requires_grad
    assert "car/ntp_loss" in logs and "car/token_acc" in logs
    if name == "softmax":
        assert aux["logits"].shape == (B, C, 64)
    idx = head.sample(torch.randn(B, 32), torch.randn(B, 32),
                      temperature=1.0, top_k=0, cfg_scale=2.0, do_cfg=True)
    assert idx.shape == (B, 1) and idx.max() < 64


# --------------------------------------------------------------------------- #
# grouped evaluator
# --------------------------------------------------------------------------- #
def test_grouped_evaluator_splits_by_tag(tmp_path):
    from cvq.eval.evaluator import GroupedEvaluator

    class FakeDS(torch.utils.data.Dataset):
        records = ([{"dataset": "easy"}] * 4) + ([{"dataset": "hard"}] * 4)

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            return {"image": torch.zeros(3, 8, 8) + i}

    ev = GroupedEvaluator(FakeDS(), device="cpu", sample_dir=tmp_path, n_eval=2)
    assert ev.group_names() == ["easy", "hard"]
    assert ev.multi
    assert ev.eval_batches["easy"]["image"].shape[0] == 2
    # untagged datasets collapse to a single "all" group
    class Untagged(FakeDS):
        records = [{}] * 4
    ev2 = GroupedEvaluator(Untagged(), device="cpu", sample_dir=tmp_path)
    assert ev2.group_names() == ["all"] and not ev2.multi
