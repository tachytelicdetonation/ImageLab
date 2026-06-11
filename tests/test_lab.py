"""Fast CPU regression tests for the framework seams: registry, data utilities,
tiers/overrides/naming, run dirs + ledger, gate verdicts. No network, no datasets.

    uv run pytest tests/ -q
"""

from __future__ import annotations

import pytest
import torch

from imagelab.registry import available, build, get, register


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_registry_register_and_build():
    @register("quantizer", "_test_dummy")
    class Dummy:
        def __init__(self, token_dim, **_ignore):
            self.token_dim = token_dim

    assert build("quantizer", "_test_dummy", token_dim=7, codebook_size=4).token_dim == 7
    assert "_test_dummy" in available("quantizer")


def test_registry_unknown_name_lists_known():
    @register("quantizer", "_test_known")
    class Known:
        pass

    with pytest.raises(KeyError, match="_test_known"):
        get("quantizer", "does-not-exist")


# --------------------------------------------------------------------------- #
# data: split + overfit clamp
# --------------------------------------------------------------------------- #
def test_record_split_is_stable_and_per_item():
    from imagelab.data.dataset import record_split
    recs = [{"file": f"img_{i:04d}.png"} for i in range(1000)]
    splits = [record_split(r, 0.1) for r in recs]
    frac = splits.count("val") / len(splits)
    assert 0.05 < frac < 0.15                       # hash buckets land near the fraction
    # per-item stability: growing the dataset must never reassign existing records
    assert [record_split(r, 0.1) for r in recs[:100]] == splits[:100]
    # explicit manifest split wins over the hash
    assert record_split({"file": "x.png", "split": "val"}, 0.0) == "val"
    # val_fraction=0 -> everything trains (legacy behavior)
    assert all(record_split(r, 0.0) == "train" for r in recs)


def test_overfit_dataset_clamps_and_repeats():
    from imagelab.data.dataset import OverfitDataset

    class DS(torch.utils.data.Dataset):
        records = [{"file": f"{i}.png", "dataset": "all"} for i in range(10)]

        def __len__(self):
            return 10

        def __getitem__(self, i):
            return i

    ds = OverfitDataset(DS(), n=1, repeat_to=64)
    assert len(ds) == 64
    assert {ds[i] for i in range(64)} == {0}        # one image, repeated
    assert len(ds.records) == 64                    # evaluator-visible view matches


# --------------------------------------------------------------------------- #
# lab layer: tiers, overrides, naming, ledger
# --------------------------------------------------------------------------- #
def base_cfg():
    return {
        "data": {"root": "", "size": 32, "hflip": True},
        "model": {"width": 32},
        "train": {"batch_size": 2, "epochs": 1, "lr": 1e-4},
    }


def test_tier_overfit_transforms_config():
    from imagelab.lab.tiers import apply_overrides, apply_tier
    cfg = base_cfg()
    steps = apply_tier(cfg, "overfit")
    assert steps == 300
    assert cfg["train"]["overfit_n"] == 1
    assert cfg["data"]["hflip"] is False
    # explicit --set wins over the tier
    deltas = apply_overrides(cfg, ["train.overfit_n=4", "model.width=64"])
    assert cfg["train"]["overfit_n"] == 4 and deltas["model.width"] == 64


def test_auto_name_encodes_deltas():
    from imagelab.lab.tiers import auto_name
    n = auto_name("tok_inet", "overfit", {"train.lr": 0.0002, "model.quant_type": "lfq"})
    assert n == "tok_inet_overfit_lr-0.0002_quant_type-lfq"


def test_rundir_ledger_roundtrip(tmp_path):
    from imagelab.lab.rundir import RunDir, ledger_rows
    rd = RunDir.create("test_run", root=tmp_path)
    rd.start(name="test_run", task="tokenizer", config_path="x.yaml", tier="smoke",
             deltas={"train.lr": 1e-4}, resolved={"task": "tokenizer", "train": {}},
             seed=0, device="cpu", amp="none",
             key_metrics=["val/recon_l2_full"], higher_is_better=[])
    assert (rd.dir / "config.yaml").exists()
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "running"
    rd.finish(status="done", steps=8, final_metrics={"val/recon_l2_full": 0.5},
              params={"tok": 1.2})
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1                            # last row per run_id wins
    assert rows[0]["status"] == "done"
    assert rows[0]["final_metrics"]["val/recon_l2_full"] == 0.5
    assert rows[0]["key_metrics"] == ["val/recon_l2_full"]   # declarations travel with the run
    meta = (rd.dir / "meta.json").read_text()
    assert "git" in meta and "steps_per_sec" in meta


def test_ledger_skips_corrupt_lines(tmp_path):
    """A half-written row (crash mid-append) must not take down every lab command."""
    from imagelab.lab.rundir import RunDir, ledger_rows
    rd = RunDir.create("ok_run", root=tmp_path)
    rd.start(name="ok_run", task="t", config_path="x.yaml", tier="smoke", deltas={},
             resolved={}, seed=0, device="cpu", amp="none")
    with open(tmp_path / "ledger.jsonl", "a") as f:
        f.write('{"run_id": "truncated", "status"')   # no newline, invalid JSON
    rows = ledger_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["name"] == "ok_run"


def test_overfit_verdict_gates():
    from imagelab.lab.criteria import gate_verdict
    gate = ("val/recon_l2_full", "<=", 0.01)
    assert gate_verdict(gate, {"val/recon_l2_full": 0.001})[0] == "pass"
    assert gate_verdict(gate, {"val/recon_l2_full": 0.2})[0] == "fail"
    assert gate_verdict(("car/token_acc", ">=", 0.9), {"car/token_acc": 0.99})[0] == "pass"
    assert gate_verdict(("car/token_acc", ">=", 0.9), {})[0] == "unknown"
    assert gate_verdict(None, {"x": 1})[0] == "unknown"
