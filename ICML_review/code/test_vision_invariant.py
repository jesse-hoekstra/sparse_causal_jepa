"""Check RGB teacher lookup, native replay, both continuation arms and the pilot gate."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import train as release
from diagnostics.vision_dimer.analysis import load_model
from scjepa.models.model import ModelOutput

from .consensus_gradient import whitening
from .vision_invariant_eval import SPLITS, decision, endpoint, spectrum
from .vision_invariant_train import continue_training, table_batch, teacher_alignment, teacher_table


def test_native_and_both_targets_on_real_frames(tmp_path: Path) -> None:
    """Continue the native and intervention models from one sparse checkpoint on CPU."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _smoke(tmp_path)
    finally:
        torch.set_num_threads(threads)


def _smoke(root: Path) -> None:
    data = root / "data"
    vdata.prepare(
        data,
        vdata.DataConfig(
            episode_frames=16, train_groups=8, calibration_groups=8, validation_groups=4
        ),
    )
    cfg = release.StageConfig(
        "joint",
        str(data),
        str(root / "joint"),
        device="cpu",
        batch_groups=2,
        code_dim=4,
        steps=1,
        log_every=1,
        checkpoint_every=1,
        eval_every=1000,
    )
    release.train(cfg)
    cfg = replace(
        cfg,
        stage="full",
        parent=str(root / "joint/final.pt"),
        output_dir=str(root / "parent"),
        tau=0.05,
    )
    release.train(cfg)
    parent = root / "parent/last.pt"
    device = torch.device("cpu")
    model, _, _ = load_model(parent, device)
    model.eval()
    source = vdata.RenderedInputs(data, "train", device)
    table = teacher_table(model, source)
    assert teacher_alignment(model, source, table)["matched"]
    chart = whitening(table.flatten(1, 2))
    assert table.shape == (8, 10, 5, 4)
    idx = torch.tensor([4, 2])
    starts = torch.arange(20).reshape(2, 10) % 5
    native = source.batch(idx, starts)
    with torch.no_grad():
        out = cast(ModelOutput, model(native["observations"], native["actions"]))
    torch.testing.assert_close(
        table_batch(table, idx, starts), out.parameters, rtol=1e-5, atol=1e-5
    )
    for arm in ("reference", "native", "same", "other"):
        variant = replace(cfg, output_dir=str(root / arm), steps=3)
        release.fork(parent, variant)
        if arm == "reference":
            release.train(variant, resume=True)
        else:
            continue_training(variant, table, chart, arm)
    reference = release.load_checkpoint(root / "reference/last.pt")
    for arm in ("native", "same", "other"):
        saved = release.load_checkpoint(root / arm / "last.pt")
        for key in ("windows", "sampler", "cpu_rng"):
            torch.testing.assert_close(reference[key], saved[key], rtol=0, atol=0)
        if arm == "native":
            for key in ("model", "optimizer", "geco"):
                torch.testing.assert_close(reference[key], saved[key], rtol=0, atol=0)
        else:
            logs = [
                json.loads(s)
                for s in (root / arm / "anchor-history.jsonl").read_text().splitlines()
            ]
            assert len(logs) == 2
            assert all(0 <= r["added_to_native_grad_norm"] <= 1.00001 for r in logs)
    own = release.load_checkpoint(root / "same/last.pt")
    other = release.load_checkpoint(root / "other/last.pt")
    assert any(
        not torch.equal(v, other["model"][k])
        for k, v in own["model"].items()
        if k.startswith("parameter_encoder.")
    )
    native_results = endpoint(root / "reference/final.pt", data, device)
    assert all(native_results[s]["mse_matched"] for s in SPLITS)


def test_rejects_a_lost_direction_and_a_weak_matched_contrast() -> None:
    """A reduction relative to the archived endpoint alone cannot admit the pilot."""
    generator = torch.Generator().manual_seed(3)
    latent = torch.randn(40, 1, 4, generator=generator)
    episode = torch.randn(40, 6, 4, generator=generator)
    assert not spectrum(latent.expand(-1, 6, -1)[:, :, :1].expand(-1, -1, 4))["full_rank"]
    records: dict[str, dict[str, Any]] = {}
    for arm, scale in (("native", 1.0), ("same", 0.5), ("other", 0.5)):
        c = latent + scale * episode
        block: dict[str, Any] = {
            "chart": spectrum(c),
            "reported_spread": {"worst_direction_share": scale},
            "mse": 0.04,
            "mse_matched": True,
            "group_mse": [0.04] * 40,
            "group_donors": [[0.04, 0.04, 0.05, 0.001]] * 40,
            "codes": c.tolist(),  # pyright: ignore[reportUnknownMemberType]
        }
        records[arm] = {
            "semantic": {
                "parameter_r2_min": 0.97,
                "mcc": 0.91,
                "state_shd": 1,
                "parameter_shd": 0,
                "action_shd": 1,
            },
            "calibration": block,
            "validation": block,
        }
    verdict = decision(records, draws=20)
    assert not verdict["passed"]
    assert verdict["checks"]["validation_spread_vs_native"]
    assert not verdict["checks"]["validation_spread_vs_same"]
    assert (
        verdict["intervals"]["validation"]["paired"]["other_minus_same"]["worst_ratio"]["estimate"]
        == 1.0
    )
    records["other"]["validation"]["chart"] = {"full_rank": False}
    assert not decision(records, draws=20)["checks"]["validation_rank"]
