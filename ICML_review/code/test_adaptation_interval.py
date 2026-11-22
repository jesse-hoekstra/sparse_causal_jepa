"""Check native group aggregation and paired context uncertainty for the RGB adaptation gate."""

from pathlib import Path

import torch

from diagnostics.vision_dimer import data, evaluate, train
from diagnostics.vision_dimer.model import build_model
from scjepa.experiments.storage import write_json

from .adaptation_interval import MARGIN, SEEDS, compare, group_errors, paired_intervals, replay


def test_native_groups_reproduce_the_released_split_mean(tmp_path: Path) -> None:
    """Recover the released two-step first-window score with a short final batch."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = data.DataConfig(calibration_groups=5, views=3, episode_frames=14)
        split = "calibration"
        path = tmp_path / f"{split}.pt"
        torch.save(data.generate_split(cfg, split), path)
        write_json(
            tmp_path / "manifest.json",
            {"config": cfg.as_record(), "splits": {split: {"sha256": data.sha256(path)}}},
        )
        inputs = data.RenderedInputs(tmp_path, split, torch.device("cpu"))
        model = build_model(0, phase="dense", code_dim=4, context=10)
        raw = group_errors(model, inputs, batch_size=3)
        original = evaluate.collect(model, train.rendered_batches(inputs, size=3))
        assert raw.shape == (cfg.calibration_groups,)
        assert replay(raw, evaluate.prediction_mse(original))["matches"]
        predicted = original["predictions"]
        target = original["targets"]
        torch.testing.assert_close(
            raw,
            torch.stack([(p - t).square().mean() for p, t in zip(predicted, target, strict=True)]),
        )
    finally:
        torch.set_num_threads(threads)


def test_resampling_whole_paired_systems_preserves_an_exact_ratio() -> None:
    """Common context indices cancel group heterogeneity in an exact proportional comparison."""
    control = torch.tensor([0.5, 1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float64)
    near = MARGIN * control + torch.tensor([0.1, -0.1, 0.1, -0.1, 0.1, -0.1], dtype=torch.float64)
    result = paired_intervals(
        {"reported": control, "direct": 2 * control, "one_head": near}, 42, draws=500
    )
    assert result["direct"]["ratio_ci95"] == [2.0, 2.0]
    assert result["direct"]["excess_ci95"][0] > 0
    assert not result["direct"]["excess_contains_zero"]
    assert result["one_head"]["excess_contains_zero"]
    assert result["one_head"]["ratio_contains_margin"]


def test_one_bad_replay_withholds_both_split_intervals() -> None:
    """Retain raw errors when one checkpoint misses its archived score."""
    cells = {
        name: torch.tensor([0.5, 1.5], dtype=torch.float64)
        for name in ("reported", "direct", "one_head")
    }
    raw = {split: cells for split in SEEDS}
    sources = {name: {"raw_references": dict.fromkeys(SEEDS, 1.0)} for name in cells}
    sources["one_head"]["raw_references"]["validation"] = 1.1
    verified, reports = compare(raw, sources, views=10)
    assert not verified
    assert all("paired" not in r for r in reports.values())
    assert not reports["validation"]["replay"]["one_head"]["matches"]
    assert reports["calibration"]["group_mse"]["reported"] == [0.5, 1.5]
