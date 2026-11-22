"""Check the frozen window-by-gate design and its context-paired uncertainty."""

from pathlib import Path

import torch

from diagnostics.vision_dimer import data, evaluate, train
from diagnostics.vision_dimer.model import build_model

from .mask_windows import CELLS, measure, summarize


def test_zero_available_offset_cancels_both_window_effects(tmp_path: Path) -> None:
    """Equal windows must give identical forecasts even with stochastic gates and a short batch."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cfg = data.DataConfig(
            train_groups=2,
            calibration_groups=5,
            validation_groups=2,
            episode_frames=12,
            views=3,
        )
        data.prepare(tmp_path, cfg)
        source = data.RenderedInputs(tmp_path, "calibration", torch.device("cpu"))
        model = build_model(2, phase="full", code_dim=4, context=10, initial_logit=0.0)
        reference = evaluate.prediction_mse(
            evaluate.collect(model, train.rendered_batches(source, 3))
        )
        report = measure(model, source, "calibration", 3, 0.1, reference)
        assert report["first_window_reference"]["matches"]
        assert report["first_window_reference"]["bitwise_constant_across_schedules"]
        evidence = report["raw_mse_by_repeat_then_group"]
        assert evidence["first_sampled"] == evidence["uniform_sampled"]
        assert evidence["first_hard"] == evidence["uniform_hard"]
        assert evidence["first_sampled"][0] != evidence["first_sampled"][1]
        assert report["contrasts"]["uniform_minus_first_sampled"]["context_bootstrap_95"] == [0, 0]
        assert all(len(row) == cfg.calibration_groups for row in evidence["first_hard"])
        assert report["batches"][1]["group_stop"] == cfg.calibration_groups
        assert all(
            row["uniform_offset_counts"] == [(row["group_stop"] - row["group_start"]) * cfg.views]
            for row in report["batches"]
        )
    finally:
        torch.set_num_threads(threads)


def test_context_bootstrap_preserves_pairs_and_separates_schedule_variation() -> None:
    """An exactly constant paired effect stays fixed despite heterogeneous contexts/schedules."""
    raw = torch.tensor([[1.0, 3.0, 2.0], [4.0, 6.0, 5.0]], dtype=torch.float64)
    cells = {k: raw.clone() for k in CELLS}
    cells["uniform_hard"] += 0.5
    result = summarize(cells, 0.5, 42)
    effect = result["contrasts"]["uniform_minus_first_hard"]
    assert effect["estimate"] == 1
    assert effect["context_bootstrap_95"] == [1, 1]
    assert effect["schedule_sd"] == 0
    assert result["cells"]["first_hard"]["schedule_sd"] > 0
    assert result["contrasts"]["hard_minus_sampled_uniform_minus_first"]["estimate"] == 1
