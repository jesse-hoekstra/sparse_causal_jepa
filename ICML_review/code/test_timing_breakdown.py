"""Check the CPU failure diagnosis against native autograd and known numerical edge cases."""

import json
import math
from pathlib import Path

import pytest
import torch

from diagnostics.masked_mlp_learned import run as native
from diagnostics.shared.oracle import objective_config
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.loop import TrainingObjective
from scjepa.training.rng import preserve_rng

from .timing_breakdown import batch_probe, bound_failure, chart, describe, model_identity


def test_finite_elements_can_overflow_single_precision_norm() -> None:
    """FP64 accumulation can resolve a norm when all its FP32 components remain finite."""
    row = describe([torch.tensor([1e22, -1e22])])
    assert row["sum_squares_fp32"] == "inf"
    assert row["nonfinite_elements"] == 0
    assert math.isclose(row["sum_squares_fp64"], 2**0.5 * 1e22, rel_tol=1e-7)
    assert describe([torch.tensor([float("nan")])])["nonfinite_elements"] == 1


def test_nearly_and_exactly_constant_codes_are_distinguished() -> None:
    """A chart varying by one representable unit has a real resolved direction."""
    codes = torch.full((16, 10, 6), 0.2)
    constant = chart(codes)
    assert constant["all_codes_exactly_equal"]
    assert constant["total_numerical_rank"] == 0
    assert constant["worst_within_fraction"] is None
    codes[0, 0, 0] = torch.nextafter(codes[0, 0, 0], torch.tensor(1.0))
    near = chart(codes)
    assert not near["all_codes_exactly_equal"]
    assert near["total_numerical_rank"] == 1
    assert near["exactly_identical_sibling_groups"] == 15


def test_step_zero_log_bounds_failure_without_claiming_exact_update(tmp_path: Path) -> None:
    """A checkpoint preceding an incomplete logging block bounds the unknown failed update."""
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps({"step": 0}) + "\n")
    row = bound_failure(history, 0, 100)
    assert row["earliest_possible_failed_update"] == 1
    assert row["latest_possible_failed_update"] == 100
    assert row["exact_failed_update_from_archive"] is None


def test_replay_preserves_model_and_rng_and_matches_native_gradient() -> None:
    """A forced-offset counterfactual matches direct native CPU backward under that draw."""
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(3)
    batch = {
        "observations": torch.randn(8, 10, 12, 6, generator=gen),
        "actions": torch.randn(8, 10, 11, 24, generator=gen),
    }
    model = native.build_control("dense", 0, predictor_activation="gelu")
    model.online_encoder.requires_grad_(False)
    model.train()
    obj = TrainingObjective(objective_config("dense", 0.0), GECOConfig())
    old = model_identity(model.state_dict())
    rng = torch.get_rng_state()
    result = batch_probe(model, obj, batch, 1, 0, None)
    matched = batch_probe(model, obj, batch, 1, 0, result["donor_offset"])
    assert matched == result
    assert torch.equal(torch.get_rng_state(), rng)
    assert model_identity(model.state_dict()) == old
    assert all(p.grad is None for p in model.parameters())
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed((1 << 40) + 1)
        native_loss, _, _ = native.forward_loss(model, batch, obj, forward_seed=1)
        params = tuple(p for p in model.parameters() if p.requires_grad)
        grads = torch.autograd.grad(native_loss, params)
    assert describe(grads) == result["global_gradients"]["native_only"]
    assert result["global_gradients"]["guided_native_cap"]["nonfinite_elements"] == 0
    with pytest.raises(ValueError, match="only observations"):
        batch_probe(model, obj, batch | {"oracle_parameters": torch.empty(0)}, 1, 0, None)


def test_recompute_heldout_r2_and_code_chart_from_bound_samples(tmp_path: Path) -> None:
    """Score the release sample layout, using the calibration fit for each physical probe."""
    from diagnostics.shared.io import sha
    from diagnostics.shared.oracle import EvaluationResult

    from .timing_breakdown import SPLITS, heldout

    gen = torch.Generator().manual_seed(5)
    samples: dict[str, EvaluationResult] = {}
    hashes: dict[str, str] = {}
    for split in SPLITS:
        codes = torch.randn(24, 10, 6, generator=gen)
        values = {
            "codes": codes,
            "predictions": torch.ones(24, 10, 2, 6),
            "targets": torch.zeros(24, 10, 2, 6),
            "theta": codes.mean(dim=1),
        }
        samples[split] = EvaluationResult({}, values)
    for split, record in samples.items():
        r2 = native.code_metrics(samples["calibration"], record)["code_r2_min"]
        metrics = {"prediction": 1.0, "code_r2_min": r2, "state_mcc": 0.5}
        torch.save({"metrics": metrics, "samples": record.samples}, tmp_path / f"{split}.pt")
        (tmp_path / f"{split}.json").write_text(json.dumps({"metrics": metrics}))
        for suffix in ("pt", "json"):
            file = tmp_path / f"{split}.{suffix}"
            hashes[file.name] = sha(file)
    result = heldout(tmp_path, {"hashes": hashes}, {})
    assert result["validation"]["r2_matches"]
    assert result["validation"]["mse_matches"]
    assert result["validation"]["code"]["total_numerical_rank"] == 6
    (tmp_path / "validation.json").write_text("{}")
    with pytest.raises(ValueError, match="changed outcome"):
        heldout(tmp_path, {"hashes": hashes}, {})


@pytest.mark.parametrize("force_overflow", [False, True])
def test_bounded_native_replay_stops_before_failed_adam_step(force_overflow: bool) -> None:
    """Exercise successful and failing clip paths of the bounded worker with portable CPU RNG."""
    import random

    import numpy as np

    from diagnostics.shared.sampling import GroupSampler

    from .timing_breakdown_gpu import replay_window

    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(8)
    batch = {
        "observations": torch.randn(12, 10, 12, 6, generator=gen),
        "actions": torch.randn(12, 10, 11, 24, generator=gen),
    }
    model = native.build_control("dense", 2, predictor_activation="gelu")
    model.online_encoder.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    saved = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),  # noqa: NPY002
        "torch_rng": torch.get_rng_state(),
        "step": 0,
        "optimizer": optimizer.state_dict(),
        "objective": TrainingObjective(objective_config("dense", 0.0), GECOConfig()).state_dict(),
        "sampler": GroupSampler(12, 8, 2).state_dict(),
    }
    config = {"seed": 2, "lr": 0.001, "batch_size": 8, "log_every": 2, "grad_clip": 1.0}
    if force_overflow:
        p = next(p for p in model.predictor.parameters() if p.requires_grad)

        def overflow(grad: torch.Tensor) -> torch.Tensor:
            return torch.full_like(grad, 1e27)

        p.register_hook(overflow)  # pyright: ignore[reportUnknownMemberType]
    starting = model_identity(model.state_dict())
    with preserve_rng(torch.device("cpu")):
        r = replay_window(model, saved, batch, config, 2, lambda: None)
    if force_overflow:
        assert r["status"] == "global_clip_reproduced"
        assert r["failed_step"] == 1
        assert r["all_counterfactual_replayed_gradients_match"]
        assert r["actual_clip_input"]["nonfinite_elements"] == 0
        assert r["actual_clip_input"]["native_clip_norm"] == "inf"
        assert model_identity(model.state_dict()) == starting
    else:
        assert r["status"] == "finite_through_archive_bound"
        assert [entry["step"] for entry in r["successes"]] == [1, 2]
        assert model_identity(model.state_dict()) != starting
