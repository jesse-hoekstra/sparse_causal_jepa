"""Check single-context probe semantics, label separation and frozen evaluation."""

import json
import math
from collections.abc import Generator
from dataclasses import asdict
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.multiquery_control.data import build_cache, module_fingerprint
from diagnostics.multiquery_control.evaluation import (
    _variance,  # pyright: ignore[reportPrivateUsage]
    evaluate,
    parameter_readouts,
)
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.types import DGPConfig
from scjepa.models.model import SCJEPA
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


@pytest.fixture(autouse=True, scope="module")
def _single_thread() -> Generator[None, None, None]:  # pyright: ignore[reportUnusedFunction]
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model() -> SCJEPA:
    model = build_model(
        ModelConfig(
            observation_dim=6,
            state_dim=6,
            param_dim=6,
            action_dim=16,
            context_len=10,
            history_len=1,
            autoregressive=True,
            encoder_hidden_dims=(8,),
            parameter_encoder_kind="transition",
            spartan_layers=1,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_mlp_layers=2,
            spartan_dense=True,
        )
    )
    model.online_encoder.requires_grad_(False).eval()
    model.target_encoder.requires_grad_(False).eval()
    return model


def test_joint_nonlinear_probes_recover_all_six_mixed_coordinates() -> None:
    """A mixed code can retain all information without coordinate-wise theta alignment."""
    generator = torch.Generator().manual_seed(1401)
    raw = torch.randn(6, 6, dtype=torch.float64, generator=generator)
    rotation, _ = cast(tuple[Tensor, Tensor], torch.linalg.qr(raw))  # pyright: ignore[reportUnknownMemberType]
    codes: dict[str, Tensor] = {}
    data: dict[str, Any] = {}
    for name, count in (("calibration", 256), ("validation", 128), ("test", 128)):
        latent = torch.rand(count, 6, dtype=torch.float64, generator=generator) * 2 - 1
        codes[name] = (latent @ rotation)[:, None].expand(-1, 3, -1).clone()
        data[name] = {"theta": latent.pow(3)}
    result = parameter_readouts(codes, data)
    assert min(result["selected_test_r2"]) > 0.999
    assert max(result["families"]["linear"]["test_r2"]) < 0.95
    assert set(result["selected_methods_per_parameter"]) == {"cubic"}
    assert len(result["selected_test_group_mse_by_parameter"]) == 128
    json.dumps(result, allow_nan=False)


def test_context_variance_distinguishes_invariance_from_constant_codes() -> None:
    """Constant codes have no total variance; invariant informative codes do."""
    shared = torch.arange(4, dtype=torch.float64)[:, None, None].expand(4, 10, 6)
    assert _variance(shared)["within_over_total"] == 0
    assert math.isclose(cast(float, _variance(shared)["total"]), 5 / 3)
    varied = shared + torch.arange(10, dtype=torch.float64)[None, :, None] / 10
    stats = _variance(varied)
    expected_within = float(varied.var(1, correction=1).mean())
    expected_between = float(varied.mean(1).var(0, correction=1).mean()) - expected_within / 10
    assert stats["within"] == expected_within
    assert stats["between"] == expected_between
    assert stats["raw_consistency"] == 2 * expected_within
    assert _variance(torch.zeros(4, 10, 6))["within_over_total"] is None


def test_cache_separates_labels_and_final_evaluation_preserves_cpu_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny exact-law evaluation changes neither frozen states nor random streams."""
    model = _model()
    source_hash = module_fingerprint(model.online_encoder)
    config = asdict(
        DGPConfig(
            family="simplified_gaussian",
            trajectory_length=12,
            episodes_per_parameter=10,
            parameter_graph="complementary",
        )
    )
    train, data = build_cache(
        model,
        VectorObservationMap(dimension=6),
        config,
        train_groups=4,
        calibration_groups=8,
        validation_groups=4,
        test_groups=4,
        batch_size=2,
        eval_seed=1402,
    )
    assert set(train) == {"online", "target", "actions"}
    assert train["online"].shape == (4, 10, 12, 6)
    assert all(
        tensor.device.type == "cpu" and not tensor.requires_grad for tensor in train.values()
    )
    assert not torch.equal(data["validation"]["theta"], data["test"]["theta"])
    assert data["test"]["true_actions"].dtype == torch.float64
    assert data["test"]["actions"].dtype == torch.float32

    def fail_cuda(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CPU evaluation touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_rng_state", fail_cuda)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_cuda)
    monkeypatch.setattr(torch.cuda, "init", fail_cuda)
    objective = ObjectiveConfig(
        state_regularizer="sigreg",
        state_regularizer_weight=0.4,
        parameter_regularizer="sigreg",
        parameter_regularizer_weight=0.4,
        parameter_consistency_weight=0.0,
        regularizer_num_projections=4,
    )
    rng = torch.get_rng_state().clone()
    result = evaluate(model, data, objective, batch_size=2, final=True)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert module_fingerprint(model.online_encoder) == source_hash
    assert model.training
    assert not model.online_encoder.training
    assert not model.target_encoder.training
    assert result["split"] == "test"
    assert result["queries_per_code"] == 9
    assert len(result["parameter_readouts"]["selected_test_r2"]) == 6
    assert len(result["true_parameter_interventions"]["by_parameter"]) == 6
    assert math.isclose(
        result["total_at_lambda1"] - result["total_at_lambda0"],
        result["raw_consistency"],
        abs_tol=1e-6,
    )
    json.dumps(result, allow_nan=False)
    validation_only = {key: value for key, value in data.items() if key != "test"}
    monitored = evaluate(model, validation_only, objective, batch_size=2)
    assert monitored["split"] == "validation"
    assert "parameter_readouts" not in monitored
