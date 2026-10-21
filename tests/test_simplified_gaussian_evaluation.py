"""Held-out, group-aware diagnostics for the six-coordinate experiment."""

import copy
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.simplified_gaussian import graphs
from scjepa.data.types import DGPConfig
from scjepa.experiments.evaluation import (
    EvaluationConfig,
    _EncodedSplit,  # pyright: ignore[reportPrivateUsage]
    _gaussianized_parameters,  # pyright: ignore[reportPrivateUsage]
    _paired_code_variance,  # pyright: ignore[reportPrivateUsage]
    _recovery_metrics,  # pyright: ignore[reportPrivateUsage]
    _shuffled_code_prediction_mse,  # pyright: ignore[reportPrivateUsage]
    calibrate_tau,
    evaluate_model,
    validate_model,
)
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.spartan import SpartanOutput
from scjepa.training.factory import ModelConfig, build_model
from scjepa.training.loop import ObjectiveConfig


def _data() -> DGPConfig:
    return DGPConfig(
        family="simplified_gaussian",
        trajectory_length=5,
        episodes_per_parameter=2,
        parameter_graph="complementary",
    )


def _model() -> SCJEPA:
    return build_model(
        ModelConfig(
            observation_dim=6,
            state_dim=6,
            param_dim=6,
            action_dim=16,
            context_len=3,
            history_len=1,
            autoregressive=True,
            encoder_hidden_dims=(8,),
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_layers=1,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_mlp_layers=2,
            spartan_dense=True,
        )
    )


def _split(seed: int, count: int = 512) -> _EncodedSplit:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(count * 2, 5, 6, dtype=torch.float64, generator=generator)
    parameters = (torch.rand(count, 6, dtype=torch.float64, generator=generator) - 0.5) * 1.5
    parameters = parameters.repeat_interleave(2, dim=0)
    truth = graphs()
    return _EncodedSplit(
        config=_data(),
        reference_graphs=truth,
        ids=torch.arange(count).repeat_interleave(2),
        states=states,
        parameters=parameters,
        online=states.clone(),
        target=states[:, 3:].clone(),
        target_states=states[:, 3:].clone(),
        codes=parameters.clone(),
        graph_frequencies={
            name: getattr(truth, name).double() for name in ("state", "action", "parameter")
        },
        prediction_mse=0.0,
        base_loss=0.0,
        sparsity=0.0,
    )


def test_aligned_graph_counts_distinguish_false_positives_from_false_negatives() -> None:
    calibration, scored = _split(1), _split(2)
    frequencies = {name: value.clone() for name, value in scored.graph_frequencies.items()}
    frequencies["state"][0, 0, 1] = 0
    frequencies["state"][0, 0, 4] = 1
    frequencies["parameter"][2, 2] = 0
    frequencies["parameter"][2, 1] = 1
    metrics, _, _ = _recovery_metrics(
        calibration, replace(scored, graph_frequencies=frequencies), 0
    )
    for name in ("state", "parameter"):
        assert metrics[f"{name}_shd"] == 2
        assert metrics[f"{name}_false_positives"] == 1
        assert metrics[f"{name}_false_negatives"] == 1
    assert metrics["action_shd"] == metrics["action_false_negatives"] == 0


def test_linear_probes_recover_group_rotations_without_refitting_on_test() -> None:
    row_rotation = torch.zeros(6, 6, dtype=torch.float64)
    for start in (0, 3):
        row_rotation[start : start + 3, start : start + 3] = (
            torch.eye(3, dtype=torch.float64) - 2 / 3
        )
    pair_rotation = torch.eye(6, dtype=torch.float64)
    for index in range(3):
        pair_rotation[index, index] = pair_rotation[index + 3, index + 3] = 2**-0.5
        pair_rotation[index, index + 3] = -(2**-0.5)
        pair_rotation[index + 3, index] = 2**-0.5
    permutation = [2, 5, 0, 4, 1, 3]

    def rotated(split: _EncodedSplit) -> _EncodedSplit:
        return replace(
            split,
            online=(split.states @ row_rotation)[..., permutation],
            codes=(split.parameters @ pair_rotation)[..., permutation],
        )

    calibration, scored = rotated(_split(3)), rotated(_split(4))
    metrics, _, _ = _recovery_metrics(calibration, scored, 0)
    assert metrics["state_mcc"] < 0.8
    assert metrics["parameter_mcc"] < 0.8
    assert metrics["state_row_block_linear_r2"] > 0.999
    assert metrics["parameter_column_pair_linear_r2"] > 0.999
    changed_basis = replace(scored, online=-scored.online, codes=-scored.codes)
    changed, _, _ = _recovery_metrics(calibration, changed_basis, 0)
    assert changed["state_row_block_linear_r2"] < -1
    assert changed["parameter_column_pair_linear_r2"] < -1


def test_gaussianized_uniform_reference_recovers_known_coordinate_change() -> None:
    normal = torch.linspace(-4, 4, 1001, dtype=torch.float64)
    parameters = 0.75 * torch.erf(normal / math.sqrt(2))
    torch.testing.assert_close(_gaussianized_parameters(parameters), normal, atol=1e-12, rtol=1e-12)
    calibration, scored = _split(5), _split(6)
    calibration = replace(calibration, codes=_gaussianized_parameters(calibration.parameters))
    scored = replace(scored, codes=_gaussianized_parameters(scored.parameters))
    metrics, matches, _ = _recovery_metrics(calibration, scored, 0)
    assert metrics["parameter_mcc"] < 0.99
    assert math.isclose(metrics["parameter_gaussianized_mcc"], 1.0)
    assert "parameter_gaussianized" in matches


def test_context_variance_separates_invariance_from_constant_codes() -> None:
    shared = torch.arange(4, dtype=torch.float64)[:, None, None].expand(4, 2, 6)
    invariant = _paired_code_variance(shared.flatten(0, 1))
    assert invariant["parameter_within_theta_variance"] == 0
    assert invariant["parameter_within_theta_variance_ratio"] == 0
    assert math.isclose(invariant["parameter_total_variance"], 5 / 3)
    varied = shared + torch.tensor([-0.25, 0.25])[None, :, None]
    diagnostic = _paired_code_variance(varied.flatten(0, 1))
    assert math.isclose(diagnostic["parameter_within_theta_variance"], 0.125)
    assert math.isclose(diagnostic["parameter_within_theta_variance_ratio"], 0.075)
    constant = _paired_code_variance(torch.zeros(8, 6))
    assert constant["parameter_total_variance"] == 0
    assert all(math.isfinite(value) for value in constant.values())


def test_code_shuffle_uses_other_theta_opposite_view_and_recomputes_each_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model().eval()
    codes = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])[..., None].expand(3, 2, 6)
    states = torch.zeros(3, 2, 5, 6)
    actions = torch.arange(3 * 2 * 4 * 16, dtype=torch.float32).reshape(3, 2, 4, 16)
    targets = codes.flip(1)[:, :, None] * torch.tensor([1.0, 2.0])[None, None, :, None]
    result = ModelOutput(targets.clone(), targets, states, codes, (), 2)
    calls: list[tuple[Tensor, Tensor, Tensor]] = []
    forward = model.predictor.forward

    def observed_forward(
        history: Tensor, parameters: Tensor, observed_actions: Tensor | None = None
    ) -> SpartanOutput:
        assert observed_actions is not None
        calls.append((history.clone(), parameters.clone(), observed_actions.clone()))
        output = forward(history, parameters, observed_actions)
        return output._replace(prediction=history[:, -1] + parameters)

    monkeypatch.setattr(model.predictor, "forward", observed_forward)
    mse = _shuffled_code_prediction_mse(model, result, actions)
    expected = codes.roll(1, dims=0).flip(1).flatten(0, 1)
    assert len(calls) == 2
    for step, (history, parameters, observed_actions) in enumerate(calls):
        torch.testing.assert_close(parameters, expected)
        torch.testing.assert_close(history[:, 0], expected * step)
        torch.testing.assert_close(
            observed_actions, actions[:, :, step + 2 : step + 3].flatten(0, 1)
        )
    expected_prediction = expected[:, None] * torch.tensor([1.0, 2.0])[None, :, None]
    assert math.isclose(mse, float((expected_prediction - targets.flatten(0, 1)).square().mean()))
    assert mse > 0


def test_final_six_dimensional_evaluation_is_isolated_and_calibration_remains_unchanged(
    tmp_path: Path,
) -> None:
    model = _model().train()
    config = EvaluationConfig(
        calibration_episodes=8, validation_episodes=4, test_episodes=4, batch_size=2
    )
    matrix = VectorObservationMap(dimension=6).matrix
    objective = ObjectiveConfig(regularizer_num_projections=4)
    state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
    metrics = evaluate_model(model, _data(), matrix, config, tmp_path, objective=objective)
    assert model.training
    torch.testing.assert_close(model.state_dict(), state, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert all(math.isfinite(value) for value in metrics.values())
    assert "recovery/state_row_block_linear_r2" in metrics
    assert "recovery/parameter_column_pair_linear_r2" in metrics
    assert "recovery/parameter_gaussianized_mcc" in metrics
    assert "recovery/parameter_within_theta_variance_ratio" in metrics
    assert math.isclose(
        metrics["test/shuffled_parameter_prediction_mse_delta"],
        metrics["test/shuffled_parameter_prediction_mse"] - metrics["test/prediction_mse"],
    )
    for name in ("state", "action", "parameter"):
        assert metrics[f"graph/{name}_shd"] == (
            metrics[f"graph/{name}_false_positives"] + metrics[f"graph/{name}_false_negatives"]
        )
    references = cast(
        dict[str, Tensor], torch.load(tmp_path / "evaluation_references.pt", weights_only=True)
    )
    assert references["matrix"].shape == (6, 6)
    assert {int(value) for value in references["calibration_episode_ids"]}.isdisjoint(
        int(value) for value in references["test_episode_ids"]
    )
    assert "parameter_gaussianized_match" in references
    assert "additional_diagnostics" in read_json(tmp_path / "protocol.json")
    calibration = calibrate_tau(model, _data(), matrix, config, objective=objective)
    assert set(calibration) == {
        "calibration/prediction_mse",
        "calibration/base_loss",
        "calibration/sparsity",
    }
    validation = validate_model(model, _data(), matrix, config, objective=objective)
    assert "validation/state_row_block_linear_r2" in validation
    assert not any("shuffle" in name for name in validation)
