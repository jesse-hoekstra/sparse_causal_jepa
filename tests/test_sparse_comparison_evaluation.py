import copy
import math
from collections.abc import Generator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from test_multiquery_control import make_model, objective_config
from torch import Tensor

from diagnostics.joint_acquisition.baseline import evaluate_responses
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.multiquery_control.data import build_cache
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.sparse_comparison import evaluation
from diagnostics.sparse_comparison.source import Inputs
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig
from scjepa.eval.recovery import fit_coordinate_match, mean_correlation_coefficient
from scjepa.experiments import evaluation as native_evaluation
from scjepa.experiments.evaluation import EvaluationConfig
from scjepa.experiments.storage import read_json
from scjepa.training.factory import ModelConfig


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def inputs() -> Inputs:
    model = make_model().float()
    observation = VectorObservationMap(dimension=6)
    data = DGPConfig(
        family="simplified_gaussian",
        num_episodes=4,
        episodes_per_parameter=10,
        trajectory_length=12,
        root_seed=53,
    )
    _, cached = build_cache(
        model,
        observation,
        asdict(data),
        train_groups=4,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        eval_seed=21731,
        batch_size=2,
    )
    return Inputs(
        model=model,
        observations={},
        evaluation=cached,
        precision_data={},
        data_config=data,
        model_config=ModelConfig(),
        objective_config=objective_config(),
        native_evaluation=EvaluationConfig(
            calibration_episodes=4, validation_episodes=4, test_episodes=4, batch_size=2
        ),
        hashes={},
        provenance={},
        archived_validation={},
        admission={},
        source_step=200000,
        seed=202610261301,
        batch_size=2,
    )


def response_rows() -> dict[str, Any]:
    return {
        "per_parameter": [
            {
                "parameter": index + 1,
                "exact_response_energy": 1.0,
                "learned_response_energy": 0.1,
                "normalized_response_error": {"defined": True, "interval_95": [0.1, 0.9]},
                "predictive_benefit": {"defined": True, "interval_95": [0.01, 0.1]},
            }
            for index in range(6)
        ]
    }


def test_all_six_retention_and_fidelity_admission_keeps_original_benefit_separate() -> None:
    response = response_rows()
    response["per_parameter"][4]["predictive_benefit"]["interval_95"] = [-0.1, 0.2]
    result = evaluation.classify([0.8] * 6, response)
    assert result["revised_admission_pass"]
    assert not result["original_retention_and_use_gate_pass"]
    assert result["all_six_retention"]
    assert result["all_six_response_fidelity"]
    assert not result["per_parameter"][4]["positive_benefit_pass"]
    response["per_parameter"][4]["predictive_benefit"]["interval_95"] = [0.0, 0.2]
    assert not evaluation.classify([1.0] * 6, response)["all_six_positive_benefit"]
    response["per_parameter"][5]["normalized_response_error"]["interval_95"] = [0.9, 1.0]
    assert not evaluation.classify([1.0] * 6, response)["revised_admission_pass"]
    response = response_rows()
    assert not evaluation.classify([1.0] * 5 + [0.7999], response)["revised_admission_pass"]
    response["per_parameter"][2]["learned_response_energy"] = 0.0
    assert not evaluation.classify([1.0] * 6, response)["revised_admission_pass"]
    response = response_rows()
    response["per_parameter"][2]["normalized_response_error"]["defined"] = False
    assert not evaluation.classify([1.0] * 6, response)["all_six_response_fidelity"]
    response["per_parameter"][2]["normalized_response_error"] = {"interval_95": [0.2, None]}
    assert not evaluation.classify([float("nan")] * 6, response)["revised_admission_pass"]
    with pytest.raises(ValueError, match="six"):
        evaluation.classify([1.0] * 5, response)
    response["per_parameter"][2]["parameter"] = 1
    with pytest.raises(ValueError, match="order"):
        evaluation.classify([1.0] * 6, response)


def test_monitor_uses_only_validation_inputs_and_preserves_training_rng_and_modes(
    inputs: Inputs,
) -> None:
    # Keep only test shape metadata: any test input inference would fail.
    data = dict(inputs.evaluation)
    data["test"] = {"theta": inputs.evaluation["test"]["theta"]}
    inputs = replace(inputs, evaluation=data)
    model = inputs.model
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    with (
        patch.object(native_evaluation, "build_dgp_dataset", wraps=build_dgp_dataset) as draws,
        patch.object(evaluation, "evaluate_model", side_effect=AssertionError("No final test")),
    ):
        report, raw = evaluation.monitor(
            model, inputs, groups=4, bootstrap_samples=16, seed=97, tau=3
        )
    assert not report["test_inference"]
    assert report["role"] == "validation_monitor"
    assert {call.args[0].split for call in draws.call_args_list} == {
        "train",
        "evaluation_calibration",
        "validation",
    }
    native = report["native"]
    assert native["validation/feasible"] == float(native["validation/base_loss"] <= 3)
    assert native["geco/tau"] == 3
    assert raw["predictive_benefit"].shape == (4, 6)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [module.training for module in model.modules()]
    assert report["cohorts"]["native_registered_dgp"]["historically_reused"]
    assert report["cohorts"]["diagnostic_direct_draws"]["historically_reused"]


def test_chunked_explicit_test_responses_match_single_batch_and_direct_vector_identity(
    inputs: Inputs,
) -> None:
    model = copy.deepcopy(inputs.model).double().eval()
    with pytest.raises(ValueError, match="fixed-final"):
        evaluation.physical_response(
            model,
            inputs.evaluation,
            split="test",
            final=False,
            chunk_size=2,
            bootstrap_samples=24,
            seed=91,
        )
    batch, _, _ = matched_batch(model, inputs.evaluation, "test", tuple(range(4)), final=True)
    expected, expected_raw = evaluate_responses(model, batch, bootstrap_samples=24, seed=91)
    with patch.object(evaluation, "matched_batch", wraps=matched_batch) as chunks:
        result, raw, vectors = evaluation.physical_response(
            model,
            inputs.evaluation,
            split="test",
            final=True,
            chunk_size=2,
            bootstrap_samples=24,
            seed=91,
        )
    assert [call.args[2] for call in chunks.call_args_list] == ["test", "test"]
    assert [call.args[3] for call in chunks.call_args_list] == [(0, 1), (2, 3)]
    assert all(call.kwargs["final"] is True for call in chunks.call_args_list)
    for name, value in expected_raw.items():
        torch.testing.assert_close(raw[name], value, atol=1e-14, rtol=1e-8)
    for before, after in zip(expected["per_parameter"], result["per_parameter"], strict=True):
        for name in ("normalized_response_error", "predictive_benefit"):
            assert math.isclose(after[name]["estimate"], before[name]["estimate"], abs_tol=1e-12)
            torch.testing.assert_close(
                torch.tensor(after[name]["interval_95"]),
                torch.tensor(before[name]["interval_95"]),
                atol=1e-12,
                rtol=1e-8,
            )
    residual = vectors["factual_predictions"] - vectors["factual_targets"]
    learned = vectors["counterfactual_predictions"] - vectors["factual_predictions"][:, :, None]
    exact = vectors["counterfactual_targets"] - vectors["factual_targets"][:, :, None]
    axes = (1, 3, 4, 5)
    finite_gain = (2 * learned * exact - learned.square()).mean(axes)
    residual_term = (2 * residual[:, :, None] * learned).mean(axes)
    torch.testing.assert_close(raw["finite_response_gain"], finite_gain, atol=0, rtol=0)
    torch.testing.assert_close(
        raw["twice_residual_dot_learned_response"], residual_term, atol=0, rtol=0
    )
    torch.testing.assert_close(
        raw["predictive_benefit"], finite_gain - residual_term, atol=2e-15, rtol=1e-9
    )
    assert result["direct_vector_identity_max_absolute_error"] < 2e-15


def write_checkpoint(inputs: Inputs, output: Path, *, step: int = 2, identity: str = "arm") -> None:
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "run_metadata": {"experiment_run_identity": identity},
            "model": inputs.model.state_dict(),
        },
        output / "last.pt",
    )


def evaluate_final(
    inputs: Inputs, output: Path, *, step: int = 2
) -> tuple[dict[str, Any], list[str]]:
    return evaluation.final(
        inputs.model,
        inputs,
        output,
        objective=inputs.objective_config,
        tau=3,
        identity="arm",
        condition="dense",
        step=step,
        fixed_budget=2,
        verification=True,
        chunk_size=2,
        bootstrap_samples=16,
        seed=97,
    )


def test_final_checkpoint_step_identity_and_state_boundary_precedes_any_test_inference(
    inputs: Inputs, tmp_path: Path
) -> None:
    with patch.object(
        evaluation, "evaluate_model", side_effect=AssertionError("Too early")
    ) as test:
        write_checkpoint(inputs, tmp_path)
        with pytest.raises(ValueError, match="fixed-budget"):
            evaluate_final(inputs, tmp_path, step=1)
        write_checkpoint(inputs, tmp_path, identity="different")
        with pytest.raises(ValueError, match="identity"):
            evaluate_final(inputs, tmp_path)
        write_checkpoint(inputs, tmp_path)
        with torch.no_grad():
            next(inputs.model.parameters()).view(-1)[0].add_(0.1)
        with pytest.raises(ValueError, match="tensors"):
            evaluate_final(inputs, tmp_path)
        write_checkpoint(inputs, tmp_path)
        inputs.evaluation["test"]["theta"][0] = inputs.evaluation["calibration"]["theta"][0]
        with pytest.raises(ValueError, match="overlap"):
            evaluate_final(inputs, tmp_path)
        test.assert_not_called()


def test_final_emits_native_paired_mcc_contract_and_full_diagnostic_test_with_frozen_matches(
    inputs: Inputs, tmp_path: Path
) -> None:
    model = inputs.model
    # Change both current charts from archived encodings to catch stale-chart evaluation.
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(0.01)
        next(model.target_encoder.parameters()).add_(0.02)
    write_checkpoint(inputs, tmp_path)
    before = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    report, artifacts = evaluate_final(inputs, tmp_path)
    assert len(artifacts) == 7
    assert all((tmp_path / path).is_file() for path in artifacts)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [module.training for module in model.modules()]
    torch.testing.assert_close(model.state_dict(), before, atol=0, rtol=0)
    boundary = read_json(tmp_path / "evaluation/evaluated_checkpoint.json")
    assert boundary["run_identity"] == "arm"
    assert boundary["evaluated_step"] == 2
    assert boundary["checkpoint_path"] == "last.pt"
    assert boundary["checkpoint_sha256"] == sha(tmp_path / "last.pt")
    assert boundary["mcc_samples_sha256"] == sha(tmp_path / "evaluation/mcc_samples.pt")
    saved = cast(
        dict[str, Tensor],
        torch.load(tmp_path / "evaluation/mcc_samples.pt", map_location="cpu", weights_only=True),
    )
    assert saved["states"].shape == (4, 10, 12, 6)
    assert not bool((saved["group_ids"][:, None] == saved["calibration_group_ids"][None]).any())
    process = build_dgp_dataset(
        replace(inputs.data_config, split="evaluation_calibration"),
        observation_matrix=inputs.evaluation["metadata"]["observation_matrix"],
    ).process
    episodes = [process.generate(index) for index in range(4)]
    observations = torch.stack([episode.observations for episode in episodes])
    truth = torch.stack([episode.ground_truth["states"] for episode in episodes])
    with torch.no_grad():
        encoded = model.online_encoder.encode_sequence(observations.flatten(0, 1))
    match = fit_coordinate_match(encoded.reshape(-1, 6), truth.reshape(-1, 6), method="pearson")
    assert tuple(int(value) for value in saved["state_match"]) == match.columns
    native = report["native"]
    expected_mcc = mean_correlation_coefficient(
        match, saved["online"].reshape(-1, 6), saved["states"].reshape(-1, 6)
    )
    assert native["recovery/state_mcc"] == expected_mcc
    assert all(isinstance(value, float | int) for value in native.values())
    assert "graph/state_shd" in native
    assert "test/shuffled_parameter_prediction_mse_delta" in native
    assert native["test/feasible"] == float(native["test/base_loss"] <= native["geco/tau"])
    assert math.isclose(
        native["test/normalized_prediction_mse"],
        native["test/prediction_mse"] / native["test/target_variance"],
        abs_tol=1e-12,
    )
    refreshed = refresh_evaluation(model, inputs.evaluation, ("calibration", "validation", "test"))
    candidate = copy.deepcopy(model).double().eval()
    codes = {
        name: encode_codes(candidate, refreshed[name])
        for name in ("calibration", "validation", "test")
    }
    expected_retention = parameter_readouts(codes, refreshed)
    assert report["diagnostic"]["retention"] == expected_retention
    diagnostic = cast(
        dict[str, Any],
        torch.load(
            tmp_path / "evaluation/diagnostic_measurements.pt",
            map_location="cpu",
            weights_only=True,
        ),
    )
    assert diagnostic["response"]["predictive_benefit"].shape == (4, 6)
    assert diagnostic["retention_group_mse"].shape == (4, 6)
    torch.testing.assert_close(diagnostic["test_codes"], codes["test"], atol=0, rtol=0)
    protocol = read_json(tmp_path / "evaluation/protocol.json")
    assert protocol["comparison_cohorts"] == report["cohorts"]
    assert "descriptive only" in report["cohorts"]["parameter_coordinate_metrics"]
    assert report["cohorts"]["native_registered_dgp"]["data_root_seed"] == 53
    assert report["cohorts"]["diagnostic_direct_draws"]["base_draw_seed"] == 21731
