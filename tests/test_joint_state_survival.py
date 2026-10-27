import copy
from collections.abc import Callable, Generator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
import test_predictor_transfer_control as transfer_fixtures
import test_predictor_transfer_precision as precision_fixtures
import torch
from omegaconf import DictConfig, OmegaConf
from test_encoder_gradient_source import refresh_hashes, write_source_fixture
from test_multiquery_control import make_model, objective_config
from torch import Tensor, nn

from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_state_survival import run as runner
from diagnostics.joint_state_survival.data import (
    check_train,
    refresh_evaluation,
    regenerate_train,
)
from diagnostics.joint_state_survival.evaluation import paired_changes, state_metrics
from diagnostics.joint_state_survival.forward import (
    Arm,
    after_update,
    configure_model,
    ordinary_objective,
    training_loss,
)
from diagnostics.joint_state_survival.probes import fit_probes, predict, score
from diagnostics.joint_state_survival.source import (
    QUALIFIED_CALIBRATED_SHA256,
    QUALIFIED_CHECKPOINT_SHA256,
    admission_gate,
    cast_replay,
    load_inputs,
    response_gate,
    response_status,
    validate_source_mode,
)
from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    build_cache,
    module_fingerprint,
)
from diagnostics.multiquery_control.evaluation import parameter_readouts
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import frozen_hashes, sha
from diagnostics.predictive_supervision_control.objective import PredictiveSupervisionObjective
from diagnostics.predictor_transfer_calibration.run import Config as CalibrationConfig
from diagnostics.predictor_transfer_calibration.run import run as run_calibration
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import SpartanOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.loop import TrainingObjective
from scjepa.training.rng import preserve_rng


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def data_config() -> dict[str, Any]:
    return {
        "family": "simplified_gaussian",
        "num_episodes": 4,
        "episodes_per_parameter": 10,
        "trajectory_length": 12,
        "split": "train",
        "root_seed": 53,
    }


@pytest.mark.parametrize("arm", ["frozen_state", "live_state"])
def test_ordinary_objective_matches_production_graph_rng_gradients_and_state_policy(
    arm: Arm,
) -> None:
    model = make_model().float()
    with torch.no_grad():
        next(model.target_encoder.parameters()).view(-1)[0].add_(0.02)
    initial_target = copy.deepcopy(model.target_encoder.state_dict())
    configure_model(model, arm)
    torch.testing.assert_close(model.target_encoder.state_dict(), initial_target, rtol=0, atol=0)
    assert module_fingerprint(model.online_encoder) != module_fingerprint(model.target_encoder)
    objective = ordinary_objective(replace(objective_config(), regularizer_num_projections=256))
    batch: TensorSplit = {
        "observations": torch.randn(2, 10, 12, 6),
        "actions": torch.randn(2, 10, 11, 16),
    }
    donors: list[Tensor] = []
    outputs: list[Tensor] = []
    histories: list[Tensor] = []

    def capture(_module: nn.Module, args: tuple[object, ...], output: SpartanOutput) -> None:
        donors.append(cast(Tensor, args[1]))
        histories.append(cast(Tensor, args[0]))
        outputs.append(output.prediction)

    hook = model.predictor.register_forward_hook(capture)
    torch.default_generator.manual_seed(893)
    try:
        loss, terms = training_loss(model, batch, objective, arm)
    finally:
        hook.remove()
    rng = torch.get_rng_state()
    assert donors[0] is donors[1]
    assert donors[0].requires_grad
    torch.testing.assert_close(histories[1], outputs[0][:, None], rtol=0, atol=0)
    gradient = torch.autograd.grad(outputs[1].square().mean(), outputs[0], retain_graph=True)[0]
    assert float(gradient.square().sum()) > 0
    for predicted in outputs:
        gradient = torch.autograd.grad(predicted.square().mean(), donors[0], retain_graph=True)[0]
        assert float(gradient.square().sum()) > 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = cast(
        tuple[Tensor | None, ...],
        torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True),
    )
    torch.default_generator.manual_seed(893)
    output = model(batch["observations"], batch["actions"])
    expected, expected_terms = TrainingObjective(objective.config)(output)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        loss,
        terms["prediction"]
        + 0.0001 * terms["logit"]
        + 0.4 * terms["state_regularizer"]
        + 0.4 * terms["parameter_regularizer"]
        + terms["parameter_consistency"],
        rtol=1e-6,
        atol=1e-7,
    )
    expected_gradients = cast(
        tuple[Tensor | None, ...], torch.autograd.grad(expected, parameters, allow_unused=True)
    )
    for actual, target in zip(gradients, expected_gradients, strict=True):
        if actual is None or target is None:
            assert actual is None
            assert target is None
        else:
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for key in expected_terms:
        torch.testing.assert_close(terms[key], expected_terms[key], rtol=0, atol=0)
    before_online = copy.deepcopy(model.online_encoder.state_dict())
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    after_update(model, arm)
    if arm == "live_state":
        assert any(
            not torch.equal(v, model.online_encoder.state_dict()[k])
            for k, v in before_online.items()
        )
        for key, value in model.target_encoder.state_dict().items():
            torch.testing.assert_close(
                value,
                initial_target[key] * 0.996 + model.online_encoder.state_dict()[key] * 0.004,
                rtol=1e-6,
                atol=1e-7,
            )
    else:
        torch.testing.assert_close(model.online_encoder.state_dict(), before_online, rtol=0, atol=0)
        torch.testing.assert_close(
            model.target_encoder.state_dict(), initial_target, rtol=0, atol=0
        )
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())
    with pytest.raises(ValueError, match="observations and actions"):
        training_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective, arm)
    with pytest.raises(ValueError, match="full ordinary"):
        ordinary_objective(replace(objective_config(), regularizer_num_projections=8))


def test_regeneration_authenticates_original_inputs_and_refreshes_both_charts() -> None:
    model = make_model().float().eval()
    with torch.no_grad():
        next(model.target_encoder.parameters()).view(-1)[0].add_(0.02)
    observation = VectorObservationMap(dimension=6)
    cached, data = build_cache(
        model,
        observation,
        data_config(),
        train_groups=4,
        calibration_groups=3,
        validation_groups=3,
        test_groups=3,
        eval_seed=318,
        batch_size=3,
    )
    rng = torch.get_rng_state().clone()
    observed, receipt = regenerate_train(model, cached, data["metadata"], data_config())
    assert torch.equal(rng, torch.get_rng_state())
    assert set(observed) == {"observations", "actions"}
    assert receipt["actions_exact"]
    assert receipt["all_initial_encodings_checked"]
    torch.testing.assert_close(observed["actions"], cached["actions"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        regenerate_train(model, cached, data["metadata"], {**data_config(), "root_seed": 54})
    broken = {key: value.clone() for key, value in cached.items()}
    broken["actions"][0, 0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="actions differ"):
        regenerate_train(model, broken, data["metadata"], data_config())
    broken = {key: value.clone() for key, value in cached.items()}
    broken["target"][3, 9, 11, 0] += 0.1
    with pytest.raises(AssertionError):
        regenerate_train(model, broken, data["metadata"], data_config())
    with pytest.raises(ValueError, match="observations and actions"):
        check_train({**observed, "online": cached["online"]})
    with torch.no_grad():
        next(model.online_encoder.parameters()).view(-1)[0].add_(0.03)
        next(model.target_encoder.parameters()).view(-1)[0].sub_(0.01)
    refreshed = refresh_evaluation(model, data, ("calibration", "validation"))
    assert set(refreshed) == {"metadata", "calibration", "validation"}
    assert refreshed["metadata"]["state_encodings_refreshed"]
    for name in ("calibration", "validation"):
        old, new = cast(TensorSplit, data[name]), cast(TensorSplit, refreshed[name])
        observed_eval = observation.forward(old["true_states"]).float().flatten(0, 1)
        for chart in ("online", "target"):
            assert not torch.equal(new[f"states_{chart}"], old[f"states_{chart}"])
            with torch.no_grad():
                expected = getattr(model, f"{chart}_encoder").encode_sequence(observed_eval)
            torch.testing.assert_close(
                new[f"states_{chart}"], expected.reshape(3, 10, 12, 6), rtol=1e-6, atol=1e-7
            )
        for key in ("true_states", "true_actions", "theta", "actions"):
            assert new[key] is old[key]


def test_float32_materialization_replays_fixed_eval_donors_and_gate_requires_every_factor() -> None:
    model = make_model().eval()
    observed, actions = torch.randn(2, 10, 12, 6), torch.randn(2, 10, 11, 16)
    rng = torch.get_rng_state().clone()
    candidate, receipt = cast_replay(model, observed, actions)
    assert torch.equal(rng, torch.get_rng_state())
    assert next(candidate.parameters()).dtype == torch.float32
    assert next(model.parameters()).dtype == torch.float64
    assert all(row["max_absolute_error"] <= row["tolerance"] for row in receipt["replay"].values())
    with torch.no_grad():
        next(model.parameter_encoder.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        cast_replay(model, observed, actions)
    rows: list[dict[str, Any]] = [
        {
            "parameter": i,
            "normalized_response_error": {"interval_95": [0.1, 0.9]},
            "predictive_benefit": {"interval_95": [0.001, 0.01]},
        }
        for i in range(1, 7)
    ]
    assert response_gate({"per_parameter": rows})
    for field, bounds in (
        ("normalized_response_error", [0.1, 1.0]),
        ("predictive_benefit", [0.0, 0.1]),
    ):
        changed = copy.deepcopy(rows)
        changed[5][field]["interval_95"] = bounds
        assert not response_gate({"per_parameter": changed})
    assert not response_gate({"per_parameter": rows[:-1]})


def test_explicit_qualified_admission_keeps_strict_use_gate_and_rejects_additional_failures() -> (
    None
):
    rows: list[dict[str, Any]] = [
        {
            "parameter": i,
            "normalized_response_error": {"estimate": 0.5, "interval_95": [0.1, 0.9]},
            "predictive_benefit": {"estimate": 0.01, "interval_95": [0.005, 0.015]},
        }
        for i in range(1, 7)
    ]
    rows[1]["predictive_benefit"] = {"estimate": 0.003, "interval_95": [-0.002, 0.008]}
    report = {"per_parameter": rows}
    assert not response_gate(report)
    assert not admission_gate(report, "all_six_use")
    assert admission_gate(report, "qualified_spartan")
    assert not response_gate(report), "Permission to diagnose must not relabel source use"
    status = response_status(report)
    assert status["all_six_fidelity_pass"]
    assert not status["all_six_use_pass"]
    assert status["benefit_pass_parameters"] == [1, 3, 4, 5, 6]
    assert status["unmet_benefit_parameters"] == [2]
    failures: list[dict[str, Any]] = []
    for factor, field, interval in (
        (2, "predictive_benefit", [-0.001, 0.015]),
        (1, "predictive_benefit", [-0.01, 0.0]),
        (1, "predictive_benefit", [-0.01, -0.005]),
        (4, "normalized_response_error", [0.1, 1.0]),
        (1, "predictive_benefit", [float("nan"), 0.008]),
        (1, "predictive_benefit", [-0.002, float("inf")]),
        (1, "predictive_benefit", [0.02, 0.01]),
        (1, "predictive_benefit", None),
    ):
        changed = copy.deepcopy(report)
        changed["per_parameter"][factor][field]["interval_95"] = interval
        failures.append(changed)
    failures.append({"per_parameter": copy.deepcopy(rows[:-1])})
    duplicate = copy.deepcopy(report)
    duplicate["per_parameter"][1]["parameter"] = 1
    failures.append(duplicate)
    negative_point = copy.deepcopy(report)
    negative_point["per_parameter"][1]["predictive_benefit"]["estimate"] = -0.001
    failures.append(negative_point)
    for failure in failures:
        assert not admission_gate(failure, "qualified_spartan")
    all_six = copy.deepcopy(report)
    all_six["per_parameter"][1]["predictive_benefit"]["interval_95"] = [0.001, 0.008]
    assert admission_gate(all_six, "all_six_use")
    assert not admission_gate(all_six, "qualified_spartan")


def test_qualified_mode_is_explicit_and_pinned_to_only_the_authorized_source(
    tmp_path: Path,
) -> None:
    assert (
        QUALIFIED_CHECKPOINT_SHA256
        == "b85b70e110f6d00a1d5bbcdddcf9756d0288daf2ccf1c51eded92eff49d864ce"
    )
    assert (
        QUALIFIED_CALIBRATED_SHA256
        == "8225738a6db85e074bf95a0af9c40055db979a1ccdca31735ba61485f1335cb9"
    )
    config = runner.Config(
        str(tmp_path / "source"),
        str(tmp_path / "endpoint"),
        str(tmp_path / "calibration"),
        str(tmp_path / "precision"),
        str(tmp_path / "output"),
        QUALIFIED_CHECKPOINT_SHA256,
        QUALIFIED_CALIBRATED_SHA256,
        "live_state",
    )
    assert config.source_gate_mode == "all_six_use"
    qualified = replace(config, source_gate_mode="qualified_spartan")
    assert not qualified.verification
    assert qualified.steps == 10000
    assert qualified.groups == 64
    validate_source_mode(
        "qualified_spartan", QUALIFIED_CHECKPOINT_SHA256, QUALIFIED_CALIBRATED_SHA256
    )
    for field in ("checkpoint_sha256", "calibrated_sha256"):
        with pytest.raises(ValueError, match="specifically pinned"):
            replace(qualified, **{field: "f" * 64})
    with pytest.raises(ValueError, match="Unknown source gate"):
        replace(config, source_gate_mode=cast(Any, "ignore_failed_gate"))
    # The source loader independently applies the pin before opening any artifacts.
    with pytest.raises(ValueError, match="specifically pinned"):
        load_inputs(
            tmp_path / "source",
            tmp_path / "endpoint",
            tmp_path / "calibration",
            tmp_path / "precision",
            checkpoint_sha256="f" * 64,
            calibrated_sha256=QUALIFIED_CALIBRATED_SHA256,
            device=torch.device("cpu"),
            source_gate_mode="qualified_spartan",
        )


def genuine_source(root: Path) -> Path:
    """Rebuild the tiny source with an actual TRAIN stream and complete data provenance."""
    source = write_source_fixture(root)
    protocol = cast(dict[str, Any], read_json(source / "protocol.json"))
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(123)
        model = build_model(cast(DictConfig, OmegaConf.create(protocol["model_config"])))
        with torch.no_grad():
            next(model.target_encoder.parameters()).view(-1)[0].add_(0.02)
        model.online_encoder.requires_grad_(False).eval()
        model.target_encoder.requires_grad_(False).eval()
        observation = VectorObservationMap(dimension=6)
        train, evaluation = build_cache(
            model,
            observation,
            data_config(),
            train_groups=4,
            calibration_groups=4,
            validation_groups=4,
            test_groups=4,
            eval_seed=319,
            batch_size=3,
        )
        objective = replace(objective_config(), regularizer_num_projections=256)
        protocol["objective_config"] = asdict(objective)
        protocol["frozen_hashes"] = frozen_hashes(model)
        protocol["source"].update(
            {
                "checkpoint": str(source / "original/best.pt"),
                "run_identity": "genuine-tiny-joint-state-source",
                "dataset_identity": evaluation["metadata"]["training_dataset_fingerprint"],
            }
        )
        (source / "original").mkdir()
        write_json(
            source / "original/run_config.json",
            {
                "data": data_config(),
                "run_identity": protocol["source"]["run_identity"],
                "dataset_identity": protocol["source"]["dataset_identity"],
            },
        )
        torch.save(train, source / "train_inputs.pt")
        torch.save(evaluation, source / "evaluation_data.pt")
        torch.default_generator.manual_seed(7)
        fresh = build_model(cast(DictConfig, OmegaConf.create(protocol["model_config"])))
        fresh.online_encoder.load_state_dict(model.online_encoder.state_dict())
        fresh.target_encoder.load_state_dict(model.target_encoder.state_dict())
        torch.save(
            {"model": fresh.state_dict(), "seed": 7, "source_sha256": protocol["source"]["sha256"]},
            source / "initial_seed_7.pt",
        )
        protocol["initializations"]["7"] = {
            "path": "initial_seed_7.pt",
            "sha256": sha(source / "initial_seed_7.pt"),
        }
        protocol["artifacts"]["initial_seed_7.pt"] = sha(source / "initial_seed_7.pt")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
        loss_function = PredictiveSupervisionObjective(objective, 1.0)
        for step in (1, 2):
            indices = list(GlobalBatchSampler(4, 2, 7, 0))[step - 1]
            model.train()
            model.online_encoder.eval()
            model.target_encoder.eval()
            torch.default_generator.manual_seed(700000 + step)
            optimizer.zero_grad(set_to_none=True)
            output = cached_multiquery_forward(
                model, {key: value[indices] for key, value in train.items()}, queries=1
            )
            loss, _ = loss_function(output)
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        path = source / "seed-7/prediction_1/last.pt"
        payload = cast(dict[str, Any], torch.load(path, weights_only=False))
        payload.update(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "frozen_hashes": protocol["frozen_hashes"],
                "initial_sha256": sha(source / "initial_seed_7.pt"),
            }
        )
        torch.save(payload, path)
        write_json(source / "protocol.json", protocol)
        refresh_hashes(source)
    return source


@pytest.fixture(scope="module")
def joint_sources(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path, Path]:
    """Use unchanged upstream fixture builders, with genuine TRAIN provenance from the start."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        build_transfer = cast(
            Callable[..., tuple[Path, Path, Path]],
            cast(Any, transfer_fixtures.transfer_sources).__wrapped__,
        )
        build_precision = cast(
            Callable[..., tuple[Path, Path, Path, Path]],
            cast(Any, precision_fixtures.precision_sources).__wrapped__,
        )
        # Only the fixture construction helper is replaced; all production loaders,
        # training, data generation, source authentication and calibration run normally.
        with patch.object(transfer_fixtures, "source_with_initialization", genuine_source):
            original = build_transfer(tmp_path_factory)
        endpoint, _mlp, source, precision = build_precision(original, tmp_path_factory)
        calibration = tmp_path_factory.mktemp("joint-state-calibration")
        run_calibration(
            CalibrationConfig(
                str(endpoint),
                str(source),
                str(precision),
                str(calibration),
                "spartan",
                sha(endpoint / "last.pt"),
                chunk_size=3,
                max_iterations=2,
                max_evaluations=6,
                verification=True,
            )
        )
        return source, endpoint, calibration, precision
    finally:
        torch.set_num_threads(old)


def test_source_authenticates_calibrated_endpoint_data_law_and_distinct_state_bytes(
    joint_sources: tuple[Path, Path, Path, Path],
    tmp_path: Path,
) -> None:
    source, endpoint, calibration, precision = joint_sources
    inputs = load_inputs(
        source,
        endpoint,
        calibration,
        precision,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        calibrated_sha256=sha(calibration / "composed_model.pt"),
        device=torch.device("cpu"),
        verification=True,
    )
    assert inputs.source_step == 2
    assert inputs.data_config == data_config()
    assert module_fingerprint(inputs.model.online_encoder) != module_fingerprint(
        inputs.model.target_encoder
    )
    for name in ("online", "target"):
        assert (
            module_fingerprint(getattr(inputs.model, f"{name}_encoder"))
            == inputs.bundle.evaluation["metadata"][f"state_{name}_sha256"]
        )
    train, _ = regenerate_train(
        inputs.model, inputs.bundle.train, inputs.bundle.evaluation["metadata"], inputs.data_config
    )
    assert set(train) == {"observations", "actions"}
    assert all(sha(Path(path)) == value for path, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="pinned"):
        load_inputs(
            source,
            endpoint,
            calibration,
            precision,
            checkpoint_sha256="f" * 64,
            calibrated_sha256=sha(calibration / "composed_model.pt"),
            device=torch.device("cpu"),
            verification=True,
        )

    record = cast(dict[str, Any], read_json(source / "original/run_config.json"))
    record["data"]["root_seed"] += 1
    write_json(tmp_path / "changed-data.json", record)
    with pytest.raises(ValueError, match="fingerprint"):
        load_inputs(
            source,
            endpoint,
            calibration,
            precision,
            checkpoint_sha256=sha(endpoint / "last.pt"),
            calibrated_sha256=sha(calibration / "composed_model.pt"),
            device=torch.device("cpu"),
            data_record=tmp_path / "changed-data.json",
            verification=True,
        )


def test_probe_selection_matches_original_six_code_assay_and_mcc_matching_is_calibration_only() -> (
    None
):
    generator = torch.Generator().manual_seed(621)
    codes = {
        name: torch.randn(8, 10, 6, generator=generator, dtype=torch.float64)
        for name in ("calibration", "validation", "test")
    }
    splits: EvaluationData = {
        name: {"theta": torch.randn(8, 6, generator=generator, dtype=torch.float64)}
        for name in codes
    }
    expected = parameter_readouts(codes, splits)
    probes = fit_probes(
        codes["calibration"],
        codes["validation"],
        splits["calibration"]["theta"],
        splits["validation"]["theta"],
    )
    predictions = predict(probes, codes["test"])
    actual, _ = score(predictions, splits["test"]["theta"])
    torch.testing.assert_close(
        torch.tensor(actual["r2"]), torch.tensor(expected["selected_test_r2"]), rtol=1e-6, atol=1e-7
    )
    assert [row["family"] for row in probes["selected"]] == expected[
        "selected_methods_per_parameter"
    ]
    truth = torch.randn(2, 10, 12, 6, generator=generator, dtype=torch.float64)
    perm = [2, 0, 5, 1, 3, 4]
    evaluation: EvaluationData = {
        "calibration": {"states_online": truth[..., perm], "true_states": truth},
        "test": {
            "states_online": truth,
            "states_target": truth + 1,
            "true_states": truth,
            "theta": torch.zeros(2, 6),
        },
    }
    result = state_metrics(evaluation, "test")
    assert result["calibration_assignment"] == [perm.index(i) for i in range(6)]
    assert result["mcc"] < 0.2  # Refitting on this test set would incorrectly give one.
    assert result["online_target_mse"] == 1
    before = torch.randn(7, 6, generator=generator, dtype=torch.float64)
    after = before + torch.linspace(-0.03, 0.02, 6, dtype=torch.float64)
    changes = paired_changes(
        {"response": {"predictive_benefit": before}},
        {"response": {"predictive_benefit": after}},
        bootstrap_samples=20,
        seed=51,
    )
    for index, row in enumerate(changes["final_minus_source_predictive_benefit"]):
        expected_change = float(after[0, index] - before[0, index])
        torch.testing.assert_close(torch.tensor(row["estimate"]), torch.tensor(expected_change))
        torch.testing.assert_close(
            torch.tensor(row["interval_95"]), torch.tensor([expected_change, expected_change])
        )


@pytest.mark.parametrize("arm", ["frozen_state", "live_state"])
@pytest.mark.parametrize("source_gate_mode", ["all_six_use", "qualified_spartan"])
def test_genuine_runner_reencodes_paired_inputs_uses_fresh_adamw_and_resumes_exactly(
    joint_sources: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
    source_gate_mode: Literal["all_six_use", "qualified_spartan"],
) -> None:
    source, endpoint, calibration, precision = joint_sources
    inputs = load_inputs(
        source,
        endpoint,
        calibration,
        precision,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        calibrated_sha256=sha(calibration / "composed_model.pt"),
        device=torch.device("cpu"),
        verification=True,
    )
    train, _ = regenerate_train(
        inputs.model, inputs.bundle.train, inputs.bundle.evaluation["metadata"], inputs.data_config
    )
    initial = copy.deepcopy(inputs.model.state_dict())
    active = tmp_path / "full"
    config = runner.Config(
        str(source),
        str(endpoint),
        str(calibration),
        str(precision),
        str(active),
        sha(endpoint / "last.pt"),
        sha(calibration / "composed_model.pt"),
        arm,
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=8,
        chunk_size=3,
        verification=True,
        source_gate_mode=source_gate_mode,
    )
    original_loss, original_monitor = runner.training_loss, runner.monitor
    original_precision, original_after = runner.precision_assessment, runner.after_update
    original_load = cast(Callable[..., Any], torch.load)
    calls, ema_calls, tests, precisions = 0, 0, 0, 0
    interrupt = False

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert "shared-confirmatory" not in str(file)
            assert Path(file).name != "fresh_test.pt"
        return original_load(file, *args, **kwargs)

    def checked_loss(
        model: SCJEPA, batch: TensorSplit, objective: TrainingObjective, given_arm: Arm
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal calls
        calls += 1
        assert given_arm == arm
        step = inputs.source_step + calls
        epoch, position = divmod(step - 1, 2)
        selected = list(GlobalBatchSampler(4, 2, 7, epoch))[position]
        torch.testing.assert_close(
            batch, {key: value[selected] for key, value in train.items()}, rtol=0, atol=0
        )
        assert set(batch) == {"observations", "actions"}
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(700000 + step).get_state()
        )
        if calls == 1:
            torch.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)
            checkpoint = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
            assert checkpoint["optimizer"]["state"] == {}
            group = checkpoint["optimizer"]["param_groups"][0]
            assert group["lr"] == 0.001
            assert group["weight_decay"] == 0
            assert len(group["params"]) == len([p for p in model.parameters() if p.requires_grad])
            assert checkpoint["step"] == inputs.source_step
        if interrupt and calls == 2:
            raise RuntimeError("joint-state operational interruption")
        return original_loss(model, batch, objective, given_arm)

    def checked_after(model: SCJEPA, given_arm: Arm) -> None:
        nonlocal ema_calls
        ema_calls += 1
        before = copy.deepcopy(model.target_encoder.state_dict())
        original_after(model, given_arm)
        for key, value in model.target_encoder.state_dict().items():
            expected = (
                before[key]
                if arm == "frozen_state"
                else before[key] * 0.996 + model.online_encoder.state_dict()[key] * 0.004
            )
            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)

    def checked_monitor(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        nonlocal tests
        if final:
            assert calls == 3
            tests += 1
        return original_monitor(
            model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    def checked_precision(
        model: SCJEPA,
        old_data: EvaluationData,
        precision_data: EvaluationData,
        *,
        bootstrap_samples: int,
        seed: int,
        chunk_size: int,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal precisions
        assert calls in (0, 3)
        precisions += 1
        torch.testing.assert_close(
            precision_data["validation"]["theta"],
            inputs.precision_data["validation"]["theta"],
            rtol=0,
            atol=0,
        )
        return original_precision(
            model,
            old_data,
            precision_data,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            chunk_size=chunk_size,
            progress=progress,
        )

    monkeypatch.setattr(torch, "load", guarded_load)
    monkeypatch.setattr(runner, "training_loss", checked_loss)
    monkeypatch.setattr(runner, "after_update", checked_after)
    monkeypatch.setattr(runner, "monitor", checked_monitor)
    monkeypatch.setattr(runner, "precision_assessment", checked_precision)
    full = runner.run(config)
    protocol = cast(dict[str, Any], read_json(active / "protocol.json"))
    assert protocol["config"]["source_gate_mode"] == source_gate_mode
    assert protocol["source_gate_mode"] == source_gate_mode
    source_report = full["source"]
    retained = source_report["original_full_validation_retention"]["all_six_above_0_80"]
    assert source_report["all_six_source_gate"] == (
        retained and response_gate(source_report["response"])
    )
    assert source_report["admission"]["mode"] == source_gate_mode
    assert source_report["admission"]["passed"] == (
        retained and admission_gate(source_report["response"], source_gate_mode)
    )
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 2)
    full_checkpoint = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    assert full_checkpoint["step"] == 5
    assert full_checkpoint["updates"] == 3
    assert full["endpoint"]["precision_validation"]["precision_retention"]["groups"] == 5
    if arm == "frozen_state":
        assert full["endpoint"]["state_test_mcc_change"] == 0
    else:
        assert full["endpoint"]["state_test_mcc_change"] != 0
    calls, ema_calls, tests, precisions = 0, 0, 0, 0
    active = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(active))
    interrupt = True
    with pytest.raises(RuntimeError, match="operational interruption"):
        runner.run(resumed_config)
    assert (calls, ema_calls, tests, precisions) == (2, 1, 0, 1)
    partial = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    assert partial["updates"] == 1
    calls = 1
    interrupt = False
    resumed = runner.run(resumed_config, resume=True)
    final_checkpoint = cast(dict[str, Any], original_load(active / "last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(full_checkpoint[key], final_checkpoint[key], rtol=0, atol=0)
    assert full["source"] == resumed["source"]
    assert full["records"] == resumed["records"]
    assert full["endpoint"] == resumed["endpoint"]
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 2)
    runner.run(resumed_config, resume=True)
    assert (calls, ema_calls, tests, precisions) == (3, 3, 1, 2)
    other_mode = "qualified_spartan" if source_gate_mode == "all_six_use" else "all_six_use"
    with pytest.raises(ValueError, match="resume|required|changed"):
        runner.run(replace(resumed_config, source_gate_mode=other_mode), resume=True)
    assert all(sha(Path(path)) == value for path, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(source / "nested"))
    changed = cast(dict[str, Any], original_load(active / "initial_model.pt", weights_only=False))
    changed["model"]["parameter_encoder.head.4.bias"][0] += 0.1
    torch.save(changed, active / "initial_model.pt")
    with pytest.raises(ValueError, match="initial model differs"):
        runner.run(resumed_config, resume=True)
