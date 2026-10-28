import copy
import shutil
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_joint_state_calibration import Arm
from test_joint_state_calibration import calibration_sources as calibration_sources
from test_joint_state_survival import joint_sources as joint_sources
from torch import Tensor, nn

from diagnostics.joint_dense_continuation.run import Config as DenseConfig
from diagnostics.joint_dense_continuation.run import run as run_dense
from diagnostics.multiquery_control.run import sha
from diagnostics.sparse_comparison import prepare as preparation
from diagnostics.sparse_comparison import run as runner
from diagnostics.sparse_comparison.source import Inputs, admission_status, load_inputs
from diagnostics.sparse_comparison.training import (
    CONDITIONS,
    OBJECTIVE_STREAM_OFFSET,
    Condition,
    assert_trainer,
    batch_at,
    make_trainer,
    step,
)
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import SpartanOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.loop import Trainer, TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def comparison_source(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Complete the genuine raw endpoint chain, retaining actual optimizer/evaluation artifacts."""
    endpoints, source, precision = calibration_sources
    endpoint = endpoints["live_state"]
    output = tmp_path_factory.mktemp("sparse-comparison-source")
    run_dense(
        DenseConfig(
            str(endpoint),
            str(source),
            str(precision),
            str(output),
            sha(endpoint / "last.pt"),
            steps=2,
            seed=7,
            groups=2,
            bootstrap_samples=8,
            chunk_size=3,
            max_iterations=2,
            max_evaluations=6,
            verification=True,
        )
    )
    return output


@pytest.fixture
def inputs(comparison_source: Path) -> Inputs:
    return load_inputs(comparison_source, device=torch.device("cpu"), verification=True)


def assert_optimizer_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(actual["state"], expected["state"], rtol=0, atol=0)
    assert actual["param_groups"] == expected["param_groups"]


def test_source_is_exact_raw_endpoint_and_admission_does_not_rewrite_benefit(
    comparison_source: Path, inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = cast(dict[str, Any], torch.load(comparison_source / "last.pt", weights_only=False))
    original_load = cast(Callable[..., Any], torch.load)

    def only_raw(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert (
                Path(file).resolve()
                != (comparison_source / "calibration/composed_model.pt").resolve()
            )
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", only_raw)
    reread = load_inputs(comparison_source, device=torch.device("cpu"), verification=True)
    torch.testing.assert_close(reread.model.state_dict(), raw["model"], rtol=0, atol=0)
    assert inputs.source_step == 6
    assert inputs.seed == 7
    assert inputs.batch_size == 2
    assert set(inputs.observations) == {"observations", "actions"}
    torch.testing.assert_close(inputs.model.state_dict(), raw["model"], rtol=0, atol=0)
    assert inputs.native_evaluation.seed == 42
    assert inputs.provenance["raw_source_only"] is True
    assert all(sha(Path(path)) == value for path, value in inputs.hashes.items())
    report = cast(dict[str, Any], read_json(comparison_source / "endpoint_assessment.json"))[
        "metrics"
    ]["precision_validation"]
    assert inputs.admission == admission_status(report)
    assert inputs.admission["historical_all_six_use_passed"] == report["all_six_source_gate"]
    with pytest.raises(ValueError, match="verification|identity"):
        load_inputs(comparison_source, device=torch.device("cpu"))
    damaged = tmp_path / "changed-source"
    shutil.copytree(comparison_source, damaged)
    changed = copy.deepcopy(raw)
    changed["model"]["parameter_encoder.head.4.bias"][0] += 0.1
    torch.save(changed, damaged / "last.pt")
    with pytest.raises(ValueError, match="hash|artifact|changed"):
        load_inputs(damaged, device=torch.device("cpu"), verification=True)


def test_response_admission_requires_all_actual_retention_values_and_preserves_failure() -> None:
    report: dict[str, Any] = {
        "response": {
            "per_parameter": [
                {
                    "parameter": index,
                    "exact_response_energy": 1.0,
                    "learned_response_energy": 0.9,
                    "normalized_response_error": {
                        "defined": True,
                        "estimate": 0.4,
                        "interval_95": [0.3, 0.5],
                    },
                    "predictive_benefit": {
                        "defined": True,
                        "estimate": -0.01 if index == 5 else 0.01,
                        "interval_95": [-0.02, 0.001] if index == 5 else [0.001, 0.02],
                    },
                }
                for index in range(1, 7)
            ]
        },
        "original_full_validation_retention": {
            "r2": [0.9] * 6,
            "minimum_r2": 0.9,
            "all_six_above_0_80": True,
        },
        "precision_retention": {
            "r2": [0.9] * 6,
            "minimum_r2": 0.9,
            "all_six_above_0_80": True,
        },
        "all_six_source_gate": False,
    }
    status = admission_status(report)
    assert status["passed"]
    assert not status["historical_all_six_use_passed"]
    assert status["unmet_benefit_parameters"] == [5]
    for bad_r2 in ([0.9] * 5, [0.9] * 5 + [0.7], [0.9] * 5 + [float("nan")]):
        changed = copy.deepcopy(report)
        changed["precision_retention"]["r2"] = bad_r2
        assert not admission_status(changed)["passed"]
    changed = copy.deepcopy(report)
    changed["response"]["per_parameter"][4]["exact_response_energy"] = 0.0
    assert not admission_status(changed)["passed"]
    changed = copy.deepcopy(report)
    changed["response"]["per_parameter"][4]["normalized_response_error"]["defined"] = False
    assert not admission_status(changed)["passed"]


@pytest.mark.parametrize("condition", CONDITIONS)
def test_native_update_matches_direct_graph_and_resumes_exactly(
    inputs: Inputs, tmp_path: Path, condition: Condition
) -> None:
    tau = 0.123
    trainer = make_trainer(inputs, condition, tmp_path / "full", tau=tau, steps=2)
    assert isinstance(trainer, Trainer)
    assert trainer.config.seed == 0
    assert trainer.optimizer.state == {}
    assert (trainer.step, trainer.optimizer_steps) == (0, 0)
    torch.testing.assert_close(
        trainer.model.state_dict(), inputs.model.state_dict(), rtol=0, atol=0
    )
    assert any(
        not torch.equal(value, trainer.model.target_encoder.state_dict()[name])
        for name, value in trainer.model.online_encoder.state_dict().items()
    )
    objective_config = trainer.config.objective
    assert (
        objective_config.state_sparsity_weight,
        objective_config.action_sparsity_weight,
        objective_config.parameter_sparsity_weight,
    ) == {"dense": (0, 0, 0), "baseline": (1, 1, 0), "full": (1, 1, 1)}[condition]
    assert (trainer.geco is None) == (condition == "dense")
    if trainer.geco is not None:
        assert trainer.geco.tau == tau
        assert int(trainer.geco.num_updates) == 0
        torch.testing.assert_close(trainer.geco.log_lambda.exp(), torch.tensor(1000.0))

    reference = copy.deepcopy(trainer.model)
    parameters = [parameter for parameter in reference.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    objective = TrainingObjective(trainer.config.objective, trainer.config.geco)
    global_step = inputs.source_step + 1
    epoch, position = divmod(global_step - 1, 2)
    indices = list(GlobalBatchSampler(4, 2, 7, epoch))[position]
    batch = batch_at(inputs, global_step)
    assert torch.equal(batch["episode_ids"], torch.tensor(indices))
    torch.testing.assert_close(
        batch["observations"], inputs.observations["observations"][indices], rtol=0, atol=0
    )
    donors: list[Tensor] = []
    predictions: list[Tensor] = []
    histories: list[Tensor] = []

    def capture(_module: nn.Module, args: tuple[object, ...], output: SpartanOutput) -> None:
        histories.append(cast(Tensor, args[0]))
        donors.append(cast(Tensor, args[1]))
        predictions.append(output.prediction)

    hook = reference.predictor.register_forward_hook(capture)
    reference.train()
    torch.default_generator.manual_seed(inputs.seed * 100000 + global_step)
    try:
        output = SCJEPA.forward(reference, batch["observations"], batch["actions"])
    finally:
        hook.remove()
    assert donors[0] is donors[1]
    assert donors[0].requires_grad
    torch.testing.assert_close(histories[1], predictions[0][:, None], rtol=0, atol=0)
    assert (
        float(
            torch.autograd.grad(predictions[1].square().mean(), predictions[0], retain_graph=True)[
                0
            ]
            .square()
            .sum()
        )
        > 0
    )
    for predicted in predictions:
        assert (
            float(
                torch.autograd.grad(predicted.square().mean(), donors[0], retain_graph=True)[0]
                .square()
                .sum()
            )
            > 0
        )
    torch.default_generator.manual_seed(
        inputs.seed * 100000 + global_step + OBJECTIVE_STREAM_OFFSET
    )
    loss, terms = objective(output)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    previous_target = copy.deepcopy(reference.target_encoder.state_dict())
    reference.update_target()
    if objective.geco is not None:
        objective.geco.update(terms["base"].detach())
    expected_rng = torch.get_rng_state().clone()
    actual = step(trainer, inputs)
    assert actual["loss/total"] == float(loss.detach())
    assert actual["train/global_step"] == global_step
    assert actual["health/updated"] == 1
    assert actual["health/skipped"] == 0
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(trainer.model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    assert_optimizer_equal(trainer.optimizer.state_dict(), optimizer.state_dict())
    torch.testing.assert_close(
        trainer.objective.state_dict(), objective.state_dict(), rtol=0, atol=0
    )
    for name, value in trainer.model.target_encoder.state_dict().items():
        torch.testing.assert_close(
            value,
            previous_target[name] * 0.996 + trainer.model.online_encoder.state_dict()[name] * 0.004,
            rtol=1e-6,
            atol=1e-7,
        )
    assert all(parameter.grad is None for parameter in trainer.model.target_encoder.parameters())
    if trainer.geco is not None:
        assert int(trainer.geco.num_updates) == 1
    checkpoint = tmp_path / "partial.pt"
    trainer.save_checkpoint(checkpoint)
    second = step(trainer, inputs)
    final_rng = torch.get_rng_state().clone()
    restored = make_trainer(inputs, condition, tmp_path / "resumed", tau=tau, steps=2)
    restored.load_checkpoint(checkpoint)
    assert_trainer(restored, inputs, condition, tau=tau)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    replay = step(restored, inputs)
    assert second == replay
    assert torch.equal(torch.get_rng_state(), final_rng)
    torch.testing.assert_close(
        restored.model.state_dict(), trainer.model.state_dict(), rtol=0, atol=0
    )
    assert_optimizer_equal(restored.optimizer.state_dict(), trainer.optimizer.state_dict())
    torch.testing.assert_close(
        restored.objective.state_dict(), trainer.objective.state_dict(), rtol=0, atol=0
    )
    assert_trainer(restored, inputs, condition, tau=tau)
    other: Condition = "baseline" if condition == "full" else "full"
    mismatched = make_trainer(inputs, other, tmp_path / "wrong", tau=tau, steps=2)
    with pytest.raises(ValueError, match="model_config"):
        mismatched.load_checkpoint(checkpoint)
    restored.run_metadata["stream_seed"] = inputs.seed + 1
    with pytest.raises(ValueError, match="stream"):
        assert_trainer(restored, inputs, condition, tau=tau)
    restored.run_metadata["stream_seed"] = inputs.seed
    if condition != "dense":
        restored.config.geco.tau = tau + 0.01
        with pytest.raises(ValueError, match="GECO"):
            assert_trainer(restored, inputs, condition, tau=tau)
        restored.config.geco.tau = tau
    restored.optimizer.state.clear()
    with pytest.raises(ValueError, match="coverage"):
        assert_trainer(restored, inputs, condition, tau=tau)


def test_conditions_match_groups_donors_gate_draws_and_both_sigreg_streams(
    inputs: Inputs, tmp_path: Path
) -> None:
    histories: dict[Condition, dict[str, Any]] = {}
    for condition in CONDITIONS:
        trainer = make_trainer(inputs, condition, tmp_path / condition, tau=0.123, steps=1)
        donors: list[Tensor] = []
        regulator_rng: list[Tensor] = []
        gate_rng: list[Tensor] = []

        def donor(_module: nn.Module, args: tuple[object, ...], _output: object) -> None:
            donors.append(cast(Tensor, args[1]).detach().clone())  # noqa: B023

        def regulator(_module: nn.Module, _args: tuple[object, ...]) -> None:
            regulator_rng.append(torch.get_rng_state().clone())  # noqa: B023

        def gate(_module: nn.Module, _args: tuple[object, ...]) -> None:
            gate_rng.append(torch.get_rng_state().clone())  # noqa: B023

        hooks = [
            trainer.model.predictor.register_forward_hook(donor),
            trainer.objective.state_regularizer.register_forward_pre_hook(regulator),
            trainer.objective.parameter_regularizer.register_forward_pre_hook(regulator),
            *[layer.register_forward_pre_hook(gate) for layer in trainer.model.predictor.layers],
        ]
        try:
            metrics = step(trainer, inputs)
        finally:
            for hook in hooks:
                hook.remove()
        histories[condition] = {
            "donors": donors,
            "regulators": regulator_rng,
            "gates": gate_rng,
            "metrics": metrics,
        }
    for condition in CONDITIONS:
        assert len(histories[condition]["donors"]) == 2
        assert len(histories[condition]["regulators"]) == 2
        torch.testing.assert_close(
            histories[condition]["donors"], histories["dense"]["donors"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            histories[condition]["regulators"], histories["dense"]["regulators"], rtol=0, atol=0
        )
        for name in ("state_regularizer", "parameter_regularizer", "parameter_consistency"):
            assert (
                histories[condition]["metrics"]["loss/" + name]
                == histories["dense"]["metrics"]["loss/" + name]
            )
    torch.testing.assert_close(
        histories["baseline"]["gates"], histories["full"]["gates"], rtol=0, atol=0
    )


def test_shared_full_base_preparation_and_fixed_final_runner_resume(
    comparison_source: Path, inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = tmp_path / "shared"
    prepare_config = preparation.Config(str(comparison_source), str(shared), verification=True)
    source_state = copy.deepcopy(inputs.model.state_dict())
    created: list[Condition] = []
    original_make = preparation.make_trainer

    def checked_make(
        given: Inputs, condition: Condition, output: Path, *, tau: float, steps: int
    ) -> Trainer:
        trainer = original_make(given, condition, output, tau=tau, steps=steps)
        assert trainer.optimizer.state == {}
        assert trainer.optimizer_steps == 0
        torch.testing.assert_close(trainer.model.state_dict(), source_state, rtol=0, atol=0)
        created.append(condition)
        return trainer

    monkeypatch.setattr(preparation, "make_trainer", checked_make)
    prepared = preparation.run(prepare_config)
    assert created == list(CONDITIONS)
    tau = prepared["tau"]["tau"]
    assert tau == prepared["tau"]["calibration/base_loss"]
    assert tau > prepared["tau"]["calibration/prediction_mse"]
    assert prepared["tau"]["tau_multiplier"] == 1
    assert prepared["tau"]["calibration"]["seed"] == 42
    assert prepared["tau"]["calibration"]["batch_size"] == 2
    assert prepared["tau"]["calibration"]["calibration_episodes"] == 4
    assert prepared["source_unchanged"]
    initial = cast(dict[str, Any], torch.load(shared / "initial_model.pt", weights_only=False))
    torch.testing.assert_close(initial["model"], source_state, rtol=0, atol=0)
    assert not list(shared.rglob("last.pt"))
    for condition in CONDITIONS:
        trial = prepared["preflight"][condition]
        assert trial["discarded"]
        assert trial["fresh_optimizer"]
        assert trial["ema_verified"]
        assert trial["metrics"]["health/optimizer_steps"] == 1
        assert trial["geco_updates"] == (0 if condition == "dense" else 1)
    shared_hashes = cast(dict[str, Any], read_json(shared / "completed.json"))["hashes"]
    assert all(sha(shared / name) == value for name, value in shared_hashes.items())
    assert preparation.run(prepare_config, resume=True) == prepared
    assert created == list(CONDITIONS)

    active = tmp_path / "full"
    config = runner.Config(
        str(shared),
        str(active),
        "full",
        steps=3,
        groups=2,
        bootstrap_samples=8,
        chunk_size=3,
        verification=True,
    )
    original_step, original_final = runner.step, runner.final
    calls, finals = 0, 0
    interrupt = False

    def checked_step(trainer: Trainer, given: Inputs) -> dict[str, float]:
        nonlocal calls
        calls += 1
        assert trainer.step == calls - 1
        if calls == 1:
            checkpoint = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
            assert checkpoint["step"] == checkpoint["optimizer_steps"] == 0
            assert checkpoint["optimizer"]["state"] == {}
            torch.testing.assert_close(checkpoint["model"], source_state, rtol=0, atol=0)
        if interrupt and calls == 2:
            raise RuntimeError("sparse comparison interruption")
        return original_step(trainer, given)

    def checked_final(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[str]]:  # noqa: ANN401
        nonlocal finals
        assert calls == 3
        assert kwargs["step"] == kwargs["fixed_budget"] == 3
        assert kwargs["condition"] == "full"
        assert kwargs["tau"] == tau
        finals += 1
        return original_final(*args, **kwargs)

    monkeypatch.setattr(runner, "step", checked_step)
    monkeypatch.setattr(runner, "final", checked_final)
    full = runner.run(config)
    full_checkpoint = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
    assert calls == 3
    assert finals == 1
    assert full["steps"] == 3
    assert full["global_step"] == inputs.source_step + 3
    assert full_checkpoint["optimizer_steps"] == 3
    assert int(full_checkpoint["objective"]["geco.num_updates"]) == 3
    assert full_checkpoint["run_metadata"]["stream_seed"] == inputs.seed
    assert full_checkpoint["run_metadata"]["bootstrap_seed"] == 0
    completion = cast(dict[str, Any], read_json(active / "completed.json"))
    assert all(sha(active / name) == value for name, value in completion["hashes"].items())

    active = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(active))
    calls, finals, interrupt = 0, 0, True
    with pytest.raises(RuntimeError, match="sparse comparison interruption"):
        runner.run(resumed_config)
    partial = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
    assert partial["step"] == 1
    assert finals == 0
    saved = (active / "last.pt").read_bytes()
    changed = copy.deepcopy(partial)
    changed["model"]["parameter_encoder.head.4.bias"][0] += 0.01
    torch.save(changed, active / "last.pt")
    with pytest.raises(ValueError, match="hash"):
        runner.run(resumed_config, resume=True)
    assert calls == 2
    (active / "last.pt").write_bytes(saved)
    calls, interrupt = 1, False
    resumed = runner.run(resumed_config, resume=True)
    resumed_checkpoint = cast(dict[str, Any], torch.load(active / "last.pt", weights_only=False))
    assert (calls, finals) == (3, 1)
    torch.testing.assert_close(
        resumed_checkpoint["model"], full_checkpoint["model"], rtol=0, atol=0
    )
    assert_optimizer_equal(resumed_checkpoint["optimizer"], full_checkpoint["optimizer"])
    torch.testing.assert_close(
        resumed_checkpoint["objective"], full_checkpoint["objective"], rtol=0, atol=0
    )
    assert torch.equal(
        resumed_checkpoint["rank_rng"][0]["rng_torch"], full_checkpoint["rank_rng"][0]["rng_torch"]
    )
    assert resumed["records"] == full["records"]
    assert full["final"]["checkpoint_sha256"] == sha(Path(config.output_dir) / "last.pt")
    assert resumed["final"]["checkpoint_sha256"] == sha(active / "last.pt")
    # Native checkpoints include their different output paths; scientific values stay exact.
    assert {
        key: value for key, value in resumed["final"].items() if key != "checkpoint_sha256"
    } == {key: value for key, value in full["final"].items() if key != "checkpoint_sha256"}
    assert runner.run(resumed_config, resume=True) == resumed
    assert (calls, finals) == (3, 1)
    with pytest.raises(ValueError, match="identity|resume"):
        runner.run(replace(resumed_config, condition="baseline"), resume=True)
    with pytest.raises(ValueError, match="identity|resume"):
        runner.run(replace(resumed_config, steps=2), resume=True)
    assert all(sha(Path(path)) == value for path, value in inputs.hashes.items())
    assert all(sha(shared / name) == value for name, value in shared_hashes.items())
    with pytest.raises(ValueError, match="fixes30000"):
        replace(config, verification=False)
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(shared / "nested"))
