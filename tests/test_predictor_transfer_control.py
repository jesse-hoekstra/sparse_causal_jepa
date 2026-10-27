import copy
import shutil
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_batch, make_model, objective_config
from test_overcomplete_forward import batch_for, gained_model
from torch import Tensor, nn

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_oracle_code_control.model import MLPPredictor, Prediction, initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.evaluation import encode_codes
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from diagnostics.multiquery_control.run import sha
from diagnostics.overcomplete_acquisition.model import (
    OvercompletePredictor,
    initialize_overcomplete,
)
from diagnostics.overcomplete_acquisition.run import Config as ParentConfig
from diagnostics.overcomplete_acquisition.run import run as run_parent
from diagnostics.predictor_transfer_control import run as runner
from diagnostics.predictor_transfer_control.forward import training_loss
from diagnostics.predictor_transfer_control.model import (
    audit_model_fingerprint,
    compose_pca,
    fit_pca,
)
from diagnostics.predictor_transfer_control.source import fixed_hashes, load_inputs, verify_sources
from diagnostics.shared_feature_acquisition.run import Config as SharedConfig
from diagnostics.shared_feature_acquisition.run import run as run_shared
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.models.spartan import Spartan, SpartanOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.loop import TrainingObjective

Arm = Literal["spartan", "mlp"]


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_train_pca_and_affine_composition_preserve_individual_context_and_two_step_map() -> None:
    scale = torch.arange(1, 13, dtype=torch.float64)
    flat = torch.zeros(40, 12, dtype=torch.float64)
    flat[:12] = torch.diag(scale)
    flat[12:24] = -torch.diag(scale)
    center = torch.arange(12, dtype=torch.float64) / 10
    mean, basis, values = fit_pca((flat + center).reshape(4, 10, 12))
    torch.testing.assert_close(mean, center)
    torch.testing.assert_close(values, (2 * scale.square() / 40).flip(0))
    torch.testing.assert_close(
        basis @ basis.T, torch.diag(torch.cat((torch.zeros(6), torch.ones(6)))).double()
    )
    model = gained_model()
    initialize_overcomplete(model, 17)
    batch = {name: value.double() for name, value in batch_for(model).items()}
    model.double().eval()
    before = audit_model_fingerprint(model)
    projected = compose_pca(model, mean, basis)
    states = batch["online"][:, :, :10].flatten(0, 1)
    actions = batch["actions"][:, :, :9].flatten(0, 1)
    original_code = model.parameter_encoder(states, actions)
    code = projected.parameter_encoder(states, actions)
    torch.testing.assert_close(code, (original_code - mean) @ basis, rtol=1e-12, atol=1e-12)
    assert not torch.equal(code[0], code[1])
    reconstructed = code @ basis.T + mean
    original_history = history = batch["online"][:, :, 9:10].flatten(0, 1)
    for step in (9, 10):
        action = batch["actions"][:, :, step : step + 1].flatten(0, 1)
        actual = projected.predictor(history, code, action).prediction
        expected = model.predictor(original_history, reconstructed, action).prediction
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        history, original_history = actual[:, None], expected[:, None]
    assert audit_model_fingerprint(model) == before


@pytest.mark.parametrize("arm", ["spartan", "mlp"])
def test_predictor_loss_matches_original_gradient_rng_and_attached_rollout(arm: Arm) -> None:
    model = make_model()
    if arm == "mlp":
        initialize_mlp(model, 17)
    else:
        assert all(layer.dense for layer in model.predictor.layers)
    model.parameter_encoder.requires_grad_(False).eval()
    _, batch = make_batch(model)
    expected_model = copy.deepcopy(model)
    frozen = [
        module_fingerprint(module)
        for module in (model.parameter_encoder, model.online_encoder, model.target_encoder)
    ]
    outputs: list[Tensor] = []
    donors: list[Tensor] = []

    def capture(
        _module: nn.Module, args: tuple[object, ...], output: Prediction | SpartanOutput
    ) -> None:
        output.prediction.retain_grad()
        outputs.append(output.prediction)
        assert isinstance(args[1], Tensor)
        donors.append(args[1].detach().clone())

    hook = model.predictor.register_forward_hook(capture)
    torch.default_generator.manual_seed(293)
    try:
        actual, terms = training_loss(model, batch, AcquisitionObjective(objective_config()))
    finally:
        hook.remove()
    final_rng = torch.get_rng_state()
    assert len(outputs) == 2
    torch.testing.assert_close(donors[0], donors[1], rtol=0, atol=0)
    derivative = torch.autograd.grad(outputs[1].square().mean(), outputs[0], retain_graph=True)[0]
    assert float(derivative.square().sum()) > 0
    parameters = list(model.predictor.parameters())
    actual_gradients = torch.autograd.grad(actual, parameters, allow_unused=True)
    torch.default_generator.manual_seed(293)
    if arm == "spartan":
        output = cached_multiquery_forward(expected_model, batch, queries=1)
        full_loss, expected_terms = TrainingObjective(objective_config())(output)
        expected = expected_terms["prediction"] + 0.0001 * expected_terms["logit"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_gradients = torch.autograd.grad(
            full_loss, list(expected_model.predictor.parameters()), allow_unused=True
        )
    else:
        output = cached_forward(expected_model, batch)
        _, expected_terms = AcquisitionObjective(objective_config())(output)
        torch.testing.assert_close(actual, expected_terms["prediction"], rtol=0, atol=0)
        expected_gradients = torch.autograd.grad(
            expected_terms["prediction"],
            list(expected_model.predictor.parameters()),
            allow_unused=True,
        )
    assert torch.equal(final_rng, torch.get_rng_state())
    for actual_gradient, expected_gradient in zip(
        cast(tuple[Tensor | None, ...], actual_gradients),
        cast(tuple[Tensor | None, ...], expected_gradients),
        strict=True,
    ):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(terms["prediction"], expected_terms["prediction"], rtol=0, atol=0)
    assert frozen == [
        module_fingerprint(module)
        for module in (model.parameter_encoder, model.online_encoder, model.target_encoder)
    ]
    assert all(
        parameter.grad is None
        for module in (model.parameter_encoder, model.online_encoder, model.target_encoder)
        for parameter in module.parameters()
    )
    with pytest.raises(ValueError, match="label-free"):
        training_loss(
            model, {**batch, "theta": torch.zeros(2, 6)}, AcquisitionObjective(objective_config())
        )
    model.parameter_encoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        training_loss(model, batch, AcquisitionObjective(objective_config()))


@pytest.fixture(scope="module")
def transfer_sources(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Create a genuinely trained tiny endpoint and a deterministic authenticated PCA artifact."""
    root = tmp_path_factory.mktemp("predictor-transfer")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = source_with_initialization(root / "source")
        parent, endpoint, audit = root / "parent", root / "endpoint", root / "audit"
        run_parent(
            ParentConfig(
                str(source),
                str(parent),
                seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=8,
                verification=True,
            )
        )
        run_shared(
            SharedConfig(
                str(source),
                str(parent),
                str(endpoint),
                source_seed=7,
                seed=7,
                evaluation_seed=7,
                steps=2,
                groups=2,
                bootstrap_samples=8,
                verification=True,
            )
        )
        bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
        config = copy.deepcopy(bundle.protocol["model_config"])
        config["param_dim"] = 12
        model = build_model(cast(DictConfig, OmegaConf.create(config)))
        model.predictor = cast(Spartan, OvercompletePredictor())
        checkpoint = cast(dict[str, Any], torch.load(endpoint / "last.pt", weights_only=False))
        model.load_state_dict(checkpoint["model"])
        model.requires_grad_(False).eval()
        source_fingerprint = audit_model_fingerprint(model)
        codes = encode_codes(
            model,
            {
                "states_online": bundle.train["online"],
                "actions": bundle.train["actions"],
            },
        )
        mean, basis, values = fit_pca(codes)
        composed = compose_pca(model.double(), mean, basis)
        audit.mkdir()
        torch.save(
            {
                "mean": mean,
                "basis": basis,
                "eigenvalues": values,
                "composed_model": composed.state_dict(),
            },
            audit / "measurements.pt",
        )
        completion = cast(dict[str, Any], read_json(endpoint / "completed.json"))
        hashes = {
            **{
                str((endpoint / name).resolve()): value
                for name, value in completion["hashes"].items()
            },
            **{
                str((source / name).resolve()): sha(source / name)
                for name in (
                    "protocol.json",
                    "train_inputs.pt",
                    "evaluation_data.pt",
                )
            },
        }
        write_json(
            audit / "results.json",
            {
                "measurement_sha256": sha(audit / "measurements.pt"),
                "source_model_sha256": source_fingerprint,
                "composed_model_sha256": audit_model_fingerprint(composed),
                "source_hashes": hashes,
                # This verification fixture exercises the gate contract; its tiny
                # training run makes no scientific parameter-recovery claim.
                "predeclared_transfer_gate": {"passed": True},
                "full_validation": {"pca6": {"selected_validation_r2": [0.9] * 6}},
            },
        )
        return source, endpoint, audit
    finally:
        torch.set_num_threads(previous)


def test_authenticated_transfer_uses_same_pca6_chart_and_fresh_arm_predictors(
    transfer_sources: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    source, endpoint, audit = transfer_sources
    arms = [
        load_inputs(source, endpoint, audit, 7, arm, verification=True)
        for arm in ("spartan", "mlp")
    ]
    first, second = arms
    assert isinstance(first.model.predictor, Spartan)
    assert all(layer.dense and not layer.identity for layer in first.model.predictor.layers)
    assert isinstance(second.model.predictor, MLPPredictor)
    assert fixed_hashes(first.model) == fixed_hashes(second.model)
    for inputs in arms:
        assert inputs.model.parameter_encoder.param_dim == 6
        assert next(inputs.model.parameters()).dtype == torch.float32
        assert all(not p.requires_grad and p.grad is None for p in inputs.model.parameters())
        assert not inputs.model.training
        assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
        replay = inputs.provenance["materialization_replay"]
        assert replay["code_rms_difference"] <= replay["code_rms_tolerance"]
        assert replay["forecast_rms_difference"] <= replay["forecast_rms_tolerance"]
        assert module_fingerprint(inputs.model.predictor) != module_fingerprint(
            inputs.reference_model.predictor
        )
        verify_sources(inputs)
    expected, _ = build_source_model(first.bundle, torch.device("cpu"))
    initialize_control(expected, first.bundle, "fresh_zero")
    torch.testing.assert_close(
        first.model.predictor.state_dict(), expected.predictor.state_dict(), rtol=0, atol=0
    )
    initialize_mlp(expected, 7)
    torch.testing.assert_close(
        second.model.predictor.state_dict(), expected.predictor.state_dict(), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="seed"):
        load_inputs(source, endpoint, audit, 17, "mlp", verification=True)
    # Falsifying either the declared gate or the saved scientific inputs must
    # reject the artifact before a new optimization run can begin.
    copied = tmp_path / "bad-audit"
    shutil.copytree(audit, copied)
    changed = cast(dict[str, Any], read_json(copied / "results.json"))
    changed["full_validation"]["pca6"]["selected_validation_r2"][4] = 0.7
    write_json(copied / "results.json", changed)
    with pytest.raises(ValueError, match="gate"):
        load_inputs(source, endpoint, copied, 7, "mlp", verification=True)
    reference_before = audit_model_fingerprint(second.reference_model)
    with torch.no_grad():
        next(second.model.parameter_encoder.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="Frozen"):
        verify_sources(second)
    assert audit_model_fingerprint(second.reference_model) == reference_before


@pytest.mark.parametrize("arm", ["spartan", "mlp"])
def test_frozen_encoder_fresh_predictor_only_updates_and_exact_resume(
    transfer_sources: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
) -> None:
    source, endpoint, audit = transfer_sources
    inputs = load_inputs(source, endpoint, audit, 7, arm, verification=True)
    initial_model = copy.deepcopy(inputs.model.state_dict())
    initial_fixed = fixed_hashes(inputs.model)
    config = runner.Config(
        str(source),
        str(endpoint),
        str(audit),
        str(tmp_path / "full"),
        arm,
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=8,
        verification=True,
    )
    loss, assess = runner.training_loss, runner.assess_model
    count, test_calls = 0, 0
    active_output = tmp_path / "full"

    def capture(
        model: SCJEPA,
        batch: TensorSplit,
        objective: AcquisitionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        assert fixed_hashes(model) == initial_fixed
        assert not any(
            p.requires_grad
            for name, module in model.named_children()
            if name != "predictor"
            for p in module.parameters()
        )
        assert set(batch) == {"online", "target", "actions"}
        epoch, position = divmod(count - 1, inputs.bundle.train_groups // inputs.bundle.batch_size)
        indices = list(
            GlobalBatchSampler(
                inputs.bundle.train_groups,
                inputs.bundle.batch_size,
                7,
                epoch,
            )
        )[position]
        torch.testing.assert_close(
            batch,
            {name: value[indices] for name, value in inputs.bundle.train.items()},
            rtol=0,
            atol=0,
        )
        assert torch.equal(
            torch.get_rng_state(), torch.Generator().manual_seed(7 * 100000 + count).get_state()
        )
        if count == 1:
            torch.testing.assert_close(model.state_dict(), initial_model, rtol=0, atol=0)
            zero = cast(dict[str, Any], torch.load(active_output / "last.pt", weights_only=False))
            assert zero["optimizer"]["state"] == {}
        return loss(model, batch, objective)

    def checked_assess(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        nonlocal test_calls
        if final:
            test_calls += 1
            saved = cast(dict[str, Any], torch.load(active_output / "last.pt", weights_only=False))
            assert saved["step"] == config.steps
        return assess(
            model, data, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", checked_assess)
    full = runner.run(config)
    assert count == 3
    assert test_calls == 1
    final = cast(dict[str, Any], torch.load(active_output / "last.pt", weights_only=False))
    assert all(int(v["step"]) == 3 for v in final["optimizer"]["state"].values())
    for name, value in initial_model.items():
        if not name.startswith("predictor."):
            torch.testing.assert_close(final["model"][name], value, rtol=0, atol=0)
    assert any(
        not torch.equal(value, final["model"][name])
        for name, value in initial_model.items()
        if name.startswith("predictor.") and isinstance(value, Tensor)
    )
    count, test_calls = 0, 0
    active_output = tmp_path / "resumed"

    def interrupt(
        model: SCJEPA,
        batch: TensorSplit,
        objective: AcquisitionObjective,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("operational predictor-transfer interruption")
        return loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(active_output))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    with pytest.raises(RuntimeError, match="operational predictor-transfer"):
        runner.run(resumed_config)
    assert test_calls == 0
    assert not (active_output / "final_test.json").exists()
    partial = cast(dict[str, Any], torch.load(active_output / "last.pt", weights_only=False))
    assert partial["step"] == 1
    monkeypatch.setattr(runner, "training_loss", loss)
    resumed = runner.run(resumed_config, resume=True)
    replay = cast(dict[str, Any], torch.load(active_output / "last.pt", weights_only=False))
    assert test_calls == 1
    for name in ("model", "optimizer"):
        torch.testing.assert_close(replay[name], final[name], rtol=0, atol=0)
    torch.testing.assert_close(replay["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert replay["rng"]["python"] == final["rng"]["python"]
    assert replay["rng"]["numpy"][0] == final["rng"]["numpy"][0]
    assert (replay["rng"]["numpy"][1] == final["rng"]["numpy"][1]).all()
    assert replay["rng"]["numpy"][2:] == final["rng"]["numpy"][2:]
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert full["reference_validation"] == resumed["reference_validation"]
    runner.run(resumed_config, resume=True)
    assert test_calls == 1
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(endpoint / "bad-output"))
    saved = cast(
        dict[str, Any], torch.load(active_output / "initial_modules.pt", weights_only=False)
    )
    cast(Tensor, saved["projection"]["basis"])[0, 0] += 1
    torch.save(saved, active_output / "initial_modules.pt")
    with pytest.raises(AssertionError):
        runner.run(resumed_config, resume=True)
