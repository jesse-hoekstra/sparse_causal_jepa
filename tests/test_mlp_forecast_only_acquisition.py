import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_group_latent_runner import source_with_initialization
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor

from diagnostics.encoder_gradient_control.source import build_source_model
from diagnostics.group_latent_control.run import initialize_control
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.joint_acquisition.source import load_primary_source
from diagnostics.mlp_acquisition_control import run as original_runner
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.forward import training_loss as original_training_loss
from diagnostics.mlp_forecast_only_acquisition import run as runner
from diagnostics.mlp_forecast_only_acquisition.run import Config, fixed_hashes, run
from diagnostics.mlp_oracle_code_control.model import initialize_mlp
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from scjepa.experiments.storage import read_json
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_joint_forecast_gradients_match_original_component_and_preserve_rng() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    parameters = (*model.parameter_encoder.parameters(), *model.predictor.parameters())
    encoder_count = len(tuple(model.parameter_encoder.parameters()))
    _, batch = make_batch(model)
    objective = AcquisitionObjective(objective_config())
    torch.default_generator.manual_seed(717)
    original, terms = original_training_loss(model, batch, objective)
    expected_rng = torch.get_rng_state()
    expected = torch.autograd.grad(terms["prediction"], parameters, retain_graph=True)
    original_gradients = torch.autograd.grad(original, parameters)
    torch.default_generator.manual_seed(717)
    actual, reported = runner.training_loss(model, batch, objective)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, terms["prediction"], rtol=0, atol=0)
    for observed, reference in zip(gradients, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert sum(float(value.square().sum()) for value in gradients[:encoder_count]) > 0
    assert sum(float(value.square().sum()) for value in gradients[encoder_count:]) > 0
    assert (
        sum(
            float((first - second).square().sum())
            for first, second in zip(
                original_gradients[:encoder_count], expected[:encoder_count], strict=True
            )
        )
        > 1e-12
    )
    for first, second in zip(
        original_gradients[encoder_count:], expected[encoder_count:], strict=True
    ):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, value in terms.items():
        if name != "optimized_total":
            torch.testing.assert_close(reported[name], value, rtol=0, atol=0)
    torch.testing.assert_close(reported["optimized_total"], actual, rtol=0, atol=0)
    torch.testing.assert_close(
        reported["ordinary_total_without_constant_state_sigreg"], original, rtol=0, atol=0
    )
    assert torch.equal(torch.get_rng_state(), expected_rng)
    with pytest.raises(ValueError, match="exclude physical labels"):
        runner.training_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective)


def test_fresh_joint_source_sampler_exact_resume_and_fixed_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_with_initialization(tmp_path / "source")
    bundle = load_primary_source(source, 7, expected_step=2, expected_checkpoint_sha256=None)
    source_model, _ = build_source_model(bundle, torch.device("cpu"))
    _, _, hashes = initialize_control(source_model, bundle, "fresh_zero")
    frozen = fixed_hashes(source_model)
    initialize_mlp(source_model, 7)
    archived = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    config = Config(
        str(source),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_load = cast(Callable[..., Any], torch.load)

    def guarded_load(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name != "audit_inputs.pt", "Joint source loaded training theta"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded_load)
    original_loss = runner.training_loss
    samples: list[tuple[TensorSplit, Tensor]] = []

    def capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        assert set(batch) == {"online", "target", "actions"}
        assert fixed_hashes(model) == frozen
        assert all(parameter.requires_grad for parameter in model.parameter_encoder.parameters())
        assert all(parameter.requires_grad for parameter in model.predictor.parameters())
        samples.append(
            ({name: value.clone() for name, value in batch.items()}, torch.get_rng_state())
        )
        return original_loss(model, batch, objective)

    calls: list[bool] = []
    original_assess = runner.assess_model

    def assess(
        model: SCJEPA,
        data: EvaluationData,
        selection: DiagnosticSelection,
        *,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        calls.append(final)
        assert fixed_hashes(model) == frozen
        guarded = data if final else {key: value for key, value in data.items() if key != "test"}
        return original_assess(
            model, guarded, selection, bootstrap_samples=bootstrap_samples, seed=seed, final=final
        )

    monkeypatch.setattr(runner, "training_loss", capture)
    monkeypatch.setattr(runner, "assess_model", assess)
    full = run(config)
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    initial = cast(
        dict[str, Any], torch.load(tmp_path / "full/initial_modules.pt", weights_only=False)
    )
    for name, value in initial["parameter_encoder"].items():
        torch.testing.assert_close(
            value, archived["model"]["parameter_encoder." + name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        initial["predictor"], source_model.predictor.state_dict(), rtol=0, atol=0
    )
    restored = copy.deepcopy(source_model)
    restored.load_state_dict(checkpoint["model"])
    assert fixed_hashes(restored) == frozen
    for prefix in ("predictor", "parameter_encoder"):
        assert any(
            not torch.equal(value, initial[prefix][name])
            for name, value in getattr(restored, prefix).state_dict().items()
        )
    protocol = cast(dict[str, Any], read_json(tmp_path / "full/protocol.json"))
    assert protocol["training_parameter_labels_absent"] is True
    assert len(full["final_test"]["retention"]["selected_test_group_mse_by_parameter"]) == len(
        bundle.evaluation["test"]["theta"]
    )
    assert len(full["final_test"]["response"]["group_indices"]) == 2
    assert len(samples) == 3

    # Re-run the original regularized arm: source weights and batch/RNG policy must match.
    baseline_samples: list[tuple[TensorSplit, Tensor]] = []

    def baseline_capture(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        baseline_samples.append(
            ({name: value.clone() for name, value in batch.items()}, torch.get_rng_state())
        )
        return original_training_loss(model, batch, objective)

    monkeypatch.setattr(original_runner, "training_loss", baseline_capture)
    original_runner.run(
        original_runner.Config(
            str(source),
            str(tmp_path / "baseline"),
            seed=7,
            steps=3,
            groups=2,
            bootstrap_samples=16,
            verification=True,
        )
    )
    for actual_sample, baseline_sample in zip(samples, baseline_samples, strict=True):
        torch.testing.assert_close(actual_sample, baseline_sample, rtol=0, atol=0)
    baseline_initial = cast(
        dict[str, Any], torch.load(tmp_path / "baseline/initial_modules.pt", weights_only=False)
    )
    for key in ("parameter_encoder", "predictor"):
        torch.testing.assert_close(initial[key], baseline_initial[key], rtol=0, atol=0)

    updates = 0

    def interrupt(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert set(batch) == {"online", "target", "actions"}
        if updates == 2:
            raise RuntimeError("interrupted joint forecast training")
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupt)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted joint forecast training"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for key in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[key], final[key], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(source / name) == value for name, value in hashes.items())
    count = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == count
    with pytest.raises(ValueError, match="Resume settings"):
        run(replace(resumed_config, bootstrap_samples=17), resume=True)
