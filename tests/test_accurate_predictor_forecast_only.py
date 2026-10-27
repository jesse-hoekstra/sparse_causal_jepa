from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_accurate_predictor_acquisition import one_thread as one_thread
from test_accurate_predictor_acquisition import source_roots as original_source_roots
from torch import Tensor

from diagnostics.accurate_predictor_acquisition.source import load_inputs
from diagnostics.accurate_predictor_forecast_only import run as runner
from diagnostics.accurate_predictor_forecast_only.run import Config, fixed_hashes, run
from diagnostics.joint_acquisition.baseline import DiagnosticSelection
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.forward import training_loss as original_training_loss
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit, module_fingerprint
from diagnostics.multiquery_control.run import sha
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig

source_roots = original_source_roots


def test_only_forecast_gradients_are_optimized_with_identical_rng_and_reporting(
    source_roots: tuple[Path, Path],
) -> None:
    inputs = load_inputs(*source_roots, 7, verification=True)
    model = inputs.model.train()
    model.parameter_encoder.requires_grad_(True)
    parameters = tuple(model.parameter_encoder.parameters())
    batch = {key: value[:2] for key, value in inputs.bundle.train.items()}
    objective = AcquisitionObjective(ObjectiveConfig(**inputs.bundle.protocol["objective_config"]))
    torch.default_generator.manual_seed(717)
    original, terms = original_training_loss(model, batch, objective)
    expected_rng = torch.get_rng_state()
    expected = torch.autograd.grad(terms["prediction"], parameters, retain_graph=True)
    full_gradient = torch.autograd.grad(original, parameters)
    torch.default_generator.manual_seed(717)
    actual, reported = runner.training_loss(model, batch, objective)
    actual_gradient = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, terms["prediction"], rtol=0, atol=0)
    for observed, reference in zip(actual_gradient, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert (
        sum(float((x - y).square().sum()) for x, y in zip(full_gradient, expected, strict=True))
        > 1e-12
    )
    for name, value in terms.items():
        if name != "optimized_total":
            torch.testing.assert_close(reported[name], value, rtol=0, atol=0)
    torch.testing.assert_close(reported["optimized_total"], actual, rtol=0, atol=0)
    torch.testing.assert_close(reported["original_total_without_constant_state_sigreg"], original)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(parameter.grad is None for parameter in model.predictor.parameters())


def test_encoder_only_training_resumes_exactly_with_fixed_final_test(
    source_roots: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, oracle = source_roots
    inputs = load_inputs(source, oracle, 7, verification=True)
    frozen = fixed_hashes(inputs.model)
    config = Config(
        str(source),
        str(oracle),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    original_assess = runner.assess_model
    calls: list[bool] = []

    def assessment(
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

    monkeypatch.setattr(runner, "assess_model", assessment)
    full = run(config)
    assert calls == [False] * 4 + [True]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    initial_encoder = module_fingerprint(inputs.model.parameter_encoder)
    inputs.model.load_state_dict(checkpoint["model"])
    assert fixed_hashes(inputs.model) == frozen
    assert module_fingerprint(inputs.model.parameter_encoder) != initial_encoder
    assert len(checkpoint["optimizer"]["param_groups"][0]["params"]) == len(
        tuple(inputs.model.parameter_encoder.parameters())
    )
    original_loss = runner.training_loss
    updates = 0

    def interrupted(
        model: SCJEPA, batch: TensorSplit, objective: AcquisitionObjective
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert set(batch) == {"online", "target", "actions"}
        assert all(not parameter.requires_grad for parameter in model.predictor.parameters())
        if updates == 2:
            raise RuntimeError("interrupted E training")
        with pytest.raises(ValueError, match="exclude physical labels"):
            original_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective)
        return original_loss(model, batch, objective)

    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    monkeypatch.setattr(runner, "training_loss", interrupted)
    calls.clear()
    with pytest.raises(RuntimeError, match="interrupted E training"):
        run(resumed_config)
    assert not any(calls)
    assert not (tmp_path / "resumed/final_test.json").exists()
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    for name in ("model", "optimizer"):
        torch.testing.assert_close(checkpoint[name], final[name], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["rng"]["torch"], final["rng"]["torch"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(Path(path)) == expected for path, expected in inputs.hashes.items())
    before = len(calls)
    assert run(resumed_config, resume=True) == resumed
    assert len(calls) == before
