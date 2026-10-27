import copy
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model
from test_table_settling_control import fixture_sources
from torch import Tensor

from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control import run as original_runner
from diagnostics.oracle_code_control.evaluation import PARAMETER_STD, Split
from diagnostics.oracle_code_control.source import load_inputs
from diagnostics.oracle_code_no_logit import run as runner
from diagnostics.oracle_code_no_logit.run import Config, fixed_hashes, run, training_loss
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


def test_only_optimized_logit_gradient_changes_with_identical_forward_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model().requires_grad_(False)
    model.predictor.requires_grad_(True)
    _, batch = make_batch(model)
    theta = torch.randn(2, 6, dtype=torch.float64, requires_grad=True) * 0.1

    def forbidden(*_args: object, **_kwargs: object) -> Tensor:
        raise AssertionError("Parameter encoder must not infer the diagnostic code")

    monkeypatch.setattr(model.parameter_encoder, "forward", forbidden)
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    actual, terms = training_loss(model, batch, theta)
    rng_after = torch.get_rng_state()
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    original, original_terms = original_runner.training_loss(model, batch, theta)
    assert torch.equal(torch.get_rng_state(), rng_after)
    torch.testing.assert_close(actual, original_terms["prediction"], rtol=0, atol=0)
    torch.testing.assert_close(terms["logit"], original_terms["logit"], rtol=0, atol=0)
    torch.testing.assert_close(terms["original_optimized_total"], original, rtol=0, atol=0)
    assert terms["effective_logit_weight"].item() == 0
    parameters = tuple(model.predictor.parameters())
    forecast_gradient = torch.autograd.grad(actual, parameters, retain_graph=True)
    original_gradient = torch.autograd.grad(original, parameters, retain_graph=True)
    logit_gradient = torch.autograd.grad(terms["logit"], parameters, retain_graph=True)
    assert sum(float(value.square().sum()) for value in logit_gradient) > 0
    for forecast, baseline, penalty in zip(
        forecast_gradient, original_gradient, logit_gradient, strict=True
    ):
        torch.testing.assert_close(baseline, forecast + 0.0001 * penalty, rtol=1e-10, atol=1e-12)
    # Independent attached rollout catches loss detaches and code/step mistakes.
    codes = (theta / PARAMETER_STD)[:, None].expand(-1, 10, -1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1), codes, batch["actions"][:, :, 9:10].flatten(0, 1)
    )
    second = model.predictor(
        first.prediction[:, None], codes, batch["actions"][:, :, 10:11].flatten(0, 1)
    )
    predictions = torch.stack((first.prediction, second.prediction), 1).reshape(2, 10, 2, 6)
    expected = (predictions - batch["target"][:, :, 10:12]).square().mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta, retain_graph=True)[0],
        torch.autograd.grad(expected, theta, retain_graph=True)[0],
        rtol=1e-14,
        atol=1e-18,
    )
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    assert all(parameter.grad is None for parameter in model.parameter_encoder.parameters())
    with pytest.raises(ValueError, match="separate"):
        training_loss(model, {**batch, "theta": theta}, theta)


def test_completed_matched_reference_resume_and_final_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum, audit = fixture_sources(tmp_path, fresh=True)
    source = Path(
        cast(dict[str, Any], read_json(curriculum / "protocol.json"))["config"]["source_dir"]
    )
    reference = tmp_path / "reference"
    original_config = original_runner.Config(
        str(source),
        str(audit),
        str(reference),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    baseline = original_runner.run(original_config)
    inputs = load_inputs(source, audit, 7, verification=True)
    initial = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    expected = copy.deepcopy(inputs.model)
    expected.load_state_dict(initial["model"])
    torch.testing.assert_close(inputs.model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    config = Config(
        str(source),
        str(audit),
        str(tmp_path / "full"),
        reference_run_dir=str(reference),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    calls: list[tuple[str, bool, int]] = []
    updates = 0
    original_loss, original_assess = runner.training_loss, runner.assess

    def spy_loss(
        model: SCJEPA, batch: TensorSplit, theta: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal updates
        updates += 1
        assert set(batch) == {"online", "target", "actions"}
        return original_loss(model, batch, theta)

    def spy_assess(
        model: SCJEPA,
        data: EvaluationData,
        split: Split,
        *,
        groups: int,
        bootstrap_samples: int,
        seed: int,
        final: bool = False,
    ) -> dict[str, Any]:
        calls.append((split, final, updates))
        assert split != "test" or (final and updates == 3)
        return original_assess(
            model,
            data,
            split,
            groups=groups,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            final=final,
        )

    monkeypatch.setattr(runner, "training_loss", spy_loss)
    monkeypatch.setattr(runner, "assess", spy_assess)
    full = run(config)
    assert calls == [("validation", False, i) for i in range(4)] + [("test", True, 3)]
    assert full["records"][0] == baseline["records"][0]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    restored = copy.deepcopy(inputs.model)
    restored.load_state_dict(checkpoint["model"])
    assert fixed_hashes(restored) == fixed_hashes(inputs.model)
    assert any(
        not torch.equal(value, initial["model"]["predictor." + name])
        for name, value in restored.predictor.state_dict().items()
    )
    original_checkpoint = cast(
        dict[str, Any], torch.load(reference / "last.pt", weights_only=False)
    )
    assert any(
        not torch.equal(value, original_checkpoint["model"]["predictor." + name])
        for name, value in restored.predictor.state_dict().items()
    )
    interrupted_calls = 0

    def interrupted(
        model: SCJEPA, batch: TensorSplit, theta: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == 2:
            raise RuntimeError("interrupt")
        return original_loss(model, batch, theta)

    monkeypatch.setattr(runner, "assess", original_assess)
    monkeypatch.setattr(runner, "training_loss", interrupted)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    with pytest.raises(RuntimeError, match="interrupt"):
        run(resumed_config)
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final_checkpoint = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    )
    torch.testing.assert_close(checkpoint["model"], final_checkpoint["model"], rtol=0, atol=0)
    torch.testing.assert_close(
        checkpoint["optimizer"], final_checkpoint["optimizer"], rtol=0, atol=0
    )
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert run(resumed_config, resume=True) == resumed
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="settings differ"):
        run(replace(config, output_dir=str(tmp_path / "bad"), threads=2))
    with (reference / "last.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="Source artifact changed"):
        run(resumed_config, resume=True)


def test_scientific_control_requires_matched_reference_and_fixed_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="completed reference"):
        Config("source", "audit", str(tmp_path / "output"))
    with pytest.raises(ValueError, match="original seed"):
        Config("source", "audit", str(tmp_path / "output"), reference_run_dir="ref", seed=7)
    with pytest.raises(ValueError, match="separate"):
        Config("source", "audit", "ref/output", reference_run_dir="ref")
