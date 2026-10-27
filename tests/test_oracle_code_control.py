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

from diagnostics.joint_acquisition.baseline import DiagnosticSelection, build_matched_batch
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.oracle_code_control import run as runner
from diagnostics.oracle_code_control.evaluation import (
    PARAMETER_STD,
    Split,
    assess,
    forecast_report,
    matched_batch,
)
from diagnostics.oracle_code_control.run import Config, fixed_hashes, metrics, run, training_loss
from diagnostics.oracle_code_control.source import load_inputs
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


def test_training_uses_exact_codes_and_both_attached_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model().requires_grad_(False)
    model.predictor.requires_grad_(True)
    _, batch = make_batch(model)
    theta = torch.randn(2, 6, dtype=torch.float64, requires_grad=True) * 0.1

    def forbidden_encoder(*_args: object, **_kwargs: object) -> Tensor:
        raise AssertionError("Oracle diagnostic invoked parameter encoder")

    monkeypatch.setattr(model.parameter_encoder, "forward", forbidden_encoder)
    actual, _ = training_loss(model, batch, theta)
    codes = (theta / PARAMETER_STD)[:, None].expand(-1, 10, -1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1), codes, batch["actions"][:, :, 9:10].flatten(0, 1)
    )
    second = model.predictor(
        first.prediction[:, None], codes, batch["actions"][:, :, 10:11].flatten(0, 1)
    )
    expected_predictions = torch.stack((first.prediction, second.prediction), 1).reshape(
        2, 10, 2, 6
    )
    expected = (expected_predictions - batch["target"][:, :, 10:12]).square().mean()
    expected = expected + 0.0001 * torch.stack((first.logit_penalty, second.logit_penalty)).mean()
    actual_gradient = torch.autograd.grad(actual, theta, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, theta, retain_graph=True)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-14, atol=1e-18)
    assert float(actual_gradient.square().sum()) > 0
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())
    assert all(parameter.grad is None for parameter in model.parameter_encoder.parameters())
    with pytest.raises(ValueError, match="separate"):
        training_loss(model, {**batch, "theta": theta}, theta)


def test_coordinate_centered_nmse_does_not_pool_coordinate_offsets() -> None:
    model = make_model().requires_grad_(False).eval()
    _, batch = make_batch(model)
    target = batch["target"].clone()
    target += torch.arange(6, dtype=torch.float64)[None, None, None] * 100
    data = {
        "states_online": batch["online"],
        "states_target": target,
        "actions": batch["actions"],
        "theta": torch.zeros(2, 6, dtype=torch.float64),
    }
    report = forecast_report(model, data)
    expected = target[:, :, 10:12].permute(2, 0, 1, 3).flatten(1, 2).var(1, correction=0).mean(-1)
    torch.testing.assert_close(
        torch.tensor(report["target_variance_by_horizon"], dtype=torch.float64), expected
    )
    assert max(report["target_variance_by_horizon"]) < 10
    assert "ablation" in report["zero_code_scope"]


def test_full_run_resume_authentication_and_explicit_test_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum, audit = fixture_sources(tmp_path, fresh=True)
    source = Path(
        cast(dict[str, Any], read_json(curriculum / "protocol.json"))["config"]["source_dir"]
    )
    inputs = load_inputs(source, audit, 7, verification=True)
    initial = cast(dict[str, Any], torch.load(source / "initial_seed_7.pt", weights_only=False))
    expected = copy.deepcopy(inputs.model)
    expected.load_state_dict(initial["model"])
    torch.testing.assert_close(inputs.model.state_dict(), expected.state_dict(), rtol=0, atol=0)
    frozen = fixed_hashes(inputs.model)
    config = Config(
        str(source),
        str(audit),
        str(tmp_path / "full"),
        seed=7,
        steps=3,
        groups=2,
        bootstrap_samples=16,
        verification=True,
    )
    calls: list[tuple[str, bool]] = []
    original_assess = runner.assess

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
        calls.append((split, final))
        assert split != "test" or final
        return original_assess(
            model,
            data,
            split,
            groups=groups,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            final=final,
        )

    monkeypatch.setattr(runner, "assess", spy_assess)
    full = run(config)
    assert calls == [("validation", False)] * 4 + [("test", True)]
    checkpoint = cast(dict[str, Any], torch.load(tmp_path / "full/last.pt", weights_only=False))
    restored = copy.deepcopy(inputs.model)
    restored.load_state_dict(checkpoint["model"])
    assert fixed_hashes(restored) == frozen
    assert any(
        not torch.equal(value, initial["model"]["predictor." + name])
        for name, value in restored.predictor.state_dict().items()
    )
    original_loss = runner.training_loss
    update_calls = 0

    def interrupted(
        model: SCJEPA, batch: TensorSplit, theta: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        nonlocal update_calls
        update_calls += 1
        if update_calls == 2:
            raise RuntimeError("interrupt")
        assert set(batch) == {"online", "target", "actions"}
        return original_loss(model, batch, theta)

    monkeypatch.setattr(runner, "training_loss", interrupted)
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    with pytest.raises(RuntimeError, match="interrupt"):
        run(resumed_config)
    monkeypatch.setattr(runner, "training_loss", original_loss)
    resumed = run(resumed_config, resume=True)
    final = cast(dict[str, Any], torch.load(tmp_path / "resumed/last.pt", weights_only=False))
    torch.testing.assert_close(checkpoint["model"], final["model"], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint["optimizer"], final["optimizer"], rtol=0, atol=0)
    assert full["records"] == resumed["records"]
    assert full["final_test"] == resumed["final_test"]
    assert all(sha(Path(name)) == expected_hash for name, expected_hash in inputs.hashes.items())
    assert run(resumed_config, resume=True) == resumed
    # The public evaluator excludes test until explicitly final; validation needs no test payload.
    candidate = copy.deepcopy(inputs.model).double().eval()
    guarded = {name: value for name, value in inputs.bundle.evaluation.items() if name != "test"}
    validation = assess(candidate, guarded, "validation", groups=2, bootstrap_samples=16, seed=7)
    assert "validation/response/theta_6/normalized_response_error/estimate" in metrics(
        validation, "validation"
    )
    with pytest.raises(ValueError, match="fixed final"):
        assess(candidate, guarded, "test", groups=2, bootstrap_samples=16, seed=7)
    expected_batch = build_matched_batch(
        candidate, inputs.bundle.evaluation, DiagnosticSelection((0, 1), (0, 1)), "validation"
    ).select_queries("heldout")
    actual_batch, codes, changed_codes = matched_batch(
        candidate, inputs.bundle.evaluation, "validation", (0, 1)
    )
    for name in ("history", "query_actions", "targets", "changed_targets", "finite_steps"):
        torch.testing.assert_close(
            getattr(actual_batch, name), getattr(expected_batch, name), rtol=1e-12, atol=1e-12
        )
    theta = cast(TensorSplit, inputs.bundle.evaluation["validation"])["theta"][:2]
    torch.testing.assert_close(codes, theta / PARAMETER_STD)
    torch.testing.assert_close(
        changed_codes,
        (theta[:, None] + torch.diag_embed(actual_batch.finite_steps)) / PARAMETER_STD,
    )
    payload = cast(dict[str, Any], torch.load(audit / "audit_inputs.pt", weights_only=False))
    payload["theta_evaluation_only"] += 0.001
    torch.save(payload, audit / "audit_inputs.pt")
    with pytest.raises(ValueError, match="Source artifact changed"):
        load_inputs(source, audit, 7, verification=True)
