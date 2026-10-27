import copy
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from test_multiquery_control import make_model
from test_predictor_transfer_control import transfer_sources as transfer_sources
from test_predictor_transfer_precision import precision_sources as precision_sources
from torch import Tensor, nn

from diagnostics.global_code_calibration import optimization as original_optimization
from diagnostics.mlp_oracle_code_control.model import Prediction, initialize_mlp
from diagnostics.multiquery_control.data import TensorSplit
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_calibration import run as runner
from diagnostics.predictor_transfer_calibration.objective import (
    TrainObjective,
    compose_translation,
    encode_train,
    prediction_loss,
)
from diagnostics.predictor_transfer_calibration.optimization import (
    EmbeddedObjective,
    active_state,
    fit,
    verify_embedding,
)
from diagnostics.predictor_transfer_calibration.source import load_training
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.experiments.storage import read_json
from scjepa.models.spartan import SpartanOutput

Arm = Literal["spartan", "mlp"]


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("arm", ["spartan", "mlp"])
def test_six_offset_factual_objective_matches_every_nonself_donor_and_attached_gradient(
    arm: Arm,
) -> None:
    model = make_model()
    if arm == "mlp":
        initialize_mlp(model, 17)
    model.requires_grad_(False).double().eval()
    generator = torch.Generator().manual_seed(1531)
    train: TensorSplit = {
        "online": torch.randn(3, 10, 12, 6, dtype=torch.float64, generator=generator),
        "target": torch.randn(3, 10, 12, 6, dtype=torch.float64, generator=generator),
        "actions": torch.randn(3, 10, 11, 16, dtype=torch.float64, generator=generator),
    }
    initial = model_fingerprint(model)
    codes = encode_train(model, train, 2)
    assert codes.shape == (3, 10, 6)
    assert not codes.requires_grad
    assert not torch.equal(codes[:, 0], codes[:, 1])
    delta = torch.linspace(-0.02, 0.03, 6, dtype=torch.float64, requires_grad=True)
    donors: list[Tensor] = []
    outputs: list[Tensor] = []
    histories: list[Tensor] = []

    def capture(
        _module: nn.Module, args: tuple[object, ...], output: Prediction | SpartanOutput
    ) -> None:
        assert isinstance(args[0], Tensor)
        assert isinstance(args[1], Tensor)
        histories.append(args[0])
        donors.append(args[1])
        outputs.append(output.prediction)

    hook = model.predictor.register_forward_hook(capture)
    try:
        actual = prediction_loss(model, train, codes, delta)
    finally:
        hook.remove()
    assert len(outputs) == 2
    assert donors[0] is donors[1]
    assert donors[0].requires_grad
    torch.testing.assert_close(histories[1], outputs[0][:, None], rtol=0, atol=0)
    path = torch.autograd.grad(outputs[1].square().mean(), outputs[0], retain_graph=True)[0]
    assert float(path.square().sum()) > 0
    actual_gradient = torch.autograd.grad(actual, delta)[0]
    terms: list[Tensor] = []
    for recipient in range(10):
        for donor in range(10):
            if donor == recipient:
                continue
            history = train["online"][:, recipient, 9:10]
            code = codes[:, donor] + delta
            predictions: list[Tensor] = []
            for step in (9, 10):
                predicted = model.predictor(
                    history, code, train["actions"][:, recipient, step : step + 1]
                ).prediction
                predictions.append(predicted)
                history = predicted[:, None]
            terms.append(
                (torch.stack(predictions, 1) - train["target"][:, recipient, 10:12]).square().mean()
            )
    expected = torch.stack(terms).mean()
    expected_gradient = torch.autograd.grad(expected, delta)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-13)
    objective = TrainObjective(model, train, codes, 2)
    value, gradient = objective(delta)
    torch.testing.assert_close(
        torch.tensor(value, dtype=torch.float64), expected, rtol=1e-12, atol=1e-13
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-10, atol=1e-13)
    candidate = compose_translation(model, delta.detach())
    new_codes = encode_train(candidate, train, 2)
    torch.testing.assert_close(new_codes, codes + delta.detach(), rtol=1e-10, atol=1e-12)
    for name, value in model.state_dict().items():
        if name != "parameter_encoder.head.4.bias":
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
    assert model_fingerprint(model) == initial
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="label-free"):
        TrainObjective(model, {**train, "theta": torch.zeros(3, 6)}, codes, 2)


def test_six_active_offsets_reuse_unchanged_bounded_optimizer_and_keep_inactive_history_zero() -> (
    None
):
    assert fit is original_optimization.fit
    weights = torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0, 11.0], dtype=torch.float64)
    target = torch.linspace(-0.3, 0.2, 6, dtype=torch.float64)

    def quadratic(x: Tensor) -> tuple[float, Tensor]:
        difference = x - target
        return float(
            (0.5 * weights * difference.square()).sum().detach()
        ), weights * difference.detach()

    wrapped = EmbeddedObjective(quadratic)
    state = fit(wrapped, device=torch.device("cpu"), max_iterations=3, max_evaluations=12)
    verify_embedding(state)
    six = active_state(state)
    assert six["translation"].shape == (6,)
    assert state["translation"].shape == (12,)
    assert state["accepted_iterations"] == 3
    assert state["full_evaluations"] <= 12
    native = nn.Parameter(torch.zeros(6, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [native],
        lr=1,
        max_iter=3,
        max_eval=50,
        history_size=10,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = (0.5 * weights * (native - target).square()).sum()
        native.grad = torch.autograd.grad(loss, native)[0]
        return loss

    optimizer.step(closure)  # pyright: ignore[reportUnknownMemberType]
    torch.testing.assert_close(six["translation"], native.detach(), rtol=1e-12, atol=1e-12)
    saved: dict[str, Any] | None = None

    def interrupt(current: dict[str, Any]) -> None:
        nonlocal saved
        saved = copy.deepcopy(current)
        verify_embedding(current)
        if current["accepted_iterations"] == 1:
            raise RuntimeError("embedding operational interruption")

    with pytest.raises(RuntimeError, match="operational interruption"):
        fit(
            wrapped,
            device=torch.device("cpu"),
            max_iterations=3,
            max_evaluations=12,
            on_state=interrupt,
        )
    assert saved is not None
    resumed = fit(
        wrapped, device=torch.device("cpu"), max_iterations=3, max_evaluations=12, state=saved
    )
    torch.testing.assert_close(resumed["translation"], state["translation"], rtol=0, atol=0)
    torch.testing.assert_close(
        resumed["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0
    )
    assert resumed["optimizer"]["param_groups"] == state["optimizer"]["param_groups"]
    assert resumed["accepted_iterations"] == state["accepted_iterations"]
    assert resumed["full_evaluations"] == state["full_evaluations"]
    capped = fit(wrapped, device=torch.device("cpu"), max_iterations=25, max_evaluations=1)
    assert capped["stop_reason"] == "evaluation_budget_rollback"
    assert capped["accepted_iterations"] == 0
    assert capped["full_evaluations"] == 1
    assert torch.equal(capped["translation"], torch.zeros(12, dtype=torch.float64))
    malformed = torch.zeros(12, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        malformed[-1] = 1
    with pytest.raises(ValueError, match="inactive zeros"):
        wrapped(malformed)
    corrupted = copy.deepcopy(state)
    cast(Tensor, corrupted["translation"])[-1] = 0.1
    with pytest.raises(ValueError, match="Inactive"):
        verify_embedding(corrupted)
    corrupted = copy.deepcopy(state)
    cast(Tensor, next(iter(corrupted["cache"].values()))["gradient"])[-1] = 0.1
    with pytest.raises(ValueError, match="inactive"):
        verify_embedding(corrupted)


@pytest.mark.parametrize("arm", ["spartan", "mlp"])
def test_six_offset_runner_keeps_heldout_data_after_fit_and_resumes_unchanged_optimizer(
    precision_sources: tuple[Path, Path, Path, Path],  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
) -> None:
    spartan, mlp, source, precision = precision_sources
    endpoint = spartan if arm == "spartan" else mlp
    finished = False
    original_load = cast(Callable[..., Any], torch.load)

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            assert name not in {"audit_inputs.pt", "fresh_test.pt"}
            if name in {"evaluation_data.pt", "validation_data.pt"}:
                assert finished, "Heldout arrays read before fixed TRAIN fit ended"
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    inputs = load_training(
        endpoint,
        source,
        precision,
        arm=arm,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        verification=True,
    )
    assert set(inputs.train) == {"online", "target", "actions"}
    assert all(not p.requires_grad and p.grad is None for p in inputs.model.parameters())
    assert not inputs.model.training
    assert runner.fit is original_optimization.fit
    baseline = model_fingerprint(inputs.model)
    config = runner.Config(
        str(endpoint),
        str(source),
        str(precision),
        str(tmp_path / "full"),
        arm,
        sha(endpoint / "last.pt"),
        chunk_size=3,
        max_iterations=2,
        max_evaluations=6,
        verification=True,
    )
    original_fit = runner.fit
    interrupt = False

    def checked_fit(
        evaluator: original_optimization.Evaluator,
        *,
        device: torch.device,
        max_iterations: int = 25,
        max_evaluations: int = 50,
        state: dict[str, Any] | None = None,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        nonlocal finished
        assert isinstance(evaluator, EmbeddedObjective)
        assert isinstance(evaluator.evaluator, TrainObjective)
        assert not finished
        assert max_iterations == 2
        assert max_evaluations == 6
        assert model_fingerprint(evaluator.evaluator.model) == baseline

        def checked_save(current: dict[str, Any]) -> None:
            verify_embedding(current)
            assert current["translation"].shape == (12,)
            assert current["full_evaluations"] <= 6
            if on_state:
                on_state(current)
            if interrupt and current["full_evaluations"] == 2:
                raise RuntimeError("six-offset calibration operational interruption")

        result = original_fit(
            evaluator,
            device=device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=checked_save,
        )
        finished = True
        return result

    monkeypatch.setattr(runner, "fit", checked_fit)
    full = runner.run(config)
    assert finished
    assert set(full["reports"]) == {
        "original64/source",
        "original64/calibrated",
        "precision8192/source",
        "precision8192/calibrated",
    }
    output = tmp_path / "full"
    public = cast(dict[str, Any], torch.load(output / "translation.pt", weights_only=False))
    assert public["translation"].shape == (6,)
    state = cast(dict[str, Any], torch.load(output / "fit_state.pt", weights_only=False))["state"]
    verify_embedding(state)
    torch.testing.assert_close(public["translation"], state["translation"][:6], rtol=0, atol=0)
    composed = cast(dict[str, Any], torch.load(output / "composed_model.pt", weights_only=False))
    expected = compose_translation(inputs.model, public["translation"])
    torch.testing.assert_close(composed["model"], expected.state_dict(), rtol=0, atol=0)
    assert model_fingerprint(inputs.model) == baseline
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"))
    finished = False
    interrupt = True
    with pytest.raises(RuntimeError, match="operational interruption"):
        runner.run(resumed_config)
    assert not finished
    assert not (tmp_path / "resumed/composed_model.pt").exists()
    partial = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/fit_state.pt", weights_only=False)
    )["state"]
    assert partial["full_evaluations"] == 2
    previous = cast(dict[str, Any], read_json(tmp_path / "resumed/tracking.json"))[
        "last_event_step"
    ]
    interrupt = False
    resumed = runner.run(resumed_config, resume=True)
    assert finished
    assert full["reports"] == resumed["reports"]
    replay = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/fit_state.pt", weights_only=False)
    )["state"]
    for name in ("translation", "accepted_gradient", "initial_gradient"):
        torch.testing.assert_close(replay[name], state[name], rtol=0, atol=0)
    torch.testing.assert_close(
        replay["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0
    )
    assert replay["optimizer"]["param_groups"] == state["optimizer"]["param_groups"]
    assert replay["full_evaluations"] == state["full_evaluations"]
    assert (
        cast(dict[str, Any], read_json(tmp_path / "resumed/tracking.json"))["last_event_step"]
        > previous
    )
    finished = False
    runner.run(resumed_config, resume=True)
    assert not finished
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(endpoint / "bad-output"))
    with pytest.raises(ValueError, match="fixes25"):
        replace(config, verification=False)
    changed = cast(
        dict[str, Any], torch.load(tmp_path / "resumed/translation.pt", weights_only=False)
    )
    changed["translation"][0] += 0.1
    torch.save(changed, tmp_path / "resumed/translation.pt")
    with pytest.raises(ValueError, match="changed"):
        runner.run(resumed_config, resume=True)
