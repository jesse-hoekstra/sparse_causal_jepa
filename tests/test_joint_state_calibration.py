import copy
import shutil
from collections.abc import Callable, Generator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import torch
from test_joint_state_survival import joint_sources as joint_sources
from torch import Tensor

from diagnostics.global_code_calibration import optimization as original_optimization
from diagnostics.joint_state_calibration import run as runner
from diagnostics.joint_state_calibration.source import (
    encode_observations,
    load_training,
)
from diagnostics.joint_state_survival.run import Config as SurvivalConfig
from diagnostics.joint_state_survival.run import run as run_survival
from diagnostics.multiquery_control.data import TensorSplit, encode_states
from diagnostics.multiquery_control.run import sha
from diagnostics.predictor_transfer_calibration.objective import (
    TrainObjective,
    compose_translation,
    encode_train,
    prediction_loss,
)
from diagnostics.predictor_transfer_calibration.optimization import (
    EmbeddedObjective,
    verify_embedding,
)
from diagnostics.table_settling_control.source import model_fingerprint
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA

Arm = Literal["frozen_state", "live_state"]


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def calibration_sources(
    joint_sources: tuple[Path, Path, Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[Arm, Path], Path, Path]:
    """Train both genuine endpoint policies from the same authenticated observation stream."""
    source, transfer, calibration, precision = joint_sources
    root = tmp_path_factory.mktemp("final-state-calibration-sources")
    endpoints: dict[Arm, Path] = {}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        arms: tuple[Arm, ...] = ("frozen_state", "live_state")
        for arm in arms:
            endpoint = root / arm
            run_survival(
                SurvivalConfig(
                    str(source),
                    str(transfer),
                    str(calibration),
                    str(precision),
                    str(endpoint),
                    sha(transfer / "last.pt"),
                    sha(calibration / "composed_model.pt"),
                    arm,
                    source_gate_mode="qualified_spartan",
                    seed=7,
                    steps=2,
                    groups=2,
                    bootstrap_samples=8,
                    chunk_size=3,
                    verification=True,
                )
            )
            endpoints[arm] = endpoint
        return endpoints, source, precision
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("arm", ["frozen_state", "live_state"])
def test_final_charts_reencode_train_without_old_inputs_and_preserve_exact_nine_donor_objective(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arm: Arm,
) -> None:
    endpoints, source, precision = calibration_sources
    endpoint = endpoints[arm]
    old_train = cast(TensorSplit, torch.load(source / "train_inputs.pt", weights_only=False))
    saved = cast(dict[str, Any], torch.load(endpoint / "last.pt", weights_only=False))
    original = cast(Callable[..., Any], torch.load)

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            assert Path(file).name not in {
                "train_inputs.pt",
                "evaluation_data.pt",
                "validation_data.pt",
                "fresh_test.pt",
                "source_assessment.pt",
                "endpoint_assessment.pt",
            }
        return original(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    inputs = load_training(
        endpoint,
        source,
        precision,
        arm=arm,
        checkpoint_sha256=sha(endpoint / "last.pt"),
        verification=True,
    )
    assert set(inputs.observations) == {"observations", "actions"}
    model = inputs.model
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    expected = {
        key: value.double() if isinstance(value, Tensor) else value
        for key, value in saved["model"].items()
    }
    torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)
    fingerprint = model_fingerprint(model)
    train = encode_observations(model, inputs.observations, chunk_size=3)
    online, target = encode_states(
        copy.deepcopy(model).float(), inputs.observations["observations"], 32
    )
    torch.testing.assert_close(train["online"], online, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(train["target"], target, rtol=1e-6, atol=1e-7)
    assert train["actions"] is inputs.observations["actions"]
    for chart in ("online", "target"):
        difference = float((train[chart] - old_train[chart]).abs().max())
        if arm == "live_state":
            assert difference > 1e-4, "Live calibration must not use initial state encodings"
        else:
            torch.testing.assert_close(train[chart], old_train[chart], rtol=1e-6, atol=1e-7)
    codes = encode_train(model, train, 3)
    delta = torch.linspace(-0.01, 0.02, 6, dtype=torch.float64, requires_grad=True)
    batch = {key: value.double() for key, value in train.items()}
    actual = prediction_loss(model, batch, codes, delta)
    gradient = torch.autograd.grad(actual, delta)[0]
    terms: list[Tensor] = []
    for recipient in range(10):
        for donor in range(10):
            if donor == recipient:
                continue
            history = batch["online"][:, recipient, 9:10]
            predictions: list[Tensor] = []
            for step in (9, 10):
                predicted = model.predictor(
                    history,
                    codes[:, donor] + delta,
                    batch["actions"][:, recipient, step : step + 1],
                ).prediction
                predictions.append(predicted)
                history = predicted[:, None]
            terms.append(
                (torch.stack(predictions, 1) - batch["target"][:, recipient, 10:12]).square().mean()
            )
    expected_loss = torch.stack(terms).mean()
    expected_gradient = torch.autograd.grad(expected_loss, delta)[0]
    torch.testing.assert_close(actual, expected_loss, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-10, atol=1e-12)
    value, chunked_gradient = TrainObjective(model, train, codes, 3)(delta)
    torch.testing.assert_close(
        torch.tensor(value, dtype=torch.float64), expected_loss, rtol=1e-12, atol=1e-13
    )
    torch.testing.assert_close(chunked_gradient, expected_gradient, rtol=1e-10, atol=1e-12)
    assert model_fingerprint(model) == fingerprint
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="pin|checkpoint"):
        load_training(
            endpoint, source, precision, arm=arm, checkpoint_sha256="f" * 64, verification=True
        )
    other_arm: Arm = "live_state" if arm == "frozen_state" else "frozen_state"
    with pytest.raises(ValueError, match="arm"):
        load_training(
            endpoint,
            source,
            precision,
            arm=other_arm,
            checkpoint_sha256=sha(endpoint / "last.pt"),
            verification=True,
        )
    copied = tmp_path / "invalid-receipt"
    shutil.copytree(endpoint, copied)
    receipt = cast(dict[str, Any], read_json(copied / "train_receipt.json"))
    receipt["all_initial_encodings_checked"] = False
    write_json(copied / "train_receipt.json", receipt)
    completion = cast(dict[str, Any], read_json(copied / "completed.json"))
    completion["hashes"]["train_receipt.json"] = sha(copied / "train_receipt.json")
    write_json(copied / "completed.json", completion)
    with pytest.raises(ValueError, match="authentication"):
        load_training(
            copied,
            source,
            precision,
            arm=arm,
            checkpoint_sha256=sha(copied / "last.pt"),
            verification=True,
        )
    with pytest.raises(ValueError, match="observations and actions"):
        encode_observations(model, {**inputs.observations, "online": train["online"]}, chunk_size=3)


@pytest.mark.parametrize("arm", ["frozen_state", "live_state"])
def test_runner_fits_only_current_train_before_heldout_replay_and_resumes_exactly(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: Arm,
) -> None:
    endpoints, source, precision = calibration_sources
    endpoint = endpoints[arm]
    finished = False
    original_load = cast(Callable[..., Any], torch.load)

    def guarded(file: object, *args: object, **kwargs: object) -> Any:  # noqa: ANN401
        if isinstance(file, str | Path):
            name = Path(file).name
            assert name not in {"fresh_test.pt", "audit_inputs.pt", "train_inputs.pt"}
            if name in {"evaluation_data.pt", "validation_data.pt"}:
                assert finished, "Heldout physical inputs opened before fixed TRAIN fit ended"
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
    expected_train = encode_observations(inputs.model, inputs.observations, chunk_size=3)
    fixed = model_fingerprint(inputs.model)
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
    assert runner.fit is original_optimization.fit
    assert runner.TrainObjective is TrainObjective
    fit = runner.fit
    original_encode = runner.encode_observations
    encoded_calls = 0
    interrupt = False

    def checked_encode(
        model: SCJEPA,
        observations: TensorSplit,
        *,
        chunk_size: int,
        progress: Callable[[int], None] | None = None,
    ) -> TensorSplit:
        nonlocal encoded_calls
        encoded_calls += 1
        assert not finished
        return original_encode(model, observations, chunk_size=chunk_size, progress=progress)

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
        assert not finished
        assert isinstance(evaluator, EmbeddedObjective)
        assert isinstance(evaluator.evaluator, TrainObjective)
        objective = evaluator.evaluator
        assert model_fingerprint(objective.model) == fixed
        torch.testing.assert_close(objective.train, expected_train, rtol=0, atol=0)
        assert set(objective.train) == {"online", "target", "actions"}
        assert max_iterations == 2
        assert max_evaluations == 6

        def save(current: dict[str, Any]) -> None:
            verify_embedding(current)
            if on_state:
                on_state(current)
            if interrupt and current["full_evaluations"] == 2:
                raise RuntimeError("current-chart calibration interruption")

        result = fit(
            evaluator,
            device=device,
            max_iterations=max_iterations,
            max_evaluations=max_evaluations,
            state=state,
            on_state=save,
        )
        finished = True
        return result

    monkeypatch.setattr(runner, "encode_observations", checked_encode)
    monkeypatch.setattr(runner, "fit", checked_fit)
    full = runner.run(config)
    assert encoded_calls == 1
    assert set(full["reports"]) == {
        "original64/source",
        "original64/calibrated",
        "precision8192/source",
        "precision8192/calibrated",
    }
    output = tmp_path / "full"
    public = cast(dict[str, Any], original_load(output / "translation.pt", weights_only=False))
    assert public["translation"].shape == (6,)
    state = cast(dict[str, Any], original_load(output / "fit_state.pt", weights_only=False))[
        "state"
    ]
    verify_embedding(state)
    torch.testing.assert_close(public["translation"], state["translation"][:6], rtol=0, atol=0)
    composed = cast(dict[str, Any], original_load(output / "composed_model.pt", weights_only=False))
    expected = compose_translation(inputs.model, public["translation"])
    torch.testing.assert_close(composed["model"], expected.state_dict(), rtol=0, atol=0)
    for key, value in inputs.model.state_dict().items():
        if key != "parameter_encoder.head.4.bias":
            torch.testing.assert_close(composed["model"][key], value, rtol=0, atol=0)
    completed = cast(dict[str, Any], read_json(output / "completed.json"))
    assert len(completed["hashes"]) == 12
    assert all(sha(output / name) == value for name, value in completed["hashes"].items())
    finished = False
    interrupt = True
    resumed_output = tmp_path / "resumed"
    resumed_config = replace(config, output_dir=str(resumed_output))
    with pytest.raises(RuntimeError, match="calibration interruption"):
        runner.run(resumed_config)
    assert not finished
    assert encoded_calls == 2
    assert not (resumed_output / "composed_model.pt").exists()
    receipt_path = resumed_output / "encoding_receipt.json"
    receipt_text = receipt_path.read_text()
    receipt = cast(dict[str, Any], read_json(receipt_path))
    receipt["source_model_sha256"] = "f" * 64
    write_json(receipt_path, receipt)
    interrupt = False
    with pytest.raises(ValueError, match="encodings changed"):
        runner.run(resumed_config, resume=True)
    receipt_path.write_text(receipt_text)
    encoding_hash = sha(resumed_output / "train_encodings.pt")
    resumed = runner.run(resumed_config, resume=True)
    assert finished
    assert encoded_calls == 2
    assert sha(resumed_output / "train_encodings.pt") == encoding_hash
    assert full["reports"] == resumed["reports"]
    replay = cast(
        dict[str, Any], original_load(resumed_output / "fit_state.pt", weights_only=False)
    )["state"]
    for name in ("translation", "accepted_gradient", "initial_gradient"):
        torch.testing.assert_close(replay[name], state[name], rtol=0, atol=0)
    torch.testing.assert_close(
        replay["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0
    )
    assert replay["optimizer"]["param_groups"] == state["optimizer"]["param_groups"]
    assert replay["full_evaluations"] == state["full_evaluations"]
    finished = False
    runner.run(resumed_config, resume=True)
    assert not finished
    assert encoded_calls == 2
    assert model_fingerprint(inputs.model) == fixed
    assert all(sha(Path(name)) == value for name, value in inputs.hashes.items())
    with pytest.raises(ValueError, match="disjoint"):
        replace(config, output_dir=str(endpoint / "bad"))
    with pytest.raises(ValueError, match="fixes25"):
        replace(config, verification=False)


def test_postfit_interruption_reuses_terminal_state_without_optimizer_reentry(
    calibration_sources: tuple[dict[Arm, Path], Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints, source, precision = calibration_sources
    endpoint = endpoints["live_state"]
    config = runner.Config(
        str(endpoint),
        str(source),
        str(precision),
        str(tmp_path / "postfit"),
        "live_state",
        sha(endpoint / "last.pt"),
        chunk_size=3,
        max_iterations=2,
        max_evaluations=6,
        verification=True,
    )
    original_evaluation = runner.load_evaluation

    def stop_before_heldout(*_args: object, **_kwargs: object) -> None:
        assert (tmp_path / "postfit/composed_model.pt").exists()
        raise RuntimeError("interrupted after completed TRAIN fit")

    monkeypatch.setattr(runner, "load_evaluation", stop_before_heldout)
    with pytest.raises(RuntimeError, match="after completed TRAIN fit"):
        runner.run(config)
    path = tmp_path / "postfit/fit_state.pt"
    fit_hash = sha(path)
    before = cast(dict[str, Any], torch.load(path, weights_only=False))["state"]
    assert before["stop_reason"] != "running"

    def no_optimizer_reentry(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Terminal optimizer budget must not reopen after fitting ended")

    monkeypatch.setattr(runner, "fit", no_optimizer_reentry)
    monkeypatch.setattr(runner, "load_evaluation", original_evaluation)
    result = runner.run(config, resume=True)
    assert len(result["reports"]) == 4
    assert sha(path) == fit_hash
    after = cast(dict[str, Any], torch.load(path, weights_only=False))["state"]
    assert after["full_evaluations"] == before["full_evaluations"]
    assert after["closure_calls"] == before["closure_calls"]
    assert after["accepted_iterations"] == before["accepted_iterations"]
