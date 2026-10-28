"""Original-model pairing, frozen observation-only inputs and exact interrupted continuation."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.neural_observation_oracle import data, run
from diagnostics.neural_observation_oracle.map import FrozenObservationMap
from diagnostics.oracle_state_learning import core
from diagnostics.oracle_state_learning import data as original_data
from diagnostics.oracle_state_learning.data import read_json, save_torch
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _prepare(root: Path) -> None:
    original_data.prepare(
        original_data.PrepareConfig(
            str(root / "source"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    base = root / "base.pt"
    save_torch(base, {"matrix": VectorObservationMap(seed=1729, dimension=6).matrix})
    data.prepare(
        data.PrepareConfig(str(root / "data"), str(root / "source"), str(base), profile="smoke")
    )


def _config(root: Path, condition: core.Condition) -> run.TrainConfig:
    return run.TrainConfig(
        str(root / condition),
        str(root / "data"),
        condition,
        profile="smoke",
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
        calibration_dir=str(root / "bound") if condition in {"full", "baseline"} else None,
    )


def _checkpoint(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, weights_only=False, map_location="cpu"))


def _bound(root: Path) -> dict[str, Any]:
    run.train(_config(root, "dense"))
    return run.calibrate(
        run.CalibrateConfig(str(root / "bound"), (str(root / "dense"),), profile="smoke")
    )


def test_exact_original_factory_initialization_and_parameter_counts() -> None:
    assert run.build_control is core.build_control
    native = run.build_control("dense", 0)
    original = core.build_control("dense", 0)
    torch.testing.assert_close(native.state_dict(), original.state_dict(), rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in native.predictor.parameters()) == 52289
    assert len(native.predictor.layers) == 2
    assert native.predictor.action_dim == 16
    for condition in ("baseline", "full", "mlp"):
        model = run.build_control(condition, 0)
        torch.testing.assert_close(
            model.online_encoder.state_dict(), native.online_encoder.state_dict(), rtol=0, atol=0
        )
        if condition == "mlp":
            assert sum(parameter.numel() for parameter in model.predictor.parameters()) == 52123
        else:
            torch.testing.assert_close(
                model.predictor.state_dict(), native.predictor.state_dict(), rtol=0, atol=0
            )
        assert model.parameter_source == "oracle"
        assert all(not value.requires_grad for value in model.parameter_encoder.parameters())


def test_original_protocol_is_fixed_for_scientific_runs() -> None:
    config = run.TrainConfig("unused", "unused", "dense")
    for changes in (
        {"seed": 1},
        {"steps": 30000},
        {"batch_size": 8},
        {"lr": 0.002},
        {"evaluate_every": 1000},
        {"log_every": 1},
        {"checkpoint_every": 100},
    ):
        with pytest.raises(ValueError, match="fixed"):
            replace(config, **changes)
    with pytest.raises(ValueError, match="prespecified"):
        run.CalibrateConfig("unused", ("unused",), tau_factor=2)


def test_training_never_uses_inverse_or_test_and_keeps_two_step_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    cached, _ = data.load_data(tmp_path / "data")
    assert set(cached) == {"train", "calibration", "validation"}
    assert cached["train"].states is None
    batch = cached["train"].batch(torch.tensor([0, 1]), "cpu")
    assert set(batch) == core.TRAINING_KEYS
    assert batch["actions"].shape == (2, 10, 11, 16)
    model = run.build_control("dense", 0)
    objective = TrainingObjective(run.objective_config("dense"))
    loss, _, output = run.forward_loss(model, batch, objective, forward_seed=71)
    derivative = torch.autograd.grad(
        output.predictions[:, :, 1].square().mean(),
        output.dynamics[0].prediction,
        retain_graph=True,
    )[0]
    assert bool(torch.isfinite(derivative).all())
    assert float(derivative.abs().sum()) > 0
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.parameter_encoder.parameters())
    evaluated: list[str] = []
    actual = run.evaluate

    def observe(
        model: SCJEPA,
        sample: core.LiveStateData,
        batch_size: int,
        device: torch.device | str,
        objective: TrainingObjective,
    ) -> core.EvaluationResult:
        evaluated.append(sample.config.split)
        return actual(model, sample, batch_size, device, objective)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Training must never use the observation inverse")

    monkeypatch.setattr(run, "evaluate", observe)
    monkeypatch.setattr(FrozenObservationMap, "inverse", forbidden)
    completed = run.train(_config(tmp_path, "dense"))
    assert "test" not in evaluated
    assert set(evaluated) == {"evaluation_calibration", "validation"}
    assert len(completed["hashes"]) == 11
    assert completed["calibration_identity"] is None
    record = read_json(tmp_path / "dense/config.json")
    assert record["scientific_config"]["native_spartan_layers"] == 2
    assert record["scientific_config"]["predictor_parameter_count"] == 52289
    assert not record["physical_state_training"]
    assert not record["parameter_encoder_training"]
    final, initial = (
        _checkpoint(tmp_path / "dense/last.pt"),
        _checkpoint(tmp_path / "dense/initial.pt"),
    )
    for name, value in initial["model"].items():
        if name.startswith("parameter_encoder."):
            torch.testing.assert_close(final["model"][name], value, rtol=0, atol=0)
    assert not (tmp_path / "dense/test.pt").exists()


def test_sparse_bound_is_own_dense_final_calibration_and_starts_fresh(tmp_path: Path) -> None:
    _prepare(tmp_path)
    bound = _bound(tmp_path)
    assert bound["tau"] == 1.1 * read_json(tmp_path / "dense/calibration.json")["metrics"]["base"]
    dense = run.verify_completion(tmp_path / "dense")
    for condition in ("full", "baseline"):
        completed = run.train(_config(tmp_path, condition))
        for key in ("initial_encoder_identity", "initial_predictor_identity"):
            assert completed[key] == dense[key]
        assert completed["calibration_identity"] == bound["identity"]
        checkpoint = _checkpoint(tmp_path / condition / "last.pt")
        assert int(checkpoint["objective"]["geco.num_updates"]) == 4
    assert (
        run.train(_config(tmp_path, "full"), resume=True)["identity"]
        == read_json(tmp_path / "full/completion.json")["identity"]
    )


@pytest.mark.parametrize("condition", ["dense", "mlp", "full"])
def test_exact_interrupted_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: core.Condition
) -> None:
    _prepare(tmp_path)
    if condition == "full":
        _bound(tmp_path)
    config = _config(tmp_path, condition)
    complete = run.train(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    actual = run.forward_loss
    updates = 0

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal updates
        updates += 1
        if updates == 3:
            raise RuntimeError("deliberate interruption")
        return actual(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(run, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            run.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = run.train(interrupted, resume=True)
    assert complete["identity"] == resumed["identity"]
    left, right = (
        _checkpoint(Path(config.output_dir) / "last.pt"),
        _checkpoint(tmp_path / "interrupted/last.pt"),
    )
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng", "cuda_rng"):
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for split in ("calibration", "validation"):
        torch.testing.assert_close(
            _checkpoint(Path(config.output_dir) / f"{split}.pt"),
            _checkpoint(tmp_path / f"interrupted/{split}.pt"),
            rtol=0,
            atol=0,
        )


def test_source_resume_guard_and_numeric_identity_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    config = _config(tmp_path, "mlp")
    run.train(config)
    before = original_data.code_identity()
    isolated = data.code_identity()
    original_read = Path.read_bytes
    changed_path = Path(run.__file__).resolve()

    def changed(path: Path) -> bytes:
        value = original_read(path)
        return value + b"\n# isolated test mutation\n" if path.resolve() == changed_path else value

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", changed)
        assert original_data.code_identity() == before
        assert data.code_identity() != isolated
    monkeypatch.setattr(run, "code_identity", lambda: "changed-source")
    with pytest.raises(ValueError, match="configuration, source"):
        run.train(config, resume=True)
