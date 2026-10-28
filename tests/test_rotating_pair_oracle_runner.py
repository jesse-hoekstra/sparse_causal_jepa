"""Fresh training, source isolation and exact resume for the rotating-pair control."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.balanced_action_oracle.data import code_identity as parent_code_identity
from diagnostics.balanced_action_oracle.data import prepare as prepare_source
from diagnostics.isotropic_oracle import data as ancestor_data
from diagnostics.isotropic_oracle.model import build_control
from diagnostics.oracle_state_learning.core import EvaluationResult, LiveStateData
from diagnostics.oracle_state_learning.data import read_json
from diagnostics.rotating_pair_oracle import run as runner
from diagnostics.rotating_pair_oracle.data import (
    PrepareConfig,
    StateData,
    code_identity,
    load_data,
    prepare,
)
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
    ancestor_data.prepare(
        ancestor_data.PrepareConfig(
            str(root / "ancestor"),
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare_source(
        SourcePrepareConfig(str(root / "source"), str(root / "ancestor"), profile="smoke")
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source"), profile="smoke"))


def _config(root: Path, condition: runner.Condition) -> runner.TrainConfig:
    return runner.TrainConfig(
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
    runner.train(_config(root, "dense"))
    return runner.calibrate(
        runner.CalibrateConfig(str(root / "bound"), (str(root / "dense"),), profile="smoke")
    )


def test_model_pairing_and_mlp_parameter_match() -> None:
    assert runner.build_control is build_control
    models = {
        condition: build_control(condition, 0) for condition in ("dense", "baseline", "full", "mlp")
    }
    for name in ("baseline", "full", "mlp"):
        torch.testing.assert_close(
            models[name].online_encoder.state_dict(),
            models["dense"].online_encoder.state_dict(),
            rtol=0,
            atol=0,
        )
    for name in ("baseline", "full"):
        torch.testing.assert_close(
            models[name].predictor.state_dict(),
            models["dense"].predictor.state_dict(),
            rtol=0,
            atol=0,
        )
    counts = {
        name: sum(p.numel() for p in model.predictor.parameters()) for name, model in models.items()
    }
    assert abs(counts["mlp"] / counts["dense"] - 1) < 0.005
    assert all(model.predictor.action_dim == 24 for model in models.values())


def test_cache_labels_addresses_and_validation_only_load(tmp_path: Path) -> None:
    _prepare(tmp_path)
    data, completion = load_data(tmp_path / "data")
    assert set(data) == {"train", "calibration", "validation"}
    assert data["train"].states is None
    assert not (tmp_path / "data/train_states.pt").exists()
    assert completion["settings"]["transport_chart"] == "isotropic"
    assert completion["settings"]["candidate"] == "rotating_pair_innovation"
    assert completion["settings"]["action_mix"] == "paley24"
    assert data["train"].tensors["actions"].shape == (4, 10, 11, 24)
    assert set(data["train"].batch(torch.tensor([0]), "cpu")) == {
        "observations",
        "actions",
        "oracle_parameters",
        "episode_ids",
    }
    all_data, _ = load_data(tmp_path / "data", include_test=True)
    assert (
        len({int(i) for sample in all_data.values() for i in sample.tensors["episode_ids"]}) == 10
    )
    payload = _checkpoint(tmp_path / "data/train.pt")
    assert set(payload) == {"tensors", "metadata"}


def test_one_seed_calibration_fresh_sparse_start_and_separate_final_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prepare(tmp_path)
    evaluated: list[str] = []
    actual = runner.evaluate

    def observe(
        model: SCJEPA,
        data: LiveStateData,
        batch_size: int,
        device: torch.device | str,
        objective: TrainingObjective,
    ) -> EvaluationResult:
        evaluated.append(cast(StateData, data).split)
        return actual(model, data, batch_size, device, objective)

    monkeypatch.setattr(runner, "evaluate", observe)
    bound = _bound(tmp_path)
    assert bound["tau"] == 1.1 * read_json(tmp_path / "dense/calibration.json")["metrics"]["base"]
    full = runner.train(_config(tmp_path, "full"))
    dense = runner.verify_completion(tmp_path / "dense")
    for key in ("initial_encoder_identity", "initial_predictor_identity"):
        assert dense[key] == full[key]
    assert "test" not in evaluated
    assert not (tmp_path / "full/test.pt").exists()
    tested = runner.score_test(
        runner.TestConfig(str(tmp_path / "tested"), str(tmp_path / "full"), str(tmp_path / "data"))
    )
    assert evaluated.count("test") == 1
    assert tested["parent_identity"] == full["identity"]
    assert read_json(tmp_path / "tested/matching.json") == read_json(
        tmp_path / "full/matching.json"
    )
    assert runner.verify_completion(tmp_path / "full") == full
    assert runner.verify_completion(tmp_path / "tested", kind="test") == tested
    assert runner.train(_config(tmp_path, "full"), resume=True) == full


@pytest.mark.parametrize("condition", ["dense", "mlp", "full"])
def test_exact_interruption_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: runner.Condition
) -> None:
    _prepare(tmp_path)
    if condition == "full":
        _bound(tmp_path)
    config = _config(tmp_path, condition)
    complete = runner.train(config)
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    actual = runner.forward_loss
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
        patch.setattr(runner, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            runner.train(interrupted)
    assert _checkpoint(tmp_path / "interrupted/last.pt")["step"] == 2
    resumed = runner.train(interrupted, resume=True)
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


def test_source_changes_and_protocol_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="fixed"):
        runner.TrainConfig("unused", "unused", "dense", seed=1)
    with pytest.raises(ValueError, match="100000"):
        runner.TrainConfig("unused", "unused", "dense", steps=10)
    with pytest.raises(ValueError, match="100000"):
        runner.TrainConfig("unused", "unused", "dense", steps=30000)
    with pytest.raises(ValueError, match="prespecified"):
        runner.CalibrateConfig("unused", ("unused",), tau_factor=2)
    _prepare(tmp_path)
    config = _config(tmp_path, "mlp")
    runner.train(config)
    monkeypatch.setattr(runner, "code_identity", lambda: "changed-source")
    with pytest.raises(ValueError, match="configuration, source"):
        runner.train(config, resume=True)


def test_balanced_cache_and_run_cannot_substitute_for_rotating_pair_control(tmp_path: Path) -> None:
    _prepare(tmp_path)
    with pytest.raises(ValueError, match="schema|source|identity|rotating"):
        runner.train(replace(_config(tmp_path, "mlp"), data_dir=str(tmp_path / "source")))
    from diagnostics.balanced_action_oracle import run as source_runner

    original = source_runner.TrainConfig(
        str(tmp_path / "original_dense"),
        str(tmp_path / "source"),
        "dense",
        profile="smoke",
        steps=4,
        batch_size=2,
    )
    source_runner.train(original)
    with pytest.raises(ValueError, match="schema"):
        runner.calibrate(
            runner.CalibrateConfig(
                str(tmp_path / "wrong_bound"),
                (str(tmp_path / "original_dense"),),
                profile="smoke",
            )
        )


def test_new_source_identity_is_isolated_from_completed_parent_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_before, current_before = parent_code_identity(), code_identity()
    assert parent_before != current_before
    original_read = Path.read_bytes
    changed_path = Path(runner.__file__).resolve()

    def hypothetical_change(path: Path) -> bytes:
        result = original_read(path)
        return (
            result + b"\n# hypothetical isolated source change\n"
            if path.resolve() == changed_path
            else result
        )

    monkeypatch.setattr(Path, "read_bytes", hypothetical_change)
    assert parent_code_identity() == parent_before
    assert code_identity() != current_before


def test_two_step_gradients_and_oracle_target_freezing(tmp_path: Path) -> None:
    _prepare(tmp_path)
    data, _ = load_data(tmp_path / "data")
    batch = data["train"].batch(torch.tensor([0, 1]), "cpu")
    batch["oracle_parameters"].requires_grad_(True)
    model = build_control("dense", 0)
    objective = TrainingObjective(runner.objective_config("dense"))
    loss, _, output = runner.forward_loss(model, batch, objective, forward_seed=71)
    first = output.dynamics[0].prediction
    second_loss = output.predictions[:, :, 1].square().mean()
    attached = torch.autograd.grad(second_loss, first, retain_graph=True)[0]
    assert bool(torch.isfinite(attached).all())
    assert float(attached.abs().sum()) > 0
    assert output.prediction_start == 9
    assert output.predictions.shape == (2, 10, 2, 6)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    for module in (model.online_encoder, model.predictor):
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
            for parameter in module.parameters()
        )
    for module in (model.target_encoder, model.parameter_encoder):
        assert all(not parameter.requires_grad for parameter in module.parameters())
        assert all(parameter.grad is None for parameter in module.parameters())
    assert batch["oracle_parameters"].grad is None
    assert not output.targets.requires_grad
    assert not output.parameters.requires_grad
