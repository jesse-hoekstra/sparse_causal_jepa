"""Frozen-code transfer preserves information, branch isolation and exact resume."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import frozen_code as control
from diagnostics.masked_mlp_learned import run as learned
from diagnostics.masked_mlp_oracle.model import MaskedTrunkPredictor
from diagnostics.shared.data import PrepareConfig, StateData, prepare
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.oracle import objective_config
from diagnostics.shared.source_data import PrepareConfig as SourcePrepareConfig
from diagnostics.shared.source_data import prepare as prepare_source
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import TrainingObjective


def _model(condition: control.Arm = "full") -> control.FrozenCodeModel:
    baseline = learned.build_control("baseline", 0, predictor_activation="gelu")
    donor = learned.build_control("full", 1, predictor_activation="gelu")
    return control.FrozenCodeModel(
        baseline,
        donor,
        condition,
        torch.arange(1, 7).float(),
        control.pair_rotation(((0, 3), (1, 4), (2, 5))),
    )


def _inputs() -> dict[str, Tensor]:
    return {"observations": torch.randn(2, 10, 12, 6), "actions": torch.randn(2, 10, 11, 24)}


def test_pair_selection_uses_only_learned_support_and_rotation_is_invertible() -> None:
    graph = torch.eye(3, dtype=torch.bool).repeat(2, 2)
    permutation = [5, 2, 0, 4, 3, 1]
    graph = graph[:, permutation]
    pairs = control.learned_pairs(graph)
    assert pairs == ((0, 1), (2, 4), (3, 5))
    rotation = control.pair_rotation(pairs)
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(6, dtype=torch.float64))
    codes = torch.randn(40, 6, dtype=torch.float64)
    scale = torch.arange(1, 7).double()
    transformed = (codes / scale) @ rotation.T
    torch.testing.assert_close((transformed @ rotation) * scale, codes)
    for i, j in pairs:
        assert torch.equal(graph[:, i], graph[:, j])
    with pytest.raises(ValueError, match="three equal-column pairs"):
        control.learned_pairs(torch.ones(6, 6, dtype=torch.bool))
    with pytest.raises(ValueError, match="no learned children"):
        control.learned_pairs(torch.zeros(6, 6, dtype=torch.bool))


def test_initial_state_predictor_and_ema_are_copied_without_compensating_weights() -> None:
    baseline = learned.build_control("baseline", 0, predictor_activation="gelu")
    donor = learned.build_control("full", 1, predictor_activation="gelu")
    with torch.no_grad():
        next(baseline.target_encoder.parameters()).add_(0.123)
    args = (baseline, donor)
    scale, rotation = torch.ones(6), control.pair_rotation(((0, 3), (1, 4), (2, 5)))
    full = control.FrozenCodeModel(*args, "full", scale, rotation)
    reference = control.FrozenCodeModel(*args, "baseline", scale, rotation)
    for model in (full, reference):
        torch.testing.assert_close(
            model.online_encoder.state_dict(), baseline.online_encoder.state_dict()
        )
        torch.testing.assert_close(
            model.target_encoder.state_dict(), baseline.target_encoder.state_dict()
        )
        torch.testing.assert_close(
            model.parameter_state_encoder.state_dict(), donor.online_encoder.state_dict()
        )
        torch.testing.assert_close(
            model.parameter_encoder.state_dict(), donor.parameter_encoder.state_dict()
        )
        for key, value in baseline.predictor.state_dict().items():
            torch.testing.assert_close(model.predictor.state_dict()[key], value, rtol=0, atol=0)
    assert full.frozen_identity() == reference.frozen_identity()
    assert isinstance(full.predictor, MaskedTrunkPredictor)
    assert isinstance(reference.predictor, MaskedTrunkPredictor)
    assert reference.predictor.parameter_logits is None
    assert full.predictor.parameter_logits is not None
    assert torch.equal(full.predictor.parameter_logits, torch.full((6, 6), 3.0))


def test_frozen_codes_ignore_state_updates_and_future_frames_with_attached_rollout() -> None:
    torch.set_num_threads(1)
    model = _model().eval()
    inputs = _inputs()
    output = cast(ModelOutput, model(**inputs))
    context = model.parameter_state_encoder.encode_sequence(
        inputs["observations"].flatten(0, 1)[:, :10]
    )
    raw = model.parameter_encoder(context, inputs["actions"].flatten(0, 1)[:, :9])
    expected = ((raw / model.code_scale) @ model.code_rotation.T).reshape(2, 10, 6)
    torch.testing.assert_close(output.parameters, expected, rtol=0, atol=0)
    first = output.dynamics[0].prediction
    first.retain_grad()
    output.predictions[:, :, 1].square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert first.grad is not None
    assert float(first.grad.abs().sum()) > 0
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in model.online_encoder.parameters()
    )
    assert all(p.grad is None for p in model.parameter_encoder.parameters())
    assert all(p.grad is None for p in model.parameter_state_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    fixed = model.frozen_identity()
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(0.2)
    model.update_target()
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["observations"][:, :, 10:] += 100
    changed["actions"][:, :, 9:] -= 100
    after = cast(ModelOutput, model(**changed))
    assert model.frozen_identity() == fixed
    torch.testing.assert_close(after.parameters, output.parameters, rtol=0, atol=0)
    model.train()
    assert model.online_encoder.training
    assert not model.parameter_encoder.training
    assert not model.parameter_state_encoder.training
    assert not model.target_encoder.training
    with pytest.raises(ValueError, match="never consumes true"):
        model(**inputs, oracle_parameters=torch.randn(2, 6))


def test_donor_routing_matches_the_existing_model_when_transform_is_identity() -> None:
    baseline = learned.build_control("baseline", 0, predictor_activation="gelu")
    donor = learned.build_control("full", 1, predictor_activation="gelu")
    baseline.parameter_encoder.load_state_dict(donor.parameter_encoder.state_dict())
    donor.online_encoder.load_state_dict(baseline.online_encoder.state_dict())
    frozen = control.FrozenCodeModel(baseline, donor, "baseline", torch.ones(6), torch.eye(6))
    inputs = _inputs()
    objective = TrainingObjective(objective_config("baseline", 0.0))
    for training in (False, True):
        baseline.train(training)
        frozen.train(training)
        torch.manual_seed(12)  # pyright: ignore[reportUnknownMemberType]
        _, _, expected = learned.forward_loss(baseline, inputs, objective, forward_seed=55)
        torch.manual_seed(12)  # pyright: ignore[reportUnknownMemberType]
        _, _, actual = learned.forward_loss(frozen, inputs, objective, forward_seed=55)
        for key in ("parameters", "online_states", "targets", "predictions"):
            torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)


def test_scale_fits_training_observations_only() -> None:
    donor = learned.build_control("full", 0, predictor_activation="gelu")
    inputs = _inputs() | {"oracle_parameters": torch.randn(2, 6), "episode_ids": torch.arange(2)}
    train = StateData("train", inputs, None, {})
    expected = control.training_scale(donor, train, torch.device("cpu"))
    poisoned = StateData("train", dict(inputs), None, {})
    poisoned.tensors["oracle_parameters"] = torch.full((2, 6), float("nan"))
    torch.testing.assert_close(
        control.training_scale(donor, poisoned, torch.device("cpu")), expected, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="training groups only"):
        control.training_scale(
            donor, StateData("validation", inputs, None, {}), torch.device("cpu")
        )


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def test_training_resume_keeps_frozen_branch_and_sources_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_source(
        SourcePrepareConfig(
            str(tmp_path / "source"),
            train_groups=4,
            calibration_groups=4,
            validation_groups=2,
            test_groups=2,
            profile="smoke",
        )
    )
    prepare(PrepareConfig(str(tmp_path / "data"), str(tmp_path / "source"), profile="smoke"))
    dense = learned.TrainConfig(
        str(tmp_path / "dense"),
        str(tmp_path / "data"),
        "dense",
        profile="smoke",
        steps=2,
        batch_size=2,
        code_dim=6,
        predictor_activation="gelu",
    )
    learned.train(dense)
    learned.calibrate(
        learned.CalibrateConfig(str(tmp_path / "bound"), dense.output_dir, profile="smoke")
    )
    for arm in ("baseline", "full"):
        learned.train(
            replace(
                dense,
                output_dir=str(tmp_path / arm),
                condition=arm,
                init_dir=dense.output_dir,
                calibration_dir=str(tmp_path / "bound"),
                code_dim=None,
            )
        )
    # A synthetic paired-support donor fixture; no physical labels are used by the control.
    path = tmp_path / "full" / "last.pt"
    checkpoint = _load(path)
    checkpoint["model"]["predictor.parameter_logits"] = torch.eye(3).repeat(2, 2) * 8 - 4
    save_torch(path, checkpoint)
    completion = read_json(path.parent / "completion.json")
    completion["hashes"]["last.pt"] = sha(path)
    write_json(path.parent / "completion.json", completion)
    before = {
        str(p): sha(p)
        for arm in ("baseline", "full")
        for p in (tmp_path / arm).iterdir()
        if p.is_file()
    }
    config = control.ControlConfig(
        str(tmp_path / "reference"),
        str(tmp_path / "baseline"),
        str(tmp_path / "full"),
        str(tmp_path / "data"),
        "full",
        0,
        profile="smoke",
        steps=4,
        batch_size=2,
        log_every=1,
        checkpoint_every=1,
        evaluate_every=3,
    )
    result = control.train(config)
    expected = _load(tmp_path / "reference" / "last.pt")
    assert result["frozen_branch_unchanged"]
    assert not result["test_evaluated"]
    initial = _load(tmp_path / "reference" / "initial.pt")
    for key, value in initial["model"].items():
        if key.startswith(("parameter_encoder.", "parameter_state_encoder.", "code_")):
            torch.testing.assert_close(value, expected["model"][key], rtol=0, atol=0)
    assert not torch.equal(
        initial["model"]["predictor.first.weight"], expected["model"]["predictor.first.weight"]
    )
    assert "code_mcc_columns" in read_json(tmp_path / "reference" / "matching.json")
    interrupted = replace(config, output_dir=str(tmp_path / "interrupted"))
    calls = 0
    forward = learned.forward_loss

    def interrupt(
        model: SCJEPA,
        batch: dict[str, Tensor],
        objective: TrainingObjective,
        *,
        forward_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], ModelOutput]:
        nonlocal calls
        if model.training:
            calls += 1
            if calls == 3:
                raise RuntimeError("deliberate interruption")
        return forward(model, batch, objective, forward_seed=forward_seed)

    with monkeypatch.context() as patch:
        patch.setattr(learned, "forward_loss", interrupt)
        with pytest.raises(RuntimeError, match="deliberate"):
            control.train(interrupted)
    assert _load(tmp_path / "interrupted" / "last.pt")["step"] == 2
    with pytest.raises(ValueError, match="resume source, protocol"):
        control.train(replace(interrupted, steps=5), resume=True)
    resumed = control.train(interrupted, resume=True)
    assert resumed["identity"] == result["identity"]
    actual = _load(tmp_path / "interrupted" / "last.pt")
    for key in ("model", "optimizer", "objective", "sampler", "torch_rng"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert {p: sha(Path(p)) for p in before} == before
    control.verify_completion(tmp_path / "interrupted")
