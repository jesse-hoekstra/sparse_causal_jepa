from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data.compatibility import validate_model_dataset
from scjepa.models import SpartanOutput
from scjepa.training import ObjectiveConfig, TrainConfig, Trainer, TrainingObjective
from scjepa.training.factory import ModelConfig, build_model


def small_config(source: str = "oracle") -> ModelConfig:
    return ModelConfig(
        observation_dim=6,
        state_dim=4,
        param_dim=3,
        action_dim=2,
        context_len=3,
        history_len=1,
        encoder_hidden_dims=(8,),
        parameter_encoder_kind="transition",
        parameter_source=source,
        parameter_embed_dim=8,
        parameter_heads=2,
        spartan_embed_dim=8,
        spartan_mlp_hidden=8,
        spartan_layers=1,
        spartan_dense=True,
        autoregressive=True,
    )


def oracle_batch() -> dict[str, Tensor]:
    return {
        "observations": torch.randn(2, 3, 5, 6),
        "actions": torch.randn(2, 3, 4, 2),
        "oracle_parameters": torch.tensor([[-0.6, 0.1, 0.7], [0.4, -0.3, 0.2]]),
        "episode_ids": torch.arange(2),
    }


class TensorEpisodes(Dataset[dict[str, Tensor]]):
    """Expose a small explicit oracle batch through the training dataset contract."""

    def __init__(self, batch: dict[str, Tensor]) -> None:
        """Retain tensors so each test can alter its intended contract violation."""
        self.batch = batch

    def __len__(self) -> int:
        """Return the number of independent parameter groups."""
        return self.batch["observations"].shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return one group without changing its declared keys or dtypes."""
        return {key: value[index] for key, value in self.batch.items()}


def test_oracle_initialization_preserves_learned_tensors_and_rng() -> None:
    torch.default_generator.manual_seed(37)
    learned = build_model(small_config("learned"))
    expected_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(37)
    oracle = build_model(small_config())
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
    for (name, expected), (actual_name, actual) in zip(
        learned.named_parameters(), oracle.named_parameters(), strict=True
    ):
        assert actual_name == name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in oracle.parameter_encoder.parameters())
    assert all(parameter.requires_grad for parameter in learned.parameter_encoder.parameters())
    oracle.train()
    assert not oracle.parameter_encoder.training
    with pytest.raises(ValueError, match="parameter_source"):
        build_model(replace(small_config(), parameter_source="unknown"))


@pytest.mark.parametrize("grouped", [False, True])
def test_oracle_bypasses_encoder_reuses_raw_values_and_keeps_state_gradients(
    grouped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build_model(small_config()).train()
    batch = oracle_batch()
    observations = batch["observations"] if grouped else batch["observations"][:, 0]
    actions = batch["actions"] if grouped else batch["actions"][:, 0]
    theta = batch["oracle_parameters"].requires_grad_()
    calls: list[tuple[Tensor, Tensor, Tensor]] = []
    original = model.predictor.forward

    def forbid_encoder(_states: Tensor, _actions: Tensor | None = None) -> Tensor:
        raise AssertionError("oracle forward must bypass the parameter encoder")

    def record(
        history: Tensor, parameters: Tensor, controls: Tensor | None = None
    ) -> SpartanOutput:
        result = original(history, parameters, controls)
        calls.append((history, parameters, result.prediction))
        return result

    monkeypatch.setattr(model.parameter_encoder, "forward", forbid_encoder)
    monkeypatch.setattr(model.predictor, "forward", record)
    result = model(observations, actions, oracle_parameters=theta)
    expected = theta[:, None].expand(-1, 3, -1) if grouped else theta
    torch.testing.assert_close(result.parameters, expected, rtol=0, atol=0)
    assert not result.parameters.requires_grad
    assert len(calls) == 2
    assert calls[0][1] is calls[1][1]
    torch.testing.assert_close(calls[0][1], expected.reshape(-1, 3), rtol=0, atol=0)
    torch.testing.assert_close(calls[1][0][:, -1], calls[0][2])
    loss, _ = TrainingObjective(ObjectiveConfig())(result)
    loss.backward()
    for module in (model.online_encoder, model.predictor):
        gradients = [
            parameter.grad for parameter in module.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert theta.grad is None
    assert all(parameter.grad is None for parameter in model.parameter_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


@pytest.mark.parametrize("problem", ["missing", "shape", "dtype", "integer", "nonfinite", "device"])
def test_oracle_rejects_missing_or_invalid_parameters(problem: str) -> None:
    model = build_model(small_config())
    batch = oracle_batch()
    theta: Tensor | None = batch["oracle_parameters"]
    if problem == "missing":
        theta = None
    elif problem == "shape":
        theta = torch.zeros(2, 3, 3)
    elif problem == "dtype":
        theta = torch.zeros(2, 3, dtype=torch.float64)
    elif problem == "integer":
        theta = torch.zeros(2, 3, dtype=torch.long)
    elif problem == "nonfinite":
        theta = torch.full((2, 3), torch.nan)
    else:
        theta = torch.empty(2, 3, device="meta")
    with pytest.raises(ValueError, match="oracle_parameters"):
        model(batch["observations"], batch["actions"], oracle_parameters=theta)


def test_learned_mode_rejects_oracle_data() -> None:
    model = build_model(small_config("learned"))
    batch = oracle_batch()
    with pytest.raises(ValueError, match="cannot consume oracle_parameters"):
        model(batch["observations"], batch["actions"], oracle_parameters=batch["oracle_parameters"])
    with pytest.raises(ValueError, match="ground-truth"):
        validate_model_dataset(model, TensorEpisodes(batch))
    trainer = Trainer(model, TrainConfig(batch_size=2))
    with pytest.raises(ValueError, match="cannot consume oracle_parameters"):
        trainer.train_step(batch)


@pytest.mark.parametrize("problem", ["missing", "shape", "dtype", "nonfinite", "extra_label"])
def test_oracle_dataset_contract_rejects_invalid_items(problem: str) -> None:
    model = build_model(small_config())
    batch = oracle_batch()
    validate_model_dataset(model, TensorEpisodes(batch))
    if problem == "missing":
        batch.pop("oracle_parameters")
    elif problem == "shape":
        batch["oracle_parameters"] = torch.zeros(2, 3, 3)
    elif problem == "dtype":
        batch["oracle_parameters"] = batch["oracle_parameters"].double()
    elif problem == "nonfinite":
        batch["oracle_parameters"][0, 0] = torch.inf
    else:
        batch["states"] = torch.zeros(2, 3, 5, 4)
    with pytest.raises(ValueError, match="oracle|ground-truth"):
        validate_model_dataset(model, TensorEpisodes(batch))


@pytest.mark.parametrize("term", ["regularizer", "consistency"])
def test_oracle_trainer_rejects_parameter_learning_penalties(term: str) -> None:
    objective = ObjectiveConfig(
        parameter_regularizer="sigreg",
        parameter_regularizer_weight=0.1 if term == "regularizer" else 0,
        parameter_consistency_weight=1 if term == "consistency" else 0,
    )
    with pytest.raises(ValueError, match="zero parameter"):
        Trainer(build_model(small_config()), TrainConfig(objective=objective))


def test_oracle_training_and_resume_exclude_parameter_encoder(tmp_path: Path) -> None:
    config = small_config()
    training = TrainConfig(batch_size=2, out_dir=str(tmp_path))
    trainer = Trainer(build_model(config), training, model_config=asdict(config))
    encoder_before = {
        name: value.clone() for name, value in trainer.model.parameter_encoder.state_dict().items()
    }
    optimized = {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }
    assert optimized.isdisjoint(
        id(parameter) for parameter in trainer.model.parameter_encoder.parameters()
    )
    batch = oracle_batch()
    assert trainer.train_step(batch)["health/updated"] == 1
    checkpoint = tmp_path / "oracle.pt"
    trainer.save_checkpoint(checkpoint)
    expected = trainer.train_step(batch)
    resumed = Trainer(build_model(config), training, model_config=asdict(config))
    resumed.load_checkpoint(checkpoint)
    assert resumed.train_step(batch) == expected
    torch.testing.assert_close(
        resumed.model.state_dict(), trainer.model.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed.model.parameter_encoder.state_dict(), encoder_before, rtol=0, atol=0
    )
    learned = Trainer(build_model(small_config("learned")), training)
    with pytest.raises(ValueError, match="parameter_source"):
        learned.load_checkpoint(checkpoint)


def test_checkpoint_source_rejection_and_legacy_learned_default(tmp_path: Path) -> None:
    learned_config = small_config("learned")
    learned, oracle = build_model(learned_config), build_model(small_config())
    with pytest.raises(ValueError, match="parameter_source"):
        oracle.load_state_dict(learned.state_dict())
    with pytest.raises(ValueError, match="parameter_source"):
        learned.load_state_dict(oracle.state_dict())
    training = TrainConfig(batch_size=2, out_dir=str(tmp_path))
    trainer = Trainer(learned, training, model_config=asdict(learned_config))
    checkpoint = tmp_path / "legacy.pt"
    trainer.save_checkpoint(checkpoint)
    payload: dict[str, Any] = torch.load(checkpoint, weights_only=False)
    payload["model_config"].pop("parameter_source")
    payload["model"]["_extra_state"].pop("oracle_parameters")
    torch.save(payload, checkpoint)
    restored = Trainer(build_model(learned_config), training, model_config=asdict(learned_config))
    restored.load_checkpoint(checkpoint)
    assert restored.model.parameter_source == "learned"
    incompatible = Trainer(
        build_model(small_config()), training, model_config=asdict(small_config())
    )
    with pytest.raises(ValueError, match="model_config"):
        incompatible.load_checkpoint(checkpoint)
