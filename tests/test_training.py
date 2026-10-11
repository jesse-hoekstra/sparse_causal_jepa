"""CPU training contracts: losses, accepted updates, accumulation, and exact resume."""

import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.hooks import RemovableHandle

from scjepa.models.encoders import VectorEncoder
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.parameter_encoder import ParameterEncoder
from scjepa.models.spartan import GraphPenalties, Spartan
from scjepa.training import ObjectiveConfig, TrainConfig, Trainer, TrainingObjective
from scjepa.training.lagrangian import SparsityLagrangian


def tiny_model() -> SCJEPA:
    return SCJEPA(
        VectorEncoder(6, 3, (8,)),
        ParameterEncoder(3, 2, action_dim=1, dim=8, num_heads=2, max_history=5),
        Spartan(3, 2, action_dim=1, history_len=2, num_layers=1, embed_dim=8, mlp_hidden_size=8),
        context_len=3,
        ema_decay=0.8,
    )


def tiny_batch() -> dict[str, Tensor]:
    return {
        "observations": torch.randn(4, 5, 6),
        "actions": torch.randn(4, 4, 1),
        "episode_ids": torch.arange(4),
    }


def tiny_trainer(tmp_path: Path, accumulation_steps: int = 1) -> Trainer:
    return Trainer(
        tiny_model(),
        TrainConfig(batch_size=4, accumulation_steps=accumulation_steps, out_dir=str(tmp_path)),
        model_config={"test": "tiny"},
    )


def test_cpu_forward_backward_and_target_is_excluded_from_optimizer(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path)
    optimized = {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }
    assert optimized.isdisjoint(
        id(parameter) for parameter in trainer.model.target_encoder.parameters()
    )
    output = trainer.model(**{k: v for k, v in tiny_batch().items() if k != "episode_ids"})
    loss, _ = trainer.objective(output)
    loss.backward()
    for module in (
        trainer.model.online_encoder,
        trainer.model.parameter_encoder,
        trainer.model.predictor,
    ):
        gradients = [
            parameter.grad for parameter in module.parameters() if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(parameter.grad is None for parameter in trainer.model.target_encoder.parameters())
    assert not trainer.model.target_encoder.training


def test_accumulation_updates_optimizer_and_ema_exactly_once(tmp_path: Path) -> None:
    trainer = tiny_trainer(tmp_path, accumulation_steps=2)
    before_online = copy.deepcopy(trainer.model.online_encoder.state_dict())
    before_target = copy.deepcopy(trainer.model.target_encoder.state_dict())
    first = trainer.train_step(tiny_batch())
    assert first["health/updated"] == 0
    assert trainer.pending_microbatches == 1
    torch.testing.assert_close(
        trainer.model.online_encoder.state_dict(), before_online, rtol=0, atol=0
    )
    torch.testing.assert_close(
        trainer.model.target_encoder.state_dict(), before_target, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="accumulation boundary"):
        trainer.save_checkpoint(tmp_path / "partial.pt")
    second = trainer.train_step(tiny_batch())
    assert second["health/updated"] == 1
    assert trainer.optimizer_steps == 1
    assert trainer.pending_microbatches == 0
    assert trainer.step == 2
    for key, value in trainer.model.target_encoder.state_dict().items():
        expected = before_target[key] * 0.8 + trainer.model.online_encoder.state_dict()[key] * 0.2
        torch.testing.assert_close(value, expected)


@pytest.mark.parametrize("rejection", ["loss", "gradient", "threshold"])
def test_rejected_group_discards_accumulation_without_ema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejection: str
) -> None:
    trainer = tiny_trainer(tmp_path, accumulation_steps=2)
    before = copy.deepcopy(trainer.model.state_dict())
    trainer.train_step(tiny_batch())
    original_forward = trainer.objective.forward
    hook: RemovableHandle | None = None
    if rejection == "loss":

        def bad_loss(output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
            total, terms = original_forward(output)
            return total * float("nan"), terms

        monkeypatch.setattr(trainer.objective, "forward", bad_loss)
    elif rejection == "gradient":

        def poison(gradient: Tensor) -> Tensor:
            return torch.full_like(gradient, float("nan"))

        parameter = next(trainer.model.online_encoder.parameters())
        hook = cast(RemovableHandle, parameter.register_hook(poison))  # pyright: ignore[reportUnknownMemberType]
    else:
        trainer.config.grad_skip_threshold = 0
    metrics = trainer.train_step(tiny_batch())
    assert metrics["health/skipped"] == 1
    assert trainer.optimizer_steps == 0
    assert trainer.pending_microbatches == 0
    assert trainer.total_skips == 1
    assert not trainer.optimizer.state
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    torch.testing.assert_close(trainer.model.state_dict(), before, rtol=0, atol=0)
    if hook is not None:
        hook.remove()
    monkeypatch.setattr(trainer.objective, "forward", original_forward)
    trainer.config.grad_skip_threshold = float("inf")
    trainer.train_step(tiny_batch())
    trainer.train_step(tiny_batch())
    assert trainer.optimizer_steps == 1
    assert trainer.consecutive_skips == 0


def test_checkpoint_restores_ema_config_rng_optimizer_and_exact_continuation(
    tmp_path: Path,
) -> None:
    trainer = tiny_trainer(tmp_path)
    trainer.config.objective.state_sparsity_weight = 0.02
    trainer.train_step(tiny_batch())
    next_batch = tiny_batch()
    checkpoint = tmp_path / "state.pt"
    trainer.save_checkpoint(checkpoint)
    metrics = trainer.train_step(next_batch)
    expected_model = copy.deepcopy(trainer.model.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    expected_rng = torch.get_rng_state().clone()
    resumed = tiny_trainer(tmp_path)
    resumed.model.ema_decay = 0.2
    resumed.load_checkpoint(checkpoint)
    assert resumed.model.ema_decay == 0.8
    assert resumed.config.objective.state_sparsity_weight == 0.02
    assert resumed.step == resumed.optimizer_steps == 1
    actual_metrics = resumed.train_step(next_batch)
    assert actual_metrics == metrics
    torch.testing.assert_close(resumed.model.state_dict(), expected_model, rtol=0, atol=0)
    torch.testing.assert_close(resumed.optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    assert not resumed.model.target_encoder.training
    assert all(
        not parameter.requires_grad for parameter in resumed.model.target_encoder.parameters()
    )


def test_separate_sparsity_weights_change_only_the_selected_loss_block() -> None:
    model = tiny_model()
    batch = tiny_batch()
    output: ModelOutput = model(batch["observations"], batch["actions"])
    baseline, _ = TrainingObjective(ObjectiveConfig())(output)
    for name, block in (("state", "state"), ("action", "action"), ("parameter", "parameter")):
        config = ObjectiveConfig()
        setattr(config, f"{name}_sparsity_weight", 0.7)
        total, terms = TrainingObjective(config)(output)
        torch.testing.assert_close(total, baseline + 0.7 * terms[f"{block}_sparsity"])
    # Parameter-conditioned predictive gradients exist with parameter sparsity off.
    baseline.backward()
    gradient = model.parameter_encoder.head.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_disabled_auxiliary_terms_cannot_poison_prediction_with_infinite_diagnostics() -> None:
    model = tiny_model()
    batch = tiny_batch()
    output: ModelOutput = model(batch["observations"], batch["actions"])
    infinite = torch.tensor(float("inf"))
    dynamics = tuple(
        item._replace(sparsity=GraphPenalties(infinite, infinite, infinite), logit_penalty=infinite)
        for item in output.dynamics
    )
    loss, terms = TrainingObjective(ObjectiveConfig())(output._replace(dynamics=dynamics))
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, terms["prediction"])
    loss.backward()


def test_parameter_regularizer_uses_one_sample_per_distinct_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(
        tiny_model(),
        TrainConfig(
            batch_size=4,
            objective=ObjectiveConfig(
                parameter_regularizer="sigreg",
                parameter_regularizer_weight=0.1,
                regularizer_num_projections=8,
            ),
        ),
    )
    seen: list[tuple[int, ...]] = []
    original = trainer.objective.parameter_regularizer.forward

    def record(samples: Tensor) -> Tensor:
        seen.append(tuple(samples.shape))
        return original(samples)

    monkeypatch.setattr(trainer.objective.parameter_regularizer, "forward", record)
    batch = tiny_batch()
    trainer.train_step(batch)
    assert seen == [(1, 4, 2)]
    batch["episode_ids"] = torch.tensor([0, 1, 1, 2])
    with pytest.raises(ValueError, match="distinct episodes"):
        trainer.train_step(batch)
    del batch["episode_ids"]
    with pytest.raises(ValueError, match="episode_ids"):
        trainer.train_step(batch)
    trainer.save_checkpoint(tmp_path / "regularized.pt")


class _TensorEpisodes(Dataset[dict[str, Tensor]]):
    def __init__(self, count: int = 12) -> None:
        self.observations = torch.randn(count, 5, 6)
        self.actions = torch.randn(count, 4, 1)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"observations": self.observations[index], "actions": self.actions[index]}


def test_dataset_loop_and_resume_replay_batch_order(tmp_path: Path) -> None:
    dataset = _TensorEpisodes()
    trainer = Trainer(
        tiny_model(), TrainConfig(steps=2, batch_size=4, out_dir=str(tmp_path)), dataset
    )
    trainer.train()
    resumed = Trainer(
        tiny_model(), TrainConfig(steps=3, batch_size=4, out_dir=str(tmp_path)), dataset
    )
    resumed.load_checkpoint(tmp_path / "last.pt")
    expected = next(trainer._batches())  # pyright: ignore[reportPrivateUsage]
    actual = next(resumed._batches())  # pyright: ignore[reportPrivateUsage]
    torch.testing.assert_close(actual, expected)
    resumed.train()
    assert resumed.step == resumed.optimizer_steps == 3


def test_partial_final_group_is_discarded_and_checkpoint_is_safe(tmp_path: Path) -> None:
    trainer = Trainer(
        tiny_model(),
        TrainConfig(steps=3, batch_size=4, accumulation_steps=2, out_dir=str(tmp_path)),
        _TensorEpisodes(),
    )
    metrics = trainer.train()
    assert trainer.optimizer_steps == 1
    assert trainer.pending_microbatches == 0
    assert metrics["health/discarded_microbatches"] == 1
    assert (tmp_path / "last.pt").exists()


def test_config_validation_and_dataset_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no full batches"):
        Trainer(tiny_model(), TrainConfig(batch_size=16), _TensorEpisodes())
    with pytest.raises(ValueError, match="requires a dataset"):
        tiny_trainer(tmp_path).train()
    with pytest.raises(ValueError, match="episode count"):
        tiny_trainer(tmp_path).train_step({"observations": torch.randn(2, 5, 6)})
    invalid: list[dict[str, Any]] = [
        {"accumulation_steps": 0},
        {"lr": float("nan")},
        {"grad_clip": 0},
    ]
    for settings in invalid:
        with pytest.raises(ValueError, match="positive|finite"):
            TrainConfig(**settings)
    with pytest.raises(ValueError, match="regularizer weight"):
        ObjectiveConfig(state_regularizer_weight=1)


def test_optional_dual_controller_has_explicit_settings_and_serialized_state() -> None:
    controller = SparsityLagrangian(tau=0.5, step_size=0.1, lambda_init=2, momentum=0)
    torch.testing.assert_close(controller.penalty_weight, torch.tensor(0.5))
    controller.update(torch.tensor(1.5))
    torch.testing.assert_close(controller.ma_error, torch.tensor(1.0))
    restored = SparsityLagrangian(tau=0.5, step_size=0.1, lambda_init=1, momentum=0)
    restored.load_state_dict(controller.state_dict())
    torch.testing.assert_close(restored.penalty_weight, controller.penalty_weight)
