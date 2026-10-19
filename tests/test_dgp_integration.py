"""Tiny CPU learning and exact-resume contracts for scientific dataset adapters."""

import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data import TensorEpisodes
from scjepa.data.bundle import load_dataset_bundle, save_dataset_bundle
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.types import DGPConfig
from scjepa.models.model import SCJEPA
from scjepa.training import ObjectiveConfig, TrainConfig, Trainer, TrainingObjective
from scjepa.training.factory import ModelConfig, build_model


def _model_config(family: str = "controlled_vector") -> ModelConfig:
    spring = family == "coupled_springs"
    return ModelConfig(
        observation_kind="image" if spring else "vector",
        observation_dim=10,
        in_channels=3,
        encoder_hidden_dims=(8,),
        image_channels=(4, 4),
        image_pool_size=1,
        state_dim=4,
        param_dim=3,
        action_dim=5 if spring else 10,
        context_len=8 if spring else 2,
        history_len=2 if spring else 1,
        parameter_embed_dim=8,
        parameter_heads=2,
        max_context_len=10,
        spartan_layers=1,
        spartan_embed_dim=8,
        spartan_mlp_hidden=8,
        spartan_mlp_layers=2,
        spartan_dense=True,
    )


def _batch(dataset: Dataset[dict[str, Tensor]]) -> dict[str, Tensor]:
    episodes = [dataset[0], dataset[1]]
    assert set(episodes[0]) == {"observations", "actions", "episode_ids"}
    return {key: torch.stack([episode[key] for episode in episodes]) for key in episodes[0]}


def _trainer(
    dataset: Dataset[dict[str, Tensor]], path: Path, config: ModelConfig | None = None
) -> Trainer:
    model_config = _model_config() if config is None else config
    return Trainer(
        build_model(model_config),
        TrainConfig(batch_size=2, steps=1, out_dir=str(path)),
        dataset=dataset,
        model_config=asdict(model_config),
    )


@pytest.mark.parametrize("family", ["controlled_vector", "coupled_springs"])
def test_tiny_native_dgp_cpu_forward_backward_without_ground_truth(
    family: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = DGPConfig(
        family=family,
        num_episodes=2,
        trajectory_length=10 if family == "coupled_springs" else 3,
        root_seed=51,
    )
    dataset = build_dgp_dataset(config)

    def forbidden_ground_truth(index: int) -> dict[str, Tensor]:
        raise AssertionError(f"training requested ground truth at {index}")

    monkeypatch.setattr(dataset, "ground_truth", forbidden_ground_truth)
    trainer = _trainer(dataset, tmp_path, _model_config(family))
    batch = _batch(dataset)
    assert batch["observations"].dtype == batch["actions"].dtype == torch.float32
    if family == "coupled_springs":
        assert batch["observations"].shape == (2, 10, 3, 320, 64)
    else:
        assert batch["observations"].shape == (2, 3, 10)
    model: SCJEPA = trainer.model
    output = model.forward(batch["observations"], batch["actions"])
    predictions = 2 if family == "coupled_springs" else 1
    assert output.predictions.shape == output.targets.shape == (2, predictions, 4)
    if family == "controlled_vector":
        assert output.prediction_start == 1
    assert output.parameters.shape == (2, 3)
    assert not output.targets.requires_grad
    loss, _ = TrainingObjective(ObjectiveConfig()).forward(output)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert torch.isfinite(loss)
    for module in (model.online_encoder, model.parameter_encoder, model.predictor):
        gradients = [value.grad for value in module.parameters() if value.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    assert all(value.grad is None for value in model.target_encoder.parameters())
    assert trainer.dataset_identity == dataset.fingerprint


def test_checkpoint_accepts_identical_lazy_dataset(tmp_path: Path) -> None:
    config = DGPConfig(num_episodes=2, trajectory_length=4, parameter_graph="complementary")
    dataset = build_dgp_dataset(config)
    source = _trainer(dataset, tmp_path / "source")
    source.train_step(_batch(dataset))
    checkpoint = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint)
    payload: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["dataset_identity"] == dataset.fingerprint
    expected_model = copy.deepcopy(source.model.state_dict())
    expected_optimizer = copy.deepcopy(source.optimizer.state_dict())
    lazy = build_dgp_dataset(config)
    resumed = _trainer(lazy, tmp_path / "resumed")
    resumed.load_checkpoint(checkpoint)
    assert resumed.step == resumed.optimizer_steps == 1
    torch.testing.assert_close(resumed.model.state_dict(), expected_model, rtol=0, atol=0)
    torch.testing.assert_close(resumed.optimizer.state_dict(), expected_optimizer, rtol=0, atol=0)
    actual_batch = _batch(lazy)
    torch.testing.assert_close(actual_batch, _batch(dataset), rtol=0, atol=0)
    assert resumed.train_step(actual_batch)["health/updated"] == 1


def test_stored_bundle_resume_binds_content_and_rejects_generated_source(tmp_path: Path) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=2, trajectory_length=4))
    generated = _trainer(dataset, tmp_path / "generated")
    generated_path = tmp_path / "generated.pt"
    generated.save_checkpoint(generated_path)
    manifest = save_dataset_bundle(dataset, tmp_path / "bundle")
    stored = load_dataset_bundle(manifest)
    assert stored.fingerprint == dataset.fingerprint
    source = _trainer(stored, tmp_path / "stored")
    assert source.dataset_identity != generated.dataset_identity
    with pytest.raises(ValueError, match="dataset_identity"):
        source.load_checkpoint(generated_path)
    source.train_step(_batch(stored))
    checkpoint = tmp_path / "stored.pt"
    source.save_checkpoint(checkpoint)
    copied = load_dataset_bundle(manifest)
    resumed = _trainer(copied, tmp_path / "resumed")
    resumed.load_checkpoint(checkpoint)
    assert resumed.step == resumed.optimizer_steps == 1
    assert resumed.dataset_identity == source.dataset_identity
    torch.testing.assert_close(
        resumed.model.state_dict(), source.model.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), source.optimizer.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(_batch(copied), _batch(dataset), rtol=0, atol=0)
    assert resumed.train_step(_batch(copied))["health/updated"] == 1


def test_mutation_after_trainer_creation_prevents_checkpoint_save_and_load(tmp_path: Path) -> None:
    generated = build_dgp_dataset(DGPConfig(num_episodes=2, trajectory_length=4))
    batch = _batch(generated)
    dataset = TensorEpisodes(batch["observations"], batch["actions"], batch["episode_ids"])
    trainer = _trainer(dataset, tmp_path)
    checkpoint = tmp_path / "before.pt"
    trainer.save_checkpoint(checkpoint)
    dataset.observations[0, 0, 0] += 0.5
    expected_model = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError, match="dataset_identity changed"):
        trainer.load_checkpoint(checkpoint)
    torch.testing.assert_close(trainer.model.state_dict(), expected_model, rtol=0, atol=0)
    with pytest.raises(ValueError, match="dataset_identity changed"):
        trainer.save_checkpoint(tmp_path / "after.pt")
    assert not (tmp_path / "after.pt").exists()


@pytest.mark.parametrize(
    "difference", ["graph", "root_seed", "observation_seed", "split", "length"]
)
def test_changed_dataset_rejected_before_restoring_any_training_state(
    difference: str, tmp_path: Path
) -> None:
    config = DGPConfig(num_episodes=2, trajectory_length=4)
    original = build_dgp_dataset(config)
    source = _trainer(original, tmp_path / "source")
    source.train_step(_batch(original))
    checkpoint = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint)
    alternatives = {
        "graph": replace(config, parameter_graph="complementary"),
        "root_seed": replace(config, root_seed=27),
        "observation_seed": replace(config, observation_seed=1730),
        "split": replace(config, split="validation"),
        "length": replace(config, trajectory_length=5),
    }
    changed = build_dgp_dataset(alternatives[difference])
    resumed = _trainer(changed, tmp_path / "resumed")
    initial_model = copy.deepcopy(resumed.model.state_dict())
    initial_optimizer = copy.deepcopy(resumed.optimizer.state_dict())
    initial_config = copy.deepcopy(resumed.config)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="dataset_identity"):
        resumed.load_checkpoint(checkpoint)
    assert resumed.step == resumed.optimizer_steps == 0
    assert resumed.config == initial_config
    torch.testing.assert_close(resumed.model.state_dict(), initial_model, rtol=0, atol=0)
    torch.testing.assert_close(resumed.optimizer.state_dict(), initial_optimizer, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_saved_tensor_route_fingerprints_content_and_rejects_changed_trajectories(
    tmp_path: Path,
) -> None:
    generated = build_dgp_dataset(DGPConfig(num_episodes=2, trajectory_length=4))
    batch = _batch(generated)
    path = tmp_path / "episodes.pt"
    torch.save(batch, path)
    tensors = TensorEpisodes.load(path)
    trainer = _trainer(tensors, tmp_path / "source")
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    exact_copy = TensorEpisodes.load(path)
    restored = _trainer(exact_copy, tmp_path / "restored")
    restored.load_checkpoint(checkpoint)
    changed_observations = batch["observations"].clone()
    changed_observations[0, 0, 0] += 0.01
    changed = TensorEpisodes(changed_observations, batch["actions"], batch["episode_ids"])
    mismatch = _trainer(changed, tmp_path / "mismatch")
    with pytest.raises(ValueError, match="dataset_identity"):
        mismatch.load_checkpoint(checkpoint)


def test_identity_cannot_override_real_dataset_and_old_unidentified_resume_fails(
    tmp_path: Path,
) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=2, trajectory_length=4))
    with pytest.raises(ValueError, match="supplied dataset fingerprint"):
        Trainer(
            build_model(_model_config()),
            TrainConfig(batch_size=2),
            dataset,
            dataset_identity="false identity",
        )
    trainer = _trainer(dataset, tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint)
    payload: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("dataset_identity")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="dataset_identity"):
        trainer.load_checkpoint(checkpoint)


def test_manual_batches_accept_explicit_identity_and_check_resume(tmp_path: Path) -> None:
    config = TrainConfig(batch_size=2)
    trainer = Trainer(build_model(_model_config()), config, dataset_identity="manual-stream-v1")
    checkpoint = tmp_path / "manual.pt"
    trainer.save_checkpoint(checkpoint)
    restored = Trainer(build_model(_model_config()), config, dataset_identity="manual-stream-v1")
    restored.load_checkpoint(checkpoint)
    unidentified = Trainer(build_model(_model_config()), config)
    with pytest.raises(ValueError, match="dataset_identity"):
        unidentified.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="nonempty"):
        Trainer(build_model(_model_config()), config, dataset_identity="")


@pytest.mark.parametrize("mismatch", ["observation", "action", "history", "length"])
def test_dataset_model_compatibility_is_rejected_during_trainer_initialization(
    mismatch: str, tmp_path: Path
) -> None:
    spring = mismatch == "history"
    dataset = build_dgp_dataset(
        DGPConfig(
            family="coupled_springs" if spring else "controlled_vector",
            num_episodes=2,
            trajectory_length=10 if spring else 4,
        )
    )
    config = _model_config("coupled_springs" if spring else "controlled_vector")
    if mismatch == "observation":
        config.observation_dim = 9
    elif mismatch == "action":
        config.action_dim = 2
    elif mismatch == "history":
        config.history_len = 1
    else:
        config.context_len = 4
    expected = {
        "observation": "observation dimension",
        "action": "actions must match",
        "history": "history is shorter",
        "length": "prediction target",
    }
    with pytest.raises(ValueError, match=expected[mismatch]):
        _trainer(dataset, tmp_path, config)
