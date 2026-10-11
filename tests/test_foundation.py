"""Generic data, config, and reusable evaluation contracts."""

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from scjepa.data import TensorEpisodes
from scjepa.eval import (
    fit_state_probe,
    nonlinear_mcc,
    oe_worst_step_nrmse,
    state_probe_metrics,
    structural_hamming_distance,
    summarize_oe,
    training_coordinate_std,
)
from scjepa.training import TrainConfig
from scjepa.training.factory import ModelConfig, build_model


def test_tensor_episode_roundtrip_and_action_indexing(tmp_path: Path) -> None:
    observations = torch.randn(3, 6, 7)
    actions = torch.randn(3, 5, 2)
    path = tmp_path / "episodes.pt"
    torch.save({"observations": observations, "actions": actions}, path)
    data = TensorEpisodes.load(path)
    assert len(data) == 3
    torch.testing.assert_close(data[1]["observations"], observations[1])
    torch.testing.assert_close(data[1]["actions"], actions[1])
    assert data[1]["episode_ids"].item() == 1
    with pytest.raises(ValueError, match="actions"):
        TensorEpisodes(observations, actions[:, :-1])


def test_config_builds_image_and_vector_with_independent_dimensions() -> None:
    for kind in ("image", "vector"):
        cfg = OmegaConf.structured(ModelConfig)
        cfg.observation_kind = kind
        cfg.state_dim, cfg.param_dim, cfg.action_dim = 5, 3, 2
        cfg.context_len, cfg.history_len = 3, 2
        cfg.spartan_dense = True
        model = build_model(cfg)
        observations = torch.randn(2, 5, 8) if kind == "vector" else torch.randn(2, 5, 3, 12, 12)
        output = model(observations, torch.randn(2, 4, 2))
        assert output.predictions.shape == (2, 2, 5)
        assert output.parameters.shape == (2, 3)


def test_checked_in_config_has_no_unknown_model_or_training_fields() -> None:
    cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/config.yaml")
    OmegaConf.merge(OmegaConf.structured(ModelConfig), cfg.model)
    OmegaConf.merge(OmegaConf.structured(TrainConfig), cfg.train)


def test_probes_support_nonphysical_dimensions_and_detach() -> None:
    features = torch.randn(60, 7, requires_grad=True)
    reference = features.detach() @ torch.randn(7, 5)
    probe = fit_state_probe(features[:40], reference[:40], ridge=1e-6)
    prediction = probe.predict(features[40:])
    assert prediction.shape == (20, 5)
    assert not prediction.requires_grad
    assert state_probe_metrics(prediction, reference[40:])["r2"] > 0.99
    assert features.grad is None


def test_generic_graph_and_trajectory_metrics() -> None:
    learned = torch.tensor([[[True, False, True], [False, True, False]]])
    target = learned.clone()
    target[0, 1, 2] = True
    assert structural_hamming_distance(learned, target).item() == 1
    prediction = torch.zeros(2, 4, 5)
    prediction[1, 2] = 2
    errors = oe_worst_step_nrmse(prediction, torch.zeros_like(prediction), torch.ones(5))
    torch.testing.assert_close(errors, torch.tensor([0.0, 2.0]))
    assert summarize_oe(errors, 1.0).satisfaction == 0.5
    data = TensorEpisodes(torch.randn(5, 4, 7))
    torch.testing.assert_close(
        training_coordinate_std(data), data.observations.reshape(-1, 7).std(dim=0, correction=0)
    )


def test_nonlinear_probe_retains_independent_coordinate_counts_and_rng() -> None:
    learned = torch.randn(12, 2)
    target = torch.randn(12, 3)
    before = torch.get_rng_state()
    report = nonlinear_mcc(learned, target, epochs=2, max_samples=12)
    assert report.matrix.shape == (3, 2)
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_factory_configuration_and_ema_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = OmegaConf.structured(
        ModelConfig(
            observation_dim=7,
            state_dim=5,
            param_dim=3,
            action_dim=2,
            context_len=3,
            history_len=2,
            encoder_hidden_dims=(8,),
            parameter_embed_dim=8,
            parameter_heads=2,
            spartan_layers=1,
            spartan_embed_dim=8,
            spartan_mlp_hidden=8,
            spartan_dense=True,
            ema_decay=0.75,
        )
    )
    original = build_model(config)
    with torch.no_grad():
        for parameter in original.online_encoder.parameters():
            parameter.add_(0.1)
    original.update_target()
    config_path, checkpoint_path = tmp_path / "config.yaml", tmp_path / "model.pt"
    OmegaConf.save(config, config_path)
    torch.save(original.state_dict(), checkpoint_path)
    restored_config = OmegaConf.load(config_path)
    assert isinstance(restored_config, DictConfig)
    restored = build_model(restored_config)
    restored.ema_decay, restored.context_len = 0.1, 1
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    assert restored.ema_decay == 0.75
    assert restored.context_len == 3
    assert restored.predictor.history_len == 2
    assert restored.predictor.state_dim == 5
    assert restored.predictor.param_dim == 3
    assert restored.predictor.action_dim == 2
    observations, actions = torch.randn(2, 5, 7), torch.randn(2, 4, 2)
    expected, actual = original(observations, actions), restored(observations, actions)
    torch.testing.assert_close(expected.predictions, actual.predictions)
    torch.testing.assert_close(expected.targets, actual.targets)
    assert not restored.target_encoder.training
    assert all(not parameter.requires_grad for parameter in restored.target_encoder.parameters())


def test_tensor_episode_load_preserves_explicit_episode_identity(tmp_path: Path) -> None:
    path = tmp_path / "episodes.pt"
    ids = torch.tensor([4, 4, 8], dtype=torch.int64)
    torch.save({"observations": torch.randn(3, 5, 7), "episode_ids": ids}, path)
    data = TensorEpisodes.load(path)
    torch.testing.assert_close(data.episode_ids, ids)
    assert "actions" not in data[0]
    torch.save({"observations": torch.randn(3, 5, 7), "unexpected": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="accepts"):
        TensorEpisodes.load(path)
    torch.save({"observations": [1, 2, 3]}, path)
    with pytest.raises(ValueError, match="tensor observations"):
        TensorEpisodes.load(path)
