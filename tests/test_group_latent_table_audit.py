import copy
from collections.abc import Generator
from dataclasses import replace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from test_multiquery_control import make_model
from torch import Tensor, nn

from diagnostics.group_latent_table_audit import donor_zero_codes, reconstruct_training_truth
from diagnostics.multiquery_control.data import build_cache
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.processes import SimplifiedGaussianDGP
from scjepa.data.types import ScientificEpisode
from scjepa.models.model import SCJEPA


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture_data() -> tuple[SCJEPA, dict[str, Tensor], dict[str, Any], Tensor, str]:
    model = make_model().float()
    observation = VectorObservationMap(matrix=torch.eye(6, dtype=torch.float64), dimension=6)
    config = {
        "family": "simplified_gaussian",
        "root_seed": 19,
        "split": "train",
        "num_episodes": 8,
        "episodes_per_parameter": 10,
        "trajectory_length": 12,
        "dtype": "float32",
    }
    train, evaluation = build_cache(
        model,
        observation,
        config,
        train_groups=8,
        calibration_groups=2,
        validation_groups=2,
        test_groups=2,
        batch_size=2,
    )
    return (
        model,
        train,
        config,
        observation.matrix,
        evaluation["metadata"]["training_dataset_fingerprint"],
    )


def test_training_labels_bind_only_to_exact_verified_cache() -> None:
    model, train, config, matrix, fingerprint = fixture_data()
    before = copy.deepcopy(train)
    theta, receipt = reconstruct_training_truth(
        model, train, config, matrix, fingerprint, batch_size=2
    )
    dataset = build_dgp_dataset(OmegaConf.create(config), observation_matrix=matrix)
    expected = torch.stack([dataset.ground_truth(index)["parameters"] for index in range(8)])
    torch.testing.assert_close(theta, expected, rtol=0, atol=0)
    assert receipt["groups_verified"] == 8
    assert receipt["all_actions_equal"]
    torch.testing.assert_close(train, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="fingerprint"):
        reconstruct_training_truth(model, train, {**config, "root_seed": 20}, matrix, fingerprint)
    changed_actions = {**train, "actions": train["actions"].clone()}
    changed_actions["actions"][-1, -1, -1, -1] += 0.001
    with pytest.raises(ValueError, match="actions differ"):
        reconstruct_training_truth(
            model, changed_actions, config, matrix, fingerprint, batch_size=2
        )
    changed_states = {**train, "target": train["target"].clone()}
    changed_states["target"][-1, -1, -1, -1] += 0.001
    with pytest.raises(ValueError, match="states differ"):
        reconstruct_training_truth(model, changed_states, config, matrix, fingerprint, batch_size=2)
    with pytest.raises(ValueError, match="label-free"):
        reconstruct_training_truth(model, {**train, "theta": theta}, config, matrix, fingerprint)


def test_encoder_audit_uses_individual_context_and_never_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, train, _, _, _ = fixture_data()
    model.eval()

    def forbidden_table(_self: nn.Embedding, _indices: Tensor) -> Tensor:
        raise AssertionError("Train group table accessed by context encoder audit")

    monkeypatch.setattr(nn.Embedding, "forward", forbidden_table)
    actual = donor_zero_codes(model, train, batch_size=2)
    with torch.no_grad():
        expected = torch.cat(
            [
                model.parameter_encoder(
                    train["online"][index : index + 1, 0, :10],
                    train["actions"][index : index + 1, 0, :9],
                )
                for index in range(8)
            ]
        )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    altered = {**train, "online": train["online"].clone()}
    altered["online"][:, 1:] += 100
    torch.testing.assert_close(
        donor_zero_codes(model, altered, batch_size=2), actual, rtol=0, atol=0
    )


def test_reconstructed_truth_must_reproduce_its_own_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, train, config, matrix, fingerprint = fixture_data()
    original = SimplifiedGaussianDGP.generate

    def mismatched_labels(process: SimplifiedGaussianDGP, index: int) -> ScientificEpisode:
        episode = original(process, index)
        truth = {**episode.ground_truth, "parameters": episode.ground_truth["parameters"] + 0.001}
        return replace(episode, ground_truth=truth)

    monkeypatch.setattr(SimplifiedGaussianDGP, "generate", mismatched_labels)
    with pytest.raises(ValueError, match="parameters do not reproduce"):
        reconstruct_training_truth(model, train, config, matrix, fingerprint, batch_size=2)
