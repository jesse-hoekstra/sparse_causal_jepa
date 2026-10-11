"""Deterministic scientific episodes, strict training interfaces, and aligned windows."""

from dataclasses import dataclass, replace
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError
from torch import Tensor
from torch.utils.data import DataLoader

from scjepa.data.datasets import EpisodeWindows
from scjepa.data.factory import build_dgp_dataset, register_dgp, resolve_dgp_config
from scjepa.data.identity import dataset_fingerprint
from scjepa.data.random import SPLITS, episode_id, stream_seed
from scjepa.data.types import DGP, DGPConfig, RendererConfig


@pytest.mark.parametrize("family", ["controlled_vector", "coupled_springs"])
def test_lazy_training_contract_and_repeatable_retrieval(family: str) -> None:
    data = build_dgp_dataset(DGPConfig(family=family, num_episodes=3, trajectory_length=4))
    expected_shape = (4, 10) if family == "controlled_vector" else (4, 3, 320, 64)
    expected_da = 10 if family == "controlled_vector" else 5
    first = data[0]
    assert set(first) == {"observations", "actions", "episode_ids"}
    assert first["observations"].shape == expected_shape
    assert first["actions"].shape == (3, expected_da)
    assert first["observations"].dtype == torch.float32
    assert first["episode_ids"].dtype == torch.int64
    assert first["episode_ids"].ndim == 0
    data[2]
    torch.testing.assert_close(data[0], first, atol=0, rtol=0)
    torch.testing.assert_close(data[-1], data[2], atol=0, rtol=0)
    assert data.ground_truth(0)["states"].dtype == torch.float64
    assert data.ground_truth(0)["parameters"].shape == (10,)
    with pytest.raises(IndexError):
        data[len(data)]


def test_graph_variants_share_exogenous_draws_and_realized_map() -> None:
    config = DGPConfig(root_seed=37, observation_seed=53, num_episodes=1, trajectory_length=6)
    datasets = [
        build_dgp_dataset(replace(config, parameter_graph=variant))
        for variant in ("dense", "complementary", "diagonal")
    ]
    truth = [dataset.ground_truth(0) for dataset in datasets]
    reference = truth[0]
    for dataset, labels in zip(datasets, truth, strict=True):
        for key in ("initial_state", "parameters", "actions"):
            torch.testing.assert_close(labels[key], reference[key], atol=0, rtol=0)
        torch.testing.assert_close(dataset.observation_matrix, datasets[0].observation_matrix)
        assert dataset[0]["episode_ids"].item() == datasets[0][0]["episode_ids"].item()
    assert not torch.equal(truth[0]["states"], truth[1]["states"])
    assert len({dataset.fingerprint for dataset in datasets}) == 3


def test_split_streams_and_ids_are_disjoint_but_observation_map_is_shared() -> None:
    datasets = [
        build_dgp_dataset(DGPConfig(split=split, num_episodes=2, trajectory_length=3))
        for split in SPLITS
    ]
    ids = [int(dataset[i]["episode_ids"]) for dataset in datasets for i in range(2)]
    assert len(set(ids)) == 8
    assert len({dataset.fingerprint for dataset in datasets}) == 4
    for dataset in datasets[1:]:
        assert not torch.equal(dataset[0]["observations"], datasets[0][0]["observations"])
        torch.testing.assert_close(dataset.observation_matrix, datasets[0].observation_matrix)
    assert episode_id(2**31 - 1, "test", 2**30 - 1) == 2**63 - 1
    assert len({stream_seed(7, "train", 0, s) for s in ("initial", "parameters", "actions")}) == 3


def test_data_workers_access_order_and_global_rng_do_not_change_episodes() -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=4, trajectory_length=4))
    rng_before = torch.get_rng_state()
    direct = {int(dataset[i]["episode_ids"]): dataset[i] for i in (3, 1, 0, 2)}
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    for workers, batch_size in ((0, 1), (2, 2)):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=torch.Generator().manual_seed(19),
        )
        observed_ids: set[int] = set()
        for raw_batch in loader:
            batch = cast(dict[str, Tensor], raw_batch)
            for position in range(batch["episode_ids"].shape[0]):
                identity = int(batch["episode_ids"][position])
                observed_ids.add(identity)
                torch.testing.assert_close(
                    {key: tensor[position] for key, tensor in batch.items()},
                    direct[identity],
                    atol=0,
                    rtol=0,
                )
        assert observed_ids == set(direct)


@pytest.mark.parametrize("family", ["controlled_vector", "coupled_springs"])
def test_windows_retain_episode_id_and_exact_source_action_alignment(family: str) -> None:
    dataset = build_dgp_dataset(DGPConfig(family=family, num_episodes=2, trajectory_length=6))
    windows = EpisodeWindows(dataset, window_length=3, stride=2)
    assert len(windows) == 4
    for i in range(len(windows)):
        episode, start, end = windows.window_bounds(i)
        window = windows[i]
        original = dataset[episode]
        assert window["episode_ids"].item() == original["episode_ids"].item()
        torch.testing.assert_close(window["observations"], original["observations"][start:end])
        torch.testing.assert_close(window["actions"], original["actions"][start : end - 1])
        torch.testing.assert_close(windows.original_ground_truth(i), dataset.ground_truth(episode))
    assert windows[0]["episode_ids"].item() == windows[1]["episode_ids"].item()
    assert windows.fingerprint != dataset.fingerprint
    assert windows.fingerprint != EpisodeWindows(dataset, 3, stride=1).fingerprint
    with pytest.raises(ValueError, match="window_length"):
        EpisodeWindows(dataset, 7)


def test_structured_config_nested_renderer_and_unknown_keys() -> None:
    config = resolve_dgp_config(
        OmegaConf.create({"family": "coupled_springs", "renderer": {"sigma": 2.5}})
    )
    assert config.parameter_graph == "diagonal"
    assert config.renderer == RendererConfig(sigma=2.5)
    with pytest.raises(ConfigKeyError):
        resolve_dgp_config(OmegaConf.create({"unrecognized_data_setting": 2}))
    with pytest.raises(ValueError, match="Unknown DGP"):
        build_dgp_dataset(DGPConfig(family="missing_family"))


def test_extensible_registry_accepts_a_typed_config_subclass() -> None:
    @dataclass(frozen=True)
    class AdditionalConfig(DGPConfig):
        scale: float = 2.0

    def constructor(config: DGPConfig) -> DGP:
        assert isinstance(config, AdditionalConfig)
        assert config.scale == 3.5
        return build_dgp_dataset(DGPConfig(num_episodes=config.num_episodes)).process

    name = "dataset_test_registered_family"
    register_dgp(name, constructor, AdditionalConfig)
    dataset = build_dgp_dataset(OmegaConf.create({"family": name, "scale": 3.5}))
    assert len(dataset) == 8
    with pytest.raises(ValueError, match="already registered"):
        register_dgp(name, constructor)


@pytest.mark.parametrize("change", ["observation_matrix", "configuration", "descriptor"])
def test_generated_definition_changes_reject_retrieval_and_checkpoint_identity(change: str) -> None:
    dataset = build_dgp_dataset(DGPConfig(num_episodes=1, trajectory_length=4))
    windows = EpisodeWindows(dataset, 3)
    if change == "observation_matrix":
        dataset.process.restore_observation_matrix(torch.eye(10, dtype=torch.float64))
    elif change == "configuration":
        dataset.process.config = replace(dataset.config, root_seed=17)
    else:
        dataset.descriptor["schema"] = "changed"
    with pytest.raises(ValueError, match="definition changed"):
        dataset[0]
    with pytest.raises(ValueError, match="definition changed"):
        dataset.ground_truth(0)
    with pytest.raises(ValueError, match="definition changed"):
        dataset_fingerprint(dataset)
    with pytest.raises(ValueError, match="definition changed"):
        dataset_fingerprint(windows)
