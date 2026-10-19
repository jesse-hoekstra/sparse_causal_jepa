"""Versioned, checksummed bundles with separate training tensors and scientific truth."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import cast

import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.data.datasets import GeneratedEpisodes, canonical_fingerprint, tensor_fingerprint
from scjepa.data.factory import build_dgp_dataset
from scjepa.data.random import episode_id as addressed_episode_id

BUNDLE_VERSION = 1


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_tensors(path: Path, mapping: dict[str, Tensor]) -> str:
    torch.save({key: value.detach().cpu().contiguous() for key, value in mapping.items()}, path)
    return _checksum(path)


def _tensor_mapping(path: Path) -> dict[str, Tensor]:
    value: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"Expected tensor mapping in {path.name}")
    mapping = cast(dict[object, object], value)
    if not mapping or any(
        not isinstance(key, str) or not isinstance(item, Tensor) for key, item in mapping.items()
    ):
        raise ValueError(f"Expected nonempty string-to-tensor mapping in {path.name}")
    return cast(dict[str, Tensor], mapping)


def save_dataset_bundle(dataset: GeneratedEpisodes, destination: str | Path) -> Path:
    """Stream episodes into a new bundle, publishing its manifest only on success.

    Existing destinations are rejected. Generation and serialization retain only
    one episode at a time, including for native-resolution floating-point video.
    """
    _ = dataset.fingerprint  # Validate the definition before generating any artifacts.
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (staging / "training").mkdir()
        (staging / "ground_truth").mkdir()
        checksums: dict[str, str] = {}
        for index in range(len(dataset)):
            episode = dataset.process.generate(index)
            train_name = f"training/{index:08d}.pt"
            truth_name = f"ground_truth/{index:08d}.pt"
            checksums[train_name] = _save_tensors(
                staging / train_name,
                {
                    "observations": episode.observations,
                    "actions": episode.actions,
                    "episode_ids": torch.tensor(episode.episode_id, dtype=torch.int64),
                },
            )
            checksums[truth_name] = _save_tensors(staging / truth_name, episode.ground_truth)
        observation_matrix = dataset.observation_matrix
        if observation_matrix is not None:
            checksums["observation_map.pt"] = _save_tensors(
                staging / "observation_map.pt", {"matrix": observation_matrix}
            )
        manifest: dict[str, object] = {
            "bundle_version": BUNDLE_VERSION,
            "fingerprint": dataset.fingerprint,
            "descriptor": dataset.descriptor,
            "checksums": checksums,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination / "manifest.json"


class BundleEpisodes(Dataset[dict[str, Tensor]]):
    """Read training shards lazily; evaluation sidecars require explicit access.

    Loading verifies every file checksum with bounded memory and checks that the
    descriptor matches the registered generator. Reads recheck their own shard,
    detecting changes after loading. Checksums provide integrity, not signatures
    or authentication of a manifest supplied by an untrusted party.
    """

    def __init__(self, path: str | Path) -> None:
        """Validate provenance and checksums before admitting training data."""
        path = Path(path)
        self.directory = path if path.is_dir() else path.parent
        manifest_path = self.directory / "manifest.json" if path.is_dir() else path
        loaded: object = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("dataset manifest must be an object")
        manifest = cast(dict[str, object], loaded)
        if set(manifest) != {"bundle_version", "fingerprint", "descriptor", "checksums"}:
            raise ValueError("dataset manifest has unexpected or missing fields")
        if manifest["bundle_version"] != BUNDLE_VERSION:
            raise ValueError("unsupported dataset bundle version")
        if not isinstance(manifest["descriptor"], dict):
            raise ValueError("dataset descriptor must be an object")
        descriptor = cast(dict[str, object], manifest["descriptor"])
        fingerprint = canonical_fingerprint(descriptor)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("dataset fingerprint does not match its descriptor")
        raw_config = descriptor.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("dataset descriptor requires resolved configuration")
        config = OmegaConf.create(cast(dict[str, object], raw_config))
        raw_checksums = manifest["checksums"]
        if not isinstance(raw_checksums, dict):
            raise ValueError("dataset checksums must be an object")
        entries = cast(dict[object, object], raw_checksums)
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in entries.items()):
            raise ValueError("dataset checksum entries must be strings")
        self._checksums = cast(dict[str, str], entries)
        self._observation_matrix: Tensor | None = None
        if descriptor.get("observation_matrix_fingerprint") is not None:
            matrix_mapping = _tensor_mapping(self._verified_path("observation_map.pt"))
            if set(matrix_mapping) != {"matrix"}:
                raise ValueError("observation map artifact must contain exactly the fixed matrix")
            matrix = matrix_mapping["matrix"]
            if tensor_fingerprint(matrix) != descriptor["observation_matrix_fingerprint"]:
                raise ValueError("persisted observation matrix does not match its fingerprint")
            self._observation_matrix = matrix
        # The persisted realization is authoritative: QR can differ between
        # numerical libraries/platforms even with an identical observation seed.
        expected = build_dgp_dataset(config, observation_matrix=self._observation_matrix)
        if expected.fingerprint != fingerprint:
            raise ValueError("dataset provenance does not match the registered generator")
        expected_files = {
            f"{kind}/{index:08d}.pt"
            for index in range(len(expected))
            for kind in ("training", "ground_truth")
        }
        if expected.observation_matrix is not None:
            expected_files.add("observation_map.pt")
        if set(self._checksums) != expected_files:
            raise ValueError("dataset checksums do not describe exactly the expected shards")
        for filename in sorted(self._checksums):
            self._verified_path(filename)
        self.config = expected.config
        self.descriptor = descriptor
        self.specification = expected.specification
        self.fingerprint = fingerprint
        self._content_fingerprint = canonical_fingerprint(
            {"generator_fingerprint": fingerprint, "checksums": self._checksums}
        )
        self.graphs = expected.graphs
        self.observation_shape = expected.observation_shape
        self.action_dim = expected.action_dim
        self.history_len = expected.history_len
        self.minimum_prediction_start = expected.minimum_prediction_start
        self.trajectory_length = expected.trajectory_length

    @property
    def observation_matrix(self) -> Tensor | None:
        """Return a copy of the persisted observation matrix without mutating provenance."""
        return None if self._observation_matrix is None else self._observation_matrix.clone()

    @property
    def content_fingerprint(self) -> str:
        """Revalidate stored bytes before declaring the checkpoint's content identity."""
        for filename in sorted(self._checksums):
            self._verified_path(filename)
        return self._content_fingerprint

    def _verified_path(self, filename: str) -> Path:
        path = self.directory / filename
        if not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("dataset shard path escapes the bundle directory")
        if not path.is_file() or _checksum(path) != self._checksums.get(filename):
            raise ValueError(f"dataset checksum mismatch or missing shard: {filename}")
        return path

    def __len__(self) -> int:
        """Return the manifest's validated number of episodes."""
        return self.config.num_episodes

    def _index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("episode index outside dataset")
        return index

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Read one strictly training-only shard with preserved action alignment."""
        index = self._index(index)
        filename = f"training/{index:08d}.pt"
        mapping = _tensor_mapping(self._verified_path(filename))
        if set(mapping) != {"observations", "actions", "episode_ids"}:
            raise ValueError("training shards accept observations, actions, and episode_ids only")
        observations, actions, episode_id = (
            mapping["observations"],
            mapping["actions"],
            mapping["episode_ids"],
        )
        dtype = torch.float32 if self.config.dtype == "float32" else torch.float64
        pair_shape = (2,) if self.config.episodes_per_parameter == 2 else ()
        if (
            observations.shape != (*pair_shape, self.trajectory_length, *self.observation_shape)
            or actions.shape != (*pair_shape, self.trajectory_length - 1, self.action_dim)
            or observations.dtype != dtype
            or actions.dtype != dtype
            or episode_id.ndim != 0
            or episode_id.dtype != torch.int64
        ):
            raise ValueError("training shard tensors do not match declared shapes and dtype")
        if int(episode_id) != addressed_episode_id(self.config.root_seed, self.config.split, index):
            raise ValueError("training shard episode ID does not match declared stream address")
        return mapping

    def ground_truth(self, index: int) -> dict[str, Tensor]:
        """Read a scientific sidecar only when explicitly requested for evaluation."""
        filename = f"ground_truth/{self._index(index):08d}.pt"
        return _tensor_mapping(self._verified_path(filename))


def load_dataset_bundle(path: str | Path) -> BundleEpisodes:
    """Load a bundle directory or its manifest without exposing labels to training."""
    return BundleEpisodes(path)
