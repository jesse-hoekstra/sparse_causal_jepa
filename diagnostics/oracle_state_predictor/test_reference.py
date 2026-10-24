"""Calibration independence and safe reuse of a completed local dense diagnostic."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.oracle_state_predictor.data import build_data
from diagnostics.oracle_state_predictor.reference import build_calibration, load_dense_reference
from diagnostics.oracle_state_predictor.run import Config, code_identity, reference_metrics, run
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.factory import build_dgp_dataset


@pytest.fixture(scope="module")
def completed_dense(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Config]:
    """Create a single one-update CPU reference shared by independent copied fixtures."""
    directory = tmp_path_factory.mktemp("dense-reference") / "dense"
    config = Config(
        output_dir=str(directory),
        steps=1,
        batch_size=2,
        train_groups=4,
        validation_groups=3,
        test_groups=3,
        views=2,
        embed_dim=8,
        mlp_hidden=8,
        threads=1,
        log_every=1,
        evaluate_every=1,
        checkpoint_every=1,
    )
    run(config)
    return directory, config


@pytest.fixture
def reference_directory(
    tmp_path: Path, completed_dense: tuple[Path, Config]
) -> tuple[Path, Config]:
    """Give each tamper test its own disposable checkpoint copies."""
    source, config = completed_dense
    directory = tmp_path / "reference"
    shutil.copytree(source, directory)
    return directory, config


def _record(directory: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((directory / "config.json").read_text()))


def _write_record(directory: Path, value: dict[str, Any]) -> None:
    (directory / "config.json").write_text(json.dumps(value))


def _checkpoint(directory: Path, name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any], torch.load(directory / name, map_location="cpu", weights_only=False)
    )


def test_calibration_uses_its_reserved_split_and_the_same_two_step_task() -> None:
    """Calibration draws must be independent of train, selection and final-test draws."""
    calibration = build_calibration(3, views=2)
    assert calibration.config.split == "evaluation_calibration"
    assert calibration.initial_states.shape == (3, 2, 1, 6)
    assert calibration.actions.shape == (3, 2, 2, 16)
    assert calibration.targets.shape == (3, 2, 2, 6)
    assert calibration.parameters.shape == (3, 6)
    assert calibration.source_fingerprint == build_dgp_dataset(calibration.config).fingerprint
    assert reference_metrics(calibration)["analytic/prediction_mse"] < 1e-12
    calibration_ids = {int(value) for value in calibration.episode_ids}
    for split in ("train", "validation", "test"):
        comparison = build_data(split, 3, views=2)
        assert calibration_ids.isdisjoint(int(value) for value in comparison.episode_ids)
        assert calibration.fingerprint != comparison.fingerprint
        assert not torch.equal(calibration.parameters, comparison.parameters)
    repeated = build_calibration(3, views=2)
    assert repeated.fingerprint == calibration.fingerprint
    torch.testing.assert_close(repeated.targets, calibration.targets, rtol=0, atol=0)


def test_dense_reference_is_validated_without_changing_source_files_or_rng(
    reference_directory: tuple[Path, Config], tmp_path: Path
) -> None:
    """The selected checkpoint and both content hashes are bound to a compatible run."""
    directory, config = reference_directory
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    rng = torch.get_rng_state().clone()
    selected, metadata = load_dense_reference(
        directory,
        replace(
            config,
            output_dir=str(tmp_path / "comparison"),
            steps=9,
            threads=2,
            log_every=9,
            evaluate_every=9,
            checkpoint_every=9,
            wandb_group="new-comparison",
        ),
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert before == {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }
    assert selected["step"] == metadata["selected_step"] == metadata["completed_step"] == 1
    assert selected["identity"] == metadata["identity"] == _record(directory)["identity"]
    assert metadata["source_fingerprint"] == code_identity()
    assert metadata["historical_source_differs"] is False
    for name in ("best", "last"):
        assert (
            metadata[f"{name}_checkpoint_sha256"]
            == hashlib.sha256((directory / f"{name}.pt").read_bytes()).hexdigest()
        )
    expected = _checkpoint(directory, "best.pt")
    for name, tensor in selected["model"].items():
        torch.testing.assert_close(tensor, expected["model"][name], rtol=0, atol=0)
    assert metadata["wandb"] == _checkpoint(directory, "last.pt")["wandb"]


@pytest.mark.parametrize("field", ["batch_size", "data_seed", "embed_dim", "lr", "logit_weight"])
def test_dense_reference_rejects_changed_scientific_settings(
    reference_directory: tuple[Path, Config], field: str
) -> None:
    """Dimensions, data, batching and objective/optimizer settings are part of the comparison."""
    directory, config = reference_directory
    value = getattr(config, field)
    requested = replace(config, **{field: value + 1})
    with pytest.raises(ValueError, match="scientific configuration"):
        load_dense_reference(directory, requested)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("identity", "identity does not match"),
        ("dataset_identities", "dataset identities"),
        ("protocol", "protocol or objective"),
        ("objective", "protocol or objective"),
        ("architecture", "architecture metadata"),
    ],
)
def test_dense_reference_rejects_inconsistent_manifest(
    reference_directory: tuple[Path, Config], field: str, message: str
) -> None:
    """A checkpoint's own declarations must agree with its scientific identity."""
    directory, config = reference_directory
    record = _record(directory)
    if field == "dataset_identities":
        record[field]["train"] = "0" * 64
    elif field == "architecture":
        record[field]["dense"] = False
    else:
        record[field] = "invalid"
    _write_record(directory, record)
    with pytest.raises(ValueError, match=message):
        load_dense_reference(directory, config)


@pytest.mark.parametrize("name", ["best.pt", "last.pt"])
@pytest.mark.parametrize("damage", ["nonfinite", "shape", "identity", "scientific"])
def test_dense_reference_validates_both_checkpoint_payloads(
    reference_directory: tuple[Path, Config], name: str, damage: str
) -> None:
    """Neither selected nor final weights may bypass model and provenance checks."""
    directory, config = reference_directory
    checkpoint = _checkpoint(directory, name)
    parameter = next(iter(checkpoint["model"]))
    if damage == "nonfinite":
        checkpoint["model"][parameter].flatten()[0] = float("nan")
    elif damage == "shape":
        checkpoint["model"][parameter] = checkpoint["model"][parameter][:-1]
    elif damage == "identity":
        checkpoint["identity"] = "0" * 64
    else:
        checkpoint["config"]["batch_size"] = 3
    torch.save(checkpoint, directory / name)
    with pytest.raises(ValueError, match="dense reference"):
        load_dense_reference(directory, config)


def test_dense_reference_rejects_incomplete_training(
    reference_directory: tuple[Path, Config],
) -> None:
    """A periodic checkpoint cannot masquerade as a completed dense reference."""
    directory, config = reference_directory
    record = _record(directory)
    record["config"]["steps"] = 2
    _write_record(directory, record)
    with pytest.raises(ValueError, match="incomplete"):
        load_dense_reference(directory, config)


def test_compatible_historical_source_uses_its_recorded_identity(
    reference_directory: tuple[Path, Config],
) -> None:
    """A declared compatible historical run keeps its old hash after strict semantic checks."""
    directory, config = reference_directory
    record = _record(directory)
    record["source_fingerprint"] = "a" * 64
    scientific = {
        key: record["config"][key]
        for key in (
            "seed",
            "data_seed",
            "batch_size",
            "train_groups",
            "validation_groups",
            "test_groups",
            "views",
            "lr",
            "logit_weight",
            "grad_clip",
            "embed_dim",
            "mlp_hidden",
        )
    }
    record["identity"] = canonical_fingerprint(
        {
            "config": scientific,
            "data": record["dataset_identities"],
            "code": record["source_fingerprint"],
        }
    )
    _write_record(directory, record)
    for name in ("best.pt", "last.pt"):
        checkpoint = _checkpoint(directory, name)
        checkpoint["identity"] = record["identity"]
        torch.save(checkpoint, directory / name)
    selected, metadata = load_dense_reference(directory, config)
    assert selected["identity"] == record["identity"]
    assert metadata["source_fingerprint"] == "a" * 64
    assert metadata["current_source_fingerprint"] == code_identity()
    assert metadata["historical_source_differs"] is True
