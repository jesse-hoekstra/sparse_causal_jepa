"""Authenticate tiny software fixtures without evaluating scientific test cohorts."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.harder_oracle import confirmation, run
from diagnostics.harder_oracle.data import PrepareConfig, prepare
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json

CONDITIONS = ("dense", "mlp", "baseline", "full")


@pytest.fixture(scope="module")
def saved_software_endpoints(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("confirmation-software-fixtures")
    prepare(
        PrepareConfig(
            str(root / "data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    for condition in CONDITIONS:
        if condition == "baseline":
            run.calibrate(
                run.CalibrateConfig(str(root / "bound"), (str(root / "dense"),), profile="smoke")
            )
        run.train(
            run.TrainConfig(
                str(root / condition),
                str(root / "data"),
                condition,
                profile="smoke",
                steps=2,
                batch_size=2,
                log_every=1,
                checkpoint_every=1,
                evaluate_every=2,
                calibration_dir=str(root / "bound") if condition in {"baseline", "full"} else None,
            )
        )
        run.score_test(
            run.TestConfig(
                str(root / f"{condition}-test"), str(root / condition), str(root / "data")
            )
        )
    try:
        yield root
    finally:
        torch.set_num_threads(previous)


def _paths(root: Path) -> tuple[list[Path], list[Path]]:
    return [root / f"{name}-test" for name in CONDITIONS], [root / name for name in CONDITIONS]


def _rehash(directory: Path, names: tuple[str, ...]) -> None:
    record = read_json(directory / "completion.json")
    record["hashes"].update({name: sha(directory / name) for name in names})
    write_json(directory / "completion.json", record)


def test_confirm_reads_saved_samples_and_never_calls_training_or_inference(
    saved_software_endpoints: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests, parents = _paths(saved_software_endpoints)
    hashes = {
        str(path): sha(path) for parent in parents for path in parent.iterdir() if path.is_file()
    }
    rng = torch.get_rng_state().clone()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Read-only confirmation must never run training or inference")

    monkeypatch.setattr(run, "train", forbidden)
    monkeypatch.setattr(run, "evaluate", forbidden)
    monkeypatch.setattr(run, "score_test", forbidden)
    result = confirmation.confirm(
        tests, list(reversed(parents)), tmp_path / "analysis", draws=10, allow_smoke=True
    )
    assert result["profile"] == "smoke"
    assert result["primary_chart"] == "mixed"
    assert not result["confirmation_passed"]
    assert not result["confirmed_requested_ordering"]
    assert "validation_confirmation_eligible" in result
    assert result["criteria"]["full_baseline_gap_minimum"] == 0.05
    assert len(result["comparisons"]) == 4
    assert all(pair["split"] == "test" for pair in result["comparisons"])
    assert "source_identity" in result
    assert all("parent_checkpoint_sha256" in arm for arm in result["arms"])
    assert all(sha(Path(path)) == value for path, value in hashes.items())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_software_fixtures_cannot_be_reported_as_scientific_confirmation(
    saved_software_endpoints: Path, tmp_path: Path
) -> None:
    tests, parents = _paths(saved_software_endpoints)
    with pytest.raises(ValueError, match="Scientific analysis"):
        confirmation.confirm(tests, parents, tmp_path / "forbidden")


def test_confirmation_rejects_parents_from_a_different_selected_chart(
    saved_software_endpoints: Path, tmp_path: Path
) -> None:
    tests, parents = _paths(saved_software_endpoints)
    with pytest.raises(ValueError, match="declared physical state transport"):
        confirmation.confirm(
            tests, parents, tmp_path / "wrong-chart", primary_chart="physical", allow_smoke=True
        )


@pytest.mark.parametrize("change", ["matching", "parent_hash", "group_ids", "mcc"])
def test_tampered_links_assignments_groups_and_metrics_are_rejected(
    saved_software_endpoints: Path, tmp_path: Path, change: str
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(saved_software_endpoints, copied)
    directory = copied / "full-test"
    if change == "matching":
        matching = read_json(directory / "matching.json")
        matching["online_columns"] = list(reversed(matching["online_columns"]))
        write_json(directory / "matching.json", matching)
        _rehash(directory, ("matching.json",))
        reason = "frozen calibration assignment"
    elif change == "parent_hash":
        evidence = read_json(directory / "config.json")
        evidence["parent_checkpoint_sha256"] = "wrong-checkpoint-hash"
        evidence["identity"] = canonical_fingerprint(
            {key: value for key, value in evidence.items() if key != "identity"}
        )
        write_json(directory / "config.json", evidence)
        record = read_json(directory / "completion.json")
        record.update(evidence)
        write_json(directory / "completion.json", record)
        _rehash(directory, ("config.json",))
        reason = "parent completion or checkpoint hash"
    else:
        payload = cast(dict[str, Any], torch.load(directory / "test.pt", weights_only=True))
        if change == "group_ids":
            payload["samples"]["group_ids"][0] += 100
            reason = "identical addressed parameter groups"
        else:
            payload["metrics"]["state_mcc"] += 0.1
            write_json(directory / "test.json", {"metrics": payload["metrics"]})
            reason = "Reported test MCC"
        save_torch(directory / "test.pt", payload)
        _rehash(directory, ("test.pt", "test.json"))
    tests, parents = _paths(copied)
    with pytest.raises(ValueError, match=reason):
        confirmation.confirm(tests, parents, tmp_path / "analysis", draws=10, allow_smoke=True)
