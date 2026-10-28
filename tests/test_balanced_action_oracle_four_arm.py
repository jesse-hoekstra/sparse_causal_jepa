"""Tiny software endpoints exercise conditional reporting without scientific test access."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.balanced_action_oracle import analysis, confirmation, run
from diagnostics.balanced_action_oracle.data import PrepareConfig, code_identity, prepare
from diagnostics.harder_oracle.analysis import ordering_evidence, screen_decision
from diagnostics.isotropic_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.isotropic_oracle.data import prepare as prepare_source
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json

CONDITIONS = ("dense", "mlp", "baseline", "full")


@pytest.fixture(scope="module")
def saved_software_endpoints(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("balanced-four-arm-software-fixtures")
    prepare_source(
        SourcePrepareConfig(
            str(root / "source-data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    prepare(PrepareConfig(str(root / "data"), str(root / "source-data"), profile="smoke"))
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
    completed = read_json(directory / "completion.json")
    completed["hashes"].update({name: sha(directory / name) for name in names})
    write_json(directory / "completion.json", completed)


def test_four_arm_reports_keep_original_criteria_and_do_not_invoke_inference(
    saved_software_endpoints: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests, parents = _paths(saved_software_endpoints)
    before = {
        path: sha(path) for root in parents + tests for path in root.iterdir() if path.is_file()
    }
    source_identity = code_identity()
    rng = torch.get_rng_state().clone()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Read-only reporting must never train or run model inference")

    monkeypatch.setattr(run, "train", forbidden)
    monkeypatch.setattr(run, "evaluate", forbidden)
    monkeypatch.setattr(run, "score_test", forbidden)
    assert analysis.screen_decision is screen_decision
    assert analysis.ordering_evidence is ordering_evidence
    result = analysis.analyze(parents, tmp_path / "validation", draws=10, allow_smoke=True)
    assert result["candidate"] == "balanced_action_innovation"
    assert result["action_mix"] == "paley24"
    assert result["criteria"]["full_baseline_gap_minimum"] == 0.05
    assert result["criteria"]["baseline_dense_gap_minimum"] == 0.03
    assert len(result["comparisons"]) == 4
    for arm in result["arms"]:
        assert arm["action_mix"] == "paley24"
        if arm["condition"] != "mlp":
            assert arm["metrics"]["validation"]["routing/action_observed_union_edges"] <= 144
    confirmed = confirmation.confirm(
        tests, list(reversed(parents)), tmp_path / "confirmed", draws=10, allow_smoke=True
    )
    assert confirmed["schema"] == "balanced-action-oracle-confirmation-analysis-v1"
    assert confirmed["candidate"] == result["candidate"]
    assert confirmed["action_mix"] == "paley24"
    assert confirmed["source_identity"] == source_identity
    assert confirmed["criteria"] == result["criteria"]
    assert not confirmed["confirmation_passed"]
    assert not confirmed["confirmed_requested_ordering"]
    assert all(item["split"] == "test" for item in confirmed["comparisons"])
    assert all(sha(path) == digest for path, digest in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert code_identity() == source_identity


def test_software_endpoints_are_not_scientific_confirmation(
    saved_software_endpoints: Path, tmp_path: Path
) -> None:
    tests, parents = _paths(saved_software_endpoints)
    with pytest.raises(ValueError, match="Scientific analysis"):
        confirmation.confirm(tests, parents, tmp_path / "forbidden")


@pytest.mark.parametrize(
    "change", ["namespace", "candidate", "action_mix", "tau", "initial_tensor"]
)
def test_wrong_candidate_calibration_or_initialization_is_rejected(
    saved_software_endpoints: Path, tmp_path: Path, change: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "full"
    if change == "namespace":
        completed = read_json(directory / "completion.json")
        completed["schema"] = "isotropic-oracle-state-learning-v1"
        write_json(directory / "completion.json", completed)
        reason = "schema differs"
    elif change == "initial_tensor":
        initial = cast(dict[str, Any], torch.load(directory / "initial.pt", weights_only=True))
        name = next(key for key in initial["model"] if key.startswith("online_encoder."))
        initial["model"][name] += 0.1
        save_torch(directory / "initial.pt", initial)
        last = cast(dict[str, Any], torch.load(directory / "last.pt", weights_only=False))
        last["initial_checkpoint_sha256"] = sha(directory / "initial.pt")
        save_torch(directory / "last.pt", last)
        _rehash(directory, ("initial.pt", "last.pt"))
        reason = "Initial tensor identity"
    else:
        record = read_json(directory / "config.json")
        if change == "tau":
            record["geco"]["tau"] += 0.1
            reason = "Sparse bound differs"
        else:
            record["candidate_settings"][change] = (
                "isotropic_innovation" if change == "candidate" else "householder24"
            )
            reason = "paley24 actions"
        write_json(directory / "config.json", record)
        _rehash(directory, ("config.json",))
    _, parents = _paths(root)
    with pytest.raises(ValueError, match=reason):
        analysis.analyze(parents, tmp_path / "bad", allow_smoke=True)


@pytest.mark.parametrize("change", ["parent_hash", "matching", "group_ids"])
def test_confirmation_rejects_changed_parent_assignment_or_unpaired_test_groups(
    saved_software_endpoints: Path, tmp_path: Path, change: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "full-test"
    if change == "parent_hash":
        evidence = read_json(directory / "config.json")
        evidence["parent_checkpoint_sha256"] = "wrong-parent"
        evidence["identity"] = canonical_fingerprint(
            {key: value for key, value in evidence.items() if key != "identity"}
        )
        write_json(directory / "config.json", evidence)
        complete = read_json(directory / "completion.json")
        complete.update(evidence)
        write_json(directory / "completion.json", complete)
        _rehash(directory, ("config.json",))
        reason = "parent completion or checkpoint hash"
    elif change == "matching":
        matching = read_json(directory / "matching.json")
        matching["online_columns"] = list(reversed(matching["online_columns"]))
        write_json(directory / "matching.json", matching)
        _rehash(directory, ("matching.json",))
        reason = "frozen calibration assignment"
    else:
        payload = cast(dict[str, Any], torch.load(directory / "test.pt", weights_only=True))
        payload["samples"]["group_ids"][0] += 100
        save_torch(directory / "test.pt", payload)
        _rehash(directory, ("test.pt",))
        reason = "identical addressed parameter groups"
    tests, parents = _paths(root)
    with pytest.raises(ValueError, match=reason):
        confirmation.confirm(tests, parents, tmp_path / "bad", draws=10, allow_smoke=True)


@pytest.mark.parametrize(
    "artifact", ["config.json", "test.pt", "test.json", "matching.json", "wandb.json"]
)
def test_confirmation_rejects_unauthenticated_test_artifacts(
    saved_software_endpoints: Path, tmp_path: Path, artifact: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "full-test"
    completed = read_json(directory / "completion.json")
    del completed["hashes"][artifact]
    write_json(directory / "completion.json", completed)
    tests, parents = _paths(root)
    with pytest.raises(ValueError, match="authenticate every required test artifact"):
        confirmation.confirm(tests, parents, tmp_path / "bad", draws=10, allow_smoke=True)


def test_new_four_arm_reports_are_excluded_from_numeric_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = code_identity()
    original_read = Path.read_bytes
    paths = {Path(analysis.__file__), Path(confirmation.__file__)}

    def hypothetical_change(path: Path) -> bytes:
        value = original_read(path)
        return value + b"\n# hypothetical reporting edit\n" if path in paths else value

    monkeypatch.setattr(Path, "read_bytes", hypothetical_change)
    assert code_identity() == before
