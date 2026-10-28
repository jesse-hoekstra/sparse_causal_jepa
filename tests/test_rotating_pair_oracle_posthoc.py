"""Tiny software endpoints exercise conditional reporting without scientific test access."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.balanced_action_oracle.data import PrepareConfig as SourcePrepareConfig
from diagnostics.balanced_action_oracle.data import prepare as prepare_source
from diagnostics.harder_oracle.analysis import ordering_evidence, screen_decision
from diagnostics.isotropic_oracle import data as ancestor_data
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from diagnostics.rotating_pair_oracle import analysis, confirmation, dgp, parameter_use, run
from diagnostics.rotating_pair_oracle.data import PrepareConfig, code_identity, prepare
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json

CONDITIONS = ("dense", "mlp", "baseline", "full")


@pytest.fixture(scope="module")
def saved_software_endpoints(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("rotating-pair-four-arm-software-fixtures")
    ancestor_data.prepare(
        ancestor_data.PrepareConfig(
            str(root / "ancestor-data"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    prepare_source(
        SourcePrepareConfig(str(root / "source-data"), str(root / "ancestor-data"), profile="smoke")
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
    assert result["candidate"] == "rotating_pair_innovation"
    assert result["action_mix"] == "paley24"
    assert result["phase_formula"] == "rotating_pair"
    assert result["phase_gain"] == dgp.PHASE_GAIN
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
    assert confirmed["schema"] == "rotating-pair-oracle-confirmation-analysis-v1"
    assert confirmed["candidate"] == result["candidate"]
    assert confirmed["action_mix"] == "paley24"
    assert confirmed["phase_formula"] == "rotating_pair"
    assert confirmed["phase_gain"] == dgp.PHASE_GAIN
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
    "change",
    [
        "namespace",
        "candidate",
        "action_mix",
        "phase_formula",
        "phase_gain",
        "tau",
        "initial_tensor",
    ],
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
            record["candidate_settings"][change] = {
                "candidate": "balanced_action_innovation",
                "action_mix": "householder24",
                "phase_formula": "linear_pair",
                "phase_gain": dgp.PHASE_GAIN + 0.01,
            }[change]
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
    paths = {Path(analysis.__file__), Path(confirmation.__file__), Path(parameter_use.__file__)}

    def hypothetical_change(path: Path) -> bytes:
        value = original_read(path)
        return value + b"\n# hypothetical reporting edit\n" if path in paths else value

    monkeypatch.setattr(Path, "read_bytes", hypothetical_change)
    assert code_identity() == before


@pytest.mark.parametrize("split", ["calibration", "validation"])
def test_validation_comparison_rejects_unpaired_calibration_or_validation_groups(
    saved_software_endpoints: Path, tmp_path: Path, split: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "full"
    payload = cast(dict[str, Any], torch.load(directory / f"{split}.pt", weights_only=True))
    payload["samples"]["group_ids"][0] += 100
    save_torch(directory / f"{split}.pt", payload)
    _rehash(directory, (f"{split}.pt",))
    _, parents = _paths(root)
    with pytest.raises(ValueError, match="paired calibration and validation states"):
        analysis.analyze(parents, tmp_path / "bad", draws=10, allow_smoke=True)


def test_report_requires_the_unchanged24_action_routing_shape(
    saved_software_endpoints: Path, tmp_path: Path
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "full"
    payload = cast(dict[str, Any], torch.load(directory / "validation.pt", weights_only=True))
    payload["samples"]["graph_action"] = payload["samples"]["graph_action"][..., :22]
    save_torch(directory / "validation.pt", payload)
    _rehash(directory, ("validation.pt",))
    _, parents = _paths(root)
    with pytest.raises(ValueError, match="Malformed local routing"):
        analysis.analyze(parents, tmp_path / "bad", draws=10, allow_smoke=True)


@pytest.mark.parametrize("artifact", ["initial.pt", "calibration.pt", "validation.pt"])
def test_validation_and_parameter_audit_require_complete_training_hashes(
    saved_software_endpoints: Path, tmp_path: Path, artifact: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    directory = root / "dense"
    completed = read_json(directory / "completion.json")
    del completed["hashes"][artifact]
    write_json(directory / "completion.json", completed)
    _, parents = _paths(root)
    with pytest.raises(ValueError, match="authenticate every required training artifact"):
        analysis.analyze(parents, tmp_path / "bad-validation", draws=10, allow_smoke=True)
    with pytest.raises(ValueError, match="authenticate every required training artifact"):
        parameter_use.audit(
            parameter_use.AuditConfig(str(directory), str(root / "data"), str(tmp_path / "bad-use"))
        )


def test_frozen_parameter_audit_reproduces_factual_predictions_and_preserves_sources(
    saved_software_endpoints: Path, tmp_path: Path
) -> None:
    source = saved_software_endpoints / "dense"
    original_hashes = {path: sha(path) for path in source.iterdir() if path.is_file()}
    rng = torch.get_rng_state().clone()
    result = parameter_use.audit(
        parameter_use.AuditConfig(
            str(source), str(saved_software_endpoints / "data"), str(tmp_path / "use")
        )
    )
    assert result["schema"] == "rotating-pair-oracle-parameter-use-v1"
    assert result["split"] == "validation"
    assert result["candidate_settings"]["phase_gain"] == dgp.PHASE_GAIN
    assert result["source_identity"] == code_identity()
    assert len(result["verified_source_artifact_hashes"]) == 11
    assert result["source_checkpoint_sha256"] == original_hashes[source / "last.pt"]
    assert all(sha(path) == digest for path, digest in original_hashes.items())
    assert torch.equal(rng, torch.get_rng_state())
    saved = cast(dict[str, Any], torch.load(tmp_path / "use/samples.pt", weights_only=True))
    original = cast(dict[str, Any], torch.load(source / "validation.pt", weights_only=True))
    torch.testing.assert_close(
        saved["factual_predictions"], original["samples"]["predictions"], rtol=0, atol=0
    )
    assert result["metrics"]["factual/normalized_prediction_change"] == 0
    assert result["metrics"]["permuted/normalized_prediction_change"] > 0
    assert result["metrics"]["zero/normalized_prediction_change"] > 0
    assert torch.equal(saved["donor_indices"].sort().values, torch.arange(2))
    assert bool((saved["donor_indices"] != torch.arange(2)).all())
    for name, digest in result["hashes"].items():
        assert sha(tmp_path / "use" / name) == digest


@pytest.mark.parametrize("change", ["phase_gain", "groups", "initial_source", "counter", "early"])
def test_parameter_audit_rejects_wrong_phase_groups_and_nonfinal_provenance_before_inference(
    saved_software_endpoints: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(saved_software_endpoints, root)
    source = root / "dense"
    record = read_json(source / "config.json")
    if change == "phase_gain":
        record["candidate_settings"]["phase_gain"] += 0.01
        write_json(source / "config.json", record)
        _rehash(source, ("config.json",))
        reason = "fixed rotating-pair phase"
    elif change == "groups":
        saved = cast(dict[str, Any], torch.load(source / "validation.pt", weights_only=True))
        saved["samples"]["group_ids"][0] += 1
        save_torch(source / "validation.pt", saved)
        _rehash(source, ("validation.pt",))
        reason = "Saved validation groups"
    elif change in {"initial_source", "counter"}:
        checkpoint = cast(dict[str, Any], torch.load(source / "last.pt", weights_only=False))
        if change == "initial_source":
            checkpoint["initial_checkpoint_sha256"] = "different-initial-source"
            reason = "Completed checkpoint"
        else:
            first = next(iter(checkpoint["optimizer"]["state"].values()))
            first["step"] -= 1
            reason = "Adam counters"
        save_torch(source / "last.pt", checkpoint)
        _rehash(source, ("last.pt",))
    else:
        record["config"].update(profile="exploratory", steps=30000, batch_size=16)
        write_json(source / "config.json", record)
        _rehash(source, ("config.json",))
        reason = "fixed 100000"

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Invalid provenance must fail before building a model")

    monkeypatch.setattr(parameter_use, "build_control", forbidden)
    with pytest.raises(ValueError, match=reason):
        parameter_use.audit(
            parameter_use.AuditConfig(str(source), str(root / "data"), str(tmp_path / "bad"))
        )
    assert not (tmp_path / "bad").exists()


def test_positive_ordering_does_not_relabel_strict_targets_or_benchmark_selection() -> None:
    metrics = {
        name: {
            "state_mcc": mcc,
            "online/variance_min": 1.0,
            "target/variance_min": 1.0,
            "online_state_probe/r2": 0.99,
            "normalized_prediction_mse": 0.04,
        }
        for name, mcc in (("full", 0.97), ("baseline", 0.94), ("dense", 0.90), ("mlp", 0.65))
    }
    comparisons = [
        {
            "left": left,
            "right": right,
            "split": "validation",
            "difference": metrics[left]["state_mcc"] - metrics[right]["state_mcc"],
            "lower_95": 0.02,
            "upper_95": 0.05,
        }
        for left, right in (("full", "baseline"), ("baseline", "dense"))
    ]
    strict = analysis.screen_decision(metrics)
    adaptive = analysis.ordering_evidence(metrics, comparisons, split="validation")
    assert not strict["screen_passed"]
    assert adaptive["confirmation_eligible"]
    assert adaptive["ordering_supported"]
    assert strict["criteria"]["full_baseline_gap_minimum"] == 0.05
    assert strict["criteria"]["baseline_dense_gap_minimum"] == 0.03
    comparisons[0]["lower_95"] = -0.001
    assert not analysis.ordering_evidence(metrics, comparisons, split="validation")[
        "confirmation_eligible"
    ]
