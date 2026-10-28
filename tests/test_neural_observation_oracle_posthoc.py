"""Tiny archived fixtures check observation provenance and frozen endpoint reporting."""

import copy
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.neural_observation_oracle import analysis, confirmation, data, parameter_use, run
from diagnostics.neural_observation_oracle.map import settings as map_settings
from diagnostics.oracle_state_learning import data as original_data
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.data.random import episode_id
from scjepa.experiments.storage import write_json

CONDITIONS = ("dense", "mlp", "baseline", "full")


@pytest.fixture(scope="module")
def archived(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("neural-observation-software-only")
    original_data.prepare(
        original_data.PrepareConfig(
            str(root / "source"),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    base = root / "base.pt"
    save_torch(base, {"matrix": VectorObservationMap(seed=1729, dimension=6).matrix})
    data.prepare(
        data.PrepareConfig(str(root / "data"), str(root / "source"), str(base), profile="smoke")
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


def _rehash(directory: Path, *names: str) -> None:
    completion = read_json(directory / "completion.json")
    completion["hashes"].update({name: sha(directory / name) for name in names})
    write_json(directory / "completion.json", completion)


def test_four_arm_reports_keep_criteria_and_do_not_invoke_models(
    archived: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parents, tests = (
        [archived / name for name in CONDITIONS],
        [archived / f"{name}-test" for name in CONDITIONS],
    )
    before = {
        path: sha(path) for root in parents + tests for path in root.iterdir() if path.is_file()
    }
    rng = torch.get_rng_state().clone()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Reporting must use archived samples without model inference")

    monkeypatch.setattr(run, "train", forbidden)
    monkeypatch.setattr(run, "evaluate", forbidden)
    monkeypatch.setattr(run, "score_test", forbidden)
    result = analysis.analyze(parents, tmp_path / "validation", draws=10, allow_smoke=True)
    assert result["candidate"] == data.CANDIDATE
    assert result["observation_map"] == map_settings()
    assert result["native_spartan_layers"] == 2
    assert result["criteria"]["full_baseline_gap_minimum"] == 0.05
    assert result["criteria"]["baseline_dense_gap_minimum"] == 0.03
    assert len(result["comparisons"]) == 4
    for arm in result["arms"]:
        if arm["condition"] != "mlp":
            assert arm["metrics"]["validation"]["routing/action_observed_union_edges"] <= 96
    confirmed = confirmation.confirm(
        tests, list(reversed(parents)), tmp_path / "confirmed", draws=10, allow_smoke=True
    )
    assert confirmed["schema"] == "neural-observation-oracle-confirmation-analysis-v1"
    assert confirmed["test_root_seed"] == 20261028
    assert confirmed["criteria"] == result["criteria"]
    assert not confirmed["confirmation_passed"]
    assert not confirmed["confirmed_requested_ordering"]
    assert all(sha(path) == digest for path, digest in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    with pytest.raises(ValueError, match="Scientific analysis"):
        analysis.analyze(parents, tmp_path / "forbidden")
    with pytest.raises(ValueError, match="four unique"):
        analysis.analyze(parents[:2], tmp_path / "two-arm", allow_smoke=True)


@pytest.mark.parametrize("condition", ["dense", "mlp", "full"])
def test_frozen_factual_replay_preserves_checkpoint_rng_and_precision(
    archived: Path, tmp_path: Path, condition: str
) -> None:
    source = archived / condition
    before = {path: sha(path) for path in source.iterdir() if path.is_file()}
    rng = torch.get_rng_state().clone()
    previous = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
    )
    result = parameter_use.audit(
        parameter_use.AuditConfig(str(source), str(archived / "data"), str(tmp_path / "use"))
    )
    assert result["schema"] == "neural-observation-oracle-parameter-use-v1"
    assert result["candidate_settings"]["observation_map"] == map_settings()
    assert result["architecture_settings"]["native_spartan_layers"] == 2
    assert len(result["verified_source_artifact_hashes"]) == 11
    saved = cast(dict[str, Any], torch.load(tmp_path / "use/samples.pt", weights_only=True))
    factual = cast(dict[str, Any], torch.load(source / "validation.pt", weights_only=True))
    torch.testing.assert_close(
        saved["factual_predictions"], factual["samples"]["predictions"], rtol=0, atol=0
    )
    assert result["metrics"]["factual/normalized_prediction_change"] == 0
    assert result["metrics"]["permuted/normalized_prediction_change"] > 0
    assert torch.equal(saved["donor_indices"].sort().values, torch.arange(2))
    assert bool((saved["donor_indices"] != torch.arange(2)).all())
    assert all(sha(path) == digest for path, digest in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    assert previous == (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_num_threads(),
    )


def test_every_train_and_test_artifact_requires_its_hash_entry(
    archived: Path, tmp_path: Path
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(archived, root)
    for name, kind in (("dense", "train"), ("dense-test", "test")):
        directory = root / name
        original = read_json(directory / "completion.json")
        assert len(original["hashes"]) == (11 if kind == "train" else 5)
        for artifact in original["hashes"]:
            modified = copy.deepcopy(original)
            del modified["hashes"][artifact]
            write_json(directory / "completion.json", modified)
            with pytest.raises(ValueError, match="required stage artifact"):
                run.verify_completion(directory, kind=kind)
        write_json(directory / "completion.json", original)


@pytest.mark.parametrize("change", ["map", "test_root", "architecture"])
def test_wrong_map_or_architecture_fails_before_model_construction(
    archived: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(archived, root)
    source = root / "dense"
    record = read_json(source / "config.json")
    if change == "map":
        record["candidate_settings"]["observation_map"]["alpha"] = 0.3
        reason = "fixed neural observation map"
    elif change == "test_root":
        record["candidate_settings"]["test_root_seed"] = 0
        reason = "fixed neural observation map"
    else:
        record["scientific_config"]["native_spartan_layers"] = 3
        reason = "original two-layer"
    write_json(source / "config.json", record)
    _rehash(source, "config.json")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Invalid provenance must fail before model inference")

    monkeypatch.setattr(parameter_use, "build_control", forbidden)
    with pytest.raises(ValueError, match=reason):
        parameter_use.audit(
            parameter_use.AuditConfig(str(source), str(root / "data"), str(tmp_path / "bad"))
        )
    with pytest.raises(ValueError, match=reason):
        analysis._load(source, allow_smoke=True)  # pyright: ignore[reportPrivateUsage]


def test_relabelled_wrong_action_tensor_and_stale_test_root_are_rejected(
    archived: Path, tmp_path: Path
) -> None:
    root = tmp_path / "copy"
    shutil.copytree(archived, root)
    source = root / "dense"
    checkpoint = cast(dict[str, Any], torch.load(source / "last.pt", weights_only=False))
    checkpoint["model"]["predictor.action_identity"] = torch.zeros(24, 64)
    save_torch(source / "last.pt", checkpoint)
    _rehash(source, "last.pt")
    with pytest.raises(ValueError, match="depth/width/count"):
        analysis._load(source, allow_smoke=True)  # pyright: ignore[reportPrivateUsage]
    test = root / "mlp-test"
    payload = cast(dict[str, Any], torch.load(test / "test.pt", weights_only=True))
    payload["samples"]["group_ids"] = torch.tensor([episode_id(0, "test", i) for i in range(2)])
    save_torch(test / "test.pt", payload)
    _rehash(test, "test.pt")
    with pytest.raises(ValueError, match="declared fresh test root"):
        confirmation._load_test(test, root / "mlp", allow_smoke=True)  # pyright: ignore[reportPrivateUsage]


def test_readonly_reports_are_excluded_from_frozen_numerical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = data.code_identity()
    original_read = Path.read_bytes
    paths = {Path(analysis.__file__), Path(confirmation.__file__), Path(parameter_use.__file__)}

    def changed(path: Path) -> bytes:
        value = original_read(path)
        return value + b"\n# reporting change\n" if path in paths else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert data.code_identity() == before
