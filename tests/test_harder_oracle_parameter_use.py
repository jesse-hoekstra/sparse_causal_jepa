"""A frozen predictive-use audit must not retrain or change calibration artifacts."""

import math
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from diagnostics.harder_oracle.data import PrepareConfig, build_data, prepare
from diagnostics.harder_oracle.model import build_control
from diagnostics.harder_oracle.parameter_use import AuditConfig, audit, cyclic_donors, measure
from diagnostics.harder_oracle.run import TrainConfig, train, verify_completion
from diagnostics.oracle_state_learning.data import read_json, sha
from scjepa.data.controlled_vector import VectorObservationMap


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _config(path: Path) -> PrepareConfig:
    return PrepareConfig(
        str(path),
        train_groups=4,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        profile="smoke",
    )


@pytest.mark.parametrize("groups", [2, 7])
def test_cycle_pairs_every_group_once_without_self_donors_or_rng_changes(groups: int) -> None:
    rng = torch.get_rng_state().clone()
    donors = cyclic_donors(groups)
    assert torch.equal(donors.sort().values, torch.arange(groups))
    assert bool((donors != torch.arange(groups)).all())
    assert torch.equal(donors, cyclic_donors(groups))
    assert torch.equal(rng, torch.get_rng_state())
    visited = [0]
    for _ in range(groups - 1):
        visited.append(int(donors[visited[-1]]))
    assert len(set(visited)) == groups


def test_measure_replaces_only_theta_and_preserves_all_model_tensors(tmp_path: Path) -> None:
    data = build_data(_config(tmp_path), "validation", VectorObservationMap(seed=1729, dimension=6))
    model = build_control("mlp", 0)
    before = deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    inputs = {name: tensor.clone() for name, tensor in data.tensors.items()}
    rng = torch.get_rng_state().clone()
    metrics, samples = measure(model, data, batch_size=2)
    assert model.predictor.action_dim == 22
    assert data.tensors["actions"].shape[-1] == 22
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(data.tensors, inputs, rtol=0, atol=0)
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert samples["targets"].shape == (4, 10, 2, 6)
    assert metrics["factual/normalized_prediction_change"] == 0
    for name in ("permuted", "zero"):
        assert metrics[f"{name}/normalized_prediction_change"] > 0
        expected = (
            float(samples[f"{name}_group_mse"].mean() - samples["factual_group_mse"].mean())
            / metrics["target_variance"]
        )
        assert math.isclose(metrics[f"{name}/normalized_mse_increase"], expected, abs_tol=1e-14)


def test_completed_audit_reproduces_factual_predictions_without_touching_source(
    tmp_path: Path,
) -> None:
    cache, source, destination = (tmp_path / name for name in ("data", "trained", "audit"))
    prepare(_config(cache))
    trained = train(
        TrainConfig(
            str(source),
            str(cache),
            "mlp",
            profile="smoke",
            steps=2,
            batch_size=2,
            log_every=1,
            evaluate_every=2,
            checkpoint_every=1,
        )
    )
    before = {path.name: sha(path) for path in source.iterdir() if path.is_file()}
    matched = read_json(source / "matching.json")
    result = audit(AuditConfig(str(source), str(cache), str(destination)))
    assert result["parent_identity"] == trained["identity"]
    assert result["split"] == "validation"
    assert result["steps"] == 2
    assert verify_completion(source, kind="train") == trained
    assert read_json(source / "matching.json") == matched
    assert before == {path.name: sha(path) for path in source.iterdir() if path.is_file()}
    assert not (source / "test.pt").exists()
    saved = cast(dict[str, Any], torch.load(source / "validation.pt", weights_only=True))
    samples = cast(dict[str, Any], torch.load(destination / "samples.pt", weights_only=True))
    torch.testing.assert_close(
        samples["factual_predictions"], saved["samples"]["predictions"], atol=0, rtol=0
    )
    for name, digest in result["hashes"].items():
        assert sha(destination / name) == digest
    with pytest.raises(FileExistsError, match="fresh"):
        audit(AuditConfig(str(source), str(cache), str(destination)))


def test_invalid_donor_population_and_nonvalidation_split_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="two independent"):
        cyclic_donors(1)
    data = build_data(_config(tmp_path), "train", VectorObservationMap(seed=1729, dimension=6))
    with pytest.raises(ValueError, match="validation"):
        measure(build_control("mlp", 0), data)
