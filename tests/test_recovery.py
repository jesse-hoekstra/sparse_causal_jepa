import math
from itertools import permutations
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from scjepa.eval.recovery import (
    CoordinateMatch,
    FrozenProbe,
    ProbeConfig,
    ProbePayload,
    fit_coordinate_match,
    fit_group_probes,
    fit_nonlinear_probe,
    probe_metrics,
    score_coordinate_match,
    spearman_matrix,
)


def _normal(rows: int, columns: int, seed: int = 123) -> Tensor:
    return torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed))


def test_average_ties_constant_columns_and_spearman_orientation() -> None:
    features = torch.tensor([[0.0, 1], [2, 1], [2, 1], [4, 1]])
    truth = torch.tensor([[-1.0], [-1], [2], [3]])
    matrix = spearman_matrix(features, truth)
    assert matrix.shape == (1, 2)
    assert math.isclose(matrix[0, 0].item(), 5 / 6)
    assert matrix[0, 1].item() == 0
    assert torch.isfinite(matrix).all()
    matched = score_coordinate_match(CoordinateMatch((0, 1)), features, features)
    assert matched["constant_learned_coordinates"] == 1
    assert matched["constant_truth_coordinates"] == 1
    assert math.isclose(matched["spearman_mean"], 0.5)


def test_match_recovers_permutation_sign_and_monotone_transform() -> None:
    truth = _normal(200, 4).double()
    features = torch.stack(
        (-truth[:, 2], torch.exp(truth[:, 0]), truth[:, 3].pow(3), truth[:, 1]), dim=1
    )
    match = fit_coordinate_match(features, truth)
    assert match.columns == (1, 3, 0, 2)
    scores = score_coordinate_match(match, features, truth)
    assert math.isclose(scores["spearman_min"], 1.0)


def test_assignment_is_exact_global_one_to_one_optimum() -> None:
    truth = _normal(30, 5)
    features = _normal(30, 5, seed=92) + truth @ _normal(5, 5, seed=100)
    scores = spearman_matrix(features, truth)
    match = fit_coordinate_match(features, truth)
    obtained = float(scores[torch.arange(5), list(match.columns)].sum())
    optimum = max(
        float(scores[torch.arange(5), list(columns)].sum()) for columns in permutations(range(5))
    )
    assert math.isclose(obtained, optimum)
    assert len(set(match.columns)) == 5


def test_test_scoring_never_rematches_coordinates() -> None:
    calibration = _normal(300, 2)
    test_truth = _normal(300, 2, seed=321)
    match = fit_coordinate_match(calibration, calibration)
    test_features = test_truth.flip(1)
    frozen = score_coordinate_match(match, test_features, test_truth)
    rematched = score_coordinate_match(
        fit_coordinate_match(test_features, test_truth), test_features, test_truth
    )
    assert frozen["spearman_mean"] < 0.15
    assert math.isclose(rematched["spearman_mean"], 1.0)
    assert match.columns == (0, 1)


@pytest.mark.parametrize("columns", [(), (0, 0), (2, 0), (-1, 0)])
def test_invalid_coordinate_permutations_rejected(columns: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="permutation"):
        CoordinateMatch(columns)


def test_matching_validation() -> None:
    with pytest.raises(ValueError, match="equal learned and true dimensions"):
        fit_coordinate_match(_normal(10, 2), _normal(10, 3))
    with pytest.raises(ValueError, match="finite"):
        spearman_matrix(torch.full((10, 2), float("nan")), _normal(10, 2))
    with pytest.raises(ValueError, match="same number"):
        spearman_matrix(_normal(11, 2), _normal(10, 2))
    with pytest.raises(ValueError, match="dimensions"):
        score_coordinate_match(CoordinateMatch((0, 1)), _normal(10, 3), _normal(10, 3))


def test_nonlinear_probe_recovers_analytic_map_on_heldout_samples() -> None:
    generator = torch.Generator().manual_seed(29)
    calibration = torch.rand(600, 2, generator=generator) * 3 - 1.5
    test = torch.rand(200, 2, generator=generator) * 3 - 1.5

    def target(features: Tensor) -> Tensor:
        return torch.stack(
            (features[:, 0] + 0.2 * features[:, 0].pow(3), torch.sin(1.5 * features[:, 1])), dim=1
        )

    probe = fit_nonlinear_probe(
        calibration,
        target(calibration),
        config=ProbeConfig(hidden_dim=32, steps=500, batch_size=128, lr=0.01, seed=12),
    )
    scores = probe_metrics(probe.predict(test), target(test))
    assert scores["r2"] > 0.99
    assert scores["mse"] < 0.01
    assert probe.predict(test).dtype == torch.float64


def test_probe_is_frozen_rng_isolated_and_uses_calibration_statistics_only() -> None:
    features = (_normal(60, 3) * 2 + 5).requires_grad_()
    truth = (_normal(60, 2) * 3 - 2).requires_grad_()
    episode_ids = torch.arange(12).repeat_interleave(5)
    before_rng = torch.get_rng_state().clone()
    config = ProbeConfig(steps=3, seed=20)
    with torch.no_grad():
        first = fit_nonlinear_probe(features, truth, config=config, episode_ids=episode_ids)
    assert torch.equal(before_rng, torch.get_rng_state())
    second = fit_nonlinear_probe(features, truth, config=config, episode_ids=episode_ids)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(first.feature_mean, features.detach().double().mean(0))
    assert torch.equal(first.target_mean, truth.detach().double().mean(0))
    assert torch.equal(first.calibration_episode_ids, torch.arange(12))
    before_mean = first.feature_mean.clone()
    prediction = first.predict(features.detach() + 1e3)
    assert not prediction.requires_grad
    assert torch.equal(first.feature_mean, before_mean)
    assert torch.equal(first.predict(features), second.predict(features))
    assert all(not parameter.requires_grad for parameter in first.network.parameters())
    assert features.grad is None
    assert truth.grad is None


def test_probe_linear_recovery_and_payload_restore(tmp_path: Path) -> None:
    features = _normal(90, 4)
    truth = features @ _normal(4, 2) + torch.tensor([1.0, -5.0])
    probe = fit_nonlinear_probe(features, truth, config=ProbeConfig(steps=2, lr=1e-5))
    test = _normal(20, 4, seed=987)
    prediction = probe.predict(test)
    expected = test @ _normal(4, 2) + torch.tensor([1.0, -5.0])
    assert probe_metrics(prediction, expected)["r2"] > 0.999
    path = tmp_path / "probe.pt"
    torch.save(probe.to_payload(), path)
    payload = cast(ProbePayload, torch.load(path, weights_only=True))
    before_rng = torch.get_rng_state().clone()
    restored = FrozenProbe.from_payload(payload)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.equal(prediction, restored.predict(test))
    assert restored.config == probe.config
    assert restored.predict(test.reshape(2, 10, 4)).shape == (2, 10, 2)
    assert restored.predict(test[:0]).shape == (0, 2)
    assert restored.predict(test[0]).shape == (2,)
    torch.testing.assert_close(
        prediction, restored.predict(test, batch_size=3), rtol=1e-6, atol=1e-6
    )


def test_group_probes_use_frozen_assigned_subsets() -> None:
    truth = _normal(100, 4)
    features = truth[:, [2, 0, 3, 1]]
    match = fit_coordinate_match(features, truth)
    groups = fit_group_probes(
        features,
        truth,
        ((0, 1), (2, 3)),
        match=match,
        config=ProbeConfig(steps=2, lr=1e-5),
    )
    assert groups[0].feature_columns == (1, 3)
    assert groups[1].feature_columns == (0, 2)
    assert groups[0].truth_columns == (0, 1)
    perturbed = features.clone()
    perturbed[:, [0, 2]] += 20
    assert torch.equal(groups[0].predict(features), groups[0].predict(perturbed))
    assert probe_metrics(groups[1].predict(features), truth[:, [2, 3]])["r2"] > 0.999
    with pytest.raises(ValueError, match="partition"):
        fit_group_probes(features, truth, ((0, 1), (1, 2, 3)), match=match)


def test_constant_probe_targets_and_degenerate_metrics_have_no_nan() -> None:
    features = torch.ones(10, 3)
    truth = torch.full((10, 2), 6.0)
    probe = fit_nonlinear_probe(features, truth, config=ProbeConfig(steps=2))
    assert torch.equal(probe.predict(features), truth.double())
    metrics = probe_metrics(probe.predict(features), truth)
    assert metrics["constant_truth_coordinates"] == 2
    assert metrics["r2"] == 0
    assert metrics["mse"] == 0
    assert probe_metrics(truth + 2, truth)["mse"] == 4


@pytest.mark.parametrize(
    "config",
    [
        {"steps": 0},
        {"hidden_dim": 0},
        {"batch_size": 0},
        {"lr": float("nan")},
        {"ridge": 0.0},
        {"seed": -1},
    ],
)
def test_probe_config_validation(config: dict[str, int | float]) -> None:
    with pytest.raises(ValueError, match="probe"):
        ProbeConfig(**config)  # type: ignore[arg-type]


def test_probe_input_and_payload_validation() -> None:
    features, truth = _normal(10, 3), _normal(10, 2)
    with pytest.raises(ValueError, match="episode_ids"):
        fit_nonlinear_probe(features, truth, episode_ids=torch.zeros(10))
    with pytest.raises(ValueError, match="same number"):
        fit_nonlinear_probe(features, truth[:5])
    probe = fit_nonlinear_probe(features, truth, config=ProbeConfig(steps=1))
    with pytest.raises(ValueError, match="feature dimension"):
        probe.predict(features[:, :2])
    with pytest.raises(ValueError, match="finite"):
        probe.predict(features * float("inf"))
    payload = probe.to_payload()
    payload["version"] = 123
    with pytest.raises(ValueError, match="version"):
        FrozenProbe.from_payload(payload)
    payload = probe.to_payload()
    payload["feature_std"].zero_()
    with pytest.raises(ValueError, match="positive"):
        FrozenProbe.from_payload(payload)
