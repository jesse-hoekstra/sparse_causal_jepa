import torch

from diagnostics.oracle_state_learning.analysis import (
    grouped_moments,
    moment_mcc,
    paired_mcc_interval,
)
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient


def test_group_moments_match_pearson_and_keep_calibration_assignment() -> None:
    generator = torch.Generator().manual_seed(6)
    truth = torch.randn(8, 10, 12, 6, generator=generator)
    columns = [2, 4, 0, 5, 3, 1]
    features = torch.empty_like(truth)
    features[..., columns] = 2 * truth + 1
    features[..., columns[0]] = 0
    actual = float(moment_mcc(grouped_moments(features, truth, columns).sum(0)))
    expected = mean_correlation_coefficient(
        CoordinateMatch(tuple(columns)), features.flatten(0, 2), truth.flatten(0, 2)
    )
    assert abs(actual - expected) < 1e-12
    assert abs(actual - 5 / 6) < 1e-12


def test_paired_bootstrap_identical_arms_zero_and_mixing_positive() -> None:
    generator = torch.Generator().manual_seed(9)
    truth = torch.randn(24, 10, 12, 6, generator=generator)
    columns = list(range(6))
    identical = paired_mcc_interval(truth, truth, truth, columns, columns, draws=40)
    assert identical["difference"] == identical["lower_95"] == identical["upper_95"] == 0
    mixed = (truth + truth.roll(1, -1)) / 2**0.5
    improved = paired_mcc_interval(truth, mixed, truth, columns, columns, draws=100)
    assert improved["lower_95"] > 0.25
    assert improved["upper_95"] < 0.35
