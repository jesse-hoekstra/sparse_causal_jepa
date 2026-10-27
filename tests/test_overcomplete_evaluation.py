"""Whole12D readouts and six-factor counterfactual axes stay distinct."""

from types import SimpleNamespace
from typing import cast

import torch
from torch import Tensor, nn

from diagnostics.joint_acquisition.baseline import MatchedBatch
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.overcomplete_acquisition import evaluation as evaluation
from scjepa.models.model import SCJEPA


def test_tail_coordinates_enter_every_readout_family() -> None:
    """Recover labels stored exclusively in7..12 using real calibration-fit probes."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(182)
        codes: dict[str, Tensor] = {}
        splits: EvaluationData = {}
        for name, count in (("calibration", 768), ("validation", 192), ("test", 192)):
            tail = torch.randn(count, 6, generator=generator, dtype=torch.float64)
            codes[name] = torch.cat((torch.zeros_like(tail), tail), 1)[:, None]
            splits[name] = {"theta": tail}
        report = evaluation.parameter_readouts(codes, splits)
        for family in report["families"].values():
            assert min(family["test_r2"]) > 0.90
        x = codes["calibration"].flatten(0, 1)
        cubic = evaluation.cubic_features(x)
        assert cubic.shape == (768, 454)
        # Degree3 includes the very last coordinate cubed, not just its linear term.
        torch.testing.assert_close(cubic[:, -1], x[:, 11] ** 3)
        assert len(report["selected_test_r2"]) == 6
    finally:
        torch.set_num_threads(old)


class TailEncoder(nn.Module):
    """Put the first frame exclusively in the last six code coordinates."""

    param_dim = 12

    def forward(self, states: Tensor, _actions: Tensor) -> Tensor:
        """Expose only tail coordinates to the evaluator."""
        return torch.cat((torch.zeros_like(states[:, 0]), states[:, 0]), -1)


class TailPredictor(nn.Module):
    """Integrate the code tail once per horizon."""

    param_dim = 12

    def forward(self, states: Tensor, code: Tensor, _actions: Tensor) -> SimpleNamespace:
        """Preserve separate physical outputs and twelve-coordinate input."""
        return SimpleNamespace(prediction=states[:, 0] + code[:, 6:])


class TailModel(nn.Module):
    """Minimal differentiable container for the real evaluator."""

    context_len = 10

    def __init__(self) -> None:
        """Supply a dtype/device reference and explicit dimension attributes."""
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.parameter_encoder = TailEncoder()
        self.predictor = TailPredictor()


def test_six_counterfactual_factors_use_all_twelve_code_coordinates() -> None:
    groups, donors, queries = 3, 2, 5
    states = torch.randn(groups, donors, 10, 6, dtype=torch.float64)
    changed = states[:, :, None].expand(-1, -1, 6, -1, -1).clone()
    for factor in range(6):
        changed[:, :, factor, 0, factor] += 0.01
    batch = MatchedBatch(
        states,
        torch.zeros(groups, donors, 9, 16),
        changed,
        torch.zeros(groups, donors, queries, 1, 6),
        torch.zeros(groups, donors, queries, 2, 16),
        torch.zeros(groups, donors, queries, 2, 6),
        torch.zeros(groups, donors, 6, queries, 2, 6),
        torch.full((groups, 6), 0.01),
        tuple(range(groups)),
        (0, 1),
        torch.arange(queries).expand(donors, -1),
        "validation",
        1,
        0.0,
    )
    result = evaluation.predict_matched(cast(SCJEPA, TailModel()), batch)
    assert result.codes.shape == (groups, donors, 12)
    assert result.changed_codes.shape == (groups, donors, 6, 12)
    delta = result.changed_predictions - result.predictions[:, :, None]
    expected = torch.eye(6, dtype=torch.float64)[None, None, :, None, None, :]
    expected = expected * torch.tensor([0.01, 0.02])[None, None, None, None, :, None]
    torch.testing.assert_close(delta, expected.expand(groups, donors, 6, queries, 2, 6))


def test_probe_hyperparameters_and_families_ignore_test_labels() -> None:
    """Changing held-out answers changes scores, never calibration/validation selection."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(5)
        codes = {
            name: torch.randn(100, 1, 12, generator=generator, dtype=torch.float64)
            for name in ("calibration", "validation", "test")
        }
        splits: EvaluationData = {name: {"theta": value[:, 0, 6:]} for name, value in codes.items()}
        first = evaluation.parameter_readouts(codes, splits)
        splits["test"]["theta"] = -splits["test"]["theta"]
        second = evaluation.parameter_readouts(codes, splits)
        assert first["selected_methods_per_parameter"] == second["selected_methods_per_parameter"]
        for name, family in first["families"].items():
            assert (
                family["selected_hyperparameters_per_parameter"]
                == second["families"][name]["selected_hyperparameters_per_parameter"]
            )
        assert max(second["selected_test_r2"]) < 0
    finally:
        torch.set_num_threads(old)


def test_validation_retention_never_requires_test_data() -> None:
    """The historical scoring-slot adapter receives validation, with no test input."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(29)
        data: EvaluationData = {}
        for name in ("calibration", "validation"):
            theta = torch.randn(32, 6, generator=generator, dtype=torch.float64)
            data[name] = {
                "theta": theta,
                "states_online": theta[:, None, None].expand(-1, 10, 10, -1),
                "actions": torch.zeros(32, 10, 9, 16, dtype=torch.float64),
            }
        selection = evaluation.DiagnosticSelection(tuple(range(32)), tuple(range(32)))
        result = evaluation.validation_retention(cast(SCJEPA, TailModel()), data, selection)
        assert min(result["selected_validation_r2"]) > 0.99
        assert "selected_test_r2" not in result
        assert result["contexts_per_group"] == 10
    finally:
        torch.set_num_threads(old)
