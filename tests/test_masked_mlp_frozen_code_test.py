"""Test scoring keeps calibration assignments and the separate frozen branch."""

from typing import cast

import torch

from diagnostics.masked_mlp_learned import frozen_code as control
from diagnostics.masked_mlp_learned import frozen_code_test as evaluation
from diagnostics.masked_mlp_learned import run
from diagnostics.shared.oracle import EvaluationResult
from scjepa.eval.recovery import fit_coordinate_match
from scjepa.models.model import ModelOutput


def test_test_parameter_mcc_does_not_refit_an_assignment_on_scored_labels() -> None:
    generator = torch.Generator().manual_seed(3)
    theta = torch.randn(256, 6, generator=generator)
    codes = theta.roll(1, dims=1)[:, None].expand(-1, 10, -1)
    result = EvaluationResult({}, {"codes": codes, "theta": theta})
    assert evaluation.parameter_mcc(result, list(range(6))) < 0.15
    best = fit_coordinate_match(codes[:, 0], theta, method="pearson")
    assert evaluation.parameter_mcc(result, list(best.columns)) > 0.999


def test_restored_test_model_preserves_frozen_full_branch_and_baseline_state_branch() -> None:
    torch.set_num_threads(1)
    baseline = run.build_control("baseline", 0, predictor_activation="gelu")
    donor = run.build_control("full", 1, predictor_activation="gelu")
    model = control.FrozenCodeModel(
        baseline,
        donor,
        "full",
        torch.arange(1, 7).float(),
        control.pair_rotation(((0, 3), (1, 4), (2, 5))),
    ).eval()
    inputs = {"observations": torch.randn(2, 10, 12, 6), "actions": torch.randn(2, 10, 11, 24)}
    with torch.no_grad():
        expected = cast(ModelOutput, model(**inputs))
        restored = evaluation.restore_model({"model": model.state_dict()}, "full", 0)
        actual = cast(ModelOutput, restored(**inputs))
    for key in ("predictions", "targets", "online_states", "parameters"):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
    assert model.frozen_identity() == restored.frozen_identity()
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert all(not module.training for module in restored.modules())
