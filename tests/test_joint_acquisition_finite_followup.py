from pathlib import Path
from typing import cast

import pytest
import torch
from test_encoder_gradient_source import write_source_fixture
from torch import Tensor, nn

from diagnostics.joint_acquisition.finite_followup import check_direction, followup
from diagnostics.joint_acquisition.run import Config, run


class ScalarModel(nn.Module):
    """A smooth nonlinear loss with an analytic directional slope."""

    def __init__(self) -> None:
        """Initialize a scalar point in float64."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self) -> Tensor:
        """Return the nonlinear scalar loss."""
        return self.weight.exp()


def loss(model: nn.Module) -> Tensor:
    return cast(Tensor, model())


def test_smaller_two_sided_steps_validate_nonlinear_direction() -> None:
    model = ScalarModel()
    direction = {"weight": torch.tensor(100.0, dtype=torch.float64)}
    slope = float(torch.exp(torch.tensor(1.0, dtype=torch.float64))) * 100
    before = model.weight.detach().clone()
    result = check_direction(model, loss, direction, slope)
    assert result["numerically_valid"]
    assert not result["finite_steps"][0]["positive"]["valid"]
    assert result["finite_steps"][-1]["positive"]["valid"]
    assert result["finite_steps"][-1]["negative"]["valid"]
    assert torch.equal(before, model.weight)


def test_wrong_saved_gradient_does_not_unlock_trajectories() -> None:
    model = ScalarModel()
    direction = {"weight": torch.tensor(1.0, dtype=torch.float64)}
    result = check_direction(model, loss, direction, 0.0)
    assert result["finite_steps"][-1]["positive"]["valid"]
    assert not result["saved_slope_matches"]
    assert not result["numerically_valid"]
    with pytest.raises(ValueError, match="direction"):
        check_direction(model, loss, {"missing": torch.tensor(1.0)}, 0.0)


def test_followup_reuses_saved_directions_and_preserves_parent(tmp_path: Path) -> None:
    source = write_source_fixture(tmp_path / "source")
    previous = tmp_path / "previous"
    run(
        Config(
            source_dir=str(source),
            output_dir=str(previous),
            seed=7,
            groups=2,
            steps=2,
            bootstrap_samples=20,
            verification=True,
            expected_source_step=2,
            expected_checkpoint_sha256=None,
        )
    )
    before = (previous / "results.json").read_bytes()
    output = tmp_path / "followup"
    result = followup(previous, output, run_trajectories=True)
    assert (previous / "results.json").read_bytes() == before
    assert set(result["finite_checks"]) == {"encoder", "predictor", "joint"}
    assert all(row["saved_slope_matches"] for row in result["finite_checks"].values())
    assert result["numerically_valid"]
    assert all(row["status"] == "complete" for row in result["trajectories"].values())
    assert all(row["records"][-1]["step"] == 2 for row in result["trajectories"].values())
    assert (output / "completed.json").is_file()
    with pytest.raises(FileExistsError):
        followup(previous, output)
    (previous / "measurements.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        followup(previous, tmp_path / "rejected")
