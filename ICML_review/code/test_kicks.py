"""Check kick responses against the release scalar protocol and physical-coordinate surgery."""

import torch
from torch import Tensor

from diagnostics.vision_dimer import downstream
from diagnostics.vision_dimer.model import build_model

from .chart import AffineChart
from .kicks import AffineWorld, responses, score


def _contexts() -> tuple[dict[str, Tensor], Tensor]:
    gen = torch.Generator().manual_seed(7)

    def draws(*dims: int) -> Tensor:
        return torch.randn(dims, generator=gen, dtype=torch.float64)

    return {"online": draws(5, 12, 8), "actions": draws(5, 11, 5), "code": draws(5, 4)}, draws(
        80, 8
    )


@torch.inference_mode()
def test_release_scalar_kicks_and_unheld_forecast() -> None:
    """Match the release Q2 matrices and the two-step free forecast in the identity chart."""
    lat, cal = _contexts()
    model = build_model(7, phase="dense", code_dim=4, context=downstream.CONTEXT).eval()
    scalar = downstream.Model(model, lat, cal, cal)
    affine = AffineWorld(model, lat, AffineChart.fit(cal, cal))
    torch.testing.assert_close(affine.rollout({}, None), scalar.forecast()[:, 1])
    for bead in range(len(downstream.BEADS)):
        for kick in downstream.kicks(bead // 2):
            torch.testing.assert_close(
                affine.rollout(affine.current(bead), kick),
                scalar.rollout(scalar.current(bead), kick),
            )
    grid = [torch.tensor([-0.5, 0.5], dtype=torch.float64) for _ in range(8)]
    reference = downstream.experiments(scalar, grid)["q2"]
    matrices = responses(scalar)
    for bead, labels in matrices.items():
        for label in ("held", "free"):
            target = torch.tensor(reference[bead][label], dtype=torch.float64)
            torch.testing.assert_close(labels[label], target)
    identity_score = score(matrices, matrices)["mean_across_beads"]
    for label in ("held", "free", "held_minus_free"):
        assert abs(identity_score[label]["relative_error"]) < 1e-12
        assert abs(identity_score[label]["cosine"] - 1.0) < 1e-12


def test_mixed_chart_holds_two_axes_per_context() -> None:
    """The edit preserves all other physical components and removes held-bead velocity."""
    lat, cal = _contexts()
    basis = torch.eye(8, dtype=cal.dtype) + torch.tril(torch.ones((8, 8), dtype=cal.dtype), -1) / 2
    chart = AffineChart.fit(cal, cal @ basis + 1.0)
    model = build_model(3, phase="dense", code_dim=4, context=downstream.CONTEXT).eval()
    world = AffineWorld(model, lat, chart)
    hist = lat["online"][:, downstream.T - 1 : downstream.T + 1]
    expected = chart.decode(hist)
    expected[:, 0, 2:4] = expected[:, 1, 2:4]
    held = world._clamp(hist, world.current(1))  # pyright: ignore[reportPrivateUsage]
    torch.testing.assert_close(chart.decode(held), expected)
