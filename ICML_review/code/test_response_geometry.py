"""Analytical and native-routing checks for the RGB response sensitivity diagnostic."""

from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.vision_dimer.evaluate import collect
from diagnostics.vision_dimer.model import CachedEncoder, assemble, fresh_code_modules
from scjepa.models.model import SCJEPA

from .response_geometry import covariance_axes, geometry


class _Predictor:
    history_len = 1


class _Linear:
    context_len = 1
    predictor = _Predictor()

    def __init__(self, effect: Tensor, offset: Tensor) -> None:
        self.effect = effect.float()
        self.offset = offset.float()

    def rollout(self, h: Tensor, c: Tensor, steps: int, actions: Tensor) -> Tensor:
        del h, actions
        return (c @ self.effect + self.offset)[:, None].expand(-1, steps, -1)


def _case(
    codes: Tensor, targets: Tensor, transform: Tensor, offset: Tensor
) -> tuple[dict[str, Tensor], Tensor]:
    groups, views, steps, ny = targets.shape
    predictions = (codes.roll(1, dims=1) @ transform + offset)[:, :, None].expand_as(targets)
    return {
        "codes": codes,
        "states": torch.zeros(groups, views, 1, ny),
        "targets": targets,
        "predictions": predictions,
    }, torch.zeros(groups, views, steps, 1)


def _measure(
    codes: Tensor,
    targets: Tensor,
    effect: Tensor,
    offset: Tensor,
    dirs: Tensor | None = None,
    *,
    whiten: bool = True,
) -> tuple[dict[str, Any], Tensor]:
    data, actions = _case(codes, targets, effect, offset)
    if whiten:
        u, s, _, _ = covariance_axes(codes)
        v, t, _, _ = covariance_axes(targets)
        code_root, target_inverse = u * s, v / t
    else:
        code_root = torch.eye(codes.shape[-1], dtype=torch.float64)
        target_inverse = torch.eye(targets.shape[-1], dtype=torch.float64)
    return geometry(
        cast(SCJEPA, _Linear(effect, offset)),
        data,
        actions,
        torch.device("cpu"),
        code_root,
        target_inverse,
        dirs,
    )


def test_rank_one_response_is_invariant_to_affine_input_and_output_units() -> None:
    """An invertible shear and translation change native units but preserve squared ratios."""
    rng = torch.Generator().manual_seed(743)
    c = torch.randn((6, 3, 2), generator=rng, dtype=torch.float64)
    error = torch.randn((6, 3, 2, 2), generator=rng, dtype=torch.float64)
    effect = torch.tensor([[1.0, -2.0], [0.5, -1.0]], dtype=torch.float64)
    offset = torch.tensor([0.2, 0.7], dtype=torch.float64)
    y = (c.roll(1, dims=1) @ effect + offset)[:, :, None] + error
    original, _ = _measure(c, y, effect, offset)
    a = c.new_tensor([[2.0, 0.5], [-0.5, 3.0]])
    b = c.new_tensor([[0.4, 0.7], [0.0, 2.5]])
    dc, dy = c.new_tensor([-2.0, 1.5]), c.new_tensor([1.0, -3.0])
    rescaled = cast(Tensor, torch.linalg.solve(a, effect @ b))  # pyright: ignore[reportUnknownMemberType]
    converted, _ = _measure(c @ a + dc, y @ b + dy, rescaled, offset @ b + dy - dc @ rescaled)
    for result in (original, converted):
        assert abs(result["minimum_response_to_residual"]["estimate"]) < 1e-10
        assert abs(result["local_response_spectrum"][0]) < 1e-10
        assert result["local_response_spectrum"][1] > 0
    for key in (
        "local_response_spectrum",
        "mean_whitened_forecast_mse",
        "within_total_in_calibration_directions",
    ):
        torch.testing.assert_close(
            torch.as_tensor(converted[key]), torch.as_tensor(original[key]), rtol=1e-5, atol=1e-6
        )
    for key in (
        "observed_within_system_change_to_residual",
        "observed_other_system_change_to_residual",
    ):
        for stat in ("estimate", "interval"):
            torch.testing.assert_close(
                torch.as_tensor(converted[key][stat]),
                torch.as_tensor(original[key][stat]),
                rtol=1e-5,
                atol=1e-6,
            )


def test_transported_calibration_direction_and_validation_minimum_can_disagree() -> None:
    """Rotate the exact null between panels and keep the calibration basis fixed."""
    c = torch.randn((4, 3, 2), generator=torch.Generator().manual_seed(64), dtype=torch.float64)
    b = c.new_zeros(1)
    cal_effect = c.new_tensor([[0.0], [2.0]])
    val_effect = c.new_tensor([[3.0], [0.0]])

    def targets(effect: Tensor) -> Tensor:
        prediction = c.roll(1, dims=1) @ effect
        return (prediction + 1)[:, :, None].expand(-1, -1, 2, -1)

    first, axes = _measure(c, targets(cal_effect), cal_effect, b, whiten=False)
    second, fixed = _measure(c, targets(val_effect), val_effect, b, axes, whiten=False)
    torch.testing.assert_close(fixed, axes)
    for result, expected in ((first, [0.0, 4.0]), (second, [9.0, 0.0])):
        for r, x in zip(
            result["calibration_directions_response_to_residual"], expected, strict=True
        ):
            torch.testing.assert_close(
                torch.as_tensor([r["estimate"], *r["interval"]]),
                torch.full((3,), x),
                atol=1e-5,
                rtol=1e-5,
            )
        assert abs(result["minimum_response_to_residual"]["estimate"]) < 1e-10


def test_replays_native_forecast_and_swaps_disjoint_alien_pairs() -> None:
    """Use the real lagged predictor, native grouped forward, cached targets and explicit donors."""
    context, steps, views, groups = 4, 2, 3, 4
    code_module, predictor = fresh_code_modules(139, 2, context)
    model = assemble(CachedEncoder(), code_module, predictor, context).eval().requires_grad_(False)
    rng = torch.Generator().manual_seed(819)
    states = torch.randn((groups, views, context + steps, 8), generator=rng)
    actions = torch.randn((groups, views, context + steps - 1, 5), generator=rng)
    targets = torch.randn((groups, views, steps, 8), generator=rng)
    targets[:2] *= 2
    batch = {"observations": states, "actions": actions, "targets": targets}
    samples = collect(model, iter([batch]))
    code_root, target_inverse = torch.eye(2, dtype=torch.float64), torch.eye(8, dtype=torch.float64)
    result, _ = geometry(
        model, samples, actions, torch.device("cpu"), code_root, target_inverse, None
    )
    assert result["maximum_reference_difference"] < 1e-6
    native = samples["predictions"]
    residual = (native - targets.double()).square().mean((1, 2, 3))
    torch.testing.assert_close(
        torch.as_tensor(result["mean_whitened_forecast_mse"], dtype=torch.float64), residual.mean()
    )
    with torch.no_grad():
        for key, indices, shift in (
            ("observed_within_system_change_to_residual", [0, 1, 2, 3], 2),
            ("observed_other_system_change_to_residual", [1, 0, 3, 2], 1),
        ):
            donor = samples["codes"][indices][:, [(i - shift) % views for i in range(views)]]
            moved = model.rollout(
                states[:, :, context - 2 : context].flatten(0, 1),
                donor.flatten(0, 1).float(),
                steps,
                actions[:, :, context - 2 :].flatten(0, 1),
            )
            numerator = (moved.reshape_as(native).double() - native).square().mean((1, 2, 3))
            estimate = torch.as_tensor(result[key]["estimate"], dtype=torch.float64)
            torch.testing.assert_close(
                estimate, numerator.mean() / residual.mean(), rtol=1e-5, atol=1e-13
            )
            if key == "observed_other_system_change_to_residual":
                # With two independent pairs, the extreme empirical ratios each have mass 1/4.
                pair_ratios = numerator.reshape(2, 2).sum(1) / residual.reshape(2, 2).sum(1)
                torch.testing.assert_close(
                    torch.as_tensor(result[key]["interval"], dtype=torch.float64),
                    pair_ratios.sort()[0],
                    rtol=1e-5,
                    atol=1e-13,
                )
    incorrect = samples | {"predictions": native + 0.01}
    with pytest.raises(RuntimeError, match="native reference"):
        geometry(model, incorrect, actions, torch.device("cpu"), code_root, target_inverse, None)
