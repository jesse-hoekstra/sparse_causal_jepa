"""Masking semantics, pairing and exact within-block chart equivariance of the masked MLP."""

from typing import cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.isotropic_oracle.model import build_control as build_native
from diagnostics.masked_mlp_oracle.model import (
    MaskedTrunkPredictor,
    build_control,
    build_predictor,
    trunk_width,
)


def _block_orthogonal(seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation = torch.zeros(6, 6, dtype=torch.float64)
    for block in (slice(0, 3), slice(3, 6)):
        sample = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        q, r = cast(tuple[Tensor, Tensor], torch.linalg.qr(sample))  # pyright: ignore[reportUnknownMemberType]
        rotation[block, block] = q * torch.sign(torch.diagonal(r))
    return rotation


def _with_block_state_masks(predictor: MaskedTrunkPredictor) -> MaskedTrunkPredictor:
    assert predictor.state_logits is not None
    with torch.no_grad():
        predictor.state_logits.fill_(-3.0)
        predictor.state_logits[:3, :3] = 3.0
        predictor.state_logits[3:, 3:] = 3.0
    return predictor


def _recharted(predictor: MaskedTrunkPredictor, rotation: Tensor) -> MaskedTrunkPredictor:
    """Absorb z' = M z into the trunk's state columns and the per-output readouts."""
    copy = MaskedTrunkPredictor(
        learn_state=predictor.state_logits is not None,
        learn_action=predictor.action_logits is not None,
        learn_parameter=predictor.parameter_logits is not None,
    ).double()
    copy.load_state_dict(predictor.state_dict())
    with torch.no_grad():
        copy.first.weight[:, :6] = predictor.first.weight[:, :6] @ rotation.T
        copy.readout_weight.copy_(rotation @ predictor.readout_weight)
        copy.readout_bias.copy_(rotation @ predictor.readout_bias)
    return copy.eval()


def _inputs(batch: int = 64) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(7)
    states = torch.randn(batch, 1, 6, generator=generator, dtype=torch.float64)
    parameters = torch.randn(batch, 6, generator=generator, dtype=torch.float64)
    actions = torch.randn(batch, 1, 24, generator=generator, dtype=torch.float64)
    return states, parameters, actions


def test_dense_is_the_parameter_matched_global_mlp_and_arms_are_paired() -> None:
    native = build_native("mlp", 0).predictor
    dense = build_predictor("dense")
    assert trunk_width() == 209
    assert [tuple(p.shape) for p in native.parameters()] == [
        (209, 36),
        (209,),
        (209, 209),
        (209,),
        (6, 209),
        (6,),
    ]
    assert sum(p.numel() for p in dense.parameters()) == sum(p.numel() for p in native.parameters())
    assert dense.learned_logits() == {}
    models = {condition: build_control(condition, 1) for condition in ("dense", "baseline", "full")}
    shared = dict(models["dense"].predictor.named_parameters())
    for condition in ("baseline", "full"):
        tensors = dict(models[condition].predictor.named_parameters())
        assert all(torch.equal(tensors[name], value) for name, value in shared.items())
        for left, right in zip(
            models[condition].online_encoder.parameters(),
            models["dense"].online_encoder.parameters(),
            strict=True,
        ):
            assert torch.equal(left, right)
    assert set(dict(models["baseline"].predictor.named_parameters())) - set(shared) == {
        "state_logits",
        "action_logits",
    }
    for online, target in zip(
        models["full"].online_encoder.parameters(),
        models["full"].target_encoder.parameters(),
        strict=True,
    ):
        assert torch.equal(online, target)
    other = cast(MaskedTrunkPredictor, cast(nn.Module, build_control("full", 2).predictor))
    reference = cast(MaskedTrunkPredictor, cast(nn.Module, models["full"].predictor))
    assert not torch.equal(other.first.weight, reference.first.weight)


def test_baseline_is_exactly_equivariant_and_complementary_theta_masks_break_it() -> None:
    rotation = _block_orthogonal(3)
    states, parameters, actions = _inputs()
    baseline = _with_block_state_masks(build_predictor("baseline").double()).eval()
    moved = _recharted(baseline, rotation)
    with torch.no_grad():
        expected = baseline(states, parameters, actions).prediction @ rotation.T
        actual = moved(states @ rotation.T, parameters, actions).prediction
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    full = _with_block_state_masks(build_predictor("full").double()).eval()
    assert full.parameter_logits is not None
    own_or_partner = torch.eye(6, dtype=torch.bool) | torch.eye(6, dtype=torch.bool).roll(3, 1)
    with torch.no_grad():
        full.parameter_logits.copy_(torch.where(own_or_partner, 3.0, -3.0))
        broken = _recharted(full, rotation)(states @ rotation.T, parameters, actions).prediction
        reference = full(states, parameters, actions).prediction @ rotation.T
    assert float((broken - reference).abs().max()) > 1e-3


def test_masks_are_static_thresholds_with_exact_expected_open_fraction() -> None:
    predictor = build_predictor("full").double()
    assert predictor.parameter_logits is not None
    with torch.no_grad():
        predictor.parameter_logits.copy_(torch.linspace(-2, 2, 36).reshape(6, 6))
    states, parameters, actions = _inputs(8)
    predictor.eval()
    output = predictor(states, parameters, actions)
    expected = (predictor.parameter_logits > 0).double().expand(8, 6, 6)
    torch.testing.assert_close(output.graphs.parameter, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        output.sparsity.parameter, torch.sigmoid(predictor.parameter_logits).mean()
    )
    assert torch.equal(predictor.thresholded_graphs()["parameter"], predictor.parameter_logits > 0)
    baseline = build_predictor("baseline").double()
    baseline_output = baseline.eval()(states, parameters, actions)
    assert float(baseline_output.sparsity.parameter) == 1.0
    assert bool((baseline_output.graphs.parameter == 1).all())
    assert bool(baseline.thresholded_graphs()["parameter"].all())


def test_training_gates_are_hard_and_pass_straight_through_gradients() -> None:
    predictor = build_predictor("full").double().train()
    states, parameters, actions = _inputs(256)
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    output = predictor(states, parameters, actions)
    values = output.graphs.state.detach()
    # Straight-through values equal hard gates up to floating-point rounding.
    assert bool(((values - values.round()).abs() < 1e-12).all())
    assert bool((values.round() == 0).any())
    output.prediction.square().mean().backward()
    for logits in predictor.learned_logits().values():
        assert logits.grad is not None
        assert bool(logits.grad.abs().sum() > 0)


def test_shapes_are_validated() -> None:
    predictor = build_predictor("full")
    states, parameters, actions = (value.float() for value in _inputs(4))
    with pytest.raises(ValueError, match="actions"):
        predictor(states, parameters, None)
    with pytest.raises(ValueError, match="states"):
        predictor(states[:, :, :5], parameters, actions)
