"""Matched physical changes preserve recipient inputs and allow mixed complete codes."""

import copy
from collections.abc import Iterator
from dataclasses import replace
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from diagnostics.balanced_action_oracle import dgp
from diagnostics.balanced_action_oracle.data import StateData
from diagnostics.masked_mlp_learned import physical_response as assay
from diagnostics.masked_mlp_learned.run import build_control
from diagnostics.masked_mlp_oracle.model import MaskedTrunkPredictor
from diagnostics.masked_mlp_oracle.model import build_control as build_oracle
from scjepa.data.controlled_vector import VectorObservationMap


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _data() -> tuple[StateData, VectorObservationMap]:
    generator = torch.Generator().manual_seed(800)
    theta = torch.rand(2, 6, generator=generator) - 0.5
    initial = torch.randn(2, 10, 6, generator=generator, dtype=torch.float64)
    actions = dgp.encode_controls(dgp.sample_controls((2, 10, 11), generator)).float()
    physical = dgp.simulate(initial, theta[:, None], actions)
    observation = VectorObservationMap(dimension=6)
    return (
        StateData(
            "validation",
            {
                "oracle_parameters": theta,
                "observations": observation.forward(physical).float(),
                "actions": actions,
                "episode_ids": torch.arange(2),
            },
            physical,
            {"action_mix": "paley24", "rho": 0.7, "rotation_scale": dgp.ROTATION_SCALE},
        ),
        observation,
    )


def test_reconstruction_guards_action_chart_and_observation_map() -> None:
    data, observation = _data()
    model = build_control("dense", 0).double().eval()
    batch = assay.matched_batch(model, data, observation, (0, 1))
    assert batch.reconstruction_error < 2e-5
    assert batch.changed_targets.shape == (2, 1, 6, 5, 2, 6)
    assert batch.donors == (0,)
    torch.testing.assert_close(
        batch.query_actions[:, 0], data.tensors["actions"][:, 5:10, 9:11].double()
    )
    with pytest.raises(ValueError, match="reconstruction"):
        assay.matched_batch(model, data, VectorObservationMap(seed=22, dimension=6), (0, 1))
    corrupted = replace(data, tensors=data.tensors | {"actions": data.tensors["actions"] + 0.1})
    with pytest.raises(ValueError, match="reconstruction"):
        assay.matched_batch(model, corrupted, observation, (0, 1))
    for forbidden in ("train", "calibration", "test"):
        wrong = replace(data, split=forbidden, states=None if forbidden == "train" else data.states)
        with pytest.raises(ValueError, match="validation"):
            assay.matched_batch(model, wrong, observation, (0, 1))


@pytest.mark.parametrize("width", [6, 12])
def test_physical_predictions_are_invariant_to_invertible_complete_code_mixing(width: int) -> None:
    data, observation = _data()
    model = build_control("dense", 0, width).double().eval()
    batch = assay.matched_batch(model, data, observation, (0, 1))
    before = assay.predict_matched(model, batch)
    mixed = copy.deepcopy(model)
    transform = torch.eye(width, dtype=torch.float64) + 0.1 * torch.ones(
        width, width, dtype=torch.float64
    )
    output = cast(nn.Linear, cast(nn.Sequential, mixed.parameter_encoder.head)[4])
    predictor = cast(MaskedTrunkPredictor, cast(nn.Module, mixed.predictor))
    inverse = cast(Tensor, torch.linalg.inv(transform))  # pyright: ignore[reportUnknownMemberType]
    with torch.no_grad():
        output.weight.copy_(transform @ output.weight)
        output.bias.copy_(transform @ output.bias)
        predictor.first.weight[:, -width:].copy_(predictor.first.weight[:, -width:] @ inverse)
    after = assay.predict_matched(mixed, batch)
    assert not torch.equal(before.codes, after.codes)
    torch.testing.assert_close(after.codes, before.codes @ transform.T, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(after.predictions, before.predictions, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        after.changed_predictions, before.changed_predictions, atol=1e-12, rtol=1e-10
    )


def test_frozen_predictor_transmits_the_second_forecast_gradient_to_codes() -> None:
    data, observation = _data()
    model = build_control("dense", 0).double().eval()
    cast(nn.Module, model.predictor).requires_grad_(False)
    batch = assay.matched_batch(model, data, observation, (0, 1))
    output = assay.predict_matched(model, batch)
    code_gradient = torch.autograd.grad(output.predictions[..., 1, :].square().sum(), output.codes)[
        0
    ]
    assert bool(torch.isfinite(code_gradient).all())
    assert float(code_gradient.abs().sum()) > 0
    report, raw = assay.assess(model, data, observation, chunk_size=1)
    assert len(report["per_parameter"]) == 6
    assert raw["factual_predictions"].shape == (2, 1, 5, 2, 6)
    assert raw["finite_steps"].shape == (2, 6)
    torch.testing.assert_close(raw["predictive_benefit"], raw["benefit_from_direct_vectors"])


@pytest.mark.parametrize("condition", ["dense", "full"])
def test_oracle_reference_bypasses_context_encoder_and_perturbs_physical_inputs(
    condition: assay.oracle_run.Condition, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, observation = _data()
    model = build_oracle(condition, 0).double().eval()

    def forbidden(*_args: Tensor) -> Tensor:
        raise AssertionError("Oracle reference must bypass the unused parameter encoder")

    monkeypatch.setattr(model.parameter_encoder, "forward", forbidden)
    batch = assay.matched_batch(model, data, observation, (0, 1))
    theta = data.tensors["oracle_parameters"].double()
    output = assay.predict_matched(model, batch, oracle_parameters=theta)
    torch.testing.assert_close(output.codes[:, 0], theta)
    torch.testing.assert_close(
        output.changed_codes[:, 0] - theta[:, None], torch.diag_embed(batch.finite_steps)
    )

    # Independently forecast a changed recipient through both steps. Its second
    # input must be the first forecast and the same changed physical parameter.
    group, factor, recipient = 1, 4, 3
    parameters = theta[group : group + 1].clone()
    parameters[:, factor] += batch.finite_steps[group, factor]
    history = batch.history[group : group + 1, 0, recipient]
    actions = batch.query_actions[group : group + 1, 0, recipient]
    first = model.predictor(history, parameters, actions[:, :1]).prediction
    second = model.predictor(first[:, None], parameters, actions[:, 1:2]).prediction
    torch.testing.assert_close(
        output.changed_predictions[group, 0, factor, recipient],
        torch.stack((first[0], second[0])),
    )
    report, raw = assay.assess(model, data, observation, chunk_size=1, parameter_source="oracle")
    assert report["parameter_source"] == "oracle"
    assert len(report["per_parameter"]) == 6
    torch.testing.assert_close(raw["factual_predictions"], output.predictions)
    torch.testing.assert_close(raw["counterfactual_predictions"], output.changed_predictions)


def test_oracle_reference_rejects_wrong_dimensions_or_split() -> None:
    data, observation = _data()
    model = build_oracle("dense", 0).double().eval()
    batch = assay.matched_batch(model, data, observation, (0, 1))
    theta = data.tensors["oracle_parameters"]
    with pytest.raises(ValueError, match="six parameters per validation group"):
        assay.predict_matched(model, batch, oracle_parameters=theta[:1])
    with pytest.raises(ValueError, match="six parameters per validation group"):
        assay.predict_matched(model, replace(batch, split="train"), oracle_parameters=theta)
    overcomplete = build_control("dense", 0, 12).double().eval()
    with pytest.raises(ValueError, match="six parameters per validation group"):
        assay.predict_matched(overcomplete, batch, oracle_parameters=theta)
