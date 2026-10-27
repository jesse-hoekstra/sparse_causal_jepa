from collections.abc import Generator

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor

from diagnostics.mlp_acquisition_control.forward import (
    AcquisitionObjective,
    cached_forward,
    training_loss,
)
from diagnostics.mlp_oracle_code_control.model import Prediction, initialize_mlp
from scjepa.losses.regularization import RepresentationRegularizer


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def test_cached_forward_preserves_individual_context_nonself_donors_and_rng() -> None:
    model = make_model()
    initialize_mlp(model, 17)
    _, batch = make_batch(model)
    q = model.parameter_encoder(
        batch["online"][:, :, :10].flatten(0, 1), batch["actions"][:, :, :9].flatten(0, 1)
    ).reshape(2, 10, 6)
    assert not torch.equal(q[:, 0], q[:, 1])
    torch.default_generator.manual_seed(293)
    shift = int(torch.randint(1, 10, ()).item())
    expected_rng = torch.get_rng_state()
    donor = q.roll(shift, dims=1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1),
        donor,
        batch["actions"][:, :, 9:10].flatten(0, 1),
    )
    second = model.predictor(
        first.prediction[:, None], donor, batch["actions"][:, :, 10:11].flatten(0, 1)
    )
    expected = torch.stack((first.prediction, second.prediction), 1).reshape(2, 10, 2, 6)
    torch.default_generator.manual_seed(293)
    actual = cached_forward(model, batch)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual.parameters, q, rtol=0, atol=0)
    torch.testing.assert_close(actual.predictions, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.targets, batch["target"][:, :, 10:12], rtol=0, atol=0)
    altered = {key: value.clone() for key, value in batch.items()}
    altered["online"][:, :, 10:] += 100
    altered["actions"][:, :, 9:] += 100
    torch.testing.assert_close(cached_forward(model, altered).parameters, q, rtol=0, atol=0)


def test_second_step_transmits_through_first_prediction_and_donor_encoder() -> None:
    model = make_model()
    initialize_mlp(model, 17)
    _, batch = make_batch(model)
    predictions: list[Tensor] = []

    def capture(_module: torch.nn.Module, _args: tuple[object, ...], output: Prediction) -> None:
        output.prediction.retain_grad()
        predictions.append(output.prediction)

    handle = model.predictor.register_forward_hook(capture)
    try:
        output = cached_forward(model, batch)
    finally:
        handle.remove()
    loss = output.predictions[:, :, 1].square().mean()
    code_gradient = torch.autograd.grad(loss, output.parameters, retain_graph=True)[0]
    assert float(code_gradient.square().sum()) > 0
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert predictions[0].grad is not None
    assert float(predictions[0].grad.square().sum()) > 0
    for module in (model.parameter_encoder, model.predictor):
        assert (
            sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None) > 0
        )
    for module in (model.online_encoder, model.target_encoder):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert all(value.grad is None for value in batch.values())


def test_objective_matches_original_per_view_sigreg_consistency_gradients_and_rng() -> None:
    model = make_model()
    initialize_mlp(model, 17)
    _, batch = make_batch(model)
    output = cached_forward(model, batch)
    config = objective_config()
    objective = AcquisitionObjective(config)
    torch.default_generator.manual_seed(992)
    actual, terms = objective(output)
    actual_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(992)
    # Original frozen-state optimization omits the constant but retains its random projection.
    torch.randn(6, config.regularizer_num_projections, dtype=output.online_states.dtype)
    regularizer = RepresentationRegularizer("sigreg", config.regularizer_num_projections)
    sigreg = regularizer(output.parameters.transpose(0, 1))
    prediction = (output.predictions - output.targets).square().mean()
    consistency = 2 * output.parameters.var(dim=1, correction=1).mean()
    expected = prediction + 0.4 * sigreg + consistency
    assert torch.equal(torch.get_rng_state(), actual_rng)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(terms["parameter_regularizer"], sigreg, rtol=0, atol=0)
    torch.testing.assert_close(terms["parameter_consistency"], consistency, rtol=0, atol=0)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    actual_gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, parameters)
    for got, wanted in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(got, wanted, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize("extra", ["theta", "group_ids"])
def test_training_rejects_parameter_labels_and_group_identifiers(extra: str) -> None:
    model = make_model()
    initialize_mlp(model, 17)
    _, batch = make_batch(model)
    with pytest.raises(ValueError, match="exclude physical labels and group IDs"):
        training_loss(
            model, {**batch, extra: torch.zeros(2, 6)}, AcquisitionObjective(objective_config())
        )
