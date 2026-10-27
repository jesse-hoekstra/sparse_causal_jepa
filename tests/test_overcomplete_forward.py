import copy
import math
from collections.abc import Generator
from typing import cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective
from diagnostics.mlp_acquisition_control.forward import cached_forward as original_forward
from diagnostics.mlp_oracle_code_control.model import Prediction, initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.overcomplete_acquisition.forward import cached_forward, training_loss
from diagnostics.overcomplete_acquisition.model import (
    OvercompletePredictor,
    initialize_overcomplete,
)
from scjepa.losses.regularization import RepresentationRegularizer
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.rng import preserve_rng


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def gained_model() -> SCJEPA:
    model = make_model().float()
    initialize_mlp(model, 17)
    apply_encoder_gains(model)
    return model


def batch_for(model: SCJEPA) -> dict[str, Tensor]:
    _, batch = make_batch(copy.deepcopy(model).double())
    return {name: value.float() for name, value in batch.items()}


def test_expansion_preserves_common_weights_and_exact_seeded_added_interfaces() -> None:
    model = gained_model()
    old_encoder, old_predictor = model.parameter_encoder, model.predictor
    original_encoder = copy.deepcopy(old_encoder.state_dict())
    original_predictor = copy.deepcopy(old_predictor.state_dict())
    fixed = (module_fingerprint(model.online_encoder), module_fingerprint(model.target_encoder))
    before_rng = torch.get_rng_state()
    receipt = initialize_overcomplete(model, 17)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert type(model.parameter_encoder) is TransitionParameterEncoder
    assert isinstance(model.predictor, OvercompletePredictor)
    assert model.parameter_encoder.param_dim == model.predictor.param_dim == 12
    assert model.parameter_encoder.state_dim == model.predictor.state_dim == 6
    assert receipt["encoder_parameter_count"] == 71308
    assert receipt["predictor_parameter_count"] == 53389
    assert receipt["additional_encoder_seed"] == 18
    assert receipt["additional_predictor_seed"] == 19
    assert receipt["encoder_sha256"] == module_fingerprint(model.parameter_encoder)
    assert receipt["predictor_sha256"] == module_fingerprint(model.predictor)
    for name, value in model.parameter_encoder.state_dict().items():
        common = value[:6] if name in {"head.4.weight", "head.4.bias"} else value
        torch.testing.assert_close(common, original_encoder[name], rtol=0, atol=0)
    for name, value in model.predictor.state_dict().items():
        common = value[:, :28] if name == "network.0.weight" else value
        torch.testing.assert_close(common, original_predictor[name], rtol=0, atol=0)
    # The original six-code source modules themselves are never rewritten.
    torch.testing.assert_close(old_encoder.state_dict(), original_encoder, rtol=0, atol=0)
    torch.testing.assert_close(old_predictor.state_dict(), original_predictor, rtol=0, atol=0)
    assert fixed == (
        module_fingerprint(model.online_encoder),
        module_fingerprint(model.target_encoder),
    )
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(18)
        extra_output = nn.Linear(128, 6)
        torch.default_generator.manual_seed(19)
        extra_columns = torch.empty(211, 6).uniform_(-1 / math.sqrt(28), 1 / math.sqrt(28))
    output = cast(nn.Linear, model.parameter_encoder.head[4])
    first = cast(nn.Linear, model.predictor.network[0])
    torch.testing.assert_close(
        output.weight[6:], extra_output.weight * math.sqrt(3), rtol=0, atol=0
    )
    torch.testing.assert_close(output.bias[6:], extra_output.bias, rtol=0, atol=0)
    torch.testing.assert_close(first.weight[:, 28:], extra_columns, rtol=0, atol=0)
    assert bool((first.weight[:, 28:].square().sum(0) > 0).all())
    second = gained_model()
    torch.default_generator.manual_seed(929)
    initialize_overcomplete(second, 17)
    torch.testing.assert_close(
        second.parameter_encoder.state_dict(), model.parameter_encoder.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        second.predictor.state_dict(), model.predictor.state_dict(), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG preservation requires CUDA")
def test_cpu_expansion_preserves_caller_cuda_rng_too() -> None:
    model = gained_model()
    torch.cuda.manual_seed(793)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    initialize_overcomplete(model, 17)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_complete_twelve_code_forward_preserves_context_boundaries_and_donor_rng() -> None:
    model = gained_model()
    baseline = copy.deepcopy(model)
    initialize_overcomplete(model, 17)
    batch = batch_for(model)
    encoder = model.parameter_encoder
    assert isinstance(encoder, TransitionParameterEncoder)
    codes = encoder(
        batch["online"][:, :, :10].flatten(0, 1), batch["actions"][:, :, :9].flatten(0, 1)
    ).reshape(2, 10, 12)
    assert codes.shape == (2, 10, 12)
    torch.default_generator.manual_seed(293)
    original_forward(baseline, batch)
    baseline_rng = torch.get_rng_state()
    torch.default_generator.manual_seed(293)
    shift = int(torch.randint(1, 10, ()))
    assert 1 <= shift <= 9
    donor = codes.roll(shift, 1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1), donor, batch["actions"][:, :, 9:10].flatten(0, 1)
    ).prediction
    second = model.predictor(
        first[:, None], donor, batch["actions"][:, :, 10:11].flatten(0, 1)
    ).prediction
    torch.default_generator.manual_seed(293)
    output = cached_forward(model, batch)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    torch.testing.assert_close(output.parameters, codes, rtol=0, atol=0)
    torch.testing.assert_close(
        output.predictions, torch.stack((first, second), 1).reshape(2, 10, 2, 6), rtol=0, atol=0
    )
    torch.testing.assert_close(output.targets, batch["target"][:, :, 10:12], rtol=0, atol=0)
    changed = {name: value.clone() for name, value in batch.items()}
    changed["online"][:, :, 10:] += 100
    changed["actions"][:, :, 9:] += 100
    torch.testing.assert_close(cached_forward(model, changed).parameters, codes, rtol=0, atol=0)
    changed["online"][:, 3, :10] += 10
    other_codes = cached_forward(model, changed).parameters
    others = [index for index in range(10) if index != 3]
    torch.testing.assert_close(other_codes[:, others], codes[:, others], rtol=0, atol=0)
    assert not torch.equal(other_codes[:, 3], codes[:, 3])
    for coordinate in range(6, 12):
        perturbed = donor.detach().clone()
        perturbed[:, coordinate] += 1
        altered = model.predictor(
            batch["online"][:, :, 9:10].flatten(0, 1),
            perturbed,
            batch["actions"][:, :, 9:10].flatten(0, 1),
        ).prediction
        assert not torch.equal(altered, first)


@pytest.mark.parametrize("frozen_predictor", [False, True])
def test_second_horizon_backpropagates_through_first_prediction_and_all_twelve_codes(
    frozen_predictor: bool,
) -> None:
    model = gained_model()
    initialize_overcomplete(model, 17)
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(not frozen_predictor)
    batch = batch_for(model)
    predictions: list[Tensor] = []

    def capture(_module: nn.Module, _args: tuple[object, ...], output: Prediction) -> None:
        output.prediction.retain_grad()
        predictions.append(output.prediction)

    handle = model.predictor.register_forward_hook(capture)
    try:
        output = cached_forward(model, batch)
    finally:
        handle.remove()
    loss = (output.predictions[:, :, 1] - output.targets[:, :, 1]).square().mean()
    gradient = torch.autograd.grad(loss, output.parameters, retain_graph=True)[0]
    assert bool((gradient.square().sum((0, 1)) > 0).all())
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert predictions[0].grad is not None
    assert float(predictions[0].grad.square().sum()) > 0
    encoder = model.parameter_encoder
    assert isinstance(encoder, TransitionParameterEncoder)
    last = cast(nn.Linear, encoder.head[4])
    assert last.weight.grad is not None
    assert bool((last.weight.grad.square().sum(1) > 0).all())
    if frozen_predictor:
        assert all(parameter.grad is None for parameter in model.predictor.parameters())
    else:
        assert isinstance(model.predictor, OvercompletePredictor)
        first = cast(nn.Linear, model.predictor.network[0])
        assert first.weight.grad is not None
        assert bool((first.weight.grad[:, 22:].square().sum(0) > 0).all())
    assert all(parameter.grad is None for parameter in model.online_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert all(value.grad is None for value in batch.values())


def test_forecast_only_gradient_and_auxiliary_rng_use_all_twelve_codes_without_labels() -> None:
    model = gained_model()
    initialize_overcomplete(model, 17)
    batch = batch_for(model)
    config = objective_config()
    objective = AcquisitionObjective(config)
    torch.default_generator.manual_seed(493)
    loss, terms = training_loss(model, batch, objective)
    actual_rng = torch.get_rng_state()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    actual_gradients = torch.autograd.grad(loss, parameters)
    torch.default_generator.manual_seed(493)
    output = cached_forward(model, batch)
    prediction = (output.predictions - output.targets).square().mean()
    torch.randn(6, config.regularizer_num_projections)
    regularizer = RepresentationRegularizer("sigreg", config.regularizer_num_projections)
    sigreg = regularizer(output.parameters.transpose(0, 1))
    consistency = 2 * output.parameters.var(dim=1, correction=1).mean()
    assert torch.equal(actual_rng, torch.get_rng_state())
    torch.testing.assert_close(loss, prediction, rtol=0, atol=0)
    torch.testing.assert_close(terms["parameter_regularizer"], sigreg, rtol=0, atol=0)
    torch.testing.assert_close(terms["parameter_consistency"], consistency, rtol=0, atol=0)
    expected_gradients = torch.autograd.grad(prediction, parameters)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="exclude physical labels"):
        training_loss(model, {**batch, "theta": torch.zeros(2, 6)}, objective)
    changed = dict(batch, online=batch["online"].clone().requires_grad_())
    with pytest.raises(ValueError, match="remain frozen"):
        training_loss(model, changed, objective)
