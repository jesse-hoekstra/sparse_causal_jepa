from collections.abc import Generator

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_forecast_only_acquisition.run import training_loss
from diagnostics.mlp_oracle_code_control.model import MLPPredictor, Prediction, initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from diagnostics.wide_encoder_acquisition.model import (
    WideTransitionParameterEncoder,
    initialize_wide_encoder,
)
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


def test_wide_topology_inherits_original_transition_pooling_and_raw_six_code() -> None:
    encoder = WideTransitionParameterEncoder(6, 6, 16, 10).double()
    assert type(encoder).forward is TransitionParameterEncoder.forward
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 272134
    original = TransitionParameterEncoder(6, 6, 16, 10)
    assert sum(parameter.numel() for parameter in original.parameters()) == 70534
    assert [
        (layer.in_features, layer.out_features)
        for layer in encoder.transition
        if isinstance(layer, nn.Linear)
    ] == [(28, 256), (256, 256), (256, 256)]
    assert [
        (layer.in_features, layer.out_features)
        for layer in encoder.head
        if isinstance(layer, nn.Linear)
    ] == [(256, 256), (256, 256), (256, 6)]
    assert sum(isinstance(layer, nn.GELU) for layer in encoder.modules()) == 5
    generator = torch.Generator().manual_seed(157)
    states = torch.randn(3, 10, 6, generator=generator, dtype=torch.float64)
    actions = torch.randn(3, 9, 16, generator=generator, dtype=torch.float64)
    transitions = torch.cat((states[:, :-1], actions, states[:, 1:]), -1)
    expected = encoder.head(encoder.transition(transitions).mean(1))
    torch.testing.assert_close(encoder(states, actions), expected, rtol=0, atol=0)
    changed = states.clone()
    changed[1] += 10
    independent = encoder(changed, actions)
    torch.testing.assert_close(independent[[0, 2]], expected[[0, 2]], rtol=0, atol=0)
    assert not torch.equal(independent[1], expected[1])
    output = encoder.head[-1]
    assert isinstance(output, nn.Linear)
    bias = torch.tensor([-3, -2, -1, 1, 2, 3], dtype=torch.float64)
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(bias)
    torch.testing.assert_close(encoder(states, actions), bias.expand(3, -1), rtol=0, atol=0)


def test_wide_initialization_preserves_original_mlp_state_maps_and_caller_rng() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    fixed = (model.online_encoder, model.target_encoder, model.predictor)
    before = [module_fingerprint(module) for module in fixed]
    torch.default_generator.manual_seed(273)
    rng = torch.get_rng_state()
    receipt = initialize_wide_encoder(model, 7)
    assert torch.equal(rng, torch.get_rng_state())
    assert [module_fingerprint(module) for module in fixed] == before
    assert next(model.parameter_encoder.parameters()).dtype == torch.float64
    assert receipt["hidden_width"] == 256
    assert receipt["scientific_seed"] == 7
    assert receipt["encoder_initialization_seed"] == 8
    assert receipt["parameter_count"] == 272134
    assert receipt["module_sha256"] == module_fingerprint(model.parameter_encoder)
    assert isinstance(model.parameter_encoder, WideTransitionParameterEncoder)
    assert isinstance(model.predictor, MLPPredictor)
    first_encoder = model.parameter_encoder.transition[0]
    first_predictor = model.predictor.network[0]
    assert isinstance(first_encoder, nn.Linear)
    assert isinstance(first_predictor, nn.Linear)
    assert first_encoder.in_features == first_predictor.in_features == 28
    assert not torch.equal(first_encoder.weight[:211], first_predictor.weight)
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(8)
        reference = WideTransitionParameterEncoder(6, 6, 16, 10).double()
    torch.testing.assert_close(
        reference.state_dict(), model.parameter_encoder.state_dict(), rtol=0, atol=0
    )
    second = make_model()
    initialize_wide_encoder(second, 7)
    initialize_mlp(second, 7)
    torch.testing.assert_close(
        second.parameter_encoder.state_dict(), model.parameter_encoder.state_dict(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        second.predictor.state_dict(), model.predictor.state_dict(), rtol=0, atol=0
    )
    assert sum(parameter.numel() for parameter in model.predictor.parameters()) == 52123


def test_wide_cached_forward_keeps_context_boundaries_and_nonself_attached_rollout() -> None:
    model = make_model()
    initialize_mlp(model, 7)
    initialize_wide_encoder(model, 7)
    _, batch = make_batch(model)
    expected_codes = model.parameter_encoder(
        batch["online"][:, :, :10].flatten(0, 1),
        batch["actions"][:, :, :9].flatten(0, 1),
    ).reshape(2, 10, 6)
    altered = {name: value.clone() for name, value in batch.items()}
    altered["online"][:, :, 10:] += 100
    altered["actions"][:, :, 9:] += 100
    torch.testing.assert_close(
        cached_forward(model, altered).parameters, expected_codes, rtol=0, atol=0
    )
    torch.default_generator.manual_seed(179)
    shift = int(torch.randint(1, 10, ()))
    assert shift != 0
    donor = expected_codes.roll(shift, dims=1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1), donor, batch["actions"][:, :, 9:10].flatten(0, 1)
    ).prediction
    second = model.predictor(
        first[:, None], donor, batch["actions"][:, :, 10:11].flatten(0, 1)
    ).prediction
    torch.default_generator.manual_seed(179)
    actual = cached_forward(model, batch)
    torch.testing.assert_close(actual.parameters, expected_codes, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.predictions, torch.stack((first, second), 1).reshape(2, 10, 2, 6), rtol=0, atol=0
    )
    for frame in (0, 9):
        context = {name: value.clone() for name, value in batch.items()}
        context["online"][:, 3, frame] += 10
        changed_codes = cached_forward(model, context).parameters
        assert not torch.equal(changed_codes[:, 3], expected_codes[:, 3])
        others = [view for view in range(10) if view != 3]
        torch.testing.assert_close(
            changed_codes[:, others], expected_codes[:, others], rtol=0, atol=0
        )


@pytest.mark.parametrize("frozen_predictor", [False, True])
def test_wide_second_step_transmits_through_first_prediction_and_updates_both_modules(
    frozen_predictor: bool,
) -> None:
    model = make_model()
    initialize_mlp(model, 7)
    initialize_wide_encoder(model, 7)
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(not frozen_predictor)
    _, batch = make_batch(model)
    captured: list[Tensor] = []

    def capture(_module: nn.Module, _args: tuple[object, ...], result: Prediction) -> None:
        result.prediction.retain_grad()
        captured.append(result.prediction)

    handle = model.predictor.register_forward_hook(capture)
    try:
        output = cached_forward(model, batch)
    finally:
        handle.remove()
    second_loss = (output.predictions[:, :, 1] - output.targets[:, :, 1]).square().mean()
    gradient = torch.autograd.grad(second_loss, output.parameters, retain_graph=True)[0]
    assert float(gradient.square().sum()) > 0
    second_loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert captured[0].grad is not None
    assert float(captured[0].grad.square().sum()) > 0
    active = (
        (model.parameter_encoder,)
        if frozen_predictor
        else (model.parameter_encoder, model.predictor)
    )
    for module in active:
        assert (
            sum(
                float(parameter.grad.square().sum())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            > 0
        )
    if frozen_predictor:
        assert all(parameter.grad is None for parameter in model.predictor.parameters())
    model.predictor.requires_grad_(True)
    frozen = (module_fingerprint(model.online_encoder), module_fingerprint(model.target_encoder))
    learned = (module_fingerprint(model.parameter_encoder), module_fingerprint(model.predictor))
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001
    )
    optimizer.zero_grad(set_to_none=True)
    loss, terms = training_loss(model, batch, AcquisitionObjective(objective_config()))
    torch.testing.assert_close(loss, terms["prediction"], rtol=0, atol=0)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert module_fingerprint(model.parameter_encoder) != learned[0]
    assert module_fingerprint(model.predictor) != learned[1]
    assert (
        module_fingerprint(model.online_encoder),
        module_fingerprint(model.target_encoder),
    ) == frozen
    assert all(value.grad is None for value in batch.values())
