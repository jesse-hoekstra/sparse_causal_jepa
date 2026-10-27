import math
from collections.abc import Generator

import pytest
import torch
from test_multiquery_control import make_batch, make_model, objective_config
from torch import Tensor, nn

from diagnostics.encoder_gain_acquisition.model import apply_encoder_gains
from diagnostics.mlp_acquisition_control.forward import AcquisitionObjective, cached_forward
from diagnostics.mlp_forecast_only_acquisition.run import training_loss
from diagnostics.mlp_oracle_code_control.model import Prediction, initialize_mlp
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.models.parameter_encoder import TransitionParameterEncoder


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_gain_changes_only_exact_six_archived_weights_without_rng_or_replacement() -> None:
    model = make_model().float()
    initialize_mlp(model, 17)
    encoder = model.parameter_encoder
    assert type(encoder) is TransitionParameterEncoder
    encoder_hash = module_fingerprint(encoder)
    before = {name: value.clone() for name, value in encoder.state_dict().items()}
    identities = {name: id(parameter) for name, parameter in encoder.named_parameters()}
    fixed = [
        module_fingerprint(module)
        for module in (model.online_encoder, model.target_encoder, model.predictor)
    ]
    rng = torch.get_rng_state()
    receipt = apply_encoder_gains(model)
    assert torch.equal(rng, torch.get_rng_state())
    assert model.parameter_encoder is encoder
    assert type(encoder).forward is TransitionParameterEncoder.forward
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 70534
    assert receipt["module_sha256"] == module_fingerprint(encoder)
    assert receipt["original_encoder_sha256"] == encoder_hash
    assert receipt["bias_hashes_before"] == receipt["bias_hashes_after"]
    assert identities == {name: id(parameter) for name, parameter in encoder.named_parameters()}
    assert fixed == [
        module_fingerprint(module)
        for module in (model.online_encoder, model.target_encoder, model.predictor)
    ]
    weights = {
        f"{name}.weight"
        for name in ("transition.0", "transition.2", "transition.4", "head.0", "head.2", "head.4")
    }
    assert {name for name in before if name.endswith("weight")} == weights
    for name, value in encoder.state_dict().items():
        gain = math.sqrt(3 if name == "head.4.weight" else 6)
        expected = before[name] * gain if name in weights else before[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_gain_rejects_unmatched_dtype_before_mutating_source_weights() -> None:
    model = make_model()
    encoder_hash = module_fingerprint(model.parameter_encoder)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="CPUfloat32"):
        apply_encoder_gains(model)
    assert module_fingerprint(model.parameter_encoder) == encoder_hash
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("frozen_predictor", [False, True])
def test_gain_keeps_single_context_codes_nonself_attached_rollout_and_gradients(
    frozen_predictor: bool,
) -> None:
    model = make_model()
    initialize_mlp(model, 17)
    _, original_batch = make_batch(model)
    batch = {key: value.float() for key, value in original_batch.items()}
    model.float()
    apply_encoder_gains(model)
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(not frozen_predictor)
    codes = model.parameter_encoder(
        batch["online"][:, :, :10].flatten(0, 1), batch["actions"][:, :, :9].flatten(0, 1)
    ).reshape(2, 10, 6)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["online"][:, :, 10:] += 100
    changed["actions"][:, :, 9:] += 100
    torch.testing.assert_close(cached_forward(model, changed).parameters, codes, rtol=0, atol=0)
    changed["online"][:, 3, :10] += 10
    changed_codes = cached_forward(model, changed).parameters
    others = [view for view in range(10) if view != 3]
    torch.testing.assert_close(changed_codes[:, others], codes[:, others], rtol=0, atol=0)
    assert not torch.equal(changed_codes[:, 3], codes[:, 3])
    torch.default_generator.manual_seed(179)
    shift = int(torch.randint(1, 10, ()))
    donor = codes.roll(shift, 1).flatten(0, 1)
    first = model.predictor(
        batch["online"][:, :, 9:10].flatten(0, 1), donor, batch["actions"][:, :, 9:10].flatten(0, 1)
    ).prediction
    second = model.predictor(
        first[:, None], donor, batch["actions"][:, :, 10:11].flatten(0, 1)
    ).prediction
    captured: list[Tensor] = []

    def capture(_module: nn.Module, _args: tuple[object, ...], output: Prediction) -> None:
        output.prediction.retain_grad()
        captured.append(output.prediction)

    handle = model.predictor.register_forward_hook(capture)
    torch.default_generator.manual_seed(179)
    try:
        output = cached_forward(model, batch)
    finally:
        handle.remove()
    torch.testing.assert_close(
        output.predictions, torch.stack((first, second), 1).reshape(2, 10, 2, 6), rtol=0, atol=0
    )
    loss = (output.predictions[:, :, 1] - output.targets[:, :, 1]).square().mean()
    gradient = torch.autograd.grad(loss, output.parameters, retain_graph=True)[0]
    assert float(gradient.square().sum()) > 0
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert captured[0].grad is not None
    assert float(captured[0].grad.square().sum()) > 0
    assert any(
        p.grad is not None and bool(p.grad.any()) for p in model.parameter_encoder.parameters()
    )
    if frozen_predictor:
        assert all(p.grad is None for p in model.predictor.parameters())
    model.predictor.requires_grad_(True)
    fixed = [module_fingerprint(module) for module in (model.online_encoder, model.target_encoder)]
    learned = [module_fingerprint(module) for module in (model.parameter_encoder, model.predictor)]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    optimized, terms = training_loss(model, batch, AcquisitionObjective(objective_config()))
    assert optimized is terms["prediction"]
    optimized.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert all(
        module_fingerprint(module) != previous
        for module, previous in zip(
            (model.parameter_encoder, model.predictor), learned, strict=True
        )
    )
    assert fixed == [
        module_fingerprint(module) for module in (model.online_encoder, model.target_encoder)
    ]
    assert all(value.grad is None for value in batch.values())
