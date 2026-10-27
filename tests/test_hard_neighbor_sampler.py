from collections.abc import Generator, Iterable
from typing import cast

import pytest
import torch
from test_multiquery_control import make_batch, make_model

from diagnostics.group_contrastive_pretext.forward import cached_codes, training_loss
from diagnostics.hard_neighbor_pretext.sampler import (
    Bank,
    hard_indices,
    refresh_bank,
    refresh_due,
)
from diagnostics.multiquery_control.data import module_fingerprint
from scjepa.data.datasets import tensor_fingerprint


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_nearest_order_self_exclusion_ties_and_independent_anchor_rng() -> None:
    means = torch.zeros(9, 6, dtype=torch.float64)
    means[:, 0] = torch.tensor([0, 1, -1, 2, -2, 3, -3, 1, -1], dtype=torch.float64)
    bank = Bank(means, 5001, "a" * 64)
    seed, step = 17, 5003
    anchor_seed = (seed * 1_000_003 + step + 202610270501) % (2**63 - 1)
    anchor = int(torch.randint(9, (), generator=torch.Generator().manual_seed(anchor_seed)))
    neighbors = sorted(
        [index for index in range(9) if index != anchor],
        key=lambda index: (float((means[index] - means[anchor]).square().sum()), index),
    )[:5]
    expected = [anchor, *neighbors]
    torch.default_generator.manual_seed(123)
    before = torch.get_rng_state()
    assert hard_indices(bank, 6, seed, step) == expected
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(111)
    assert hard_indices(bank, 6, seed, step) == expected
    assert len(set(expected)) == 6
    collapsed = Bank(torch.zeros_like(means), 5001, "b" * 64)
    assert hard_indices(collapsed, 6, seed, step) == [
        anchor,
        *[index for index in range(9) if index != anchor][:5],
    ]
    assert len({hard_indices(bank, 2, seed, current)[0] for current in range(5001, 5017)}) > 1


def test_refresh_boundary_is_after_warmup_and_counts_hard_updates() -> None:
    assert not any(refresh_due(step) for step in (1, 4999, 5000, 5002, 5512, 5514))
    assert all(refresh_due(step) for step in (5001, 5513, 6025, 6537))
    assert [step for step in range(1, 8) if refresh_due(step, warmup_steps=1, interval=2)] == [
        2,
        4,
        6,
    ]


@pytest.mark.parametrize("groups", [2, 5])
def test_bank_refresh_is_detached_train_only_and_preserves_model_rng_and_modes(groups: int) -> None:
    model = make_model()
    model.train()
    model.online_encoder.eval()
    model.target_encoder.eval()
    model.predictor.eval()
    _, train = make_batch(model)
    train = {name: torch.cat([value] * 3, dim=0)[:groups] for name, value in train.items()}
    named_modules = list(cast(Iterable[tuple[str, torch.nn.Module]], model.named_modules()))
    before_modes = {name: module.training for name, module in named_modules}
    modules = (model.online_encoder, model.target_encoder, model.parameter_encoder, model.predictor)
    hashes = [module_fingerprint(module) for module in modules]
    torch.default_generator.manual_seed(29)
    rng = torch.get_rng_state()
    bank = refresh_bank(model, train, 5001, group_batch_size=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert {name: module.training for name, module in named_modules} == before_modes
    assert [module_fingerprint(module) for module in modules] == hashes
    with torch.no_grad():
        expected = cached_codes(model.eval(), train).mean(1).double().cpu()
    torch.testing.assert_close(bank.means, expected, rtol=1e-12, atol=1e-14)
    assert bank.means.dtype == torch.float64
    assert bank.means.device.type == "cpu"
    assert not bank.means.requires_grad
    assert bank.refresh_step == 5001
    assert bank.encoder_sha256 == hashes[2]
    assert all(parameter.grad is None for parameter in model.parameters())
    for extra in ("theta", "group_ids", "validation"):
        with pytest.raises(ValueError, match="label|keys|cache|TRAIN|physical"):
            refresh_bank(model, {**train, extra: torch.zeros(2, 6)}, 5001)


def test_bank_payload_replays_exactly_and_detects_mean_tampering() -> None:
    bank = Bank(torch.arange(30, dtype=torch.float64).reshape(5, 6), 5513, "c" * 64)
    restored = Bank.from_payload(bank.payload())
    torch.testing.assert_close(restored.means, bank.means, rtol=0, atol=0)
    assert restored.refresh_step == bank.refresh_step
    assert restored.encoder_sha256 == bank.encoder_sha256
    assert hard_indices(restored, 3, 7, 5520) == hard_indices(bank, 3, 7, 5520)
    payload = bank.payload()
    payload["means"] = payload["means"].clone()
    payload["means"][0, 0] += 1
    with pytest.raises(ValueError, match="hash|changed|fingerprint"):
        Bank.from_payload(payload)
    for key, changed in (("refresh_step", 5514), ("encoder_sha256", "d" * 64)):
        metadata = bank.payload()
        metadata[key] = changed
        with pytest.raises(ValueError, match="metadata changed"):
            Bank.from_payload(metadata)
    with pytest.raises(ValueError, match="detached group bank"):
        Bank(bank.means.clone().requires_grad_(), bank.refresh_step, bank.encoder_sha256)


def test_selected_contexts_are_recomputed_with_gradients_not_bank_values() -> None:
    model = make_model()
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    _, original = make_batch(model)
    train = {name: torch.cat((value, value + 0.1), dim=0) for name, value in original.items()}
    bank = refresh_bank(model, train, 5001, group_batch_size=2)
    means_hash = tensor_fingerprint(bank.means)
    transformed = Bank(100 * bank.means + 73, bank.refresh_step, bank.encoder_sha256)
    first = hard_indices(bank, 2, 7, 5001)
    second = hard_indices(transformed, 2, 7, 5001)
    assert first == second
    parameters = tuple(model.parameter_encoder.parameters())
    batch = {name: value[first] for name, value in train.items()}
    actual, _ = training_loss(model, batch)
    gradients = torch.autograd.grad(actual, parameters, retain_graph=True)
    repeated, _ = training_loss(model, {name: value[second] for name, value in train.items()})
    repeated_gradients = torch.autograd.grad(repeated, parameters)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    for observed, expected in zip(gradients, repeated_gradients, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert sum(float(value.square().sum()) for value in gradients) > 0
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    actual.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    assert module_fingerprint(model.parameter_encoder) != bank.encoder_sha256
    assert tensor_fingerprint(bank.means) == means_hash
    assert bank.means.grad is None
    assert all(parameter.grad is None for parameter in model.predictor.parameters())
    assert all(value.grad is None for value in train.values())
