"""Certify action orientation and physical/cache equivalence of the paired Paley control."""

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from diagnostics.balanced_action_oracle import dgp
from diagnostics.balanced_action_oracle.data import PrepareConfig, load_data, prepare
from diagnostics.isotropic_oracle import data as source_data
from diagnostics.isotropic_oracle import dgp as source_dgp
from diagnostics.oracle_state_learning.data import read_json, sha


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _source(root: Path) -> Path:
    path = root / "original"
    source_data.prepare(
        source_data.PrepareConfig(
            str(path),
            profile="smoke",
            train_groups=4,
            calibration_groups=2,
            validation_groups=2,
            test_groups=2,
        )
    )
    return path


def test_paley_integer_certificate_and_nonsymmetric_encoding_orientation() -> None:
    integer = dgp.paley_hadamard()
    assert bool((integer.abs() == 1).all())
    assert torch.equal(integer @ integer.T, 24 * torch.eye(24, dtype=torch.int64))
    assert not torch.equal(integer, integer.T)
    assert integer[1, 2] == 1
    assert integer[2, 1] == -1
    rotation = dgp.action_rotation()
    assert bool((rotation.abs() == 1 / math.sqrt(24)).all())
    controls = torch.arange(24, dtype=torch.float64)
    torch.testing.assert_close(dgp.encode_controls(controls), rotation @ controls)
    torch.testing.assert_close(dgp.decode_actions(rotation @ controls), controls)
    assert not torch.allclose(dgp.encode_controls(controls) @ rotation.T, controls)


def test_paired_physical_transition_rollout_and_attached_parameter_gradients() -> None:
    generator = torch.Generator().manual_seed(1067)
    initial = torch.randn(7, 6, dtype=torch.float64, generator=generator)
    parameters = torch.linspace(-0.7, 0.7, 6, dtype=torch.float64, requires_grad=True)
    controls = source_dgp.sample_controls((7, 4), generator)
    source_actions = source_dgp.encode_controls(controls)
    paired_actions = dgp.encode_controls(controls)
    original = source_dgp.simulate(initial, parameters, source_actions)
    paired = dgp.simulate(initial, parameters, paired_actions)
    torch.testing.assert_close(paired, original, atol=1e-14, rtol=1e-13)
    assert not torch.allclose(source_actions, paired_actions)
    gradient = torch.autograd.grad(paired[:, -1].square().sum(), parameters)[0]
    assert bool((gradient.abs() > 1e-8).all())
    torch.testing.assert_close(
        dgp.state_rotation(paired_actions),
        source_dgp.state_rotation(source_actions),
        atol=1e-14,
        rtol=1e-13,
    )


def test_all_action_jacobians_use_correct_transpose_and_preserve_true_graphs() -> None:
    generator = torch.Generator().manual_seed(187)
    state = torch.randn(6, dtype=torch.float64, generator=generator)
    parameter = 1.5 * torch.rand(6, dtype=torch.float64, generator=generator) - 0.75
    action = dgp.encode_controls(dgp.sample_controls((), generator))
    analytic = dgp.analytic_jacobians(state, parameter, action)
    automatic = cast(
        tuple[Tensor, Tensor, Tensor],
        torch.autograd.functional.jacobian(  # pyright: ignore[reportUnknownMemberType]
            dgp.transition, (state, parameter, action)
        ),
    )
    for calculated, expected in (
        (analytic.state[:, 0], automatic[0]),
        (analytic.parameter, automatic[1]),
        (analytic.action[:, 0], automatic[2]),
    ):
        torch.testing.assert_close(calculated, expected, atol=2e-14, rtol=2e-13)
    masks = dgp.graphs()
    for derivative, mask, count in (
        (analytic.state, masks.state, 18),
        (analytic.action, masks.action, 144),
        (analytic.parameter, masks.parameter, 12),
    ):
        assert int(mask.sum()) == count
        assert torch.equal(derivative != 0, mask)
    for column in range(24):
        step = torch.zeros(24, dtype=torch.float64)
        step[column] = 1e-5
        finite = (
            dgp.transition(state, parameter, action + step)
            - dgp.transition(state, parameter, action - step)
        ) / 2e-5
        torch.testing.assert_close(analytic.action[:, 0, column], finite, atol=1e-10, rtol=1e-8)


def test_cache_pairing_is_bitwise_and_test_state_labels_are_never_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    source_before = source_data.code_identity()
    source_completion_before = sha(source / "completion.json")
    original_load = torch.load
    loaded: list[str] = []

    def observe(path: Path, *, weights_only: bool) -> object:
        name = path.name
        loaded.append(name)
        if name == "test_states.pt":
            raise AssertionError("Paired preparation must copy test label bytes without loading")
        return original_load(path, weights_only=weights_only)

    output = tmp_path / "paired"
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", observe)
        completion = prepare(PrepareConfig(str(output), str(source), profile="smoke"))
        loaded.clear()
        data, _ = load_data(output)
        assert "test.pt" not in loaded
        assert "test_states.pt" not in loaded
    assert set(data) == {"train", "calibration", "validation"}
    assert data["train"].states is None
    assert source_data.code_identity() == source_before
    assert sha(source / "completion.json") == source_completion_before
    for split in source_data.SPLITS:
        old = cast(dict[str, Any], torch.load(source / f"{split}.pt", weights_only=True))
        new = cast(dict[str, Any], torch.load(output / f"{split}.pt", weights_only=True))
        for name in ("observations", "oracle_parameters", "episode_ids"):
            assert torch.equal(old["tensors"][name], new["tensors"][name])
        assert not torch.equal(old["tensors"]["actions"], new["tensors"]["actions"])
        assert new["metadata"]["original_actions_regenerated_bitwise"]
        if split != "train":
            assert sha(source / f"{split}_states.pt") == sha(output / f"{split}_states.pt")
    assert sha(source / "observation_map.pt") == sha(output / "observation_map.pt")
    assert completion == prepare(
        PrepareConfig(str(output), str(source), profile="smoke"), resume=True
    )
    assert read_json(output / "protocol.json")["source_completion_sha256"] == (
        source_completion_before
    )


def test_original_action_regeneration_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    original = source_dgp.encode_controls

    def mismatched(controls: Tensor) -> Tensor:
        return original(controls) + 0.01

    monkeypatch.setattr(source_dgp, "encode_controls", mismatched)
    with pytest.raises(ValueError, match="Regenerated original actions differ"):
        prepare(PrepareConfig(str(tmp_path / "paired"), str(source), profile="smoke"))


def test_source_hash_tampering_is_rejected_before_copying(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with (source / "train.pt").open("ab") as stream:
        stream.write(b"tampered")
    output = tmp_path / "paired"
    with pytest.raises(ValueError, match="Completed artifact changed"):
        prepare(PrepareConfig(str(output), str(source), profile="smoke"))
    assert not output.exists()
