import copy
from collections.abc import Generator
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import Tensor, nn

import scjepa.models.spartan as spartan_module
from diagnostics.gate_replay.core import replay_matched
from diagnostics.gate_replay.run import evaluate_arm
from diagnostics.joint_acquisition.baseline import MatchedBatch, evaluate_responses
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.multiquery_control.data import build_cache
from diagnostics.oracle_code_control.evaluation import matched_batch
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.models.spartan import SpartanLayer, SpartanOutput
from scjepa.training.factory import ModelConfig, build_model


@pytest.fixture(autouse=True)
def one_thread() -> Generator[None, None, None]:
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def model_and_batch(*, dense: bool = False, baseline: bool = False) -> tuple[SCJEPA, MatchedBatch]:
    torch.default_generator.manual_seed(2811)
    model = (
        build_model(
            ModelConfig(
                observation_dim=6,
                encoder_hidden_dims=(8,),
                state_dim=6,
                param_dim=6,
                action_dim=16,
                context_len=10,
                history_len=1,
                parameter_encoder_kind="transition",
                spartan_layers=2,
                spartan_embed_dim=8,
                spartan_mlp_hidden=16,
                spartan_mlp_layers=2,
                spartan_dense=dense,
                spartan_parameter_gates_open=baseline,
                autoregressive=True,
            )
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    # A wide code range exercises influential routing changes, rather than
    # incidental flips in unused token workspaces at initialization.
    with torch.no_grad():
        head = cast(nn.Linear, cast(TransitionParameterEncoder, model.parameter_encoder).head[-1])
        head.weight.mul_(50)
        assert head.bias is not None
        head.bias.zero_()
    groups, donors, queries = 2, 1, 3
    batch = MatchedBatch(
        donor_states=torch.randn(groups, donors, 10, 6, dtype=torch.float64),
        donor_actions=torch.randn(groups, donors, 9, 16, dtype=torch.float64),
        changed_donor_states=4 * torch.randn(groups, donors, 6, 10, 6, dtype=torch.float64),
        history=torch.randn(groups, donors, queries, 1, 6, dtype=torch.float64),
        query_actions=torch.randn(groups, donors, queries, 2, 16, dtype=torch.float64),
        targets=torch.randn(groups, donors, queries, 2, 6, dtype=torch.float64),
        changed_targets=torch.randn(groups, donors, 6, queries, 2, 6, dtype=torch.float64),
        finite_steps=torch.full((groups, 6), 0.01, dtype=torch.float64),
        group_indices=(0, 1),
        donors=(0,),
        query_indices=torch.tensor([[5, 6, 7]]),
        split="validation",
        fit_queries=0,
        reconstruction_error=0,
    )
    return model, batch


def explicit_masked_rollout(
    model: SCJEPA, batch: MatchedBatch, codes: Tensor, masks: Tensor
) -> Tensor:
    """Independent two-step attention calculation with factual supports and live weights."""
    predictor = model.predictor
    groups, donors, queries = batch.history.shape[:3]
    history = batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3)
    actions = batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3)
    codes = codes[:, :, :, None].expand(-1, -1, -1, queries, -1).flatten(0, 3)
    expanded = masks[:, :, None].expand(-1, -1, 6, -1, -1, -1, -1, -1).flatten(0, 3)
    predictions: list[Tensor] = []
    for horizon in range(2):
        assert predictor.action_project is not None
        tokens = (
            torch.cat(
                (
                    predictor.state_project(history.flatten(1).unsqueeze(-1)),
                    predictor.action_project(actions[:, horizon].unsqueeze(-1)),
                    predictor.param_project(codes.unsqueeze(-1)),
                ),
                dim=1,
            )
            + predictor.token_identities()
        )
        for layer_index, raw_layer in enumerate(predictor.layers):
            layer = cast(SpartanLayer, raw_layer)
            normalized = layer.norm(tokens)
            logits = torch.einsum(
                "bid,bjd->bij", layer.project_q(normalized), layer.project_k(normalized)
            )
            mask = expanded[:, horizon, layer_index].to(logits)
            scores = logits * layer.scale
            row_max = scores.masked_fill(mask == 0, -float("inf")).amax(-1, keepdim=True)
            row_max = torch.where(torch.isfinite(row_max), row_max, 0.0)
            weights = mask * (scores - row_max).clamp(max=0).exp()
            denominator = weights.sum(-1, keepdim=True)
            weights = weights / (denominator + (denominator < 0.5).to(weights.dtype))
            attended = torch.einsum("bij,bjd->bid", weights, layer.project_v(normalized))
            tokens = layer.mlp(tokens + attended)
        predicted = predictor.out_project(tokens[:, :6]).squeeze(-1)
        predictions.append(predicted)
        history = predicted[:, None]
    return torch.stack(predictions, dim=1).reshape(groups, donors, 6, queries, 2, 6)


@pytest.mark.parametrize("baseline", [False, True])
def test_actual_mask_intervention_matches_independent_attention_and_preserves_forced_routes(
    baseline: bool,
) -> None:
    model, batch = model_and_batch(baseline=baseline)
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    result = replay_matched(model, batch)
    assert result.factual_masks.shape == (2, 1, 3, 2, 2, 28, 28)
    assert result.natural_changed_masks.shape == (2, 1, 6, 3, 2, 2, 28, 28)
    assert result.factual_masks.dtype == torch.bool
    assert result.factual_masks.device.type == "cpu"
    assert result.flip_counts.shape == (2, 1, 6, 3, 2, 2)
    assert int(result.flip_counts.sum()) > 0
    expected_counts = (result.natural_changed_masks != result.factual_masks[:, :, None]).sum(
        (-1, -2)
    )
    torch.testing.assert_close(result.flip_counts, expected_counts, rtol=0, atol=0)
    torch.testing.assert_close(result.factual_replay, result.natural.predictions, rtol=0, atol=0)
    expected = explicit_masked_rollout(
        model, batch, result.natural.changed_codes, result.factual_masks
    )
    torch.testing.assert_close(result.replayed_predictions, expected, rtol=1e-12, atol=1e-12)
    assert (
        float((result.replayed_predictions - result.natural.changed_predictions).abs().max()) > 1e-8
    )
    if baseline:
        assert bool(result.factual_masks[..., :6, -6:].all())
        assert bool(result.natural_changed_masks[..., :6, -6:].all())
    torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_dense_replay_is_exact_identity_and_never_invokes_sampler() -> None:
    model, batch = model_and_batch(dense=True)
    with patch.object(
        spartan_module, "_sample_hard_adjacency", side_effect=AssertionError("Dense sampler")
    ):
        result = replay_matched(model, batch)
    assert bool(result.factual_masks.all())
    assert int(result.flip_counts.sum()) == 0
    torch.testing.assert_close(
        result.replayed_predictions, result.natural.changed_predictions, rtol=0, atol=0
    )


@pytest.mark.parametrize("baseline", [False, True])
def test_closed_rows_and_forced_open_baseline_are_replayed_before_normalization(
    baseline: bool,
) -> None:
    model, batch = model_and_batch(baseline=baseline)
    with torch.no_grad():
        for raw_layer in model.predictor.layers:
            layer = cast(SpartanLayer, raw_layer)
            layer.norm.weight.zero_()
            layer.norm.bias.fill_(1)
            layer.project_q.weight.copy_(torch.eye(8))
            layer.project_k.weight.copy_(-100 * torch.eye(8))
    result = replay_matched(model, batch)
    assert bool(torch.isfinite(result.replayed_predictions).all())
    assert int(result.flip_counts.sum()) == 0
    expected = torch.zeros(28, 28, dtype=torch.bool)
    if baseline:
        expected[:6, -6:] = True
    torch.testing.assert_close(
        result.factual_masks, expected.expand_as(result.factual_masks), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result.replayed_predictions, result.natural.changed_predictions, rtol=0, atol=0
    )


def test_replay_second_step_uses_own_first_prediction_with_attached_code_gradient() -> None:
    model, batch = model_and_batch(baseline=True)
    batch.changed_donor_states.requires_grad_(True)
    histories: list[Tensor] = []
    outputs: list[Tensor] = []

    def capture(_module: nn.Module, args: tuple[object, ...], output: object) -> None:
        histories.append(cast(Tensor, args[0]))
        outputs.append(cast(SpartanOutput, output).prediction)

    hook = model.predictor.register_forward_hook(capture)
    try:
        result = replay_matched(model, batch)
    finally:
        hook.remove()
    assert len(outputs) == 8
    torch.testing.assert_close(histories[7][:, 0], outputs[6], rtol=0, atol=0)
    assert not torch.equal(outputs[6], outputs[2])
    first_gradient = torch.autograd.grad(outputs[7].square().sum(), outputs[6], retain_graph=True)[
        0
    ]
    code_gradient = torch.autograd.grad(
        result.replayed_predictions.square().sum(), result.natural.changed_codes, retain_graph=True
    )[0]
    donor_gradient = torch.autograd.grad(
        result.replayed_predictions.square().sum(), batch.changed_donor_states
    )[0]
    assert float(first_gradient.square().sum()) > 0
    assert float(code_gradient.square().sum()) > 0
    assert float(donor_gradient.square().sum()) > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_replay_restores_sampler_and_hooks_after_counterfactual_exception() -> None:
    model, batch = model_and_batch()
    sampler = spartan_module._sample_hard_adjacency  # pyright: ignore[reportPrivateUsage]
    original = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    calls = 0

    def fail_late(_module: nn.Module, _args: tuple[object, ...]) -> None:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("Injected during counterfactual replay")

    hook = model.predictor.register_forward_pre_hook(fail_late)
    try:
        with pytest.raises(RuntimeError, match="Injected"):
            replay_matched(model, batch)
    finally:
        hook.remove()
    assert spartan_module._sample_hard_adjacency is sampler  # pyright: ignore[reportPrivateUsage]
    assert all(not layer._forward_hooks for layer in model.predictor.layers)  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(model.state_dict(), original, rtol=0, atol=0)
    assert int(replay_matched(model, batch).flip_counts.sum()) > 0
    with pytest.raises(ValueError, match="validation"):
        replay_matched(model, replace(batch, split="test"))
    with pytest.raises(ValueError, match="eval-mode"):
        replay_matched(model.train(), batch)


@pytest.mark.parametrize("condition", ["dense", "baseline", "full"])
def test_arm_evaluation_uses_every_existing_validation_group_and_one_paired_bootstrap(
    condition: str,
) -> None:
    model, _ = model_and_batch(dense=condition == "dense", baseline=condition == "baseline")
    model.float()
    _, cached = build_cache(
        model,
        VectorObservationMap(dimension=6),
        {
            "family": "simplified_gaussian",
            "root_seed": 53,
            "episodes_per_parameter": 10,
            "trajectory_length": 12,
        },
        train_groups=4,
        calibration_groups=4,
        validation_groups=4,
        test_groups=4,
        eval_seed=281503,
        batch_size=2,
    )
    data = {name: cached[name] for name in ("metadata", "validation")}
    model.eval()
    frozen = copy.deepcopy(model.state_dict())
    data_before = copy.deepcopy(data)
    rng = torch.get_rng_state().clone()
    progress: list[int] = []
    report, raw = evaluate_arm(
        model, data, chunk_size=2, bootstrap_samples=24, seed=97, progress=progress.append
    )
    assert progress == [2, 4]
    assert report["groups"] == 4
    assert report["split"] == "original_diagnostic_validation"
    assert report["dense_identity_control"] == (condition == "dense")
    assert raw["effective_flip_counts"].shape == (4, 1, 6, 5, 2, 2)
    torch.testing.assert_close(raw["natural"]["group_indices"], torch.arange(4), atol=0, rtol=0)
    torch.testing.assert_close(
        raw["bootstrap_group_indices"],
        torch.randint(4, (24, 4), generator=torch.Generator().manual_seed(97)),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(model.state_dict(), frozen, atol=0, rtol=0)
    torch.testing.assert_close(data["validation"], data_before["validation"], atol=0, rtol=0)
    assert set(data["metadata"]) == set(data_before["metadata"])
    for name, value in data["metadata"].items():
        if isinstance(value, Tensor):
            torch.testing.assert_close(value, data_before["metadata"][name], atol=0, rtol=0)
        else:
            assert value == data_before["metadata"][name]
    assert torch.equal(torch.get_rng_state(), rng)
    refreshed = refresh_evaluation(model, data, ("validation",))
    candidate = copy.deepcopy(model).double().eval()
    matched, _, _ = matched_batch(candidate, refreshed, "validation", tuple(range(4)))
    _, expected = evaluate_responses(candidate, matched, bootstrap_samples=24, seed=97)
    for name, value in expected.items():
        torch.testing.assert_close(raw["natural"][name], value, atol=1e-13, rtol=1e-8)
    paired = raw["paired_group_numerators"]["normalized_response_error"]
    torch.testing.assert_close(
        paired,
        raw["replayed_factual_gates"]["response_error_energy"]
        - raw["natural"]["response_error_energy"],
        atol=0,
        rtol=0,
    )
    if condition == "dense":
        assert not bool(raw["effective_flip_counts"].any())
        assert not bool(paired.any())
    with pytest.raises(ValueError, match="validation-only"):
        evaluate_arm(model, cached, chunk_size=2, bootstrap_samples=24, seed=97)
