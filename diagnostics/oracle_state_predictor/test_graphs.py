"""Routing support and attached-forecast contracts for the isolated comparison."""

import json
import pickle
import random
from copy import deepcopy
from math import isclose
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from diagnostics.oracle_state_predictor.data import build_data
from diagnostics.oracle_state_predictor.graphs import (
    Condition,
    build_predictor,
    evaluate_graphs,
    rollout_graphs,
)
from diagnostics.oracle_state_predictor.run import Config
from diagnostics.oracle_state_predictor.run import build_predictor as build_dense_predictor
from scjepa.data.simplified_gaussian import graphs as reference_graphs
from scjepa.models.spartan import GraphBlocks, Spartan, SpartanLayer, SpartanOutput


@pytest.mark.parametrize("condition", ["dense", "baseline", "full"])
def test_conditions_share_parameter_initialization_and_the_existing_architecture(
    condition: Condition,
) -> None:
    """Only gate policy changes; all learned tensor initializations stay paired."""
    config = Config(embed_dim=8, mlp_hidden=8)
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    original = build_dense_predictor(config)
    torch.manual_seed(17)  # pyright: ignore[reportUnknownMemberType]
    model = build_predictor(config, condition)
    torch.testing.assert_close(model.state_dict(), original.state_dict(), rtol=0, atol=0)
    assert (model.state_dim, model.param_dim, model.action_dim, model.history_len) == (6, 6, 16, 1)
    assert model.num_tokens == 28
    assert len(model.layers) == 2
    assert model.dense == (condition == "dense")
    assert all(layer.dense == (condition == "dense") for layer in model.layers)
    assert model.parameter_gates_open == (condition == "baseline")
    assert not model.action_gates_open
    if condition == "baseline":
        mask = model.parameter_gate_mask
        assert mask is not None
        assert bool(mask[:6, 22:].all())
        assert int(mask.sum()) == 36
    else:
        assert model.parameter_gate_mask is None


def test_dense_support_and_path_penalties_include_every_layer_and_residual() -> None:
    """Two dense layers yield (I+J)^2, while scientific reference edges stay sparse."""
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "dense")
    data = build_data("test", 3, views=2)
    metrics, payload = evaluate_graphs(model, data, 2, torch.device("cpu"), 1e-4)
    truth = reference_graphs()
    assert (int(truth.state.sum()), int(truth.action.sum()), int(truth.parameter.sum())) == (
        18,
        96,
        12,
    )
    for name, edges, true_edges in (("state", 36, 18), ("action", 96, 96), ("parameter", 36, 12)):
        assert metrics[f"{name}_edges"] == edges
        assert metrics[f"{name}_true_positives"] == true_edges
        assert metrics[f"{name}_false_positives"] == edges - true_edges
        assert metrics[f"{name}_false_negatives"] == 0
        assert metrics[f"{name}_shd"] == edges - true_edges
    assert isclose(metrics["state_sparsity"], 30 + 1 / 6, rel_tol=1e-6)
    assert metrics["action_sparsity"] == metrics["parameter_sparsity"] == 30
    assert metrics["sparsity"] == 0
    assert payload["predictor_calls"] == 3 * 2 * 2
    blocks = cast(dict[str, dict[str, object]], payload["graphs"])
    for name in ("state", "action", "parameter"):
        frequencies = torch.tensor(cast(list[list[float]], blocks[name]["frequencies"]))
        torch.testing.assert_close(frequencies, torch.ones_like(frequencies), rtol=0, atol=0)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize("condition", ["baseline", "full"])
def test_closed_learned_gates_keep_residuals_and_only_baseline_parameter_overrides(
    condition: Condition,
) -> None:
    """Zero logits close learned evaluation gates while forced parameter paths survive."""
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), condition)
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, SpartanLayer)
            layer.project_q.weight.zero_()
            layer.project_k.weight.zero_()
    data = build_data("test", 2, views=2)
    metrics, _ = evaluate_graphs(model, data, 2, torch.device("cpu"), 0.0)
    assert metrics["state_edges"] == 6
    assert metrics["state_shd"] == metrics["state_false_negatives"] == 12
    assert metrics["action_edges"] == 0
    assert metrics["action_shd"] == 96
    assert isclose(metrics["state_sparsity"], 1 / 6, rel_tol=1e-6)
    assert metrics["action_sparsity"] == 0
    if condition == "baseline":
        assert metrics["parameter_edges"] == 36
        assert metrics["parameter_shd"] == metrics["parameter_false_positives"] == 24
        assert metrics["parameter_false_negatives"] == 0
        assert metrics["parameter_sparsity"] == 2
    else:
        assert metrics["parameter_edges"] == 0
        assert metrics["parameter_shd"] == metrics["parameter_false_negatives"] == 12
        assert metrics["parameter_sparsity"] == 0
    assert isclose(metrics["sparsity"], 1 / 6, rel_tol=1e-6)
    assert metrics["base_loss"] == metrics["prediction_mse"]


class _ScriptedPredictor(Spartan):
    """Expose two disjoint local state edges on the same subset of examples."""

    def __init__(self) -> None:
        super().__init__(6, 6, 16, embed_dim=8, mlp_hidden_size=8)
        self.calls = 0

    def forward(
        self, states: Tensor, parameters: Tensor, actions: Tensor | None = None
    ) -> SpartanOutput:
        """Use input-dependent support whose expected frequencies are easy to count."""
        assert parameters.shape[-1] == 6
        assert actions is not None
        assert actions.shape[-1] == 16
        horizon = self.calls % 2
        self.calls += 1
        prediction = states[:, 0]
        batch = len(prediction)
        selected = prediction[:, 0] > 0
        state = torch.eye(6, device=states.device).expand(batch, -1, -1).clone().unsqueeze(2)
        state[selected, horizon + 1, 0, horizon] = 1
        parameter = states.new_zeros(batch, 6, 6)
        parameter[selected, 0, 0] = 1
        blocks = GraphBlocks(state, states.new_zeros(batch, 6, 1, 16), parameter)
        zero = states.new_zeros(())
        return SpartanOutput(
            prediction,
            states.new_zeros(batch, 28, 28),
            blocks,
            blocks.penalties(),
            prediction[:, 0].square().mean(),
            zero,
            zero,
            zero,
        )


def test_graph_union_is_local_and_frequencies_weight_samples_not_batches() -> None:
    """Union z0→z1 and z1→z2 without inventing a composed z0→z2 routing edge."""
    data = build_data("test", 3, views=2)
    selected_fraction = float((data.initial_states[:, :, 0, 0] > 0).double().mean())
    assert 0 < selected_fraction < 1
    model = _ScriptedPredictor()
    metrics, payload = evaluate_graphs(model, data, 2, torch.device("cpu"), 1e-4)
    whole_metrics, whole_payload = evaluate_graphs(model, data, 3, torch.device("cpu"), 1e-4)
    for key, value in metrics.items():
        assert isclose(value, whole_metrics[key], rel_tol=1e-6, abs_tol=1e-8)
    assert payload == whole_payload
    blocks = cast(dict[str, dict[str, object]], payload["graphs"])
    frequency = torch.tensor(cast(list[list[float]], blocks["state"]["frequencies"]))
    parameter_frequency = torch.tensor(cast(list[list[float]], blocks["parameter"]["frequencies"]))
    assert isclose(float(frequency[1, 0]), selected_fraction / 2, rel_tol=1e-6)
    assert isclose(float(frequency[2, 1]), selected_fraction / 2, rel_tol=1e-6)
    assert frequency[2, 0] == 0
    assert isclose(float(parameter_frequency[0, 0]), selected_fraction, rel_tol=1e-6)
    assert metrics["state_edges"] == 8
    assert metrics["parameter_edges"] == 1
    assert isclose(
        metrics["sparsity"],
        metrics["state_sparsity"] + metrics["action_sparsity"] + metrics["parameter_sparsity"],
    )


def test_graph_rollout_preserves_second_horizon_gradient_and_reuses_raw_theta() -> None:
    """Graph diagnostics must not detach the forecast used by the training objective."""
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "dense")
    data = build_data("train", 2, views=2)
    batch = data.batch(torch.arange(2), "cpu")
    received: list[tuple[Tensor, Tensor, Tensor]] = []

    def before(_module: nn.Module, inputs: tuple[object, ...]) -> None:
        received.append(cast(tuple[Tensor, Tensor, Tensor], inputs))

    handle = model.register_forward_pre_hook(before)
    prediction, logit, penalties, outputs = rollout_graphs(
        model, batch.initial_states, batch.parameters, batch.actions
    )
    handle.remove()
    assert len(outputs) == len(received) == 2
    outputs[0].prediction.retain_grad()
    for horizon, (_, theta, actions) in enumerate(received):
        assert theta is batch.parameters
        torch.testing.assert_close(actions[:, 0], batch.actions[:, horizon])
    torch.testing.assert_close(received[1][0][:, 0], outputs[0].prediction)
    torch.testing.assert_close(logit, torch.stack([item.logit_penalty for item in outputs]).mean())
    for index, penalty in enumerate(penalties):
        torch.testing.assert_close(
            penalty, torch.stack([item.sparsity[index] for item in outputs]).mean()
        )
    prediction[:, 1].square().sum().backward()  # pyright: ignore[reportUnknownMemberType]
    gradient = outputs[0].prediction.grad
    assert gradient is not None
    assert bool(gradient.abs().sum() > 0)


@pytest.mark.parametrize("fail", [False, True])
def test_graph_evaluation_restores_modes_tensors_and_rng_even_on_failure(
    monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    """Graph reporting cannot perturb the next optimizer step or its random streams."""
    model = build_predictor(Config(embed_dim=8, mlp_hidden=8), "full").train()
    model.layers[0].eval()
    data = build_data("validation", 2, views=2)
    original_forward = model.forward
    calls = 0

    def forward(states: Tensor, parameters: Tensor, actions: Tensor) -> SpartanOutput:
        nonlocal calls
        calls += 1
        assert not any(module.training for module in model.modules())
        random.random()
        np.random.random()  # noqa: NPY002 - deliberately test global RNG preservation.
        torch.rand(3)
        if fail and calls == 2:
            raise RuntimeError("intentional graph evaluation failure")
        return original_forward(states, parameters, actions)

    monkeypatch.setattr(model, "forward", forward)
    modes = tuple(module.training for module in model.modules())
    weights = deepcopy(model.state_dict())
    host_rng = random.getstate()
    numpy_rng = pickle.dumps(np.random.get_state())  # noqa: NPY002
    torch_rng = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="intentional graph evaluation failure"):
            evaluate_graphs(model, data, 2, torch.device("cpu"), 1e-4)
    else:
        evaluate_graphs(model, data, 2, torch.device("cpu"), 1e-4)
    assert tuple(module.training for module in model.modules()) == modes
    torch.testing.assert_close(model.state_dict(), weights, rtol=0, atol=0)
    assert random.getstate() == host_rng
    assert pickle.dumps(np.random.get_state()) == numpy_rng  # noqa: NPY002
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
