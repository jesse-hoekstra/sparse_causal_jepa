"""Finite-lag coordinate graphs and local sparse-attention contracts."""

from typing import cast

import pytest
import torch
from torch import Tensor

import scjepa.models.spartan as spartan_module
from scjepa.models.spartan import Spartan, SpartanLayer

DZ, DP, DA, DIM = 3, 5, 2, 16


def tiny(
    history: int = 1,
    actions: int = 0,
    dense: bool = False,
    identity: bool = False,
    layers: int = 2,
    parameter_gates_open: bool = False,
) -> Spartan:
    """Construct a small model with independently chosen coordinate dimensions."""
    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    return Spartan(
        state_dim=DZ,
        param_dim=DP,
        action_dim=actions,
        history_len=history,
        num_layers=layers,
        embed_dim=DIM,
        mlp_hidden_size=DIM,
        mlp_num_layers=2,
        dense=dense,
        identity=identity,
        parameter_gates_open=parameter_gates_open,
    )


@pytest.mark.parametrize("history", [1, 3])
@pytest.mark.parametrize("action_dim", [0, DA])
def test_output_shapes_and_independent_dimensions(history: int, action_dim: int) -> None:
    model = tiny(history=history, actions=action_dim)
    states, parameters = torch.randn(2, history, DZ), torch.randn(2, DP)
    actions = torch.randn(2, history, action_dim) if action_dim else None
    output = model(states, parameters, actions)
    tokens = history * (DZ + action_dim) + DP
    assert output.prediction.shape == (2, DZ)
    assert output.path_matrix.shape == (2, tokens, tokens)
    assert output.graphs.state.shape == (2, DZ, history, DZ)
    assert output.graphs.action.shape == (2, DZ, history, action_dim)
    assert output.graphs.parameter.shape == (2, DZ, DP)
    assert all(penalty.ndim == 0 for penalty in output.sparsity)
    assert all(torch.isfinite(penalty) for penalty in output.sparsity)
    if not action_dim:
        assert output.sparsity.action.item() == 0.0


def test_graph_extraction_uses_lag_major_sources_and_latest_state_receivers() -> None:
    model = tiny(history=3, actions=DA)
    paths = torch.arange(2 * model.num_tokens**2).reshape(2, model.num_tokens, model.num_tokens)
    graphs = model.extract_graphs(paths)
    for coordinate in range(DZ):
        row = 2 * DZ + coordinate
        for lag in range(3):
            torch.testing.assert_close(
                graphs.state[:, coordinate, lag], paths[:, row, lag * DZ : (lag + 1) * DZ]
            )
            start = 3 * DZ + lag * DA
            torch.testing.assert_close(
                graphs.action[:, coordinate, lag], paths[:, row, start : start + DA]
            )
        torch.testing.assert_close(graphs.parameter[:, coordinate], paths[:, row, -DP:])
    aggregate = graphs.aggregate_lags()
    torch.testing.assert_close(aggregate.state, graphs.state.amax(dim=2))
    torch.testing.assert_close(aggregate.action, graphs.action.amax(dim=2))
    torch.testing.assert_close(aggregate.parameter, graphs.parameter)


def test_no_action_graph_aggregation_keeps_empty_block() -> None:
    model = tiny(history=3)
    output = model(torch.randn(2, 3, DZ), torch.randn(2, DP))
    assert output.graphs.aggregate_lags().action.shape == (2, DZ, 0)


def test_path_counts_include_every_residual() -> None:
    model = tiny(history=3, actions=DA, dense=True, layers=3)
    output = model(torch.randn(2, 3, DZ), torch.randn(2, DP), torch.randn(2, 3, DA))
    single_layer = torch.ones(model.num_tokens, model.num_tokens) + torch.eye(model.num_tokens)
    expected = (single_layer @ single_layer @ single_layer).expand(2, -1, -1)
    torch.testing.assert_close(output.path_matrix, expected)
    torch.testing.assert_close(output.sparsity.state, output.graphs.state.mean())
    torch.testing.assert_close(output.sparsity.action, output.graphs.action.mean())
    torch.testing.assert_close(output.sparsity.parameter, output.graphs.parameter.mean())


def test_path_counts_keep_float32_under_autocast() -> None:
    model = tiny(history=3, actions=DA, dense=True, layers=4)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(torch.randn(2, 3, DZ), torch.randn(2, DP), torch.randn(2, 3, DA))
    assert output.path_matrix.dtype == torch.float32
    assert torch.isfinite(output.path_matrix).all()
    single = torch.ones(model.num_tokens, model.num_tokens) + torch.eye(model.num_tokens)
    torch.testing.assert_close(
        output.path_matrix, (single @ single @ single @ single).expand(2, -1, -1)
    )


def test_identity_mode_depends_only_on_same_latest_state_coordinate() -> None:
    model = tiny(history=3, actions=DA, identity=True).eval()
    states = torch.randn(2, 3, DZ)
    parameters, actions = torch.randn(2, DP), torch.randn(2, 3, DA)
    original = model(states, parameters, actions)
    torch.testing.assert_close(original.path_matrix, torch.eye(model.num_tokens).expand(2, -1, -1))
    assert torch.count_nonzero(original.graphs.action).item() == 0
    assert torch.count_nonzero(original.graphs.parameter).item() == 0
    assert torch.count_nonzero(original.graphs.state[:, :, :-1]).item() == 0
    torch.testing.assert_close(original.graphs.state[:, :, -1], torch.eye(DZ).expand(2, -1, -1))
    changed = states.clone()
    changed[:, :-1] += 10
    changed[:, -1, 1:] -= 10
    perturbed = model(changed, parameters + 5, actions - 7)
    torch.testing.assert_close(original.prediction[:, 0], perturbed.prediction[:, 0])


def test_sparse_path_products_account_for_indirect_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny(history=2, actions=DA, layers=2).eval()
    first = torch.zeros(1, model.num_tokens, model.num_tokens)
    second = first.clone()
    # Latest state 0 receives old action 1, which receives parameter 2 in the
    # preceding layer. The graph must include the resulting indirect path.
    receiver, intermediary, source = DZ, 2 * DZ + 1, model.num_tokens - DP + 2
    first[0, intermediary, source] = 1
    second[0, receiver, intermediary] = 1
    prescribed = iter([first, second])

    def fixed_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        return next(prescribed).to(logits).expand_as(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", fixed_adjacency)
    output = model(torch.randn(1, 2, DZ), torch.randn(1, DP), torch.randn(1, 2, DA))
    eye = torch.eye(model.num_tokens)
    torch.testing.assert_close(output.path_matrix, (second + eye) @ (first + eye))
    assert output.graphs.parameter[0, 0, 2].item() == 1
    assert output.graphs.action[0, 0, 0, 1].item() == 1


def test_fixed_closed_routes_block_unrouted_values(monkeypatch: pytest.MonkeyPatch) -> None:
    model = tiny(history=2, actions=DA, layers=1).eval()
    prescribed = torch.zeros(1, model.num_tokens, model.num_tokens)
    prescribed[0, DZ, 1] = 1  # latest output 0 may read oldest state 1

    def fixed_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        return prescribed.to(logits).expand_as(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", fixed_adjacency)
    states = torch.randn(1, 2, DZ, requires_grad=True)
    parameters = torch.randn(1, DP, requires_grad=True)
    actions = torch.randn(1, 2, DA, requires_grad=True)
    output = model(states, parameters, actions)
    state_grad, parameter_grad, action_grad = torch.autograd.grad(
        output.prediction[0, 0], (states, parameters, actions)
    )
    allowed = output.graphs.state[0, 0].bool()
    assert torch.count_nonzero(state_grad[0][~allowed]).item() == 0
    assert torch.count_nonzero(parameter_grad).item() == 0
    assert torch.count_nonzero(action_grad).item() == 0
    changed_states = states.detach().clone()
    changed_states[0][~allowed] += 100
    changed = model(changed_states, parameters.detach() - 100, actions.detach() + 100)
    torch.testing.assert_close(output.prediction[:, 0], changed.prediction[:, 0])


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("action_dim", [0, DA])
def test_baseline_opens_only_parameter_routes_in_every_layer(
    monkeypatch: pytest.MonkeyPatch, training: bool, action_dim: int
) -> None:
    baseline = tiny(history=3, actions=action_dim, layers=3, parameter_gates_open=True)
    full = tiny(history=3, actions=action_dim, layers=3)
    baseline.train(training)
    full.train(training)
    sampled_modes: list[bool] = []
    adjacencies: list[Tensor] = []

    def closed_adjacency(logits: Tensor, _temperature: float, is_training: bool) -> Tensor:
        sampled_modes.append(is_training)
        return torch.zeros_like(logits)

    def capture_layer(
        _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ) -> None:
        adjacencies.append(cast(tuple[Tensor, ...], output)[1].detach())

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", closed_adjacency)
    handles = [layer.register_forward_hook(capture_layer) for layer in baseline.layers]
    states, parameters = torch.randn(2, 3, DZ), torch.randn(2, DP)
    actions = torch.randn(2, 3, action_dim) if action_dim else None
    output = baseline(states, parameters, actions)
    for handle in handles:
        handle.remove()
    closed = full(states, parameters, actions)
    expected = torch.zeros(2, baseline.num_tokens, baseline.num_tokens)
    expected[:, 2 * DZ : 3 * DZ, -DP:] = 1
    assert len(adjacencies) == 3
    assert sampled_modes == [training] * 6
    for adjacency in adjacencies:
        torch.testing.assert_close(adjacency, expected, rtol=0, atol=0)
    identity = torch.eye(baseline.num_tokens).expand(2, -1, -1)
    torch.testing.assert_close(output.path_matrix, (expected + identity).matrix_power(3))
    assert (output.graphs.parameter > 0).all()
    assert not closed.graphs.parameter.any()
    torch.testing.assert_close(output.graphs.state, closed.graphs.state)
    torch.testing.assert_close(output.graphs.action, closed.graphs.action)


def test_forced_parameter_gates_have_no_straight_through_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny(history=2, actions=DA, parameter_gates_open=True)
    sampled: list[Tensor] = []

    def closed_straight_through(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        adjacency = logits - logits.detach()
        adjacency.retain_grad()
        sampled.append(adjacency)
        return adjacency

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", closed_straight_through)
    output = model(torch.randn(2, 2, DZ), torch.randn(2, DP), torch.randn(2, 2, DA))
    output.path_matrix.sum().backward()  # pyright: ignore[reportUnknownMemberType]
    forced = torch.zeros(model.num_tokens, model.num_tokens, dtype=torch.bool)
    forced[DZ : 2 * DZ, -DP:] = True
    assert len(sampled) == len(model.layers)
    for adjacency in sampled:
        assert adjacency.grad is not None
        assert torch.isfinite(adjacency.grad).all()
        assert not adjacency.grad[:, forced].any()
        assert adjacency.grad[:, ~forced].abs().sum() > 0


def test_open_parameter_routes_still_learn_attention_and_parameter_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny(history=2, actions=DA, layers=1, parameter_gates_open=True)

    def closed_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        return torch.zeros_like(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", closed_adjacency)
    parameters = torch.randn(2, DP, requires_grad=True)
    output = model(torch.randn(2, 2, DZ), parameters, torch.randn(2, 2, DA))
    output.prediction.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert parameters.grad is not None
    assert parameters.grad.abs().sum() > 0
    layer = model.layers[0]
    assert isinstance(layer, SpartanLayer)
    for projection in (layer.project_q, layer.project_k, layer.project_v):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0


def test_fixed_parameter_gate_diagnostics_report_probability_one_and_zero_entropy() -> None:
    model = tiny(history=2, actions=DA, parameter_gates_open=True)
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, SpartanLayer)
            layer.project_q.weight.zero_()
    output = model(torch.randn(2, 2, DZ), torch.randn(2, DP), torch.randn(2, 2, DA))
    fixed_fraction = DZ * DP / model.num_tokens**2
    torch.testing.assert_close(
        output.mean_gate_probability, torch.tensor(0.5 + 0.5 * fixed_fraction)
    )
    torch.testing.assert_close(output.gate_entropy, (1 - fixed_fraction) * torch.tensor(2.0).log())


def test_local_gates_change_with_content_within_the_same_batch() -> None:
    layer = SpartanLayer(dim=2, mlp_hidden_size=4, mlp_num_layers=2, temperature=1).eval()
    with torch.no_grad():
        layer.project_q.weight.copy_(torch.eye(2))
        layer.project_k.weight.copy_(torch.eye(2))
    # Same query token, but an opposite/same-direction key changes the route.
    tokens = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]], [[1.0, -1.0], [1.0, -1.0]]])
    _, adjacency, *_ = layer(tokens)
    assert adjacency[0, 0, 1].item() == 0
    assert adjacency[1, 0, 1].item() == 1


def test_paper_uses_unscaled_gate_logits_and_scaled_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = SpartanLayer(dim=DIM, mlp_hidden_size=DIM, mlp_num_layers=2, temperature=1)
    tokens = torch.randn(2, 3, DIM)
    normalized = layer.norm(tokens)
    raw = layer.project_q(normalized) @ layer.project_k(normalized).transpose(-1, -2)
    captured: list[Tensor] = []

    def open_adjacency(logits: Tensor, _temperature: float, _training: bool) -> Tensor:
        captured.append(logits)
        return torch.ones_like(logits)

    monkeypatch.setattr(spartan_module, "_sample_hard_adjacency", open_adjacency)
    updated, _, penalty, _, probability, _ = layer(tokens)
    expected_attention = (raw * layer.scale).softmax(dim=-1) @ layer.project_v(normalized)
    torch.testing.assert_close(captured[0], raw)
    torch.testing.assert_close(updated, layer.mlp(tokens + expected_attention))
    torch.testing.assert_close(penalty, (raw.exp() + (-raw).exp()).mean())
    torch.testing.assert_close(probability, raw.sigmoid().mean())


def test_sparsity_blocks_can_be_weighted_independently() -> None:
    model = tiny(history=2, actions=DA)
    paths = torch.rand(2, model.num_tokens, model.num_tokens, requires_grad=True)
    penalties = model.extract_graphs(paths).penalties()
    loss = 2 * penalties.state + 3 * penalties.action + 0 * penalties.parameter
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    assert paths.grad is not None
    gradient = model.extract_graphs(paths.grad)
    assert torch.all(gradient.state > 0)
    assert torch.all(gradient.action > 0)
    assert torch.count_nonzero(gradient.parameter).item() == 0


def test_dense_mode_deterministic_without_gate_sampling() -> None:
    model = tiny(dense=True)
    states, parameters = torch.randn(2, 1, DZ), torch.randn(2, DP)
    first, second = model(states, parameters), model(states, parameters)
    torch.testing.assert_close(first.prediction, second.prediction)
    assert torch.all(first.path_matrix > 0)


def test_dense_and_sparse_predictors_have_identical_initial_parameters() -> None:
    sparse_state = tiny(actions=DA).state_dict()
    for model in (tiny(actions=DA, dense=True), tiny(actions=DA, parameter_gates_open=True)):
        assert sparse_state.keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, sparse_state[name], rtol=0, atol=0)


def test_sparse_evaluation_deterministic_training_resamples() -> None:
    model = tiny()
    states, parameters = torch.randn(2, 1, DZ), torch.randn(2, DP)
    model.eval()
    first, second = model(states, parameters), model(states, parameters)
    torch.testing.assert_close(first.prediction, second.prediction)
    torch.testing.assert_close(first.path_matrix, second.path_matrix)
    model.train()
    first, second = model(states, parameters), model(states, parameters)
    assert not torch.equal(first.path_matrix, second.path_matrix)
    torch.testing.assert_close(first.path_matrix, first.path_matrix.round())


def test_gradients_reach_states_actions_parameters_and_attention_logits() -> None:
    model = tiny(history=2, actions=DA, dense=True)
    states = torch.randn(2, 2, DZ, requires_grad=True)
    parameters = torch.randn(2, DP, requires_grad=True)
    actions = torch.randn(2, 2, DA, requires_grad=True)
    output = model(states, parameters, actions)
    (output.prediction.square().mean() + 1e-3 * output.logit_penalty).backward()  # pyright: ignore[reportUnknownMemberType]
    for value in (states, parameters, actions):
        assert value.grad is not None
        assert value.grad.abs().sum() > 0
    layer = model.layers[0]
    assert isinstance(layer, SpartanLayer)
    gradient = layer.project_q.weight.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("parameter_gates_open", [False, True])
def test_sparse_path_penalty_trains_straight_through_gates(parameter_gates_open: bool) -> None:
    model = tiny(history=2, actions=DA, parameter_gates_open=parameter_gates_open)
    output = model(torch.randn(2, 2, DZ), torch.randn(2, DP), torch.randn(2, 2, DA))
    (output.sparsity.state + output.sparsity.action + output.sparsity.parameter).backward()  # pyright: ignore[reportUnknownMemberType]
    for layer in model.layers:
        assert isinstance(layer, SpartanLayer)
        for projection in (layer.project_q, layer.project_k):
            gradient = projection.weight.grad
            assert gradient is not None
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0


def test_fully_closed_rows_remain_finite_in_forward_and_backward() -> None:
    model = tiny(identity=True)
    states, parameters = torch.randn(2, 1, DZ, requires_grad=True), torch.randn(2, DP)
    output = model(states, parameters)
    output.prediction.square().mean().backward()  # pyright: ignore[reportUnknownMemberType]
    assert torch.isfinite(output.prediction).all()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_logit_penalty_and_diagnostics_finite_at_large_logits() -> None:
    model = tiny(dense=True)
    states, parameters = torch.randn(2, 1, DZ), torch.randn(2, DP)
    regular = model(states, parameters)
    assert regular.logit_penalty.item() >= 2
    assert 0 <= regular.mean_gate_probability.item() <= 1
    assert 0 <= regular.gate_entropy.item() <= 0.6932
    assert regular.mean_abs_logit.item() >= 0
    with torch.no_grad():
        for layer in model.layers:
            assert isinstance(layer, SpartanLayer)
            layer.project_q.weight.mul_(1e4)
    large = model(states, parameters)
    assert torch.isfinite(large.prediction).all()
    assert torch.isfinite(large.logit_penalty)
    assert large.logit_penalty > regular.logit_penalty


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("state_dim", 0), ("param_dim", 0), ("history_len", 0), ("action_dim", -1), ("num_layers", 0)],
)
def test_invalid_dimensions(keyword: str, value: int) -> None:
    config = {"state_dim": DZ, "param_dim": DP, keyword: value}
    with pytest.raises(ValueError, match="positive|nonnegative"):
        Spartan(
            state_dim=config["state_dim"],
            param_dim=config["param_dim"],
            history_len=config.get("history_len", 1),
            action_dim=config.get("action_dim", 0),
            num_layers=config.get("num_layers", 2),
        )


def test_input_guards() -> None:
    states, parameters = torch.randn(2, 1, DZ), torch.randn(2, DP)
    with pytest.raises(ValueError, match="expected states"):
        tiny()(states[:, 0], parameters)
    with pytest.raises(ValueError, match="expected parameters"):
        tiny()(states, parameters[:, :2])
    with pytest.raises(ValueError, match="expected actions"):
        tiny(actions=DA)(states, parameters)
    with pytest.raises(ValueError, match="expected actions"):
        tiny(actions=DA)(states, parameters, torch.randn(2, 2, DA))
    with pytest.raises(ValueError, match="actions must be None"):
        tiny()(states, parameters, torch.randn(2, 1, 0))
    with pytest.raises(ValueError, match="nonempty"):
        tiny()(states[:0], parameters[:0])
    with pytest.raises(ValueError, match="mutually exclusive"):
        tiny(dense=True, identity=True)
    with pytest.raises(ValueError, match="identity"):
        tiny(identity=True, parameter_gates_open=True)
    with pytest.raises(ValueError, match="temperature"):
        Spartan(state_dim=DZ, param_dim=DP, temperature=0)
    with pytest.raises(ValueError, match="path_matrix"):
        tiny().extract_graphs(torch.ones(2, 3, 3))
