"""Finite-lag coordinate graphs and sparse information-flow contracts."""

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


def test_closed_edges_have_no_attention_control_bypass(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_graph_and_attention_logits_are_independent_of_input_values() -> None:
    model = tiny(history=2, actions=DA).eval()
    first = model(torch.randn(2, 2, DZ), torch.randn(2, DP), torch.randn(2, 2, DA))
    changed = model(100 * torch.randn(2, 2, DZ), torch.randn(2, DP), torch.randn(2, 2, DA))
    torch.testing.assert_close(first.path_matrix, changed.path_matrix)
    torch.testing.assert_close(first.logit_penalty, changed.logit_penalty)
    torch.testing.assert_close(first.mean_abs_logit, changed.mean_abs_logit)


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


def test_gradients_reach_states_actions_parameters_and_structural_logits() -> None:
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


def test_sparse_path_penalty_trains_straight_through_gates() -> None:
    model = tiny(history=2, actions=DA)
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
    with pytest.raises(ValueError, match="temperature"):
        Spartan(state_dim=DZ, param_dim=DP, temperature=0)
    with pytest.raises(ValueError, match="path_matrix"):
        tiny().extract_graphs(torch.ones(2, 3, 3))
