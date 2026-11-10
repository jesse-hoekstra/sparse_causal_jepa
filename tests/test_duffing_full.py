from typing import cast

import torch

from diagnostics.coupled_duffing.evaluate import graph_metrics
from diagnostics.coupled_duffing.model import LaggedPredictor, build_model
from diagnostics.coupled_duffing.run import objective_for


def test_parameter_sparsity_changes_only_gates_and_closes_code_paths() -> None:
    torch.set_num_threads(1)
    baseline = build_model(9, 6, sparse=True, parameter_encoder="temporal-cnn").eval()
    full = build_model(
        9, 6, sparse=True, parameter_encoder="temporal-cnn", parameter_sparsity=True
    ).eval()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[name], rtol=0, atol=0)
    predictor = cast(LaggedPredictor, full.predictor)
    states, actions = torch.randn(8, 2, 2), torch.randn(8, 2, 1)
    codes = torch.randn(8, 4, requires_grad=True)
    torch.testing.assert_close(
        baseline.predictor(states, codes, actions).prediction,
        predictor(states, codes, actions).prediction,
        rtol=0,
        atol=0,
    )
    assert predictor.parameter_logits is not None
    with torch.no_grad():
        predictor.parameter_logits[0, 2:] = -3
    predictor(states, codes, actions).prediction[:, 0].sum().backward()  # pyright: ignore[reportUnknownMemberType]
    assert codes.grad is not None
    assert torch.count_nonzero(codes.grad[:, 2:]) == 0
    assert bool((codes.grad[:, :2].abs().sum(0) > 0).all())
    objective = objective_for(
        "full", 0.4, torch.device("cpu"), state_regularization="projected-aligned"
    )
    assert objective.config.parameter_sparsity_weight == 1
    assert objective.geco is not None


def test_graph_alignment_allows_pair_mixing_and_counts_missing_and_false_edges() -> None:
    generator = torch.Generator().manual_seed(19)
    states = torch.randn(256, 3, 5, 2, generator=generator, dtype=torch.float64)
    theta = torch.randn(256, 4, generator=generator, dtype=torch.float64)
    mixing = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 2.0, 1.0], [0.0, 0.0, -1.0, 2.0]],
        dtype=torch.float64,
    )
    order = [2, 0, 3, 1]
    codes = (theta @ mixing)[:, order, None].transpose(1, 2).expand(-1, 3, -1)
    calibration = {"states": states.flip(-1), "codes": codes}
    labels = {"states": states, "theta": theta}
    truth = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool)
    graphs = {
        "state": torch.ones(2, 2, dtype=torch.bool),
        "action": torch.ones(2, 1, dtype=torch.bool),
        "parameter": truth.flip(0)[:, order],
    }
    scores = graph_metrics(graphs, calibration, labels)
    for name in ("state", "action", "parameter_code"):
        assert scores[f"masks/{name}_aligned_shd"] == 0
    graphs["parameter"][0, 0] = False
    graphs["parameter"][0, 1] = True
    scores = graph_metrics(graphs, calibration, labels)
    assert scores["masks/parameter_code_aligned_shd"] == 2
    assert scores["masks/parameter_code_aligned_missing_true_edges"] == 1
    assert scores["masks/parameter_code_aligned_false_edges"] == 1
