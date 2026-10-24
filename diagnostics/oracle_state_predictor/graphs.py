"""True-coordinate routing diagnostics for the isolated predictor comparison.

Support is the union of residual-inclusive local one-step routing paths over
held-out inputs and both predictor calls. Paths are never composed across the
autoregressive forecast, and routing support is not a causal guarantee.
"""

from typing import Literal

import torch
from torch import Tensor

from diagnostics.oracle_state_predictor.data import OracleStateData
from diagnostics.oracle_state_predictor.run import Config, forecast_metrics
from scjepa.data.simplified_gaussian import graphs as reference_graphs
from scjepa.models.spartan import GraphPenalties, Spartan, SpartanOutput
from scjepa.training.rng import preserve_rng

Condition = Literal["dense", "baseline", "full"]


def build_predictor(config: Config, condition: Condition) -> Spartan:
    """Keep one architecture while selecting dense or learned routing gates."""
    if condition not in {"dense", "baseline", "full"}:
        raise ValueError("condition must be dense, baseline, or full")
    return Spartan(
        state_dim=6,
        param_dim=6,
        action_dim=16,
        history_len=1,
        num_layers=2,
        embed_dim=config.embed_dim,
        mlp_hidden_size=config.mlp_hidden,
        mlp_num_layers=3,
        temperature=1.0,
        dense=condition == "dense",
        parameter_gates_open=condition == "baseline",
    )


def rollout_graphs(
    model: Spartan, initial_states: Tensor, parameters: Tensor, actions: Tensor
) -> tuple[Tensor, Tensor, GraphPenalties, tuple[SpartanOutput, ...]]:
    """Reuse raw theta and feed the attached first prediction into horizon two."""
    if actions.ndim != 3 or actions.shape[1:] != (2, 16):
        raise ValueError("actions must have shape (batch * views, 2, 16)")
    state = initial_states
    outputs: list[SpartanOutput] = []
    for horizon in range(2):
        result = model(state, parameters, actions[:, horizon : horizon + 1])
        outputs.append(result)
        state = result.prediction.unsqueeze(1)
    penalties = GraphPenalties(
        torch.stack([output.sparsity.state for output in outputs]).mean(),
        torch.stack([output.sparsity.action for output in outputs]).mean(),
        torch.stack([output.sparsity.parameter for output in outputs]).mean(),
    )
    return (
        torch.stack([output.prediction for output in outputs], dim=1),
        torch.stack([output.logit_penalty for output in outputs]).mean(),
        penalties,
        tuple(outputs),
    )


@torch.no_grad()
def evaluate_graphs(
    model: Spartan,
    data: OracleStateData,
    batch_size: int,
    device: torch.device,
    logit_weight: float,
) -> tuple[dict[str, float], dict[str, object]]:
    """Evaluate fixed true targets and unaligned support, preserving model/RNG state."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    condition: Condition = (
        "dense" if model.dense else "baseline" if model.parameter_gates_open else "full"
    )
    flags = [(module, module.training) for module in model.modules()]
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    term_sums = {name: 0.0 for name in ("logit", "state", "action", "parameter")}
    truth = reference_graphs()
    references = {
        "state": truth.state.any(dim=1),
        "action": truth.action.any(dim=1),
        "parameter": truth.parameter,
    }
    counts = {
        name: torch.zeros_like(value, dtype=torch.float64) for name, value in references.items()
    }
    predictor_calls = 0
    try:
        with preserve_rng(device):
            model.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                batch = data.batch(indices, device)
                predicted, logit, penalties, outputs = rollout_graphs(
                    model, batch.initial_states, batch.parameters, batch.actions
                )
                predictions.append(predicted.cpu())
                targets.append(batch.targets.cpu())
                term_sums["logit"] += float(logit) * len(indices)
                for name, penalty in zip(("state", "action", "parameter"), penalties, strict=True):
                    term_sums[name] += float(penalty) * len(indices)
                for output in outputs:
                    blocks = output.graphs.aggregate_lags()
                    for name, block in zip(("state", "action", "parameter"), blocks, strict=True):
                        counts[name] += (block > 0).sum(dim=0).cpu().double()
                    predictor_calls += len(output.prediction)
    finally:
        for module, training in flags:
            module.training = training
    metrics = forecast_metrics(torch.cat(predictions), torch.cat(targets))
    metrics["logit"] = term_sums["logit"] / len(data)
    metrics["base_loss"] = metrics["prediction_mse"] + logit_weight * metrics["logit"]
    for name in ("state", "action", "parameter"):
        metrics[f"{name}_sparsity"] = term_sums[name] / len(data)
    metrics["sparsity"] = (
        0.0
        if condition == "dense"
        else metrics["state_sparsity"]
        + metrics["action_sparsity"]
        + (metrics["parameter_sparsity"] if condition == "full" else 0.0)
    )
    graph_payload: dict[str, object] = {}
    for name, reference in references.items():
        frequencies = counts[name] / predictor_calls
        support = frequencies > 0
        metrics[f"{name}_edges"] = float(support.sum())
        metrics[f"{name}_true_positives"] = float((support & reference).sum())
        metrics[f"{name}_false_positives"] = float((support & ~reference).sum())
        metrics[f"{name}_false_negatives"] = float((~support & reference).sum())
        metrics[f"{name}_shd"] = float((support != reference).sum())
        graph_payload[name] = {
            "frequencies": [[float(value) for value in row] for row in frequencies],
            "support": [[bool(value) for value in row] for row in support],
            "reference": [[bool(value) for value in row] for row in reference],
        }
    payload: dict[str, object] = {
        "schema": "oracle-state-predictor-routing-v1",
        "condition": condition,
        "coordinate_system": "true state, observed action, raw true parameter; no matching",
        "orientation": "receiver/output rows, source/input columns; any-lag support with L=1",
        "support_threshold": 0.0,
        "support_semantics": (
            "union of residual-inclusive local one-step routing paths across all evaluated "
            "samples and both predictor calls; no composition across rollout steps; "
            "input-dependent routing support is not a causal guarantee"
        ),
        "parameter_groups": len(data),
        "views_per_group": data.views,
        "predictor_calls": predictor_calls,
        "graphs": graph_payload,
    }
    return metrics, payload
