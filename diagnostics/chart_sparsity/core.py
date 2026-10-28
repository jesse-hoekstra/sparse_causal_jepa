"""Fixed state charts and independently auditable predictor measurements.

The supplied true states and parameters are explicit diagnostic controls. Only
state coordinates change between charts; the physical DGP, observed actions and
parameter coordinates remain fixed. Routing metrics describe local value paths,
not a certificate of the global functional graph.
"""

import math
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from diagnostics.oracle_state_predictor.data import OracleStateData
from diagnostics.oracle_state_predictor.graphs import rollout_graphs
from diagnostics.oracle_state_predictor.run import forecast_metrics
from scjepa.data.datasets import canonical_fingerprint
from scjepa.data.simplified_gaussian import graphs as physical_graphs
from scjepa.models.spartan import Spartan, SpartanLayer
from scjepa.training.rng import preserve_rng

Chart = Literal["physical", "mixed"]
BLOCKS = ("state", "action", "parameter")


def chart_matrix(
    chart: Chart,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return I6 or blockdiag(H,H), H=I3-(2/3)J3, without drawing randomness."""
    if chart not in {"physical", "mixed"}:
        raise ValueError("chart must be physical or mixed")
    if not dtype.is_floating_point:
        raise ValueError("chart matrices require a floating-point dtype")
    if chart == "physical":
        return torch.eye(6, dtype=dtype, device=device)
    block = torch.eye(3, dtype=torch.float64, device=device) - torch.full(
        (3, 3), 2 / 3, dtype=torch.float64, device=device
    )
    zero = torch.zeros_like(block)
    return torch.cat((torch.cat((block, zero), dim=1), torch.cat((zero, block), dim=1))).to(
        dtype=dtype
    )


def chart_metadata(chart: Chart) -> dict[str, object]:
    """Describe the fixed coordinate change and analytic support references."""
    matrix = chart_matrix(chart)
    return {
        "chart": chart,
        "matrix": [[float(value) for value in row] for row in matrix],
        "state_map": "z = s @ Q.T; column-vector convention z = Q s",
        "parameter_map": "q = theta, supplied raw true parameters",
        "action_map": "unchanged observed mixed actions",
        "matrix_definition": "I6" if chart == "physical" else "blockdiag(H,H); H=I3-(2/3)J3",
        "population_state_mcc": 1.0 if chart == "physical" else 2 / 3,
        "true_global_edge_counts": {
            name: int(mask.sum()) for name, mask in true_graphs(chart).items()
        },
        "scope": (
            "Fixed, information-preserving oracle coordinates; no learned state recovery. "
            "Global support references are not training budgets or path-loss targets."
        ),
    }


def transform_data(data: OracleStateData, chart: Chart) -> OracleStateData:
    """Transform original physical inputs/targets, preserving actions and parameters.

    Call once on the original physical data. Float64 matrix multiplication
    precedes conversion back to the source dtype. No source tensor is modified;
    unchanged action/parameter tensors and episode identities are shared read-only.
    """
    matrix = chart_matrix(chart)
    if data.initial_states.shape[-2:] != (1, 6) or data.targets.shape[-2:] != (2, 6):
        raise ValueError("chart control requires six states and two forecast targets")
    if data.initial_states.device.type != "cpu" or data.targets.device.type != "cpu":
        raise ValueError("chart data must be stored on the CPU")
    if not data.initial_states.is_floating_point() or not data.targets.is_floating_point():
        raise ValueError("chart data must use floating-point states")
    return replace(
        data,
        initial_states=(data.initial_states.double() @ matrix.T).to(data.initial_states.dtype),
        targets=(data.targets.double() @ matrix.T).to(data.targets.dtype),
        fingerprint=canonical_fingerprint(
            {
                "schema": "chart-sparsity-data-v1",
                "physical_data_fingerprint": data.fingerprint,
                "chart": chart_metadata(chart),
            }
        ),
    )


def true_graphs(chart: Chart) -> dict[str, Tensor]:
    """Return 2D output-row support masks in the declared fixed coordinates."""
    if chart not in {"physical", "mixed"}:
        raise ValueError("chart must be physical or mixed")
    masks = physical_graphs()
    return {
        "state": masks.state.any(dim=1),
        "action": masks.action.any(dim=1),
        "parameter": masks.parameter if chart == "physical" else torch.ones(6, 6, dtype=torch.bool),
    }


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics, graph summaries and CPU samples for independent audits."""

    metrics: dict[str, float]
    graphs: dict[str, object]
    samples: dict[str, Tensor]


class _ProjectionCapture:
    """Capture actual Q/K outputs, preserving the production forward computation."""

    def __init__(self) -> None:
        self.queries: list[Tensor] = []
        self.keys: list[Tensor] = []

    def query(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.queries.append(output.detach())

    def key(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.keys.append(output.detach())

    def clear(self) -> None:
        self.queries.clear()
        self.keys.clear()

    def per_example_logit(self, layers: int) -> Tensor:
        """Reproduce the actual float32 logit penalty for each sample and horizon."""
        if len(self.queries) != 2 * layers or len(self.keys) != 2 * layers:
            raise RuntimeError("expected one Q/K capture per layer and forecast horizon")
        penalties: list[Tensor] = []
        for query, key in zip(self.queries, self.keys, strict=True):
            dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
            logits = torch.einsum("bid,bjd->bij", query.to(dtype), key.to(dtype)).float()
            magnitude = logits.abs()
            core = magnitude.clamp(max=30)
            tail = (magnitude - 30).clamp(min=0)
            penalty = core.exp() + (-core).exp() + core.exp() * tail
            penalties.append(penalty.double().mean(dim=(1, 2)))
        stacked = torch.stack(penalties).reshape(2, layers, -1)
        return stacked.mean(dim=1).T


@contextmanager
def _capture_projections(model: Spartan) -> Generator[_ProjectionCapture, None, None]:
    captured = _ProjectionCapture()
    handles: list[RemovableHandle] = []
    try:
        for raw_layer in model.layers:
            layer = cast(SpartanLayer, raw_layer)
            handles.append(layer.project_q.register_forward_hook(captured.query))
            handles.append(layer.project_k.register_forward_hook(captured.key))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _graph_summary(
    samples: dict[str, Tensor], chart: Chart, metrics: dict[str, float]
) -> dict[str, object]:
    graphs: dict[str, object] = {}
    for name, reference in true_graphs(chart).items():
        frequency = samples[f"group_{name}_reachability"].mean(dim=(0, 1))
        support = frequency > 0
        metrics[f"{name}_edges"] = float(support.sum())
        metrics[f"mean_local_{name}_edges"] = float(frequency.sum())
        metrics[f"{name}_true_positives"] = float((support & reference).sum())
        metrics[f"{name}_false_positives"] = float((support & ~reference).sum())
        metrics[f"{name}_false_negatives"] = float((~support & reference).sum())
        metrics[f"{name}_shd"] = float((support != reference).sum())
        graphs[name] = {
            "frequencies": [[float(value) for value in row] for row in frequency],
            "support": [[bool(value) for value in row] for row in support],
            "reference": [[bool(value) for value in row] for row in reference],
        }
    return graphs


@torch.no_grad()
def evaluate(
    model: Spartan,
    data: OracleStateData,
    batch_size: int,
    device: torch.device,
    logit_weight: float,
    chart: Chart,
) -> EvaluationResult:
    """Measure all groups with deterministic gates, restoring modes, hooks and RNG.

    Samples retain group/horizon errors, actual per-example logit penalties,
    path means and local binary reachability averaged over views. Group is the
    independent resampling unit. Graphs concern each predictor call separately;
    they are never composed across the autoregressive forecast horizons.
    """
    if batch_size < 1 or len(data) < 1:
        raise ValueError("batch size and number of groups must be positive")
    if not math.isfinite(logit_weight) or logit_weight < 0:
        raise ValueError("logit weight must be finite and nonnegative")
    if (model.state_dim, model.action_dim, model.param_dim, model.history_len) != (6, 16, 6, 1):
        raise ValueError(
            "chart control requires dimensions state/action/parameter/history=6/16/6/1"
        )
    true_graphs(chart)
    flags = [(module, module.training) for module in model.modules()]
    parts: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "predictions",
            "targets",
            "group_horizon_mse",
            "group_horizon_logit",
            *(f"group_{name}_path_means" for name in BLOCKS),
            *(f"group_{name}_path_sums" for name in BLOCKS),
            *(f"group_{name}_reachability" for name in BLOCKS),
        )
    }
    try:
        with preserve_rng(device), _capture_projections(model) as captured:
            model.eval()
            for start in range(0, len(data), batch_size):
                indices = torch.arange(start, min(start + batch_size, len(data)))
                batch = data.batch(indices, device)
                captured.clear()
                prediction, scalar_logit, _, outputs = rollout_graphs(
                    model, batch.initial_states, batch.parameters, batch.actions
                )
                per_example_logit = captured.per_example_logit(len(model.layers))
                if not math.isclose(
                    float(per_example_logit.mean()), float(scalar_logit), rel_tol=2e-6, abs_tol=1e-6
                ):
                    raise RuntimeError("per-example logit reconstruction differs from model output")
                predicted = prediction.cpu().reshape(batch.groups, batch.views, 2, 6)
                target = batch.targets.cpu().reshape_as(predicted)
                parts["predictions"].append(predicted)
                parts["targets"].append(target)
                parts["group_horizon_mse"].append(
                    (predicted.double() - target.double()).square().mean(dim=(1, 3))
                )
                parts["group_horizon_logit"].append(
                    per_example_logit.reshape(batch.groups, batch.views, 2).mean(dim=1).cpu()
                )
                horizon_paths: dict[str, list[Tensor]] = {name: [] for name in BLOCKS}
                horizon_reach: dict[str, list[Tensor]] = {name: [] for name in BLOCKS}
                for output in outputs:
                    for name, block in zip(BLOCKS, output.graphs.aggregate_lags(), strict=True):
                        block = block.double().reshape(batch.groups, batch.views, 6, -1)
                        horizon_paths[name].append(block.mean(dim=(1, 2, 3)).cpu())
                        horizon_reach[name].append((block > 0).double().mean(dim=1).cpu())
                for name in BLOCKS:
                    path_means = torch.stack(horizon_paths[name], dim=1)
                    parts[f"group_{name}_path_means"].append(path_means)
                    parts[f"group_{name}_path_sums"].append(
                        path_means * (96 if name == "action" else 36)
                    )
                    parts[f"group_{name}_reachability"].append(
                        torch.stack(horizon_reach[name], dim=1)
                    )
    finally:
        for module, training in flags:
            module.training = training
    samples = {name: torch.cat(values, dim=0) for name, values in parts.items()}
    samples["group_ids"] = data.episode_ids.clone()
    if not all(bool(torch.isfinite(value).all()) for value in samples.values()):
        raise FloatingPointError("nonfinite chart evaluation measurements")
    metrics = forecast_metrics(
        samples["predictions"].flatten(0, 1), samples["targets"].flatten(0, 1)
    )
    metrics["logit"] = float(samples["group_horizon_logit"].mean())
    metrics["base_loss"] = metrics["prediction_mse"] + logit_weight * metrics["logit"]
    for name in BLOCKS:
        metrics[f"{name}_sparsity"] = float(samples[f"group_{name}_path_means"].mean())
    metrics["sparsity"] = (
        0.0
        if model.dense
        else metrics["state_sparsity"]
        + metrics["action_sparsity"]
        + (0.0 if model.parameter_gates_open else metrics["parameter_sparsity"])
    )
    graphs: dict[str, object] = {
        "schema": "chart-sparsity-routing-v1",
        "chart": chart_metadata(chart),
        "groups": len(data),
        "views": data.views,
        "predictor_calls": len(data) * data.views * 2,
        "support_threshold": 0.0,
        "orientation": "output/receiver rows, input/source columns",
        "semantics": (
            "Union of local binary one-step reachability across all inputs and both horizons; "
            "no graph composition across horizons; not a global functional-support certificate."
        ),
        "graphs": _graph_summary(samples, chart, metrics),
    }
    return EvaluationResult(metrics, graphs, samples)
