"""Held-out identifiability and trajectory diagnostics for state-to-state SCJEPA.

The GECO calibration quantity mirrors the training predictive objective::

    constraint_loss = L_TF + lambda_rollout_t2 * L_AR2
                      + lambda_logit * L_logit.

For deterministic calibration, ``L_AR2`` is averaged exhaustively over every
valid two-step offset. This is the exact uniform-anchor expectation estimated by
the eight-window training sample, without evaluation sampling noise.

A separate no-gradient open-loop rollout reports approximate trajectory
agreement on a fixed held-out sample. It is monitoring only: it never enters
``constraint_loss`` and does not prove population observational equivalence.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.graph import (
    gt_causal_graph_from_contacts,
    read_learned_graph,
    structural_hamming_distance,
)
from scjepa.eval.observational_equivalence import oe_worst_step_nrmse, summarize_oe
from scjepa.eval.parameters import nonlinear_mcc
from scjepa.losses import aligned_mse, rollout_t2_endpoint_mse
from scjepa.models.state_to_state import (
    StateToStateModel,
    TransitionOutput,
    num_valid_rollout_t2_offsets,
)


class IdentifiabilityReport(NamedTuple):
    """Periodic metrics plus final-only recovery diagnostics."""

    metrics: dict[str, float]
    diagnostics: dict[str, float]
    learned_coordinates: Tensor
    true_parameters: Tensor
    recovery_matrix: Tensor


def _weighted_mean(values: list[Tensor], weights: list[int]) -> float:
    """Average per-batch scalar values without overweighting a short last batch."""
    if len(values) != len(weights) or not values:
        raise ValueError("weighted mean requires one positive weight per value")
    denominator = float(sum(weights))
    numerator = sum(float(value) * weight for value, weight in zip(values, weights, strict=True))
    return numerator / denominator


@torch.random.fork_rng(devices=[])  # pyright: ignore[reportUnknownMemberType]
@torch.no_grad()
def evaluate_identifiability(
    model: StateToStateModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 32,
    max_batches: int | None = None,
    device: str = "cpu",
    context_len: int | None = None,
    lambda_logit: float = 0.0,
    lambda_rollout_t2: float = 0.0,
    num_rollout_t2_anchors: int = 8,
    rollout_t2_horizon: int = 2,
    oe_eval_horizon: int | None = None,
    oe_tolerance_nrmse: float = 0.10,
    oe_coordinate_std: Tensor | Sequence[float] | None = None,
) -> IdentifiabilityReport:
    """Evaluate prediction, graph, recovery, and sampled trajectory agreement.

    ``L_AR2`` is exhaustive over valid offsets during evaluation. Since the
    training loss is a mean over a uniform sample without replacement, this is
    its deterministic held-out expectation and therefore the appropriate
    quantity for fresh tau calibration.
    """
    if rollout_t2_horizon != 2:
        raise ValueError(f"rollout_t2_horizon must equal 2, got {rollout_t2_horizon}")
    if not math.isfinite(lambda_rollout_t2) or lambda_rollout_t2 < 0:
        raise ValueError("lambda_rollout_t2 must be finite and non-negative")
    if num_rollout_t2_anchors < 1:
        raise ValueError("num_rollout_t2_anchors must be positive")
    if oe_eval_horizon is not None and oe_eval_horizon < 1:
        raise ValueError("oe_eval_horizon must be positive or None")
    if not math.isfinite(oe_tolerance_nrmse) or oe_tolerance_nrmse < 0:
        raise ValueError("oe_tolerance_nrmse must be finite and non-negative")
    coordinate_std = None
    if oe_eval_horizon is not None:
        if oe_coordinate_std is None:
            raise ValueError("oe_coordinate_std is required when OE evaluation is enabled")
        coordinate_std = torch.as_tensor(oe_coordinate_std, dtype=torch.float32, device=device)

    model = model.to(device).eval()
    # The loader has a private generator and no shuffle so evaluation neither
    # depends on nor advances the training sampling stream.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    num_slots: int | None = None
    teacher_forcing_losses: list[Tensor] = []
    rollout_t2_losses: list[Tensor] = []
    logit_penalties: list[Tensor] = []
    learned_graphs: list[Tensor] = []
    true_graphs: list[Tensor] = []
    learned_coordinates: list[Tensor] = []
    true_parameters: list[Tensor] = []
    path_density: list[Tensor] = []
    path_density_full: list[Tensor] = []
    mean_abs_logits: list[Tensor] = []
    mean_gate_probabilities: list[Tensor] = []
    gate_entropies: list[Tensor] = []
    oe_errors: list[Tensor] = []
    batch_weights: list[int] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        states = batch["states"].to(device)
        length = states.shape[1]
        tpar = context_len if context_len is not None else length - 1
        output: TransitionOutput = model(states, context_len=tpar)
        batch_weights.append(states.shape[0])
        num_slots = output.prediction.shape[1]
        teacher_forcing_losses.append(aligned_mse(output.prediction, output.target).cpu())

        if lambda_rollout_t2 > 0:
            valid = num_valid_rollout_t2_offsets(length, tpar)
            if num_rollout_t2_anchors > valid:
                raise ValueError(
                    f"num_rollout_t2_anchors={num_rollout_t2_anchors} exceeds {valid} valid offsets"
                )
            offsets = torch.arange(valid, device=states.device).expand(states.shape[0], -1)
            endpoint, endpoint_target, _ = model.rollout_t2_from_offsets(
                states,
                tpar,
                output.causal_params,
                offsets,
            )
            rollout_t2_losses.append(rollout_t2_endpoint_mse(endpoint, endpoint_target).cpu())

        if oe_eval_horizon is not None:
            assert coordinate_std is not None
            prediction, target = model.rollout_for_evaluation(
                states,
                tpar,
                oe_eval_horizon,
                output.causal_params,
            )
            oe_errors.append(oe_worst_step_nrmse(prediction, target, coordinate_std).cpu())

        logit_penalties.append(output.logit_penalty.cpu())
        contacts = batch["contacts"].to(device)
        per_transition = contacts[:, tpar - 1 :].flatten(0, 1).unsqueeze(1)
        graph_gt = gt_causal_graph_from_contacts(per_transition)
        graph_learned = read_learned_graph(output.path_matrix, num_slots)
        learned_graphs.append(graph_learned.cpu())
        true_graphs.append(graph_gt.cpu())
        path_density.append((output.path_matrix[:, :num_slots] >= 0.5).float().mean().cpu())
        path_density_full.append((output.path_matrix >= 0.5).float().mean().cpu())
        mean_abs_logits.append(output.mean_abs_logit.cpu())
        mean_gate_probabilities.append(output.mean_gate_probability.cpu())
        gate_entropies.append(output.gate_entropy.cpu())
        learned_coordinates.append(output.causal_params.flatten(1).cpu())
        true_parameters.append(batch["params"].flatten(1).cpu())

    if num_slots is None:
        raise ValueError("dataset yielded no batches")
    episode_learned = torch.cat(learned_coordinates)
    episode_true = torch.cat(true_parameters)
    if episode_learned.shape[1] != num_slots or episode_true.shape[1] != num_slots:
        raise ValueError(
            "mass recovery requires one scalar parameter slot per object; "
            f"got {tuple(episode_learned.shape)} and {tuple(episode_true.shape)}"
        )
    recovery = nonlinear_mcc(episode_learned, episode_true)
    learned_graph = torch.cat(learned_graphs)
    true_graph = torch.cat(true_graphs)

    teacher_forcing = _weighted_mean(teacher_forcing_losses, batch_weights)
    raw_rollout_t2 = _weighted_mean(rollout_t2_losses, batch_weights) if rollout_t2_losses else 0.0
    weighted_rollout_t2 = lambda_rollout_t2 * raw_rollout_t2
    logit_penalty = _weighted_mean(logit_penalties, batch_weights)
    weighted_logit = lambda_logit * logit_penalty
    constraint_loss = teacher_forcing + weighted_rollout_t2 + weighted_logit
    metrics = {
        # ``pred_loss`` remains the historical teacher-forcing/MCC sweep key.
        "pred_loss": teacher_forcing,
        "loss_rollout_t2_raw": raw_rollout_t2,
        "loss_rollout_t2_weighted": weighted_rollout_t2,
        "mean_abs_logit": _weighted_mean(mean_abs_logits, batch_weights),
        "gate_entropy": _weighted_mean(gate_entropies, batch_weights),
        "constraint_loss": constraint_loss,
        "shd": structural_hamming_distance(learned_graph, true_graph).item(),
        "mcc": recovery.score.item(),
        "path_density": _weighted_mean(path_density, batch_weights),
    }
    if oe_errors:
        oe = summarize_oe(torch.cat(oe_errors), oe_tolerance_nrmse)
        suffix = f"k{oe_eval_horizon}"
        metrics |= {
            f"oe_sample_satisfaction_{suffix}": oe.satisfaction,
            f"oe_{suffix}_worst_step_nrmse_p50": oe.p50,
            f"oe_{suffix}_worst_step_nrmse_p95": oe.p95,
        }
    diagnostics = {
        "logit_penalty": logit_penalty,
        "logit_weighted": weighted_logit,
        "logit_fraction": weighted_logit / max(constraint_loss, 1e-12),
        "mean_gate_probability": _weighted_mean(mean_gate_probabilities, batch_weights),
        "path_density_full": _weighted_mean(path_density_full, batch_weights),
        "num_samples": float(recovery.num_samples),
    }
    return IdentifiabilityReport(
        metrics=metrics,
        diagnostics=diagnostics,
        learned_coordinates=episode_learned,
        true_parameters=episode_true,
        recovery_matrix=recovery.matrix,
    )


__all__ = ["IdentifiabilityReport", "evaluate_identifiability"]
