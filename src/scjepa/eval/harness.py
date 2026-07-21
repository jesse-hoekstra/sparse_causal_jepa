"""Identifiability evaluation for globally indexed latent coordinates.

Consumes any JEPA variant with the ``JepaOutput`` contract plus a dataset whose
items carry the bounce-style ground truth (``params``, ``contacts``). Returns
the compact scalar metrics used during training and the full held-out recovery
matrix used for final analysis.

State rows are object-aligned in the GT-state regime. Parameter coordinates are
*not* assumed to have the same order as masses: one global Hungarian mapping is
learned on a held-out alignment split, frozen on the final score split, and
also used to align parameter-graph columns before SHD is computed.
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.graph import (
    align_parameter_columns,
    gt_graphs_from_contacts,
    read_learned_graphs,
    structural_hamming_distance,
)
from scjepa.eval.parameters import one_to_one_recovery
from scjepa.losses import (
    prediction_mse,
    resolve_constraint_normalization,
    resolve_prediction_matching,
)
from scjepa.models.jepa import JepaOutput, SCJepa
from scjepa.models.state_jepa import StateJepa


class IdentifiabilityReport(NamedTuple):
    """Periodic metrics plus final-only recovery diagnostics.

    ``metrics`` is deliberately compact because every entry becomes a W&B
    curve. ``diagnostics`` contains useful final scalars that should not create
    redundant periodic curves. Recovery matrices use
    ``[learned_coordinate, true_mass]`` order; ``target_to_learned[i]`` is the
    one globally assigned learned coordinate for mass ``i``.
    """

    metrics: dict[str, float]
    diagnostics: dict[str, float]
    learned_coordinates: Tensor
    true_parameters: Tensor
    recovery_matrix: Tensor
    recovery_linear_matrix: Tensor
    target_to_learned: Tensor


def _weighted_mean(values: list[Tensor], weights: list[int]) -> float:
    """Average per-batch scalar values without overweighting the last batch."""
    if len(values) != len(weights) or not values:
        raise ValueError("weighted mean requires one positive weight per value")
    denominator = float(sum(weights))
    numerator = sum(float(value) * weight for value, weight in zip(values, weights, strict=True))
    return numerator / denominator


@torch.random.fork_rng(devices=[])  # pyright: ignore[reportUnknownMemberType]
@torch.no_grad()
def evaluate_identifiability(
    model: SCJepa | StateJepa,
    dataset: Dataset[dict[str, Tensor]],
    input_key: str = "states",
    batch_size: int = 32,
    max_batches: int | None = None,
    device: str = "cpu",
    context_len: int | None = None,
    rollout_horizon: int | None = None,
    lambda_logit: float = 0.0,
    prediction_matching: str = "auto",
    constraint_normalization: str = "auto",
) -> IdentifiabilityReport:
    """Evaluate SHD / MCC / prediction error over the dataset.

    Args:
        model: A JEPA variant (eval mode is set here).
        dataset: Items with ``input_key`` plus ``params`` and ``contacts``.
        input_key: "states" (GT-embedding regime) or "frames" (only with
            aligned slots — see module docstring).
        batch_size: Eval batch size.
        max_batches: Optional cap for quick runs.
        device: Device string.
        context_len: Context window Th (must match training); None = single
            transition per episode.
        rollout_horizon: D16 autoregressive chain length (must match training
            — pred_loss is then measured under the SAME rollout objective the
            constraint uses); None = one chain over all K transitions.
        lambda_logit: Training's attention-logit weight, included in the
            reported constraint just as it is in the training dual.
        prediction_matching: ``auto`` selects aligned MSE for StateJepa's
            persistent object rows. Visual slots are refused until a persistent
            trajectory-level alignment exists.
        constraint_normalization: ``auto`` uses raw prediction MSE for literal
            GT states and D17 target-variance normalization for learned target
            spaces. It must match training when calibrating tau.
    """
    model = model.to(device).eval()
    if isinstance(model, SCJepa):
        raise ValueError(
            "vision-slot identifiability requires one persistent trajectory-level "
            "slot-to-object alignment; independent frame matching is insufficient"
        )
    resolved_matching = resolve_prediction_matching(
        prediction_matching,
        object_aligned=True,
    )
    resolved_normalization = resolve_constraint_normalization(
        constraint_normalization,
        gt_states=model.gt_states,
    )
    # DataLoader draws a worker/base seed even with shuffle=False. Give it a
    # private generator so periodic evaluation cannot advance training RNG.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    num_slots: int | None = None
    pred_losses: list[Tensor] = []
    pred_losses_normalized: list[Tensor] = []
    target_vars: list[Tensor] = []
    logit_penalties: list[Tensor] = []
    shd_state: list[Tensor] = []
    learned_param_graphs: list[Tensor] = []
    true_param_graphs: list[Tensor] = []
    learned_coordinates: list[Tensor] = []
    true_parameters: list[Tensor] = []
    path_density: list[Tensor] = []
    path_density_full: list[Tensor] = []
    mean_abs_logits: list[Tensor] = []
    mean_gate_probabilities: list[Tensor] = []
    gate_entropies: list[Tensor] = []
    batch_weights: list[int] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs = batch[input_key].to(device)
        output: JepaOutput = model(inputs, context_len=context_len, rollout_horizon=rollout_horizon)
        batch_weights.append(inputs.shape[0])
        num_slots = output.prediction.shape[1]
        batch_pred = prediction_mse(output.prediction, output.target_slots, resolved_matching).cpu()
        pred_losses.append(batch_pred)
        # D17: per-batch target variance, same formula as the trainer's dual
        # measurement (mean per-dim variance of the target slots, floored).
        target_var = (
            output.target_slots.reshape(-1, output.target_slots.shape[-1])
            .var(dim=0)
            .mean()
            .clamp_min(1e-6)
            .cpu()
        )
        target_vars.append(target_var)
        pred_losses_normalized.append(batch_pred / target_var)
        logit_penalties.append(output.logit_penalty.cpu())
        # D15: with a sliding window, predictions cover transitions th-1 .. L-2;
        # build one gt graph per transition and flatten to match (B*K, ...).
        length = inputs.shape[1]
        th = context_len if context_len is not None else length - 1
        contacts = batch["contacts"].to(device)  # (B, L-1, N, N)
        per_transition = contacts[:, th - 1 :]  # (B, K, N, N)
        flat_contacts = per_transition.flatten(0, 1).unsqueeze(1)  # (B*K, 1, N, N)
        state_gt, param_gt = gt_graphs_from_contacts(flat_contacts)
        state_learned, param_learned = read_learned_graphs(output.path_matrix, num_slots)
        shd_state.append(structural_hamming_distance(state_learned, state_gt).cpu())
        learned_param_graphs.append(param_learned.cpu())
        true_param_graphs.append(param_gt.cpu())
        # Primary density mirrors the optimized state-output rows. Parameter
        # rows after the last layer are not decoded, so report their inclusion
        # only under the explicit full-token diagnostic.
        path_density.append((output.path_matrix[:, :num_slots] >= 0.5).float().mean().cpu())
        path_density_full.append((output.path_matrix >= 0.5).float().mean().cpu())
        mean_abs_logits.append(output.mean_abs_logit.cpu())
        mean_gate_probabilities.append(output.mean_gate_probability.cpu())
        gate_entropies.append(output.gate_entropy.cpu())
        # One row per episode. Coordinate order is persistent but deliberately
        # not assumed to match the physical mass order.
        learned_coordinates.append(output.causal_params.flatten(1).cpu())
        true_parameters.append(batch["params"].flatten(1).cpu())

    if num_slots is None:
        raise ValueError("dataset yielded no batches")
    episode_learned = torch.cat(learned_coordinates)
    episode_true = torch.cat(true_parameters)
    if episode_learned.shape[1] != num_slots or episode_true.shape[1] != num_slots:
        raise ValueError(
            "strict graph-aligned mass recovery requires exactly one global latent "
            f"coordinate per object; got learned {tuple(episode_learned.shape)} and "
            f"true {tuple(episode_true.shape)} for {num_slots} objects"
        )
    recovery = one_to_one_recovery(episode_learned, episode_true)

    # The state rows already follow GT object order. Reorder only the learned
    # parameter columns so column i denotes true mass i, using the mapping
    # selected globally from a disjoint recovery alignment fold.
    learned_param = torch.cat(learned_param_graphs)
    true_param = torch.cat(true_param_graphs)
    aligned_param = align_parameter_columns(learned_param, recovery.target_to_learned)

    pred_loss = _weighted_mean(pred_losses, batch_weights)
    pred_loss_normalized = _weighted_mean(pred_losses_normalized, batch_weights)
    logit_penalty = _weighted_mean(logit_penalties, batch_weights)
    constraint_prediction = pred_loss if resolved_normalization == "raw" else pred_loss_normalized
    weighted_logit = lambda_logit * logit_penalty
    constraint_loss = constraint_prediction + weighted_logit
    metrics = {
        "pred_loss": pred_loss,
        "mean_abs_logit": _weighted_mean(mean_abs_logits, batch_weights),
        "gate_entropy": _weighted_mean(gate_entropies, batch_weights),
        # The exact tau quantity, on the same configured scale as training.
        "constraint_loss": constraint_loss,
        "shd_state": _weighted_mean(shd_state, batch_weights),
        "shd_param_aligned": structural_hamming_distance(aligned_param, true_param).item(),
        # The only periodic mass-recovery curve: strict nonlinear one-to-one
        # recovery under one global permutation.
        "mass_mcc": recovery.nonlinear_score.item(),
        "path_density": _weighted_mean(path_density, batch_weights),
    }
    diagnostics = {
        "logit_penalty": logit_penalty,
        "logit_weighted": weighted_logit,
        "logit_fraction": weighted_logit / max(constraint_loss, 1e-12),
        "mean_gate_probability": _weighted_mean(mean_gate_probabilities, batch_weights),
        "pred_loss_normalized": pred_loss_normalized,
        "target_var": _weighted_mean(target_vars, batch_weights),
        "mass_mcc_linear": recovery.linear_score.item(),
        "path_density_full": _weighted_mean(path_density_full, batch_weights),
        "num_samples": float(recovery.num_samples),
    }
    return IdentifiabilityReport(
        metrics=metrics,
        diagnostics=diagnostics,
        learned_coordinates=episode_learned,
        true_parameters=episode_true,
        recovery_matrix=recovery.nonlinear_matrix,
        recovery_linear_matrix=recovery.linear_matrix,
        target_to_learned=recovery.target_to_learned,
    )


__all__ = ["IdentifiabilityReport", "evaluate_identifiability"]
