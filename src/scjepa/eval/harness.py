"""Identifiability evaluation for the state-to-state regime (experiments.pdf §6.1.3 / §6.7).

Consumes an ``StateToStateModel`` plus a bounce dataset whose items carry the
ground truth (``params``, ``contacts``). Two headline numbers, one from each
source paper: ``mcc`` is Baumgartner App. F.1's recovery score (D27, see
``scjepa.eval.parameters``) and ``shd`` is SPARTAN's Structural Hamming
Distance between the learned graph and the ground-truth causal graph (D28, see
``scjepa.eval.graph``). NEITHER is meaningful alone — see D28.

The constraint is reported exactly as the dual sees it (raw units):
``constraint_loss = pred_loss + lambda_roll * rollout_loss + lambda_logit *
logit_penalty``, the scalarised hybrid §4.3 bound. Calibrate τ on THAT
quantity, and pass the SAME ``rollout_len``/``lambda_roll`` used in training —
a mismatch on either side silently invalidates τ. Evaluation runs in eval mode,
so gates are the deterministic Eq. 34 thresholds; the rollout is anchored at
the fixed t = Tpar-1 in both modes, so the reported constraint is reproducible.

Everything is in tracked-object order by construction (ζ = id, Eq. 132): no
permutation of parameter coordinates or graph axes is fitted anywhere.
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.graph import (
    gt_causal_graph_from_contacts,
    read_learned_graph,
    structural_hamming_distance,
)
from scjepa.eval.parameters import nonlinear_mcc
from scjepa.losses import aligned_mse, rollout_weights, weighted_rollout_mse
from scjepa.models.state_to_state import StateToStateModel, TransitionOutput


class IdentifiabilityReport(NamedTuple):
    """Periodic metrics plus final-only recovery diagnostics.

    ``metrics`` is deliberately compact because every entry becomes a W&B
    curve. ``recovery_matrix[i, j]`` is the held-out R² of learned coordinate
    ``j`` predicting true mass ``i`` — App. F.1's I x J orientation.
    """

    metrics: dict[str, float]
    diagnostics: dict[str, float]
    learned_coordinates: Tensor
    true_parameters: Tensor
    recovery_matrix: Tensor


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
    model: StateToStateModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 32,
    max_batches: int | None = None,
    device: str = "cpu",
    context_len: int | None = None,
    lambda_logit: float = 0.0,
    rollout_len: int | None = None,
    lambda_roll: float = 0.0,
) -> IdentifiabilityReport:
    """Evaluate prediction / constraint / SHD / MCC over the dataset.

    Args:
        model: The state-to-state model (eval mode is set here).
        dataset: Items with ``states``, ``params`` and ``contacts``.
        batch_size: Eval batch size.
        max_batches: Optional cap for quick runs.
        device: Device string.
        context_len: Tpar — must match training.
        lambda_logit: Training's attention-logit weight, included in the
            reported constraint exactly as in the training dual.
        rollout_len: K for the hybrid rollout branch — must match training,
            since τ is calibrated on the constraint this function reports.
            None disables the branch (pure teacher-forced constraint).
        lambda_roll: Training's rollout weight inside the scalarised bound.
    """
    model = model.to(device).eval()
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
    rollout_losses: list[Tensor] = []
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
    batch_weights: list[int] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        states = batch["states"].to(device)
        output: TransitionOutput = model(states, context_len=context_len, rollout_len=rollout_len)
        batch_weights.append(states.shape[0])
        num_slots = output.prediction.shape[1]
        pred_losses.append(aligned_mse(output.prediction, output.target).cpu())
        if output.rollout_prediction is not None and output.rollout_target is not None:
            weights = rollout_weights(
                output.rollout_prediction.shape[1], device=output.rollout_prediction.device
            )
            rollout_losses.append(
                weighted_rollout_mse(
                    output.rollout_prediction, output.rollout_target, weights
                ).cpu()
            )
        logit_penalties.append(output.logit_penalty.cpu())
        # One ground-truth local graph per predicted transition (Eqs. 8/9),
        # flattened to match the model's (B·K, ...) transition rows.
        length = states.shape[1]
        tpar = context_len if context_len is not None else length - 1
        contacts = batch["contacts"].to(device)  # (B, T-1, N, N)
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
        # One row per episode, matching App. F.1's "encoding all trajectories
        # in a validation dataset into the learnt parameters".
        learned_coordinates.append(output.causal_params.flatten(1).cpu())
        true_parameters.append(batch["params"].flatten(1).cpu())

    if num_slots is None:
        raise ValueError("dataset yielded no batches")
    episode_learned = torch.cat(learned_coordinates)
    episode_true = torch.cat(true_parameters)
    if episode_learned.shape[1] != num_slots or episode_true.shape[1] != num_slots:
        raise ValueError(
            "mass recovery requires exactly one scalar parameter slot per object; "
            f"got learned {tuple(episode_learned.shape)} and "
            f"true {tuple(episode_true.shape)} for {num_slots} objects"
        )
    recovery = nonlinear_mcc(episode_learned, episode_true)

    # Every axis is already in tracked-object order: state row i, state column
    # i and parameter column i all descend from simulator track i.
    learned_graph = torch.cat(learned_graphs)
    true_graph = torch.cat(true_graphs)

    pred_loss = _weighted_mean(pred_losses, batch_weights)
    logit_penalty = _weighted_mean(logit_penalties, batch_weights)
    weighted_logit = lambda_logit * logit_penalty
    # Hybrid §4.3 dual form, scalarised: the bound covers the teacher-forced
    # AND rollout errors. τ is calibrated on exactly this number.
    raw_rollout = _weighted_mean(rollout_losses, batch_weights) if rollout_losses else 0.0
    rollout_loss = lambda_roll * raw_rollout  # the term as it enters the bound
    constraint_loss = pred_loss + rollout_loss + weighted_logit  # raw units
    metrics = {
        "pred_loss": pred_loss,
        "rollout_loss": rollout_loss,
        "mean_abs_logit": _weighted_mean(mean_abs_logits, batch_weights),
        "gate_entropy": _weighted_mean(gate_entropies, batch_weights),
        "constraint_loss": constraint_loss,
        # SPARTAN's graph metric (their Table 1): SHD between the learned
        # graph and the ground-truth causal graph over the decoded rows.
        # Lower is better; range [0, 2N^2]. D28.
        "shd": structural_hamming_distance(learned_graph, true_graph).item(),
        # The one mass-recovery number: Baumgartner et al. App. F.1 MCC. D27.
        "mcc": recovery.score.item(),
        "path_density": _weighted_mean(path_density, batch_weights),
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
