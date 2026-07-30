"""Held-out evaluation for the visual-to-visual regime (experiments.pdf §6.6/§6.7).

Reports the same two metrics as the state-to-state regime — ``mcc`` (D27) and ``shd`` (D28) —
so the three experiments sit on one axis, plus the latent prediction error and
the Eq. 123 constraint the dual actually sees.

The one structural difference is that nothing here can be read off directly.
The state-to-state regime's parameter row ``i`` IS physical object ``i`` (zeta = id, Eq. 132),
whereas the visual-to-visual regime's rows are anonymous visual tracks whose order changes
between episodes. Every quantity is therefore mapped into physical-object order
first, using the evaluation-only geometric assignment of Eqs. 137-138 — which
never sees the learned parameters or the true masses, so it cannot be a
permutation chosen to improve the score.
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
from scjepa.eval.visual_alignment import physical_assignment, slot_centroids
from scjepa.losses.alignment import align_to_assignment, invert_assignment
from scjepa.losses.predictive import rollout_weights, weighted_rollout_mse
from scjepa.models.visual_to_visual import VisualToVisualModel

__all__ = ["VisualToVisualReport", "evaluate_visual_to_visual"]


class VisualToVisualReport(NamedTuple):
    """Metrics plus the artifacts the recovery grid needs."""

    metrics: dict[str, float]
    learned_params: Tensor
    """Learned coordinates in PHYSICAL-object order, (episodes, N)."""
    true_masses: Tensor


@torch.no_grad()
def evaluate_visual_to_visual(
    model: VisualToVisualModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 8,
    max_batches: int | None = 16,
    device: str = "cpu",
    context_len: int | None = None,
    lambda_logit: float = 0.0,
    rollout_len: int | None = None,
    lambda_roll: float = 0.0,
    resolution: int = 64,
) -> VisualToVisualReport:
    """Run the held-out evaluation; returns metrics in the shared key namespace."""
    was_training = model.training
    model.eval()
    torch_device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pred_losses: list[Tensor] = []
    rollout_losses: list[Tensor] = []
    logit_penalties: list[Tensor] = []
    abs_logits: list[Tensor] = []
    entropies: list[Tensor] = []
    densities: list[Tensor] = []
    variances: list[Tensor] = []
    shds: list[Tensor] = []
    params: list[Tensor] = []
    masses: list[Tensor] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        frames = batch["frames"].to(torch_device)
        states = batch["states"].to(torch_device)
        output = model(frames, context_len=context_len, rollout_len=rollout_len)
        tpar = context_len if context_len is not None else frames.shape[1] - 1
        num_slots = output.causal_params.shape[1]

        pred_losses.append((output.prediction - output.target).square().mean())
        if output.rollout_prediction is not None and output.rollout_target is not None:
            rollout_losses.append(
                weighted_rollout_mse(
                    output.rollout_prediction,
                    output.rollout_target,
                    rollout_weights(
                        output.rollout_prediction.shape[1],
                        device=output.rollout_prediction.device,
                    ),
                )
            )
        logit_penalties.append(output.logit_penalty)
        abs_logits.append(output.mean_abs_logit)
        entropies.append(output.gate_entropy)
        variances.append(output.target_variance)

        # Eqs. 134-138: match anonymous tracks to physical objects on geometry.
        allocations = model.online.encoder.allocations(frames[:, : frames.shape[1] - 1])
        centroids = slot_centroids(allocations[:, :tpar], resolution)
        zeta = physical_assignment(centroids, states[:, :tpar, :, :2])
        inverse = invert_assignment(zeta)

        # Eq. 139: express learned coordinates in physical-object order so they
        # can be pooled across episodes at all.
        params.append(align_to_assignment(output.causal_params, inverse, track_dim=1).squeeze(-1))
        masses.append(batch["params"].to(torch_device).squeeze(-1))

        # The learned graph is indexed by visual track on BOTH axes, so the
        # ground truth is conjugated into that order instead (Eq. 84's P^tok):
        # comparing the two in the SAME per-episode order is what matters, and
        # the graph score is not pooled across episodes the way MCC is.
        learned = read_learned_graph(output.path_matrix, num_slots)
        contacts = batch["contacts"][:, tpar - 1 :].bool().to(torch_device)
        truth = gt_causal_graph_from_contacts(contacts)
        repeats = learned.shape[0] // truth.shape[0]
        truth_visual = align_to_assignment(truth, zeta, track_dim=1)
        truth_visual = torch.cat(
            [
                align_to_assignment(truth_visual[:, :, :num_slots], zeta, track_dim=2),
                align_to_assignment(truth_visual[:, :, num_slots:], zeta, track_dim=2),
            ],
            dim=2,
        ).repeat_interleave(repeats, dim=0)
        shds.append(structural_hamming_distance(learned, truth_visual).float().mean())
        densities.append((output.path_matrix[:, :num_slots] >= 0.5).float().mean())

    if not pred_losses:
        raise ValueError("evaluation dataset produced no batches")

    learned_params = torch.cat(params).cpu()
    true_masses = torch.cat(masses).cpu()
    report = nonlinear_mcc(learned_params, true_masses)
    pred_loss = torch.stack(pred_losses).mean()
    variance = torch.stack(variances).mean().clamp(min=model.variance_floor)
    # Hybrid §4.3 dual form, scalarised: the SAME numerator the trainer builds.
    rollout_loss = (
        lambda_roll * torch.stack(rollout_losses).mean()
        if rollout_losses
        else torch.zeros((), device=pred_loss.device)
    )
    metrics = {
        "pred_loss": float(pred_loss),
        "rollout_loss": float(rollout_loss),
        # Eq. 123: exactly the scalar the dual is held to. tau_3 is calibrated
        # on this, so rollout_len/lambda_roll MUST match training.
        "constraint_loss": float(
            (pred_loss + rollout_loss) / variance
            + lambda_logit * torch.stack(logit_penalties).mean()
        ),
        "target_variance": float(variance),
        "mean_abs_logit": float(torch.stack(abs_logits).mean()),
        "gate_entropy": float(torch.stack(entropies).mean()),
        "path_density": float(torch.stack(densities).mean()),
        "shd": float(torch.stack(shds).mean()),
        "mcc": float(report.score),
        "num_samples": float(learned_params.shape[0]),
    }
    if was_training:
        model.train()
    return VisualToVisualReport(
        metrics=metrics, learned_params=learned_params, true_masses=true_masses
    )
