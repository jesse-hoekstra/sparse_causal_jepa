"""Held-out evaluation for the visual-to-state regime (experiments.pdf §6.5/§6.7).

**Two different alignments are in play, deliberately.** Getting them the wrong
way round would either make tau meaningless or make the recovery score
self-fulfilling:

* ``pred_loss`` and ``constraint_loss`` use the TRAINING assignment, Eq. 99's
  prediction-error Hungarian. tau_2 is calibrated as the held-out constraint of
  a converged dense reference, so the held-out constraint has to be the same
  quantity the training constraint is — anything else calibrates the dual
  against a number it never sees.
* ``mcc`` and ``shd`` use the EVALUATION assignment, Eqs. 137-138's geometric
  match of slot-attention mask centroids to true rendered centres. That one
  never touches the learned parameters, the true masses, or prediction quality,
  so it cannot be a permutation chosen to flatter the score — which is exactly
  the guarantee §6.7 demands and which the prediction-error assignment cannot
  give.

Metric keys are the shared namespace (D27/D28) so all three regimes sit on one
axis.
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
from scjepa.losses.alignment import (
    align_to_assignment,
    invert_assignment,
    trajectory_assignment,
)
from scjepa.models.visual_to_state import VisualToStateModel

__all__ = ["VisualToStateReport", "evaluate_visual_to_state"]


class VisualToStateReport(NamedTuple):
    """Metrics plus the artifacts the recovery grid needs."""

    metrics: dict[str, float]
    learned_params: Tensor
    """Learned coordinates in PHYSICAL-object order, (episodes, N)."""
    true_masses: Tensor


@torch.no_grad()
def evaluate_visual_to_state(
    model: VisualToStateModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 8,
    max_batches: int | None = 16,
    device: str = "cpu",
    context_len: int | None = None,
    lambda_logit: float = 0.0,
    resolution: int = 64,
) -> VisualToStateReport:
    """Run the held-out evaluation; returns metrics in the shared key namespace."""
    was_training = model.training
    model.eval()
    torch_device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pred_losses: list[Tensor] = []
    logit_penalties: list[Tensor] = []
    abs_logits: list[Tensor] = []
    entropies: list[Tensor] = []
    densities: list[Tensor] = []
    shds: list[Tensor] = []
    params: list[Tensor] = []
    masses: list[Tensor] = []
    switch_rates: list[Tensor] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        frames = batch["frames"].to(torch_device)
        states = batch["states"].to(torch_device)
        output = model(frames, states, context_len=context_len)
        tpar = context_len if context_len is not None else frames.shape[1] - 1
        num_slots = output.causal_params.shape[1]
        scales = model.coordinate_scales.to(torch_device)

        # (a) TRAINING assignment -> the loss and the constraint tau is set from.
        training_assignment = trajectory_assignment(output.prediction, output.target, scales)
        aligned = align_to_assignment(output.target, training_assignment, track_dim=2)
        pred_losses.append((output.prediction - aligned).square().mean())
        logit_penalties.append(output.logit_penalty)
        abs_logits.append(output.mean_abs_logit)
        entropies.append(output.gate_entropy)

        # (b) EVALUATION assignment -> parameter recovery and graph axes.
        allocations = model.visual.encoder.allocations(frames[:, : frames.shape[1] - 1])
        zeta = physical_assignment(
            slot_centroids(allocations[:, :tpar], resolution), states[:, :tpar, :, :2]
        )
        params.append(
            align_to_assignment(output.causal_params, invert_assignment(zeta), track_dim=1).squeeze(
                -1
            )
        )
        masses.append(batch["params"].to(torch_device).squeeze(-1))
        # Disagreement between the two is a tracking tell, not a metric: they
        # answer different questions, but a model that predicts well for the
        # right object should mostly agree with the geometry.
        switch_rates.append((training_assignment != zeta).float().mean())

        learned = read_learned_graph(output.path_matrix, num_slots)
        contacts = batch["contacts"][:, tpar - 1 :].bool().to(torch_device)
        truth = gt_causal_graph_from_contacts(contacts)
        truth_visual = align_to_assignment(truth, zeta, track_dim=1)
        truth_visual = torch.cat(
            [
                align_to_assignment(truth_visual[:, :, :num_slots], zeta, track_dim=2),
                align_to_assignment(truth_visual[:, :, num_slots:], zeta, track_dim=2),
            ],
            dim=2,
        ).repeat_interleave(learned.shape[0] // truth.shape[0], dim=0)
        shds.append(structural_hamming_distance(learned, truth_visual).float().mean())
        densities.append((output.path_matrix[:, :num_slots] >= 0.5).float().mean())

    if not pred_losses:
        raise ValueError("evaluation dataset produced no batches")

    learned_params = torch.cat(params).cpu()
    true_masses = torch.cat(masses).cpu()
    report = nonlinear_mcc(learned_params, true_masses)
    pred_loss = torch.stack(pred_losses).mean()
    penalty = torch.stack(logit_penalties).mean()
    metrics = {
        "pred_loss": float(pred_loss),
        "constraint_loss": float(pred_loss + lambda_logit * penalty),  # Eq. 103, raw
        "logit_penalty": float(penalty),
        "mean_abs_logit": float(torch.stack(abs_logits).mean()),
        "gate_entropy": float(torch.stack(entropies).mean()),
        "path_density": float(torch.stack(densities).mean()),
        "shd": float(torch.stack(shds).mean()),
        "mcc": float(report.score),
        "assignment_disagreement": float(torch.stack(switch_rates).mean()),
        "num_samples": float(learned_params.shape[0]),
    }
    if was_training:
        model.train()
    return VisualToStateReport(
        metrics=metrics, learned_params=learned_params, true_masses=true_masses
    )
