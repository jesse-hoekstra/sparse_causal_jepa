"""Experiment 2 evaluation: mass/graph recovery, object tracking and frozen state probes.

Simulator truth is used exclusively for these diagnostics. One geometric
assignment, inferred from the context prefix, is held fixed across the episode.
The predictive constraint mirrors training's raw TF + T=2 + logit objective.
Its scale is specific to the current learned target representation; matching
loss values does not establish equal physical fidelity across models.
"""

from collections import defaultdict
from typing import NamedTuple, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.graph import (
    gt_causal_graph_from_contacts,
    read_learned_graph,
    structural_hamming_distance,
)
from scjepa.eval.parameters import nonlinear_mcc
from scjepa.eval.state_probes import StateProbe, fit_state_probe, state_probe_metrics
from scjepa.eval.visual_alignment import physical_assignment, slot_centroids, tracking_metrics
from scjepa.losses.alignment import align_to_assignment, invert_assignment
from scjepa.models.visual_to_visual import VisualToVisualModel

__all__ = ["VisualToVisualReport", "evaluate_visual_to_visual", "graph_in_slot_order"]


class VisualToVisualReport(NamedTuple):
    """Metrics and compact final-report artifacts, all detached on the CPU."""

    metrics: dict[str, float]
    learned_params: Tensor
    true_masses: Tensor
    recovery_matrix: Tensor
    slot_example: dict[str, Tensor]


def graph_in_slot_order(contacts: Tensor, assignment: Tensor) -> Tensor:
    """Return each transition's graph in the episode's fixed visual-slot order.

    Flatten BEFORE constructing graphs: the graph helper reads the last contact
    transition, so passing a whole window would silently repeat its final graph.
    """
    batch, transitions, slots, _ = contacts.shape
    truth = gt_causal_graph_from_contacts(contacts.reshape(batch * transitions, 1, slots, slots))
    order = assignment.repeat_interleave(transitions, dim=0)
    truth = align_to_assignment(truth, order, track_dim=1)
    return torch.cat(
        [
            align_to_assignment(truth[:, :, :slots], order, track_dim=2),
            align_to_assignment(truth[:, :, slots:], order, track_dim=2),
        ],
        dim=2,
    )


@torch.no_grad()
def _fit_probe(
    model: VisualToVisualModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int,
    max_batches: int,
    context_len: int | None,
    resolution: int,
    device: torch.device,
) -> StateProbe:
    features: list[Tensor] = []
    labels: list[Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator())
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        frames = batch["frames"].to(device)
        tpar = context_len if context_len is not None else frames.shape[1] - 1
        online = model.online(frames[:, :-1], capture_allocations=True)
        assert online.allocations is not None
        centroids = slot_centroids(online.allocations, resolution)
        states = batch["states"][:, :-1].to(device)
        order = physical_assignment(centroids[:, :tpar], states[:, :tpar, :, :2])
        # Probe the actual prediction-source window; initial slot transients
        # must not conceal an uninformative steady representation.
        features.append(online.states[:, tpar - 1 :].flatten(0, 2).cpu())
        labels.append(
            align_to_assignment(states[:, tpar - 1 :], order, track_dim=2).flatten(0, 2).cpu()
        )
    if not features:
        raise ValueError("probe training dataset produced no batches")
    return fit_state_probe(torch.cat(features), torch.cat(labels))


def _collapse_metrics(states: Tensor) -> dict[str, float]:
    """Compute three label-free checks for scale, temporal, and rank collapse."""
    flat = states.detach().float().flatten(0, 2)
    centred = flat - flat.mean(dim=0)
    covariance = centred.T @ centred / max(flat.shape[0] - 1, 1)
    eigenvalues = cast(Tensor, torch.linalg.eigvalsh(covariance)).clamp(min=0)  # pyright: ignore[reportUnknownMemberType]
    probabilities = eigenvalues / eigenvalues.sum().clamp(min=1e-12)
    rank = (-(probabilities * probabilities.clamp(min=1e-12).log()).sum()).exp()
    return {
        "target_temporal_variance": float(states.float().var(dim=1, unbiased=False).mean()),
        "target_effective_rank": float(rank),
    }


@torch.no_grad()
def evaluate_visual_to_visual(
    model: VisualToVisualModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int = 4,
    max_batches: int | None = 16,
    device: str = "cpu",
    context_len: int | None = None,
    lambda_logit: float = 0.0,
    lambda_rollout_t2: float = 1.0,
    num_rollout_t2_anchors: int = 8,
    rollout_t2_horizon: int = 2,
    oe_eval_horizon: int | None = None,
    resolution: int = 64,
    probe_dataset: Dataset[dict[str, Tensor]] | None = None,
    max_probe_batches: int = 16,
) -> VisualToVisualReport:
    """Evaluate fixed held-out episodes, preserving training mode and RNG state.

    ``probe_dataset`` must be the TRAINING split; its labels fit a frozen linear
    decoder only after encoding, and are never used by the representation loss.
    Omit it to skip state probes. MCC retains its existing episode-separated
    fit/score protocol inside the held-out representations (Baumgartner App. F.1).
    ``oe_eval_horizon`` requests a no-gradient LATENT rollout diagnostic; it does
    not certify physical observational equivalence.
    """
    if rollout_t2_horizon != 2:
        raise ValueError("Experiment 2 uses exactly T=2 training endpoints")
    if lambda_rollout_t2 < 0 or num_rollout_t2_anchors < 1 or max_probe_batches < 1:
        raise ValueError("rollout coefficient must be non-negative and sample counts positive")
    if probe_dataset is dataset:
        raise ValueError("probe fitting and scoring require separate episode datasets")
    was_training = model.training
    torch_device = torch.device(device)
    cuda_devices = []
    if torch_device.type == "cuda":
        device_index = cast(int | None, torch_device.index)
        cuda_devices = [torch.cuda.current_device() if device_index is None else device_index]
    try:
        model.to(torch_device).eval()
        with torch.random.fork_rng(devices=cuda_devices):  # pyright: ignore[reportUnknownMemberType]
            torch.default_generator.manual_seed(0)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(0)
            return _evaluate(
                model,
                dataset,
                batch_size,
                max_batches,
                torch_device,
                context_len,
                lambda_logit,
                lambda_rollout_t2,
                num_rollout_t2_anchors,
                oe_eval_horizon,
                resolution,
                probe_dataset,
                max_probe_batches,
            )
    finally:
        model.train(was_training)


@torch.no_grad()
def _evaluate(
    model: VisualToVisualModel,
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
    context_len: int | None,
    lambda_logit: float,
    lambda_rollout_t2: float,
    num_rollout_t2_anchors: int,
    oe_eval_horizon: int | None,
    resolution: int,
    probe_dataset: Dataset[dict[str, Tensor]] | None,
    max_probe_batches: int,
) -> VisualToVisualReport:
    probe = (
        _fit_probe(
            model, probe_dataset, batch_size, max_probe_batches, context_len, resolution, device
        )
        if probe_dataset is not None
        else None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=torch.Generator())
    totals: dict[str, float] = defaultdict(float)
    params: list[Tensor] = []
    masses: list[Tensor] = []
    probe_predictions: list[Tensor] = []
    probe_targets: list[Tensor] = []
    slot_example: dict[str, Tensor] = {}
    num_samples = 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        frames, states = batch["frames"].to(device), batch["states"].to(device)
        tpar = context_len if context_len is not None else frames.shape[1] - 1
        output = model(
            frames,
            context_len=context_len,
            capture_allocations=True,
            num_rollout_t2_anchors=num_rollout_t2_anchors if lambda_rollout_t2 > 0 else 0,
        )
        assert output.context_allocations is not None
        assert output.target_allocations is not None
        centroids = slot_centroids(output.context_allocations, resolution)
        order = physical_assignment(centroids[:, :tpar], states[:, :tpar, :, :2])
        inverse = invert_assignment(order)
        params.append(
            align_to_assignment(output.causal_params, inverse, track_dim=1).squeeze(-1).cpu()
        )
        masses.append(batch["params"].squeeze(-1).cpu())
        tf = (output.prediction - output.target).square().mean()
        t2 = torch.zeros((), device=device)
        if output.rollout_t2_prediction is not None:
            assert output.rollout_t2_target is not None
            t2 = (output.rollout_t2_prediction - output.rollout_t2_target).square().mean()
        learned = read_learned_graph(output.path_matrix, output.causal_params.shape[1])
        truth = graph_in_slot_order(batch["contacts"][:, tpar - 1 :].bool().to(device), order)
        metrics = {
            "pred_loss": float(tf),
            "loss_teacher_forcing": float(tf),
            "loss_rollout_t2_raw": float(t2),
            "loss_rollout_t2_weighted": float(lambda_rollout_t2 * t2),
            "loss_logit_weighted": float(lambda_logit * output.logit_penalty),
            "constraint_loss": float(
                tf + lambda_rollout_t2 * t2 + lambda_logit * output.logit_penalty
            ),
            "target_variance": float(output.target_variance),
            "mean_abs_logit": float(output.mean_abs_logit),
            "gate_entropy": float(output.gate_entropy),
            "path_density": float(learned.float().mean()),
            "shd": float(structural_hamming_distance(learned, truth)),
        }
        metrics.update(_collapse_metrics(output.target_states[:, tpar:]))
        metrics.update(tracking_metrics(centroids, states[:, :-1, :, :2], order))
        # Check agreement after the matching prefix, where correspondence was
        # not optimized; a one-transition evaluation has only its anchor left.
        start = min(tpar, centroids.shape[1] - 1)
        branch_centroids = centroids[:, start:]
        target_centroids = slot_centroids(output.target_allocations[:, start:-1], resolution)
        batch_count, length, slots, _ = branch_centroids.shape
        branch_assignment = physical_assignment(
            branch_centroids.reshape(batch_count * length, 1, slots, 2),
            target_centroids.reshape(batch_count * length, 1, slots, 2),
        )
        metrics["branch_slot_disagreement"] = float(
            (branch_assignment != torch.arange(slots, device=device)).float().mean()
        )
        if oe_eval_horizon is not None:
            prediction, target = model.rollout_for_evaluation(
                output.context_states,
                output.target_states,
                tpar,
                oe_eval_horizon,
                output.causal_params,
                output.episode_keys,
            )
            # Preserve the evaluation-only rollout ruler. This numerical floor
            # and normalization never enter the training loss or GECO bound.
            denominator = output.target_variance.clamp(min=1e-4)
            metrics[f"latent_rollout_k{oe_eval_horizon}_nrmse"] = float(
                ((prediction - target).square().mean() / denominator).sqrt()
            )
        if probe is not None:
            probe_predictions.append(
                probe.predict(output.context_states[:, tpar - 1 :].flatten(0, 2))
            )
            probe_targets.append(
                align_to_assignment(states[:, tpar - 1 : -1], order, track_dim=2)
                .flatten(0, 2)
                .cpu()
            )
        if not slot_example:
            times = torch.linspace(0, frames.shape[1] - 2, 4).long()
            slot_example = {
                "frames": frames[0, times].cpu(),
                "allocations": output.context_allocations[0, times].cpu(),
                "target_allocations": output.target_allocations[0, times].cpu(),
                "centroids": centroids[0, times].cpu(),
                "true_centres": states[0, times, :, :2].cpu(),
                "assignment": order[0].cpu(),
                "timesteps": times,
            }
        for key, value in metrics.items():
            totals[key] += value * batch_count
        num_samples += batch_count
    if num_samples < 2:
        raise ValueError("visual recovery evaluation requires at least two episodes")
    metrics = {key: value / num_samples for key, value in totals.items()}
    learned_params, true_masses = torch.cat(params), torch.cat(masses)
    recovery = nonlinear_mcc(learned_params, true_masses)
    metrics.update(mcc=float(recovery.score), num_samples=float(num_samples))
    if probe_predictions:
        metrics.update(state_probe_metrics(torch.cat(probe_predictions), torch.cat(probe_targets)))
    return VisualToVisualReport(metrics, learned_params, true_masses, recovery.matrix, slot_example)
