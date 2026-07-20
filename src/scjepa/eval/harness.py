"""Identifiability evaluation harness: run a model over episodes, report metrics.

Consumes any JEPA variant with the ``JepaOutput`` contract plus a dataset whose
items carry the bounce-style ground truth (``params``, ``contacts``). Returns
scalar metrics and the marginal-recovery scatter data.

Alignment caveat (see graph.py): metrics assume slot i ≡ object i. That holds
by construction in the GT-embedding regime (StateJepa). In the vision regime
learned slots are NOT aligned to objects; numbers computed there without an
alignment step are meaningless, so this harness refuses them until persistent
trajectory-level alignment is implemented.
"""

from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.graph import (
    gt_graphs_from_contacts,
    read_learned_graphs,
    structural_hamming_distance,
)
from scjepa.eval.parameters import marginal_recovery, mean_max_correlation, nonlinear_mcc
from scjepa.losses import (
    prediction_mse,
    resolve_constraint_normalization,
    resolve_prediction_matching,
)
from scjepa.models.jepa import JepaOutput, SCJepa
from scjepa.models.state_jepa import StateJepa


class IdentifiabilityReport(NamedTuple):
    """Scalar metrics + recovery scatter data.

    ``recovery_*``: pooled (episode, object) samples for the single best-dim
    scatter. ``per_slot_learned`` (E, N, d) / ``per_slot_true`` (E, N): kept
    unpooled for the Baumgartner-Fig.-5/12-style grid (true mass of ball i vs
    slot j's learned parameter — diagonal sharp + off-diagonal blobs = each
    mass identified in ITS OWN slot).
    """

    metrics: dict[str, float]
    recovery_best_dim: int
    recovery_learned: Tensor
    recovery_true: Tensor
    per_slot_learned: Tensor
    per_slot_true: Tensor


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
    shd_param: list[Tensor] = []
    learned_params: list[Tensor] = []
    true_params: list[Tensor] = []
    path_density: list[Tensor] = []
    path_density_full: list[Tensor] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs = batch[input_key].to(device)
        output: JepaOutput = model(inputs, context_len=context_len, rollout_horizon=rollout_horizon)
        num_slots = output.prediction.shape[1]
        batch_pred = prediction_mse(
            output.prediction, output.target_slots, resolved_matching
        ).cpu()
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
        shd_param.append(structural_hamming_distance(param_learned, param_gt).cpu())
        # Primary density mirrors the optimized state-output rows. Parameter
        # rows after the last layer are not decoded, so report their inclusion
        # only under the explicit full-token diagnostic.
        path_density.append(
            (output.path_matrix[:, :num_slots] >= 0.5).float().mean().cpu()
        )
        path_density_full.append((output.path_matrix >= 0.5).float().mean().cpu())
        learned_params.append(
            output.causal_params.reshape(-1, output.causal_params.shape[-1]).cpu()
        )
        true_params.append(batch["params"].reshape(-1, batch["params"].shape[-1]).cpu())

    if num_slots is None:
        raise ValueError("dataset yielded no batches")
    learned_flat = torch.cat(learned_params)
    true_flat = torch.cat(true_params)
    per_slot_learned = learned_flat.reshape(-1, num_slots, learned_flat.shape[-1])
    per_slot_true = true_flat.reshape(-1, num_slots)
    # Baumgartner App. F.1 treats each trajectory as one sample and compares its
    # complete learned parameter vector against all five true masses. For the
    # exact scalar bottleneck this is E x 5 versus E x 5. Flattening only the
    # coordinate axes (not episode and object) also gives a well-defined
    # overcomplete diagnostic for legacy d>1 representations.
    episode_learned = per_slot_learned.flatten(1)
    episode_true = per_slot_true
    best_dim, recovery_learned, recovery_true = marginal_recovery(learned_flat, true_flat)
    pred_loss = torch.stack(pred_losses).mean().item()
    pred_loss_normalized = torch.stack(pred_losses_normalized).mean().item()
    logit_penalty = torch.stack(logit_penalties).mean().item()
    constraint_prediction = (
        pred_loss if resolved_normalization == "raw" else pred_loss_normalized
    )
    metrics = {
        "pred_loss": pred_loss,
        "pred_loss_normalized": pred_loss_normalized,
        "target_var": torch.stack(target_vars).mean().item(),
        "logit_penalty": logit_penalty,
        # The exact tau quantity, on the same configured scale as training.
        "constraint_loss": constraint_prediction + lambda_logit * logit_penalty,
        "shd_state": torch.stack(shd_state).mean().item(),
        "shd_param": torch.stack(shd_param).mean().item(),
        "mcc": nonlinear_mcc(episode_learned, episode_true).item(),
        "mcc_linear": mean_max_correlation(episode_learned, episode_true).item(),
        # Retain the former shared-per-object probe under an explicit name. It
        # can be useful diagnostically, but it is not the paper's MCC.
        "mcc_pooled": nonlinear_mcc(learned_flat, true_flat).item(),
        "mcc_linear_pooled": mean_max_correlation(learned_flat, true_flat).item(),
        "path_density": torch.stack(path_density).mean().item(),
        "path_density_full": torch.stack(path_density_full).mean().item(),
        "num_samples": float(episode_learned.shape[0]),
        "num_pooled_samples": float(learned_flat.shape[0]),
    }
    return IdentifiabilityReport(
        metrics=metrics,
        recovery_best_dim=int(best_dim.item()),
        recovery_learned=recovery_learned,
        recovery_true=recovery_true,
        per_slot_learned=per_slot_learned,
        per_slot_true=per_slot_true,
    )


__all__ = ["IdentifiabilityReport", "evaluate_identifiability"]
