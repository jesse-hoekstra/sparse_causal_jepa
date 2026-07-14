"""Identifiability evaluation harness: run a model over episodes, report metrics.

Consumes any JEPA variant with the ``JepaOutput`` contract plus a dataset whose
items carry the bounce-style ground truth (``params``, ``contacts``). Returns
scalar metrics and the marginal-recovery scatter data.

Alignment caveat (see graph.py): metrics assume slot i ≡ object i. That holds
by construction in the GT-embedding regime (StateJepa). In the vision regime
learned slots are NOT aligned to objects; numbers computed there without an
alignment step are meaningless — the caller (script) must refuse or align.
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
from scjepa.losses import hungarian_mse
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
        lambda_logit: Training's attention-logit weight; used to report
            ``constraint_loss`` = pred / Var(target batch) + lambda_logit *
            logit_penalty — the SAME scale-free quantity the Lagrangian dual
            compares against tau (Baumgartner Eq. 9 with the D17 variance
            normalization; raw MSE lives in a trainable space whose scale is
            solution-dependent, see docs/decisions.md D17). tau calibration
            must read this, not ``pred_loss``. Raw ``pred_loss`` and
            ``target_var`` are reported alongside for diagnostics.
    """
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        inputs = batch[input_key].to(device)
        output: JepaOutput = model(inputs, context_len=context_len, rollout_horizon=rollout_horizon)
        num_slots = output.prediction.shape[1]
        batch_pred = hungarian_mse(output.prediction, output.target_slots).cpu()
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
        # Fraction of thresholded path-matrix edges (same >= 0.5 rule as
        # graph.py); entries of the path matrix are PATH COUNTS, so the old
        # sum/(tokens^2) could exceed 1 and was not a density.
        path_density.append((output.path_matrix >= 0.5).float().mean().cpu())
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
    best_dim, recovery_learned, recovery_true = marginal_recovery(learned_flat, true_flat)
    pred_loss = torch.stack(pred_losses).mean().item()
    pred_loss_normalized = torch.stack(pred_losses_normalized).mean().item()
    logit_penalty = torch.stack(logit_penalties).mean().item()
    metrics = {
        "pred_loss": pred_loss,
        "pred_loss_normalized": pred_loss_normalized,
        "target_var": torch.stack(target_vars).mean().item(),
        "logit_penalty": logit_penalty,
        # D17 scale-free constraint — the τ quantity (mean of per-batch ratios,
        # matching what the trainer's dual EMA averages).
        "constraint_loss": pred_loss_normalized + lambda_logit * logit_penalty,
        "shd_state": torch.stack(shd_state).mean().item(),
        "shd_param": torch.stack(shd_param).mean().item(),
        "mcc": nonlinear_mcc(learned_flat, true_flat).item(),
        "mcc_linear": mean_max_correlation(learned_flat, true_flat).item(),
        "path_density": torch.stack(path_density).mean().item(),
        "num_samples": float(learned_flat.shape[0]),
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
