"""The unchanged ordinary dense objective, with one explicit state/EMA policy choice."""

from typing import Literal, cast

import torch
from torch import Tensor

from diagnostics.joint_state_survival.data import check_train
from diagnostics.multiquery_control.data import TensorSplit
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.loop import ObjectiveConfig, TrainingObjective

Arm = Literal["frozen_state", "live_state"]


def configure_model(model: SCJEPA, arm: Arm) -> None:
    """Keep the exact distinct initial target; change only trainability of the online map."""
    if arm not in ("frozen_state", "live_state"):
        raise ValueError("Unknown joint-state survival arm")
    model.requires_grad_(False)
    model.parameter_encoder.requires_grad_(True)
    model.predictor.requires_grad_(True)
    model.online_encoder.requires_grad_(arm == "live_state")
    model.target_encoder.eval()
    if (
        model.ema_decay != 0.996
        or not model.autoregressive
        or model.context_len != 10
        or model.parameter_source != "learned"
    ):
        raise ValueError("Require original attached K10 rollout and EMA.996")
    if not all(layer.dense and not layer.identity for layer in model.predictor.layers):
        raise ValueError("Require original all-open dense SPARTAN")


def ordinary_objective(config: ObjectiveConfig) -> TrainingObjective:
    """Require every saved ordinary loss term, with no sparsity controller or schedule."""
    if (config.state_regularizer, config.parameter_regularizer) != ("sigreg", "sigreg") or (
        config.state_regularizer_weight,
        config.parameter_regularizer_weight,
        config.parameter_consistency_weight,
        config.logit_weight,
        config.regularizer_num_projections,
        config.state_sparsity_weight,
        config.action_sparsity_weight,
        config.parameter_sparsity_weight,
    ) != (0.4, 0.4, 1.0, 0.0001, 256, 0.0, 0.0, 0.0):
        raise ValueError(
            "Require saved full ordinary SIGReg.4/consistency1/logit.0001/256 objective"
        )
    return TrainingObjective(config)


def training_loss(
    model: SCJEPA, batch: TensorSplit, objective: TrainingObjective, arm: Arm
) -> tuple[Tensor, dict[str, Tensor]]:
    """Re-encode actual observations; donors and both autoregressive steps stay attached."""
    check_train(batch)
    if arm not in ("frozen_state", "live_state") or objective.geco is not None:
        raise ValueError("Require a declared state arm and no GECO")
    if any(p.requires_grad for p in model.target_encoder.parameters()) or any(
        p.requires_grad != (arm == "live_state") for p in model.online_encoder.parameters()
    ):
        raise ValueError("State trainability differs from arm")
    if not all(
        p.requires_grad
        for block in (model.parameter_encoder, model.predictor)
        for p in block.parameters()
    ):
        raise ValueError("Both E and P must train")
    model.train()
    model.target_encoder.eval()
    if arm == "frozen_state":
        model.online_encoder.eval()
    output = cast(ModelOutput, model(batch["observations"], batch["actions"]))
    loss, terms = objective(output)
    return loss, {
        **terms,
        "optimized_total": loss,
        "online_variance": output.online_states.detach().flatten(0, 2).var(0).mean(),
        "ema_enabled": loss.new_tensor(arm == "live_state"),
    }


@torch.no_grad()
def after_update(model: SCJEPA, arm: Arm) -> None:
    """Advance the original target EMA exactly once after an accepted live-state update."""
    if arm == "live_state":
        model.update_target()
    elif arm != "frozen_state":
        raise ValueError("Unknown state policy")
