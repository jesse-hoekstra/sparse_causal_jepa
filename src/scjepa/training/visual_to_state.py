"""Training loop for the visual-to-state regime: frames in, true states as targets.

Inherits every guard, checkpoint and resume path from
:class:`scjepa.training.loop.Trainer`. Two things differ from the state-to-state regime:

1. the model reads FRAMES, and its predictions come out in visual-track order
   while the targets are in simulator-row order, so a detached trajectory-level
   assignment (Eqs. 98-100) is applied before the loss;
2. the loss is therefore computed here rather than by the base class's plain
   ``aligned_mse``.

The constraint is Eq. 103 — ``L_pred + lambda_logit * L_logit``, RAW and
unnormalized, exactly as in the state-to-state regime. Only visual-to-visual
normalizes by a target variance, because only it has a target whose scale can
drift. There is no grounding loss and no representation regularizer here.

The assignment is loss-side supervision only: neither it nor any true state
reaches the encoder, the state head, the parameter encoder or SPARTAN.
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from scjepa.eval.visual_to_state import evaluate_visual_to_state
from scjepa.losses.alignment import (
    align_to_assignment,
    coordinate_scales,
    trajectory_assignment,
)
from scjepa.models.visual_to_state import VisualToStateModel, VisualToStateOutput
from scjepa.training.loop import MetricLogger, TrainConfig, Trainer

__all__ = ["VisualToStateTrainer"]


class VisualToStateTrainer(Trainer):
    """Trainer for the visual-context, true-state-target experiment."""

    model: VisualToStateModel

    def __init__(
        self,
        model: VisualToStateModel,
        dataset: Dataset[dict[str, Tensor]],
        config: TrainConfig,
        logger: MetricLogger | None = None,
        eval_dataset: Dataset[dict[str, Tensor]] | None = None,
        scale_episodes: int = 512,
    ) -> None:
        """Build the trainer and FREEZE the assignment's coordinate scales.

        sigma_a (Eq. 98) is estimated once from the training split and then held
        fixed for the whole run: a scale that drifted with the model would make
        the assignment cost incomparable between steps. It only ranks
        permutations — the reported loss (Eq. 101) is raw and unstandardized —
        so sampling error here is harmless.
        """
        super().__init__(model, dataset, config, logger, eval_dataset=eval_dataset)
        scales = coordinate_scales(
            dataset,
            num_episodes=scale_episodes,
            context_len=config.context_len if config.context_len is not None else 1,
        )
        # Written into the MODEL so it serializes: a later standalone evaluation
        # must reproduce the same assignment, or its constraint is not the one
        # tau was calibrated from.
        self.model.coordinate_scales.copy_(scales.to(self.device))

    def _forward(  # type: ignore[override]
        self, batch: dict[str, Tensor], rollout_len: int | None = None
    ) -> VisualToStateOutput:
        """Frames are the only predictor input; states supply targets only."""
        del rollout_len  # This regime has no autoregressive rollout branch.
        return self.model(
            batch["frames"].to(self.device),
            batch["states"].to(self.device),
            context_len=self.config.context_len,
        )

    def _prediction_loss(self, output: VisualToStateOutput) -> Tensor:
        """Eq. 101 under the Eq. 99 assignment: raw, unstandardized next-state MSE.

        Gradients reach the model through the selected predictions, the latent
        visual states, the parameter representation and the visual encoder — but
        never through the discrete assignment, which is chosen on detached
        tensors and applied by gathering the (gradient-free) TARGETS.
        """
        assignment = trajectory_assignment(
            output.prediction, output.target, self.model.coordinate_scales
        )
        aligned = align_to_assignment(output.target, assignment, track_dim=2)
        return (output.prediction - aligned).square().mean()

    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Identical to the base step except that the loss is assignment-aware."""
        output = self._forward(batch)
        sparsity_active = self._sparsity_active()
        pred_loss = self._prediction_loss(output)
        logit_loss = self.config.lambda_logit * output.logit_penalty
        total = pred_loss + logit_loss
        if sparsity_active:
            total = total + self.lagrangian.penalty_weight * output.sparsity
        constraint = (pred_loss + logit_loss).detach()  # Eq. 103, raw

        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {self.step}: pred={pred_loss.item():.4g} "
                f"sparsity={output.sparsity.item():.4g}"
            )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()  # pyright: ignore[reportUnknownMemberType]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        skip = (not bool(torch.isfinite(grad_norm))) or (
            float(grad_norm) > self.config.grad_skip_threshold
        )
        if skip:
            self.optimizer.zero_grad(set_to_none=True)
            self.total_skips += 1
            self.consecutive_skips += 1
            if self.consecutive_skips >= self.config.grad_skip_max_consecutive:
                raise RuntimeError(
                    f"{self.consecutive_skips} consecutive grad-spike skips at step "
                    f"{self.step} (grad_norm={float(grad_norm):.3g}): the model is no "
                    "longer trainable — resume from the last healthy checkpoint."
                )
        else:
            self.consecutive_skips = 0
            self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if sparsity_active:
                self.lagrangian.update(constraint)
            self.successful_updates += 1

        num_decoded = output.causal_params.shape[1]
        with torch.no_grad():
            latent_std = float(output.context_states.flatten(0, 2).std(dim=0).mean())
        return {
            "loss/total": total.item(),
            "loss/pred": pred_loss.item(),
            "loss/logit": logit_loss.item(),
            "loss/sparsity": output.sparsity.item(),
            "attention/logit_penalty": output.logit_penalty.item(),
            "attention/mean_abs_logit": output.mean_abs_logit.item(),
            "attention/gate_entropy": output.gate_entropy.item(),
            "sparsity/constraint": constraint.item(),
            "sparsity/lambda": float(torch.exp(self.lagrangian.log_lambda).item()),
            "sparsity/path_density": (output.path_matrix[:, :num_decoded] >= 0.5)
            .float()
            .mean()
            .item(),
            "sparsity/path_density_full": (output.path_matrix >= 0.5).float().mean().item(),
            "sparsity/active": float(sparsity_active),
            # Not an anti-collapse objective (§6.5 needs none) — a cheap tell
            # that the visual encoder is still producing varied states.
            "health/latent_std": latent_std,
            "health/grad_norm": float(grad_norm.item()),
            "health/skipped_steps": float(self.total_skips),
            "schedule/successful_updates": float(self.successful_updates),
        }

    def _eval_step(self) -> dict[str, float]:
        """Held-out metrics; see :func:`evaluate_visual_to_state` for which alignment."""
        assert self.eval_dataset is not None
        report = evaluate_visual_to_state(
            self.model,
            self.eval_dataset,
            batch_size=self.config.batch_size,
            device=self.config.device,
            context_len=self.config.context_len,
            lambda_logit=self.config.lambda_logit,
        )
        self.model.train()
        return {f"eval/{key}": value for key, value in report.metrics.items()}
