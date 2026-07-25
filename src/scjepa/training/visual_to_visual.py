"""Training loop for the visual-to-visual regime: three overrides on the shared Trainer.

Everything that makes a run survivable — the D18 grad-spike skip guard, rolling
checkpoints, exact resume, deterministic epoch order, the GECO dual — is
inherited unchanged from :class:`scjepa.training.loop.Trainer`, because none of
it depends on where the target comes from. Exactly three things differ:

1. the model reads FRAMES, not true states;
2. the dual is fed Eq. 123's variance-normalized constraint rather than Eq. 13's
   raw one, because a learned target's scale drifts during training and an
   unnormalized c would silently retune itself;
3. the EMA target is stepped after every optimizer step (Eq. 111).

A fourth addition is monitoring rather than objective: Eq. 124's collapse
diagnostics are logged every step. The EMA asymmetry does NOT make a constant
representation mathematically impossible, so §6.6 makes non-collapse an
empirical continuation requirement — a run with vanishing content variance or
degenerate effective rank is rejected, not reported.
"""

import torch
from torch import Tensor

from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.models.visual_to_visual import VisualToVisualModel, VisualToVisualOutput
from scjepa.training.loop import Trainer

__all__ = ["VisualToVisualTrainer", "collapse_metrics"]


def collapse_metrics(states: Tensor, prefix: str) -> dict[str, float]:
    """Eq. 124's representation diagnostics for one branch.

    ``std``: mean per-coordinate standard deviation. ``content_var``: variance
    across (episode, time) averaged over tracks and coordinates — the quantity
    Eq. 122 feeds to the constraint. ``effective_rank``: the exponential of the
    entropy of the normalized covariance eigenvalues, so a representation using
    one direction scores ~1 and an isotropic one scores d_s. All three go to
    their floor together under collapse.
    """
    with torch.no_grad():
        flat = states.detach().flatten(0, 2).float()  # (episodes * time * tracks, d_s)
        centred = flat - flat.mean(dim=0, keepdim=True)
        covariance = centred.T @ centred / max(flat.shape[0] - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0.0)
        total = eigenvalues.sum()
        if float(total) <= 0.0:
            effective_rank = 1.0
        else:
            weights = eigenvalues / total
            entropy = -(weights * (weights + 1e-12).log()).sum()
            effective_rank = float(entropy.exp())
        return {
            f"{prefix}/std": float(flat.std(dim=0).mean()),
            f"{prefix}/content_var": float(centred.square().mean()),
            f"{prefix}/effective_rank": effective_rank,
        }


class VisualToVisualTrainer(Trainer):
    """Trainer for the fully visual, learned-target experiment."""

    model: VisualToVisualModel

    def _forward(self, batch: dict[str, Tensor]) -> VisualToVisualOutput:  # type: ignore[override]
        """Read frames; the true states in the batch are for evaluation only."""
        frames = batch["frames"].to(self.device)
        return self.model(frames, context_len=self.config.context_len)

    def _constraint(
        self,
        pred_loss: Tensor,
        logit_loss: Tensor,
        output: VisualToVisualOutput,  # type: ignore[override]
    ) -> Tensor:
        """Eq. 123: normalize ONLY the scalar handed to the dual controller.

        The gradient objective (Eq. 121) keeps the raw latent MSE; dividing that
        by a moving denominator would change what is optimized. The floor
        epsilon_var stops a collapsing target from making the constraint look
        satisfiable by shrinking itself.
        """
        denominator = torch.clamp(output.target_variance, min=self.model.variance_floor)
        return (pred_loss.detach() / denominator + logit_loss.detach()).detach()

    def _after_optimizer_step(self, output: VisualToVisualOutput) -> None:  # type: ignore[override]
        """Eq. 111: the target moves only through the EMA, never by gradient."""
        del output
        self.model.update_target()

    def _extra_metrics(self, output: VisualToVisualOutput) -> dict[str, float]:  # type: ignore[override]
        """Eq. 124 collapse diagnostics plus the raw (unnormalized) predictor loss."""
        return (
            collapse_metrics(output.context_states, "collapse/online")
            | collapse_metrics(output.target_states, "collapse/target")
            | {"collapse/target_variance": float(output.target_variance)}
        )

    def _eval_step(self) -> dict[str, float]:
        """Held-out metrics through the geometric track alignment (§6.7)."""
        assert self.eval_dataset is not None
        report = evaluate_visual_to_visual(
            self.model,
            self.eval_dataset,
            batch_size=self.config.batch_size,
            device=self.config.device,
            context_len=self.config.context_len,
            lambda_logit=self.config.lambda_logit,
        )
        self.model.train()
        return {f"eval/{key}": value for key, value in report.metrics.items()}
