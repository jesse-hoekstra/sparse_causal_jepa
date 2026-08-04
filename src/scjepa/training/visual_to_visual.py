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
    one direction scores ~1 and an isotropic one scores d_s. Those three go to
    their floor together under SCALE collapse.

    ``temporal_var`` covers a mode the other three are jointly blind to: the
    variance across TIME WITHIN an episode, averaged over episodes. The other
    three pool episode and time, so a representation that is frozen in time but
    still varies across episodes keeps all of them healthy — while making both
    prediction branches trivially satisfiable by the identity map, which is also
    the sparsest possible graph. That degenerate optimum exists under teacher
    forcing alone; the K-step rollout raises its payoff (an honest model pays
    L_TF + lambda_roll*L_roll, a frozen one pays about zero for both), so this
    number is the one that separates "learned the dynamics" from "stopped
    moving". Healthy: same order as ``content_var``. Degenerate: heads for zero
    while ``content_var`` and ``effective_rank`` sit still.
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
        # Time is axis 1 of (B, T, N, d_s): reduce over it FIRST, so episode
        # variation cannot mask a temporally frozen representation.
        temporal_var = float(states.detach().float().var(dim=1, unbiased=False).mean())
        return {
            f"{prefix}/std": float(flat.std(dim=0).mean()),
            f"{prefix}/content_var": float(centred.square().mean()),
            f"{prefix}/effective_rank": effective_rank,
            f"{prefix}/temporal_var": temporal_var,
        }


class VisualToVisualTrainer(Trainer):
    """Trainer for the fully visual, learned-target experiment."""

    model: VisualToVisualModel

    def _forward(  # type: ignore[override]
        self, batch: dict[str, Tensor], rollout_len: int | None
    ) -> VisualToVisualOutput:
        """Read frames; the true states in the batch are for evaluation only."""
        frames = batch["frames"].to(self.device)
        return self.model(
            frames,
            context_len=self.config.context_len,
            rollout_len=rollout_len,
        )

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

        Under the hybrid objective ``pred_loss`` arrives as L_TF +
        lambda_roll*L_roll (the caller scalarises §4.3's two bounds into one).
        Both are squared errors in the same target space, so both scale with the
        representation exactly as ``target_variance`` does: the ratio stays
        scale-free with the rollout term in it. What the normalization does NOT
        see is a target frozen in time — watch ``collapse/*/temporal_var``.
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
            # Periodic eval mirrors the live stage; final evaluation reads the
            # configured terminal horizon from resolved_config.yaml.
            rollout_len=self._current_rollout_len(),
            lambda_roll=self.config.lambda_roll,
        )
        self.model.train()
        return {f"eval/{key}": value for key, value in report.metrics.items()}
