"""Training loop for learned visual states and EMA targets.

Everything that makes a run survivable — the D18 grad-spike skip guard, rolling
checkpoints, exact resume, deterministic epoch order, the GECO dual — is
inherited unchanged from :class:`scjepa.training.loop.Trainer`, because none of
it depends on where the target comes from. The visual-specific behavior is:

1. the model reads FRAMES, not true states;
2. the EMA target is stepped after every accepted optimizer step;
3. representation variation is monitored on the predictive suffix.

The predictive objective and raw GECO constraint match Experiment 1: teacher
forcing plus sampled T=2 endpoint loss, with the logit term in the constraint.
The path term stays outside the constraint. Collapse diagnostics are monitoring,
not normalization or an anti-collapse objective. A low latent loss alone is
insufficient evidence of success; object grounding and held-out state probes
must also improve.
"""

from typing import cast

import torch
from torch import Tensor

from scjepa.eval.visual_to_visual import evaluate_visual_to_visual
from scjepa.losses import rollout_t2_endpoint_mse
from scjepa.models.visual_to_visual import VisualToVisualModel, VisualToVisualOutput
from scjepa.training.distributed import gather_batch_tensor
from scjepa.training.loop import Trainer

__all__ = ["VisualToVisualTrainer", "collapse_metrics"]


def collapse_metrics(states: Tensor, prefix: str) -> dict[str, float]:
    """Representation diagnostics for one branch, independent of the constraint.

    ``std``: mean per-coordinate standard deviation. ``content_var``: variance
    across (episode, time) averaged over tracks and coordinates.
    ``effective_rank``: the exponential of the
    entropy of the normalized covariance eigenvalues, so a representation using
    one direction scores ~1 and an isotropic one scores d_s. Those three go to
    their floor together under SCALE collapse.

    ``temporal_var`` measures variation within each episode. Together with
    per-row ``content_var`` it catches fixed slot identities and a representation
    frozen in time despite variation between episodes. Effective rank pools
    rows and is not sufficient evidence against either failure mode.
    """
    with torch.no_grad():
        flat = states.detach().flatten(0, 2).float()  # (episodes * time * tracks, d_s)
        centred = flat - flat.mean(dim=0, keepdim=True)
        covariance = centred.T @ centred / max(flat.shape[0] - 1, 1)
        eigenvalues = cast(
            Tensor,
            torch.linalg.eigvalsh(covariance),  # pyright: ignore[reportUnknownMemberType]
        ).clamp(min=0.0)
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
            f"{prefix}/content_var": float(
                states.detach().float().flatten(0, 1).var(dim=0, unbiased=False).mean()
            ),
            f"{prefix}/effective_rank": effective_rank,
            f"{prefix}/temporal_var": temporal_var,
        }


class VisualToVisualTrainer(Trainer):
    """Trainer for the fully visual, learned-target experiment."""

    model: VisualToVisualModel

    def _forward(self, batch: dict[str, Tensor]) -> VisualToVisualOutput:  # type: ignore[override]
        """Read frames; simulator labels in the batch are evaluation-only."""
        frames = batch["frames"].to(self.device, non_blocking=self.device.type == "cuda")
        anchors = self.config.num_rollout_t2_anchors if self.config.lambda_rollout_t2 > 0 else 0
        return cast(
            VisualToVisualOutput,
            self._forward_model(
                frames,
                context_len=self.config.context_len,
                num_rollout_t2_anchors=anchors,
            ),
        )

    def _auxiliary_loss(self, output: VisualToVisualOutput) -> Tensor:  # type: ignore[override]
        """Mean endpoint loss through two attached transitions, as in Experiment 1."""
        if output.rollout_t2_prediction is None:
            return torch.zeros((), device=self.device)
        assert output.rollout_t2_target is not None
        return rollout_t2_endpoint_mse(output.rollout_t2_prediction, output.rollout_t2_target)

    def _after_optimizer_step(self, output: VisualToVisualOutput) -> None:  # type: ignore[override]
        """Eq. 111: the target moves only through the EMA, never by gradient."""
        del output
        self.model.update_target()

    def _extra_metrics(self, output: VisualToVisualOutput) -> dict[str, float]:  # type: ignore[override]
        """Predictive-suffix collapse diagnostics and context alignment changes."""
        # Initialization transients in frames 0..C-2 can hide a frozen suffix.
        # Report precisely the online anchors and EMA targets used by the loss.
        transitions = output.target.shape[0] // output.causal_params.shape[0]
        online_metrics = collapse_metrics(
            gather_batch_tensor(output.context_states[:, -transitions:]), "collapse/online"
        )
        target_metrics = collapse_metrics(
            gather_batch_tensor(output.target_states[:, -transitions:]), "collapse/target"
        )
        return (
            online_metrics
            | target_metrics
            | {
                # Reuse the gathered predictive-suffix statistic. Rank-local
                # variances omit variation between ranks' episode means.
                "collapse/target_variance": target_metrics["collapse/target/content_var"],
                "alignment/context_target_nonidentity_fraction": float(
                    (
                        output.target_assignment
                        != torch.arange(
                            output.target_assignment.shape[1],
                            device=output.target_assignment.device,
                        )
                    )
                    .float()
                    .mean()
                ),
            }
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
            lambda_rollout_t2=self.config.lambda_rollout_t2,
            num_rollout_t2_anchors=self.config.num_rollout_t2_anchors,
            rollout_t2_horizon=self.config.rollout_t2_horizon,
            oe_eval_horizon=self.config.oe_eval_horizon,
            probe_dataset=self.dataset,
        )
        self.model.train()
        return {f"eval/{key}": value for key, value in report.metrics.items()}
