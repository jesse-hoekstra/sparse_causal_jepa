"""Training loop for the state-to-state regime (experiments.pdf §6.1.3 / §6.2).

Hybrid objective per step (hybrid write-up Eq. 36, dual form):

    sparse:          L = L_TF + λ_roll·L_roll + λ_logit·L_logit + λ⁻¹·L_path
    dense/identity:  L = L_TF + λ_roll·L_roll + λ_logit·L_logit  (no path penalty)

with L_TF the aligned raw-state MSE (Eq. 32/39) over every suffix transition,
L_roll the dense K-step autoregressive rollout term (Eq. 35), and
the dual constraint

    c = L_TF + λ_roll·L_roll + λ_logit·L_logit ≤ τ

driving the GECO controller. §4.3 specifies the dual form as minimising
sparsity subject to upper bounds on the teacher-forced AND rollout errors; that
is scalarised here into ONE bound with the same λ_roll that weights the
objective, so a single dual variable and a single τ remain. c still excludes
the path penalty, and no representation regularizer exists in this experiment.

For the K=30 experiment, λ_roll stays fixed at its declared value (1.0). The
autoregressive depth instead follows a curriculum indexed by SUCCESSFUL
optimizer updates: teacher forcing only, then K=2, 5, 10, 20, and finally 30.
Rejected gradient-spike batches do not advance this schedule. In sparse runs,
the path penalty and GECO controller remain frozen until the terminal K=30
stage, because τ is calibrated on the full K=30 constraint; applying that τ to
a shorter loss would create artificial slack and premature pruning.

CRITICAL: τ is calibrated as 1.0x the held-out c of a dense reference, so
``scjepa.eval.harness`` must compute c with the identical rollout settings —
changing ``rollout_len``/``lambda_roll`` on one side alone silently
invalidates τ. Both are read from the same config keys and passed through.

Optimizer: Adam at the SPARTAN default learning rate over all trainable
parameters jointly.

Run infrastructure retained from the audited loop (health, not objective):
D18 gradient-spike skip guard (a finite ~1e30 loss once froze a run silently
for 230k steps), rolling step-tagged checkpoints, exact resume including RNG
and dataloader position, and the periodic held-out identifiability eval.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.harness import evaluate_identifiability
from scjepa.losses import aligned_mse, rollout_weights, weighted_rollout_mse
from scjepa.models.state_to_state import StateToStateModel, TransitionOutput
from scjepa.training.lagrangian import SparsityLagrangian

RolloutCurriculum = tuple[tuple[int, int | None], ...]


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG is what libs consume
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


class MetricLogger(Protocol):
    """Minimal logging interface so W&B never blocks tests/CI."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record one step's scalar metrics."""
        ...


class NoopLogger:
    """Logger that drops everything (tests, CI, WANDB_MODE=disabled)."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Drop the metrics."""


@dataclass
class TrainConfig:
    """Everything the loop needs; every field maps 1:1 to a Hydra config key."""

    steps: int
    batch_size: int
    lr: float = 5e-5  # SPARTAN App. A.1 default (Adam)
    grad_clip: float = 1.0
    sparsity_enabled: bool = True  # false: dense / token-local references
    sparsity_tau: float = 0.1
    sparsity_step_size: float = 2e-2
    sparsity_lambda_init: float = 1e6  # §6.1.3: initial path weight 10⁻⁶
    sparsity_momentum: float = 0.99
    lambda_logit: float = 0.0  # selected by the dense sweep (§6.1.3); 0 disables
    # Hybrid rollout branch (write-up §4.2). rollout_len = K in Eqs. 33-35;
    # None disables the branch and recovers the pure teacher-forced objective.
    # lambda_roll weights L_roll in BOTH the objective and the constraint
    # (Eq. 36 dual form), so it must be identical in the dense τ-calibration
    # run and the sparse run.
    # 30 matches configs/experiment/bounce_baumgartner.yaml: a direct-Trainer
    # smoke must exercise the SAME objective the real run trains, so this
    # default tracks the experiment config rather than sitting below it.
    # Episodes shorter than Tpar-1+K raise — set it explicitly for tiny configs.
    rollout_len: int | None = 30
    lambda_roll: float = 1.0
    # Optional (successful_updates, K) stages. ``None`` K disables only the
    # autoregressive branch; teacher forcing still covers the whole suffix.
    # ``rollout_len`` remains the terminal/post-hoc-evaluation horizon.
    rollout_curriculum: RolloutCurriculum | None = None
    seed: int = 0
    device: str = "cpu"
    context_len: int | None = None  # Tpar (the state-to-state regime: 30); None -> T-1
    # Periodic held-out identifiability eval (the analog of Baumgartner
    # Fig. 17's MCC-over-steps). None = off; needs an eval_dataset.
    eval_every: int | None = None
    # D18 grad-spike guards: skip the update (optimizer AND dual) when the
    # pre-clip grad norm is non-finite or absurd; raise after too many
    # consecutive skips — the weights are then already broken, fail loudly.
    grad_skip_threshold: float = 1e3
    grad_skip_max_consecutive: int = 2000
    # DataLoader worker processes. 0 (the default) renders/loads inline, which
    # is right for the state-to-state regime: its episodes are plain tensor slices. The
    # visual experiments draw frames per batch, so without workers that cost is
    # SERIAL with the optimizer — measured 1.7 h over 300k steps at batch 8.
    # Episodes are deterministic per index and the shuffle order comes from the
    # main-process generator, so raising this does not change which data a step
    # sees.
    num_workers: int = 0
    prefetch_factor: int = 4
    log_every: int = 10
    checkpoint_every: int = 200
    # ALSO keep step-tagged checkpoints every N steps (last.pt is overwritten).
    checkpoint_keep_every: int | None = None
    out_dir: str = "outputs"


class Trainer:
    """Explicit single-device training loop; fails loudly, resumes exactly."""

    def __init__(
        self,
        model: StateToStateModel,
        dataset: Dataset[dict[str, Tensor]],
        config: TrainConfig,
        logger: MetricLogger | None = None,
        eval_dataset: Dataset[dict[str, Tensor]] | None = None,
    ) -> None:
        """Build optimizer and dual controller around the model."""
        seed_everything(config.seed)
        if config.eval_every is not None and eval_dataset is None:
            raise ValueError("eval_every set but no eval_dataset provided")
        self.config = config
        self._validate_rollout_curriculum()
        self.eval_dataset = eval_dataset
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.logger: MetricLogger = logger if logger is not None else NoopLogger()
        self.lagrangian = SparsityLagrangian(
            tau=config.sparsity_tau,
            step_size=config.sparsity_step_size,
            lambda_init=config.sparsity_lambda_init,
            momentum=config.sparsity_momentum,
        ).to(self.device)
        # ONE optimizer over everything trainable (P_η and f_gamma jointly).
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        # Eq. 35 weights depend on the live curriculum horizon. Cache the tiny
        # vectors by K rather than pinning one tensor for the whole run.
        self._rollout_weight_cache: dict[int, Tensor] = {}
        # The loader uses drop_last=True, so a dataset smaller than one batch
        # yields ZERO batches. `_batches` would then spin through empty epochs
        # forever — regenerating data, never stepping, never erroring. Fail here
        # instead: the run is misconfigured, not slow.
        batches_per_epoch = len(self._epoch_loader(0))
        if batches_per_epoch == 0:
            raise ValueError(
                f"dataset yields no batches: {len(self._epoch_loader(0).dataset)} episodes "  # pyright: ignore[reportArgumentType]
                f"with batch_size={config.batch_size} and drop_last=True. Raise "
                "data.num_clips to at least the batch size, or lower train.batch_size."
            )
        self.step = 0
        self.successful_updates = 0
        self.total_skips = 0  # D18: batches whose update was rejected
        self.consecutive_skips = 0

    def _validate_rollout_curriculum(self) -> None:
        """Reject schedules that could silently change the terminal objective."""
        curriculum = self.config.rollout_curriculum
        if curriculum is None:
            return
        if self.config.lambda_roll != 1.0:
            raise ValueError("rollout curriculum requires lambda_roll=1.0")
        if self.config.rollout_len is None:
            raise ValueError("rollout curriculum requires a terminal train.rollout_len")
        if not curriculum or curriculum[0][0] != 0:
            raise ValueError("rollout_curriculum must start at successful update 0")

        previous_start = -1
        previous_horizon = 0
        for index, (start, horizon) in enumerate(curriculum):
            if start <= previous_start:
                raise ValueError("rollout_curriculum starts must be strictly increasing")
            if horizon is None:
                if index != 0:
                    raise ValueError("only the first rollout_curriculum stage may disable rollout")
            else:
                if horizon < 2:
                    raise ValueError("curriculum rollout horizons must be >= 2")
                if horizon < previous_horizon:
                    raise ValueError("curriculum rollout horizons must be non-decreasing")
                if horizon == self.config.rollout_len and index != len(curriculum) - 1:
                    raise ValueError(
                        "terminal rollout_len may only appear in the final curriculum stage"
                    )
                previous_horizon = horizon
            previous_start = start

        if curriculum[-1][1] != self.config.rollout_len:
            raise ValueError("rollout_curriculum terminal horizon must equal train.rollout_len")

    # ------------------------------------------------------------- data ----
    def _epoch_loader(self, epoch: int) -> DataLoader[dict[str, Tensor]]:
        """Deterministic per-epoch shuffling so resume can replay the order."""
        generator = torch.Generator()
        generator.manual_seed(self.config.seed * 100_003 + epoch)
        workers = self.config.num_workers
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=self.config.prefetch_factor if workers > 0 else None,
        )

    def _batches(self) -> Iterator[dict[str, Tensor]]:
        """Endless batch stream; fast-forwards within the epoch on resume."""
        first_loader = self._epoch_loader(0)
        # steps_per_epoch >= 1 is guaranteed by __init__'s guard, so this is a
        # plain length — NOT max(len, 1), which is what used to turn an empty
        # loader into a silent infinite loop instead of an error.
        steps_per_epoch = len(first_loader)
        epoch = self.step // steps_per_epoch
        skip = self.step % steps_per_epoch
        while True:
            for index, batch in enumerate(self._epoch_loader(epoch)):
                if skip and index < skip:
                    continue
                yield batch
            skip = 0
            epoch += 1

    # ------------------------------------------------------------- steps ----
    # --------------------------------------------------- experiment seams ----
    # The visual-to-visual regime keeps this loop's objective, guards, checkpointing and resume
    # but changes three things: it reads frames, its dual sees a
    # variance-normalized constraint (Eq. 123), and it must step an EMA target
    # after each optimizer step. Those are the only overrides.
    def _current_rollout_len(self) -> int | None:
        """Return K for the next batch, indexed by accepted optimizer updates."""
        curriculum = self.config.rollout_curriculum
        if curriculum is None:
            return self.config.rollout_len
        horizon: int | None = None
        for start, stage_horizon in curriculum:
            if self.successful_updates < start:
                break
            horizon = stage_horizon
        return horizon

    def _sparsity_active(self) -> bool:
        """Enable path/GECO only when their calibrated constraint is live."""
        if not self.config.sparsity_enabled:
            return False
        if self.config.rollout_curriculum is None:
            return True
        return self._current_rollout_len() == self.config.rollout_len

    def _forward(self, batch: dict[str, Tensor], rollout_len: int | None) -> TransitionOutput:
        """Run the model on one batch (the state-to-state regime: true states)."""
        states = batch["states"].to(self.device)
        return self.model(
            states,
            context_len=self.config.context_len,
            rollout_len=rollout_len,
        )

    def _rollout_loss(self, output: TransitionOutput) -> Tensor:
        """Eq. 35; zero when the branch is off.

        The visual regimes reach this through the shared step but their outputs
        carry no rollout fields, so they return the disabled path unchanged.
        """
        prediction = getattr(output, "rollout_prediction", None)
        if prediction is None:
            return torch.zeros((), device=self.device)
        target = output.rollout_target
        assert target is not None
        horizon = prediction.shape[1]
        weights = self._rollout_weight_cache.get(horizon)
        if weights is None:
            weights = rollout_weights(horizon, device=self.device)
            self._rollout_weight_cache[horizon] = weights
        return weighted_rollout_mse(prediction, target, weights)

    def _branch_gradient_norm(self, loss: Tensor) -> float:
        """Global pre-clip norm of one loss branch without touching ``.grad``."""
        trainable_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        gradients = torch.autograd.grad(
            loss,
            trainable_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        norms = [gradient.detach().norm(2) for gradient in gradients if gradient is not None]
        if not norms:
            return 0.0
        return float(torch.stack(norms).norm(2))

    def _constraint(
        self, pred_loss: Tensor, logit_loss: Tensor, output: TransitionOutput
    ) -> Tensor:
        """The dual constraint excludes the path penalty (Eq. 13).

        ``pred_loss`` is the full prediction-side term the bound applies to,
        i.e. L_TF + λ_roll·L_roll under the hybrid §4.3 dual form — the caller
        scalarises the two bounds into one before calling this.
        """
        del output
        return (pred_loss + logit_loss).detach()

    def _after_optimizer_step(self, output: TransitionOutput) -> None:
        """Hook for post-update work (the visual-to-visual regime's EMA, Eq. 111)."""

    def _extra_metrics(self, output: TransitionOutput) -> dict[str, float]:
        """Experiment-specific metrics merged into the logged dict."""
        del output
        return {}

    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """One optimizer step over the hybrid Eq. 36; returns scalar metrics."""
        current_rollout_len = self._current_rollout_len()
        sparsity_active = self._sparsity_active()
        output = self._forward(batch, current_rollout_len)

        pred_loss = aligned_mse(output.prediction, output.target)  # L_TF, Eq. 32
        # Eq. 36: both prediction branches enter the objective, and (dual form,
        # §4.3) the same scalarised sum is what the bound applies to. Only the
        # weighted term is the quantity in the objective; the raw term is also
        # reported so a changing K cannot disguise an unstable chain.
        raw_rollout_loss = self._rollout_loss(output)  # Eq. 35
        rollout_loss = self.config.lambda_roll * raw_rollout_loss
        logit_loss = self.config.lambda_logit * output.logit_penalty
        total = pred_loss + rollout_loss + logit_loss
        if sparsity_active:
            total = total + self.lagrangian.penalty_weight * output.sparsity
        constraint = self._constraint(pred_loss + rollout_loss, logit_loss, output)

        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {self.step}: pred={pred_loss.item():.4g} "
                f"rollout={rollout_loss.item():.4g} "
                f"sparsity={output.sparsity.item():.4g}"
            )

        # Branch-specific pre-clip norms are diagnostic autograd traversals;
        # compute them only on steps that W&B will retain. The raw rollout norm
        # exposes an unstable K-step chain independently of lambda_roll.
        branch_grad_metrics: dict[str, float] = {}
        diagnostic_step = self.step + 1
        if diagnostic_step % self.config.log_every == 0 or diagnostic_step == self.config.steps:
            tf_grad_norm = self._branch_gradient_norm(pred_loss)
            branch_grad_metrics["health/grad_norm_tf"] = tf_grad_norm
            if getattr(output, "rollout_prediction", None) is not None:
                raw_rollout_grad_norm = self._branch_gradient_norm(raw_rollout_loss)
                branch_grad_metrics |= {
                    "health/grad_norm_rollout_raw": raw_rollout_grad_norm,
                    "health/grad_norm_rollout": self.config.lambda_roll * raw_rollout_grad_norm,
                }

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()  # pyright: ignore[reportUnknownMemberType]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        # D18 skip guard: clip_grad_norm_ returns the PRE-clip norm. A
        # non-finite norm means every gradient was already zeroed by the clip
        # coefficient; an absurd finite norm is the kick that starts an
        # explosion spiral. Either way, reject the whole update (optimizer AND
        # dual — a pathological batch must not jolt the moving average).
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
            self._after_optimizer_step(output)
            if sparsity_active:
                self.lagrangian.update(constraint)
            self.successful_updates += 1

        num_decoded = output.prediction.shape[1]
        # λ_roll·L_roll, as it enters the objective. Emitted ONLY when the branch
        # is live: the visual regimes share this step but carry no rollout, and a
        # constant-zero curve is noise on their W&B pages.
        rollout_metric = (
            {
                "loss/rollout_raw": raw_rollout_loss.item(),
                "loss/rollout": rollout_loss.item(),
            }
            if getattr(output, "rollout_prediction", None) is not None
            else {}
        )
        return (
            self._extra_metrics(output)
            | rollout_metric
            | branch_grad_metrics
            | {
                "loss/total": total.item(),
                "loss/pred": pred_loss.item(),
                "loss/logit": logit_loss.item(),
                "loss/sparsity": output.sparsity.item(),
                "attention/logit_penalty": output.logit_penalty.item(),
                "attention/mean_abs_logit": output.mean_abs_logit.item(),
                "attention/gate_entropy": output.gate_entropy.item(),
                "sparsity/constraint": constraint.item(),
                "sparsity/lambda": float(torch.exp(self.lagrangian.log_lambda).item()),
                # rho_path over the decoded state rows (Eq. 11); full-token density
                # stays a diagnostic (parameter rows are not decoded).
                "sparsity/path_density": (output.path_matrix[:, :num_decoded] >= 0.5)
                .float()
                .mean()
                .item(),
                "sparsity/path_density_full": (output.path_matrix >= 0.5).float().mean().item(),
                "sparsity/active": float(sparsity_active),
                "health/grad_norm": float(grad_norm.item()),
                "health/skipped_steps": float(self.total_skips),
                "schedule/successful_updates": float(self.successful_updates),
                "schedule/rollout_len": float(current_rollout_len or 0),
                "schedule/lambda_roll": self.config.lambda_roll,
            }
        )

    def train(self) -> dict[str, float]:
        """Run until ``config.steps``; returns the final step's metrics."""
        self.model.train()
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}
        batches = self._batches()
        while self.step < self.config.steps:
            rollout_len_before = self._current_rollout_len()
            metrics = self._train_step(next(batches))
            self.step += 1
            rollout_len_after = self._current_rollout_len()
            if rollout_len_after != rollout_len_before:
                next_stage = "tf" if rollout_len_after is None else f"k{rollout_len_after}"
                self.save_checkpoint(
                    out_dir / f"curriculum_success_{self.successful_updates}_before_{next_stage}.pt"
                )
            if self.step % self.config.log_every == 0 or self.step == self.config.steps:
                self.logger.log(self.step, metrics)
            if (
                self.config.eval_every is not None
                and self.eval_dataset is not None
                and (self.step % self.config.eval_every == 0 or self.step == self.config.steps)
            ):
                self.logger.log(self.step, self._eval_step())
            if self.step % self.config.checkpoint_every == 0:
                self.save_checkpoint(out_dir / "last.pt")
            if (
                self.config.checkpoint_keep_every is not None
                and self.step % self.config.checkpoint_keep_every == 0
            ):
                self.save_checkpoint(out_dir / f"step_{self.step}.pt")
        self.save_checkpoint(out_dir / "last.pt")
        return metrics

    def _eval_step(self) -> dict[str, float]:
        """Held-out identifiability metrics, prefixed for separate W&B charts."""
        assert self.eval_dataset is not None
        report = evaluate_identifiability(
            self.model,
            self.eval_dataset,
            batch_size=self.config.batch_size,
            device=self.config.device,
            context_len=self.config.context_len,
            lambda_logit=self.config.lambda_logit,
            # Periodic eval follows the live curriculum stage. Final/post-hoc
            # calibration deliberately uses config.rollout_len (terminal K).
            rollout_len=self._current_rollout_len(),
            lambda_roll=self.config.lambda_roll,
        )
        self.model.train()  # the harness switches to eval mode
        # Every run mode logs the FULL eval key set (2026-07-25, Jesse): in the
        # reference modes shd/path_density are constants, but constant curves
        # are cheap and their absence has previously hidden dead runs.
        return {f"eval/{key}": value for key, value in report.metrics.items()}

    # ------------------------------------------------------- checkpoints ----
    def save_checkpoint(self, path: Path) -> None:
        """Save model/optimizer/controller/step/RNG for exact resume."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lagrangian": self.lagrangian.state_dict(),
                "step": self.step,
                "successful_updates": self.successful_updates,
                "total_skips": self.total_skips,
                "consecutive_skips": self.consecutive_skips,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),  # noqa: NPY002
                "rng_torch": torch.get_rng_state(),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """Restore everything ``save_checkpoint`` wrote (exact resume)."""
        payload = torch.load(path, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.lagrangian.load_state_dict(payload["lagrangian"])
        self.step = int(payload["step"])
        self.total_skips = int(payload.get("total_skips", 0))
        self.successful_updates = int(
            payload.get("successful_updates", max(self.step - self.total_skips, 0))
        )
        self.consecutive_skips = int(payload.get("consecutive_skips", 0))
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])  # noqa: NPY002
        torch.set_rng_state(payload["rng_torch"])


__all__ = [
    "MetricLogger",
    "NoopLogger",
    "RolloutCurriculum",
    "TrainConfig",
    "Trainer",
    "seed_everything",
]
