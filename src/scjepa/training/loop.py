"""Plain-PyTorch training loop for SCJepa (module 6).

Re-expresses le-wm's JEPA training pattern (see third_party/lewm/PROVENANCE.md:
loss assembled in one place, ONE optimizer step over encoders + heads +
predictor jointly, regularizer as a swappable module call) in this project's
own stack — no Lightning, no EMA machinery, nothing architectural against
collapse (D3/D7).

Objective per step (D6 + SPARTAN App. A.2):

    total = prediction_mse(Ŝ_{t+1}, S_{t+1})
          + λ_logit · logit_penalty                                (Eq. 11)
          + λ_reg · [reg(context slots) + reg(target slots)]      (both branches)
          + (1/λ_s) · |Ā|                                          (if enabled)

with λ_s driven by the GECO-style dual update in ``SparsityLagrangian``. For
learned targets the dual compares the scale-free D17 constraint

    constraint = pred / Var(target batch, detached) + λ_logit · logit_penalty

against τ because raw MSE lives in a trainable representation. Literal GT
states instead use raw MSE, matching Baumgartner/SPARTAN's fixed observation-
space constraint. ``constraint_normalization=auto`` selects between them.
The ±sparsity ablation is the ``sparsity`` config toggle.

Checkpoints carry model/optimizer/controller/step/RNG state; resume is exact
(verified by test) — data order is reproduced by re-seeding the epoch generator
and fast-forwarding within the current epoch.
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
from scjepa.losses import (
    SlotRegularizer,
    prediction_constraint,
    prediction_mse,
    resolve_constraint_normalization,
    resolve_prediction_matching,
)
from scjepa.models.jepa import JepaOutput, SCJepa
from scjepa.models.state_jepa import StateJepa
from scjepa.training.lagrangian import SparsityLagrangian


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
    lambda_reg: float = 1.0
    sparsity_enabled: bool = True  # the ±sparsity ablation toggle
    sparsity_warmup_steps: int = 0
    sparsity_tau: float = 0.1
    sparsity_step_size: float = 1e-3
    sparsity_lambda_init: float = 1e3
    # Upper clamp on the dual λ (D22): GECO's ascent is unbounded while the
    # constraint sits above τ, but the descent is rate-limited by how far the
    # model can UNDERSHOOT τ — run maj7im56 peaked at λ=3.8e5 and pruning
    # never engaged in 300k steps. Clamping caps the overshoot so the
    # reversal is immediate once τ is crossed.
    sparsity_lambda_max: float = 1e6
    sparsity_momentum: float = 0.99
    # Attention-logit regularisation (Baumgartner Eq. 11): keeps softmax
    # gradients alive during the pruning phase; 0 disables. When enabled it is
    # part of the Lagrangian constraint (their Eq. 9), so calibrate tau with it
    # on the configured raw/normalized scale reported by the eval harness.
    lambda_logit: float = 0.0
    regularizer: str = "visreg"  # D3: "visreg" | "sigreg"
    num_projections: int = 256
    seed: int = 0
    device: str = "cpu"
    input_key: str = "frames"  # "frames" (vision, SCJepa) | "states" (StateJepa)
    # "auto": aligned MSE for StateJepa's tracked object rows; visual learned-
    # slot targets remain unordered and use Hungarian matching.
    prediction_matching: str = "auto"  # "auto" | "aligned" | "hungarian"
    # "auto": raw MSE on literal GT states; variance-normalized MSE when the
    # target representation itself is trainable.
    constraint_normalization: str = "auto"  # auto | raw | target_variance
    # Pool one S^ph from the first context_len steps; the remaining
    # K = L - context_len transitions are predicted. None -> L-1 (K = 1).
    context_len: int | None = None
    # D16 autoregressive rollout: chains of this length feed predictions back
    # (must divide K); None -> one chain over all K (paper-literal S_Tp).
    rollout_horizon: int | None = None
    # Periodic held-out identifiability eval (W&B curves "eval/*", the analog
    # of Baumgartner Fig. 17's MCC-over-steps). None = off. Requires an
    # eval_dataset passed to the Trainer; states regime only (slot i = object i).
    eval_every: int | None = None
    # D18 grad-spike guards (post-mortem of run 7wupt6pw, 2026-07-17): a rare
    # batch kicked the predictor into a >1 per-step rollout gain, the Tp=30
    # chain amplified it to a FINITE ~1e30 loss (passes the isfinite guard),
    # BPTT overflowed to grad_norm=inf, and clip_grad_norm_'s inf denominator
    # silently multiplied every gradient by ZERO — the run finished 230k steps
    # as a frozen zombie. Guards: skip the optimizer step (and dual update)
    # when the pre-clip grad norm is non-finite or above the threshold; raise
    # after too many consecutive skips (weights are then already broken —
    # fail loudly, resume from a rolling checkpoint).
    grad_skip_threshold: float = 1e3
    # Weights are FROZEN during consecutive skips, so patience is free — the
    # model cannot degrade while the guard holds, and every retry is a fresh
    # draw (epochs reshuffle batch compositions; gates resample per forward).
    # The counter resets on ANY calm batch, so reaching N consecutive means
    # the calm-batch rate is below ~1/N — at 2000 (~8 bounce epochs, ~10 min)
    # that is a dead run, not an unlucky one. The original 50 executed run
    # 0ta5ymcw mid-episode (its first episode recovered after 149 skips
    # interleaved with calm batches at a ~50% pass rate).
    grad_skip_max_consecutive: int = 2000
    log_every: int = 10
    checkpoint_every: int = 200
    # Also keep a step-tagged checkpoint every N steps (None = only last.pt).
    # last.pt is OVERWRITTEN every checkpoint_every, so without this a late
    # failure leaves no healthy state to resume from (the 7wupt6pw lesson).
    checkpoint_keep_every: int | None = None
    out_dir: str = "outputs"


class Trainer:
    """Explicit single-device training loop; fails loudly, resumes exactly."""

    def __init__(
        self,
        model: SCJepa | StateJepa,
        dataset: Dataset[dict[str, Tensor]],
        config: TrainConfig,
        logger: MetricLogger | None = None,
        eval_dataset: Dataset[dict[str, Tensor]] | None = None,
    ) -> None:
        """Build optimizer, regularizer, and sparsity controller around the model."""
        seed_everything(config.seed)
        if config.sparsity_warmup_steps < 0:
            raise ValueError("sparsity_warmup_steps must be non-negative")
        self.config = config
        self.eval_dataset = eval_dataset
        self.prediction_matching = resolve_prediction_matching(
            config.prediction_matching,
            object_aligned=isinstance(model, StateJepa),
        )
        self.constraint_normalization = resolve_constraint_normalization(
            config.constraint_normalization,
            gt_states=isinstance(model, StateJepa) and model.gt_states,
        )
        if config.eval_every is not None:
            if eval_dataset is None:
                raise ValueError("eval_every set but no eval_dataset provided")
            if config.input_key != "states":
                raise ValueError(
                    "periodic identifiability eval requires input_key='states' "
                    "(vision-regime slots are unaligned; see scjepa/eval/harness.py)"
                )
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.logger: MetricLogger = logger if logger is not None else NoopLogger()
        if config.regularizer not in ("visreg", "sigreg"):
            raise ValueError(f"unknown regularizer {config.regularizer!r}")
        self.regularizer = SlotRegularizer(
            kind=config.regularizer,  # pyright: ignore[reportArgumentType]
            num_projections=config.num_projections,
        ).to(self.device)
        self.lagrangian = SparsityLagrangian(
            tau=config.sparsity_tau,
            step_size=config.sparsity_step_size,
            lambda_init=config.sparsity_lambda_init,
            lambda_max=config.sparsity_lambda_max,
            momentum=config.sparsity_momentum,
        ).to(self.device)
        # ONE optimizer over everything (D7: encoders + heads + predictor jointly).
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.step = 0
        self.total_skips = 0  # D18: batches whose update was rejected
        self.consecutive_skips = 0

    # ------------------------------------------------------------- data ----
    def _epoch_loader(self, epoch: int) -> DataLoader[dict[str, Tensor]]:
        """Deterministic per-epoch shuffling so resume can replay the order."""
        generator = torch.Generator()
        generator.manual_seed(self.config.seed * 100_003 + epoch)
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )

    def _batches(self) -> Iterator[dict[str, Tensor]]:
        """Endless batch stream; fast-forwards within the epoch on resume."""
        first_loader = self._epoch_loader(0)
        steps_per_epoch = max(len(first_loader), 1)
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
    def _train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """One optimizer step over the full objective; returns scalar metrics."""
        inputs = batch[self.config.input_key].to(self.device)
        aux = batch.get("aux")
        output: JepaOutput = self.model(
            inputs,
            aux.to(self.device) if aux is not None else None,
            context_len=self.config.context_len,
            rollout_horizon=self.config.rollout_horizon,
        )

        pred_loss = prediction_mse(
            output.prediction, output.target_slots, self.prediction_matching
        )
        reg_loss = self.regularizer(output.context_slots) + self.regularizer(output.target_slots)
        logit_loss = self.config.lambda_logit * output.logit_penalty
        # Gradient objective (Baumgartner Eq. 10): raw pred + logit
        # regularisation. VISReg stays OUTSIDE the constraint — it is the
        # collapse/scale anchor (D12) and must not trade off against sparsity.
        total = pred_loss + logit_loss + self.config.lambda_reg * reg_loss
        sparsity_active = (
            self.config.sparsity_enabled and self.step >= self.config.sparsity_warmup_steps
        )
        if sparsity_active:
            total = total + self.lagrangian.penalty_weight * output.sparsity

        with torch.no_grad():
            # Collapse indicator: per-dimension std of target slots (D3 — nothing
            # architectural prevents collapse, so this must be watched).
            slot_std = output.target_slots.reshape(-1, output.target_slots.shape[-1]).std(dim=0)
            # Learned target spaces use D17's detached variance normalization;
            # literal GT states already are Baumgartner's fixed ruler and use
            # raw MSE. This quantity only drives the dual update and logging.
            target_var = slot_std.pow(2).mean().clamp_min(1e-6)
            constraint_loss = prediction_constraint(
                pred_loss.detach(), target_var, self.constraint_normalization
            ) + logit_loss.detach()

        if not torch.isfinite(total):
            raise RuntimeError(
                f"non-finite loss at step {self.step}: pred={pred_loss.item():.4g} "
                f"reg={reg_loss.item():.4g} sparsity={output.sparsity.item():.4g}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()  # pyright: ignore[reportUnknownMemberType]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        # D18 skip guard: clip_grad_norm_ returns the PRE-clip norm. A
        # non-finite norm means clip's coefficient max_norm/inf is 0 — every
        # gradient is already zeroed and stepping would freeze the model
        # silently; an absurd finite norm is the batch kick that starts the
        # explosion spiral. Either way: reject this batch's update entirely
        # (optimizer AND dual — a pathological batch must not jolt the EMA).
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
                    f"{self.step} (grad_norm={float(grad_norm):.3g}, threshold="
                    f"{self.config.grad_skip_threshold:.3g}): the model is no longer "
                    "trainable — weights are likely already broken. Resume from the "
                    "last healthy step-tagged checkpoint instead of continuing."
                )
        else:
            self.consecutive_skips = 0
            self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if sparsity_active:
                self.lagrangian.update(constraint_loss)

        return {
            "loss/total": total.item(),
            "loss/pred": pred_loss.item(),
            "loss/reg": reg_loss.item(),
            "loss/sparsity": output.sparsity.item(),
            "loss/logit": logit_loss.item(),
            "sparsity/constraint": constraint_loss.item(),
            "sparsity/lambda": float(torch.exp(self.lagrangian.log_lambda).item()),
            "sparsity/active": float(sparsity_active),
            # Primary density covers the same decoded state rows optimized by
            # output.sparsity. Keep the full-token density only as a diagnostic:
            # final parameter rows are not decoded and need not prune.
            "sparsity/path_density": (
                output.path_matrix[:, : output.prediction.shape[1]] >= 0.5
            )
            .float()
            .mean()
            .item(),
            "sparsity/path_density_full": (output.path_matrix >= 0.5).float().mean().item(),
            "health/target_slot_std_mean": slot_std.mean().item(),
            "health/target_slot_std_min": slot_std.min().item(),
            "health/grad_norm": float(grad_norm.item()),
            "health/skipped_steps": float(self.total_skips),
        }

    def train(self) -> dict[str, float]:
        """Run until ``config.steps``; returns the final step's metrics."""
        self.model.train()
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}
        batches = self._batches()
        while self.step < self.config.steps:
            metrics = self._train_step(next(batches))
            self.step += 1
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
                # D18: last.pt gets overwritten — keep dated fallbacks so a
                # late-run failure is a resume, not a rerun.
                self.save_checkpoint(out_dir / f"step_{self.step}.pt")
        self.save_checkpoint(out_dir / "last.pt")
        return metrics

    def _eval_step(self) -> dict[str, float]:
        """Held-out identifiability metrics, prefixed for separate W&B charts."""
        assert self.eval_dataset is not None
        report = evaluate_identifiability(
            self.model,
            self.eval_dataset,
            input_key=self.config.input_key,
            batch_size=self.config.batch_size,
            device=self.config.device,
            context_len=self.config.context_len,
            rollout_horizon=self.config.rollout_horizon,
            lambda_logit=self.config.lambda_logit,
            prediction_matching=self.prediction_matching,
            constraint_normalization=self.constraint_normalization,
        )
        self.model.train()  # the harness switches to eval mode
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
        self.total_skips = int(payload.get("total_skips", 0))  # absent pre-D18
        self.consecutive_skips = int(payload.get("consecutive_skips", 0))
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])  # noqa: NPY002
        torch.set_rng_state(payload["rng_torch"])


__all__ = ["MetricLogger", "NoopLogger", "TrainConfig", "Trainer", "seed_everything"]
