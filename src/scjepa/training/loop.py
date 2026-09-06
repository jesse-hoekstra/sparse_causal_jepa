"""Training loop for the state-to-state SCJEPA regime.

The fixed predictive objective is::

    L_pred = L_TF + lambda_rollout_t2 * L_AR2

where ``L_TF`` uses every true suffix transition and ``L_AR2`` is the mean
endpoint error from independently sampled, true-anchored two-step windows. The
GECO constraint sees that same predictive sum plus the existing logit term. No
rollout curriculum, horizon schedule, recurrent gradient cut, or full-rollout
backpropagation exists in this loop. K=30 is evaluated only by the held-out,
no-gradient harness.
"""

import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from scjepa.eval.harness import evaluate_identifiability
from scjepa.losses import aligned_mse, rollout_t2_endpoint_mse
from scjepa.models.state_to_state import StateToStateModel, TransitionOutput
from scjepa.models.visual_to_visual import VISUAL_CONSTRAINT_VERSION, VisualToVisualModel
from scjepa.training.distributed import (
    GlobalBatchSampler,
    any_rank,
    distributed_barrier,
    gather_rank_objects,
    is_main_process,
    mean_metrics,
    mean_tensor,
    rank,
    world_size,
)
from scjepa.training.lagrangian import SparsityLagrangian


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - external libraries consume the global RNG
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


class MetricLogger(Protocol):
    """Minimal logging interface so W&B never blocks tests or CI."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record one step's scalar metrics."""
        ...


class NoopLogger:
    """Logger that drops everything."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Drop the metrics."""


@dataclass
class TrainConfig:
    """Fixed training protocol and ordinary run-health configuration."""

    steps: int
    batch_size: int
    lr: float = 5e-5
    grad_clip: float = 1.0
    sparsity_enabled: bool = True
    sparsity_tau: float = 0.1
    sparsity_step_size: float = 2e-2
    sparsity_lambda_init: float = 1e4
    sparsity_momentum: float = 0.99
    lambda_logit: float = 0.0
    # Both experiments use endpoint-only T=2 windows, without a schedule.
    lambda_rollout_t2: float = 1.0
    num_rollout_t2_anchors: int = 8
    rollout_t2_horizon: int = 2
    # No-gradient held-out trajectory diagnostic.
    oe_eval_horizon: int = 30
    oe_tolerance_nrmse: float = 0.10
    oe_coordinate_std: tuple[float, ...] | None = None
    seed: int = 0
    device: str = "cpu"
    context_len: int | None = None
    eval_every: int | None = None
    grad_skip_threshold: float = 1e3
    grad_skip_max_consecutive: int = 2000
    num_workers: int = 0
    prefetch_factor: int = 4
    log_every: int = 10
    checkpoint_every: int = 200
    checkpoint_keep_every: int | None = None
    out_dir: str = "outputs"


class Trainer:
    """Train on one device or synchronized replicas, with a fixed global batch."""

    def __init__(
        self,
        model: StateToStateModel | VisualToVisualModel,
        dataset: Dataset[dict[str, Tensor]],
        config: TrainConfig,
        logger: MetricLogger | None = None,
        eval_dataset: Dataset[dict[str, Tensor]] | None = None,
    ) -> None:
        """Build the optimizer and dual controller around a fixed objective."""
        # Model construction uses one common seed; stochastic episodes, gates,
        # and T=2 anchors then receive an independent stream on each rank.
        seed_everything(config.seed + rank() * 1_000_003)
        if config.eval_every is not None and eval_dataset is None:
            raise ValueError("eval_every set but no eval_dataset provided")
        self.config = config
        self.eval_dataset = eval_dataset
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.dataset = dataset
        self.logger: MetricLogger = logger if logger is not None else NoopLogger()
        self._validate_fixed_protocol()
        self._forward_model: nn.Module = self.model
        if world_size() > 1:
            self._forward_model = DistributedDataParallel(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                # Buffers are fixed architecture/codebooks. EMA updates follow
                # synchronized optimizer updates; no forward broadcast needed.
                broadcast_buffers=False,
                # SAVi retains unused upstream prior parameters, and reference
                # predictors can bypass gates. Detect these rather than hang.
                find_unused_parameters=True,
            )
        self.lagrangian = SparsityLagrangian(
            tau=config.sparsity_tau,
            step_size=config.sparsity_step_size,
            lambda_init=config.sparsity_lambda_init,
            momentum=config.sparsity_momentum,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        batches_per_epoch = len(self._epoch_loader(0))
        if batches_per_epoch == 0:
            raise ValueError(
                f"dataset yields no batches: {len(self._epoch_loader(0).dataset)} episodes "  # pyright: ignore[reportArgumentType]
                f"with batch_size={config.batch_size} and drop_last=True. Raise "
                "data.num_clips to at least the batch size, or lower train.batch_size."
            )
        self.step = 0
        self.total_skips = 0
        self.consecutive_skips = 0

    def _validate_fixed_protocol(self) -> None:
        """Reject invalid fixed T=2/OE settings before the first batch."""
        config = self.config
        if config.rollout_t2_horizon != 2:
            raise ValueError(f"rollout_t2_horizon must equal 2, got {config.rollout_t2_horizon}")
        if not math.isfinite(config.lambda_rollout_t2) or config.lambda_rollout_t2 < 0:
            raise ValueError("lambda_rollout_t2 must be finite and non-negative")
        if config.num_rollout_t2_anchors < 1:
            raise ValueError("num_rollout_t2_anchors must be positive")
        if config.oe_eval_horizon < 1:
            raise ValueError("oe_eval_horizon must be positive")
        if not math.isfinite(config.oe_tolerance_nrmse) or config.oe_tolerance_nrmse < 0:
            raise ValueError("oe_tolerance_nrmse must be finite and non-negative")
        if config.eval_every is not None and isinstance(self.model, StateToStateModel):
            scales = config.oe_coordinate_std
            if scales is None or not scales:
                raise ValueError("state-to-state evaluation requires fixed oe_coordinate_std")
            if any(not math.isfinite(value) or value <= 0 for value in scales):
                raise ValueError("oe_coordinate_std must contain finite positive values")

    # ------------------------------------------------------------- data ----
    def _epoch_loader(self, epoch: int) -> DataLoader[dict[str, Tensor]]:
        """Create deterministic per-epoch shuffling so resume replays order."""
        generator = torch.Generator()
        generator.manual_seed(self.config.seed * 100_003 + epoch)
        workers = self.config.num_workers
        if world_size() > 1:
            return DataLoader(
                self.dataset,
                batch_sampler=GlobalBatchSampler(
                    dataset_size=len(self.dataset),  # pyright: ignore[reportArgumentType]
                    batch_size=self.config.batch_size,
                    seed=self.config.seed,
                    epoch=epoch,
                ),
                generator=generator,
                num_workers=workers,
                pin_memory=self.device.type == "cuda",
                persistent_workers=workers > 0,
                prefetch_factor=self.config.prefetch_factor if workers > 0 else None,
                # NCCL is not fork-safe after group initialization.
                multiprocessing_context="spawn" if workers > 0 else None,
            )
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            prefetch_factor=self.config.prefetch_factor if workers > 0 else None,
        )

    def _batches(self) -> Iterator[dict[str, Tensor]]:
        """Yield an endless batch stream, fast-forwarding on resume."""
        first_loader = self._epoch_loader(0)
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

    # ------------------------------------------------------------ hooks ----
    def _sparsity_active(self) -> bool:
        """The fixed protocol applies sparsity from the first sparse update."""
        return self.config.sparsity_enabled

    def _forward(self, batch: dict[str, Tensor]) -> TransitionOutput:
        """Run teacher forcing and, when weighted, exactly eight T=2 windows."""
        states = batch["states"].to(self.device, non_blocking=self.device.type == "cuda")
        # A zero coefficient bypasses sampling and both auxiliary predictor
        # calls. This preserves TF tensors and the post-forward RNG state.
        anchors = self.config.num_rollout_t2_anchors if self.config.lambda_rollout_t2 > 0 else 0
        return cast(
            TransitionOutput,
            self._forward_model(
                states,
                context_len=self.config.context_len,
                num_rollout_t2_anchors=anchors,
            ),
        )

    def _teacher_forcing_loss(self, output: TransitionOutput) -> Tensor:
        """Teacher-forced prediction loss; subclasses may change alignment."""
        return aligned_mse(output.prediction, output.target)

    def _auxiliary_loss(self, output: TransitionOutput) -> Tensor:
        """Endpoint-only T=2 loss, or exact zero for the lambda=0 ablation."""
        if output.rollout_t2_prediction is None:
            return torch.zeros((), device=self.device)
        assert output.rollout_t2_target is not None
        return rollout_t2_endpoint_mse(output.rollout_t2_prediction, output.rollout_t2_target)

    def _auxiliary_weight(self) -> float:
        """Return the fixed state-to-state T=2 coefficient."""
        return self.config.lambda_rollout_t2

    def _constraint(self, predictive_loss: Tensor, logit_loss: Tensor, output: object) -> Tensor:
        """GECO bound: predictive sum plus logit term, excluding path/OE."""
        del output
        return mean_tensor(predictive_loss + logit_loss)

    def _after_optimizer_step(self, output: TransitionOutput) -> None:
        """Hook for the visual-to-visual EMA update."""

    def _extra_metrics(self, output: TransitionOutput) -> dict[str, float]:
        """Return experiment-specific metrics."""
        del output
        return {}

    def _predictive_metrics(
        self,
        teacher_forcing: Tensor,
        raw_auxiliary: Tensor,
        weighted_auxiliary: Tensor,
        total: Tensor,
    ) -> dict[str, float]:
        """Minimal state-to-state loss logging requested by the fixed protocol."""
        return {
            "train/loss_teacher_forcing": teacher_forcing.item(),
            "train/loss_rollout_t2_raw": raw_auxiliary.item(),
            "train/loss_rollout_t2_weighted": weighted_auxiliary.item(),
            "train/loss_total": total.item(),
        }

    def _branch_gradient_metrics(
        self,
        teacher_forcing: Tensor,
        weighted_auxiliary: Tensor,
        auxiliary_enabled: bool,
    ) -> dict[str, float]:
        """Compute branch norms only on logging steps to limit retained work."""
        metrics = {"train/grad_norm_teacher_forcing": self._branch_gradient_norm(teacher_forcing)}
        if auxiliary_enabled:
            metrics["train/grad_norm_rollout_t2_weighted"] = self._branch_gradient_norm(
                weighted_auxiliary
            )
        return metrics

    def _branch_gradient_norm(self, loss: Tensor) -> float:
        """Global pre-clip norm of one branch without changing ``.grad``."""
        parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        gradients = cast(
            tuple[Tensor | None, ...],
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            ),
        )
        squared_norm: Tensor | None = None
        for gradient in gradients:
            if gradient is None:
                continue
            contribution = gradient.detach().square().sum()
            squared_norm = contribution if squared_norm is None else squared_norm + contribution
        return 0.0 if squared_norm is None else float(squared_norm.sqrt())

    # ------------------------------------------------------------- steps ----
    def _train_step(
        self, batch: dict[str, Tensor], *, collect_metrics: bool = True
    ) -> dict[str, float]:
        """Execute one optimizer attempt; materialize metrics only when requested.

        Loss/gradient rejection and optimizer, EMA, and GECO updates always run.
        Direct callers retain scalar results by default; the main loop avoids
        monitoring reductions and device-to-host copies on unlogged steps.
        """
        output = self._forward(batch)
        teacher_forcing = self._teacher_forcing_loss(output)
        raw_auxiliary = self._auxiliary_loss(output)
        weighted_auxiliary = self._auxiliary_weight() * raw_auxiliary
        predictive_loss = teacher_forcing + weighted_auxiliary
        logit_loss = self.config.lambda_logit * output.logit_penalty
        total = predictive_loss + logit_loss
        sparsity_active = self._sparsity_active()
        if sparsity_active:
            total = total + self.lagrangian.penalty_weight * output.sparsity
        constraint = self._constraint(predictive_loss, logit_loss, output)

        if any_rank(~torch.isfinite(total), self.device):
            raise RuntimeError(
                f"non-finite loss on at least one rank at step {self.step}: "
                f"local tf={teacher_forcing.item():.4g} "
                f"rollout_t2={weighted_auxiliary.item():.4g} "
                f"sparsity={output.sparsity.item():.4g}"
            )

        branch_grad_metrics: dict[str, float] = {}
        diagnostic_step = self.step + 1
        log_step = (
            diagnostic_step % self.config.log_every == 0 or diagnostic_step == self.config.steps
        )
        if collect_metrics and world_size() == 1 and log_step:
            # DDP requires .backward() accumulation and does not support these
            # extra autograd.grad traversals. Global total gradient norm is
            # still logged in every distributed run.
            branch_grad_metrics = self._branch_gradient_metrics(
                teacher_forcing,
                weighted_auxiliary,
                getattr(output, "rollout_t2_prediction", None) is not None,
            )

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()  # pyright: ignore[reportUnknownMemberType]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        skip = any_rank(
            ~torch.isfinite(grad_norm) | (grad_norm > self.config.grad_skip_threshold),
            self.device,
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

        if not collect_metrics:
            return {}

        # The episode-level parameter width names the decoded state-token rows
        # in both experiments, independently of flattened transition axes.
        num_decoded = output.causal_params.shape[1]
        return (
            # Collapse spectra and distributed gathers are monitoring only;
            # calculate them when their values will actually be logged.
            (self._extra_metrics(output) if log_step else {})
            | self._predictive_metrics(
                teacher_forcing,
                raw_auxiliary,
                weighted_auxiliary,
                total,
            )
            | branch_grad_metrics
            | {
                "loss/logit": logit_loss.item(),
                "loss/sparsity": output.sparsity.item(),
                "attention/logit_penalty": output.logit_penalty.item(),
                "attention/mean_abs_logit": output.mean_abs_logit.item(),
                "attention/gate_entropy": output.gate_entropy.item(),
                "sparsity/constraint": constraint.item(),
                "sparsity/lambda": float(torch.exp(self.lagrangian.log_lambda)),
                "sparsity/path_density": (output.path_matrix[:, :num_decoded] >= 0.5)
                .float()
                .mean()
                .item(),
                "sparsity/path_density_full": (output.path_matrix >= 0.5).float().mean().item(),
                "sparsity/active": float(sparsity_active),
                "health/grad_norm": float(grad_norm),
                "health/skipped_steps": float(self.total_skips),
            }
        )

    def train(self) -> dict[str, float]:
        """Run through ``config.steps`` attempted batches."""
        self.model.train()
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics: dict[str, float] = {}
        batches = self._batches()
        self._synchronize_device()
        interval_started = time.perf_counter()
        interval_step = self.step
        interval_training_seconds = 0.0
        while self.step < self.config.steps:
            upcoming_step = self.step + 1
            log_step = (
                upcoming_step % self.config.log_every == 0 or upcoming_step == self.config.steps
            )
            metrics = self._train_step(next(batches), collect_metrics=log_step)
            self.step += 1
            eval_step = (
                self.config.eval_every is not None
                and self.eval_dataset is not None
                and (self.step % self.config.eval_every == 0 or self.step == self.config.steps)
            )
            rolling_checkpoint = self.step % self.config.checkpoint_every == 0
            kept_checkpoint = (
                self.config.checkpoint_keep_every is not None
                and self.step % self.config.checkpoint_keep_every == 0
            )
            pause_training_timer = log_step or eval_step or rolling_checkpoint or kept_checkpoint
            if pause_training_timer:
                # CUDA launches are asynchronous. Finish the measured training
                # work before pausing, without imposing a new per-step sync.
                self._synchronize_device()
                interval_training_seconds += time.perf_counter() - interval_started
            if log_step:
                metrics = mean_metrics(metrics, self.device)
                seconds_per_step = interval_training_seconds / (self.step - interval_step)
                metrics["train/seconds_per_step"] = seconds_per_step
                metrics["train/episodes_per_second"] = self.config.batch_size / seconds_per_step
                if is_main_process():
                    self.logger.log(self.step, metrics)
                interval_training_seconds, interval_step = 0.0, self.step
            if eval_step:
                # Evaluation uses the unwrapped model and the original global
                # batch size, preserving dense calibration's variance ruler.
                distributed_barrier()
                if is_main_process():
                    self.logger.log(self.step, self._eval_step())
                distributed_barrier()
            if rolling_checkpoint:
                self.save_checkpoint(out_dir / "last.pt")
            if kept_checkpoint:
                self.save_checkpoint(out_dir / f"step_{self.step}.pt")
            if pause_training_timer:
                # Evaluation/checkpoint/logging time is excluded, including any
                # queued CUDA work from those operations. Data loading stays in.
                self._synchronize_device()
                interval_started = time.perf_counter()
        self.save_checkpoint(out_dir / "last.pt")
        return metrics

    def _synchronize_device(self) -> None:
        """Finish pending CUDA work only when starting or pausing a timed interval."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _eval_step(self) -> dict[str, float]:
        """Run deterministic held-out identifiability and K=30 OE evaluation."""
        assert self.eval_dataset is not None
        report = evaluate_identifiability(
            cast(StateToStateModel, self.model),
            self.eval_dataset,
            batch_size=self.config.batch_size,
            device=self.config.device,
            context_len=self.config.context_len,
            lambda_logit=self.config.lambda_logit,
            lambda_rollout_t2=self.config.lambda_rollout_t2,
            num_rollout_t2_anchors=self.config.num_rollout_t2_anchors,
            rollout_t2_horizon=self.config.rollout_t2_horizon,
            oe_eval_horizon=self.config.oe_eval_horizon,
            oe_tolerance_nrmse=self.config.oe_tolerance_nrmse,
            oe_coordinate_std=self.config.oe_coordinate_std,
        )
        self.model.train()
        return {f"eval/{key}": value for key, value in report.metrics.items()}

    # ------------------------------------------------------- checkpoints ----
    def save_checkpoint(self, path: Path) -> None:
        """Collect each rank's RNG and write one portable checkpoint on rank zero.

        All ranks must call this method together in distributed training.
        The model is saved unwrapped so ordinary single-device evaluation can
        load it directly. Resuming training requires the same world/global batch.
        """
        local_rng = {
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),  # noqa: NPY002
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
        }
        rank_rng = gather_rank_objects(local_rng)
        if is_main_process():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lagrangian": self.lagrangian.state_dict(),
                    "step": self.step,
                    "total_skips": self.total_skips,
                    "consecutive_skips": self.consecutive_skips,
                    "oe_coordinate_std": self.config.oe_coordinate_std,
                    **local_rng,
                    "rank_rng": rank_rng,
                    "world_size": world_size(),
                    "global_batch_size": self.config.batch_size,
                    **(
                        {"visual_constraint_version": VISUAL_CONSTRAINT_VERSION}
                        if isinstance(self.model, VisualToVisualModel)
                        else {}
                    ),
                },
                temporary,
            )
            temporary.replace(path)
        # Returning means all ranks can immediately load the complete file.
        distributed_barrier()

    def load_checkpoint(self, path: Path) -> None:
        """Restore everything written by :meth:`save_checkpoint`."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            isinstance(self.model, VisualToVisualModel)
            and payload.get("visual_constraint_version") != VISUAL_CONSTRAINT_VERSION
        ):
            raise ValueError(
                "visual checkpoint uses a legacy or unknown constraint; "
                "start a fresh raw-constraint run and recalibrate tau (D42)"
            )
        if int(payload.get("world_size", 1)) != world_size():
            raise ValueError("exact training resume requires the checkpoint's world_size")
        if int(payload.get("global_batch_size", self.config.batch_size)) != self.config.batch_size:
            raise ValueError("exact training resume requires the checkpoint's global_batch_size")
        checkpoint_scales = payload.get("oe_coordinate_std")
        configured_scales = self.config.oe_coordinate_std
        if checkpoint_scales is not None:
            saved_scale_tensor = torch.as_tensor(checkpoint_scales, dtype=torch.float64)
            configured_scale_tensor = (
                torch.as_tensor(configured_scales, dtype=torch.float64)
                if configured_scales is not None
                else None
            )
            if (
                configured_scale_tensor is None
                or saved_scale_tensor.shape != configured_scale_tensor.shape
                or not torch.allclose(
                    saved_scale_tensor,
                    configured_scale_tensor,
                    rtol=1e-7,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "checkpoint oe_coordinate_std differs from the resolved training set"
                )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.lagrangian.load_state_dict(payload["lagrangian"])
        self.step = int(payload["step"])
        self.total_skips = int(payload.get("total_skips", 0))
        self.consecutive_skips = int(payload.get("consecutive_skips", 0))
        rng_state = cast(
            dict[str, Any], payload["rank_rng"][rank()] if "rank_rng" in payload else payload
        )
        random.setstate(rng_state["rng_python"])
        np.random.set_state(rng_state["rng_numpy"])  # noqa: NPY002
        torch.set_rng_state(rng_state["rng_torch"])
        cuda_rng = rng_state.get("rng_cuda")
        if cuda_rng is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(cuda_rng, self.device)


__all__ = ["MetricLogger", "NoopLogger", "TrainConfig", "Trainer", "seed_everything"]
