"""Generic latent-prediction training, accumulation, EMA, and portable checkpoints."""

import math
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sized
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from scjepa.data.compatibility import validate_model_dataset
from scjepa.data.identity import dataset_fingerprint
from scjepa.losses import RegularizerKind, RepresentationRegularizer, prediction_mse
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.distributed import (
    GlobalBatchSampler,
    any_rank,
    distributed_barrier,
    gather_batch_tensor,
    gather_rank_objects,
    is_main_process,
    mean_metrics,
    mean_tensor,
    rank,
    world_size,
)
from scjepa.training.lagrangian import GECOConfig, SparsityLagrangian
from scjepa.training.rng import preserve_rng


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - external libraries consume the global RNG
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


class MetricLogger(Protocol):
    """Small optional logging interface."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Record scalar metrics."""
        ...


class NoopLogger:
    """Logger that drops metrics."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Drop metrics."""


@dataclass
class ObjectiveConfig:
    """Explicit additive loss weights; all auxiliary terms are opt-in."""

    state_sparsity_weight: float = 0.0
    action_sparsity_weight: float = 0.0
    parameter_sparsity_weight: float = 0.0
    logit_weight: float = 0.0
    state_regularizer: str = "none"
    parameter_regularizer: str = "none"
    state_regularizer_weight: float = 0.0
    parameter_regularizer_weight: float = 0.0
    regularizer_num_projections: int = 256

    def __post_init__(self) -> None:
        """Reject invalid weights and silently ineffective regularization settings."""
        for name, value in asdict(self).items():
            if name.endswith("_weight") and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        for kind, weight in (
            (self.state_regularizer, self.state_regularizer_weight),
            (self.parameter_regularizer, self.parameter_regularizer_weight),
        ):
            if kind not in {"none", "sigreg"}:
                raise ValueError("regularizer must be none or sigreg")
            if kind == "none" and weight != 0:
                raise ValueError("a nonzero regularizer weight requires sigreg")
        if self.regularizer_num_projections < 1:
            raise ValueError("regularizer_num_projections must be positive")


class TrainingObjective(nn.Module):
    """JEPA base loss with fixed or GECO-controlled independent graph penalties.

    Representation statistics use the local batch on each distributed rank.
    State samples have shape ``(T, B, d_z)``; parameter samples ``(1, B, d_theta)``.
    Paired episodes use ``(2T, B, d_z)`` and ``(2, B, d_theta)``: each view has
    one sample per independent parameter group, rather than pooling both views.
    Parameter regularization requires unique episode IDs, validated by Trainer
    across all ranks. It never counts repeated times as new parameter samples.
    """

    def __init__(self, config: ObjectiveConfig, geco: GECOConfig | None = None) -> None:
        """Construct SIGReg and an optional controller, without updating its state."""
        super().__init__()
        self.config = config
        self.state_regularizer = RepresentationRegularizer(
            cast(RegularizerKind, config.state_regularizer), config.regularizer_num_projections
        )
        self.parameter_regularizer = RepresentationRegularizer(
            cast(RegularizerKind, config.parameter_regularizer), config.regularizer_num_projections
        )
        self.geco: SparsityLagrangian | None = None
        if geco is not None and geco.enabled:
            assert geco.tau is not None
            self.geco = SparsityLagrangian(
                tau=geco.tau,
                step_size=geco.step_size,
                lambda_init=geco.lambda_init,
                momentum=geco.momentum,
                log_lambda_min=geco.log_lambda_min,
                log_lambda_max=geco.log_lambda_max,
            )

    def forward(self, output: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute scalar terms without a rollout or experiment-specific schedule."""
        config = self.config
        terms = {
            "prediction": prediction_mse(output.predictions, output.targets),
            "state_sparsity": torch.stack([item.sparsity.state for item in output.dynamics]).mean(),
            "action_sparsity": torch.stack(
                [item.sparsity.action for item in output.dynamics]
            ).mean(),
            "parameter_sparsity": torch.stack(
                [item.sparsity.parameter for item in output.dynamics]
            ).mean(),
            "logit": torch.stack([item.logit_penalty for item in output.dynamics]).mean(),
        }
        base = terms["prediction"]
        sparsity = base.new_zeros(())
        for name, weight in (
            ("state_sparsity", config.state_sparsity_weight),
            ("action_sparsity", config.action_sparsity_weight),
            ("parameter_sparsity", config.parameter_sparsity_weight),
        ):
            if weight:
                sparsity = sparsity + weight * terms[name]
        if config.logit_weight:
            base = base + config.logit_weight * terms["logit"]
        if config.state_regularizer_weight:
            states = output.online_states
            if states.ndim == 4:
                states = states.flatten(1, 2)
            terms["state_regularizer"] = self.state_regularizer(states.transpose(0, 1))
            base = base + config.state_regularizer_weight * terms["state_regularizer"]
        if config.parameter_regularizer_weight:
            parameters = output.parameters
            terms["parameter_regularizer"] = self.parameter_regularizer(
                parameters.transpose(0, 1) if parameters.ndim == 3 else parameters.unsqueeze(0)
            )
            base = base + config.parameter_regularizer_weight * terms["parameter_regularizer"]
        terms["base"] = base
        terms["sparsity"] = sparsity
        weight = self.geco.penalty_weight if self.geco is not None else base.new_ones(())
        total = base + weight * sparsity
        return total, terms


class _LossForward(nn.Module):
    """Expose only the optimized graph to DDP's unused-parameter discovery."""

    def __init__(self, model: SCJEPA, objective: TrainingObjective) -> None:
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(
        self, observations: Tensor, actions: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        output = cast(ModelOutput, self.model(observations, actions))
        total, terms = self.objective.forward(output)
        return total, {name: value.detach() for name, value in terms.items()}


@dataclass
class TrainConfig:
    """Optimizer and run-health settings, independent of scientific protocols.

    ``steps`` counts consumed microbatches. ``batch_size`` is the total number
    of independent episodes or parameter groups per microbatch over all
    distributed ranks. Paired groups keep both episodes together. Updates average
    ``accumulation_steps`` equal-sized microbatches. Optional cosine decay uses
    ``lr_decay_steps`` successful updates and ends at ``lr * lr_final_ratio``.
    """

    steps: int = 1
    batch_size: int = 8
    lr: float = 1e-3
    lr_decay_steps: int | None = None
    lr_final_ratio: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_skip_threshold: float = math.inf
    grad_skip_max_consecutive: int = 100
    accumulation_steps: int = 1
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    prefetch_factor: int = 2
    log_every: int = 1
    checkpoint_every: int | None = None
    out_dir: str = "outputs"
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    geco: GECOConfig = field(default_factory=GECOConfig)

    def __post_init__(self) -> None:
        """Validate numerical and batching controls."""
        for name in (
            "steps",
            "batch_size",
            "accumulation_steps",
            "grad_skip_max_consecutive",
            "prefetch_factor",
            "log_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or (
            self.checkpoint_every is not None and self.checkpoint_every < 1
        ):
            raise ValueError("num_workers must be non-negative and checkpoint_every positive")
        for name in ("lr", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lr_decay_steps is not None and self.lr_decay_steps < 1:
            raise ValueError("lr_decay_steps must be positive when supplied")
        if not math.isfinite(self.lr_final_ratio) or not 0 < self.lr_final_ratio <= 1:
            raise ValueError("lr_final_ratio must lie in (0, 1]")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if math.isnan(self.grad_skip_threshold) or self.grad_skip_threshold < 0:
            raise ValueError("grad_skip_threshold must be non-negative")


class Trainer:
    """Train an SCJEPA with synchronized skip decisions and one EMA per update."""

    def __init__(
        self,
        model: SCJEPA,
        config: TrainConfig,
        dataset: Dataset[dict[str, Tensor]] | None = None,
        logger: MetricLogger | None = None,
        model_config: Mapping[str, object] | None = None,
        dataset_identity: str | None = None,
    ) -> None:
        """Validate inputs and bind dataset identity before constructing the optimizer.

        Built-in datasets declare their provenance fingerprint. Other deterministic
        datasets receive a streamed content fingerprint. An explicit identity may
        describe caller-managed ``train_step`` data when ``dataset`` is absent;
        it cannot replace or bypass the identity of a supplied dataset.
        """
        if dataset_identity is not None and not dataset_identity:
            raise ValueError("dataset_identity must be a nonempty string when supplied")
        if dataset is not None:
            validate_model_dataset(model, dataset)
            inferred_identity = dataset_fingerprint(dataset)
            if dataset_identity is not None and dataset_identity != inferred_identity:
                raise ValueError("dataset_identity differs from the supplied dataset fingerprint")
            dataset_identity = inferred_identity
        self.dataset_identity = dataset_identity
        # Generic tracking metadata is independent of model/data/optimizer state.
        # The CLI fills this after resuming and initializing its rank-zero logger.
        self.run_metadata: dict[str, object] = {}
        self.validation_state: dict[str, str | float | int | None] = {}
        self.interrupted = False
        seed_everything(config.seed + rank() * 1_000_003)
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.model_config = dict(model_config) if model_config is not None else None
        self.dataset = dataset
        self.logger: MetricLogger = logger if logger is not None else NoopLogger()
        if config.batch_size % world_size():
            raise ValueError("batch_size must be divisible by world_size")
        self.objective = TrainingObjective(config.objective, config.geco).to(self.device)
        self._loss_model = _LossForward(self.model, self.objective)
        self._forward_model: nn.Module = self._loss_model
        if world_size() > 1:
            self._forward_model = DistributedDataParallel(
                self._loss_model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        self._parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        target_ids = {id(parameter) for parameter in self.model.target_encoder.parameters()}
        if any(id(parameter) in target_ids for parameter in self._parameters):
            raise ValueError("target encoder parameters must be frozen")
        self.optimizer = torch.optim.AdamW(
            self._parameters, lr=config.lr, weight_decay=config.weight_decay
        )
        self.step = 0
        self.optimizer_steps = 0
        self.total_skips = 0
        self.consecutive_skips = 0
        self._accumulated = 0
        self._accumulation_batch_size: int | None = None
        self._accumulated_base_loss = torch.zeros((), device=self.device)
        self.discarded_microbatches = 0
        if dataset is not None and len(self._epoch_loader(0)) == 0:
            raise ValueError("dataset yields no full batches; lower batch_size or add episodes")

    @property
    def geco(self) -> SparsityLagrangian | None:
        """The optional checkpointed controller attached to the training objective."""
        return self.objective.geco

    @property
    def pending_microbatches(self) -> int:
        """Number of accumulated microbatches awaiting an optimizer update."""
        return self._accumulated

    def _epoch_loader(self, epoch: int) -> DataLoader[dict[str, Tensor]]:
        if self.dataset is None:
            raise ValueError("train() requires a dataset")
        generator = torch.Generator().manual_seed(self.config.seed * 100_003 + epoch)
        workers = self.config.num_workers
        return DataLoader(
            self.dataset,
            batch_sampler=GlobalBatchSampler(
                dataset_size=len(cast(Sized, self.dataset)),
                batch_size=self.config.batch_size,
                seed=self.config.seed,
                epoch=epoch,
            ),
            generator=generator,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
            prefetch_factor=self.config.prefetch_factor if workers > 0 else None,
            multiprocessing_context="spawn" if workers > 0 else None,
        )

    def _batches(self) -> Iterator[dict[str, Tensor]]:
        steps_per_epoch = len(self._epoch_loader(0))
        epoch, skip = divmod(self.step, steps_per_epoch)
        while True:
            for index, batch in enumerate(self._epoch_loader(epoch)):
                if index >= skip:
                    yield batch
            epoch, skip = epoch + 1, 0

    def _validate_regularization_batch(self, batch: Mapping[str, Tensor]) -> None:
        config = self.config.objective
        if (config.state_regularizer_weight or config.parameter_regularizer_weight) and any_rank(
            batch["observations"].shape[0] < 2, self.device
        ):
            raise ValueError("regularization requires at least two episodes per local batch")
        if not config.parameter_regularizer_weight:
            return
        ids = batch.get("episode_ids")
        bad_ids = ids is None or ids.ndim != 1 or ids.shape[0] != batch["observations"].shape[0]
        if any_rank(bad_ids, self.device):
            raise ValueError("parameter regularization requires episode_ids with shape (B,)")
        assert ids is not None
        ids = gather_batch_tensor(ids.to(self.device))
        unique_ids = cast(Tensor, ids.unique())  # pyright: ignore[reportUnknownMemberType]
        if unique_ids.numel() != ids.numel():
            raise ValueError("parameter regularization requires distinct episodes in each batch")

    def _forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        observations = batch["observations"].to(self.device, non_blocking=True)
        actions = batch.get("actions")
        if actions is not None:
            actions = actions.to(self.device, non_blocking=True)
        return cast(tuple[Tensor, dict[str, Tensor]], self._forward_model(observations, actions))

    def discard_accumulation(self) -> None:
        """Explicitly discard a partial accumulation group without an EMA update."""
        self.discarded_microbatches += self._accumulated
        self._accumulated = 0
        self._accumulation_batch_size = None
        self._accumulated_base_loss.zero_()
        self.optimizer.zero_grad(set_to_none=True)

    def _skip(self) -> None:
        self.discard_accumulation()
        self.total_skips += 1
        self.consecutive_skips += 1
        if self.consecutive_skips >= self.config.grad_skip_max_consecutive:
            raise RuntimeError("too many consecutive rejected updates")

    def train_step(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        """Consume one microbatch, updating EMA/GECO only after an accepted step.

        A nonfinite loss or gradient on any rank discards the entire accumulated
        group on every rank. Gradients are synchronized on every microbatch;
        they are clipped only when the complete group is ready for an update.
        GECO uses the detached mean base loss over the complete accumulation
        group and all ranks, with the pre-update multiplier shared by the group.
        """
        self.model.train()
        batch_size = batch["observations"].shape[0]
        if any_rank(batch_size != self.config.batch_size // world_size(), self.device):
            raise ValueError("local episode count must equal batch_size / world_size")
        self._validate_regularization_batch(batch)
        if (
            self._accumulation_batch_size is not None
            and batch_size != self._accumulation_batch_size
        ):
            raise ValueError("accumulated microbatches must have equal episode counts")
        self._accumulation_batch_size = batch_size
        total, terms = self._forward(batch)
        self.step += 1
        rejected = any_rank(~torch.isfinite(total), self.device)
        if rejected:
            # Finish DDP's reducer traversal even for a rejected forward. Any
            # gradients from this backward are discarded, never applied.
            if world_size() > 1:
                (total.nan_to_num() * 0).backward()  # pyright: ignore[reportUnknownMemberType]
            self._skip()
            grad_norm = total.new_tensor(math.nan)
            updated = False
        else:
            (total / self.config.accumulation_steps).backward()  # pyright: ignore[reportUnknownMemberType]
            self._accumulated += 1
            self._accumulated_base_loss.add_(terms["base"])
            grad_norm = torch.nn.utils.clip_grad_norm_(self._parameters, math.inf)
            rejected = any_rank(~torch.isfinite(grad_norm), self.device)
            updated = False
            if not rejected and self._accumulated == self.config.accumulation_steps:
                rejected = any_rank(grad_norm > self.config.grad_skip_threshold, self.device)
                if not rejected:
                    torch.nn.utils.clip_grad_norm_(self._parameters, self.config.grad_clip)
                    self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
                    self.model.update_target()
                    if self.geco is not None:
                        self.geco.update(
                            mean_tensor(self._accumulated_base_loss / self._accumulated)
                        )
                    self.optimizer_steps += 1
                    if self.config.lr_decay_steps is not None:
                        progress = min(self.optimizer_steps / self.config.lr_decay_steps, 1.0)
                        ratio = self.config.lr_final_ratio
                        factor = ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.config.lr * factor
                    self.consecutive_skips = 0
                    self._accumulated = 0
                    self._accumulation_batch_size = None
                    self._accumulated_base_loss.zero_()
                    self.optimizer.zero_grad(set_to_none=True)
                    updated = True
            if rejected:
                self._skip()
        metrics = {f"loss/{name}": float(value.detach()) for name, value in terms.items()} | {
            "loss/total": float(total.detach()),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
            "health/grad_norm": float(grad_norm),
            "health/skipped": float(rejected),
            "health/skipped_steps": float(self.total_skips),
            "health/optimizer_steps": float(self.optimizer_steps),
            "health/updated": float(updated),
            "health/pending_microbatches": float(self._accumulated),
        }
        if self.geco is not None:
            metrics.update(
                {
                    "geco/tau": self.geco.tau,
                    "geco/constraint": float(terms["base"]) - self.geco.tau,
                    "geco/sparsity_weight": float(self.geco.penalty_weight),
                    "geco/multiplier": float(self.geco.log_lambda.exp()),
                }
            )
        return metrics

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        error: Exception | None = None
        if is_main_process():
            try:
                self.logger.log(self.step, metrics)
            except Exception as caught:
                error = caught
        if any_rank(error is not None, self.device):
            raise RuntimeError("metric logging failed on rank zero") from error

    def _run_validation(
        self, callback: Callable[[SCJEPA], dict[str, float]], selection_metric: str
    ) -> dict[str, float]:
        """Protect training RNG/modes and synchronize a rank-zero callback result."""
        metrics: dict[str, float] = {}
        error: Exception | None = None
        if is_main_process():
            modes = [(module, module.training) for module in self.model.modules()]
            self.model.eval()
            try:
                # Evaluation can fit its own probes. Learned model extraction
                # should use no_grad inside the callback, without preventing
                # gradients for those separate evaluation-only modules.
                with preserve_rng(self.device):
                    metrics = callback(self.model)
                if selection_metric not in metrics:
                    raise ValueError(f"validation metrics require {selection_metric!r}")
                if any(not math.isfinite(value) for value in metrics.values()):
                    raise ValueError("validation metrics must be finite")
            except Exception as caught:
                error = caught
            finally:
                for module, mode in modes:
                    module.training = mode
                self.model.target_encoder.eval()
        if any_rank(error is not None, self.device):
            raise RuntimeError("validation failed on rank zero") from error
        metrics = cast(dict[str, float], gather_rank_objects(metrics)[0])
        score = metrics[selection_metric]
        previous = self.validation_state.get("best_score")
        improved = previous is None or score < float(previous)
        self.validation_state["last_step"] = self.step
        if improved:
            self.validation_state["best_score"] = score
            self.validation_state["best_step"] = self.step
        self._log_metrics(metrics)
        if improved:
            self.save_checkpoint(Path(self.config.out_dir) / "best.pt")
        return metrics

    def train(
        self,
        validation_callback: Callable[[SCJEPA], dict[str, float]] | None = None,
        validation_every: int | None = None,
        selection_metric: str = "validation/base_loss",
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict[str, float]:
        """Consume ``config.steps`` microbatches, resuming deterministic dataset order.

        Final partial groups are explicitly discarded. Periodic checkpoints
        wait for an accumulation boundary; direct saves reject pending groups.
        Dataset transforms must be deterministic for exact data replay.
        Validation intervals count microbatches and defer to a completed group;
        callbacks run only on rank zero, with protected RNG and module modes.
        Normal completion validates once at the final step. A synchronized stop
        finishes the current group and checkpoints without a final evaluation.
        Resume services any validation due at that saved boundary before taking
        another optimizer step, preserving the uninterrupted selection candidates.
        """
        if self.dataset is None:
            raise ValueError("train() requires a dataset")
        if validation_every is not None and validation_every < 1:
            raise ValueError("validation_every must be positive")
        if validation_every is not None and validation_callback is None:
            raise ValueError("validation_every requires a validation_callback")
        if not selection_metric:
            raise ValueError("selection_metric must be nonempty")
        if validation_callback is not None:
            if (
                self.validation_state
                and self.validation_state["selection_metric"] != selection_metric
            ):
                raise ValueError("selection_metric differs from resumed validation state")
            if not self.validation_state:
                self.validation_state = {
                    "last_step": None,
                    "best_score": None,
                    "best_step": None,
                    "selection_metric": selection_metric,
                }
        batches = self._batches()
        metrics: dict[str, float] = {}
        checkpoint_due = False
        last_validation = self.validation_state.get("last_step")
        validation_due = (
            validation_every is not None
            and self.step // validation_every > int(last_validation or 0) // validation_every
        )
        stopping = False
        self.interrupted = False
        while self.step < self.config.steps:
            if stop_requested is not None:
                stopping = any_rank(stop_requested(), self.device) or stopping
            if stopping and self._accumulated == 0:
                self.interrupted = True
                break
            if validation_callback is not None and validation_due and self._accumulated == 0:
                metrics.update(self._run_validation(validation_callback, selection_metric))
                validation_due = False
                self.save_checkpoint(Path(self.config.out_dir) / "last.pt")
                # A signal may arrive during validation. Recheck before
                # consuming the next batch, even though evaluation is complete.
                continue
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            metrics = self.train_step(next(batches))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            metrics["train/seconds_per_microbatch"] = time.perf_counter() - started
            if self.step % self.config.log_every == 0 or self.step == self.config.steps:
                metrics = mean_metrics(metrics, self.device)
                self._log_metrics(metrics)
            if stop_requested is not None:
                stopping = any_rank(stop_requested(), self.device) or stopping
            if stopping and self._accumulated == 0:
                self.interrupted = True
                break
            validation_due = validation_due or (
                validation_every is not None and self.step % validation_every == 0
            )
            if validation_callback is not None and validation_due and self._accumulated == 0:
                metrics.update(self._run_validation(validation_callback, selection_metric))
                validation_due = False
            every = self.config.checkpoint_every
            checkpoint_due = checkpoint_due or (every is not None and self.step % every == 0)
            if checkpoint_due and self._accumulated == 0:
                self.save_checkpoint(Path(self.config.out_dir) / "last.pt")
                checkpoint_due = False
        self.discard_accumulation()
        self.interrupted = self.interrupted or stopping
        if (
            validation_callback is not None
            and not self.interrupted
            and self.validation_state["last_step"] != self.step
        ):
            metrics.update(self._run_validation(validation_callback, selection_metric))
        metrics["health/discarded_microbatches"] = float(self.discarded_microbatches)
        metrics["health/pending_microbatches"] = 0.0
        metrics["health/interrupted"] = float(self.interrupted)
        self.save_checkpoint(Path(self.config.out_dir) / "last.pt")
        return metrics

    def _validate_dataset_identity(self) -> None:
        changed = (
            self.dataset is not None and dataset_fingerprint(self.dataset) != self.dataset_identity
        )
        if any_rank(changed, self.device):
            raise ValueError("dataset_identity changed after Trainer initialization")

    def save_checkpoint(self, path: Path) -> None:
        """Atomically save online/target weights, configs, optimizer, counters, and RNGs.

        Every rank must participate. Saves at partial accumulation boundaries
        are rejected because pending gradients are deliberately not serialized.
        A dataset changed since initialization cannot produce an exact checkpoint.
        """
        self._validate_dataset_identity()
        if any_rank(self._accumulated != 0, self.device):
            raise ValueError("save checkpoints at an accumulation boundary")
        local_rng = {
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),  # noqa: NPY002
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
        }
        rank_rng = gather_rank_objects(local_rng)
        saving_error: Exception | None = None
        if is_main_process():
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format_version": 1,
                        "model": self.model.state_dict(),
                        "model_config": self.model_config,
                        "dataset_identity": self.dataset_identity,
                        "run_metadata": self.run_metadata,
                        "validation_state": self.validation_state,
                        "train_config": asdict(self.config),
                        "objective": self.objective.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "step": self.step,
                        "optimizer_steps": self.optimizer_steps,
                        "total_skips": self.total_skips,
                        "consecutive_skips": self.consecutive_skips,
                        "discarded_microbatches": self.discarded_microbatches,
                        "rank_rng": rank_rng,
                        "world_size": world_size(),
                    },
                    temporary,
                )
                temporary.replace(path)
            except Exception as error:
                saving_error = error
        if any_rank(saving_error is not None, self.device):
            raise RuntimeError("checkpoint writing failed on rank zero") from saving_error
        distributed_barrier()

    def load_checkpoint(self, path: Path) -> None:
        """Restore training state; retain current run length, device, and output settings.

        Only load trusted checkpoints: full optimizer and Python/NumPy RNG
        restoration requires pickle. Exact resume requires the same world size,
        model configuration, global batch size, and identical dataset fingerprint.
        Dataset compatibility is checked before restoring any training state.
        Older identity-free checkpoints remain usable for caller-managed steps
        with no dataset identity; they cannot establish exact dataset-backed replay.
        """
        if self._accumulated:
            raise ValueError("load checkpoints at an accumulation boundary")
        self._validate_dataset_identity()
        payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if payload.get("dataset_identity") != self.dataset_identity:
            raise ValueError("exact training resume requires the checkpoint dataset_identity")
        if payload["world_size"] != world_size():
            raise ValueError("exact training resume requires the checkpoint's world_size")
        saved_config = cast(dict[str, Any], payload["train_config"])
        if saved_config["batch_size"] != self.config.batch_size:
            raise ValueError("exact training resume requires the checkpoint's batch_size")
        saved_model_config = cast(dict[str, object] | None, payload["model_config"])
        if self.model_config is not None and self.model_config != saved_model_config:
            raise ValueError("model_config differs from checkpoint")
        saved_run_metadata = payload.get("run_metadata", {})
        if not isinstance(saved_run_metadata, dict):
            raise ValueError("checkpoint run_metadata must be a mapping")
        saved_validation = payload.get("validation_state", {})
        if not isinstance(saved_validation, dict):
            raise ValueError("checkpoint validation_state must be a mapping")
        validation = cast(dict[str, Any], saved_validation)
        if validation:
            if set(validation) != {"last_step", "best_score", "best_step", "selection_metric"}:
                raise ValueError("invalid checkpoint validation_state fields")
            if (
                not isinstance(validation["selection_metric"], str)
                or not validation["selection_metric"]
            ):
                raise ValueError("invalid checkpoint validation selection_metric")
            for name in ("last_step", "best_step"):
                value = validation[name]
                if value is not None and (
                    not isinstance(value, int) or not 0 <= value <= int(payload["step"])
                ):
                    raise ValueError("invalid checkpoint validation step")
            score = validation["best_score"]
            if score is not None and (
                not isinstance(score, int | float) or not math.isfinite(score)
            ):
                raise ValueError("invalid checkpoint validation score")
            if (score is None) != (validation["best_step"] is None):
                raise ValueError("inconsistent checkpoint validation score and step")
        self.run_metadata = dict(cast(dict[str, object], saved_run_metadata))
        self.validation_state = dict(validation)
        self.model_config = saved_model_config
        for name in (
            "steps",
            "device",
            "out_dir",
            "log_every",
            "checkpoint_every",
            "num_workers",
            "prefetch_factor",
        ):
            saved_config[name] = getattr(self.config, name)
        saved_config["objective"] = ObjectiveConfig(**saved_config["objective"])
        saved_config["geco"] = GECOConfig(**saved_config.get("geco", {}))
        self.config = TrainConfig(**saved_config)
        self.objective = TrainingObjective(self.config.objective, self.config.geco).to(self.device)
        self._loss_model.objective = self.objective
        self.objective.load_state_dict(payload["objective"])
        if self.geco is not None:
            self.geco.validate_state()
            if int(self.geco.num_updates) != int(payload["optimizer_steps"]):
                raise ValueError("checkpoint GECO updates must match successful optimizer steps")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.step = int(payload["step"])
        self.optimizer_steps = int(payload["optimizer_steps"])
        self.total_skips = int(payload["total_skips"])
        self.consecutive_skips = int(payload["consecutive_skips"])
        self.discarded_microbatches = int(payload["discarded_microbatches"])
        self.optimizer.zero_grad(set_to_none=True)
        rng_state = cast(dict[str, Any], payload["rank_rng"][rank()])
        random.setstate(rng_state["rng_python"])
        np.random.set_state(rng_state["rng_numpy"])  # noqa: NPY002
        torch.set_rng_state(rng_state["rng_torch"])
        if rng_state["rng_cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state(rng_state["rng_cuda"], self.device)


__all__ = [
    "GECOConfig",
    "MetricLogger",
    "NoopLogger",
    "ObjectiveConfig",
    "TrainConfig",
    "Trainer",
    "TrainingObjective",
    "seed_everything",
]
