"""Native Trainer updates with matched batch/donor/gate and separate SIGReg streams."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor

from diagnostics.sparse_comparison.source import Inputs
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.distributed import GlobalBatchSampler
from scjepa.training.factory import build_model
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.loop import ObjectiveConfig, TrainConfig, Trainer
from scjepa.training.rng import preserve_rng

Condition = Literal["dense", "baseline", "full"]
CONDITIONS: tuple[Condition, ...] = ("dense", "baseline", "full")
OBJECTIVE_STREAM_OFFSET = 1 << 40


class IsolatedForward(SCJEPA):
    """Use one explicit forward stream without advancing the surrounding auxiliary stream."""

    forward_seed: int | None = None

    def forward(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        prediction_start: int | None = None,
        *,
        oracle_parameters: Tensor | None = None,
    ) -> ModelOutput:
        """Preserve ordinary donor/gate draws and attached gradients; isolate only RNG state."""
        if not self.training or self.forward_seed is None:
            return super().forward(
                observations, actions, prediction_start, oracle_parameters=oracle_parameters
            )
        with preserve_rng(observations.device):
            torch.manual_seed(self.forward_seed)  # pyright: ignore[reportUnknownMemberType]
            return super().forward(
                observations, actions, prediction_start, oracle_parameters=oracle_parameters
            )


def condition_objective(inputs: Inputs, condition: Condition) -> ObjectiveConfig:
    """Use original coefficients with only the declared structural penalties changed."""
    if condition not in CONDITIONS:
        raise ValueError("Unknown comparison condition")
    return replace(
        inputs.objective_config,
        state_sparsity_weight=0.0 if condition == "dense" else 1.0,
        action_sparsity_weight=0.0 if condition == "dense" else 1.0,
        parameter_sparsity_weight=1.0 if condition == "full" else 0.0,
    )


def make_trainer(
    inputs: Inputs, condition: Condition, output: Path, *, tau: float, steps: int
) -> Trainer:
    """Load identical raw weights into native gate policies and create fresh AdamW/GECO."""
    condition_objective(inputs, condition)
    model_config = replace(
        inputs.model_config,
        spartan_dense=condition == "dense",
        spartan_parameter_gates_open=condition == "baseline",
        spartan_action_gates_open=False,
        spartan_identity=False,
    )
    with preserve_rng(next(inputs.model.parameters()).device):
        base = build_model(model_config)
        model = IsolatedForward(
            base.online_encoder,
            base.parameter_encoder,
            base.predictor,
            base.context_len,
            base.ema_decay,
            autoregressive=base.autoregressive,
            parameter_source=base.parameter_source,
        )
    model.load_state_dict(inputs.model.state_dict(), strict=True)
    model.requires_grad_(True)
    model.target_encoder.requires_grad_(False).eval()
    config = TrainConfig(
        steps=steps,
        batch_size=inputs.batch_size,
        lr=0.001,
        lr_final_ratio=1.0,
        lr_decay_steps=None,
        weight_decay=0.0,
        grad_clip=1.0,
        accumulation_steps=1,
        seed=0,
        device=str(next(inputs.model.parameters()).device),
        log_every=100,
        checkpoint_every=250,
        out_dir=str(output),
        objective=condition_objective(inputs, condition),
        geco=GECOConfig(
            enabled=condition != "dense",
            tau=tau if condition != "dense" else None,
            step_size=0.1,
            momentum=0.99,
            lambda_init=1000.0,
            log_lambda_min=-12.0,
            log_lambda_max=12.0,
        ),
    )
    trainer = Trainer(
        model,
        config,
        model_config=asdict(model_config),
        dataset_identity=inputs.provenance["dataset_identity"],
    )
    trainer.run_metadata.update(
        {
            "source_global_step": inputs.source_step,
            "stream_seed": inputs.seed,
            "bootstrap_seed": 0,
            "objective_stream_offset": OBJECTIVE_STREAM_OFFSET,
            "condition": condition,
        }
    )
    if (
        trainer.optimizer.state
        or trainer.optimizer_steps
        or (trainer.geco is not None and int(trainer.geco.num_updates))
    ):
        raise RuntimeError("All conditions require fresh AdamW and fresh GECO")
    return trainer


def batch_at(inputs: Inputs, global_step: int) -> dict[str, Tensor]:
    """Continue original global group order and expose unique group IDs for native SIGReg."""
    groups = len(inputs.observations["observations"])
    epoch, position = divmod(global_step - 1, groups // inputs.batch_size)
    indices = list(GlobalBatchSampler(groups, inputs.batch_size, inputs.seed, epoch))[position]
    return {
        **{name: value[indices] for name, value in inputs.observations.items()},
        "episode_ids": torch.tensor(indices, dtype=torch.long),
    }


def step(trainer: Trainer, inputs: Inputs) -> dict[str, float]:
    """Perform exactly one original native update; reject every numerical skip explicitly."""
    global_step = inputs.source_step + trainer.step + 1
    forward_seed = inputs.seed * 100000 + global_step
    if not isinstance(trainer.model, IsolatedForward):
        raise TypeError("Require the isolated-forward diagnostic model")
    trainer.model.forward_seed = forward_seed
    torch.manual_seed(forward_seed + OBJECTIVE_STREAM_OFFSET)  # pyright: ignore[reportUnknownMemberType]
    metrics = trainer.train_step(batch_at(inputs, global_step))
    if (
        metrics["health/updated"] != 1
        or metrics["health/skipped"] != 0
        or trainer.optimizer_steps != trainer.step
    ):
        raise FloatingPointError(
            "Native Trainer skipped a comparison update; abort without replacement"
        )
    if trainer.geco is not None and int(trainer.geco.num_updates) != trainer.optimizer_steps:
        raise RuntimeError("GECO count differs from successful optimizer updates")
    if any(
        p.requires_grad or p.grad is not None for p in trainer.model.target_encoder.parameters()
    ):
        raise FloatingPointError("Target gradient contract violated")
    return {**metrics, "train/global_step": float(global_step)}


def assert_trainer(trainer: Trainer, inputs: Inputs, condition: Condition, *, tau: float) -> None:
    """Validate restored optimizer membership, native counters and immutable condition policy."""
    model = trainer.model
    if (
        not isinstance(model, IsolatedForward)
        or model.get_extra_state() != inputs.provenance["rollout_flags"]
    ):
        raise ValueError("Restored model class or rollout flags differ")
    if (
        trainer.config.objective != condition_objective(inputs, condition)
        or trainer.config.seed != 0
        or trainer.total_skips
        or trainer.optimizer_steps != trainer.step
    ):
        raise ValueError("Restored native objective/bootstrap/counters differ")
    if any(p.requires_grad for p in model.target_encoder.parameters()) or not all(
        p.requires_grad
        for block in (model.online_encoder, model.parameter_encoder, model.predictor)
        for p in block.parameters()
    ):
        raise ValueError("All online/E/P parameters must train and target must remain frozen")
    if (
        model.predictor.dense != (condition == "dense")
        or model.predictor.parameter_gates_open != (condition == "baseline")
        or model.predictor.action_gates_open
        or any(
            layer.dense != (condition == "dense") or layer.identity
            for layer in model.predictor.layers
        )
    ):
        raise ValueError("Restored condition gate policy differs")
    expected_geco = GECOConfig(
        enabled=condition != "dense",
        tau=tau if condition != "dense" else None,
        step_size=0.1,
        momentum=0.99,
        lambda_init=1000.0,
        log_lambda_min=-12.0,
        log_lambda_max=12.0,
    )
    if trainer.config.geco != expected_geco or (trainer.geco is None) != (condition == "dense"):
        raise ValueError("Restored shared GECO bound/controller settings differ")
    for name, expected in {
        "source_global_step": inputs.source_step,
        "stream_seed": inputs.seed,
        "bootstrap_seed": 0,
        "objective_stream_offset": OBJECTIVE_STREAM_OFFSET,
        "condition": condition,
    }.items():
        if trainer.run_metadata.get(name) != expected:
            raise ValueError("Restored explicit stream/condition metadata differs")
    if trainer.geco is not None:
        trainer.geco.validate_state()
        if trainer.geco.tau != tau or int(trainer.geco.num_updates) != trainer.optimizer_steps:
            raise ValueError("Restored GECO bound/update count differs")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite comparison model")
    parameters = [p for p in model.parameters() if p.requires_grad]
    states = trainer.optimizer.state_dict()
    groups = states["param_groups"]
    if (
        len(groups) != 1
        or groups[0]["params"] != list(range(len(parameters)))
        or (groups[0]["lr"], groups[0]["weight_decay"], tuple(groups[0]["betas"]), groups[0]["eps"])
        != (0.001, 0.0, (0.9, 0.999), 1e-8)
    ):
        raise ValueError("Restored AdamW group/order differs")
    if trainer.step == 0:
        if states["state"]:
            raise ValueError("Fresh comparison optimizer has nonempty moments")
    else:
        if set(states["state"]) != set(range(len(parameters))):
            raise ValueError("Restored AdamW moment coverage differs")
        for index, parameter in enumerate(parameters):
            row = states["state"][index]
            if float(row["step"]) != trainer.step:
                raise ValueError("Restored AdamW counter differs")
            for name in ("exp_avg", "exp_avg_sq"):
                moment = cast(Tensor, row[name])
                if (
                    moment.shape != parameter.shape
                    or moment.dtype != torch.float32
                    or not bool(torch.isfinite(moment).all())
                    or (name == "exp_avg_sq" and bool((moment < 0).any()))
                ):
                    raise ValueError("Restored AdamW moments differ")
