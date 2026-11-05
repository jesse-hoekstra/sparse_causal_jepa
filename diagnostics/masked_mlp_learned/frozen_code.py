"""Recover states from baseline weights with a frozen, pair-mixed learned code.

This is a transfer control, not an ordinary joint-learning run. A frozen copy of
the full model's observation and parameter encoders supplies single-context codes.
Identical learned parameter-mask columns define pairs without physical labels.
TRAIN standard deviations and fixed 45-degree rotations mix those pairs. Both
arms inherit baseline state/predictor weights and its EMA target, with fresh
AdamW/GECO/sampler states. Only the full arm learns sparse parameter gates.
The historical trainers and their source identities remain unchanged.
"""

import argparse
import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
import torch
from torch import Tensor, nn

from diagnostics.masked_mlp_learned import run as learned
from diagnostics.masked_mlp_oracle import paper_analysis as paper
from diagnostics.masked_mlp_oracle.model import INITIAL_LOGIT, MaskedTrunkPredictor
from diagnostics.shared.data import StateData, load_data
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.io import verify_completion as verify_artifacts
from diagnostics.shared.oracle import EvaluationResult, objective_config
from diagnostics.shared.sampling import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.eval.recovery import fit_coordinate_match, mean_correlation_coefficient
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, ModelOutput, cross_episode_parameters
from scjepa.models.spartan import SpartanOutput
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective, seed_everything

SCHEMA = "masked-mlp-frozen-code-control-v1"
STEPS = 300_000
type Arm = Literal["baseline", "full"]


def code_identity() -> str:
    """Bind this control, its evaluation and the unchanged historical runtime."""
    return canonical_fingerprint(
        {
            "learned": learned.code_identity(),
            "control": sha(Path(__file__)),
            "graph_matching": sha(Path(paper.__file__)),
        }
    )


def learned_pairs(graph: Tensor) -> tuple[tuple[int, int], ...]:
    """Require three nonempty, distinct pairs of equal learned mask columns."""
    if graph.shape != (6, 6) or graph.dtype != torch.bool:
        raise ValueError("require a six-by-six boolean learned parameter graph")
    groups: dict[tuple[bool, ...], list[int]] = {}
    for j in range(6):
        key = tuple(bool(v) for v in graph[:, j])
        if not any(key):
            raise ValueError("a frozen code coordinate has no learned children")
        groups.setdefault(key, []).append(j)
    if len(groups) != 3 or any(len(group) != 2 for group in groups.values()):
        raise ValueError("learned parameter graph must contain exactly three equal-column pairs")
    return tuple((group[0], group[1]) for group in groups.values())


def pair_rotation(pairs: tuple[tuple[int, int], ...]) -> Tensor:
    """Construct a fixed orthogonal 45-degree rotation, in learned-code order."""
    if sorted(index for pair in pairs for index in pair) != list(range(6)):
        raise ValueError("pairs must partition six code coordinates")
    matrix = torch.zeros(6, 6, dtype=torch.float64)
    for i, j in pairs:
        matrix[i, i] = matrix[i, j] = matrix[j, j] = 1 / math.sqrt(2)
        matrix[j, i] = -1 / math.sqrt(2)
    return matrix


@torch.no_grad()
def training_scale(donor: SCJEPA, train: StateData, device: torch.device) -> Tensor:
    """Fit population SDs on all TRAIN contexts, without reading any labels."""
    if train.split != "train":
        raise ValueError("code scales must be fitted on training groups only")
    donor.eval()
    codes: list[Tensor] = []
    for start in range(0, len(train), 64):
        observations = train.tensors["observations"][start : start + 64].to(device)
        actions = train.tensors["actions"][start : start + 64].to(device)
        states = donor.online_encoder.encode_sequence(
            observations.flatten(0, 1)[:, : donor.context_len]
        )
        values = donor.parameter_encoder(states, actions.flatten(0, 1)[:, : donor.context_len - 1])
        codes.append(values.cpu().double())
    scale = torch.cat(codes).std(0, correction=0)
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 1e-8).any()):
        raise ValueError("frozen donor codes must have finite nonzero TRAIN variance")
    return scale


class FrozenCodeModel(SCJEPA):
    """Separate frozen parameter inference from trainable framewise state inference."""

    code_scale: Tensor
    code_rotation: Tensor

    def __init__(
        self, baseline: SCJEPA, donor: SCJEPA, condition: Arm, scale: Tensor, rotation: Tensor
    ) -> None:
        """Copy the two source branches; never compensate predictor input weights."""
        if condition not in {"baseline", "full"}:
            raise ValueError("control requires baseline or full")
        if scale.shape != (6,) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("require six finite positive code scales")
        if rotation.shape != (6, 6) or not torch.allclose(
            rotation.double() @ rotation.double().T, torch.eye(6, dtype=torch.float64), atol=1e-6
        ):
            raise ValueError("code rotation must be orthogonal")
        if (baseline.context_len, baseline.autoregressive, baseline.predictor.history_len) != (
            10,
            True,
            1,
        ) or donor.context_len != 10:
            raise ValueError("control preserves the ten-frame context and attached two-step model")
        predictor = copy.deepcopy(baseline.predictor)
        if (
            not isinstance(predictor, MaskedTrunkPredictor)
            or predictor.parameter_logits is not None
        ):
            raise ValueError("state/predictor source must have fixed-open parameter gates")
        if condition == "full":
            predictor.parameter_logits = nn.Parameter(
                predictor.first.weight.new_full((6, 6), INITIAL_LOGIT)
            )
        super().__init__(
            copy.deepcopy(baseline.online_encoder),
            copy.deepcopy(donor.parameter_encoder),
            cast(Any, predictor),
            context_len=10,
            ema_decay=baseline.ema_decay,
            autoregressive=True,
        )
        self.target_encoder.load_state_dict(baseline.target_encoder.state_dict())
        self.parameter_state_encoder = copy.deepcopy(donor.online_encoder)
        self.parameter_state_encoder.requires_grad_(False)
        self.parameter_encoder.requires_grad_(False)
        self.register_buffer("code_scale", scale.float().clone())
        self.register_buffer("code_rotation", rotation.float().clone())
        self.train()

    def train(self, mode: bool = True) -> Self:
        """Keep both parts of the parameter branch frozen, including their modes."""
        super().train(mode)
        self.parameter_state_encoder.eval()
        self.parameter_encoder.eval()
        return self

    def frozen_identity(self) -> str:
        """Fingerprint every tensor that determines the fixed context-to-code map."""
        return canonical_fingerprint(
            {
                "state_encoder": learned._state_identity(self.parameter_state_encoder),  # pyright: ignore[reportPrivateUsage]
                "parameter_encoder": learned._state_identity(self.parameter_encoder),  # pyright: ignore[reportPrivateUsage]
                "scale": tensor_fingerprint(self.code_scale),
                "rotation": tensor_fingerprint(self.code_rotation),
            }
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor | None = None,
        prediction_start: int | None = None,
        *,
        oracle_parameters: Tensor | None = None,
    ) -> ModelOutput:
        """Infer frozen donor codes, then predict two attached future state encodings."""
        if oracle_parameters is not None:
            raise ValueError("frozen learned-code control never consumes true parameters")
        if observations.ndim != 4 or observations.shape[1:] != (10, 12, 6):
            raise ValueError("require grouped observations with shape (B,10,12,6)")
        groups = len(observations)
        if groups < 1 or actions is None or actions.shape != (groups, 10, 11, 24):
            raise ValueError("require matching observed actions")
        if prediction_start not in {None, 9}:
            raise ValueError("the control fixes prediction_start=9")
        flat_observations, flat_actions = observations.flatten(0, 1), actions.flatten(0, 1)
        online = self.online_encoder.encode_sequence(flat_observations)
        with torch.no_grad():
            context = self.parameter_state_encoder.encode_sequence(flat_observations[:, :10])
            raw_codes = self.parameter_encoder(context, flat_actions[:, :9])
            codes = (raw_codes / self.code_scale) @ self.code_rotation.T
            targets = self.target_encoder.encode_sequence(flat_observations[:, 10:12])
        routed = cross_episode_parameters(codes.reshape(groups, 10, 6), training=self.training)
        history = online[:, 9:10]
        transitions: list[SpartanOutput] = []
        for source in (9, 10):
            result = self.predictor(
                history, routed.flatten(0, 1), flat_actions[:, source : source + 1]
            )
            transitions.append(result)
            history = result.prediction[:, None]  # Attached; second-step loss reaches the first.
        predictions = torch.stack([item.prediction for item in transitions], dim=1)
        return ModelOutput(
            predictions.reshape(groups, 10, 2, 6),
            targets.reshape(groups, 10, 2, 6),
            online.reshape(groups, 10, 12, 6),
            codes.reshape(groups, 10, 6),
            tuple(transitions),
            9,
        )


@dataclass(frozen=True)
class ControlConfig:
    """The fixed transfer protocol, with only operational settings configurable."""

    output_dir: str
    baseline_dir: str
    donor_dir: str
    data_dir: str
    condition: Arm
    seed: int
    steps: int = STEPS
    profile: learned.Profile = "exploratory"
    batch_size: int = 16
    device: str = "cpu"
    threads: int = 1
    log_every: int = 100
    evaluate_every: int = 10_000
    checkpoint_every: int = 5000
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None
    wandb_group: str = "masked-mlp-frozen-code"

    def __post_init__(self) -> None:
        """Reject changes to the declared 300k scientific protocol."""
        if self.condition not in {"baseline", "full"} or self.seed not in learned.SEEDS:
            raise ValueError("require baseline/full and one of the five declared seeds")
        if self.profile not in {"exploratory", "smoke"}:
            raise ValueError("unknown profile")
        for name in (
            "steps",
            "batch_size",
            "threads",
            "log_every",
            "evaluate_every",
            "checkpoint_every",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("SIGReg needs at least two groups")
        if self.profile == "exploratory" and (self.steps, self.batch_size) != (STEPS, 16):
            raise ValueError("the declared control uses exactly 300000 updates and batch size 16")
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)


def _parent(
    directory: Path, condition: Arm, config: ControlConfig
) -> tuple[SCJEPA, dict[str, Any]]:
    completion = learned.verify_completion(directory, kind="train", require_current_source=True)
    if (
        completion["condition"] != condition
        or completion["seed"] != config.seed
        or completion["profile"] != config.profile
        or completion["code_dim"] != 6
        or learned.endpoint_activation(completion) != "gelu"
    ):
        raise ValueError("parent must be the seed-matched six-code GELU sparse endpoint")
    if config.profile == "exploratory" and completion["steps"] != 350_000:
        raise ValueError("the declared parents are the fixed 350000-update endpoints")
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        checkpoint["identity"] != completion["identity"]
        or checkpoint["step"] != completion["steps"]
    ):
        raise ValueError("parent checkpoint and completion disagree")
    model = learned.build_control(condition, config.seed, predictor_activation="gelu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, {
        key: completion[key]
        for key in (
            "identity",
            "data_identity",
            "code_identity",
            "steps",
            "tau_prediction",
            "hashes",
        )
    }


def _code_rows(result: EvaluationResult) -> tuple[Tensor, Tensor]:
    codes = result.samples["codes"].double()
    theta = result.samples["theta"].double()[:, None].expand_as(codes)
    return codes.flatten(0, 1), theta.flatten(0, 1)


def collect(
    model: FrozenCodeModel,
    data: dict[str, StateData],
    batch_size: int,
    device: torch.device,
    objective: TrainingObjective,
    tau: float,
) -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
    """Use calibration-only state/code assignments and score held-out validation."""
    results, matching = learned._collect(model, data, batch_size, device, objective, tau)  # pyright: ignore[reportPrivateUsage]
    codes, theta = _code_rows(results["calibration"])
    code_match = fit_coordinate_match(codes, theta, method="pearson")
    order = paper._code_order(codes, theta)  # pyright: ignore[reportPrivateUsage]
    graph = learned._predictor(model).thresholded_graphs()["parameter"]  # pyright: ignore[reportPrivateUsage]
    aligned = graph[matching["online_columns"]][:, order].numpy()
    for result in results.values():
        scored_codes, scored_theta = _code_rows(result)
        result.metrics["code_mcc"] = mean_correlation_coefficient(
            code_match, scored_codes, scored_theta
        )
        result.metrics.update(paper._code_aligned_metrics(aligned))  # pyright: ignore[reportPrivateUsage]
    matching["code_mcc_columns"] = list(code_match.columns)
    matching["parameter_graph_columns"] = order
    return results, matching


def verify_completion(directory: Path) -> dict[str, Any]:
    """Authenticate a completed transfer run, including its immutable frozen branch."""
    result = verify_artifacts(directory)
    if result.get("schema") != SCHEMA or not result.get("frozen_branch_unchanged"):
        raise ValueError("not a completed frozen-code control")
    return result


def train(config: ControlConfig, *, resume: bool = False) -> dict[str, Any]:
    """Train one matched arm in a fresh directory, or resume its exact saved state."""
    output = Path(config.output_dir).resolve()
    with file_lock(output.parent / f".{output.name}.lock"):
        return _train(config, output, resume=resume)


def _train(config: ControlConfig, output: Path, *, resume: bool) -> dict[str, Any]:
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(config.device)
    sources = (Path(config.baseline_dir).resolve(), Path(config.donor_dir).resolve())
    if output in sources:
        raise ValueError("control must preserve its source directories")
    if not resume:
        learned._fresh_output(output)  # pyright: ignore[reportPrivateUsage]
    elif not (output / "last.pt").is_file():
        raise FileNotFoundError("resume requires last.pt")
    data, cache = load_data(Path(config.data_dir))
    if cache["settings"]["profile"] != config.profile or config.batch_size > len(data["train"]):
        raise ValueError("cache profile or batch size differs")
    baseline, baseline_record = _parent(sources[0], "baseline", config)
    donor, donor_record = _parent(sources[1], "full", config)
    if any(
        record["data_identity"] != cache["identity"] for record in (baseline_record, donor_record)
    ):
        raise ValueError("parents must use this exact data cache")
    tau = float(baseline_record["tau_prediction"])
    if not math.isfinite(tau) or tau <= 0 or tau != float(donor_record["tau_prediction"]):
        raise ValueError("parents must share the same positive calibrated prediction bound")
    pairs = learned_pairs(learned._predictor(donor).thresholded_graphs()["parameter"])  # pyright: ignore[reportPrivateUsage]
    saved_initial: dict[str, Any] = {}
    if resume:
        saved_initial = cast(
            dict[str, Any],
            torch.load(output / "initial.pt", map_location="cpu", weights_only=False),
        )
        scale = cast(Tensor, saved_initial["model"]["code_scale"])
    else:
        scale = training_scale(donor.to(device), data["train"], device).float()
        donor.cpu()
    seed_everything(config.seed)
    model = FrozenCodeModel(baseline, donor, config.condition, scale, pair_rotation(pairs)).to(
        device
    )
    if resume:
        model.load_state_dict(saved_initial["model"], strict=True)
    frozen_identity = model.frozen_identity()
    initial_identities = learned._initial_identities(model)  # pyright: ignore[reportPrivateUsage]
    controller = GECOConfig(enabled=True, tau=1.0, step_size=0.1)
    objective = TrainingObjective(objective_config(config.condition, 0.0), controller).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0)
    sampler = GroupSampler(len(data["train"]), config.batch_size, config.seed)
    source_identity = code_identity()
    scientific_config = {
        "condition": config.condition,
        "seed": config.seed,
        "steps": config.steps,
        "profile": config.profile,
        "batch_size": config.batch_size,
        "lr": 0.001,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
        "objective": asdict(objective_config(config.condition, 0.0)),
        "optimizer_policy": "fresh AdamW, GECO and group sampler in both arms",
        "state_initialization": (
            "baseline online encoder, EMA target, predictor and state/action gates"
        ),
        "parameter_initialization": "full online encoder and parameter encoder, both frozen",
        "parameter_gate_initialization": "fresh +3 in full; fixed-open in baseline",
        "code_transform": (
            "TRAIN population-SD division, no centering, fixed 45-degree pair rotations"
        ),
        "pair_selection": "identical columns of the donor's thresholded learned parameter graph",
        "predictor_compensation": False,
        "target_policy": "baseline EMA target continued with decay .996",
        "test_evaluated": False,
    }
    specification: dict[str, Any] = {
        "schema": SCHEMA,
        "scientific_config": scientific_config,
        "data_identity": cache["identity"],
        "code_identity": source_identity,
        "baseline_parent": baseline_record,
        "donor_parent": donor_record,
        "tau_prediction": tau,
        "geco": asdict(controller),
        "learned_pairs": pairs,
        "scale": [float(value) for value in scale],
        "frozen_identity": frozen_identity,
        **initial_identities,
    }
    identity = canonical_fingerprint(specification)
    restored: dict[str, Any] = {}
    start = 0
    if resume:
        restored = cast(
            dict[str, Any], torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        )
        if restored.get("identity") != identity or restored.get("initial_checkpoint_sha256") != sha(
            output / "initial.pt"
        ):
            raise ValueError("resume source, protocol, parents or initial checkpoint differs")
        model.load_state_dict(restored["model"], strict=True)
        objective.load_state_dict(restored["objective"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        sampler.load_state_dict(restored["sampler"])
        start = int(restored["step"])
        if not 0 <= start <= config.steps or model.frozen_identity() != frozen_identity:
            raise ValueError("resume budget or frozen branch differs")
        assert objective.geco is not None
        objective.geco.validate_state()
        if int(objective.geco.num_updates) != start:
            raise ValueError("GECO and successful update counts disagree")
        torch.set_rng_state(restored["torch_rng"])
        random.setstate(restored["python_rng"])
        np.random.set_state(restored["numpy_rng"])  # noqa: NPY002
        if device.type == "cuda":
            torch.cuda.set_rng_state(restored["cuda_rng"], device)
        if (output / "completion.json").exists():
            return verify_completion(output)
    else:
        save_torch(output / "initial.pt", {"model": model.state_dict(), "identity": identity})
    initial_sha = sha(output / "initial.pt")
    record = specification | {
        "identity": identity,
        "config": asdict(config),
        "initial_checkpoint_sha256": initial_sha,
        "physical_state_training": False,
        "true_parameter_training": False,
        "runtime": {
            "torch": str(torch.__version__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
    }
    write_json(output / "config.json", record)
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=f"frozen-mixed-code-{config.condition}-s{config.seed}",
            job_type="frozen-code-control",
            tags=["diagnostic", "frozen-mixed-code", config.condition],
        ),
        resolved_config=cast(dict[str, object], record),
        dataset_identity=str(cache["identity"]),
        out_dir=output,
        resume_state=restored.get("wandb"),
    )
    write_json(output / "wandb.json", logger.metadata)

    def save(step: int) -> None:
        save_torch(
            output / "last.pt",
            {
                "schema": SCHEMA,
                "identity": identity,
                "initial_checkpoint_sha256": initial_sha,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "objective": objective.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),  # noqa: NPY002
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
            },
        )

    def evaluate_and_log(step: int) -> tuple[dict[str, EvaluationResult], dict[str, list[int]]]:
        results, matching = collect(model, data, config.batch_size, device, objective, tau)
        learned._log(  # pyright: ignore[reportPrivateUsage]
            logger,
            output,
            step,
            {  # pyright: ignore[reportPrivateUsage]
                f"{split}/{key}": value
                for split, result in results.items()
                for key, value in result.metrics.items()
            },
        )
        return results, matching

    started = time.monotonic()
    try:
        if not resume:
            initial, _ = evaluate_and_log(0)
            write_json(
                output / "startup.json",
                {
                    "metrics": {split: result.metrics for split, result in initial.items()},
                    "code_fingerprints": {
                        split: tensor_fingerprint(result.samples["codes"])
                        for split, result in initial.items()
                    },
                },
            )
            save(0)
        model.train()
        objective.train()
        for step in range(start + 1, config.steps + 1):
            indices = sampler.next()
            inputs = {
                key: data["train"].tensors[key][indices].to(device) for key in learned.INPUT_KEYS
            }
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed((1 << 40) + config.seed + step)  # pyright: ignore[reportUnknownMemberType]
            loss, terms, _ = learned.forward_loss(
                model, inputs, objective, forward_seed=config.seed + step
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss at update {step}")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            model.update_target()
            assert objective.geco is not None
            objective.geco.update(terms["prediction"].detach() / tau)
            if step % config.log_every == 0 or step == config.steps:
                metrics = {f"loss/{key}": float(value.detach()) for key, value in terms.items()}
                metrics.update(
                    {
                        "loss/total": float(loss.detach()),
                        "train/gradient_norm": float(gradient),
                        "train/successful_updates": float(step),
                        "train/elapsed_seconds": time.monotonic() - started,
                        "geco/penalty_weight": float(objective.geco.penalty_weight),
                        "geco/log10_penalty_weight": -float(objective.geco.log_lambda)
                        / math.log(10),
                        "geco/moving_violation": float(objective.geco.ma_error),
                    }
                )
                for name, logits in learned._predictor(model).learned_logits().items():  # pyright: ignore[reportPrivateUsage]
                    metrics[f"masks/{name}_edges"] = float((logits.detach() > 0).sum())
                    metrics[f"masks/{name}_open_probability"] = float(
                        torch.sigmoid(logits.detach()).mean()
                    )
                learned._log(logger, output, step, metrics)  # pyright: ignore[reportPrivateUsage]
            if step % config.evaluate_every == 0 and step != config.steps:
                evaluate_and_log(step)
            if step % config.checkpoint_every == 0 or step == config.steps:
                if model.frozen_identity() != frozen_identity:
                    raise RuntimeError("frozen parameter branch changed")
                save(step)
        final, matching = evaluate_and_log(config.steps)
        expected = read_json(output / "startup.json")["code_fingerprints"]
        if any(
            tensor_fingerprint(result.samples["codes"]) != expected[split]
            for split, result in final.items()
        ):
            raise RuntimeError("frozen context-to-code outputs changed")
        write_json(output / "matching.json", matching)
        names = [
            "initial.pt",
            "last.pt",
            "config.json",
            "startup.json",
            "history.jsonl",
            "wandb.json",
            "matching.json",
        ]
        for split, result in final.items():
            write_json(output / f"{split}.json", {"metrics": result.metrics})
            save_torch(
                output / f"{split}.pt", {"metrics": result.metrics, "samples": result.samples}
            )
            names.extend([f"{split}.json", f"{split}.pt"])
        logger.finish()
        if code_identity() != source_identity or model.frozen_identity() != frozen_identity:
            raise RuntimeError("source or frozen branch changed during training")
        for source, parent in zip(sources, (baseline_record, donor_record), strict=True):
            if learned.verify_completion(source, kind="train")["hashes"] != parent["hashes"]:
                raise RuntimeError("parent artifacts changed")
        completion = {
            "schema": SCHEMA,
            "identity": identity,
            "code_identity": source_identity,
            "condition": config.condition,
            "seed": config.seed,
            "steps": config.steps,
            "successful_updates": config.steps,
            "data_identity": cache["identity"],
            "tau_prediction": tau,
            "baseline_parent_identity": baseline_record["identity"],
            "donor_parent_identity": donor_record["identity"],
            "frozen_identity": frozen_identity,
            "frozen_branch_unchanged": True,
            "test_evaluated": False,
            "hashes": {name: sha(output / name) for name in names},
        }
        write_json(output / "completion.json", completion)
        return completion
    except BaseException:
        logger.finish(exit_code=1)
        raise


def main() -> None:
    """Launch or resume the fixed-budget, validation-only frozen-code control."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "baseline-dir", "donor-dir", "data-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--condition", choices=("baseline", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--profile", choices=("exploratory", "smoke"), default="exploratory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--evaluate-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--wandb-mode", default="disabled")
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="masked-mlp-frozen-code")
    parser.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    resume = bool(arguments.pop("resume"))
    result = train(ControlConfig(**arguments), resume=resume)
    print(json.dumps({key: result[key] for key in ("identity", "condition", "seed", "steps")}))


if __name__ == "__main__":
    main()
