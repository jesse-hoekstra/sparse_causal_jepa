"""Audit aggregate theta use in a frozen endpoint on its existing validation groups.

Factual observations/actions/targets stay fixed. Only the supplied parameter
vector changes: factual, another group's theta under one seeded cycle, or zero.
This is predictive dependence, not a per-parameter causal-fidelity certificate.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.harder_oracle.data import StateData, load_data
from diagnostics.harder_oracle.model import build_control
from diagnostics.harder_oracle.run import TrainConfig, verify_completion
from diagnostics.oracle_state_learning.data import read_json, save_torch, sha
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.rng import preserve_rng

PERMUTATION_SEED = 20261028


def cyclic_donors(groups: int, seed: int = PERMUTATION_SEED) -> Tensor:
    """Create one deterministic cycle: every group donates once and never to itself."""
    if groups < 2:
        raise ValueError("Parameter permutation requires at least two independent groups")
    order = torch.randperm(groups, generator=torch.Generator().manual_seed(seed))
    donors = torch.empty_like(order)
    donors[order] = order.roll(1)
    return donors


def _model_identity(model: SCJEPA) -> str:
    state = cast(dict[str, object], model.state_dict())
    return canonical_fingerprint(
        {
            key: tensor_fingerprint(value) if isinstance(value, Tensor) else value
            for key, value in state.items()
        }
    )


@torch.no_grad()
def measure(
    model: SCJEPA, data: StateData, *, batch_size: int = 16, device: str = "cpu"
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """Measure three deterministic rollouts without updating weights, EMA or RNG."""
    if data.split != "validation" or batch_size < 1:
        raise ValueError("Use the original validation split and a positive batch size")
    donors = cyclic_donors(len(data))
    theta = data.tensors["oracle_parameters"]
    modes = [(module, module.training) for module in model.modules()]
    identity = _model_identity(model)
    samples: dict[str, Tensor] = {"donor_indices": donors, "group_ids": data.tensors["episode_ids"]}
    try:
        with preserve_rng(torch.device(device)):
            model.eval()
            for name, values in (
                ("factual", theta),
                ("permuted", theta[donors]),
                ("zero", theta * 0),
            ):
                predictions: list[Tensor] = []
                target_parts: list[Tensor] = []
                for start in range(0, len(data), batch_size):
                    indices = torch.arange(start, min(start + batch_size, len(data)))
                    batch = data.batch(indices, device)
                    batch["oracle_parameters"] = values[indices].to(device)
                    result = cast(
                        ModelOutput,
                        model(
                            batch["observations"],
                            batch["actions"],
                            oracle_parameters=batch["oracle_parameters"],
                        ),
                    )
                    predictions.append(result.predictions.cpu())
                    target_parts.append(result.targets.cpu())
                samples[f"{name}_predictions"] = torch.cat(predictions)
                current_targets = torch.cat(target_parts)
                if "targets" in samples:
                    torch.testing.assert_close(current_targets, samples["targets"], rtol=0, atol=0)
                samples["targets"] = current_targets
    finally:
        for module, training in modes:
            module.training = training
    if identity != _model_identity(model):
        raise RuntimeError("Frozen parameter-use inference changed model tensors")
    targets = samples["targets"].double()
    variance = float(targets.flatten(0, -2).var(0, correction=0).mean())
    if not variance > 0:
        raise ValueError("Parameter-use normalization requires nonconstant target encodings")
    metrics = {"target_variance": variance}
    factual = samples["factual_predictions"].double()
    for name in ("factual", "permuted", "zero"):
        prediction = samples[f"{name}_predictions"].double()
        samples[f"{name}_group_mse"] = (prediction - targets).square().mean((1, 2, 3))
        samples[f"{name}_group_prediction_change"] = (prediction - factual).square().mean((1, 2, 3))
        metrics[f"{name}/mse"] = float(samples[f"{name}_group_mse"].mean())
        metrics[f"{name}/normalized_mse"] = metrics[f"{name}/mse"] / variance
        metrics[f"{name}/normalized_prediction_change"] = (
            float(samples[f"{name}_group_prediction_change"].mean()) / variance
        )
        metrics[f"{name}/normalized_mse_increase"] = (
            metrics[f"{name}/normalized_mse"] - metrics["factual/normalized_mse"]
        )
    return metrics, samples


@dataclass(frozen=True)
class AuditConfig:
    """Point to immutable source artifacts and a separate new audit directory."""

    train_dir: str
    data_dir: str
    output_dir: str
    device: str = "cpu"
    threads: int = 1
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None


def audit(config: AuditConfig) -> dict[str, Any]:
    """Authenticate a final checkpoint, evaluate theta replacements and save evidence."""
    if config.threads < 1:
        raise ValueError("threads must be positive")
    source, output = Path(config.train_dir).resolve(), Path(config.output_dir).resolve()
    completed = verify_completion(source, kind="train", require_current_source=True)
    record = read_json(source / "config.json")
    settings = TrainConfig(**record["config"])
    data, cache = load_data(Path(config.data_dir))
    checkpoint = cast(
        dict[str, Any], torch.load(source / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        checkpoint["identity"] != completed["identity"]
        or checkpoint["step"] != completed["steps"]
        or checkpoint["step"] != settings.steps
        or cache["identity"] != completed["data_identity"]
    ):
        raise ValueError("Completed checkpoint, data or fixed update budget differs")
    saved = cast(dict[str, Any], torch.load(source / "validation.pt", weights_only=True))
    previous_threads = torch.get_num_threads()
    previous_precision = torch.get_float32_matmul_precision()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    logger: WandbLogger | None = None
    try:
        torch.set_num_threads(config.threads)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with (
            file_lock(output.parent / f".{output.name}.lock"),
            preserve_rng(torch.device(config.device)),
        ):
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Parameter-use audit requires a fresh output directory")
            output.mkdir(parents=True, exist_ok=True)
            model = build_control(settings.condition, settings.seed).to(config.device)
            model.load_state_dict(checkpoint["model"], strict=True)
            metrics, samples = measure(
                model, data["validation"], batch_size=settings.batch_size, device=config.device
            )
            for name in ("predictions", "targets"):
                actual = samples["factual_predictions" if name == "predictions" else name]
                torch.testing.assert_close(actual, saved["samples"][name], rtol=2e-5, atol=2e-5)
            evidence: dict[str, Any] = {
                "schema": "harder-oracle-parameter-use-v1",
                "parent_identity": completed["identity"],
                "source_completion_sha256": sha(source / "completion.json"),
                "source_checkpoint_sha256": sha(source / "last.pt"),
                "source_matching_sha256": sha(source / "matching.json"),
                "analysis_source_sha256": sha(Path(__file__)),
                "data_identity": cache["identity"],
                "condition": settings.condition,
                "steps": settings.steps,
                "split": "validation",
                "permutation_seed": PERMUTATION_SEED,
                "group_count": len(data["validation"]),
                "interpretation": "aggregate parameter predictive use, not per-factor fidelity",
                "metrics": metrics,
            }
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"harder-parameter-use-{settings.condition}",
                    group="harder-oracle-parameter-use",
                    job_type="frozen-parameter-use",
                ),
                resolved_config=evidence,
                dataset_identity=str(cache["identity"]),
                out_dir=output,
            )
            save_torch(output / "samples.pt", samples)
            write_json(output / "summary.json", evidence)
            write_json(output / "wandb.json", logger.metadata)
            logger.log(settings.steps, metrics | {"diagnostic/complete": 1.0})
            logger.finish()
            if verify_completion(source, kind="train", require_current_source=True) != completed:
                raise RuntimeError("Source artifacts changed during the read-only audit")
            completion = evidence | {
                "wandb": logger.metadata,
                "hashes": {
                    name: sha(output / name)
                    for name in ("summary.json", "samples.pt", "wandb.json")
                },
            }
            write_json(output / "completion.json", completion)
            return completion
    except BaseException:
        if logger is not None:
            logger.finish(exit_code=1)
        raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def main() -> None:
    """Evaluate an authenticated endpoint without invoking training or test selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-dir", "data-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--wandb-entity")
    print(json.dumps(audit(AuditConfig(**vars(parser.parse_args()))), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
