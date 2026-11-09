"""Learn a complete parameter code from reference positions: diagnostic, not the visual model."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.coupled_duffing.acquisition import initialize_shared_output
from diagnostics.coupled_duffing.evaluate import collect, prediction_mse, selected_retention
from diagnostics.coupled_duffing.model import LaggedParameterEncoder, LaggedPredictor
from diagnostics.coupled_duffing.run import code_identity as parent_identity
from diagnostics.coupled_duffing_probe.data import window
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.sampling import GroupSampler, resolve_device
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import write_json
from scjepa.models.encoders import FramewiseEncoder
from scjepa.models.model import SCJEPA, ModelOutput
from scjepa.models.spartan import Spartan
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import seed_everything
from scjepa.training.rng import preserve_rng


class PhysicalEncoder(FramewiseEncoder):
    """Expose the exact visible positions only in this explicitly declared diagnostic."""

    observation_ndim = 1
    state_dim = 2

    def forward(self, observations: Tensor) -> Tensor:
        """Preserve physical coordinates exactly, with no trainable state map."""
        if observations.ndim != 2 or observations.shape[-1] != 2:
            raise ValueError("reference positions must have shape (B,2)")
        return observations


def build_model(seed: int, context: int, code_dim: int = 8) -> SCJEPA:
    """Use the existing parameter/predictor modules with fixed exact online/target states."""
    with preserve_rng(torch.device("cpu")):
        torch.default_generator.manual_seed(seed)
        return SCJEPA(
            PhysicalEncoder(),
            LaggedParameterEncoder(context, code_dim),
            cast(Spartan, LaggedPredictor(sparse=False, code_dim=code_dim)),
            context,
            autoregressive=True,
        )


def code_identity() -> str:
    """Bind this diagnostic and all existing numerical dependencies to completed runs."""
    return canonical_fingerprint(
        {
            "parent": parent_identity(),
            "files": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        }
    )


@dataclass(frozen=True)
class ProbeConfig:
    """One fixed-budget initialization; no acquisition selection or automatic retries."""

    data_dir: str
    output_dir: str
    context: int = 60
    horizon: int = 40
    seed: int = 0
    steps: int = 50000
    batch_size: int = 16
    eval_every: int = 5000
    log_every: int = 100
    device: str = "cuda:0"
    threads: int = 1
    wandb_mode: str = "online"
    wandb_group: str = "duffing-acquisition-screen-20261109"

    def __post_init__(self) -> None:
        """Reject invalid diagnostic windows and budgets."""
        if (
            self.context < 3
            or min(
                self.horizon,
                self.steps,
                self.batch_size,
                self.eval_every,
                self.log_every,
                self.threads,
            )
            < 1
        ):
            raise ValueError("invalid window or budget")


def forward_loss(model: SCJEPA, inputs: dict[str, Tensor]) -> tuple[Tensor, ModelOutput]:
    """Train through attached donor codes and every autoregressive prediction, never theta."""
    if set(inputs) != {"observations", "actions"}:
        raise ValueError("parameter labels are forbidden in diagnostic prediction training")
    output = cast(ModelOutput, model(inputs["observations"], inputs["actions"]))
    return (output.predictions - output.targets).square().mean(), output


def load_data(
    directory: Path, context: int, horizon: int, device: torch.device
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    """Read inputs only; TRAIN parameter labels are never opened."""
    record = read_json(directory / "manifest.json")
    if (
        canonical_fingerprint({k: v for k, v in record.items() if k != "identity"})
        != record["identity"]
    ):
        raise ValueError("data manifest changed")
    inputs: dict[str, dict[str, Tensor]] = {}
    for split in ("train", "calibration", "validation"):
        name = f"{split}-inputs.pt"
        if sha(directory / name) != record["hashes"][name]:
            raise ValueError(f"changed input: {name}")
        payload = cast(dict[str, Tensor], torch.load(directory / name, weights_only=True))
        selected = window(payload, record["config"]["context"], context, horizon)
        inputs[split] = {key: value.to(device) for key, value in selected.items()}
    return record, inputs


def train(config: ProbeConfig, *, resume: bool = False) -> None:
    """Run prediction-only acquisition and report retention/use on calibration/validation."""
    torch.set_num_threads(config.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(config.device)
    seed_everything(config.seed)
    data_dir, output_dir = Path(config.data_dir), Path(config.output_dir)
    record, inputs = load_data(data_dir, config.context, config.horizon, device)
    model = build_model(config.seed, config.context).to(device)
    specification: dict[str, Any] = {
        "schema": "duffing-physical-acquisition-v1",
        "config": asdict(config),
        "data_identity": record["identity"],
        "code_identity": code_identity(),
        "oracle_state_diagnostic": True,
        "training_parameter_labels": False,
        "test_evaluation": False,
        "code_dim": 8,
        "state_encoder": "fixed identity, no learned or lost physical state information",
        "objective": "mean physical-position forecast MSE",
        "state_regularization": "none: state encoder is fixed",
        "sparsity": "none: acquisition diagnostic",
        "optimizer": "AdamW lr=.001 weight_decay=0 gradient_clip=1",
        "donor": "attached non-self episode from same system for all forecast steps",
        "forcing": record["config"]["forcing"],
    }
    specification["identity"] = canonical_fingerprint(specification)
    if output_dir.exists() and not resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=resume)
    saved: dict[str, Any] = {}
    start = 0
    if resume:
        saved = cast(
            dict[str, Any],
            torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False),
        )
        if saved["identity"] != specification["identity"]:
            raise ValueError("resume settings or source changed")
        model.load_state_dict(saved["model"], strict=True)
        start = int(saved["step"])
    else:
        write_json(
            output_dir / "initialization.json", initialize_shared_output(model, inputs["train"])
        )
    write_json(output_dir / "config.json", specification)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    sampler = GroupSampler(len(inputs["train"]["observations"]), config.batch_size, config.seed)
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        sampler.load_state_dict(saved["sampler"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        for name in ("history.jsonl", "validation.jsonl"):
            path = output_dir / name
            path.with_name(f"{name}.before-resume-{time.time_ns()}").write_bytes(path.read_bytes())
            path.write_text(
                "".join(
                    line + "\n"
                    for line in path.read_text().splitlines()
                    if json.loads(line)["step"] <= start
                )
            )
    labels: dict[str, Tensor] = {}
    for split in ("calibration", "validation"):
        name = f"{split}-labels.pt"
        if sha(data_dir / name) != record["hashes"][name]:
            raise ValueError("evaluation labels changed")
        labels[split] = cast(dict[str, Tensor], torch.load(data_dir / name, weights_only=True))[
            "theta"
        ]
    logger = WandbLogger(
        WandbConfig(
            mode=config.wandb_mode,
            entity="jesse-hoekstra-university-of-oxford",
            project="sparse_causal_jepa",
            group=config.wandb_group,
            name=f"physical-{specification['forcing']}-c{config.context}-h{config.horizon}-s{config.seed}",
            job_type="oracle-state-acquisition-diagnostic",
            tags=["duffing", "oracle-state-control", "learned-parameters", "validation-only"],
        ),
        resolved_config=cast(dict[str, object], specification),
        dataset_identity=record["identity"],
        out_dir=output_dir,
        resume_state=saved.get("wandb"),
    )
    write_json(output_dir / "wandb.json", logger.metadata)

    def save(step: int, name: str = "last.pt") -> None:
        save_torch(
            output_dir / name,
            {
                "identity": specification["identity"],
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sampler": sampler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "wandb": logger.metadata,
            },
        )

    def evaluate(step: int, *, final: bool = False) -> dict[str, float]:
        with preserve_rng(device):
            cal = collect(model, inputs["calibration"], config.batch_size)
            val = collect(model, inputs["validation"], config.batch_size)
            retention = selected_retention(
                cal["codes"], labels["calibration"], val["codes"], labels["validation"]
            )
            shuffled = collect(model, inputs["validation"], config.batch_size, shuffle=True)
            mse = prediction_mse(val)
            variance = float(val["targets"].reshape(-1, 2).var(0, correction=0).mean())
            scores = {f"parameter_retention/r2_{i + 1}": float(v) for i, v in enumerate(retention)}
            scores.update(
                {
                    "parameter_retention/min_r2": float(retention.min()),
                    "parameter_retention/mean_r2": float(retention.mean()),
                    "forecast/mse": mse,
                    "forecast/nmse": mse / max(variance, 1e-12),
                    "forecast/endpoint_nmse": float(
                        (val["predictions"][:, :, -1] - val["targets"][:, :, -1]).square().mean()
                    )
                    / max(
                        float(val["targets"][:, :, -1].reshape(-1, 2).var(0, correction=0).mean()),
                        1e-12,
                    ),
                    "code_shuffle/mse_ratio": prediction_mse(shuffled) / max(mse, 1e-12),
                    "code_std/min": float(val["codes"].reshape(-1, 8).std(0).min()),
                    "retention_all_four_pass": float(bool((retention >= 0.8).all())),
                }
            )
            with (output_dir / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": step, **scores}) + "\n")
            logger.log(step, {f"validation/{key}": value for key, value in scores.items()})
            if final:
                save_torch(output_dir / "calibration.pt", cal)
                save_torch(output_dir / "validation.pt", val)
            return scores

    started, status = time.monotonic(), 1
    try:
        if not resume:
            save(0, "initial.pt")
            save(0)
            evaluate(0)
        final_scores: dict[str, float] = {}
        for step in range(start + 1, config.steps + 1):
            model.train()
            indices = sampler.next().to(device)
            batch = {key: values[indices] for key, values in inputs["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            with preserve_rng(device):
                torch.default_generator.manual_seed(config.seed + step)
                if device.type == "cuda":
                    torch.cuda.manual_seed(config.seed + step)
                loss, result = forward_loss(model, batch)
            if len(result.dynamics) != config.horizon or not bool(torch.isfinite(loss)):
                raise FloatingPointError("invalid attached physical forecast")
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            if step % config.log_every == 0 or step == config.steps:
                scores = {
                    "loss/prediction": float(loss.detach()),
                    "train/gradient_norm": float(gradient),
                    "runtime/seconds_per_step": (time.monotonic() - started) / (step - start),
                }
                if device.type == "cuda":
                    scores["runtime/gpu_reserved_gb"] = torch.cuda.max_memory_reserved(device) / 1e9
                with (output_dir / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": step, **scores}) + "\n")
                logger.log(step, scores)
            if step % config.eval_every == 0 or step == config.steps:
                final_scores = evaluate(step, final=step == config.steps)
                save(step)
        if start == config.steps:
            final_scores = evaluate(start, final=True)
        if code_identity() != specification["code_identity"]:
            raise RuntimeError("numerical source changed during the diagnostic")
        names = (
            "config.json",
            "initialization.json",
            "initial.pt",
            "last.pt",
            "history.jsonl",
            "validation.jsonl",
            "calibration.pt",
            "validation.pt",
        )
        write_json(
            output_dir / "completion.json",
            {
                **specification,
                "metrics": final_scores,
                "elapsed_seconds": time.monotonic() - started,
                "hashes": {name: sha(output_dir / name) for name in names},
            },
        )
        status = 0
    finally:
        logger.finish(status)


def main() -> None:
    """Run a single, explicitly labelled reference-state acquisition diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    for name, value in (
        ("context", 60),
        ("horizon", 40),
        ("seed", 0),
        ("steps", 50000),
        ("batch-size", 16),
        ("eval-every", 5000),
        ("log-every", 100),
        ("threads", 1),
    ):
        parser.add_argument(f"--{name}", type=int, default=value)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-group", default="duffing-acquisition-screen-20261109")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    train(ProbeConfig(**args), resume=resume)


if __name__ == "__main__":
    main()
