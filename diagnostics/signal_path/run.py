"""Locate signal loss in finished frozen-state prediction-weight comparisons."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from diagnostics.multiquery_control.data import EvaluationData, module_fingerprint
from diagnostics.multiquery_control.run import atomic_save, frozen_hashes, resolve_device, sha
from diagnostics.predictive_supervision_control.evaluation import weight_name
from diagnostics.predictive_supervision_control.run import code_identity as training_code_identity
from diagnostics.signal_path.geometry import geometry_diagnostic
from diagnostics.signal_path.gradients import gradient_path_diagnostic
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, source_fingerprint, write_json
from scjepa.models.model import SCJEPA
from scjepa.models.parameter_encoder import TransitionParameterEncoder
from scjepa.training.factory import build_model
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """Bounded evaluation settings; neither source weights nor Adam moments are updated."""

    experiment_dir: str
    output_dir: str
    prediction_weights: tuple[float, ...] = (1.0, 4.0, 16.0)
    seed: int = 202610261301
    device: str = "cpu"
    groups: int = 64
    donor_contexts: int = 2
    fit_queries: int = 4
    test_queries: int = 5
    gradient_groups: int = 64
    batch_size: int = 16
    rcond: float = 1e-6
    displacement: float = 0.01
    threads: int = 1

    def __post_init__(self) -> None:
        """Reject ineffective counts, leaking query splits and unsafe output destinations."""
        for name in (
            "groups",
            "donor_contexts",
            "fit_queries",
            "test_queries",
            "gradient_groups",
            "batch_size",
            "threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.batch_size < 2 or self.gradient_groups % self.batch_size:
            raise ValueError("gradient_groups must be divisible by batch_size>=2")
        if self.donor_contexts > 10 or self.fit_queries + self.test_queries > 9:
            raise ValueError("Use at most 10 donors and at most 9 distinct non-donor queries")
        if not math.isfinite(self.rcond) or not 0 < self.rcond < 1:
            raise ValueError("rcond must be finite and lie in (0,1)")
        if not math.isfinite(self.displacement) or not 0 < self.displacement <= 0.1:
            raise ValueError("displacement must be finite and lie in (0,0.1]")
        if self.seed < 0 or torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("Use a nonnegative seed and CPU or CUDA device")
        if not self.prediction_weights or len(set(self.prediction_weights)) != len(
            self.prediction_weights
        ):
            raise ValueError("prediction_weights must be nonempty and distinct")
        for weight in self.prediction_weights:
            weight_name(weight)
        source, output = Path(self.experiment_dir).resolve(), Path(self.output_dir).resolve()
        if source == output or source in output.parents:
            raise ValueError("Write diagnostic outputs outside the source experiment directory")


def code_identity() -> str:
    """Hash the new diagnostic and reused helpers without changing old training identities."""
    digest = hashlib.sha256(source_fingerprint().encode())
    root = Path(__file__).parents[1]
    for directory in (
        "signal_path",
        "predictive_supervision_control",
        "parameter_sigreg_control",
        "multiquery_control",
    ):
        for path in sorted((root / directory).glob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def model_fingerprint(model: SCJEPA) -> str:
    """Include tensor values and SCJEPA's non-tensor prediction-mode extra state."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def source_records(config: Config) -> tuple[dict[str, Any], EvaluationData, dict[str, Any]]:
    """Verify finished source artifacts before loading the label-bearing evaluation cache."""
    root = Path(config.experiment_dir)
    protocol = cast(dict[str, Any], read_json(root / "protocol.json"))
    if protocol["code_identity"] != training_code_identity():
        raise ValueError(
            "Source training code changed; use its original repository/helper versions "
            "to preserve model semantics and optimizer parameter ordering"
        )
    if config.seed not in protocol["config"]["seeds"]:
        raise ValueError("Requested seed is absent from the source comparison")
    if config.batch_size != protocol["config"]["batch_size"]:
        raise ValueError("Use the source batch size for matched SIGReg gradient measurements")
    evaluation_path = root / "evaluation_data.pt"
    if sha(evaluation_path) != protocol["artifacts"]["evaluation_data.pt"]:
        raise ValueError("Source evaluation cache hash mismatch")
    sources: dict[str, Any] = {}
    for weight in config.prediction_weights:
        condition = weight_name(weight)
        if weight not in protocol["config"]["prediction_weights"]:
            raise ValueError("Requested prediction weight is absent from the source comparison")
        dest = root / f"seed-{config.seed}" / condition
        completed = read_json(dest / "completed.json")
        for filename, key in (
            ("last.pt", "checkpoint_sha256"),
            ("results.json", "results_sha256"),
            ("gradient_history.json", "gradient_history_sha256"),
        ):
            if sha(dest / filename) != completed[key]:
                raise ValueError(f"Completed source artifact changed: {condition}/{filename}")
        if completed["identity"] != protocol["identity"]:
            raise ValueError("Completed source belongs to another protocol")
        result = cast(dict[str, Any], read_json(dest / "results.json"))
        if (
            result["identity"] != protocol["identity"]
            or result["checkpoint_sha256"] != completed["checkpoint_sha256"]
            or result["steps"] != protocol["config"]["steps"]
            or result["seed"] != config.seed
            or result["condition"] != condition
            or result["prediction_weight"] != weight
        ):
            raise ValueError("Source result does not describe the requested final checkpoint")
        sources[condition] = {
            "checkpoint_sha256": completed["checkpoint_sha256"],
            "results_sha256": completed["results_sha256"],
            "gradient_history_sha256": completed["gradient_history_sha256"],
        }
    data = cast(EvaluationData, torch.load(evaluation_path, map_location="cpu", weights_only=True))
    metadata = data["metadata"]
    if (
        metadata["generator_version"] != dgp.GENERATOR_VERSION
        or metadata["observation_map_fingerprint"]
        != protocol["source"]["observation_map_fingerprint"]
    ):
        raise ValueError("Evaluation metadata differ from the frozen source or current DGP")
    if max(config.groups, config.gradient_groups) > len(data["test"]["theta"]):
        raise ValueError("Requested more groups than the source evaluation test split contains")
    return protocol, data, sources


def load_checkpoint(
    config: Config,
    protocol: dict[str, Any],
    weight: float,
    device: torch.device,
    metadata: dict[str, Any],
) -> tuple[SCJEPA, dict[str, Any], dict[str, Any]]:
    """Validate the final model in its stored dtype, then make a float64 evaluation copy."""
    condition = weight_name(weight)
    dest = Path(config.experiment_dir) / f"seed-{config.seed}" / condition
    payload = cast(
        dict[str, Any], torch.load(dest / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        payload["identity"] != protocol["identity"]
        or payload["condition"] != condition
        or payload["prediction_weight"] != weight
        or payload["seed"] != config.seed
        or payload["step"] != protocol["config"]["steps"]
        or payload["source_sha256"] != protocol["source"]["sha256"]
        or payload["initial_sha256"] != protocol["initializations"][str(config.seed)]["sha256"]
        or payload["parameter_regularizer_weight"] != 0.4
        or payload["consistency_weight"] != 1.0
        or payload["queries"] != 1
    ):
        raise ValueError("Checkpoint provenance or coefficients do not match the selected source")
    model = build_model(cast(DictConfig, OmegaConf.create(protocol["model_config"])))
    model.load_state_dict(payload["model"], strict=True)
    if (
        not isinstance(model.parameter_encoder, TransitionParameterEncoder)
        or model.parameter_source != "learned"
        or model.context_len != 10
        or model.predictor.history_len != 1
        or not model.autoregressive
        or (model.predictor.state_dim, model.predictor.param_dim, model.predictor.action_dim)
        != (6, 6, 16)
        or not all(layer.dense for layer in model.predictor.layers)
        or frozen_hashes(model) != protocol["frozen_hashes"]
        or payload["frozen_hashes"] != protocol["frozen_hashes"]
    ):
        raise ValueError("Use the finished original-MLP dense frozen-state comparison")
    if metadata["state_online_sha256"] != module_fingerprint(model.online_encoder) or metadata[
        "state_target_sha256"
    ] != module_fingerprint(model.target_encoder):
        raise ValueError("Evaluation metadata differ from the frozen checkpoint encoders")
    model.online_encoder.requires_grad_(False)
    model.target_encoder.requires_grad_(False)
    model.to(device=device, dtype=torch.float64).eval()
    prior = cast(dict[str, Any], read_json(dest / "results.json"))
    return model, payload, prior


def _one_condition(
    config: Config,
    protocol: dict[str, Any],
    data: EvaluationData,
    source: dict[str, Any],
    identity: str,
    weight: float,
    device: torch.device,
    *,
    resume: bool,
) -> dict[str, Any]:
    condition = weight_name(weight)
    dest = Path(config.output_dir) / f"seed-{config.seed}" / condition
    dest.mkdir(parents=True, exist_ok=True)
    result_path, tensors_path = dest / "results.json", dest / "signal_tensors.pt"
    if (dest / "completed.json").exists():
        complete = read_json(dest / "completed.json")
        if (
            not resume
            or complete["identity"] != identity
            or complete["results_sha256"] != sha(result_path)
            or complete["tensors_sha256"] != sha(tensors_path)
        ):
            raise ValueError("Completed diagnostic changed or --resume was not supplied")
        print(json.dumps({"skipping_completed": condition}), flush=True)
        return cast(dict[str, Any], read_json(result_path))
    started = time.perf_counter()
    model, payload, prior = load_checkpoint(config, protocol, weight, device, data["metadata"])
    before = model_fingerprint(model)
    print(
        json.dumps({"condition": condition, "phase": "geometry", "groups": config.groups}),
        flush=True,
    )
    geometry_model = copy.deepcopy(model).requires_grad_(False)
    geometry, geometry_tensors = geometry_diagnostic(
        geometry_model,
        data,
        groups=config.groups,
        donor_contexts=config.donor_contexts,
        fit_queries=config.fit_queries,
        test_queries=config.test_queries,
        rcond=config.rcond,
        displacement=config.displacement,
    )
    if model_fingerprint(geometry_model) != before:
        raise RuntimeError("Geometry evaluation mutated model weights or buffers")
    del geometry_model
    print(
        json.dumps(
            {
                "condition": condition,
                "phase": "gradient_transmission",
                "groups": config.gradient_groups,
            }
        ),
        flush=True,
    )
    gradient, gradient_tensors = gradient_path_diagnostic(
        model,
        data,
        ObjectiveConfig(**protocol["objective_config"]),
        weight,
        groups=config.gradient_groups,
        batch_size=config.batch_size,
        optimizer_state=payload["optimizer"],
        rcond=config.rcond,
    )
    if model_fingerprint(model) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Evaluation mutated model state or gradient buffers")
    if (
        sha(Path(config.experiment_dir) / f"seed-{config.seed}" / condition / "last.pt")
        != source["checkpoint_sha256"]
    ):
        raise RuntimeError("Source checkpoint changed during evaluation")
    atomic_save(tensors_path, {"geometry": geometry_tensors, "gradients": gradient_tensors})
    result = {
        "identity": identity,
        "condition": condition,
        "prediction_weight": weight,
        "seed": config.seed,
        "source": source,
        "source_step": payload["step"],
        "geometry": geometry,
        "gradients": gradient,
        "prior_final_evaluation": {
            key: prior["metrics"][key]
            for key in (
                "prediction_mse",
                "linear_r2",
                "parameter_readouts",
                "true_parameter_interventions",
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limits": [
            "Float64 local evaluation of stored weights, not byte-identical CUDA training "
            "arithmetic.",
            "Local sensitivities and finite readouts do not certify global information "
            "preservation or absence.",
            "Code spectra and Euclidean code displacement sizes depend on parameterization; "
            "code axes are never assigned physical meanings.",
            "Fitted code directions may leave the encoder image; inspect held-out responses, "
            "required displacements and nonlinear checks together.",
            "Projected residual losses are local tangent-aligned diagnostics, not attribution "
            "of error to an incorrect physical parameter; they overlap and do not sum to MSE.",
            "Adam directional changes use fixed checkpoint moments independently on held-out "
            "batches; no optimizer step is taken and this is not the historical trajectory.",
        ],
    }
    write_json(result_path, result)
    write_json(
        dest / "completed.json",
        {
            "identity": identity,
            "results_sha256": sha(result_path),
            "tensors_sha256": sha(tensors_path),
        },
    )
    print(
        json.dumps({"completed": condition, "elapsed_seconds": result["elapsed_seconds"]}),
        flush=True,
    )
    return result


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Read finished arms sequentially and save diagnostics outside the source directory."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use ordinary Python on one device, not torchrun")
    torch.set_num_threads(config.threads)
    device = resolve_device(config.device)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".lock"), preserve_rng(device):
        protocol, data, sources = source_records(config)
        settings = asdict(config)
        settings.pop("output_dir")
        settings.pop("experiment_dir")
        evidence = {
            "settings": settings,
            "source_identity": protocol["identity"],
            "sources": sources,
            "evaluation_sha256": protocol["artifacts"]["evaluation_data.pt"],
            "code": code_identity(),
            "torch": str(torch.__version__),
        }
        identity = canonical_fingerprint(evidence)
        record_path = out / "protocol.json"
        if resume:
            if read_json(record_path)["identity"] != identity:
                raise ValueError(
                    "Diagnostic settings, code or source changed; use a new output directory"
                )
        elif any(p.name != ".lock" for p in out.iterdir()):
            raise FileExistsError(
                "Diagnostic output directory is not empty; use --resume or a new directory"
            )
        else:
            write_json(
                record_path,
                {
                    "identity": identity,
                    **evidence,
                    "config": asdict(config),
                    "scope": "Read-only checkpoint diagnosis. "
                    "No training, EMA or optimizer-state updates.",
                },
            )
        results: list[dict[str, Any]] = []
        for weight in config.prediction_weights:
            results.append(
                _one_condition(
                    config,
                    protocol,
                    data,
                    sources[weight_name(weight)],
                    identity,
                    weight,
                    device,
                    resume=resume,
                )
            )
            write_json(out / "comparison.json", {"identity": identity, "results": results})
    return {"identity": identity, "results": results}


def main() -> None:
    """Parse a bounded read-only diagnostic of final prediction-weight checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prediction-weights", nargs="+", type=float, choices=[1, 4, 16], default=[1, 4, 16]
    )
    parser.add_argument("--device", default="cpu")
    for name, default in (
        ("seed", 202610261301),
        ("groups", 64),
        ("donor-contexts", 2),
        ("fit-queries", 4),
        ("test-queries", 5),
        ("gradient-groups", 64),
        ("batch-size", 16),
        ("threads", 1),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--rcond", type=float, default=1e-6)
    parser.add_argument("--displacement", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    for key in ("experiment_dir", "output_dir"):
        args[key] = str(Path(args[key]).resolve())
    args["prediction_weights"] = tuple(args["prediction_weights"])
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
