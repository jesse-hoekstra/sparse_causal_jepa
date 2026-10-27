"""Run bounded joint-acquisition diagnostics on verified disposable copies."""

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.encoder_gradient_control.source import (
    build_source_model,
    validate_output_destination,
)
from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    build_matched_batch,
    compare_compatible_codes,
    evaluate_responses,
    numerical_checks,
    predict_matched,
    response_loss,
    selected_split,
    validation_retention,
)
from diagnostics.joint_acquisition.code_repair import bounded_code_report
from diagnostics.joint_acquisition.corrections import (
    CASES,
    CorrectionSettings,
    run_shared_corrections,
)
from diagnostics.joint_acquisition.report import decision_report
from diagnostics.joint_acquisition.source import (
    JOINT_CHECKPOINT_SHA256,
    load_comparison_sources,
    load_primary_source,
)
from diagnostics.joint_acquisition.tracking import publish_results
from diagnostics.joint_acquisition.trajectories import TrajectorySettings, run_trajectory
from diagnostics.multiquery_control.run import atomic_save, sha
from scjepa.data.datasets import canonical_fingerprint
from scjepa.experiments.storage import file_lock, read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng


@dataclass(frozen=True)
class Config:
    """Predeclared diagnostic policy; verification mode cannot claim scientific results."""

    source_dir: str
    output_dir: str
    training_source_root: str | None = None
    comparison_bundle: str | None = None
    all_six_checkpoint: str | None = None
    current_checkpoint: str | None = None
    seed: int = 202610261301
    groups: int = 64
    steps: int = 1000
    bootstrap_samples: int = 1000
    displacement: float = 0.01
    rcond: float = 1e-6
    learning_rate: float = 0.001
    device: str = "cpu"
    threads: int = 1
    verification: bool = False
    expected_source_step: int = 30000
    expected_checkpoint_sha256: str | None = JOINT_CHECKPOINT_SHA256
    wandb_mode: str = "disabled"
    wandb_project: str = "sparse_causal_jepa"
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        """Preserve the agreed scientific budget and allow only tiny software fixtures."""
        if self.groups < 2 or self.threads < 1 or self.bootstrap_samples < 2:
            raise ValueError("Need at least two groups/bootstrap draws and a positive thread count")
        if self.verification:
            if self.steps > 3 or self.groups > 4 or self.device != "cpu":
                raise ValueError("Verification mode is CPU-only, at most four groups/three updates")
        elif self.groups != 64 or self.steps != 1000 or self.expected_source_step != 30000:
            raise ValueError(
                "Scientific protocol requires 64 groups, 1000 updates and the 30k source"
            )
        if not self.verification and self.expected_checkpoint_sha256 != JOINT_CHECKPOINT_SHA256:
            raise ValueError("Scientific source must match the recorded original joint checkpoint")
        optional = (self.comparison_bundle, self.all_six_checkpoint, self.current_checkpoint)
        if any(optional) and not all(optional):
            raise ValueError(
                "Compatible comparison requires bundle, actual source and current checkpoint"
            )
        TrajectorySettings(updates=self.steps, learning_rate=self.learning_rate)
        WandbConfig(mode=self.wandb_mode, project=self.wandb_project, entity=self.wandb_entity)
        for source in (self.source_dir, self.comparison_bundle, self.training_source_root):
            if source:
                validate_output_destination(Path(source), Path(self.output_dir))
        for checkpoint in (self.all_six_checkpoint, self.current_checkpoint):
            if checkpoint:
                validate_output_destination(Path(checkpoint).parent, Path(self.output_dir))


def code_identity() -> str:
    """Identify runtime diagnostics and their production dependencies by actual bytes."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for directory in ("src/scjepa", "diagnostics"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _source_unchanged(root: Path, hashes: dict[str, str]) -> None:
    for relative, expected in hashes.items():
        if sha(root / relative) != expected:
            raise RuntimeError(f"Source artifact changed during diagnosis: {relative}")


def run(config: Config, *, resume: bool = False) -> dict[str, Any]:
    """Produce traceable local and trajectory evidence without modifying source artifacts."""
    # Imports remain local to keep preflight/CLI useful without evaluating stages.
    from diagnostics.joint_acquisition.incentives import (
        analyze_incentives,
        recorded_objective_components,
    )

    device = torch.device(config.device)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(out / ".lock"), preserve_rng(device):
            bundle = load_primary_source(
                Path(config.source_dir),
                config.seed,
                expected_step=config.expected_source_step,
                expected_checkpoint_sha256=config.expected_checkpoint_sha256,
                training_source_root=Path(config.training_source_root)
                if config.training_source_root
                else None,
            )
            model, optimizer = build_source_model(bundle, device)
            model.eval()
            selection = DiagnosticSelection(
                calibration_indices=tuple(range(config.groups)),
                validation_indices=tuple(range(config.groups)),
            )
            compatible = None
            if config.comparison_bundle and config.all_six_checkpoint and config.current_checkpoint:
                compatible = load_comparison_sources(
                    Path(config.comparison_bundle),
                    Path(config.all_six_checkpoint),
                    Path(config.current_checkpoint),
                    device=device,
                )
            settings = asdict(config)
            for key in (
                "source_dir",
                "output_dir",
                "training_source_root",
                "wandb_mode",
                "wandb_entity",
                "wandb_project",
            ):
                settings.pop(key)
            evidence = {
                "settings": settings,
                "source_hashes": bundle.hashes,
                "training_code_identity": bundle.protocol["code_identity"],
                "diagnostic_code_identity": code_identity(),
                "selection": asdict(selection),
                "torch": str(torch.__version__),
                "comparison": compatible.identity if compatible else None,
                "numerical_policy": {
                    "shared_correction": asdict(CorrectionSettings()),
                    "code_repair": {"max_displacement_std": 0.1, "fractions": [1, 0.1, 0.01]},
                    "derivatives": {
                        "chain_rule_relative_error": 1e-6,
                        "finite_relative_error": 1e-3,
                        "reconstruction_rms": 2e-5,
                        "scope": "first selected calibration group, donor 0",
                    },
                    "bootstrap": {
                        "unit": "parameter group",
                        "confidence": 0.95,
                        "samples": config.bootstrap_samples,
                        "seed": config.seed,
                    },
                    "ordinary_reference_batch": {
                        "split": "calibration",
                        "indices": list(range(bundle.batch_size)),
                        "donor_shift": 1,
                        "projection_seed": config.seed,
                    },
                    "trajectory": asdict(
                        TrajectorySettings(
                            updates=config.steps,
                            learning_rate=config.learning_rate,
                            seed=config.seed,
                        )
                    ),
                },
            }
            identity = canonical_fingerprint(evidence)
            protocol_path = out / "protocol.json"
            if resume:
                if read_json(protocol_path)["identity"] != identity:
                    raise ValueError(
                        "Diagnostic settings, sources or code changed; incompatible resume"
                    )
            elif any(path.name != ".lock" for path in out.iterdir()):
                raise FileExistsError("Use a new output directory or explicit --resume")
            else:
                write_json(
                    protocol_path,
                    {
                        "identity": identity,
                        **evidence,
                        "config": asdict(config),
                        "hypothesis": (
                            "Useful physical responses may be accessible through codes, shared "
                            "encoder/predictor changes, or their interaction but unsupported by "
                            "ordinary updates."
                        ),
                        "controlled_change": (
                            "Diagnostic code/weight corrections on copies; frozen states, original "
                            "DGP and ordinary objective unchanged."
                        ),
                        "budget": (
                            "1000 diagnostic updates per case after valid finite checks; tiny "
                            "fixtures are software checks only."
                        ),
                        "solver": (
                            "Full calibration fit-query normalized finite-response loss; per-block "
                            "gradient local repair, then fresh diagnostic Adam lr as configured."
                        ),
                        "decision_branches": (
                            "Code accessibility; encoder, predictor or coupled limitation; "
                            "objective/Adam conflict; delayed improvement; "
                            "unresolved within budget."
                        ),
                        "random_state": (
                            "Initial caller RNG in initial_rng.pt; stage policies use fixed seeds; "
                            "trajectory checkpoints retain their own RNG."
                        ),
                    },
                )
                atomic_save(
                    out / "initial_rng.pt",
                    {
                        "torch": torch.get_rng_state(),
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),  # noqa: NPY002
                        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    },
                )
            if (out / "completed.json").exists():
                completed = cast(dict[str, Any], read_json(out / "completed.json"))
                if completed["identity"] != identity or any(
                    sha(out / name) != value for name, value in completed["outputs"].items()
                ):
                    raise ValueError("Completed diagnostic output identity changed")
                return cast(dict[str, Any], read_json(out / "results.json"))
            tracking_path = out / "tracking.json"
            logger = WandbLogger(
                WandbConfig(
                    mode=config.wandb_mode,
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    name=f"joint-acquisition-{out.name}",
                    job_type="diagnostic",
                ),
                resolved_config={"diagnostic_identity": identity, **asdict(config)},
                dataset_identity=bundle.protocol["identity"],
                out_dir=out,
                resume_state=read_json(tracking_path) if tracking_path.exists() else None,
            )
            write_json(tracking_path, logger.metadata)
            try:
                geometry_model = copy.deepcopy(model).double().eval()
                calibration = build_matched_batch(
                    geometry_model,
                    bundle.evaluation,
                    selection,
                    "calibration",
                    displacement=config.displacement,
                )
                validation = build_matched_batch(
                    geometry_model,
                    bundle.evaluation,
                    selection,
                    "validation",
                    displacement=config.displacement,
                )
                assessment = validation.select_queries("heldout")
                fitting = calibration.select_queries("fit")
                baseline, baseline_raw = evaluate_responses(
                    geometry_model,
                    assessment,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                retention = validation_retention(geometry_model, bundle.evaluation, selection)
                numerical, numerical_raw = numerical_checks(
                    geometry_model, bundle.evaluation, selection
                )
                code_repair, code_raw = bounded_code_report(
                    geometry_model,
                    calibration,
                    code_std=retention["calibration_code_std"],
                    rcond=config.rcond,
                    bootstrap_samples=config.bootstrap_samples,
                    seed=config.seed,
                )

                def fit_loss(candidate: SCJEPA) -> Tensor:
                    return response_loss(candidate, fitting)

                def assess(candidate: SCJEPA) -> dict[str, Any]:
                    summary, raw = evaluate_responses(
                        candidate,
                        assessment,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    return {
                        **summary,
                        "raw_groups": {
                            name: value.tolist()  # pyright: ignore[reportUnknownMemberType]
                            for name, value in raw.items()
                        },
                    }

                def assess_trajectory(candidate: SCJEPA) -> dict[str, Any]:
                    return {
                        **assess(candidate),
                        "retention": validation_retention(candidate, bundle.evaluation, selection),
                    }

                split = selected_split(bundle.evaluation, selection, "calibration")
                fitting_batch = {
                    name: split[key][: bundle.batch_size]
                    for name, key in (
                        ("online", "states_online"),
                        ("target", "states_target"),
                        ("actions", "actions"),
                    )
                }

                def components(candidate: SCJEPA) -> dict[str, Tensor]:
                    return recorded_objective_components(
                        candidate,
                        fitting_batch,
                        ObjectiveConfig(**bundle.protocol["objective_config"]),
                        seed=config.seed,
                    )

                shared = run_shared_corrections(model, optimizer, fit_loss, assess, components)
                shared.report["numerically_valid"] = (
                    shared.report["numerically_valid"] and numerical["passed"]
                )
                incentives = analyze_incentives(model, components, shared, fit_loss=fit_loss)
                comparison: dict[str, Any] = {
                    "status": "unavailable",
                    "reason": "Actual compatible source/current checkpoints were not supplied.",
                }
                comparison_raw: dict[str, Any] = {}
                if compatible is not None:
                    comparison, comparison_raw = compare_compatible_codes(
                        compatible.source_model,
                        compatible.current_model,
                        compatible.evaluation,
                        selection,
                        displacement=config.displacement,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                    comparison["status"] = "measured"
                trajectories: dict[str, Any] = {}
                for case in CASES:
                    if not shared.report["numerically_valid"]:
                        trajectories[case] = {"status": "not_run_invalid_local_checks"}
                        continue

                    def progress(row: dict[str, Any], case: str = case) -> None:
                        logger.log(
                            int(row["step"]),
                            {
                                f"{case}/response_fit_loss": float(row["fit_loss"]),
                                f"{case}/code_displacement_rms": float(
                                    row["code_displacement_rms"]
                                ),
                            },
                        )

                    trajectory_path = out / case / "last.pt"
                    trajectories[case] = run_trajectory(
                        geometry_model,
                        fit_loss,
                        assess_trajectory,
                        lambda candidate: predict_matched(candidate, calibration).codes,
                        case=case,
                        checkpoint_path=trajectory_path,
                        identity=identity,
                        local_checks=shared.report,
                        settings=TrajectorySettings(
                            updates=config.steps,
                            learning_rate=config.learning_rate,
                            seed=config.seed,
                        ),
                        resume=resume and trajectory_path.exists(),
                        progress=progress,
                    )
                result: dict[str, Any] = {
                    "identity": identity,
                    "scientific_status": "software verification only"
                    if config.verification
                    else "diagnostic checkpoint repair; acquisition unresolved",
                    "baseline": baseline,
                    "numerical_checks": numerical,
                    "retention": retention,
                    "code_repair": code_repair,
                    "shared_corrections": shared.report,
                    "objective_incentives": incentives,
                    "compatible_control": comparison,
                    "trajectories": trajectories,
                    "tracking": logger.metadata,
                }
                atomic_save(
                    out / "measurements.pt",
                    {
                        "baseline": baseline_raw,
                        "numerical_checks": numerical_raw,
                        "code_repair": code_raw,
                        "comparison": comparison_raw,
                        "shared_directions": shared.directions,
                        "saved_optimizer_update": shared.saved_update,
                    },
                )
                write_json(out / "results.json", result)
                (out / "report.md").write_text(decision_report(result))
                _source_unchanged(Path(config.source_dir), bundle.hashes)
                outputs = {
                    name: sha(out / name)
                    for name in ("results.json", "measurements.pt", "report.md", "initial_rng.pt")
                }
                for case in CASES:
                    relative = f"{case}/last.pt"
                    if (out / relative).exists():
                        outputs[relative] = sha(out / relative)
                publish_results(logger, result, out)
                logger.finish()
                write_json(out / "completed.json", {"identity": identity, "outputs": outputs})
                return cast(dict[str, Any], read_json(out / "results.json"))
            except BaseException:
                logger.finish(exit_code=1)
                raise
    finally:
        torch.set_num_threads(old_threads)


def main() -> None:
    """Parse a fixed-budget scientific diagnostic or explicitly tiny software verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "output-dir"):
        parser.add_argument("--" + name, required=True)
    for name in (
        "training-source-root",
        "comparison-bundle",
        "all-six-checkpoint",
        "current-checkpoint",
        "wandb-entity",
    ):
        parser.add_argument("--" + name)
    for name, default in (
        ("seed", 202610261301),
        ("groups", 64),
        ("steps", 1000),
        ("bootstrap-samples", 1000),
        ("threads", 1),
        ("expected-source-step", 30000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("displacement", 0.01), ("rcond", 1e-6), ("learning-rate", 0.001)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="disabled"
    )
    parser.add_argument("--wandb-project", default="sparse_causal_jepa")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    resume = bool(args.pop("resume"))
    if args["verification"]:
        args["expected_checkpoint_sha256"] = None
    result = run(Config(**args), resume=resume)
    print(json.dumps({"identity": result["identity"], "status": result["scientific_status"]}))


if __name__ == "__main__":
    main()
