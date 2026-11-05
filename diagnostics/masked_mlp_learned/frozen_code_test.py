"""Predeclare and score the immutable frozen-code control on reserved test groups.

All state and parameter assignments are frozen from saved calibration data.
The historical parameter branch, predictor and state encoders are restored as
one complete FrozenCodeModel. Test scores never feed back into training.
"""

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from diagnostics.masked_mlp_learned import frozen_code as control
from diagnostics.masked_mlp_learned import frozen_test, run
from diagnostics.shared.data import load_data
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.io import verify_completion as verify_artifacts
from diagnostics.shared.oracle import EvaluationResult, objective_config
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.eval.recovery import CoordinateMatch, mean_correlation_coefficient
from scjepa.experiments.storage import write_json
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective

SCHEMA = "masked-mlp-frozen-code-test-v1"


def dependencies() -> dict[str, str]:
    """Bind the complete numerical evaluator and historical training runtime."""
    return {
        "control_code_identity": control.code_identity(),
        "evaluator_sha256": sha(Path(__file__)),
        "shared_test_scoring_sha256": sha(Path(frozen_test.__file__)),
    }


def load_payload(path: Path) -> dict[str, Any]:
    """Load one authenticated diagnostic checkpoint or saved representation bundle."""
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def parameter_mcc(result: EvaluationResult, columns: list[int]) -> float:
    """Score the supplied calibration assignment without optimizing it on test."""
    codes, theta = control._code_rows(result)  # pyright: ignore[reportPrivateUsage]
    return mean_correlation_coefficient(CoordinateMatch(tuple(columns)), codes, theta)


def restore_model(
    checkpoint: dict[str, Any], arm: control.Arm, seed: int
) -> control.FrozenCodeModel:
    """Restore the separate frozen parameter/state branches from the saved control."""
    baseline = run.build_control("baseline", seed, predictor_activation="gelu")
    donor = run.build_control("full", seed, predictor_activation="gelu")
    state = checkpoint["model"]
    model = control.FrozenCodeModel(
        baseline, donor, arm, state["code_scale"], state["code_rotation"]
    )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval()
    return model


def declare(batch: Path, data_dir: Path, output: Path) -> dict[str, Any]:
    """Freeze all ten 300k endpoints and calibration matchings before loading test."""
    if output.exists():
        raise FileExistsError(output)
    cache = verify_artifacts(data_dir)
    records: list[dict[str, Any]] = []
    for seed in run.SEEDS:
        for arm in ("baseline", "full"):
            directory = (batch / f"{arm}-s{seed}" / "results").resolve()
            completion = control.verify_completion(directory)
            if (
                completion["steps"] != control.STEPS
                or completion["successful_updates"] != control.STEPS
                or completion["condition"] != arm
                or completion["seed"] != seed
                or completion["code_identity"] != control.code_identity()
                or completion["data_identity"] != cache["identity"]
            ):
                raise ValueError("endpoint differs from the fixed five-seed control")
            matching = read_json(directory / "matching.json")
            calibration = load_payload(directory / "calibration.pt")
            sample = EvaluationResult(calibration["metrics"], calibration["samples"])
            if not math.isclose(
                parameter_mcc(sample, matching["code_mcc_columns"]),
                sample.metrics["code_mcc"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("saved calibration parameter assignment did not reproduce")
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "directory": str(directory),
                    "identity": completion["identity"],
                    "steps": control.STEPS,
                    "completion_sha256": sha(directory / "completion.json"),
                    "checkpoint_sha256": completion["hashes"]["last.pt"],
                    "calibration_sha256": completion["hashes"]["calibration.pt"],
                    "matching": {
                        key: matching[key] for key in ("online_columns", "target_columns")
                    },
                    "parameter_columns": matching["parameter_graph_columns"],
                    "code_mcc_columns": matching["code_mcc_columns"],
                    "tau_prediction": completion["tau_prediction"],
                    "batch_size": read_json(directory / "config.json")["config"]["batch_size"],
                }
            )
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "batch_dir": str(batch.resolve()),
        "data_dir": str(data_dir.resolve()),
        "data_identity": cache["identity"],
        "cache_completion_sha256": sha(data_dir / "completion.json"),
        "test_groups": cache["settings"]["test_groups"],
        "test_inputs_sha256": cache["hashes"]["test.pt"],
        "test_states_sha256": cache["hashes"]["test_states.pt"],
        "dependencies": dependencies(),
        "runs": records,
        "policy": {
            "training": False,
            "test_selection": False,
            "split": "test",
            "assignments": "saved calibration assignments for states, code MCC and parameter graph",
            "retention": "established calibration split-half fitting and family selection",
            "curves": "recorded validation only; test reserved for final endpoints",
            "representative_seed": 0,
            "comparisons": ["full - baseline"],
            "paired_test": "exact seed-paired sign flips, one- and two-sided",
            "decision": "report every fixed endpoint; no retraining or test-selected changes",
            "holdout_provenance": (
                "same retained test cohort previously used for the main experiment"
            ),
        },
    }
    protocol["identity"] = canonical_fingerprint(protocol)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    return protocol


def evaluate_seed(protocol_path: Path, seed: int, device_name: str = "cuda:0") -> dict[str, Any]:
    """Score one pair of declared endpoints, reproducing validation forecasts first."""
    protocol = read_json(protocol_path)
    identity = canonical_fingerprint({k: v for k, v in protocol.items() if k != "identity"})
    if protocol["schema"] != SCHEMA or protocol["identity"] != identity:
        raise ValueError("test declaration changed")
    if protocol["dependencies"] != dependencies() or seed not in run.SEEDS:
        raise ValueError("test evaluator or declared seed differs")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    data_dir = Path(protocol["data_dir"])
    if sha(data_dir / "completion.json") != protocol["cache_completion_sha256"]:
        raise ValueError("test data cache changed")
    data, cache = load_data(data_dir, include_test=True)
    if (
        cache["identity"] != protocol["data_identity"]
        or len(data["test"]) != protocol["test_groups"]
    ):
        raise ValueError("test data differ from the declaration")
    output = protocol_path.parent / f"seed-{seed}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    logger = WandbLogger(
        WandbConfig(
            mode="online",
            project="sparse_causal_jepa",
            entity="jesse-hoekstra-university-of-oxford",
            group=protocol_path.parent.name,
            name=f"frozen-code-test-s{seed}",
            job_type="test-evaluation",
            tags=["frozen-code-control", "test", "evaluation-only"],
        ),
        resolved_config={"protocol_identity": identity, "seed": seed},
        dataset_identity=protocol["data_identity"],
        out_dir=output,
    )
    reports: list[dict[str, Any]] = []
    try:
        for frozen in [r for r in protocol["runs"] if r["seed"] == seed]:
            directory = Path(frozen["directory"])
            completion = control.verify_completion(directory)
            if (
                sha(directory / "completion.json") != frozen["completion_sha256"]
                or completion["hashes"]["last.pt"] != frozen["checkpoint_sha256"]
                or completion["hashes"]["calibration.pt"] != frozen["calibration_sha256"]
            ):
                raise ValueError("declared endpoint changed")
            checkpoint = load_payload(directory / "last.pt")
            if checkpoint["identity"] != frozen["identity"] or checkpoint["step"] != control.STEPS:
                raise ValueError("checkpoint and declaration disagree")
            model = restore_model(checkpoint, frozen["arm"], seed).to(device)
            config = read_json(directory / "config.json")
            objective = TrainingObjective(
                objective_config(frozen["arm"], 0.0), GECOConfig(**config["geco"])
            ).to(device)
            objective.load_state_dict(checkpoint["objective"], strict=True)
            objective.requires_grad_(False).eval()
            saved = load_payload(directory / "calibration.pt")
            calibration = EvaluationResult(saved["metrics"], saved["samples"])
            reproduced = run.evaluate(
                model, data["validation"], frozen["batch_size"], device, objective
            )
            expected = read_json(directory / "validation.json")["metrics"]
            reproduction = {
                key: {"saved": expected[key], "reproduced": reproduced.metrics[key]}
                for key in ("prediction", "forecast/normalized_mse", "forecast/code_use_ratio")
            }
            if any(
                not math.isclose(value["saved"], value["reproduced"], rel_tol=1e-5, abs_tol=1e-7)
                for value in reproduction.values()
            ):
                raise ValueError("saved validation forecasts did not reproduce")
            before = run._state_identity(model)  # pyright: ignore[reportPrivateUsage]
            held = run.evaluate(model, data["test"], frozen["batch_size"], device, objective)
            metrics, graphs = frozen_test.score_result(model, calibration, held, frozen)
            metrics["code_mcc"] = parameter_mcc(held, frozen["code_mcc_columns"])
            if run._state_identity(model) != before:  # pyright: ignore[reportPrivateUsage]
                raise RuntimeError("evaluation modified the model")
            arm_dir = output / frozen["arm"]
            arm_dir.mkdir()
            save_torch(arm_dir / "test.pt", {"metrics": metrics, "samples": held.samples})
            report = {
                "schema": SCHEMA,
                "protocol_identity": identity,
                "split": "test",
                "groups": len(data["test"]),
                "seed": seed,
                "arm": frozen["arm"],
                "steps": control.STEPS,
                "source": frozen,
                "metrics": metrics,
                "graphs": graphs,
                "validation_reproduction": reproduction,
                "model_unchanged": True,
                "physical_samples_sha256": tensor_fingerprint(held.samples["states"]),
            }
            write_json(arm_dir / "report.json", report)
            write_json(
                arm_dir / "completion.json",
                {
                    "protocol_identity": identity,
                    "hashes": {name: sha(arm_dir / name) for name in ("test.pt", "report.json")},
                },
            )
            if sha(directory / "completion.json") != frozen["completion_sha256"]:
                raise RuntimeError("training endpoint changed during scoring")
            logger.log(len(reports), {f"test/{frozen['arm']}/{k}": v for k, v in metrics.items()})
            reports.append(report)
        first = load_payload(output / "baseline" / "test.pt")["samples"]["codes"]
        second = load_payload(output / "full" / "test.pt")["samples"]["codes"]
        if not torch.equal(first, second):
            raise ValueError("paired arms received different frozen test codes")
        summary = {
            "protocol_identity": identity,
            "seed": seed,
            "paired_codes_identical": True,
            "wandb": logger.metadata,
            "runs": reports,
        }
        write_json(output / "seed.json", summary)
        logger.finish()
        return summary
    except BaseException:
        logger.finish(exit_code=1)
        raise


def main() -> None:
    """Declare all ten test endpoints, or score one predeclared seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("declare")
    declaration.add_argument("--batch-dir", type=Path, required=True)
    declaration.add_argument("--data-dir", type=Path, required=True)
    declaration.add_argument("--output-dir", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "declare":
        result = declare(args.batch_dir, args.data_dir, args.output_dir)
        print(json.dumps({"identity": result["identity"], "runs": len(result["runs"])}))
    else:
        result = evaluate_seed(args.protocol, args.seed, args.device)
        print(
            json.dumps(
                {"seed": args.seed, "paired_codes_identical": result["paired_codes_identical"]}
            )
        )


if __name__ == "__main__":
    main()
