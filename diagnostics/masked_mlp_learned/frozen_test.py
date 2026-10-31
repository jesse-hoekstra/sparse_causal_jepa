"""Explicit reserved-test evaluation of predeclared, immutable learned-code endpoints.

``declare`` authenticates every endpoint and freezes graph alignment without
loading test tensors. ``evaluate`` scores one seed using saved calibration
representations. ``summarize`` collects all seeds without selecting any model.
This module is separate from the numerical training identity.
"""

import argparse
import csv
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import physical_response, run
from diagnostics.masked_mlp_oracle import paper_analysis as analysis
from diagnostics.masked_mlp_oracle import run as oracle
from diagnostics.masked_mlp_oracle.model import MaskedTrunkPredictor
from diagnostics.shared.data import StateData, load_data
from diagnostics.shared.io import read_json, save_torch, sha
from diagnostics.shared.io import verify_completion as verify_artifacts
from diagnostics.shared.oracle import EvaluationResult, objective_config
from diagnostics.shared.recovery import recovery_metrics
from diagnostics.shared.source_geometry import graphs as true_graphs
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.data.observations import VectorObservationMap
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA
from scjepa.training.lagrangian import GECOConfig
from scjepa.training.logging import WandbConfig, WandbLogger
from scjepa.training.loop import TrainingObjective

SCHEMA = "masked-mlp-frozen-test-v1"


def _load(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))


def _dependencies() -> dict[str, str]:
    return {
        "training_code_identity": run.code_identity(),
        "test_evaluator_sha256": sha(Path(__file__)),
        "physical_response_sha256": sha(Path(physical_response.__file__)),
        "paper_analysis_sha256": sha(Path(analysis.__file__)),
    }


def declare(
    batch: Path,
    data_dir: Path,
    output: Path,
    *,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    dense_steps: int = 50_000,
    sparse_steps: int = 300_000,
) -> dict[str, Any]:
    """Freeze endpoints and existing validation graph orders before opening test tensors."""
    if output.exists():
        raise FileExistsError(output)
    cache = verify_artifacts(data_dir)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for arm in analysis.ARMS:
            directory = (batch / f"{arm}-s{seed}" / "results").resolve()
            completion = run.verify_completion(directory, kind="train", require_current_source=True)
            steps = dense_steps if arm == "dense" else sparse_steps
            if (
                completion["steps"] != steps
                or completion["successful_updates"] != steps
                or completion["skipped_updates"] != 0
                or completion["condition"] != arm
                or completion["seed"] != seed
                or completion["code_dim"] != 6
                or completion["data_identity"] != cache["identity"]
            ):
                raise ValueError("Endpoint differs from the declared fixed-budget comparison")
            samples = _load(directory / "validation.pt")["samples"]
            codes = cast(Tensor, samples["codes"]).double()
            theta = cast(Tensor, samples["theta"]).double()[:, None].expand_as(codes)
            order = analysis._code_order(  # pyright: ignore[reportPrivateUsage]
                codes.flatten(0, 1), theta.flatten(0, 1)
            )
            config = read_json(directory / "config.json")
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "directory": str(directory),
                    "steps": steps,
                    "identity": completion["identity"],
                    "completion_sha256": sha(directory / "completion.json"),
                    "checkpoint_sha256": completion["hashes"]["last.pt"],
                    "calibration_sha256": completion["hashes"]["calibration.pt"],
                    "validation_sha256": completion["hashes"]["validation.pt"],
                    "matching": read_json(directory / "matching.json"),
                    "parameter_columns": order,
                    "parameter_alignment_fit_split": "validation",
                    "batch_size": config["config"]["batch_size"],
                    "tau_prediction": completion["tau_prediction"],
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
        "seeds": list(seeds),
        "split": "test",
        "dependencies": _dependencies(),
        "runs": records,
        "policy": {
            "training": False,
            "test_selection": False,
            "calibration": "Reuse saved calibration encodings; fit/select probes only there",
            "graph_alignment": "Saved calibration state match and validation-frozen code order",
            "forecast": "Unchanged two-step evaluator, forward seed 42, saved group batch size",
            "comparisons": ["full - baseline", "baseline - dense"],
            "paired_test": "Exact seed-level sign flips; report one- and two-sided p-values",
            "group_bootstrap": {"draws": 1000, "seed": 2026103001},
            "physical_response": {
                "donor": 0,
                "recipients": [5, 6, 7, 8, 9],
                "draws": 1000,
                "seed": physical_response.BOOTSTRAP_SEED,
            },
        },
    }
    protocol["identity"] = canonical_fingerprint(protocol)
    output.mkdir(parents=True)
    write_json(output / "protocol.json", protocol)
    return protocol


def score_result(
    model: SCJEPA,
    calibration: EvaluationResult,
    result: EvaluationResult,
    frozen: dict[str, Any],
) -> tuple[dict[str, float], dict[str, list[list[bool]]]]:
    """Score held-out samples with calibration probes and predeclared graph assignments."""
    metrics, matching = recovery_metrics(calibration, result)
    if matching != frozen["matching"]:
        raise ValueError("Calibration state assignment changed")
    metrics |= oracle.block_purity(calibration, result)
    metrics |= run.code_metrics(calibration, result)
    predictor = cast(MaskedTrunkPredictor, model.predictor)
    metrics |= oracle.graph_metrics(predictor, matching["online_columns"])
    columns = torch.tensor(matching["online_columns"])
    code_columns = torch.tensor(frozen["parameter_columns"])
    learned = predictor.thresholded_graphs()
    graphs = {
        "state": learned["state"].cpu()[columns][:, columns],
        "action": learned["action"].cpu()[columns],
        "parameter": learned["parameter"].cpu()[columns][:, code_columns],
    }
    truth = true_graphs().parameter
    graph = graphs["parameter"]
    metrics.update(
        {
            analysis.CODE_ALIGNED_SHD: float((graph != truth).sum()),
            "masks/parameter_code_aligned_missing_true_edges": float((~graph & truth).sum()),
            "masks/parameter_code_aligned_false_edges": float((graph & ~truth).sum()),
            "forecast_quality_pass": float(
                metrics["normalized_prediction_mse"] <= run.NORMALIZED_MSE_CAP
            ),
        }
    )
    if frozen["tau_prediction"] is not None:
        tau = float(frozen["tau_prediction"])
        relative = result.metrics["prediction"] / tau
        metrics.update(
            {
                "geco/tau_prediction": tau,
                "geco/relative_prediction": relative,
                "geco/feasible": float(relative <= 1),
            }
        )
    return result.metrics | metrics, {
        name: [[bool(graph[i, j]) for j in range(graph.shape[1])] for i in range(6)]
        for name, graph in graphs.items()
    }


def _evaluate_run(
    frozen: dict[str, Any],
    data: dict[str, StateData],
    observation: VectorObservationMap,
    output: Path,
    device: torch.device,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    directory = Path(frozen["directory"])
    before = sha(directory / "completion.json")
    if before != frozen["completion_sha256"]:
        raise ValueError("Declared endpoint completion changed")
    completion = run.verify_completion(directory, kind="train", require_current_source=True)
    checkpoint = _load(directory / "last.pt")
    if (
        checkpoint["identity"] != frozen["identity"]
        or checkpoint["step"] != frozen["steps"]
        or completion["hashes"]["last.pt"] != frozen["checkpoint_sha256"]
        or completion["hashes"]["calibration.pt"] != frozen["calibration_sha256"]
    ):
        raise ValueError("Checkpoint or calibration differs from the declaration")
    condition = cast(run.Condition, frozen["arm"])
    model = run.build_control(condition, frozen["seed"], 6)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).requires_grad_(False).eval()
    config = read_json(directory / "config.json")
    objective = TrainingObjective(
        objective_config(condition, 0.0), GECOConfig(**config["geco"])
    ).to(device)
    objective.load_state_dict(checkpoint["objective"], strict=True)
    objective.requires_grad_(False).eval()
    calibration_payload = _load(directory / "calibration.pt")
    calibration = EvaluationResult(calibration_payload["metrics"], calibration_payload["samples"])
    reproduced = run.evaluate(model, data["validation"], frozen["batch_size"], device, objective)
    expected = read_json(directory / "validation.json")["metrics"]
    reproduction = {
        name: {"saved": expected[name], "reproduced": reproduced.metrics[name]}
        for name in ("prediction", "forecast/normalized_mse", "forecast/code_use_ratio")
    }
    if any(
        not math.isclose(values["saved"], values["reproduced"], rel_tol=1e-5, abs_tol=1e-7)
        for values in reproduction.values()
    ):
        raise ValueError(f"Saved validation forecast did not reproduce: {reproduction}")
    model_before = run._state_identity(model)  # pyright: ignore[reportPrivateUsage]
    held = run.evaluate(model, data["test"], frozen["batch_size"], device, objective)
    metrics, graphs = score_result(model, calibration, held, frozen)
    if run._state_identity(model) != model_before:  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("Evaluation modified model weights")
    output.mkdir(parents=True)
    save_torch(output / "test.pt", {"metrics": metrics, "samples": held.samples})
    moments = analysis.matched_group_moments(
        held.samples["online_states"], held.samples["states"], frozen["matching"]["online_columns"]
    )
    model.cpu().double()
    response, raw = physical_response.assess(
        model, data["test"], observation, expected_split="test"
    )
    save_torch(output / "responses.pt", raw)
    write_json(output / "physical_response.json", response)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_identity": protocol["identity"],
        "split": "test",
        "groups": len(data["test"]),
        "arm": condition,
        "seed": frozen["seed"],
        "steps": frozen["steps"],
        "source": frozen,
        "data_identity": protocol["data_identity"],
        "physical_samples_sha256": tensor_fingerprint(held.samples["states"]),
        "validation_reproduction": reproduction,
        "metrics": metrics,
        "graphs": graphs,
        "group_moments": moments.tolist(),
        "runtime": {"torch": str(torch.__version__), "device": str(device)},
        "physical_response": response,
    }
    write_json(output / "report.json", report)
    if sha(directory / "completion.json") != before:
        raise RuntimeError("Source completion changed during evaluation")
    run.verify_completion(directory, kind="train", require_current_source=True)
    write_json(
        output / "completion.json",
        {
            "protocol_identity": protocol["identity"],
            "hashes": {
                name: sha(output / name)
                for name in ("report.json", "physical_response.json", "test.pt", "responses.pt")
            },
        },
    )
    return report


def evaluate_seed(
    protocol_path: Path,
    seed: int,
    *,
    device: str = "cuda",
    wandb_mode: str = "online",
    wandb_entity: str | None = None,
) -> None:
    """Evaluate every declared arm for one seed, with no optimizer or checkpoint choice."""
    torch.set_num_threads(1)
    protocol = read_json(protocol_path)
    if protocol["identity"] != canonical_fingerprint(
        {key: value for key, value in protocol.items() if key != "identity"}
    ):
        raise ValueError("Test declaration changed")
    if protocol["dependencies"] != _dependencies() or seed not in protocol["seeds"]:
        raise ValueError("Evaluation source or seed differs from the declaration")
    output = protocol_path.parent / f"seed{seed}"
    if output.exists():
        raise FileExistsError(output)
    data_dir = Path(protocol["data_dir"])
    if sha(data_dir / "completion.json") != protocol["cache_completion_sha256"]:
        raise ValueError("Declared cache changed")
    data, cache = load_data(data_dir, include_test=True)
    if cache["identity"] != protocol["data_identity"] or (
        len(data["test"]) != protocol["test_groups"]
    ):
        raise ValueError("Test cohort differs from the declaration")
    observation = VectorObservationMap(
        matrix=_load(data_dir / "observation_map.pt")["matrix"], dimension=6
    )
    output.mkdir()
    write_json(output / "started.json", {"utc": datetime.now(UTC).isoformat(), "seed": seed})
    logger = WandbLogger(
        WandbConfig(
            mode=wandb_mode,
            entity=wandb_entity,
            project="sparse_causal_jepa",
            group=f"frozen-test-{protocol['identity'][:12]}",
            name=f"frozen-test-s{seed}",
            job_type="reserved-test-evaluation",
            tags=["test", "frozen-checkpoint", "no-training"],
        ),
        resolved_config=protocol,
        dataset_identity=cache["identity"],
        out_dir=output,
    )
    write_json(output / "wandb.json", logger.metadata)
    try:
        for step, frozen in enumerate(r for r in protocol["runs"] if r["seed"] == seed):
            arm = frozen["arm"]
            report = _evaluate_run(
                frozen, data, observation, output / arm, torch.device(device), protocol
            )
            logger.log(step, {f"{arm}/test/{k}": v for k, v in report["metrics"].items()})
            logger.log_artifact(output / arm / "report.json", name=f"test-{arm}-s{seed}")
            print(f"Completed test {arm} seed {seed}", flush=True)
        if protocol["dependencies"] != _dependencies():
            raise RuntimeError("Evaluation source changed during assessment")
        logger.finish()
        write_json(
            output / "complete.json",
            {
                "protocol_identity": protocol["identity"],
                "finished_utc": datetime.now(UTC).isoformat(),
            },
        )
    except BaseException:
        logger.finish(exit_code=1)
        raise


def summarize(protocol_path: Path) -> dict[str, Any]:
    """Report every declared endpoint and paired comparison, with no exclusion or selection."""
    protocol = read_json(protocol_path)
    root = protocol_path.parent
    reports: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in analysis.ARMS}
    rows: list[dict[str, Any]] = []
    for seed in protocol["seeds"]:
        for arm in analysis.ARMS:
            directory = root / f"seed{seed}" / arm
            completion = verify_artifacts(directory)
            report = read_json(directory / "report.json")
            if completion["protocol_identity"] != protocol["identity"] or (
                report["protocol_identity"] != protocol["identity"]
            ):
                raise ValueError("Test result does not belong to this declaration")
            reports[arm][seed] = report
            rows.append({"arm": arm, "seed": seed, **report["metrics"]})
    keys = set[str].intersection(
        *(set(cast(dict[str, float], row["metrics"])) for row in reports["full"].values())
    )
    summaries = {
        arm: {
            key: analysis.describe([report["metrics"][key] for report in found.values()])
            for key in sorted(keys)
            if all(key in report["metrics"] for report in found.values())
        }
        for arm, found in reports.items()
    }
    comparisons: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    for left, right in (("full", "baseline"), ("baseline", "dense")):
        name = f"{left} - {right}"
        comparisons[name] = analysis.paired(
            [reports[left][seed]["metrics"]["state_mcc"] for seed in protocol["seeds"]],
            [reports[right][seed]["metrics"]["state_mcc"] for seed in protocol["seeds"]],
        )
        groups[name] = {}
        for seed in protocol["seeds"]:
            a, b = reports[left][seed], reports[right][seed]
            if a["physical_samples_sha256"] != b["physical_samples_sha256"]:
                raise ValueError("Paired comparison uses different ordered test groups")
            groups[name][str(seed)] = analysis.paired_group_mcc(
                cast(analysis.Array, np.asarray(a["group_moments"], dtype=np.float64)),
                cast(analysis.Array, np.asarray(b["group_moments"], dtype=np.float64)),
            )
    result: dict[str, Any] = {
        "protocol_identity": protocol["identity"],
        "split": "test",
        "groups": protocol["test_groups"],
        "summary": summaries,
        "paired": comparisons,
        "paired_groups": groups,
        "physical_response": {
            arm: {str(seed): report["physical_response"] for seed, report in found.items()}
            for arm, found in reports.items()
        },
    }
    write_json(root / "summary.json", result)
    fields = {key for row in rows for key in row} - {"arm", "seed"}
    columns = ["arm", "seed", *sorted(fields)]
    with (root / "per_run.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    """Declare or execute an explicitly authorized reserved-test confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("declare")
    declaration.add_argument("--batch-dir", type=Path, required=True)
    declaration.add_argument("--data-dir", type=Path, required=True)
    declaration.add_argument("--output-dir", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--seed", type=int, required=True)
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluation.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online"), default="online"
    )
    evaluation.add_argument("--wandb-entity")
    collection = commands.add_parser("summarize")
    collection.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "declare":
        result = declare(args.batch_dir, args.data_dir, args.output_dir)
        print(f"Declared {len(result['runs'])} endpoints; test tensors remain unopened")
    elif args.command == "evaluate":
        evaluate_seed(
            args.protocol,
            args.seed,
            device=args.device,
            wandb_mode=args.wandb_mode,
            wandb_entity=args.wandb_entity,
        )
    else:
        summarize(args.protocol)
        print("Collected all declared test results")


if __name__ == "__main__":
    main()
