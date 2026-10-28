"""Audit fixed endpoints and compare charts using validation groups only."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.chart_sparsity.core import BLOCKS, Chart, chart_matrix
from diagnostics.chart_sparsity.run import (
    SCHEMA,
    calibration_rule,
    read_json,
    save_torch,
    sha,
    verify_completion,
)
from diagnostics.oracle_state_predictor.run import write_json
from scjepa.data.datasets import canonical_fingerprint

Policy = Literal["baseline", "full"]
ArmKey = tuple[int, Chart, Policy]
STEPS = 100_000
MSE_BOUND = 0.02
MATCH_MARGIN = 0.002
REQUIRED_ARTIFACTS = {
    "config.json",
    "wandb.json",
    "startup.json",
    "history.jsonl",
    "last.pt",
    "validation.json",
    "validation.pt",
    "test.json",
    "test.pt",
}


@dataclass(frozen=True)
class Arm:
    """One authenticated fixed sparse endpoint and its validation measurements."""

    directory: Path
    seed: int
    chart: Chart
    policy: Policy
    completion: dict[str, Any]
    record: dict[str, Any]
    metrics: dict[str, float]
    samples: dict[str, Tensor]


def _same(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs")


def _finite_float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def group_measures(samples: dict[str, Tensor], logit_weight: float) -> dict[str, Tensor]:
    """Reduce raw group/horizon statistics without treating episodes as independent."""
    mse = samples["group_horizon_mse"].mean(dim=1)
    logit = samples["group_horizon_logit"].mean(dim=1)
    measures = {"prediction_mse": mse, "logit": logit, "base_loss": mse + logit_weight * logit}
    for block in BLOCKS:
        measures[f"{block}_paths"] = samples[f"group_{block}_path_means"].mean(dim=1)
        measures[f"{block}_reachable_pairs"] = (
            samples[f"group_{block}_reachability"].sum(dim=(2, 3)).mean(dim=1)
        )
    measures["common_score"] = sum(
        (measures[f"{block}_paths"] for block in BLOCKS), torch.zeros_like(mse)
    )
    measures["baseline_objective_paths"] = measures["state_paths"] + measures["action_paths"]
    return measures


def _audit_samples(
    samples: dict[str, Tensor], metrics: dict[str, float], logit_weight: float
) -> None:
    shapes = {
        "predictions": (256, 10, 2, 6),
        "targets": (256, 10, 2, 6),
        "group_ids": (256,),
        "group_horizon_mse": (256, 2),
        "group_horizon_logit": (256, 2),
    }
    for block in BLOCKS:
        shapes[f"group_{block}_path_means"] = (256, 2)
        shapes[f"group_{block}_path_sums"] = (256, 2)
        shapes[f"group_{block}_reachability"] = (256, 2, 6, 16 if block == "action" else 6)
    for name, shape in shapes.items():
        value = samples.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"validation sample {name} has the wrong shape")
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"validation sample {name} must be finite and on CPU")
    if len({int(value) for value in samples["group_ids"]}) != 256:
        raise ValueError("validation parameter-group identities must be unique")
    expected_mse = (
        (samples["predictions"].double() - samples["targets"].double()).square().mean(dim=(1, 3))
    )
    if not torch.allclose(expected_mse, samples["group_horizon_mse"], rtol=1e-10, atol=1e-12):
        raise ValueError("saved group errors disagree with raw forecasts")
    measures = group_measures(samples, logit_weight)
    for name in ("prediction_mse", "logit", "base_loss"):
        if not math.isclose(
            metrics[name], float(measures[name].mean()), rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError(f"aggregate metric {name} disagrees with saved groups")
    variance = float(samples["targets"].double().flatten(0, 2).var(dim=0, correction=0).mean())
    if variance <= 0 or not math.isclose(metrics["target_variance"], variance, rel_tol=1e-8):
        raise ValueError("target variance disagrees with raw targets")
    for block in BLOCKS:
        means = samples[f"group_{block}_path_means"]
        if bool((means < 0).any()):
            raise ValueError(f"{block} path counts must be nonnegative")
        expected_sums = means * (96 if block == "action" else 36)
        if not torch.equal(expected_sums, samples[f"group_{block}_path_sums"]):
            raise ValueError(f"{block} path sums disagree with their block means")
        if not math.isclose(
            metrics[f"{block}_sparsity"], float(means.mean()), rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError(f"{block} aggregate path penalty disagrees with groups")
        reach = samples[f"group_{block}_reachability"]
        if bool(((reach < 0) | (reach > 1)).any()):
            raise ValueError(f"{block} reachability frequency is outside [0,1]")
        if bool((reach.sum(dim=(2, 3)) > expected_sums + 1e-8).any()):
            raise ValueError(f"{block} reachable pairs exceed total path counts")
        frequency = reach.mean(dim=(0, 1))
        _same(metrics[f"{block}_edges"], float((frequency > 0).sum()), f"{block} union count")


def load_arms(directories: Sequence[Path]) -> dict[ArmKey, Arm]:
    """Authenticate eight scientific endpoints and their paired data/initialization."""
    if len(directories) != 8 or len({path.resolve() for path in directories}) != 8:
        raise ValueError("require eight distinct sparse endpoint directories")
    arms: dict[ArmKey, Arm] = {}
    for directory in directories:
        completion = verify_completion(directory, kind="train")
        if not REQUIRED_ARTIFACTS.issubset(completion["hashes"]):
            raise ValueError("completion omits required endpoint artifacts")
        record = read_json(directory / "config.json")
        scientific = cast(dict[str, Any], record["scientific_config"])
        seed, chart, policy = scientific["seed"], scientific["chart"], scientific["condition"]
        if (
            seed not in (0, 1)
            or chart not in ("physical", "mixed")
            or policy
            not in (
                "baseline",
                "full",
            )
        ):
            raise ValueError("unexpected seed, chart or sparse policy")
        key: ArmKey = seed, chart, policy
        if key in arms:
            raise ValueError("duplicate seed/chart/policy endpoint")
        for name, expected in (
            ("profile", "scientific"),
            ("steps", STEPS),
            ("successful_updates", STEPS),
            ("skipped_updates", 0),
            ("seed", seed),
            ("chart", chart),
            ("condition", policy),
        ):
            _same(completion.get(name), expected, f"completion {name}")
        _same(scientific["steps"], STEPS, "configured budget")
        _same(scientific["profile"], "scientific", "configured profile")
        _same(record["identity"], completion["identity"], "run identity")
        checkpoint = cast(
            dict[str, Any],
            torch.load(directory / "last.pt", map_location="cpu", weights_only=False),
        )
        for name, expected in (
            ("schema", SCHEMA),
            ("step", STEPS),
            ("identity", record["identity"]),
        ):
            _same(checkpoint.get(name), expected, f"final checkpoint {name}")
        states = cast(dict[object, dict[str, Any]], checkpoint["optimizer"]["state"])
        if not states or any(int(value["step"]) != STEPS for value in states.values()):
            raise ValueError("Adam updates disagree with the fixed endpoint")
        _same(int(checkpoint["controller"]["num_updates"]), STEPS, "GECO update count")
        specification = {
            name: record[name]
            for name in (
                "scientific_config",
                "data_identity",
                "data_sha256",
                "code_identity",
                "dense_source",
                "calibration",
                "initial_state_identity",
            )
        }
        _same(canonical_fingerprint(specification), record["identity"], "scientific identity")
        for name in ("data_identity", "code_identity", "initial_state_identity"):
            _same(record[name], completion[name], f"record {name}")
        dense = record["dense_source"]
        for name, expected in (("seed", seed), ("chart", chart), ("steps", STEPS)):
            _same(dense[name], expected, f"dense parent {name}")
        calibration = record["calibration"]
        _same(calibration["admitted"], True, "dense admission")
        _same(calibration["prediction_mse_bound"], MSE_BOUND, "prediction-quality bound")
        if _finite_float(calibration["tau"], "GECO bound") <= 0:
            raise ValueError("GECO bound must be positive")
        sources = cast(list[dict[str, Any]], calibration["dense_sources"])
        _same(
            {(row["chart"], row["seed"]) for row in sources},
            {
                (chart_name, seed_index)
                for chart_name in ("physical", "mixed")
                for seed_index in (0, 1)
            },
            "four dense calibration endpoints",
        )
        if len(sources) != 4 or any(row["steps"] != STEPS for row in sources):
            raise ValueError("calibration must pin four fixed100k dense endpoints")
        recomputed = calibration_rule(sources, MSE_BOUND)
        for name in ("admitted", "tau", "prediction_mse_bound", "failures"):
            _same(calibration[name], recomputed[name], f"recomputed calibration {name}")
        if not any(
            all(
                source[name] == dense[name]
                for name in ("identity", "checkpoint_sha256", "completion_sha256")
            )
            for source in calibration["dense_sources"]
        ):
            raise ValueError("dense parent is not pinned by calibration")
        payload = cast(
            dict[str, dict[str, Tensor]],
            torch.load(directory / "validation.pt", map_location="cpu", weights_only=True),
        )
        samples = payload["samples"]
        metrics = cast(dict[str, float], read_json(directory / "validation.json")["metrics"])
        _audit_samples(samples, metrics, float(scientific["logit_weight"]))
        arms[key] = Arm(directory, seed, chart, policy, completion, record, metrics, samples)
    reference = next(iter(arms.values()))
    shared_science = {
        name: value
        for name, value in reference.record["scientific_config"].items()
        if name not in ("seed", "chart", "condition")
    }
    for arm in arms.values():
        for name in ("data_identity", "data_sha256", "code_identity", "calibration"):
            _same(arm.record[name], reference.record[name], f"shared {name}")
        _same(
            {
                k: v
                for k, v in arm.record["scientific_config"].items()
                if k not in ("seed", "chart", "condition")
            },
            shared_science,
            "shared scientific settings",
        )
        if not torch.equal(arm.samples["group_ids"], reference.samples["group_ids"]):
            raise ValueError("validation groups are not paired")
    for seed in (0, 1):
        for chart in ("physical", "mixed"):
            baseline, full = arms[seed, chart, "baseline"], arms[seed, chart, "full"]
            for name in ("initial_state_identity", "dense_source"):
                _same(baseline.record[name], full.record[name], f"paired {name}")
            if not torch.equal(baseline.samples["targets"], full.samples["targets"]):
                raise ValueError("policy targets differ within one chart")
        physical, mixed = arms[seed, "physical", "baseline"], arms[seed, "mixed", "baseline"]
        expected = (physical.samples["targets"].double() @ chart_matrix("mixed").T).to(
            mixed.samples["targets"].dtype
        )
        if not torch.equal(expected, mixed.samples["targets"]):
            raise ValueError("mixed validation targets are not the declared fixed chart")
    return arms


def interval(values: Tensor, indices: Tensor) -> dict[str, object]:
    """Report a paired group-mean estimate and percentile bootstrap interval."""
    values = values.double()
    draws = values[indices].mean(dim=1)
    bounds = torch.quantile(draws, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "mean": float(values.mean()),
        "interval_95": [float(value) for value in bounds],
    }


def compare(arms: dict[ArmKey, Arm], indices: Tensor) -> dict[str, Any]:
    """Apply the predeclared common-score, parameter-gap and fit-quality checks."""
    seed_results: list[dict[str, Any]] = []
    all_quality, all_preference = True, True
    for seed in (0, 1):
        differences: dict[str, dict[str, Tensor]] = {}
        policies: dict[str, Any] = {}
        seed_quality = True
        for policy in ("baseline", "full"):
            physical, mixed = arms[seed, "physical", policy], arms[seed, "mixed", policy]
            weight = float(physical.record["scientific_config"]["logit_weight"])
            a, b = group_measures(physical.samples, weight), group_measures(mixed.samples, weight)
            differences[policy] = {name: b[name] - a[name] for name in a}
            rows = {name: interval(value, indices) for name, value in differences[policy].items()}
            optimized = "common_score" if policy == "full" else "baseline_objective_paths"
            mse_bounds = cast(list[float], rows["prediction_mse"]["interval_95"])
            matched = mse_bounds[0] >= -MATCH_MARGIN and mse_bounds[1] <= MATCH_MARGIN
            endpoint_quality = all(
                arm.metrics["prediction_mse"] <= MSE_BOUND
                and arm.metrics["base_loss"] <= float(arm.record["calibration"]["tau"])
                for arm in (physical, mixed)
            )
            seed_quality &= matched and endpoint_quality
            policies[policy] = {
                "mixed_minus_physical": rows,
                "optimized_path_score": optimized,
                "prediction_difference_within_margin": matched,
                "both_endpoints_meet_quality_bounds": endpoint_quality,
                "endpoints": {arm.chart: arm.metrics for arm in (physical, mixed)},
            }
        did_values = differences["full"]["common_score"] - differences["baseline"]["common_score"]
        did = interval(did_values, indices)
        full_gap = policies["full"]["mixed_minus_physical"]
        preference = all(
            cast(list[float], row["interval_95"])[0] > 0
            for row in (full_gap["common_score"], full_gap["parameter_paths"], did)
        )
        all_quality &= seed_quality
        all_preference &= preference
        seed_results.append(
            {
                "seed": seed,
                "policies": policies,
                "common_score_difference_in_differences": did,
                "quality_pass": seed_quality,
                "preference_pass": preference,
            }
        )
    decision = (
        "inconclusive_prediction_quality"
        if not all_quality
        else "positive_exploratory_preference"
        if all_preference
        else "bounded_preference_not_established"
    )
    return {
        "seeds": seed_results,
        "all_quality_pass": all_quality,
        "all_preference_pass": all_preference,
        "decision": decision,
    }


def analyze(
    directories: Sequence[Path],
    output: Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 2026102801,
) -> dict[str, Any]:
    """Write a new read-only-artifact analysis without fitting or test selection."""
    if bootstrap_samples < 1000 or seed < 0:
        raise ValueError("require at least1000 bootstrap draws and a nonnegative seed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty: {output}")
    arms = load_arms(directories)
    indices = torch.randint(
        256, (bootstrap_samples, 256), generator=torch.Generator().manual_seed(seed)
    )
    result = compare(arms, indices)
    sources = [
        {
            "directory": str(arm.directory.resolve()),
            "identity": arm.record["identity"],
            "completion_sha256": sha(arm.directory / "completion.json"),
            "verified_artifacts": len(arm.completion["hashes"]),
            "seed": arm.seed,
            "chart": arm.chart,
            "condition": arm.policy,
        }
        for arm in arms.values()
    ]
    result |= {
        "schema": "fixed-chart-sparsity-analysis-v1",
        "analysis_source_hashes": {
            name: sha(Path(__file__).with_name(name))
            for name in ("analysis.py", "core.py", "run.py")
        },
        "sources": sources,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "unit": "parameter_group",
            "shared_indices": True,
        },
        "quality_rule": {
            "validation_mse_max": MSE_BOUND,
            "paired_mse_margin": MATCH_MARGIN,
            "base_loss_max": "shared calibration tau",
        },
        "common_score": (
            "state_path_mean + action_path_mean + parameter_path_mean, for every policy"
        ),
        "score_scope": (
            "Evaluation-only common score. Baseline optimized state+action; full optimized "
            "all three. Gate policies differ as declared. All component contrasts remain reported."
        ),
        "decision_scope": (
            "Exploratory conditional group-bootstrap intervals at two predeclared training seeds; "
            "no familywise adjustment, population optimizer claim, learned state MCC claim, "
            "or theorem certification."
        ),
        "test_descriptive_only": [
            {
                "seed": arm.seed,
                "chart": arm.chart,
                "condition": arm.policy,
                "metrics": read_json(arm.directory / "test.json")["metrics"],
            }
            for arm in arms.values()
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    save_torch(output / "bootstrap_indices.pt", indices)
    write_json(output / "results.json", result)
    write_json(
        output / "completion.json",
        {
            "schema": result["schema"],
            "hashes": {
                name: sha(output / name) for name in ("results.json", "bootstrap_indices.pt")
            },
        },
    )
    return result


def main() -> None:
    """Analyze exactly eight completed scientific sparse arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", default=1000, type=int)
    parser.add_argument("--seed", default=2026102801, type=int)
    args = parser.parse_args()
    result = analyze(
        args.run_dir, args.output_dir, bootstrap_samples=args.bootstrap, seed=args.seed
    )
    print(result["decision"], flush=True)


if __name__ == "__main__":
    main()
