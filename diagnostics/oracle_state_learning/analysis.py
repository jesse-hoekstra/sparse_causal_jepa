"""Audit fixed oracle endpoints and quantify paired state-MCC differences."""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from scjepa.eval.recovery import CoordinateMatch, fit_coordinate_match, mean_correlation_coefficient

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 2026102803
QUALITY_CAP = 0.05


def read_json(path: Path) -> dict[str, Any]:
    """Read a diagnostic object without silently accepting a list payload."""
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], value)


def grouped_moments(features: Tensor, truth: Tensor, columns: Sequence[int]) -> Tensor:
    """Keep exact Pearson sufficient statistics for each independent parameter group."""
    if features.shape != truth.shape or features.ndim != 4 or features.shape[-1] != 6:
        raise ValueError("Expected aligned (group, view, time, 6) state tensors")
    CoordinateMatch(tuple(columns))
    if not bool(torch.isfinite(features).all() and torch.isfinite(truth).all()):
        raise ValueError("State samples must be finite")
    x = features.detach().cpu().double().flatten(1, 2)[..., list(columns)]
    y = truth.detach().cpu().double().flatten(1, 2)
    return torch.stack(
        (
            torch.full_like(x.sum(1), x.shape[1]),
            x.sum(1),
            y.sum(1),
            x.square().sum(1),
            y.square().sum(1),
            (x * y).sum(1),
        ),
        dim=1,
    )


def moment_mcc(moments: Tensor) -> Tensor:
    """Compute frozen-assignment MCC from summed statistics, including constant codes."""
    n, sx, sy, sxx, syy, sxy = moments.unbind(-2)
    numerator = sxy - sx * sy / n
    denominator = (
        (sxx - sx.square() / n).clamp_min(0) * (syy - sy.square() / n).clamp_min(0)
    ).sqrt()
    return (numerator.abs() / denominator.clamp_min(1e-15)).clamp(0, 1).mean(-1)


def paired_mcc_interval(
    left: Tensor,
    right: Tensor,
    truth: Tensor,
    left_columns: Sequence[int],
    right_columns: Sequence[int],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap parameter groups jointly, retaining calibration-frozen assignments."""
    if draws < 2 or len(truth) < 2:
        raise ValueError("Require at least two bootstrap draws and parameter groups")
    a = grouped_moments(left, truth, left_columns)
    b = grouped_moments(right, truth, right_columns)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(truth), (draws, len(truth)), generator=generator)
    weights = torch.zeros(draws, len(truth), dtype=torch.float64)
    weights.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    values = moment_mcc((weights @ a.flatten(1)).reshape(draws, 6, 6)) - moment_mcc(
        (weights @ b.flatten(1)).reshape(draws, 6, 6)
    )
    bounds = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "difference": float(moment_mcc(a.sum(0)) - moment_mcc(b.sum(0))),
        "lower_95": float(bounds[0]),
        "upper_95": float(bounds[1]),
        "draws": float(draws),
        "bootstrap_seed": float(seed),
    }


def audit_samples(payload: dict[str, Any], groups: int) -> dict[str, Tensor]:
    """Validate saved tensors and recompute latent prediction errors independently."""
    samples = cast(dict[str, Tensor], payload["samples"])
    shapes = {
        "online_states": (groups, 10, 12, 6),
        "target_states": (groups, 10, 12, 6),
        "states": (groups, 10, 12, 6),
        "predictions": (groups, 10, 2, 6),
        "targets": (groups, 10, 2, 6),
        "group_ids": (groups,),
    }
    for name, shape in shapes.items():
        value = samples.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"Malformed endpoint samples: {name}")
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Endpoint samples must be finite and on CPU: {name}")
    if len({int(value) for value in samples["group_ids"]}) != groups:
        raise ValueError("Repeated independent parameter-group ID")
    # Different GEMM batch shapes can produce rounding-level differences.
    if not torch.allclose(
        samples["targets"], samples["target_states"][:, :, 10:12], atol=1e-6, rtol=1e-6
    ):
        raise ValueError("Forecast targets differ from final EMA state coordinates")
    error = float((samples["predictions"].double() - samples["targets"].double()).square().mean())
    metrics = cast(dict[str, float], payload["metrics"])
    if not math.isclose(error, metrics["prediction"], abs_tol=1e-8, rel_tol=2e-6):
        raise ValueError("Reported forecast MSE disagrees with saved samples")
    return samples


def routing_metrics(samples: dict[str, Tensor], matching: dict[str, Any]) -> dict[str, float]:
    """Describe observed path unions; finite local routes do not certify causal support."""
    result: dict[str, float] = {}
    rows = list(matching["target_columns"])
    columns = list(matching["online_columns"])
    state_truth = torch.zeros(6, 6, dtype=torch.bool)
    state_truth[:3, :3] = True
    state_truth[3:, 3:] = True
    parameter_truth = torch.eye(6, dtype=torch.bool)
    parameter_truth[torch.arange(6), [3, 4, 5, 0, 1, 2]] = True
    truths = {
        "state": state_truth,
        "action": torch.ones(6, 16, dtype=torch.bool),
        "parameter": parameter_truth,
    }
    for name, truth in truths.items():
        if f"graph_{name}" not in samples:
            continue
        frequencies = samples[f"graph_{name}"].double().mean(0)
        if frequencies.shape != truth.shape or not bool(
            ((frequencies >= 0) & (frequencies <= 1)).all()
        ):
            raise ValueError("Malformed local routing frequencies")
        aligned = frequencies[rows]
        if name == "state":
            aligned = aligned[:, columns]
        support = aligned > 0
        result[f"routing/{name}_observed_union_edges"] = float(support.sum())
        result[f"routing/{name}_mean_local_edges"] = float(frequencies.sum())
        result[f"routing/{name}_aligned_union_shd"] = float((support ^ truth).sum())
    return result


def controller_history(path: Path) -> dict[str, float]:
    """Summarize logged accepted updates without treating samples as every update."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    logged = {int(row["step"]): row for row in rows if "geco/penalty_weight" in row}
    if not logged:
        return {}
    weights = [float(logged[step]["geco/penalty_weight"]) for step in sorted(logged)]
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Nonfinite or nonpositive logged GECO penalty weight")
    return {
        "logged_updates": float(len(weights)),
        "minimum_penalty_weight": min(weights),
        "maximum_penalty_weight": max(weights),
        "final_penalty_weight": weights[-1],
        "fraction_logged_at_penalty_floor": sum(
            value <= math.exp(-12) * (1 + 1e-5) for value in weights
        )
        / len(weights),
    }


def _load(directory: Path) -> dict[str, Any]:
    from diagnostics.oracle_state_learning.run import verify_completion

    complete = verify_completion(directory, require_current_source=False)
    config = read_json(directory / "config.json")
    scientific = config["scientific_config"]
    condition, seed = scientific["condition"], scientific["seed"]
    if condition not in {"dense", "baseline", "full", "mlp"} or seed not in (0, 1):
        raise ValueError("Unexpected condition or training seed")
    if complete["steps"] != 100_000 or complete["successful_updates"] != 100_000:
        raise ValueError("Only final fixed 100k endpoints may be interpreted")
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    if checkpoint["step"] != 100_000 or checkpoint["identity"] != complete["identity"]:
        raise ValueError("Final checkpoint budget or identity differs")
    if any(int(state["step"]) != 100_000 for state in checkpoint["optimizer"]["state"].values()):
        raise ValueError("Adam update counts do not match the fixed endpoint")
    if complete["profile"] != "scientific" or complete["skipped_updates"] != 0:
        raise ValueError("Endpoint must be scientific with no skipped updates")
    matching = read_json(directory / "matching.json")
    splits: dict[str, dict[str, Any]] = {}
    for split, groups in (("calibration", 512), ("validation", 256), ("test", 512)):
        if condition == "dense" and split == "test":
            if (directory / "test.pt").exists():
                raise ValueError("Dense references must not use test data")
            continue
        payload = cast(
            dict[str, Any],
            torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True),
        )
        samples = audit_samples(payload, groups)
        for encoder, key in (("online", "state_mcc"), ("target", "target_state_mcc")):
            columns = tuple(matching[f"{encoder}_columns"])
            features = samples[f"{encoder}_states"].flatten(0, 2)
            truth = samples["states"].flatten(0, 2)
            if split == "calibration":
                fitted = fit_coordinate_match(features, truth, method="pearson")
                if fitted.columns != columns:
                    raise ValueError("Saved assignment was not fitted on calibration")
            value = mean_correlation_coefficient(CoordinateMatch(columns), features, truth)
            if not math.isclose(value, payload["metrics"][key], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError("Reported state MCC disagrees with frozen calibration assignment")
        payload["metrics"].update(routing_metrics(samples, matching))
        splits[split] = payload
    return {
        "directory": str(directory),
        "completion": complete,
        "config": config,
        "condition": condition,
        "seed": seed,
        "matching": matching,
        "startup": read_json(directory / "startup.json"),
        "splits": splits,
    }


def analyze(directories: Sequence[Path], output: Path) -> dict[str, Any]:
    """Authenticate all eight runs and report bounded conclusions without selecting fits."""
    arms = [_load(directory) for directory in directories]
    indexed = {(arm["condition"], arm["seed"]): arm for arm in arms}
    expected = {
        (condition, seed) for condition in ("dense", "baseline", "full", "mlp") for seed in (0, 1)
    }
    if set(indexed) != expected or len(arms) != len(expected):
        raise ValueError("Require exactly the eight predeclared final endpoints")
    identities = {arm["completion"]["data_identity"] for arm in arms}
    sources = {arm["completion"]["code_identity"] for arm in arms}
    if len(identities) != 1 or len(sources) != 1:
        raise ValueError("Arms must share one source and cache identity")
    sparse = [arm for arm in arms if arm["condition"] in {"full", "baseline"}]
    tau = max(
        indexed[("dense", seed)]["splits"]["calibration"]["metrics"]["base"] for seed in (0, 1)
    )
    if any(arm["config"]["geco"]["tau"] != tau for arm in sparse):
        raise ValueError("Sparse arms do not share the declared maximum dense calibration bound")
    if len({arm["completion"]["calibration_identity"] for arm in sparse}) != 1:
        raise ValueError("Sparse arms do not share one frozen calibration artifact")
    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for arm in arms:
        results.append(
            {
                "condition": arm["condition"],
                "seed": arm["seed"],
                "directory": arm["directory"],
                "metrics": {split: payload["metrics"] for split, payload in arm["splits"].items()},
                "verified_artifact_count": len(arm["completion"]["hashes"]),
                "controller_history": controller_history(Path(arm["directory"]) / "history.jsonl"),
            }
        )
    for seed in (0, 1):
        paired = [indexed[(condition, seed)] for condition in ("dense", "baseline", "full", "mlp")]
        if len({arm["startup"]["encoder_identity"] for arm in paired}) != 1:
            raise ValueError("Within-seed initial state encoders differ")
        if len({arm["startup"]["predictor_identity"] for arm in paired[:3]}) != 1:
            raise ValueError("Within-seed initial SPARTAN predictors differ")
        for left, right, split in (
            ("full", "baseline", "test"),
            ("baseline", "mlp", "test"),
            ("baseline", "dense", "validation"),
        ):
            a, b = indexed[(left, seed)], indexed[(right, seed)]
            av, bv = a["splits"][split]["samples"], b["splits"][split]["samples"]
            if not torch.equal(av["group_ids"], bv["group_ids"]) or not torch.equal(
                av["states"], bv["states"]
            ):
                raise ValueError("Paired inference requires identical addressed evaluation groups")
            comparisons.append(
                {
                    "seed": seed,
                    "split": split,
                    "left": left,
                    "right": right,
                    **paired_mcc_interval(
                        av["online_states"],
                        bv["online_states"],
                        av["states"],
                        a["matching"]["online_columns"],
                        b["matching"]["online_columns"],
                    ),
                }
            )
    main = [arm for arm in arms if arm["condition"] != "dense"]
    adequate = all(
        arm["splits"]["test"]["metrics"]["forecast/normalized_mse"] <= QUALITY_CAP
        and arm["splits"]["test"]["metrics"]["online/variance_min"] > 1e-12
        and arm["splits"]["test"]["metrics"]["target/variance_min"] > 1e-12
        for arm in main
    )
    primary = [pair for pair in comparisons if pair["left"] == "full"]
    positive = all(pair["difference"] > 0 for pair in primary)
    intervals_positive = all(pair["lower_95"] > 0 for pair in primary)
    feasible = all(arm["splits"]["test"]["metrics"]["base"] <= tau for arm in sparse)
    decision = (
        "prediction_or_code_health_inadequate"
        if not adequate
        else "sparsity_constraint_not_met"
        if not feasible
        else "full_improvement_supported_both_seeds"
        if intervals_positive
        else "positive_but_uncertain_full_improvement"
        if positive
        else "no_consistent_full_improvement"
    )
    summary = {
        "schema": "oracle-state-learning-analysis-v1",
        "decision": decision,
        "quality_cap": QUALITY_CAP,
        "geco_tau": tau,
        "all_sparse_test_constraints_feasible": feasible,
        "adequate_prediction_all_main_arms": adequate,
        "full_improvement_both_seeds": positive,
        "full_group_intervals_positive_both_seeds": intervals_positive,
        "arms": results,
        "comparisons": comparisons,
        "limitations": [
            "Group bootstrap conditions on trained endpoints; two seeds do not establish "
            "training-seed robustness.",
            "Oracle parameters remove acquisition; results do not establish ordinary "
            "label-free parameter recovery.",
            "Observed routing unions use calibration coordinate assignments; they are "
            "finite path diagnostics, not certified functional causal graphs.",
            "Finite training does not certify global functional graph minima or falsify "
            "the identifiability theorem.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Analyze the completed bounded protocol without any new training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=8, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.runs, args.output_dir)
    print(
        json.dumps({"decision": result["decision"], "comparisons": result["comparisons"]}, indent=2)
    )


if __name__ == "__main__":
    main()
