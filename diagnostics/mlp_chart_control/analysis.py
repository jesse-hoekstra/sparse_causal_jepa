"""Audit four fixed MLP endpoints and compare forecast accuracy across state charts."""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.chart_sparsity.core import Chart, chart_matrix
from diagnostics.chart_sparsity.run import read_json, sha
from diagnostics.mlp_chart_control.core import (
    EvaluationResult,
    chart_transport,
    closure_audit,
    conjugate,
)
from diagnostics.mlp_chart_control.run import (
    PARAMETER_COUNT,
    QUALITY_CAP,
    SCHEMA,
    verify_completion,
)
from diagnostics.mlp_oracle_code_control.model import MLPPredictor
from diagnostics.oracle_state_predictor.run import forecast_metrics, write_json
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.training.rng import preserve_rng

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 2026102802
EQUIVALENCE_MARGIN = 0.002
STEPS = 100_000
REQUIRED_ARTIFACTS = {
    "config.json",
    "wandb.json",
    "startup.json",
    "history.jsonl",
    "last.pt",
    "transport.json",
    "transported.pt",
    *(
        f"{prefix}{split}.{suffix}"
        for prefix in ("", "transported_")
        for split in ("calibration", "validation")
        for suffix in ("json", "pt")
    ),
}


@dataclass(frozen=True)
class Arm:
    """One authenticated endpoint and its native and transported measurements."""

    directory: Path
    seed: int
    chart: Chart
    completion: dict[str, Any]
    record: dict[str, Any]
    evaluations: dict[str, EvaluationResult]
    closure: dict[str, dict[str, float]]


def _same(value: object, expected: object, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} differs")


def _state_identity(state: dict[str, Tensor]) -> str:
    return canonical_fingerprint({name: tensor_fingerprint(value) for name, value in state.items()})


def audit_samples(evaluation: EvaluationResult, groups: int) -> None:
    """Recompute all forecast metrics from saved samples before interpreting an endpoint."""
    samples = evaluation.samples
    shapes = {
        "predictions": (groups, 10, 2, 6),
        "targets": (groups, 10, 2, 6),
        "group_ids": (groups,),
        "group_horizon_mse": (groups, 2),
    }
    for name, shape in shapes.items():
        value = samples.get(name)
        if not isinstance(value, Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"saved {name} has the wrong shape")
        if value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"saved {name} must be finite and on CPU")
    if len({int(value) for value in samples["group_ids"]}) != groups:
        raise ValueError("independent parameter-group identities must be unique")
    errors = (samples["predictions"].double() - samples["targets"].double()).square()
    if not torch.allclose(
        errors.mean(dim=(1, 3)), samples["group_horizon_mse"], rtol=1e-10, atol=1e-12
    ):
        raise ValueError("saved group errors disagree with raw forecasts")
    metrics = forecast_metrics(
        samples["predictions"].flatten(0, 1), samples["targets"].flatten(0, 1)
    )
    metrics["base_loss"] = metrics["prediction_mse"]
    for name, expected in metrics.items():
        actual = evaluation.metrics.get(name)
        if not isinstance(actual, int | float) or not math.isclose(
            actual, expected, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"saved metric {name} disagrees with raw forecasts")


def audit_conjugated_weights(
    source: dict[str, Tensor], transported: dict[str, Tensor], matrix: Tensor
) -> None:
    """Reconstruct the analytic copy without optimizing it or trusting lineage labels."""
    with preserve_rng(torch.device("cpu")):
        model = MLPPredictor()
    model.load_state_dict(source, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("MLP parameter count differs")
    expected = cast(dict[str, Tensor], conjugate(model, matrix).state_dict())
    if expected.keys() != transported.keys():
        raise ValueError("analytic copy's parameter names differ")
    for name, value in expected.items():
        actual = transported[name]
        if value.shape != actual.shape or value.dtype != actual.dtype:
            raise ValueError("analytic copy's parameter shapes or dtypes differ")
        if not torch.allclose(value, actual, atol=1e-6, rtol=1e-6):
            raise ValueError(f"analytic copy weights are not the declared conjugation: {name}")
        if name not in {
            "network.0.weight",
            "network.4.weight",
            "network.4.bias",
        } and not torch.equal(source[name], actual):
            raise ValueError("analytic conjugation changed an interior parameter")
    if not torch.equal(source["network.0.weight"][:, 6:], transported["network.0.weight"][:, 6:]):
        raise ValueError("analytic conjugation changed action or parameter input columns")


def _load_arm(directory: Path) -> Arm:
    completion = verify_completion(directory)
    if not REQUIRED_ARTIFACTS.issubset(completion["hashes"]):
        raise ValueError("completion omits required endpoint artifacts")
    if any(name.startswith("test.") for name in completion["hashes"]):
        raise ValueError("this diagnostic prohibits test inference")
    record = read_json(directory / "config.json")
    scientific = cast(dict[str, Any], record["scientific_config"])
    seed, chart = scientific["seed"], scientific["chart"]
    if seed not in (0, 1) or chart not in ("physical", "mixed"):
        raise ValueError("unexpected training seed or state chart")
    for name, expected in {
        "schema": SCHEMA,
        "kind": "train",
        "profile": "scientific",
        "seed": seed,
        "chart": chart,
        "steps": STEPS,
        "successful_updates": STEPS,
        "skipped_updates": 0,
        "closure_passed": True,
    }.items():
        _same(completion.get(name), expected, f"completion {name}")
    for name, expected in {
        "profile": "scientific",
        "steps": STEPS,
        "batch_size": 16,
        "lr": 0.001,
        "grad_clip": 1.0,
        "weight_decay": 0.0,
        "scheduler": "constant",
        "parameter_count": PARAMETER_COUNT,
        "objective": "attached two-step raw forecast MSE",
        "architecture": "28-211-211-6 ReLU; biased linear layers; no normalization",
    }.items():
        _same(scientific.get(name), expected, f"scientific setting {name}")
    for name in ("identity", "code_identity", "data_identity", "initial_state_identity"):
        _same(record.get(name), completion.get(name), f"run {name}")
    specification = {
        name: record[name]
        for name in (
            "scientific_config",
            "data_identity",
            "data_sha256",
            "cache_code_identity",
            "code_identity",
            "initial_state_identity",
        )
    }
    _same(canonical_fingerprint(specification), completion["identity"], "run identity fingerprint")
    _same(record.get("quality_cap"), QUALITY_CAP, "quality cap")
    for name in ("cuda_matmul_allow_tf32", "cudnn_allow_tf32"):
        _same(record["runtime"][name], False, name)
    _same(record["runtime"]["float32_matmul_precision"], "highest", "matmul precision")
    native = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    for name, expected in (
        ("schema", SCHEMA),
        ("identity", completion["identity"]),
        ("step", STEPS),
    ):
        _same(native.get(name), expected, f"native checkpoint {name}")
    optimizer = cast(dict[str, Any], native["optimizer"])
    if not optimizer["state"] or any(
        int(state["step"]) != STEPS for state in optimizer["state"].values()
    ):
        raise ValueError("optimizer state does not match the fixed endpoint")
    with preserve_rng(torch.device("cpu")):
        torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
        initial = MLPPredictor()
    _same(
        _state_identity(cast(dict[str, Tensor], initial.state_dict())),
        record["initial_state_identity"],
        "fresh native initialization",
    )
    transported = cast(
        dict[str, Any],
        torch.load(directory / "transported.pt", map_location="cpu", weights_only=True),
    )
    transport = read_json(directory / "transport.json")
    opposite: Chart = "mixed" if chart == "physical" else "physical"
    matrix = chart_transport(chart, opposite)
    for name, expected in {
        "schema": SCHEMA,
        "kind": "analytic_conjugate",
        "source_identity": completion["identity"],
        "source_chart": chart,
        "target_chart": opposite,
        "optimization_steps": 0,
        "source_checkpoint_sha256": sha(directory / "last.pt"),
    }.items():
        _same(transported.get(name), expected, f"conjugate checkpoint {name}")
    for name in ("source_chart", "target_chart", "optimization_steps", "source_checkpoint_sha256"):
        _same(transport.get(name), transported[name], f"transport metadata {name}")
    if not torch.equal(transported["matrix"], matrix) or not torch.equal(
        torch.tensor(transport["matrix"], dtype=torch.float64), matrix
    ):
        raise ValueError("transport coordinate matrix differs")
    _same(
        _state_identity(native["model"]),
        transport["source_model_identity"],
        "transport source weights",
    )
    _same(
        _state_identity(transported["model"]),
        transport["transported_model_identity"],
        "transport derived weights",
    )
    audit_conjugated_weights(native["model"], transported["model"], matrix)
    evaluations: dict[str, EvaluationResult] = {}
    closure: dict[str, dict[str, float]] = {}
    for split, groups in (("calibration", 512), ("validation", 256)):
        for prefix in ("", "transported_"):
            name = prefix + split
            metrics = read_json(directory / f"{name}.json")["metrics"]
            samples = torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)[
                "samples"
            ]
            evaluation = EvaluationResult(metrics, samples)
            audit_samples(evaluation, groups)
            evaluations[name] = evaluation
        closure[split] = closure_audit(
            evaluations[split], evaluations[f"transported_{split}"], matrix
        )
        _same(transport["checks"][split]["passed"], 1.0, "recorded numerical closure")
        _same(
            completion["native_quality"][split],
            evaluations[split].metrics["prediction_mse"] <= QUALITY_CAP,
            "recorded quality flag",
        )
    return Arm(directory, seed, chart, completion, record, evaluations, closure)


def load_arms(directories: Sequence[Path]) -> dict[tuple[int, Chart], Arm]:
    """Authenticate four fresh fixed endpoints and their paired initialization/data."""
    if len(directories) != 4 or len({path.resolve() for path in directories}) != 4:
        raise ValueError("require four distinct fixed endpoint directories")
    arms: dict[tuple[int, Chart], Arm] = {}
    for directory in directories:
        arm = _load_arm(directory)
        key = arm.seed, arm.chart
        if key in arms:
            raise ValueError("duplicate seed/chart endpoint")
        arms[key] = arm
    reference = next(iter(arms.values()))
    for arm in arms.values():
        for name in ("code_identity", "data_identity", "data_sha256", "cache_code_identity"):
            _same(arm.record[name], reference.record[name], f"common {name}")
    for seed in (0, 1):
        physical, mixed = arms[seed, "physical"], arms[seed, "mixed"]
        _same(
            physical.record["initial_state_identity"],
            mixed.record["initial_state_identity"],
            "paired initial weights",
        )
        for split in ("calibration", "validation"):
            a, b = physical.evaluations[split].samples, mixed.evaluations[split].samples
            if not torch.equal(a["group_ids"], b["group_ids"]):
                raise ValueError("paired charts use different ordered parameter groups")
            if not torch.equal(
                a["group_ids"], arms[0, "physical"].evaluations[split].samples["group_ids"]
            ):
                raise ValueError("seeds use different ordered parameter groups")
            expected = a["targets"].double() @ chart_matrix("mixed").T
            if not torch.allclose(expected, b["targets"].double(), atol=1e-6, rtol=1e-6):
                raise ValueError("mixed targets are not the declared fixed coordinate change")
            if not torch.equal(
                a["targets"], arms[0, "physical"].evaluations[split].samples["targets"]
            ):
                raise ValueError("seeds use different physical targets")
    return arms


def summarize(arms: dict[tuple[int, Chart], Arm]) -> dict[str, Any]:
    """Apply the prospective fit/equivalence rules with shared independent-group draws."""
    if set(arms) != {(seed, chart) for seed in (0, 1) for chart in ("physical", "mixed")}:
        raise ValueError("summary requires both charts and both predeclared seeds")
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(256, (BOOTSTRAP_DRAWS, 256), generator=generator)
    results: list[dict[str, Any]] = []
    for seed in (0, 1):
        physical, mixed = arms[seed, "physical"], arms[seed, "mixed"]
        fit: dict[str, bool] = {}
        errors: dict[str, dict[str, float]] = {}
        for chart, arm in (("physical", physical), ("mixed", mixed)):
            errors[chart] = {
                split: arm.evaluations[split].metrics["prediction_mse"]
                for split in ("calibration", "validation")
            }
            fit[chart] = all(value <= QUALITY_CAP for value in errors[chart].values())
        difference = (
            mixed.evaluations["validation"].samples["group_horizon_mse"]
            - physical.evaluations["validation"].samples["group_horizon_mse"]
        ).mean(dim=1)
        interval = torch.quantile(
            difference[indices].mean(dim=1), torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        low, high = (float(value) for value in interval)
        equivalent = low >= -EQUIVALENCE_MARGIN and high <= EQUIVALENCE_MARGIN
        if all(fit.values()):
            decision = "comparable_fresh_fits" if equivalent else "residual_chart_dependence"
        elif any(fit.values()):
            decision = "attainable_but_fresh_optimization_gap"
        else:
            decision = "fit_limitation_inconclusive"
        results.append(
            {
                "seed": seed,
                "native_mse": errors,
                "native_fit": fit,
                "mixed_minus_physical_validation_mse": float(difference.mean()),
                "paired_group_bootstrap_95_interval": [low, high],
                "equivalence_passed": equivalent,
                "decision": decision,
            }
        )
    if all(all(row["native_fit"].values()) for row in results):
        decision = (
            "comparable_fresh_fits"
            if all(row["equivalence_passed"] for row in results)
            else "residual_chart_dependence"
        )
    elif results[0]["decision"] == results[1]["decision"]:
        decision = results[0]["decision"]
    else:
        decision = "mixed_outcomes"
    return {
        "schema": "fixed-chart-mlp-analysis-v1",
        "decision": decision,
        "quality_cap": QUALITY_CAP,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling_unit": (
            "independent parameter group; shared draws across the two fixed training seeds"
        ),
        "interval_scope": (
            "conditional on four trained endpoints; not across-training-seed uncertainty"
        ),
        "test_inference": False,
        "seeds": results,
        "interpretation": (
            "Predictor feasibility and coordinate dependence; "
            "no learned-state MCC or parameter-sparsity comparison."
        ),
    }


def analyze(directories: Sequence[Path], output: Path) -> dict[str, Any]:
    """Save a checked endpoint summary and human-readable report without new training."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("analysis output must be empty")
    arms = load_arms(directories)
    result = summarize(arms)
    result["provenance"] = [
        {
            "directory": str(arm.directory.resolve()),
            "seed": arm.seed,
            "chart": arm.chart,
            "completion_sha256": sha(arm.directory / "completion.json"),
            "checkpoint_sha256": sha(arm.directory / "last.pt"),
            "source_identity": arm.record["code_identity"],
            "data_identity": arm.record["data_identity"],
            "closure": arm.closure,
        }
        for _, arm in sorted(arms.items())
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", result)
    lines = [
        "# Fixed-chart MLP diagnostic",
        "",
        f"Decision: **{result['decision']}**.",
        "",
        "All four fixed 100,000-update endpoints and their analytic coordinate copies "
        "were authenticated.",
        "No test inference, new optimization, or checkpoint selection was performed.",
        "",
        "| Seed | Chart | Calibration MSE | Validation MSE | Both <= .02 |",
        "|---|---|---:|---:|---|",
    ]
    for row in result["seeds"]:
        for chart in ("physical", "mixed"):
            mse = row["native_mse"][chart]
            lines.append(
                f"| {row['seed']} | {chart} | {mse['calibration']:.9f} | "
                f"{mse['validation']:.9f} | {row['native_fit'][chart]} |"
            )
    lines += [
        "",
        "Paired mixed-minus-physical validation-MSE intervals "
        "(1,000 parameter-group bootstrap draws):",
        "",
    ]
    for row in result["seeds"]:
        low, high = row["paired_group_bootstrap_95_interval"]
        lines.append(
            f"- Seed {row['seed']}: [{low:.9f}, {high:.9f}]; equivalence within +/- .002: "
            f"{row['equivalence_passed']}; {row['decision']}."
        )
    lines += [
        "",
        "Intervals condition on the fixed trained endpoints. Analytic copies establish "
        "same-architecture coordinate attainability, not success of fresh optimization.",
        "This diagnostic does not measure learned-state MCC or compare baseline "
        "against full sparsity.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(
        output / "completion.json",
        {
            "schema": result["schema"],
            "decision": result["decision"],
            "analysis_source_sha256": sha(Path(__file__)),
            "hashes": {name: sha(output / name) for name in ("summary.json", "report.md")},
        },
    )
    return result


def main() -> None:
    """Analyze exactly four saved endpoints using the fixed scientific protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.run_dir, args.output_dir)
    print(result["decision"], flush=True)


if __name__ == "__main__":
    main()
