"""Audit four fixed-depth rotating-pair validation endpoints without training or test inference."""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.depth_oracle.model import (
    MLP_HIDDEN_DIM,
    MLP_PREDICTOR_PARAMETERS,
    NATIVE_LAYERS,
    NATIVE_PREDICTOR_PARAMETERS,
    NATIVE_WIDTH,
)
from diagnostics.depth_oracle.run import SCHEMA, verify_completion
from diagnostics.harder_oracle.analysis import ordering_evidence, screen_decision
from diagnostics.oracle_state_learning.analysis import (
    audit_samples,
    controller_history,
    paired_mcc_interval,
    read_json,
)
from diagnostics.oracle_state_learning.data import sha
from diagnostics.rotating_pair_oracle.dgp import PHASE_GAIN, ROTATION_SCALE
from scjepa.data.datasets import canonical_fingerprint, tensor_fingerprint
from scjepa.eval.recovery import CoordinateMatch, fit_coordinate_match, mean_correlation_coefficient
from scjepa.eval.state_probes import fit_state_probe, state_probe_metrics

TRAINING_ARTIFACTS = frozenset(
    {
        "config.json",
        "initial.pt",
        "last.pt",
        "matching.json",
        "startup.json",
        "history.jsonl",
        "wandb.json",
        "calibration.pt",
        "calibration.json",
        "validation.pt",
        "validation.json",
    }
)


def require_candidate(settings: dict[str, Any]) -> None:
    """Reject parent phases or action encodings before reading endpoint samples."""
    if (
        settings.get("candidate") != "rotating_pair_innovation"
        or settings.get("action_mix") != "paley24"
        or settings.get("transport_chart") != "isotropic"
        or settings.get("phase_formula") != "rotating_pair"
        or settings.get("phase_gain") != PHASE_GAIN
        or settings.get("rho") != 0.7
        or settings.get("rotation_scale") != ROTATION_SCALE
    ):
        raise ValueError("Require the fixed rotating-pair phase, gain and paley24 actions")


def require_architecture(settings: dict[str, Any]) -> None:
    """Bind the fixed third layer and its matched MLP capacity before reading samples."""
    mlp = settings.get("condition") == "mlp"
    expected = {
        "native_spartan_layers": NATIVE_LAYERS,
        "native_spartan_width": NATIVE_WIDTH,
        "native_spartan_parameter_count": NATIVE_PREDICTOR_PARAMETERS,
        "predictor_parameter_count": MLP_PREDICTOR_PARAMETERS
        if mlp
        else NATIVE_PREDICTOR_PARAMETERS,
        "mlp_hidden_dim": MLP_HIDDEN_DIM if mlp else None,
    }
    if any(settings.get(name) != value for name, value in expected.items()):
        raise ValueError("Require the fixed three-layer architecture and matched MLP capacity")


def require_predictor_tensors(state: dict[str, Any], condition: str) -> None:
    """Reject two-layer or wrong-width checkpoints even if their metadata was relabelled."""
    values = {
        name: value
        for name, value in state.items()
        if name.startswith("predictor.") and isinstance(value, Tensor)
    }
    expected_count = MLP_PREDICTOR_PARAMETERS if condition == "mlp" else NATIVE_PREDICTOR_PARAMETERS
    if condition == "mlp":
        shapes = {
            "predictor.network.0.weight": (MLP_HIDDEN_DIM, 36),
            "predictor.network.2.weight": (MLP_HIDDEN_DIM, MLP_HIDDEN_DIM),
            "predictor.network.4.weight": (6, MLP_HIDDEN_DIM),
        }
    else:
        layers = {name.split(".")[2] for name in values if name.startswith("predictor.layers.")}
        if layers != {str(index) for index in range(NATIVE_LAYERS)}:
            raise ValueError("Checkpoint predictor depth/width/count differs")
        shapes = {
            "predictor.state_identity": (6, NATIVE_WIDTH),
            "predictor.action_identity": (24, NATIVE_WIDTH),
            "predictor.parameter_identity": (6, NATIVE_WIDTH),
            **{
                f"predictor.layers.{index}.project_q.weight": (NATIVE_WIDTH, NATIVE_WIDTH)
                for index in range(NATIVE_LAYERS)
            },
        }
    if sum(value.numel() for value in values.values()) != expected_count or any(
        name not in values or values[name].shape != shape for name, shape in shapes.items()
    ):
        raise ValueError("Checkpoint predictor depth/width/count differs")


def routing_metrics(samples: dict[str, Tensor], matching: dict[str, Any]) -> dict[str, float]:
    """Describe finite observed routing unions for the 6/24/6 innovation process."""
    state_truth = torch.zeros(6, 6, dtype=torch.bool)
    state_truth[:3, :3] = state_truth[3:, 3:] = True
    parameter_truth = torch.eye(6, dtype=torch.bool)
    parameter_truth[torch.arange(6), [3, 4, 5, 0, 1, 2]] = True
    truths = {
        "state": state_truth,
        "action": torch.ones(6, 24, dtype=torch.bool),
        "parameter": parameter_truth,
    }
    result: dict[str, float] = {}
    for name, truth in truths.items():
        if f"graph_{name}" not in samples:
            continue
        frequencies = samples[f"graph_{name}"].double().mean(0)
        if frequencies.shape != truth.shape or not bool(
            ((frequencies >= 0) & (frequencies <= 1)).all()
        ):
            raise ValueError("Malformed local routing frequencies")
        aligned = frequencies[matching["target_columns"]]
        if name == "state":
            aligned = aligned[:, matching["online_columns"]]
        support = aligned > 0
        result[f"routing/{name}_observed_union_edges"] = float(support.sum())
        result[f"routing/{name}_mean_local_edges"] = float(frequencies.sum())
        result[f"routing/{name}_aligned_union_shd"] = float((support ^ truth).sum())
    return result


def _load(directory: Path, *, allow_smoke: bool) -> dict[str, Any]:
    completion = verify_completion(directory, kind="train", require_current_source=False)
    if not completion["hashes"].keys() >= TRAINING_ARTIFACTS:
        raise ValueError("Completion does not authenticate every required training artifact")
    config = read_json(directory / "config.json")
    candidate = config["candidate_settings"]
    require_candidate(candidate)
    if (
        config["identity"] != completion["identity"]
        or config["code_identity"] != completion["code_identity"]
        or config["data_identity"] != completion["data_identity"]
    ):
        raise ValueError("Configuration and completion identities disagree")
    scientific = config["scientific_config"]
    require_architecture(scientific)
    condition = scientific["condition"]
    if condition not in {"dense", "baseline", "full", "mlp"} or scientific["seed"] != 0:
        raise ValueError("Expected a supported seed-zero condition")
    if (
        completion["condition"] != condition
        or completion["seed"] != scientific["seed"]
        or config["config"]["condition"] != condition
        or config["config"]["seed"] != scientific["seed"]
        or config["config"]["steps"] != scientific["steps"]
    ):
        raise ValueError("Endpoint condition, seed or configuration budget differs")
    profile = "smoke" if allow_smoke else "exploratory"
    steps = scientific["steps"]
    if completion["profile"] != profile or (not allow_smoke and steps != 100_000):
        raise ValueError("Scientific analysis requires fixed 100000-update exploratory endpoints")
    if (
        completion["steps"] != steps
        or completion["successful_updates"] != steps
        or completion["skipped_updates"] != 0
    ):
        raise ValueError("Endpoint accepted-update count differs from its declared budget")
    checkpoint = cast(
        dict[str, Any], torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    )
    if (
        checkpoint.get("schema") != SCHEMA
        or checkpoint["step"] != steps
        or checkpoint["identity"] != completion["identity"]
    ):
        raise ValueError("Final checkpoint budget or identity differs")
    require_predictor_tensors(checkpoint["model"], condition)
    if checkpoint["initial_checkpoint_sha256"] != sha(directory / "initial.pt"):
        raise ValueError("Final checkpoint initial-source link differs")
    initial = cast(
        dict[str, Any], torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)
    )
    require_predictor_tensors(initial["model"], condition)
    for prefix, field in (
        ("online_encoder", "initial_encoder_identity"),
        ("predictor", "initial_predictor_identity"),
    ):
        state = {
            name[len(prefix) + 1 :]: tensor_fingerprint(value)
            if isinstance(value, Tensor)
            else value
            for name, value in initial["model"].items()
            if name.startswith(prefix + ".")
        }
        if not state or not (
            canonical_fingerprint(state) == initial[field] == completion[field] == config[field]
        ):
            raise ValueError("Initial tensor identity differs from its authenticated record")
    optimizer_states = checkpoint["optimizer"]["state"]
    if not optimizer_states or any(
        int(state["step"]) != steps for state in optimizer_states.values()
    ):
        raise ValueError("Adam update counters differ from the fixed endpoint")
    if (
        condition in {"baseline", "full"}
        and int(checkpoint["objective"]["geco.num_updates"]) != steps
    ):
        raise ValueError("GECO update counter differs from the fixed endpoint")
    if (directory / "test.pt").exists() or (directory / "test.json").exists():
        raise ValueError("Candidate screen artifacts must never contain test inference")
    matching = read_json(directory / "matching.json")
    splits: dict[str, dict[str, Any]] = {}
    for split in ("calibration", "validation"):
        groups = int(config["candidate_settings"][f"{split}_groups"])
        payload = cast(
            dict[str, Any],
            torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True),
        )
        samples = audit_samples(payload, groups)
        for encoder, key in (("online", "state_mcc"), ("target", "target_state_mcc")):
            columns = tuple(matching[f"{encoder}_columns"])
            features, truth = (
                samples[f"{encoder}_states"].flatten(0, 2),
                samples["states"].flatten(0, 2),
            )
            if (
                split == "calibration"
                and fit_coordinate_match(features, truth, method="pearson").columns != columns
            ):
                raise ValueError("Saved assignment was not fitted on calibration")
            actual = mean_correlation_coefficient(CoordinateMatch(columns), features, truth)
            if not math.isclose(actual, payload["metrics"][key], rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(
                    "Reported MCC differs from saved samples and calibration assignment"
                )
        variance = samples["targets"].double().flatten(0, -2).var(0, correction=0).mean()
        normalized = float(
            (samples["predictions"].double() - samples["targets"].double()).square().mean()
            / variance
        )
        if not math.isclose(
            normalized, payload["metrics"]["normalized_prediction_mse"], abs_tol=1e-8, rel_tol=2e-6
        ):
            raise ValueError("Reported normalized forecast error differs from saved samples")
        payload["metrics"].update(routing_metrics(samples, matching))
        splits[split] = payload
    calibration_samples = splits["calibration"]["samples"]
    if set(calibration_samples["group_ids"].tolist()) & set(
        splits["validation"]["samples"]["group_ids"].tolist()
    ):
        raise ValueError("Calibration and validation parameter groups overlap")
    calibration_truth = calibration_samples["states"].flatten(0, 2)
    for encoder in ("online", "target"):
        probe = fit_state_probe(
            calibration_samples[f"{encoder}_states"].flatten(0, 2), calibration_truth, ridge=1e-3
        )
        for payload in splits.values():
            samples = payload["samples"]
            actual = state_probe_metrics(
                probe.predict(samples[f"{encoder}_states"].flatten(0, 2)),
                samples["states"].flatten(0, 2),
            )
            if not math.isclose(
                actual["r2"],
                payload["metrics"][f"{encoder}_state_probe/r2"],
                rel_tol=1e-9,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Reported state information readout differs from calibration-only probe"
                )
    return {
        "directory": str(directory),
        "completion": completion,
        "config": config,
        "condition": condition,
        "matching": matching,
        "splits": splits,
    }


def _summary(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": arm["condition"],
        "directory": arm["directory"],
        "transport_chart": arm["config"]["candidate_settings"]["transport_chart"],
        "candidate": arm["config"]["candidate_settings"]["candidate"],
        "action_mix": arm["config"]["candidate_settings"]["action_mix"],
        "phase_formula": arm["config"]["candidate_settings"]["phase_formula"],
        "phase_gain": arm["config"]["candidate_settings"]["phase_gain"],
        "native_spartan_layers": NATIVE_LAYERS,
        "predictor_parameter_count": arm["config"]["scientific_config"][
            "predictor_parameter_count"
        ],
        "rho": arm["config"]["candidate_settings"]["rho"],
        "rotation_scale": arm["config"]["candidate_settings"]["rotation_scale"],
        "data_identity": arm["completion"]["data_identity"],
        "source_identity": arm["completion"]["code_identity"],
        "metrics": {split: result["metrics"] for split, result in arm["splits"].items()},
        "verified_artifact_count": len(arm["completion"]["hashes"]),
        "artifact_sha256": arm["completion"]["hashes"],
        "completion_sha256": sha(Path(arm["directory"]) / "completion.json"),
        "controller_history": controller_history(Path(arm["directory"]) / "history.jsonl"),
        "wandb": arm["completion"]["wandb"],
    }


def analyze(
    directories: Sequence[Path],
    output: Path,
    *,
    primary_chart: str = "isotropic",
    draws: int = 1000,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Audit exactly four matched three-layer oracle comparison arms."""
    if primary_chart != "isotropic":
        raise ValueError("Primary chart must be isotropic")
    arms = [_load(directory, allow_smoke=allow_smoke) for directory in directories]
    indexed = {arm["condition"]: arm for arm in arms}
    if len(arms) != 4 or set(indexed) != {"dense", "mlp", "baseline", "full"}:
        raise ValueError("Require exactly four unique primary-condition endpoints")
    if any(arm["config"]["candidate_settings"]["transport_chart"] != primary_chart for arm in arms):
        raise ValueError(
            f"Primary comparison must use only the declared {primary_chart} state transport"
        )
    for key in ("data_identity", "code_identity", "initial_encoder_identity"):
        if len({arm["completion"][key] for arm in arms}) != 1:
            raise ValueError(f"Primary arms differ in {key}")
    if (
        len(
            {
                indexed[name]["completion"]["initial_predictor_identity"]
                for name in ("dense", "baseline", "full")
            }
        )
        != 1
    ):
        raise ValueError("Primary SPARTAN predictor initialization differs")
    if len({arm["completion"]["steps"] for arm in arms}) != 1:
        raise ValueError("Primary fixed budgets differ")
    sparse = [indexed[name] for name in ("baseline", "full")]
    tau = 1.1 * indexed["dense"]["splits"]["calibration"]["metrics"]["base"]
    if any(arm["config"]["geco"]["tau"] != tau for arm in sparse):
        raise ValueError("Sparse bound differs from 1.1 times final dense calibration loss")
    if any(
        not arm["config"]["geco"]["enabled"]
        or arm["completion"]["calibration_identity"] is None
        or arm["config"]["calibration_identity"] != arm["completion"]["calibration_identity"]
        for arm in sparse
    ):
        raise ValueError("Sparse conditions lack an enabled shared calibration identity")
    if len({arm["completion"]["calibration_identity"] for arm in sparse}) != 1:
        raise ValueError("Sparse conditions use different calibration artifacts")
    reference = indexed["dense"]
    for arm in arms:
        if arm["config"]["candidate_settings"] != reference["config"]["candidate_settings"]:
            raise ValueError("Primary candidate settings differ")
        for split in ("calibration", "validation"):
            left, right = arm["splits"][split]["samples"], reference["splits"][split]["samples"]
            if not torch.equal(left["group_ids"], right["group_ids"]) or not torch.equal(
                left["states"], right["states"]
            ):
                raise ValueError("Primary arms require paired calibration and validation states")
    comparisons: list[dict[str, Any]] = []
    for left, right in (
        ("full", "baseline"),
        ("baseline", "dense"),
        ("full", "dense"),
        ("baseline", "mlp"),
    ):
        a, b = indexed[left], indexed[right]
        av, bv = a["splits"]["validation"]["samples"], b["splits"]["validation"]["samples"]
        if not torch.equal(av["group_ids"], bv["group_ids"]) or not torch.equal(
            av["states"], bv["states"]
        ):
            raise ValueError("Paired comparisons require identical addressed physical states")
        comparisons.append(
            {
                "left": left,
                "right": right,
                "split": "validation",
                **paired_mcc_interval(
                    av["online_states"],
                    bv["online_states"],
                    av["states"],
                    a["matching"]["online_columns"],
                    b["matching"]["online_columns"],
                    draws=draws,
                ),
            }
        )
    endpoint_metrics = {
        name: arm["splits"]["validation"]["metrics"] for name, arm in indexed.items()
    }
    result: dict[str, Any] = {
        "schema": "depth-oracle-validation-analysis-v1",
        "profile": "smoke" if allow_smoke else "exploratory",
        "primary_chart": primary_chart,
        "candidate": "rotating_pair_innovation",
        "action_mix": "paley24",
        "phase_formula": "rotating_pair",
        "phase_gain": PHASE_GAIN,
        "native_spartan_layers": NATIVE_LAYERS,
        "native_spartan_width": NATIVE_WIDTH,
        "geco_tau": tau,
        **screen_decision(endpoint_metrics),
        **ordering_evidence(endpoint_metrics, comparisons, split="validation"),
        "arms": [_summary(arm) for arm in arms],
        "comparisons": comparisons,
        "all_sparse_validation_constraints_feasible": all(
            arm["splits"]["validation"]["metrics"]["base"] <= tau for arm in sparse
        ),
        "limitations": [
            "Validation selected the candidate; success still requires frozen-endpoint "
            "test confirmation.",
            "One seed is exploratory. Group intervals condition on trained models and "
            "do not estimate training-seed variability.",
            "The adaptive ordering rule does not replace the original .05/.03 effect "
            "targets or imply that screen_passed is true.",
            "Benchmark selection, including a near-perfect baseline ceiling, remains "
            "a separate recorded decision; these flags do not choose a setting or authorize test.",
            "Finite routing unions do not certify population functional causal supports "
            "or global L0 minima.",
            "Oracle supplied parameters do not establish ordinary learned-parameter recovery.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    """Analyze final validation endpoints without touching training or test caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=4, type=Path, required=True)
    parser.add_argument("--primary-chart", choices=("isotropic",), default="isotropic")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.runs, args.output_dir, primary_chart=args.primary_chart)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "screen_passed": result["screen_passed"],
                "ordering_supported": result["ordering_supported"],
                "confirmation_eligible": result["confirmation_eligible"],
                "comparisons": result["comparisons"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
