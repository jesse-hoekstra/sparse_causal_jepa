"""Separate native MCC/graph evaluation from whole-code physical-parameter diagnostics."""

import copy
import math
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    MatchedBatch,
    _interval,  # pyright: ignore[reportPrivateUsage]
    predict_matched,
    response_report,
)
from diagnostics.joint_state_survival.data import refresh_evaluation
from diagnostics.joint_state_survival.evaluation import monitor as diagnostic_monitor
from diagnostics.joint_state_survival.evaluation import state_metrics
from diagnostics.multiquery_control.data import EvaluationData
from diagnostics.multiquery_control.evaluation import encode_codes, parameter_readouts
from diagnostics.multiquery_control.run import atomic_save, sha
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.overcomplete_use_precision.evaluation import direct_decomposition, join_batches
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.experiments.evaluation import evaluate_model, validate_model
from scjepa.experiments.storage import read_json, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig
from scjepa.training.rng import preserve_rng

if TYPE_CHECKING:
    from diagnostics.sparse_comparison.source import Inputs


def _bound(metric: dict[str, Any], index: int) -> float | None:
    interval = metric.get("interval_95")
    if metric.get("defined", True) is False or not isinstance(interval, list):
        return None
    entries = cast(list[Any], interval)
    if len(entries) != 2 or not all(isinstance(value, int | float) for value in entries):
        return None
    values = [float(value) for value in entries]
    if not all(math.isfinite(value) for value in values) or values[0] > values[1]:
        return None
    return values[index]


def classify(retention_r2: Sequence[float], response: dict[str, Any]) -> dict[str, Any]:
    """Keep revised retention/fidelity admission distinct from the original benefit criterion."""
    if len(retention_r2) != 6:
        raise ValueError("All six whole-code parameter readouts are required")
    rows = response["per_parameter"]
    if len(rows) != 6 or [row["parameter"] for row in rows] != list(range(1, 7)):
        raise ValueError("All six physical response rows must be present in parameter order")
    factors: list[dict[str, Any]] = []
    for index, (r2, row) in enumerate(zip(retention_r2, rows, strict=True), 1):
        upper = _bound(row["normalized_response_error"], 1)
        lower = _bound(row["predictive_benefit"], 0)
        energies = {
            name: row.get(name) for name in ("exact_response_energy", "learned_response_energy")
        }
        energy_valid = all(
            isinstance(value, int | float) and math.isfinite(value) and value > 0
            for value in energies.values()
        )
        factors.append(
            {
                "parameter": index,
                "whole_code_r2": float(r2),
                "retention_pass": math.isfinite(r2) and r2 >= 0.8,
                "response_error_upper95": upper,
                **energies,
                "response_energy_valid": energy_valid,
                "response_fidelity_pass": energy_valid and upper is not None and upper < 1,
                "benefit_lower95": lower,
                "positive_benefit_pass": lower is not None and lower > 0,
            }
        )
    retention = all(row["retention_pass"] for row in factors)
    fidelity = all(row["response_fidelity_pass"] for row in factors)
    benefit = all(row["positive_benefit_pass"] for row in factors)
    return {
        "per_parameter": factors,
        "all_six_retention": retention,
        "all_six_response_fidelity": fidelity,
        "all_six_positive_benefit": benefit,
        "revised_admission_pass": retention and fidelity,
        "original_retention_and_use_gate_pass": retention and fidelity and benefit,
        "scope": (
            "Revised exploratory admission is retention plus fidelity; original positive-benefit "
            "criterion remains separately reported. Coordinate parameter MCC/SHD never "
            "determines either gate."
        ),
    }


@contextmanager
def _evaluation_context(model: SCJEPA) -> Generator[None]:
    """Preserve training modes and RNG across both existing evaluator families."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        with preserve_rng(next(model.parameters()).device):
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def cohorts(inputs: "Inputs") -> dict[str, Any]:
    """Name the two distinct historically reused cohorts and their different draw protocols."""
    metadata = inputs.evaluation["metadata"]
    return {
        "native_registered_dgp": {
            "data_root_seed": inputs.data_config.root_seed,
            "data_splits": ["evaluation_calibration", "validation", "test"],
            "evaluation_settings": asdict(inputs.native_evaluation),
            "evaluation_seed_role": (
                "Fixed evaluation/projection stream, not a replacement DGP root seed"
            ),
            "historically_reused": True,
        },
        "diagnostic_direct_draws": {
            "base_draw_seed": metadata.get("seed"),
            "split_seed_policy": "Archived direct generator draws with 1009 split offsets",
            "groups": {
                name: len(inputs.evaluation[name]["theta"])
                for name in ("calibration", "validation", "test")
            },
            "response": (
                "Donor 0, queries 5..9, positive min(.01,.5*(.75-abs(theta))), "
                "two attached horizons"
            ),
            "historically_reused": True,
        },
        "interpretation": (
            "These are distinct existing cohorts. Native MCC/SHD and diagnostic "
            "retention/physical-response scores must not be treated as the same held-out "
            "groups or fresh confirmation."
        ),
        "parameter_coordinate_metrics": (
            "Native parameter-coordinate MCC and routing SHD are descriptive only; "
            "whole-code physical-parameter recovery defines retention."
        ),
    }


def _feasibility(metrics: dict[str, float], tau: float | None, prefix: str) -> None:
    if tau is not None:
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("The frozen scalar GECO bound must be finite and positive")
        metrics["geco/tau"] = tau
        metrics[f"{prefix}/feasible"] = float(metrics[f"{prefix}/base_loss"] <= tau)
        metrics[f"{prefix}/base_minus_tau"] = metrics[f"{prefix}/base_loss"] - tau


def monitor(
    model: SCJEPA,
    inputs: "Inputs",
    *,
    groups: int = 64,
    bootstrap_samples: int = 1000,
    seed: int = 202610261301,
    objective: ObjectiveConfig | None = None,
    tau: float | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Combine native validation and existing original64 diagnostics without test inference."""
    if groups < 2 or groups > min(
        len(inputs.evaluation[name]["theta"]) for name in ("calibration", "validation")
    ):
        raise ValueError("Monitoring groups must fit both original calibration and validation")
    if inputs.native_evaluation.graph_support_threshold != 0:
        raise ValueError("The comparison preserves native union graph support threshold zero")
    with _evaluation_context(model):
        native = validate_model(
            model,
            inputs.data_config,
            inputs.evaluation["metadata"]["observation_matrix"],
            inputs.native_evaluation,
            objective=objective or inputs.objective_config,
        )
        _feasibility(native, tau, "validation")
        diagnostic, raw = diagnostic_monitor(
            model,
            inputs.evaluation,
            DiagnosticSelection(tuple(range(groups)), tuple(range(groups))),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            final=False,
        )
        gate = classify(diagnostic["retention"]["selected_validation_r2"], diagnostic["response"])
    return {
        "role": "validation_monitor",
        "native": native,
        "diagnostic": diagnostic,
        "admission": gate,
        "cohorts": cohorts(inputs),
        "test_inference": False,
    }, raw


@torch.no_grad()
def physical_response(
    model: SCJEPA,
    data: EvaluationData,
    *,
    split: Literal["validation", "test"],
    final: bool,
    chunk_size: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor], dict[str, Tensor]]:
    """Use explicit test authorization and one group bootstrap after chunked physical prediction."""
    if split == "test" and not final:
        raise ValueError("Test physical responses require the fixed-final evaluation boundary")
    if chunk_size < 1 or model.training or model.parameter_encoder.param_dim != 6:
        raise ValueError("Require eval-mode six-code model and positive chunks")
    groups = len(data[split]["theta"])
    batches: list[MatchedBatch] = []
    predictions: list[Tensor] = []
    changed: list[Tensor] = []
    for start in range(0, groups, chunk_size):
        indices = tuple(range(start, min(start + chunk_size, groups)))
        batch, _, _ = matched_batch(model, data, split, indices, final=final)
        result = predict_matched(model, batch)
        batches.append(batch.to(torch.zeros((), dtype=torch.float64)))
        predictions.append(result.predictions.cpu())
        changed.append(result.changed_predictions.cpu())
    joined = join_batches(batches)
    base, counterfactual = torch.cat(predictions), torch.cat(changed)
    report, raw = response_report(
        base, counterfactual, joined, bootstrap_samples=bootstrap_samples, seed=seed
    )
    decomposition = direct_decomposition(
        base, counterfactual, joined.targets, joined.changed_targets
    )
    torch.testing.assert_close(
        raw["predictive_benefit"],
        decomposition["benefit_from_direct_vectors"],
        atol=2e-15,
        rtol=1e-9,
    )
    raw.update(decomposition)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.randint(groups, (bootstrap_samples, groups), generator=generator)
    report["decomposition_means"] = {
        name: [float(x) for x in value.mean(0)] for name, value in decomposition.items()
    }
    report["decomposition_intervals"] = {
        name: [
            _interval(value[:, factor], torch.ones(groups, dtype=torch.float64), samples)
            for factor in range(6)
        ]
        for name, value in decomposition.items()
    }
    report["direct_vector_identity_max_absolute_error"] = float(
        (raw["predictive_benefit"] - decomposition["benefit_from_direct_vectors"]).abs().max()
    )
    report["interpretation"] = (
        "Historically reused diagnostic test at final fixed budget"
        if split == "test"
        else "Historically reused diagnostic validation"
    )
    return (
        report,
        raw,
        {
            "factual_predictions": base,
            "counterfactual_predictions": counterfactual,
            "factual_targets": joined.targets,
            "counterfactual_targets": joined.changed_targets,
        },
    )


def _checkpoint_boundary(
    model: SCJEPA, path: Path, identity: str, step: int, fixed_budget: int
) -> str:
    if step != fixed_budget or not path.is_file() or path.name != "last.pt":
        raise ValueError("Test inference requires the final fixed-budget last.pt checkpoint")
    payload = cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=False))
    if (
        payload["step"] != fixed_budget
        or payload["run_metadata"]["experiment_run_identity"] != identity
    ):
        raise ValueError("Final checkpoint step/run identity differs")
    state = model.state_dict()
    if set(state) != set(payload["model"]):
        raise ValueError("Model does not match the final checkpoint")
    for name, value in state.items():
        expected = payload["model"][name]
        if isinstance(value, Tensor):
            if value.dtype != expected.dtype or not torch.equal(value.detach().cpu(), expected):
                raise ValueError("Evaluated model differs from final checkpoint tensors")
        elif value != expected:
            raise ValueError("Evaluated model differs from final checkpoint flags")
    return sha(path)


@torch.no_grad()
def _native_target_variance(model: SCJEPA, inputs: "Inputs", samples: Path) -> float:
    payload = cast(dict[str, Tensor], torch.load(samples, map_location="cpu", weights_only=True))
    states = payload["states"]
    observation = VectorObservationMap(
        matrix=inputs.evaluation["metadata"]["observation_matrix"], dimension=6
    )
    reference = next(model.target_encoder.parameters())
    flat = observation.forward(states).flatten(0, 1)
    outputs: list[Tensor] = []
    for start in range(0, len(flat), 128):
        outputs.append(
            model.target_encoder.encode_sequence(flat[start : start + 128].to(reference))[
                :, 10:12
            ].cpu()
        )
    targets = torch.cat(outputs).reshape(len(states), 10, 2, 6).double()
    return float(targets.var(0, correction=0).mean())


def final(
    model: SCJEPA,
    inputs: "Inputs",
    output: Path,
    *,
    objective: ObjectiveConfig,
    tau: float | None,
    identity: str,
    condition: str,
    step: int,
    fixed_budget: int = 30000,
    verification: bool = False,
    chunk_size: int = 64,
    bootstrap_samples: int = 1000,
    seed: int = 202610261301,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate only final last.pt on both named cohorts and emit native paired-MCC artifacts."""
    if condition not in ("dense", "baseline", "full"):
        raise ValueError("Require a predeclared comparison condition")
    if verification:
        if not 1 <= fixed_budget <= 3:
            raise ValueError("Verification budgets must be 1..3 updates")
    elif fixed_budget != 30000 or bootstrap_samples != 1000 or seed != 202610261301:
        raise ValueError("Scientific final budget and diagnostic bootstrap protocol are fixed")
    if inputs.native_evaluation.graph_support_threshold != 0:
        raise ValueError("Keep the declared native graph union threshold zero")
    if not verification and (
        inputs.native_evaluation.calibration_episodes,
        inputs.native_evaluation.validation_episodes,
        inputs.native_evaluation.test_episodes,
        *(len(inputs.evaluation[name]["theta"]) for name in ("calibration", "validation", "test")),
    ) != (512, 256, 512, 512, 256, 512):
        raise ValueError("Preserve both original full512/256/512 held-out cohort sizes")
    checkpoint = output / "last.pt"
    checkpoint_sha = _checkpoint_boundary(model, checkpoint, identity, step, fixed_budget)
    for first, second in (
        ("calibration", "validation"),
        ("calibration", "test"),
        ("validation", "test"),
    ):
        if bool(
            (inputs.evaluation[first]["theta"][:, None] == inputs.evaluation[second]["theta"][None])
            .all(-1)
            .any()
        ):
            raise ValueError("Diagnostic calibration/validation/test parameter groups overlap")
    directory = output / "evaluation"
    with _evaluation_context(model), torch.no_grad():
        native = evaluate_model(
            model,
            inputs.data_config,
            inputs.evaluation["metadata"]["observation_matrix"],
            inputs.native_evaluation,
            directory,
            objective=objective,
        )
        _feasibility(native, tau, "test")
        native["evaluation/evaluated_step"] = float(step)
        variance = _native_target_variance(model, inputs, directory / "mcc_samples.pt")
        native["test/target_variance"] = variance
        native["test/normalized_prediction_mse"] = native["test/prediction_mse"] / max(
            variance, 1e-12
        )
        refreshed = refresh_evaluation(
            model, inputs.evaluation, ("calibration", "validation", "test")
        )
        candidate = copy.deepcopy(model).double().eval()
        codes = {
            name: encode_codes(candidate, refreshed[name])
            for name in ("calibration", "validation", "test")
        }
        retention = parameter_readouts(codes, refreshed)
        response, raw, vectors = physical_response(
            candidate,
            refreshed,
            split="test",
            final=True,
            chunk_size=chunk_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        gate = classify(retention["selected_test_r2"], response)
        diagnostic = {
            "retention": retention,
            "response": response,
            "state": state_metrics(refreshed, "test"),
        }
        for row in gate["per_parameter"]:
            prefix = f"predictive_use/diagnostic_test/theta_{row['parameter']}"
            native[prefix + "/whole_code_r2"] = row["whole_code_r2"]
            if row["response_error_upper95"] is not None:
                native[prefix + "/response_error_upper95"] = row["response_error_upper95"]
            if row["benefit_lower95"] is not None:
                native[prefix + "/benefit_lower95"] = row["benefit_lower95"]
        native["diagnostic_test/revised_admission_pass"] = float(gate["revised_admission_pass"])
        native["diagnostic_test/original_retention_and_use_gate_pass"] = float(
            gate["original_retention_and_use_gate_pass"]
        )
        report = {
            "role": "final_fixed_budget",
            "step": step,
            "condition": condition,
            "native": native,
            "diagnostic": diagnostic,
            "admission": gate,
            "cohorts": cohorts(inputs),
            "checkpoint_sha256": checkpoint_sha,
            "scope": (
                "Exploratory comparison on two historically reused cohorts; no fresh "
                "confirmation or bias calibration."
            ),
        }
        atomic_save(
            directory / "diagnostic_measurements.pt",
            {
                "identity": identity,
                "response": raw,
                "vectors": vectors,
                "test_codes": codes["test"],
                "retention_group_mse": torch.tensor(
                    retention["selected_test_group_mse_by_parameter"], dtype=torch.float64
                ),
            },
        )
        write_json(directory / "diagnostic_final.json", report)
        write_json(directory / "metrics.json", native)
        native_protocol = read_json(directory / "protocol.json")
        native_protocol["comparison_cohorts"] = report["cohorts"]
        write_json(directory / "protocol.json", native_protocol)
        write_json(
            directory / "evaluated_checkpoint.json",
            {
                "run_identity": identity,
                "checkpoint_path": "last.pt",
                "checkpoint_sha256": checkpoint_sha,
                "mcc_samples_sha256": sha(directory / "mcc_samples.pt"),
                "evaluated_step": step,
                "selection_metric": "fixed_budget",
                "selection_score": None,
                "selected_step": step,
            },
        )
    if sha(checkpoint) != checkpoint_sha:
        raise RuntimeError("Final checkpoint changed during evaluation")
    return report, [
        "evaluation/" + name
        for name in (
            "metrics.json",
            "protocol.json",
            "mcc_samples.pt",
            "evaluation_references.pt",
            "evaluated_checkpoint.json",
            "diagnostic_final.json",
            "diagnostic_measurements.pt",
        )
    ]
