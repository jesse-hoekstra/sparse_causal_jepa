"""Twelve-code inference with unchanged six-physical-factor recovery and use criteria."""

import copy
import math
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.joint_acquisition.baseline import (
    DiagnosticSelection,
    MatchedBatch,
    MatchedPredictions,
    response_report,
    selected_split,
)
from diagnostics.mlp_acquisition_control.evaluation import (
    measurement_metrics as measurement_metrics,
)
from diagnostics.multiquery_control.data import EvaluationData, TensorSplit
from diagnostics.multiquery_control.evaluation import (
    _Candidate,  # pyright: ignore[reportPrivateUsage]
    _matrix,  # pyright: ignore[reportPrivateUsage]
    _ridge_candidates,  # pyright: ignore[reportPrivateUsage]
    _select_family,  # pyright: ignore[reportPrivateUsage]
    _vector,  # pyright: ignore[reportPrivateUsage]
    encode_codes,
)
from diagnostics.oracle_code_control.evaluation import matched_batch
from diagnostics.signal_path.geometry import predict_rollout
from scjepa.models.model import SCJEPA
from scjepa.training.rng import preserve_rng


def cubic_features(x: Tensor) -> Tensor:
    """All nonconstant monomials through degree three:454 at twelve inputs."""
    dimension = x.shape[1]
    columns = [x]
    columns += [(x[:, i] * x[:, j])[:, None] for i in range(dimension) for j in range(i, dimension)]
    columns += [
        (x[:, i] * x[:, j] * x[:, k])[:, None]
        for i in range(dimension)
        for j in range(i, dimension)
        for k in range(j, dimension)
    ]
    return torch.cat(columns, 1)


def parameter_readouts(codes: dict[str, Tensor], splits: EvaluationData) -> dict[str, Any]:
    """Fit linear/cubic/RFF probes on calibration; select only on validation."""
    if any(value.ndim != 3 or value.shape[-1] != 12 for value in codes.values()):
        raise ValueError("Overcomplete readouts require individual-context12D codes")
    ordered = ("calibration", "validation", "test")
    features = [codes[key].flatten(0, 1).double() for key in ordered]
    labels = [
        cast(TensorSplit, splits[key])["theta"].repeat_interleave(codes[key].shape[1], 0).double()
        for key in ordered
    ]
    mu = features[0].mean(0)
    eigenvalues, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(torch.cov(features[0].T)))  # pyright: ignore[reportUnknownMemberType]
    floor = max(float(eigenvalues.max()) * 1e-12, 1e-20)
    transform = vectors / eigenvalues.clamp_min(floor).sqrt()
    features = [(part - mu) @ transform for part in features]
    test_theta = cast(TensorSplit, splits["test"])["theta"].double()
    views = codes["test"].shape[1]
    families: dict[str, dict[str, Any]] = {
        "linear": _select_family(_ridge_candidates(features, labels), labels, test_theta, views)
    }
    polynomial = [cubic_features(x) for x in features]
    families["cubic"] = _select_family(
        _ridge_candidates(polynomial, labels), labels, test_theta, views
    )
    generator = torch.Generator().manual_seed(20261026072)
    omega = torch.randn(12, 256, dtype=torch.float64, generator=generator)
    phase = 2 * math.pi * torch.rand(256, dtype=torch.float64, generator=generator)
    candidates: list[_Candidate] = []
    for width in (0.5, 1.0, 2.0, 4.0):
        random_features = [math.sqrt(2) * torch.cos(x @ omega / width + phase) for x in features]
        candidates.extend(_ridge_candidates(random_features, labels, width))
    families["rbf_256_random_features"] = _select_family(candidates, labels, test_theta, views)
    names = tuple(families)
    best = [
        max(names, key=lambda key, j=j: float(families[key]["validation_r2"][j])) for j in range(6)
    ]
    scores = [float(families[best[j]]["test_r2"][j]) for j in range(6)]
    selected_errors = torch.stack(
        [torch.as_tensor(families[best[j]]["test_group_mse_by_parameter"])[:, j] for j in range(6)],
        1,
    )
    return {
        "families": families,
        "selected_methods_per_parameter": best,
        "selected_test_r2": scores,
        "selected_test_r2_mean": sum(scores) / 6,
        "selected_test_r2_minimum": min(scores),
        "selected_test_group_mse_by_parameter": _matrix(selected_errors),
        "calibration_code_covariance_eigenvalues": _vector(eigenvalues),
        "protocol": (
            "All twelve code coordinates of one context; calibration-only whitening/fit, "
            "validation-only family/hyperparameter selection per theta; independent test groups"
        ),
        "limits": (
            "A finite probe cannot prove absence or completeness of information; "
            "mixed parameter codes are allowed"
        ),
    }


def predict_matched(model: SCJEPA, batch: MatchedBatch) -> MatchedPredictions:
    """Separate12-code axes from six physical interventions and six state outputs."""
    if model.predictor.param_dim != 12 or model.parameter_encoder.param_dim != 12:
        raise ValueError("Expected12D encoder and predictor codes")
    batch = batch.to(next(model.parameters()))
    groups, donors, queries = batch.history.shape[:3]
    codes = cast(
        Tensor,
        model.parameter_encoder(
            batch.donor_states.flatten(0, 1), batch.donor_actions.flatten(0, 1)
        ),
    ).reshape(groups, donors, 12)
    changed_codes = cast(
        Tensor,
        model.parameter_encoder(
            batch.changed_donor_states.flatten(0, 2),
            batch.donor_actions[:, :, None].expand(-1, -1, 6, -1, -1).flatten(0, 2),
        ),
    ).reshape(groups, donors, 6, 12)
    predictions = predict_rollout(
        model,
        batch.history.flatten(0, 2),
        codes[:, :, None].expand(-1, -1, queries, -1).flatten(0, 2),
        batch.query_actions.flatten(0, 2),
    ).reshape(groups, donors, queries, 2, 6)
    changed_predictions = predict_rollout(
        model,
        batch.history[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
        changed_codes[:, :, :, None].expand(-1, -1, -1, queries, -1).flatten(0, 3),
        batch.query_actions[:, :, None].expand(-1, -1, 6, -1, -1, -1).flatten(0, 3),
    ).reshape(groups, donors, 6, queries, 2, 6)
    return MatchedPredictions(predictions, changed_predictions, codes, changed_codes)


@torch.no_grad()
def validation_retention(
    model: SCJEPA, data: EvaluationData, selection: DiagnosticSelection
) -> dict[str, Any]:
    """Reuse whole-code probe families while reporting diagnostic validation only.

    The historical evaluator requires three split slots. Its scoring slot is
    deliberately supplied the same selected validation data, never final-test
    data. All outputs are relabelled, and their selection optimism is explicit.
    """
    calibration = selected_split(data, selection, "calibration")
    validation = selected_split(data, selection, "validation")
    cal_codes = encode_codes(model, calibration)
    val_codes = encode_codes(model, validation)
    proxy: EvaluationData = {
        "calibration": calibration,
        "validation": validation,
        "test": validation,
    }
    result = parameter_readouts(
        {"calibration": cal_codes, "validation": val_codes, "test": val_codes}, proxy
    )
    families: dict[str, Any] = {}
    for name, family in result["families"].items():
        families[name] = {
            "validation_r2": family["validation_r2"],
            "selected_hyperparameters_per_parameter": family[
                "selected_hyperparameters_per_parameter"
            ],
            "validation_group_mse_by_parameter": family["test_group_mse_by_parameter"],
        }
    return {
        "families": families,
        "selected_methods_per_parameter": result["selected_methods_per_parameter"],
        "selected_validation_r2": result["selected_test_r2"],
        "validation_group_mse_by_parameter": result["selected_test_group_mse_by_parameter"],
        "calibration_code_covariance_eigenvalues": result[
            "calibration_code_covariance_eigenvalues"
        ],
        "calibration_code_std": float(
            cal_codes.double().flatten(0, 1).var(0, correction=0).mean().sqrt()
        ),
        "contexts_per_group": int(cal_codes.shape[1]),
        "protocol": (
            "Fit full individual-context code probes on selected calibration groups; select and "
            "score on selected validation groups. No context averaging. Selection-optimistic "
            "diagnostic scores, not independent final test evidence."
        ),
    }


def assess_model(
    model: SCJEPA,
    data: EvaluationData,
    selection: DiagnosticSelection,
    *,
    bootstrap_samples: int,
    seed: int,
    final: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Evaluate inferred codes only; exact theta serves solely to generate counterfactuals."""
    with preserve_rng(next(model.parameters()).device), torch.no_grad():
        candidate = copy.deepcopy(model).double().eval()
        split = "test" if final else "validation"
        # The shared builder explicitly handles test; its supplied-code outputs are discarded.
        batch, _, _ = matched_batch(
            candidate, data, split, selection.validation_indices, final=final
        )
        predictions = predict_matched(candidate, batch)
        response, raw = response_report(
            predictions.predictions,
            predictions.changed_predictions,
            batch,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        if final:
            names = ("calibration", "validation", "test")
            for first, second in (("calibration", "test"), ("validation", "test")):
                if bool(
                    (data[first]["theta"][:, None] == data[second]["theta"][None]).all(-1).any()
                ):
                    raise ValueError("Final-test groups overlap probe-fitting groups")
            retention = parameter_readouts(
                {name: encode_codes(candidate, data[name]) for name in names},
                {name: data[name] for name in names},
            )
            response["interpretation"] = "Fixed-budget independent-test inferred-code response"
        else:
            retention = validation_retention(candidate, data, selection)
            retention.pop("families")
        return {
            "response": response,
            "retention": retention,
            "overcomplete_architecture_diagnostic": True,
            "code_dimension": 12,
            "physical_parameter_dimension": 6,
        }, raw
