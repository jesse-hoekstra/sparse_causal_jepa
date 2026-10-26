"""Held-out information, consistency and prediction checks for frozen states.

All readouts take one context's complete code. They permit mixed parameter codes
and never require parameter-coordinate disentanglement or average input codes.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.multiquery_control.data import (
    EvaluationData,
    TensorSplit,
    encode_states,
    module_fingerprint,
)
from diagnostics.multiquery_control.forward import cached_multiquery_forward
from scjepa.data import simplified_gaussian as dgp
from scjepa.data.controlled_vector import VectorObservationMap
from scjepa.models.model import SCJEPA
from scjepa.training.loop import ObjectiveConfig, TrainingObjective
from scjepa.training.rng import preserve_rng


def _r2(truth: Tensor, prediction: Tensor) -> Tensor:
    return 1 - (truth - prediction).square().sum(0) / (
        (truth - truth.mean(0)).square().sum(0).clamp_min(1e-20)
    )


def _vector(value: Tensor) -> list[float]:
    return cast(list[float], value.tolist())  # pyright: ignore[reportUnknownMemberType]


def _matrix(value: Tensor) -> list[list[float]]:
    return cast(list[list[float]], value.tolist())  # pyright: ignore[reportUnknownMemberType]


@torch.no_grad()
def encode_codes(model: SCJEPA, split: TensorSplit, batch_size: int = 128) -> Tensor:
    """Infer codes separately from each context; return grouped CPU tensors."""
    states, actions = split["states_online"], split["actions"]
    groups, views = states.shape[:2]
    flat_states, flat_actions = states.flatten(0, 1), actions.flatten(0, 1)
    reference = next(model.parameters())
    pieces: list[Tensor] = []
    for start in range(0, len(flat_states), batch_size):
        code = model.parameter_encoder(
            flat_states[start : start + batch_size, : model.context_len].to(reference),
            flat_actions[start : start + batch_size, : model.context_len - 1].to(reference),
        )
        pieces.append(code.detach().cpu())
    return torch.cat(pieces).reshape(groups, views, -1)


@torch.no_grad()
def _predict(
    model: SCJEPA, history: Tensor, codes: Tensor, actions: Tensor, batch_size: int
) -> Tensor:
    reference = next(model.parameters())
    pieces: list[Tensor] = []
    for start in range(0, len(codes), batch_size):
        h, c, a = (
            part[start : start + batch_size].to(reference) for part in (history, codes, actions)
        )
        horizons: list[Tensor] = []
        for index in range(a.shape[1]):
            prediction = model.predictor(h, c, a[:, index : index + 1]).prediction
            horizons.append(prediction.detach().cpu())
            h = torch.cat((h[:, 1:], prediction[:, None]), dim=1)
        pieces.append(torch.stack(horizons, 1))
    return torch.cat(pieces)


def _variance(codes: Tensor) -> dict[str, float | None]:
    values = codes.double()
    within = float(values.var(1, correction=1).mean())
    between = float(values.mean(1).var(0, correction=1).mean()) - within / codes.shape[1]
    total = within + between
    return {
        "within": within,
        "between": between,
        "total": total,
        "within_over_total": within / total if total > 0 else None,
        "raw_consistency": 2 * within,
    }


@dataclass(frozen=True)
class _Candidate:
    hyperparameters: dict[str, float]
    validation_r2: Tensor
    test_prediction: Tensor


def _ridge_candidates(
    features: list[Tensor], labels: list[Tensor], width: float | None = None
) -> list[_Candidate]:
    center, scale = features[0].mean(0), features[0].std(0).clamp_min(1e-10)
    x = [torch.cat(((part - center) / scale, part.new_ones(len(part), 1)), 1) for part in features]
    covariance, rhs = x[0].T @ x[0] / len(x[0]), x[0].T @ labels[0] / len(x[0])
    identity = torch.eye(covariance.shape[0], dtype=torch.float64)
    identity[-1, -1] = 0
    rows: list[_Candidate] = []
    for strength in (1e-5, 1e-3, 0.1, 1.0, 10.0):
        weights = cast(Tensor, torch.linalg.solve(covariance + strength * identity, rhs))  # pyright: ignore[reportUnknownMemberType]
        settings = {"ridge": strength}
        if width is not None:
            settings["width"] = width
        rows.append(_Candidate(settings, _r2(labels[1], x[1] @ weights), x[2] @ weights))
    return rows


def _select_family(
    rows: list[_Candidate], labels: list[Tensor], test_theta: Tensor, views: int
) -> dict[str, Any]:
    selected = torch.stack([row.validation_r2 for row in rows]).argmax(0)
    predictions = torch.stack([rows[int(selected[j])].test_prediction[:, j] for j in range(6)], 1)
    errors = (predictions.reshape(len(test_theta), views, 6) - test_theta[:, None]).square().mean(1)
    return {
        "test_r2": _vector(_r2(labels[2], predictions)),
        "validation_r2": [float(rows[int(selected[j])].validation_r2[j]) for j in range(6)],
        "selected_hyperparameters_per_parameter": [rows[int(i)].hyperparameters for i in selected],
        "test_group_mse_by_parameter": _matrix(errors),
    }


def parameter_readouts(codes: dict[str, Tensor], splits: EvaluationData) -> dict[str, Any]:
    """Fit linear/cubic/RFF probes on calibration; select only on validation."""
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
    polynomial: list[Tensor] = []
    for x in features:
        columns = [x]
        columns += [(x[:, i] * x[:, j])[:, None] for i in range(6) for j in range(i, 6)]
        columns += [
            (x[:, i] * x[:, j] * x[:, k])[:, None]
            for i in range(6)
            for j in range(i, 6)
            for k in range(j, 6)
        ]
        polynomial.append(torch.cat(columns, 1))
    families["cubic"] = _select_family(
        _ridge_candidates(polynomial, labels), labels, test_theta, views
    )
    generator = torch.Generator().manual_seed(20261026072)
    omega = torch.randn(6, 256, dtype=torch.float64, generator=generator)
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
            "All six code coordinates of one context; calibration-only whitening/fit, "
            "validation-only family/hyperparameter selection per theta; independent test groups"
        ),
        "limits": (
            "A finite probe cannot prove absence or completeness of information; "
            "mixed parameter codes are allowed"
        ),
    }


def _linear_progress(
    calibration: Tensor, validation: Tensor, calibration_theta: Tensor, validation_theta: Tensor
) -> list[float]:
    x, y = (
        calibration.flatten(0, 1).double(),
        calibration_theta.repeat_interleave(calibration.shape[1], 0).double(),
    )
    center, scale = x.mean(0), x.std(0).clamp_min(1e-10)
    design = torch.cat(((x - center) / scale, x.new_ones(len(x), 1)), 1)
    weights = cast(Tensor, torch.linalg.lstsq(design, y).solution)  # pyright: ignore[reportUnknownMemberType]
    test = validation.flatten(0, 1).double()
    prediction = torch.cat(((test - center) / scale, test.new_ones(len(test), 1)), 1) @ weights
    return _vector(
        _r2(validation_theta.repeat_interleave(validation.shape[1], 0).double(), prediction)
    )


@torch.no_grad()
def _common_objective(
    model: SCJEPA, split: TensorSplit, config: ObjectiveConfig, batch_size: int, final: bool
) -> dict[str, Any]:
    common = copy.deepcopy(config)
    common.parameter_consistency_weight = 1.0
    objective = TrainingObjective(common).eval()
    n = len(split["theta"])
    if n % batch_size:
        raise ValueError("Evaluation group count must divide into complete objective batches")
    reference = next(model.parameters())
    seeds = (43207, 43208, 43209, 43210) if final else (43207,)
    records: list[dict[str, float]] = []
    group_errors: list[Tensor] = []
    with preserve_rng(torch.device("cpu")):
        for projection_seed in seeds:
            torch.default_generator.manual_seed(projection_seed)
            totals: dict[str, float] = {}
            for start in range(0, n, batch_size):
                batch = {
                    "online": split["states_online"][start : start + batch_size].to(reference),
                    "target": split["states_target"][start : start + batch_size].to(reference),
                    "actions": split["actions"][start : start + batch_size].to(reference),
                }
                output = cached_multiquery_forward(model, batch, queries=9 if final else 1)
                total, terms = objective(output)
                if projection_seed == seeds[0]:
                    group_errors.append(
                        (output.predictions - output.targets).square().flatten(1).mean(1).cpu()
                    )
                consistency = terms["parameter_consistency"]
                for key, value in {
                    **terms,
                    "total_at_lambda1": total,
                    "total_at_lambda0": total - consistency,
                    "total_at_training_lambda": total
                    + (config.parameter_consistency_weight - 1) * consistency,
                }.items():
                    totals[key] = totals.get(key, 0.0) + float(value) * batch_size / n
            records.append(totals)
    mean = {key: sum(row[key] for row in records) / len(records) for key in records[0]}
    return {
        "mean": mean,
        "by_projection_seed": [
            dict(seed=seed, **row) for seed, row in zip(seeds, records, strict=True)
        ],
        "queries_per_code": 9 if final else 1,
        "batch_groups": batch_size,
        "group_two_step_mse": _vector(torch.cat(group_errors)),
        "protocol": (
            "Full objective including frozen-state SIGReg; original code tensor enters "
            "regularizers once; identical projection seeds across arms"
        ),
    }


@torch.no_grad()
def _shuffle_metrics(
    model: SCJEPA, split: TensorSplit, codes: Tensor, batch_size: int
) -> dict[str, Any]:
    history = split["states_online"][:, :, 9:10].flatten(0, 1)
    actions = split["actions"][:, :, 9:11].flatten(0, 1)
    target = split["states_target"][:, :, 10:12]
    groups, views = codes.shape[:2]
    original: list[Tensor] = []
    shuffled: list[Tensor] = []
    for shift in range(1, 10):
        for donor, collection in (
            (codes.roll(shift, 1), original),
            (codes.roll(1, 0).roll(shift, 1), shuffled),
        ):
            prediction = _predict(model, history, donor.flatten(0, 1), actions, batch_size)
            collection.append(
                (prediction.reshape(groups, views, 2, 6) - target).square().mean((1, 2, 3))
            )
    base, changed = torch.stack(original).mean(0), torch.stack(shuffled).mean(0)
    return {
        "two_step_mse": float(base.mean()),
        "shuffled_two_step_mse": float(changed.mean()),
        "shuffle_delta_mse": float((changed - base).mean()),
        "group_shuffle_delta_mse": _vector(changed - base),
        "queries_per_code": 9,
        "protocol": (
            "Shuffle all donor codes together across independent theta groups, keep recipient "
            "states/actions fixed; average all90 ordered source/query pairs"
        ),
    }


@torch.no_grad()
def _parameter_interventions(
    model: SCJEPA, data: EvaluationData, codes: Tensor, batch_size: int
) -> dict[str, Any]:
    """Change one physical theta, regenerate its donor and fix recipient state/action."""
    split = cast(TensorSplit, data["test"])
    groups = min(128, len(split["theta"]))
    subset = {key: value[:groups] for key, value in split.items()}
    matrix = cast(Tensor, data["metadata"]["observation_matrix"])
    observation_map = VectorObservationMap(matrix=matrix, dimension=6)
    theta, states, true_actions = subset["theta"], subset["true_states"], subset["true_actions"]
    views = subset["actions"].shape[1]
    history = subset["states_online"][:, :, 9:10].flatten(0, 1)
    actions = subset["actions"][:, :, 9:10].flatten(0, 1)
    original_codes = codes[:groups].roll(1, 1).flatten(0, 1)
    unchanged_prediction = _predict(model, history, original_codes, actions, batch_size)[:, 0]
    indices = torch.randint(
        groups, (2000, groups), generator=torch.Generator().manual_seed(202610261411)
    )
    rows: dict[str, dict[str, Any]] = {}
    for parameter in range(6):
        changed_parameters: list[Tensor] = []
        predictions: list[Tensor] = []
        exact_targets: list[Tensor] = []
        for sign in (1, -1):
            changed = theta.clone()
            changed[:, parameter] = (changed[:, parameter] + sign * 0.1).clamp(
                -dgp.PARAMETER_BOUND, dgp.PARAMETER_BOUND
            )
            donor_states = dgp.simulate(states[:, :, 0], changed[:, None], true_actions[:, :, :9])
            donor_online, _ = encode_states(
                model, observation_map.forward(donor_states), batch_size
            )
            donor = {"states_online": donor_online, "actions": subset["actions"][:, :, :9]}
            changed_codes = encode_codes(model, donor, batch_size).roll(1, 1).flatten(0, 1)
            prediction = _predict(model, history, changed_codes, actions, batch_size)[:, 0]
            exact_states = dgp.transition(states[:, :, 9], changed[:, None], true_actions[:, :, 9])
            _, exact_target = encode_states(
                model, observation_map.forward(exact_states)[:, :, None], batch_size
            )
            changed_parameters.append(changed)
            predictions.append(prediction)
            exact_targets.append(exact_target[:, :, 0].flatten(0, 1))
        denominator = (
            changed_parameters[0][:, parameter] - changed_parameters[1][:, parameter]
        ).repeat_interleave(views)[:, None]
        predicted = (predictions[0].double() - predictions[1].double()) / denominator
        exact = (exact_targets[0].double() - exact_targets[1].double()) / denominator
        predicted_energy = predicted.square().sum(1).reshape(groups, views).mean(1)
        exact_energy = exact.square().sum(1).reshape(groups, views).mean(1)
        product = (predicted * exact).sum(1).reshape(groups, views).mean(1)
        changed_error = torch.stack(
            [(p - y).square().mean(1) for p, y in zip(predictions, exact_targets, strict=True)]
        ).mean(0)
        original_error = torch.stack(
            [(unchanged_prediction - y).square().mean(1) for y in exact_targets]
        ).mean(0)
        benefit = (original_error - changed_error).reshape(groups, views).mean(1).double()
        interval = (
            benefit[indices].mean(1).quantile(torch.tensor([0.025, 0.975], dtype=torch.float64))
        )
        rows[f"theta_{parameter + 1}"] = {
            "response_cosine": float(
                product.mean()
                / (predicted_energy.mean() * exact_energy.mean()).sqrt().clamp_min(1e-30)
            ),
            "response_energy_ratio": float(
                predicted_energy.mean() / exact_energy.mean().clamp_min(1e-30)
            ),
            "counterfactual_mse_improvement": float(benefit.mean()),
            "counterfactual_mse_improvement_ci95": _vector(interval),
            "group_counterfactual_mse_improvement": _vector(benefit),
            "group_predicted_response_energy": _vector(predicted_energy),
            "group_exact_response_energy": _vector(exact_energy),
            "group_response_inner_product": _vector(product),
        }
    return {
        "groups": groups,
        "displacement": 0.1,
        "by_parameter": rows,
        "protocol": (
            "Perturb each theta by +/-0.1 within support; regenerate donor K10 from the same "
            "initial state/actions; hold recipient latest true/encoded state and action fixed; "
            "compare one-step predictions with exact dynamics in frozen target coordinates"
        ),
        "limits": (
            "Finite local responses with shift1 donor; confidence intervals resample theta "
            "groups conditional on this checkpoint. Inspect direction, response energy and "
            "absolute benefit together: tiny positive benefit alone is not meaningful use."
        ),
    }


@torch.no_grad()
def evaluate(
    model: SCJEPA,
    evaluation_data: EvaluationData,
    objective_config: ObjectiveConfig,
    *,
    batch_size: int = 16,
    final: bool = False,
) -> dict[str, Any]:
    """Return JSON-ready validation monitoring or fixed-endpoint test diagnostics.

    The caller should pass an evaluation copy on CPU. Probes run on CPU in float64
    regardless; state hashes reject mismatched caches. No model update occurs.
    """
    if batch_size < 2:
        raise ValueError("SIGReg evaluation requires at least two independent groups")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError(
            "Pass a CPU evaluation copy to keep evaluation separate from CUDA training"
        )
    metadata = cast(dict[str, Any], evaluation_data["metadata"])
    for key, module in (
        ("state_online_sha256", model.online_encoder),
        ("state_target_sha256", model.target_encoder),
    ):
        if metadata[key] != module_fingerprint(module):
            raise ValueError(f"Frozen state cache mismatch: {key}")
    was_training = model.training
    model.eval()
    try:
        names = ("calibration", "validation", "test") if final else ("calibration", "validation")
        splits = {key: cast(TensorSplit, evaluation_data[key]) for key in names}
        codes = {
            key: encode_codes(model, split, max(128, batch_size)) for key, split in splits.items()
        }
        key = "test" if final else "validation"
        objective = _common_objective(model, splits[key], objective_config, batch_size, final)
        linear = _linear_progress(
            codes["calibration"], codes[key], splits["calibration"]["theta"], splits[key]["theta"]
        )
        result: dict[str, Any] = {
            "split": key,
            "groups": len(codes[key]),
            "queries_per_code": 9 if final else 1,
            "prediction_mse": objective["mean"]["prediction"],
            "total_at_lambda0": objective["mean"]["total_at_lambda0"],
            "total_at_lambda1": objective["mean"]["total_at_lambda1"],
            "raw_consistency": objective["mean"]["parameter_consistency"],
            "parameter_code_variance": _variance(codes[key]),
            "linear_r2": linear,
            "linear_r2_mean": sum(linear) / 6,
            "linear_r2_minimum": min(linear),
            "objective": objective,
        }
        if final:
            result["parameter_readouts"] = parameter_readouts(codes, evaluation_data)
            shuffle = _shuffle_metrics(model, splits["test"], codes["test"], max(128, batch_size))
            if abs(float(shuffle["two_step_mse"]) - float(result["prediction_mse"])) > 2e-6:
                raise RuntimeError("All-query objective and shuffle predictor disagree")
            result["parameter_use_shuffle"] = shuffle
            result["true_parameter_interventions"] = _parameter_interventions(
                model, evaluation_data, codes["test"], max(128, batch_size)
            )
            result["limits"] = [
                "Probe recovery, code consistency and predictive use are separate properties.",
                "Whole-code shuffling tests aggregate use, not use of each physical parameter.",
                "Low finite-probe R2 does not prove information is absent; high average R2 "
                "does not establish recovery of all six parameters.",
                "Frozen state encoders mean this experiment cannot change state MCC "
                "or test sparse graph recovery.",
            ]
        return result
    finally:
        model.train(was_training)
        model.online_encoder.eval()
        model.target_encoder.eval()
