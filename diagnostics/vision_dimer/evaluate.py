"""Calibration-fitted evaluation of states, codes, forecasts and graphs.

Every alignment and probe is fitted on calibration and scored on the evaluated
split (validation during development; test once at the end). Definitions follow
the Experiment 1 paper: Pearson MCC with a calibration-fitted one-to-one match,
block purity from the best single true block, whole-state ridge retention,
single-context complete-code parameter retention, forecast NMSE and SHD.

Rendering makes the learned chart a smooth nonlinear function of the state, so
the Pearson/linear scores (affine identification, as in Experiment 1's linear
observation map) are complemented by the theorem-compatible ones for
identification up to permutation and elementwise diffeomorphisms: Spearman MCC,
monotone coordinatewise recovery and nearest-neighbour block purity.
"""

from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy
import torch
from scipy.optimize import (  # pyright: ignore[reportUnknownVariableType]
    isotonic_regression,  # pyright: ignore[reportUnknownVariableType]
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)
from scipy.spatial import cKDTree  # pyright: ignore[reportUnknownVariableType]
from torch import Tensor, nn

from diagnostics.masked_mlp_learned.run import (
    _design,  # pyright: ignore[reportPrivateUsage]
    _r2,  # pyright: ignore[reportPrivateUsage]
    _ridge,  # pyright: ignore[reportPrivateUsage]
)
from diagnostics.vision_dimer import dgp
from scjepa.eval.recovery import fit_coordinate_match, mean_correlation_coefficient
from scjepa.models.model import SCJEPA, ModelOutput

BLOCKS = dgp.BLOCKS
D = dgp.STATE_DIM


class _ShuffledCodes(nn.Module):
    """Give every group another group's codes (predictive-use diagnostic)."""

    def __init__(self, inner: nn.Module, views: int) -> None:
        super().__init__()
        self.inner, self.views = inner, views
        self.param_dim = cast(Any, inner).param_dim
        self.state_dim = cast(Any, inner).state_dim
        self.action_dim = cast(Any, inner).action_dim
        self.max_history = cast(Any, inner).max_history

    def forward(self, states: Tensor, actions: Tensor | None = None) -> Tensor:
        codes = cast(Tensor, self.inner(states, actions))
        return codes.reshape(-1, self.views, codes.shape[-1]).roll(1, 0).flatten(0, 1)


@contextmanager
def shuffled_codes(model: SCJEPA, views: int) -> Generator[None]:
    """Temporarily route codes across groups."""
    original = model.parameter_encoder
    model.parameter_encoder = cast(Any, _ShuffledCodes(original, views))
    try:
        yield
    finally:
        model.parameter_encoder = original


@torch.no_grad()
def collect(
    model: SCJEPA, batches: Iterator[dict[str, Tensor]], *, shuffle: bool = False
) -> dict[str, Tensor]:
    """Encodings, single-context codes and two-step forecasts, deterministic gates."""
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    parts: dict[str, list[Tensor]] = {k: [] for k in ("states", "codes", "predictions", "targets")}
    try:
        for batch in batches:
            x, a = batch["observations"], batch["actions"]
            if shuffle:
                with shuffled_codes(model, x.shape[1]):
                    output = cast(ModelOutput, model(x, a))
            else:
                output = cast(ModelOutput, model(x, a))
            if "targets" in batch:  # cached frozen-encoder replay supplies EMA targets
                output = output._replace(targets=batch["targets"])
            for key, value in (
                ("states", output.online_states),
                ("codes", output.parameters),
                ("predictions", output.predictions),
                ("targets", output.targets),
            ):
                parts[key].append(value.detach().double().cpu())
    finally:
        for module, training in modes:
            module.training = training
    return {key: torch.cat(values) for key, values in parts.items()}


def prediction_mse(samples: dict[str, Tensor]) -> float:
    """Mean squared two-step forecast error in learned coordinates."""
    return float((samples["predictions"] - samples["targets"]).square().mean())


def _standardize(train: Tensor, *others: Tensor) -> list[Tensor]:
    mean, std = train.mean(0), train.std(0).clamp_min(1e-12)
    return [(value - mean) / std for value in (train, *others)]


def _fit_predict(train_x: Tensor, train_y: Tensor, test_x: Tensor, family: str) -> Tensor:
    coefficients = _ridge(_design(train_x, family), train_y)
    return _design(test_x, family) @ coefficients


def _monotone_recovery_r2(cal_x: Tensor, cal_z: Tensor, x: Tensor, z: Tensor) -> float:
    """Validation R2 of a calibration-fitted monotone map from one learned to one true coordinate.

    Isotonic (pool-adjacent-violators) regression in the direction of the calibration
    rank correlation; validation inputs use linear interpolation between fitted
    calibration inputs and constant extension beyond their range. A univariate cubic
    underfits inverses of saturating maps, which elementwise identification permits.
    """
    order = cal_x.argsort()
    xs, zs = cal_x[order].numpy(), cal_z[order].numpy()
    ranks = torch.stack((cal_x.argsort().argsort(), cal_z.argsort().argsort())).double()
    increasing = bool(torch.corrcoef(ranks)[0, 1] >= 0)
    fitted = isotonic_regression(zs, increasing=increasing).x
    prediction = torch.tensor(numpy.interp(x.numpy(), xs, fitted), dtype=z.dtype)
    return float(_r2(prediction[:, None], z[:, None])[0])


def _knn_r2(
    train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor, k: int = 16
) -> Tensor:
    """Validation R2 of k-nearest-neighbour regression (calibration rows as neighbours)."""
    tree = cKDTree(train_x.numpy())
    _, index = tree.query(test_x.numpy(), k=k)
    prediction = train_y[torch.as_tensor(index, dtype=torch.long)].mean(1)
    return _r2(prediction, test_y)


def state_metrics(
    cal_states: Tensor, cal_truth: Tensor, states: Tensor, truth: Tensor
) -> dict[str, Any]:
    """MCC, Spearman MCC, block purity and whole-state retention."""
    cal_x, cal_z = cal_states.reshape(-1, D), cal_truth.reshape(-1, D)
    x, z = states.reshape(-1, D), truth.reshape(-1, D)
    match = fit_coordinate_match(cal_x, cal_z, method="pearson")
    mcc = mean_correlation_coefficient(match, x, z)
    spearman_match = fit_coordinate_match(cal_x, cal_z, method="spearman")
    rank_x = x.argsort(0).argsort(0).double()
    rank_z = z.argsort(0).argsort(0).double()
    spearman = mean_correlation_coefficient(spearman_match, rank_x, rank_z)
    corr = torch.corrcoef(torch.cat((x, z), 1).T)[:D, D:].abs()  # learned rows, true columns
    # CoordinateMatch.columns maps each TRUE index to its learned index.
    per_coordinate = [float(corr[match.columns[t], t]) for t in range(D)]
    sx, sx_test = _standardize(cal_x, x)
    sz, sz_test = _standardize(cal_z, z)
    retention = {
        family: [float(v) for v in _r2(_fit_predict(sx, sz, sx_test, family), sz_test)]
        for family in ("linear", "cubic")
    }
    purity: dict[str, list[float]] = {
        "linear": [
            max(
                float(
                    _r2(
                        _fit_predict(
                            sz[:, list(block)], sx[:, j : j + 1], sz_test[:, list(block)], "linear"
                        ),
                        sx_test[:, j : j + 1],
                    )[0]
                )
                for block in BLOCKS
            )
            for j in range(D)
        ]
    }
    knn = [_knn_r2(sz[:, list(block)], sx, sz_test[:, list(block)], sx_test) for block in BLOCKS]
    purity["knn"] = [max(float(scores[j]) for scores in knn) for j in range(D)]
    # Theorem-compatible scores (identification up to permutation and elementwise
    # diffeomorphisms): under the calibration Spearman assignment, the rank
    # correlation and a monotone recovery of each true coordinate from its one
    # matched learned coordinate. kNN block purity is the nonparametric analogue of
    # the linear purity above (a learned coordinate explained by one true block).
    rank_corr = torch.corrcoef(torch.cat((rank_x, rank_z), 1).T)[:D, D:].abs()
    per_coordinate_spearman = [float(rank_corr[c, t]) for t, c in enumerate(spearman_match.columns)]
    coordinatewise = [
        _monotone_recovery_r2(cal_x[:, c], cal_z[:, t], x[:, c], z[:, t])
        for t, c in enumerate(spearman_match.columns)
    ]
    return {
        "mcc": mcc,
        "spearman_mcc": spearman,
        "per_coordinate_correlation": per_coordinate,
        "per_coordinate_spearman": per_coordinate_spearman,
        "assignment": list(match.columns),
        "spearman_assignment": list(spearman_match.columns),
        "coordinatewise_r2_monotone": coordinatewise,
        "coordinatewise_r2_monotone_min": min(coordinatewise),
        "state_r2_linear": retention["linear"],
        "state_r2_cubic": retention["cubic"],
        "state_r2_min": min(retention["cubic"]),
        "block_purity_mean": sum(purity["linear"]) / D,
        "block_purity_min": min(purity["linear"]),
        "block_purity_knn_mean": sum(purity["knn"]) / D,
        "block_purity_knn_min": min(purity["knn"]),
    }


def parameter_retention(
    cal_codes: Tensor, cal_theta: Tensor, codes: Tensor, theta: Tensor
) -> dict[str, Any]:
    """Single-context complete-code retention.

    Fit on half the calibration groups, select linear/cubic per parameter on the
    other half, and score on the evaluated split.
    """
    groups, views = cal_codes.shape[:2]
    half = groups // 2

    def flatten(c: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        return c.reshape(-1, c.shape[-1]), t[:, None].expand(-1, c.shape[1], -1).reshape(
            -1, dgp.PARAM_DIM
        )

    fit_x, fit_y = flatten(cal_codes[:half], cal_theta[:half])
    sel_x, sel_y = flatten(cal_codes[half:], cal_theta[half:])
    test_x, test_y = flatten(codes, theta)
    fx, sx, tx = _standardize(fit_x, sel_x, test_x)
    scores: list[float] = []
    families: list[str] = []
    for p in range(dgp.PARAM_DIM):
        best_family = max(
            ("linear", "cubic"),
            key=lambda f: float(
                _r2(_fit_predict(fx, fit_y[:, p : p + 1], sx, f), sel_y[:, p : p + 1])[0]
            ),
        )
        families.append(best_family)
        scores.append(
            float(
                _r2(_fit_predict(fx, fit_y[:, p : p + 1], tx, best_family), test_y[:, p : p + 1])[0]
            )
        )
    del views
    return {"parameter_r2": scores, "parameter_r2_min": min(scores), "parameter_family": families}


def code_parameter_assignment(cal_codes: Tensor, cal_theta: Tensor) -> list[int]:
    """One-to-one calibration assignment of learned code coordinates to physical parameters.

    Each parameter is identified individually here (each has its own children), so a
    code coordinate is matched to the parameter whose cubic regression explains it best;
    ``assignment[j]`` is the parameter of learned code ``j``.
    """
    x = cal_codes.reshape(-1, cal_codes.shape[-1])
    y = cal_theta[:, None].expand(-1, cal_codes.shape[1], -1).reshape(-1, dgp.PARAM_DIM)
    (sx,) = _standardize(x)
    (sy,) = _standardize(y)
    scores = torch.zeros(x.shape[-1], dgp.PARAM_DIM, dtype=torch.float64)
    for parameter in range(dgp.PARAM_DIM):
        design = _design(sy[:, [parameter]], "cubic")
        scores[:, parameter] = _r2(design @ _ridge(design, sx), sx)
    rows, columns = linear_sum_assignment(-scores.numpy())
    assignment = [0] * x.shape[-1]
    for row, column in zip(rows, columns, strict=True):
        assignment[int(row)] = int(column)
    return assignment


def graph_metrics(
    model: SCJEPA, state_assignment: list[int], code_parameters: list[int]
) -> dict[str, Any]:
    """Thresholded gate graphs aligned to the truth; SHD = missing + extra edges.

    ``state_assignment[t]`` is the learned index of true coordinate ``t`` (the
    ``CoordinateMatch.columns`` convention stored as ``assignment`` by
    ``state_metrics``); ``code_parameters[j]`` is the parameter of learned code ``j``.
    """
    predictor = cast(Any, model.predictor)
    learned = {k: v.bool() for k, v in predictor.thresholded_graphs().items()}
    true = dgp.graphs()
    rows = torch.tensor(list(state_assignment))  # reordered row t = learned row of true t
    state = learned["state"][rows][:, rows]
    action = learned["action"][rows]
    parameter_true = true["parameter"][:, torch.tensor(list(code_parameters))]
    parameter = learned["parameter"][rows]
    return {
        "state_edges": int(state.sum()),
        "action_edges": int(action.sum()),
        "parameter_edges": int(parameter.sum()),
        "state_shd": int((state != true["state"]).sum()),
        "action_shd": int((action != true["action"]).sum()),
        "parameter_shd": int((parameter != parameter_true).sum()),
    }


def evaluate(
    model: SCJEPA,
    calibration: dict[str, Tensor],
    calibration_labels: dict[str, Tensor],
    split: dict[str, Tensor],
    split_labels: dict[str, Tensor],
    shuffled: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Complete metric set for collected samples of calibration and one scored split."""
    result: dict[str, Any] = {}
    result.update(
        state_metrics(
            calibration["states"],
            calibration_labels["states"],
            split["states"],
            split_labels["states"],
        )
    )
    result.update(
        parameter_retention(
            calibration["codes"], calibration_labels["theta"], split["codes"], split_labels["theta"]
        )
    )
    targets = split["targets"]
    result["prediction_mse"] = prediction_mse(split)
    result["nmse"] = result["prediction_mse"] / float(targets.reshape(-1, D).var(0).mean())
    if shuffled is not None:
        result["code_shuffle_ratio"] = prediction_mse(shuffled) / result["prediction_mse"]
    if calibration["codes"].shape[-1] == dgp.PARAM_DIM:
        parameters = code_parameter_assignment(calibration["codes"], calibration_labels["theta"])
        result["code_parameters"] = parameters
        result.update(graph_metrics(model, result["assignment"], parameters))
    return result


__all__ = [
    "code_parameter_assignment",
    "collect",
    "evaluate",
    "graph_metrics",
    "parameter_retention",
    "prediction_mse",
    "shuffled_codes",
    "state_metrics",
]
