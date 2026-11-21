"""Fit continuous linear code directions on a declared angle grid under three loss rules."""

import argparse
import hashlib
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .excitation import verify_head

type Array = NDArray[np.float64]
SPLITS = {"train": 2048, "calibration": 1024, "validation": 1024}
LEVELS = {"low": 0.5, "high": 2.0}
ANGLES = tuple(range(-90, 90, 5))
POLICIES = ("forecast", "additive", "constrained")
DONORS = 10


@dataclass(frozen=True)
class Observed:
    """Observable support responses, shared inert controls, recipient controls and outcomes."""

    first: Array
    second: Array
    inert: Array
    actions: Array
    outcomes: Array


def observe(seed: int, split: int, groups: int, noise: float) -> Observed:
    """Generate observed systems and pair draws across the two support-noise levels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, split]))
    theta = rng.standard_normal((groups, 2))
    inert = rng.standard_normal(groups)
    error = rng.standard_normal((groups, DONORS))
    actions = rng.standard_normal((groups, 4, 2))
    return Observed(
        theta[:, 0].copy(),
        theta[:, 1, None] + noise * error,
        inert,
        actions,
        np.einsum("gqa,ga->gq", actions, theta),
    )


@dataclass(frozen=True)
class Normalizer:
    """Center and scale the two candidate second-coordinate sources using TRAIN only."""

    center: Array
    scales: Array

    @classmethod
    def from_train(cls, data: Observed) -> "Normalizer":
        """Fit marginal scalings without using the hidden generating coefficients."""
        scales = np.array([data.second.std(), data.inert.std()])
        if np.min(scales) <= 0:
            raise ValueError("Calibration of a constant source is undefined")
        return cls(np.array([data.second.mean(), data.inert.mean()]), scales)


@dataclass(frozen=True)
class Moments:
    """Factorized action-code normal equations for regression and independent group errors."""

    input_gram: Array
    action_gram: Array
    input_action_y: Array
    y2: Array
    gram: Array
    within: Array
    total: Array

    @classmethod
    def from_data(cls, data: Observed, normalizer: Normalizer) -> "Moments":
        """Compute sufficient statistics for crossing every donor with every recipient control."""
        shape = data.second.shape
        x = np.stack(
            (
                np.ones(shape),
                np.broadcast_to(data.first[:, None], shape),
                (data.second - normalizer.center[0]) / normalizer.scales[0],
                np.broadcast_to(
                    ((data.inert - normalizer.center[1]) / normalizer.scales[1])[:, None], shape
                ),
            ),
            axis=-1,
        )
        xmean = x.mean(axis=1)
        xx = np.einsum("gri,grj->gij", x, x) / x.shape[1]
        uu = np.einsum("gqa,gqb->gab", data.actions, data.actions) / data.actions.shape[1]
        uy = np.einsum("gqa,gq->ga", data.actions, data.outcomes) / data.actions.shape[1]
        uxy = uy[:, :, None] * xmean[:, None, :]
        gram = np.einsum("gab,gij->aibj", uu, xx).reshape(8, 8) / len(x)
        within = xx.mean(axis=0) - (xmean.T @ xmean) / len(xmean)
        total = xx.mean(axis=0) - np.outer(xmean.mean(axis=0), xmean.mean(axis=0))
        return cls(xx, uu, uxy, (data.outcomes**2).mean(axis=1), gram, within, total)

    def errors(self, weights: Array) -> Array:
        """Evaluate mean squared error per system over all donors and query actions."""
        quadratic = np.einsum(
            "gab,ai,bj,gij->g", self.action_gram, weights, weights, self.input_gram
        )
        linear = np.einsum("ai,gai->g", weights, self.input_action_y)
        return quadratic - 2.0 * linear + self.y2


def loss_fit(train: Moments, angle: int) -> tuple[Array, float]:
    """Solve the exact TRAIN least-squares head conditional on one candidate encoder angle."""
    alpha = math.radians(angle)
    mix = np.array([0.0, 0.0, math.cos(alpha), math.sin(alpha)])
    transform = np.column_stack((np.eye(4)[:, :2], mix))
    basis = np.kron(np.eye(2), transform)
    rhs = train.input_action_y.mean(axis=0).reshape(-1)
    weights = basis @ np.linalg.solve(basis.T @ train.gram @ basis, basis.T @ rhs)
    d = 0.5 * max(0.0, float(mix @ train.within @ mix)) / float(mix @ train.total @ mix)
    return weights.reshape(2, 4), d


@dataclass(frozen=True)
class Candidate:
    """A shared grid fit and its TRAIN/CAL admission diagnostics."""

    angle: int
    weights: Array
    disagreement: float
    train_risk: float
    cal_risk: float


def choose(candidates: list[Candidate], scale: float) -> dict[str, Candidate]:
    """Apply three declared selection rules to the exact same fitted candidate heads."""
    forecast = min(candidates, key=lambda c: (c.train_risk, abs(c.angle), c.angle))
    additive = min(
        candidates, key=lambda c: (c.train_risk + scale * c.disagreement, abs(c.angle), c.angle)
    )
    eligible = [c for c in candidates if c.cal_risk <= forecast.cal_risk]
    constrained = min(eligible, key=lambda c: (c.disagreement, abs(c.angle), c.angle))
    return {"forecast": forecast, "additive": additive, "constrained": constrained}


def assess(data: dict[str, Observed]) -> tuple[dict[str, object], dict[str, bool]]:
    """Fit the objectives on train, constrain on calibration, and assess on validation."""
    scale_data = data["calibration"]
    residual = scale_data.outcomes - scale_data.actions[:, :, 0] * scale_data.first[:, None]
    scale = float((residual**2).mean())
    normalizer = Normalizer.from_train(data["train"])
    moments = {name: Moments.from_data(x, normalizer) for name, x in data.items()}
    train, cal, val = (moments[name] for name in ("train", "calibration", "validation"))
    candidates: list[Candidate] = []
    for angle in ANGLES:
        weights, d = loss_fit(train, angle)
        candidates.append(
            Candidate(
                angle,
                weights,
                d,
                float(train.errors(weights).mean()),
                float(cal.errors(weights).mean()),
            )
        )
    selected = choose(candidates, scale)
    reference = val.errors(selected["forecast"].weights)
    summaries: dict[str, dict[str, float]] = {}
    for label, c in selected.items():
        error = val.errors(c.weights)
        change = error - reference
        summaries[label] = {
            "angle_degrees": float(c.angle),
            "response_weight_squared": math.cos(math.radians(c.angle)) ** 2,
            "train_raw_mse": c.train_risk,
            "train_disagreement": c.disagreement,
            "train_additive": c.train_risk + scale * c.disagreement,
            "calibration_raw_mse": c.cal_risk,
            "validation_raw_mse": float(error.mean()),
            "validation_raw_mse_se": float(error.std(ddof=1) / np.sqrt(len(error))),
            "paired_validation_change": float(change.mean()),
            "paired_validation_change_se": float(change.std(ddof=1) / np.sqrt(len(change))),
        }
    identity = selected["constrained"].cal_risk <= selected["forecast"].cal_risk
    finite = all(math.isfinite(v) for s in summaries.values() for v in s.values())
    is_low = all(s["response_weight_squared"] >= 0.8 for s in summaries.values())
    low_equiv = all(abs(s["paired_validation_change"]) <= 0.05 * scale for s in summaries.values())
    high_sep = (
        summaries["forecast"]["response_weight_squared"] >= 0.8
        and summaries["constrained"]["response_weight_squared"] >= 0.8
        and summaries["additive"]["response_weight_squared"] <= 0.2
    )
    high_cost = (
        summaries["additive"]["paired_validation_change"] >= 0.08 * scale
        and abs(summaries["constrained"]["paired_validation_change"]) <= 0.05 * scale
    )
    info: dict[str, object] = {
        "scale_joint_calibration_mse": scale,
        "normalization_center": normalizer.center.tolist(),
        "normalization_sd": normalizer.scales.tolist(),
        "selected": summaries,
        "candidates": [
            {
                "angle_degrees": c.angle,
                "train_raw_mse": c.train_risk,
                "train_disagreement": c.disagreement,
                "calibration_raw_mse": c.cal_risk,
                "augmented_weights": c.weights.tolist(),
            }
            for c in candidates
        ],
    }
    return info, {
        "finite_and_feasible": finite and identity,
        "low_separation": is_low and low_equiv,
        "high_separation": high_sep and high_cost,
    }


def pinned(revision: str) -> None:
    """Authenticate both CLI source and its checkout helper."""
    verify_head(revision)
    subprocess.run(["git", "diff", "--quiet", revision, "--", __file__], check=True)


def main() -> None:
    """Write the prospective declaration before fitting any candidate or reading fit metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", default=0, type=int, choices=range(5))
    args = parser.parse_args()
    pinned(args.revision)
    if args.output.resolve() != Path(f"ICML_review/outputs/noisy-support-s{args.seed}").resolve():
        raise ValueError("Use the designated new private run folder for the seed")
    if args.seed:
        pilot = json.loads(Path("ICML_review/outputs/noisy-support-s0/metrics.json").read_text())
        if not pilot["passed"] or pilot["commit"] != args.revision:
            raise RuntimeError("Expansion requires a passed same-revision pilot.")
    collections: dict[str, dict[str, Observed]] = {}
    digest = hashlib.sha256()
    for label, noise in LEVELS.items():
        collections[label] = {}
        for i, (name, n) in enumerate(SPLITS.items()):
            obs = observe(args.seed, i, n, noise)
            collections[label][name] = obs
            digest.update(f"{label}/{name}/{n}\n".encode())
            for values in (obs.first, obs.second, obs.inert, obs.actions, obs.outcomes):
                digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    data_id = digest.hexdigest()
    command = shlex.join(
        [
            ".venv/bin/python",
            "-m",
            "ICML_review.code.noisy_support",
            args.revision,
            str(args.output),
            "--seed",
            str(args.seed),
        ]
    )
    args.output.mkdir(exist_ok=False)
    note = (
        "# Optimized donor code with noisy support\n\n"
        f"Declared {datetime.now(UTC).isoformat()}. Commit {args.revision}; generator seed "
        f"{args.seed}; data identity SHA-256 {data_id}; NumPy {np.__version__}.\n\n"
        "Hypothesis: an additive disagreement penalty can discard an informative but noisy "
        "measurement in favor of a useless shared control. The controlled change between "
        "environments is support measurement noise SD (0.5 or 2); all draws are paired. "
        "Within an environment all policies select from the same encoder directions and fitted "
        "heads. The observed channels are a clean first response, noisy second response and "
        "independent group-shared inert input. A fresh Gaussian control produces an observed "
        "linear response. Use 2048 TRAIN, 1024 CAL and 1024 VAL systems; ten separate support "
        "episodes and four independent query controls per system. Only observed channels enter "
        "fitting and scoring. Compare donor disagreement at fixed risk with additive "
        "regularization on these controlled observations; this is a linear encoder assay, "
        "not a test on either paper dataset.\n\n"
        "TRAIN-standardize the noisy and inert channels. Optimize their unit-norm mixture "
        "direction over angles -90 to +85 degrees in five-degree increments. For every angle "
        "fit the same raw-MSE least-squares predictor affine in code and linear in controls. "
        "Select by (i) minimum TRAIN raw MSE, (ii) minimum TRAIN raw MSE plus scale times "
        "mean within/total variance ratio across the two coordinates, or (iii) minimum TRAIN "
        "disagreement subject to CAL raw MSE no higher than (i)'s. The scale is the raw CAL "
        "MSE of using the observed first response times the first query control. Use population "
        "variances within groups and across all observations. Prefer lower absolute angle, "
        "then lower signed angle, on an exact tie.\n\n"
        "Gate: all selected quantities finite and (iii) CAL-feasible in both regimes. Low "
        "noise: all three squared response weights at least .80 and absolute paired VAL-MSE "
        "change from (i) at most .05 times scale. High noise: squared response weight (i)/(iii) "
        "at least .80, (ii) at most .20; (ii) increases VAL MSE by at least .08 times scale, "
        "and absolute change for (iii) is at most .05 times scale. Passing all licenses "
        "seeds 1-4. Pilot failure ends expansion. A pass supports the predicted reversal in "
        "this family. A failed separation or risk gate is recorded and restricts the claim.\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src {command}\n"
    )
    (args.output / "NOTE.md").write_text(note)
    results: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for label, data in collections.items():
        pinned(args.revision)
        record, local = assess(data)
        results[label] = record
        checks[f"{label}/finite_and_feasible"] = local["finite_and_feasible"]
        checks[f"{label}/separation_and_risk"] = local[f"{label}_separation"]
    pinned(args.revision)
    receipt = {
        "seed": args.seed,
        "commit": args.revision,
        "data_identity": data_id,
        "command": command,
        "numpy": np.__version__,
        "results": results,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (args.output / "metrics.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("Completed noisy-support comparison", args.output, flush=True)


if __name__ == "__main__":
    main()
