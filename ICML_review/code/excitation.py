"""Measure what forecast and within-group agreement losses distinguish under limited excitation."""

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
CHARTS = ("response", "episode", "system")
SPLITS = {"train": 2048, "calibration": 1024, "validation": 1024}


@dataclass(frozen=True)
class Observations:
    """Observed support responses, inert controls and distinct recipient action-response pairs."""

    support: Array
    inert_controls: Array
    query_actions: Array
    query_responses: Array


def observations(seed: int, split: int, groups: int, excitation: int) -> Observations:
    """Generate paired environments with common random systems across the two excitation levels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, split]))
    parameters = rng.standard_normal((groups, 2))
    shared = rng.standard_normal((groups, 1))
    episodic = rng.standard_normal((groups, 2))
    inert = np.stack((np.repeat(shared, 2, axis=1), episodic), axis=-1)
    support = np.repeat((parameters * [1, excitation])[:, None, :], 2, axis=1)
    actions: Array = np.array(
        [[-1, -excitation], [-1, excitation], [1, -excitation], [1, excitation]], dtype=np.float64
    )
    return Observations(support, inert, actions, parameters @ actions.T)


def encode(data: Observations, chart: str) -> Array:
    """Extract a prescribed chart from the observations and support actions only."""
    if chart == "response":
        return data.support
    if chart not in ("episode", "system"):
        raise ValueError(f"Unknown chart {chart}")
    return np.stack(
        (data.support[..., 0], data.inert_controls[..., int(chart == "episode")]), axis=-1
    )


@dataclass(frozen=True)
class Fit:
    """Frozen TRAIN normalizer and bilinear regression weights."""

    center: Array
    whitening: Array
    weights: Array


def features(codes: Array, actions: Array) -> Array:
    """Form action times affine-in-code features for each donor and recipient query."""
    affine = np.concatenate((np.ones_like(codes[..., :1]), codes), axis=-1)
    return np.einsum("qa,ged->geqad", actions, affine).reshape(*codes.shape[:2], len(actions), -1)


def fit(data: Observations, chart: str) -> Fit:
    """Fit whitening on observed support and learn an identical least-squares head per chart."""
    raw = encode(data, chart)
    flat = raw.reshape(-1, raw.shape[-1])
    center = flat.mean(axis=0)
    centered = flat - center
    covariance = centered.T @ centered / len(centered)
    eigs, basis = np.linalg.eigh(covariance)
    live = eigs > 1e-10 * eigs.max()
    whitening = (basis[:, live] / np.sqrt(eigs[live])) @ basis[:, live].T
    codes = (raw - center) @ whitening
    design = features(codes, data.query_actions)
    target = np.broadcast_to(data.query_responses[:, None, :], design.shape[:-1])
    weights = np.linalg.lstsq(design.reshape(-1, design.shape[-1]), target.reshape(-1), rcond=None)[
        0
    ]
    return Fit(center, whitening, weights.reshape(2, 3))


def ratio(numerator: Array, denominator: Array) -> dict[str, float]:
    """Estimate a ratio of group means with its system-level delta-method standard error."""
    denom = float(denominator.mean())
    if denom <= 0:
        raise ValueError("Undefined energy ratio")
    estimate = float(numerator.mean()) / denom
    influence = (numerator - estimate * denominator) / denom
    return {"estimate": estimate, "se": float(influence.std(ddof=1) / np.sqrt(len(influence)))}


def score(data: Observations, chart: str, model: Fit) -> dict[str, float]:
    """Evaluate predictive risk, normalized donor disagreement, rank and code sensitivity."""
    codes = (encode(data, chart) - model.center) @ model.whitening
    prediction = features(codes, data.query_actions) @ model.weights.reshape(-1)
    squared_error = ((prediction - data.query_responses[:, None, :]) ** 2).mean(axis=(1, 2))
    risk = ratio(squared_error, (data.query_responses**2).mean(axis=1))
    centered = codes - codes.mean(axis=(0, 1))
    spread = 0.5 * ((codes[:, 0] - codes[:, 1]) ** 2).sum(axis=-1)
    disagreement = ratio(spread, (centered**2).sum(axis=-1).mean(axis=1))
    flat = centered.reshape(-1, centered.shape[-1])
    spectrum = np.linalg.eigvalsh(flat.T @ flat / len(flat))
    gradient = data.query_actions @ model.weights[:, 1:]
    sensitivity = np.linalg.eigvalsh(gradient.T @ gradient / len(gradient))
    return {
        "forecast_nmse": risk["estimate"],
        "forecast_nmse_se": risk["se"],
        "disagreement": disagreement["estimate"],
        "disagreement_se": disagreement["se"],
        "forecast_plus_disagreement": risk["estimate"] + disagreement["estimate"],
        "code_covariance_min": float(spectrum[0]),
        "code_covariance_max": float(spectrum[-1]),
        "forecast_sensitivity_min_over_max": float(sensitivity[0] / sensitivity[-1]),
    }


def criteria(results: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, bool]:
    """Evaluate the predeclared ordering on independent calibration and validation systems."""
    checks: dict[str, bool] = {}
    for split in ("calibration", "validation"):
        limited, excited = results["limited"][split], results["excited"][split]
        checks[f"{split}/limited_forecast"] = all(
            v["forecast_nmse"] < 1e-10 for v in limited.values()
        )
        checks[f"{split}/limited_agreement"] = (
            limited["episode"]["disagreement"] > 0.3
            and limited["system"]["disagreement"] < 1e-10
            and limited["response"]["disagreement"] < 1e-10
        )
        checks[f"{split}/limited_rank"] = (
            all(limited[k]["code_covariance_min"] > 0.7 for k in ("episode", "system"))
            and limited["response"]["code_covariance_min"] < 1e-10
            and all(v["forecast_sensitivity_min_over_max"] < 1e-8 for v in limited.values())
        )
        checks[f"{split}/excited_forecast"] = (
            excited["response"]["forecast_nmse"] < 1e-10
            and all(excited[k]["forecast_nmse"] > 0.2 for k in ("episode", "system"))
            and excited["response"]["forecast_sensitivity_min_over_max"] > 0.5
        )
    return checks


def verify_head(revision: str) -> None:
    """Require the pinned checkout before producing a declaration or assessment."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != revision:
        raise ValueError(f"Expected {revision}, found {head}")
    subprocess.run(["git", "diff", "--quiet", revision, "--", __file__], check=True)


def main() -> None:
    """Declare and run a compact controlled experiment entirely on a CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    verify_head(args.revision)
    if args.output.resolve().parent != Path("ICML_review/outputs").resolve():
        raise ValueError("Choose a new run directly under ICML_review/outputs")
    levels = {"limited": 0, "excited": 1}
    collections: dict[str, dict[str, Observations]] = {}
    digest = hashlib.sha256()
    for label, level in levels.items():
        collections[label] = {}
        for i, (key, n) in enumerate(SPLITS.items()):
            obs = observations(args.seed, i, n, level)
            collections[label][key] = obs
            digest.update(f"{label}/{key}/{n}\n".encode())
            for values in (obs.support, obs.inert_controls, obs.query_actions, obs.query_responses):
                digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    data_id = digest.hexdigest()
    command = shlex.join(
        [
            ".venv/bin/python",
            "-m",
            "ICML_review.code.excitation",
            args.revision,
            str(args.output),
            "--seed",
            str(args.seed),
        ]
    )
    args.output.mkdir(parents=False, exist_ok=False)
    note = (
        "# Excitation and falsifiability of two label-free losses\n\n"
        f"Declared {datetime.now(UTC).isoformat()}. Commit {args.revision}. "
        f"Observed data SHA-256 {data_id}. Generator seed {args.seed}, NumPy {np.__version__}. "
        "Separate TRAIN/calibration/validation cohorts of 2048/1024/1024 systems, two support "
        "episodes and four queries per system. Matched draws under both excitation conditions. "
        "This local CPU calculation reads no benchmark data and produces no tensor checkpoints.\n\n"
        "Hypothesis: donor agreement can remove an episode nuisance but a perfectly shared "
        "irrelevant coordinate also satisfies it. Excitation makes the omitted mechanism "
        "predictively visible. The single controlled environment change is the amplitude of "
        "the second control (zero or one). In each condition compare the same three prescribed "
        "charts, using identical whitening and least squares. The two responses are "
        "y_j = theta dot u_j; the inert control is group-constant on the first support step and "
        "episode-specific on the second. Only observed responses and actions enter fitting. "
        "No chart is selected from held-out data.\n\n"
        "Measure forecast NMSE F, normalized paired-donor disagreement D and F+D. D equals "
        "half the mean squared donor difference divided by total centered code variance. "
        "Assess minimum/maximal eigenvalues of code covariance and the eigenvalue ratio of "
        "the observed-query forecast Jacobian Gram matrix. On each held-out split, limited "
        "excitation should give F < 1e-10 for all charts, D > 0.3 for episode nuisance, and "
        "D < 1e-10 for the other charts. Both nuisance charts should have minimum code "
        "eigenvalue > 0.7, the response chart < 1e-10, and all forecast Gram ratios < 1e-8. "
        "With full excitation the response chart should give F < 1e-10 and Gram ratio > 0.5; "
        "both nuisance charts should give F > 0.2. A passing conjunction supports the proposed "
        "counterexample and sensitivity check in this controlled family. Failure of any bound "
        "is reported and limits that empirical conclusion. These are comparisons of fixed "
        "observable charts; this experiment does not estimate neural optimization performance.\n\n"
        f"    PYTHONPATH=$PWD:$PWD/src {command}\n"
    )
    (args.output / "NOTE.md").write_text(note)
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    fits: dict[str, dict[str, object]] = {}
    for label, splits in collections.items():
        verify_head(args.revision)
        results[label] = {split: {} for split in splits}
        fits[label] = {}
        for chart in CHARTS:
            model = fit(splits["train"], chart)
            fits[label][chart] = {
                "center": model.center.tolist(),
                "whitening": model.whitening.tolist(),
                "weights": model.weights.tolist(),
            }
            for split, data in splits.items():
                results[label][split][chart] = score(data, chart, model)
    checks = criteria(results)
    receipt = {
        "commit": args.revision,
        "seed": args.seed,
        "data_identity": data_id,
        "command": command,
        "numpy": np.__version__,
        "fit": fits,
        "results": results,
        "checks": checks,
        "passed": all(checks.values()),
    }
    verify_head(args.revision)
    (args.output / "metrics.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print("Completed controlled CPU assay", args.output, flush=True)


if __name__ == "__main__":
    main()
