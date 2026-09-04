"""Create paired eight-seed dense-versus-sparse Experiment-1 box plots.

The dense and sparse inputs are run directories containing both ``metrics.json``
and ``resolved_config.yaml``. Runs are paired by their recorded seed, never by
glob order. Only terminal 5,000-episode reports on seed offset 29 are accepted.

Expected layout and usage::

    ROOT=outputs/l40_exp1_8seed_job123
    python scripts/aggregate_dense_sparse.py "$ROOT" --expected-tau 0.05 \
      --seeds 0 1 2 3 4 5 6 7

Each seed lives at ``ROOT/seedN/{dense,sparse}``. Artifacts default to
``ROOT/aggregate``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Phase = Literal["dense", "sparse"]
EXPECTED_SEEDS = tuple(range(8))
OUTPUT_SUMMARY = "dense_sparse_loss_mcc_summary.json"
OUTPUT_CONSTRAINT = "dense_sparse_constraint_mcc_boxplots"
OUTPUT_MSE = "dense_sparse_mse_mcc_boxplots"

_REQUIRED_PROTOCOL: dict[str, int | float] = {
    "step": 300_000,
    "eval_seed_offset": 29,
    "num_samples": 5_000,
    "lambda_rollout_t2": 1.0,
    "num_rollout_t2_anchors": 8,
    "rollout_t2_horizon": 2,
    "oe_eval_horizon": 30,
    "oe_tolerance_nrmse": 0.1,
}
_VALUE_KEYS = (
    "pred_loss",
    "trajectory_reconstruction_mse_k30",
    "loss_rollout_t2_raw",
    "loss_rollout_t2_weighted",
    "tf_t2_predictive_loss",
    "constraint_loss",
    "mcc",
)
_CONFIG_IGNORES = (
    ("model", "spartan_dense"),
    ("model", "spartan_identity"),
    ("train", "sparsity_enabled"),
    ("train", "sparsity_tau"),
    ("train", "seed"),
    ("train", "oe_coordinate_std"),
    ("data", "seed"),
    ("data", "preload"),
    ("wandb", "run_tag"),
)


@dataclass(frozen=True)
class EvaluatedRun:
    """One validated terminal run and the values used by the comparison."""

    directory: Path
    phase: Phase
    seed: int
    metrics: dict[str, object]
    values: dict[str, float]
    coordinate_std: tuple[float, ...]
    normalized_config: str
    git_sha: str
    sparsity_tau: float | None


@dataclass(frozen=True)
class SeedPair:
    """Dense and sparse terminal reports for one training/data seed."""

    seed: int
    dense: EvaluatedRun
    sparse: EvaluatedRun


@dataclass(frozen=True)
class Comparison:
    """A fully validated ordered eight-seed comparison."""

    pairs: tuple[SeedPair, ...]
    provenance: dict[str, int | float]
    config_sha256: str
    git_sha: str


def _json_mapping(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], loaded)


def _number(mapping: dict[str, object], key: str, source: Path) -> float:
    if key not in mapping:
        raise ValueError(f"{source} is missing required field {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{source} field {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source} field {key!r} must be finite")
    return result


def _seed(mapping: dict[str, object], source: Path) -> int:
    value = mapping.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} field 'seed' must be an integer")
    return value


def _coordinate_std(mapping: dict[str, object], source: Path) -> tuple[float, ...]:
    raw = mapping.get("oe_coordinate_std")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source} field 'oe_coordinate_std' must be a non-empty list")
    values: list[float] = []
    for value in cast(list[object], raw):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{source} field 'oe_coordinate_std' must be numeric")
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{source} field 'oe_coordinate_std' must be finite and positive")
        values.append(number)
    return tuple(values)


def _mapping_value(mapping: dict[str, object], section: str, key: str, source: Path) -> object:
    nested = mapping.get(section)
    if not isinstance(nested, dict) or key not in nested:
        raise ValueError(f"{source} is missing configuration field {section}.{key}")
    typed = cast(dict[str, object], nested)
    return typed[key]


def _config_integer(mapping: dict[str, object], section: str, key: str, source: Path) -> int:
    value = _mapping_value(mapping, section, key, source)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source} configuration field {section}.{key} must be an integer")
    return value


def _config_bool(mapping: dict[str, object], section: str, key: str, source: Path) -> bool:
    value = _mapping_value(mapping, section, key, source)
    if not isinstance(value, bool):
        raise ValueError(f"{source} configuration field {section}.{key} must be boolean")
    return value


def _remove_nested(mapping: dict[str, object], path: tuple[str, str]) -> None:
    section = mapping.get(path[0])
    if isinstance(section, dict):
        cast(dict[str, object], section).pop(path[1], None)


def _normalized_config(config: dict[str, object]) -> str:
    normalized = copy.deepcopy(config)
    # Revision provenance is checked explicitly across every run below.  Keeping
    # it out of the protocol hash makes a revision mismatch produce the useful
    # git_sha error instead of looking like a hyperparameter mismatch.
    normalized.pop("git_sha", None)
    for path in _CONFIG_IGNORES:
        _remove_nested(normalized, path)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _validate_phase(config: dict[str, object], phase: Phase, source: Path) -> None:
    dense = _config_bool(config, "model", "spartan_dense", source)
    identity = _config_bool(config, "model", "spartan_identity", source)
    sparsity = _config_bool(config, "train", "sparsity_enabled", source)
    expected = (True, False, False) if phase == "dense" else (False, False, True)
    if (dense, identity, sparsity) != expected:
        raise ValueError(
            f"{source} is not a {phase} run: "
            f"spartan_dense={dense}, spartan_identity={identity}, "
            f"sparsity_enabled={sparsity}"
        )


def _validate_protocol(
    metrics: dict[str, object], config: dict[str, object], source: Path
) -> dict[str, int | float]:
    actual: dict[str, int | float] = {}
    for key, expected in _REQUIRED_PROTOCOL.items():
        value = _number(metrics, key, source)
        if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{source} is not the required final evaluation: "
                f"{key}={value:g}, expected {expected}"
            )
        actual[key] = expected
    configured_steps = _config_integer(config, "train", "steps", source)
    if configured_steps != int(actual["step"]):
        raise ValueError(
            f"{source} checkpoint step {actual['step']} != configured steps {configured_steps}"
        )
    return actual


def _load_run(directory: Path, phase: Phase) -> EvaluatedRun:
    metrics_path = directory / "metrics.json"
    config_path = directory / "resolved_config.yaml"
    if not metrics_path.is_file() or not config_path.is_file():
        raise ValueError(f"{directory} must contain metrics.json and resolved_config.yaml")
    metrics = _json_mapping(metrics_path)
    loaded_config = OmegaConf.load(config_path)
    if not isinstance(loaded_config, DictConfig):
        raise ValueError(f"{config_path} must contain a mapping")
    raw_config: object = OmegaConf.to_container(loaded_config, resolve=False)
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    config = cast(dict[str, object], raw_config)

    seed = _seed(metrics, metrics_path)
    train_seed = _config_integer(config, "train", "seed", config_path)
    data_seed = _config_integer(config, "data", "seed", config_path)
    if seed != train_seed or seed != data_seed:
        raise ValueError(
            f"{directory} seed mismatch: metrics={seed}, train={train_seed}, data={data_seed}"
        )
    _validate_phase(config, phase, config_path)
    _validate_protocol(metrics, config, metrics_path)

    pred_loss = _number(metrics, "pred_loss", metrics_path)
    trajectory_mse = _number(metrics, "trajectory_reconstruction_mse_k30", metrics_path)
    rollout_raw = _number(metrics, "loss_rollout_t2_raw", metrics_path)
    rollout_weighted = _number(metrics, "loss_rollout_t2_weighted", metrics_path)
    constraint = _number(metrics, "constraint_loss", metrics_path)
    mcc = _number(metrics, "mcc", metrics_path)
    for key, value in {
        "pred_loss": pred_loss,
        "trajectory_reconstruction_mse_k30": trajectory_mse,
        "loss_rollout_t2_raw": rollout_raw,
        "loss_rollout_t2_weighted": rollout_weighted,
        "constraint_loss": constraint,
    }.items():
        if value < 0:
            raise ValueError(f"{metrics_path} field {key!r} must be non-negative")
    if not 0.0 <= mcc <= 1.0:
        raise ValueError(f"{metrics_path} field 'mcc' must lie in [0, 1]")

    lambda_t2 = _number(metrics, "lambda_rollout_t2", metrics_path)
    if not math.isclose(rollout_weighted, lambda_t2 * rollout_raw, rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError(f"{metrics_path} has inconsistent raw and weighted T2 losses")
    tf_t2 = pred_loss + rollout_weighted
    logit_weighted = _number(metrics, "logit_weighted", metrics_path)
    if logit_weighted < 0:
        raise ValueError(f"{metrics_path} field 'logit_weighted' must be non-negative")
    if constraint != tf_t2 + logit_weighted:
        raise ValueError(f"{metrics_path} constraint_loss does not match its components")

    config_std_raw = _mapping_value(config, "train", "oe_coordinate_std", config_path)
    if not isinstance(config_std_raw, list):
        raise ValueError(f"{config_path} train.oe_coordinate_std must be a list")
    config_std = _coordinate_std(
        {"oe_coordinate_std": cast(list[object], config_std_raw)}, config_path
    )
    coordinate_std = _coordinate_std(metrics, metrics_path)
    if coordinate_std != config_std:
        raise ValueError(f"{directory} metrics/config oe_coordinate_std mismatch")

    git_sha = config.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha:
        raise ValueError(f"{config_path} must contain non-empty git_sha")
    sparsity_tau: float | None = None
    if phase == "sparse":
        raw_tau = _mapping_value(config, "train", "sparsity_tau", config_path)
        if isinstance(raw_tau, bool) or not isinstance(raw_tau, int | float):
            raise ValueError(f"{config_path} train.sparsity_tau must be numeric")
        sparsity_tau = float(raw_tau)
        if not math.isfinite(sparsity_tau) or sparsity_tau <= 0:
            raise ValueError(f"{config_path} train.sparsity_tau must be finite and positive")
    return EvaluatedRun(
        directory=directory,
        phase=phase,
        seed=seed,
        metrics=metrics,
        values={
            "pred_loss": pred_loss,
            "trajectory_reconstruction_mse_k30": trajectory_mse,
            "loss_rollout_t2_raw": rollout_raw,
            "loss_rollout_t2_weighted": rollout_weighted,
            "tf_t2_predictive_loss": tf_t2,
            "logit_weighted": logit_weighted,
            "constraint_loss": constraint,
            "mcc": mcc,
        },
        coordinate_std=coordinate_std,
        normalized_config=_normalized_config(config),
        git_sha=git_sha,
        sparsity_tau=sparsity_tau,
    )


def _by_seed(runs: list[EvaluatedRun], phase: Phase) -> dict[int, EvaluatedRun]:
    indexed: dict[int, EvaluatedRun] = {}
    for run in runs:
        if run.seed in indexed:
            raise ValueError(
                f"duplicate {phase} seed {run.seed}: "
                f"{indexed[run.seed].directory} and {run.directory}"
            )
        indexed[run.seed] = run
    return indexed


def build_comparison(
    root: Path,
    expected_tau: float,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> Comparison:
    """Load, validate, and pair terminal dense/sparse reports by seed."""
    if len(expected_seeds) != 8 or len(set(expected_seeds)) != 8:
        raise ValueError("the comparison requires exactly eight unique expected seeds")
    if not math.isfinite(expected_tau) or expected_tau <= 0:
        raise ValueError("expected tau must be finite and positive")
    if not root.is_dir():
        raise ValueError(f"comparison root does not exist or is not a directory: {root}")
    discovered_seeds = {
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := re.fullmatch(r"seed(-?\d+)", child.name))
    }
    unexpected_seeds = discovered_seeds - set(expected_seeds)
    if unexpected_seeds:
        raise ValueError(f"unexpected seed directories: {sorted(unexpected_seeds)}")
    dense = _by_seed(
        [_load_run(root / f"seed{seed}" / "dense", "dense") for seed in expected_seeds],
        "dense",
    )
    sparse = _by_seed(
        [_load_run(root / f"seed{seed}" / "sparse", "sparse") for seed in expected_seeds],
        "sparse",
    )
    expected = set(expected_seeds)
    for phase, indexed in (("dense", dense), ("sparse", sparse)):
        actual = set(indexed)
        if actual != expected:
            raise ValueError(
                f"{phase} seeds do not match expected set: "
                f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
            )

    pairs: list[SeedPair] = []
    normalized_configs: set[str] = set()
    git_shas: set[str] = set()
    for seed in expected_seeds:
        dense_run, sparse_run = dense[seed], sparse[seed]
        assert sparse_run.sparsity_tau is not None
        if not math.isclose(sparse_run.sparsity_tau, expected_tau, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"seed {seed} sparse tau {sparse_run.sparsity_tau:.17g} "
                f"!= expected tau {expected_tau:.17g}"
            )
        if dense_run.coordinate_std != sparse_run.coordinate_std:
            raise ValueError(f"seed {seed} dense/sparse oe_coordinate_std mismatch")
        if dense_run.normalized_config != sparse_run.normalized_config:
            raise ValueError(f"seed {seed} dense/sparse resolved configuration mismatch")
        pairs.append(SeedPair(seed=seed, dense=dense_run, sparse=sparse_run))
        normalized_configs.update((dense_run.normalized_config, sparse_run.normalized_config))
        git_shas.update((dense_run.git_sha, sparse_run.git_sha))
    if len(normalized_configs) != 1:
        raise ValueError("resolved protocol configuration differs across seeds")
    if len(git_shas) != 1:
        raise ValueError(f"git_sha differs across runs: {sorted(git_shas)}")
    normalized_config = next(iter(normalized_configs))
    return Comparison(
        pairs=tuple(pairs),
        provenance=dict(_REQUIRED_PROTOCOL),
        config_sha256=hashlib.sha256(normalized_config.encode()).hexdigest(),
        git_sha=next(iter(git_shas)),
    )


def five_number(values: list[float]) -> dict[str, float]:
    """Return the inclusive five-number summary used by the plotted boxes."""
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "min": ordered[0],
        "q1": quartiles[0],
        "median": quartiles[1],
        "q3": quartiles[2],
        "max": ordered[-1],
    }


def _summary(values: list[float]) -> dict[str, float]:
    result = {"mean": statistics.mean(values), "sd": statistics.stdev(values)}
    result.update(five_number(values))
    return result


def _metric_values(comparison: Comparison, phase: Phase, key: str) -> list[float]:
    values: list[float] = []
    for pair in comparison.pairs:
        run = pair.dense if phase == "dense" else pair.sparse
        value = run.values[key]
        values.append(value)
    return values


def _boxplot(
    comparison: Comparison,
    output_root: Path,
    *,
    loss_key: str,
    loss_label: str,
    output_stem: str,
) -> tuple[Path, Path]:
    dense_mcc = _metric_values(comparison, "dense", "mcc")
    sparse_mcc = _metric_values(comparison, "sparse", "mcc")
    dense_loss = _metric_values(comparison, "dense", loss_key)
    sparse_loss = _metric_values(comparison, "sparse", loss_key)

    figure, axes = plt.subplots(2, 1, figsize=(5.2, 6.8), sharex=True)
    # Baumgartner Figure 3 uses a neutral Transformer and green SPARTAN.
    colors = ("#858585", "#55A868")
    for axis, dense_values, sparse_values in (
        (axes[0], dense_mcc, sparse_mcc),
        (axes[1], dense_loss, sparse_loss),
    ):
        boxes = axis.boxplot(
            [dense_values, sparse_values],
            positions=[1, 2],
            widths=0.48,
            whis=(0, 100),
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for dense_value, sparse_value in zip(dense_values, sparse_values, strict=True):
            axis.plot(
                [1, 2],
                [dense_value, sparse_value],
                color="#6F6F6F",
                linewidth=0.8,
                alpha=0.45,
                marker="o",
                markersize=3.2,
                markerfacecolor="white",
            )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("MCC")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_title(f"Dense vs sparse ({len(comparison.pairs)} paired seeds)")
    axes[0].text(-0.12, 1.02, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].set_ylabel(loss_label)
    axes[1].set_xticks([1, 2], ["Dense", "Sparse"])
    axes[1].text(-0.12, 1.02, "b", transform=axes[1].transAxes, fontweight="bold")
    if loss_key == "constraint_loss":
        tau = comparison.pairs[0].sparse.sparsity_tau
        assert tau is not None
        axes[1].axhline(
            tau,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            label=r"$\tau$",
        )
        axes[1].legend(frameon=False, loc="best")
    if all(value > 0 for value in dense_loss + sparse_loss):
        axes[1].set_yscale("log")

    figure.tight_layout()
    png_path = output_root / f"{output_stem}.png"
    pdf_path = output_root / f"{output_stem}.pdf"
    figure.savefig(  # pyright: ignore[reportUnknownMemberType]
        png_path, dpi=240, bbox_inches="tight"
    )
    figure.savefig(pdf_path, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    plt.close(figure)
    return png_path, pdf_path


def _run_payload(run: EvaluatedRun) -> dict[str, object]:
    return {
        "run_dir": str(run.directory),
        **run.values,
        "total_skips": run.metrics.get("total_skips"),
    }


def write_artifacts(comparison: Comparison, output_root: Path) -> dict[str, Path]:
    """Write both paired figures and one machine-readable summary."""
    output_root.mkdir(parents=True, exist_ok=True)
    constraint_png, constraint_pdf = _boxplot(
        comparison,
        output_root,
        loss_key="constraint_loss",
        loss_label=(r"Constraint $C=L_{TF}+\lambda_{T2}L_{AR2}+\lambda_{logit}L_{logit}$"),
        output_stem=OUTPUT_CONSTRAINT,
    )
    mse_png, mse_pdf = _boxplot(
        comparison,
        output_root,
        loss_key="trajectory_reconstruction_mse_k30",
        loss_label="Validation autoregressive trajectory MSE (K=30)",
        output_stem=OUTPUT_MSE,
    )

    metric_summary: dict[str, object] = {}
    summary_keys = [*_VALUE_KEYS, "logit_weighted"]
    for key in summary_keys:
        dense = [pair.dense.values[key] for pair in comparison.pairs]
        sparse = [pair.sparse.values[key] for pair in comparison.pairs]
        dense_values = list(dense)
        sparse_values = list(sparse)
        deltas = [
            sparse_value - dense_value
            for dense_value, sparse_value in zip(dense_values, sparse_values, strict=True)
        ]
        metric_summary[key] = {
            "dense": _summary(dense_values),
            "sparse": _summary(sparse_values),
            "paired_delta_sparse_minus_dense": _summary(deltas),
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "seeds": [pair.seed for pair in comparison.pairs],
        "loss_definitions": {
            "pred_loss": "validation teacher-forced one-step MSE",
            "trajectory_reconstruction_mse_k30": (
                "validation K=30 open-loop autoregressive trajectory MSE; "
                "closest Baumgartner Figure 3 analogue"
            ),
            "tf_t2_predictive_loss": "pred_loss + loss_rollout_t2_weighted",
            "constraint_loss": ("pred_loss + loss_rollout_t2_weighted + logit_weighted"),
        },
        "provenance": {
            **comparison.provenance,
            "git_sha": comparison.git_sha,
            "normalized_config_sha256": comparison.config_sha256,
            "sparsity_tau": comparison.pairs[0].sparse.sparsity_tau,
            "dense_phase": {
                "spartan_dense": True,
                "spartan_identity": False,
                "sparsity_enabled": False,
            },
            "sparse_phase": {
                "spartan_dense": False,
                "spartan_identity": False,
                "sparsity_enabled": True,
            },
            "oe_coordinate_std_by_seed": {
                str(pair.seed): list(pair.dense.coordinate_std) for pair in comparison.pairs
            },
        },
        "pairs": [
            {
                "seed": pair.seed,
                "dense": _run_payload(pair.dense),
                "sparse": _run_payload(pair.sparse),
            }
            for pair in comparison.pairs
        ],
        "metrics": metric_summary,
    }
    summary_path = output_root / OUTPUT_SUMMARY
    summary_path.write_text(json.dumps(payload, indent=2))
    return {
        "constraint_png": constraint_png,
        "constraint_pdf": constraint_pdf,
        "mse_png": mse_png,
        "mse_pdf": mse_pdf,
        "summary": summary_path,
    }


def main() -> None:
    """Parse paths, validate the paired protocol, and write artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="root containing seedN/{dense,sparse}")
    parser.add_argument("--expected-tau", required=True, type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (default: ROOT/aggregate)",
    )
    parser.add_argument(
        "--seeds",
        nargs=8,
        type=int,
        default=list(EXPECTED_SEEDS),
        metavar="SEED",
        help="exactly eight expected seeds (default: 0 1 2 3 4 5 6 7)",
    )
    args = parser.parse_args()
    try:
        comparison = build_comparison(args.root, args.expected_tau, tuple(args.seeds))
        output_dir = args.output_dir if args.output_dir is not None else args.root / "aggregate"
        paths = write_artifacts(comparison, output_dir)
    except ValueError as error:
        parser.error(str(error))
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
