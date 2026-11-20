"""Paper analyses of trained endpoints (read-only; fitted on calibration, test only scored).

Subcommands (``python -m diagnostics.vision_dimer.analysis <command> ...``):

- ``figure-data``: for one vision checkpoint, the absolute Pearson and Spearman correlation matrices
  (true rows, learned columns ordered by the calibration assignment), the thresholded graphs aligned
  to the truth, and the code-parameter correlations; the metrics recomputed from the same collection
  are checked against the stored output of ``train evaluate``.
- ``charts``: which chart the learned states follow (medium, camera, pixel, per-dimer
  centre/separation, bead axes rotated by 0-89 degrees), and the MCC of structured and random
  mixings of the true states (the reference lines of the paper's state-recovery figure).
- ``code-invariance``: within- versus between-system spread of single-context parameter
  codes (context invariance, Assumption 8), for the vision endpoints or for Experiment 1's
  saved test codes.
- ``curves``: validation learning curves of the sparse arms (300,000-update runs and their
  continuation to 500,000), starting from the dense initializer's validation evaluation.

Endpoints are found under ``ROOT/seed<s>/adapt/final.pt`` (dense initializer) and
``ROOT/seed<s>/sparse-500k/{baseline,full}/final.pt``.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.shared.sampling import resolve_device
from diagnostics.vision_dimer import data as vdata
from diagnostics.vision_dimer import dgp
from diagnostics.vision_dimer import evaluate as veval
from diagnostics.vision_dimer.train import (
    SPARSE,
    load_checkpoint,
    model_from_checkpoint,
    rendered_batches,
)
from scjepa.eval.recovery import (
    fit_coordinate_match,
    mean_correlation_coefficient,
    pearson_matrix,
    spearman_matrix,
)
from scjepa.models.model import SCJEPA

SEEDS = range(5)
ARMS = ("dense", "baseline", "full")
ANGLES = range(90)
FLOATS = (
    "mcc", "spearman_mcc", "block_purity_mean", "block_purity_knn_mean", "nmse", "state_r2_min",
    "parameter_r2_min", "coordinatewise_r2_monotone_min",
)  # fmt: skip
EXACT = (
    "assignment", "spearman_assignment", "code_parameters", "state_shd", "action_shd",
    "parameter_shd",
)  # fmt: skip
# ``train evaluate`` runs the convolutional encoder with PyTorch's default cuDNN TF32
# convolutions; the analyses run it in full float32 (TF32 off, as in training). The
# stored test metrics and the figure-data recomputation differed by at most 1.8e-5.
REPORTED = 1e-6  # differences above this are recorded
TOLERANCE = 1e-4  # differences above this fail the check


def endpoint(root: Path, seed: int, arm: str, sub: str = "sparse-500k") -> Path:
    """Checkpoint of one reported endpoint."""
    return root / f"seed{seed}" / ("adapt" if arm == "dense" else f"{sub}/{arm}") / "final.pt"


def load_model(path: Path, device: torch.device) -> tuple[SCJEPA, str, dict[str, Any]]:
    """Rebuild a checkpoint's model in its own condition (dense for non-sparse stages)."""
    payload = load_checkpoint(path)
    stage = payload.get("config", {}).get("stage", payload["meta"].get("stage"))
    model = model_from_checkpoint(payload, phase=stage if stage in SPARSE else "dense")
    return model.to(device), str(stage), payload


def _setup_device(device: str) -> torch.device:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return resolve_device(device)


# ------------------------------------------------------------------------------- figure data


def _ordered(matrix: Tensor, columns: list[int]) -> list[list[float]]:
    """Rows: true coordinates; column k: the learned coordinate assigned to true coordinate k."""
    return cast(list[list[float]], matrix[:, columns].tolist())  # pyright: ignore[reportUnknownMemberType]


def figure_data(
    checkpoint: Path, data_dir: Path, stored_path: Path, device_name: str = "cuda"
) -> dict[str, Any]:
    """Correlation matrices, aligned graphs and code correlations of one checkpoint on test."""
    device = _setup_device(device_name)
    model, stage, payload = load_model(checkpoint, device)
    config = vdata.load_config(data_dir)
    samples: dict[str, dict[str, Tensor]] = {}
    labels: dict[str, dict[str, Tensor]] = {}
    with torch.no_grad():
        for name in ("calibration", "test"):
            inputs = vdata.RenderedInputs(data_dir, name, device)
            samples[name] = veval.collect(model, rendered_batches(inputs))
            raw = vdata.load_labels(data_dir, name)
            labels[name] = {"states": raw["states"][:, :, : config.frames], "theta": raw["theta"]}
    result = veval.evaluate(
        model, samples["calibration"], labels["calibration"], samples["test"], labels["test"]
    )
    stored = json.loads(stored_path.read_text())
    if stored.get("scored_split") != "test":
        raise ValueError(f"{stored_path} is not a test evaluation")
    gaps = {key: abs(float(result[key]) - float(stored[key])) for key in FLOATS}
    differences: dict[str, Any] = {
        key: [result.get(key), stored.get(key)] for key in FLOATS if gaps[key] > REPORTED
    }
    mismatched = [key for key in EXACT if result.get(key) != stored.get(key)]
    differences.update({key: [result.get(key), stored.get(key)] for key in mismatched})
    x = samples["test"]["states"].reshape(-1, dgp.STATE_DIM)
    z = labels["test"]["states"].reshape(-1, dgp.STATE_DIM).to(x.dtype)
    record: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "stage": stage,
        "step": payload.get("step"),
        "stored_evaluation": str(stored_path),
        "matches_stored": not mismatched and max(gaps.values()) <= TOLERANCE,
        "max_metric_difference": max(gaps.values()),
        "differences": differences,
        "assignment": result["assignment"],
        "spearman_assignment": result["spearman_assignment"],
        "pearson": _ordered(pearson_matrix(x, z), result["assignment"]),
        "spearman": _ordered(spearman_matrix(x, z), result["spearman_assignment"]),
    }
    if "code_parameters" in result:
        parameters: list[int] = result["code_parameters"]  # parameters[j]: parameter of code j
        by_parameter = [parameters.index(p) for p in range(dgp.PARAM_DIM)]
        codes = samples["test"]["codes"]
        theta = labels["test"]["theta"][:, None].expand(-1, codes.shape[1], -1).to(codes.dtype)
        record["code_parameters"] = parameters
        record["code_pearson"] = _ordered(
            pearson_matrix(codes.reshape(-1, codes.shape[-1]), theta.reshape(-1, dgp.PARAM_DIM)),
            by_parameter,
        )
        predictor = cast(Any, model.predictor)
        if hasattr(predictor, "thresholded_graphs"):
            learned = {k: v.bool().cpu() for k, v in predictor.thresholded_graphs().items()}
            rows = torch.tensor(result["assignment"])
            record["graphs"] = {
                "state": learned["state"][rows][:, rows].int().tolist(),
                "action": learned["action"][rows].int().tolist(),
                "parameter": learned["parameter"][rows][:, by_parameter].int().tolist(),
            }
    return record


# ------------------------------------------------------------------------------------ charts


def rotate_beads(z: Tensor, degrees: float) -> Tensor:
    """Rotate each bead's (u, v) pair by ``degrees`` (state order u1a, v1a, u1b, v1b, ...)."""
    t = math.radians(degrees)
    pairs = z.reshape(-1, 4, 2)
    u, v = pairs[..., 0], pairs[..., 1]
    rotated = (math.cos(t) * u - math.sin(t) * v, math.sin(t) * u + math.cos(t) * v)
    return torch.stack(rotated, -1).reshape(-1, dgp.STATE_DIM)


def centre_separation(z: Tensor) -> Tensor:
    """Per dimer and axis: (bead a + bead b, bead a - bead b) / sqrt 2, in the input frame."""
    d = z.reshape(-1, 2, 2, 2)  # (dimer, slot, axis)
    a, b = d[:, :, 0], d[:, :, 1]
    return torch.stack(((a + b) / math.sqrt(2), (a - b) / math.sqrt(2)), 2).reshape(-1, 8)


def chart_mcc(cal_x: Tensor, cal_c: Tensor, x: Tensor, c: Tensor) -> tuple[float, list[float]]:
    """Calibration-fitted Pearson assignment scored on ``x``; per chart coordinate |corr|."""
    match = fit_coordinate_match(cal_x, cal_c, method="pearson")
    corr = pearson_matrix(x, c)  # chart rows, learned columns
    per = [float(corr[t, match.columns[t]]) for t in range(dgp.STATE_DIM)]
    return mean_correlation_coefficient(match, x, c), per


def true_charts(z: Tensor, renderer: dgp.Renderer) -> dict[str, Tensor]:
    """True charts of states ``z`` (N, 8)."""
    camera = renderer.camera(z).reshape(-1, dgp.STATE_DIM)
    return {
        "medium": z,
        "camera": camera,
        "pixel": renderer.pixel_coordinates(z).reshape(-1, dgp.STATE_DIM),
        "medium_centre_separation": centre_separation(z),
        "camera_centre_separation": centre_separation(camera),
    }


def chart_references(z: Tensor, renderer: dgp.Renderer, draws: int = 2000) -> dict[str, Any]:
    """MCC against the medium chart of structured and random mixings of standardized states."""
    s = (z - z.mean(0)) / z.std(0)
    generator = torch.Generator().manual_seed(0)

    def matched(x: Tensor) -> float:
        corr = pearson_matrix(x, s)
        match = fit_coordinate_match(x, s, method="pearson")
        return float(corr[torch.arange(dgp.STATE_DIM), list(match.columns)].mean())

    def orthogonal(n: int) -> Tensor:
        gaussian = torch.randn(n, n, generator=generator, dtype=torch.float64)
        q, r = cast(tuple[Tensor, Tensor], torch.linalg.qr(gaussian))  # pyright: ignore[reportUnknownMemberType]
        return q * torch.sign(torch.diagonal(r))

    def block_diag(*blocks: Tensor) -> Tensor:
        return cast(Tensor, torch.block_diag(*blocks))  # pyright: ignore[reportUnknownMemberType]

    out: dict[str, Any] = {}
    for name, block in (("within_block", (4, 4)), ("within_bead", (2, 2, 2, 2))):
        values = torch.tensor(
            [matched(s @ block_diag(*[orthogonal(n) for n in block]).T) for _ in range(draws)]
        )
        quantiles = (values.mean(), values.quantile(0.05), values.quantile(0.95), values.min())
        out[name] = {
            k: float(v) for k, v in zip(("mean", "p05", "p95", "min"), quantiles, strict=True)
        }
    hadamard = torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64
    )
    out["hadamard_within_block"] = matched(s @ block_diag(hadamard / 2, hadamard / 2).T)
    out["camera_chart"] = matched(renderer.camera(z).reshape(-1, dgp.STATE_DIM))
    out["pixel_chart"] = matched(renderer.pixel_coordinates(z).reshape(-1, dgp.STATE_DIM))
    out["medium_centre_separation_chart"] = matched(centre_separation(z))
    out["rotation_curve"] = [matched(rotate_beads(z, float(a))) for a in ANGLES]
    return out


def charts(data_dir: Path, root: Path, device_name: str = "cuda") -> dict[str, Any]:
    """Score every endpoint's learned states against every true chart (calibration fitted)."""
    device = _setup_device(device_name)
    config = vdata.load_config(data_dir)
    truth = {
        name: vdata.load_labels(data_dir, name)["states"][:, :, : config.frames]
        .reshape(-1, dgp.STATE_DIM)
        .double()
        for name in ("calibration", "test")
    }
    charted = {name: true_charts(z, config.renderer) for name, z in truth.items()}
    record: dict[str, Any] = {"references": chart_references(truth["test"], config.renderer)}
    record["runs"] = {}
    inputs = {name: vdata.RenderedInputs(data_dir, name, device) for name in truth}
    for s in SEEDS:
        for arm in ARMS:
            model, _, _ = load_model(endpoint(root, s, arm), device)
            with torch.no_grad():
                learned = {
                    name: veval.collect(model, rendered_batches(inputs[name]))["states"].reshape(
                        -1, dgp.STATE_DIM
                    )
                    for name in truth
                }
            run: dict[str, Any] = {}
            for chart in charted["test"]:
                value, per = chart_mcc(
                    learned["calibration"], charted["calibration"][chart],
                    learned["test"], charted["test"][chart],
                )  # fmt: skip
                run[chart] = {"mcc": value, "per_coordinate": per}
            curve: list[float] = []
            for a in ANGLES:
                rotated = rotate_beads(truth["calibration"], float(a))
                curve.append(
                    chart_mcc(learned["calibration"], rotated, learned["calibration"], rotated)[0]
                )
            best = max(ANGLES, key=lambda a: curve[a])
            at_best = chart_mcc(
                learned["calibration"],
                rotate_beads(truth["calibration"], float(best)),
                learned["test"],
                rotate_beads(truth["test"], float(best)),
            )[0]
            run["rotation"] = {
                "best_angle_calibration": best,
                "test_mcc_at_best": at_best,
                "calibration_curve": curve,
            }
            record["runs"][f"seed{s}/{arm}"] = run
            print(json.dumps({"run": f"seed{s}/{arm}", "best_angle": best}), flush=True)
    return record


# ------------------------------------------------------------------------------------ curves

CURVE_KEYS = (
    "mcc", "spearman_mcc", "block_purity_mean", "block_purity_knn_mean", "nmse", "state_shd",
    "action_shd", "parameter_shd", "state_edges", "action_edges", "parameter_edges",
)  # fmt: skip


def curves(root: Path) -> dict[str, Any]:
    """Validation evaluations per seed and sparse arm, step 0 being the dense initializer."""
    out: dict[str, Any] = {}
    for s in SEEDS:
        dense = json.loads((root / f"seed{s}" / "adapt" / "final.json").read_text())
        for arm in ("baseline", "full"):
            rows: dict[int, dict[str, Any]] = {0: {k: dense.get(k) for k in CURVE_KEYS}}
            for sub in ("sparse-300k", "sparse-500k"):
                path = root / f"seed{s}" / sub / arm / "evaluations.jsonl"
                for line in path.read_text().splitlines():
                    entry = json.loads(line)
                    rows.setdefault(int(entry["step"]), {k: entry.get(k) for k in CURVE_KEYS})
            out[f"seed{s}/{arm}"] = [{"step": step, **rows[step]} for step in sorted(rows)]
    return out


# --------------------------------------------------------------------------- code invariance


def spread(codes: Tensor) -> dict[str, float]:
    """Within- and between-system variance of codes ``(G, R, D)``, scale-free summaries.

    ``within_share``: mean over coordinates of within-system over total variance;
    ``worst_direction_share``: the largest within-system share along any direction of code
    space (invariant to invertible linear maps of the code); ``sd_ratio``: within-system s.d.
    relative to the s.d. of the system means, after standardizing coordinates.
    """
    c = codes.double()
    groups, views, dim = c.shape
    total = torch.cov(c.reshape(-1, dim).T, correction=1)
    centred = c - c.mean(1, keepdim=True)
    within = torch.einsum("grd,gre->de", centred, centred) / (groups * (views - 1))
    between = torch.cov(c.mean(1).T, correction=1)
    diag_t, diag_w, diag_b = torch.diagonal(total), torch.diagonal(within), torch.diagonal(between)
    ratio = cast(Tensor, torch.linalg.solve(total, within))  # pyright: ignore[reportUnknownMemberType]
    worst = float(cast(Tensor, torch.linalg.eigvals(ratio)).real.max())  # pyright: ignore[reportUnknownMemberType]
    return {
        "within_share": float((diag_w / diag_t).mean()),
        "within_share_max_coordinate": float((diag_w / diag_t).max()),
        "worst_direction_share": worst,
        "sd_ratio": float(torch.sqrt((diag_w / diag_b).mean())),
        "groups": groups,
        "contexts_per_group": views,
        "code_dim": dim,
    }


def code_invariance_vision(data_dir: Path, root: Path, device_name: str = "cuda") -> dict[str, Any]:
    """Single-context test codes of every vision endpoint."""
    device = _setup_device(device_name)
    inputs = vdata.RenderedInputs(data_dir, "test", device)
    out: dict[str, Any] = {}
    for s in SEEDS:
        for arm in ARMS:
            model, _, _ = load_model(endpoint(root, s, arm), device)
            with torch.no_grad():
                codes = veval.collect(model, rendered_batches(inputs))["codes"]
            out[f"seed{s}/{arm}"] = spread(codes)
    return out


def code_invariance_experiment1(test_dir: Path) -> dict[str, Any]:
    """Experiment 1: single-context codes saved by its test assessment (``seed*/*/test.pt``)."""
    out: dict[str, Any] = {}
    for path in sorted(test_dir.glob("seed*/*/test.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        out[f"{path.parent.parent.name}/{path.parent.name}"] = spread(payload["samples"]["codes"])
    return out


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fig = commands.add_parser("figure-data")
    fig.add_argument("--checkpoint", type=Path, required=True)
    fig.add_argument("--data-dir", type=Path, required=True)
    fig.add_argument(
        "--stored", type=Path, required=True, help="train evaluate --split test output"
    )
    fig.add_argument("--output", type=Path, required=True)
    fig.add_argument("--device", default="cuda")
    chart = commands.add_parser("charts")
    chart.add_argument("--data-dir", type=Path, required=True)
    chart.add_argument("--root", type=Path, required=True)
    chart.add_argument("--output", type=Path, required=True)
    chart.add_argument("--device", default="cuda")
    inv = commands.add_parser("code-invariance")
    inv.add_argument("--data-dir", type=Path, help="vision cohort (with --root)")
    inv.add_argument("--root", type=Path, help="vision endpoints")
    inv.add_argument("--experiment1-test-dir", type=Path, help="Experiment 1 saved test outputs")
    inv.add_argument("--output", type=Path, required=True)
    inv.add_argument("--device", default="cuda")
    curve = commands.add_parser("curves")
    curve.add_argument("--root", type=Path, required=True)
    curve.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "figure-data":
        record = figure_data(args.checkpoint, args.data_dir, args.stored, args.device)
    elif args.command == "charts":
        record = charts(args.data_dir, args.root, args.device)
    elif args.command == "curves":
        record = curves(args.root)
    elif args.experiment1_test_dir is not None:
        record = code_invariance_experiment1(args.experiment1_test_dir)
    elif args.data_dir is not None and args.root is not None:
        record = code_invariance_vision(args.data_dir, args.root, args.device)
    else:
        raise SystemExit("code-invariance needs --data-dir and --root, or --experiment1-test-dir")
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
