"""Test a predictive TRAIN projection of one fixed full-length RGB acquisition."""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.masked_mlp_learned.acquisition import group_covariances
from diagnostics.shared.sampling import resolve_device
from diagnostics.vision_dimer import data, dgp, train
from diagnostics.vision_dimer.model import LaggedCodeEncoder
from scjepa.experiments.storage import write_json
from scjepa.models.model import SCJEPA

from .adaptation_interval import group_errors
from .predictive_projection import orient_columns, reduced_rank_map
from .single_acquisition import _head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import OUTPUT, PYTHON, ROOT, adaptation_checks, read, sha
from .vision_one_head import (
    PHASES,
    OneHeadTrial,
    _stage_identity,  # pyright: ignore[reportPrivateUsage]
    run_phase,
    scientific_trigger,
)

MODULE = "ICML_review.code.rgb_predictive"
PROTOCOL = (
    "Test whether first-layer predictive compression permits a single fixed a0 head to match "
    "the reported sixteen-head initializer. The sole change from the previous one-head trial "
    "is the TRAIN subspace used for the eight-to-four projection. With native centered TRAIN codes "
    "c, total covariance T and frozen first-layer code columns W, minimize squared first-layer "
    "change E||W(c-RPc)||^2 at rank four. Use orthonormal code coordinates, orient the retained "
    "space by native shared variance, fold the same centered OLS decoder R into the weights, "
    "and keep the original optimizer and data. A numerical predictive rank less than four, "
    "unresolved cutoff tie, singular selected covariance or failed native replay is a technical "
    "failure. Report conditioning, raw code and preactivation distortions, and zero-update "
    "paired raw forecast changes (system-bootstrap 95% intervals) against both upstream a0 "
    "and its native shared-variance projection on CAL/VAL. The surrogate is optimal only "
    "for first-layer preactivations. Immediate nonlinear forecast changes are descriptive. "
    "Authenticate a0 and joint, replay both first-window forecasts, and require a0 CAL MSE at "
    "most joint. Use the original 100k joint, one full 100k a0, 50k frozen adaptation and the "
    "original published same-seed tau. Admit after adaptation only if native CAL and VAL MSE "
    "are each at most 1.10 times the best-of-sixteen adapted initializer and minimum validation "
    "drag R-squared is at least .95. Then run both native sparse arms for 300k plus 200k. "
    "Require full MCC within .01 of reported or higher, at least .80 of its paired MCC gain, "
    "no higher total full SHD, full NMSE at most 1.10 times reported and minimum validation "
    "drag R-squared at least .95. Only a complete passing seed-zero pilot licenses seeds 1-4. "
    "The historical same-a0 native adaptation is available only on seed zero. Physical labels "
    "enter explicitly labelled validation diagnostics and endpoint evaluation only. A failed "
    "gate stops descendants. On failure retain the study as a controlled negative result."
)


def _training_codes(model: SCJEPA, data_dir: Path, device: torch.device) -> Tensor:
    source = data.RenderedInputs(data_dir, "train", device)
    cache = train.FrozenCache(model, {"train": source})
    encoder = cast(LaggedCodeEncoder, model.parameter_encoder)
    codes: list[Tensor] = []
    for begin in range(0, len(source), 64):
        states = cache.online["train"][begin : begin + 64, :, : model.context_len]
        actions = cache.actions["train"][begin : begin + 64, :, : model.context_len - 1]
        g, v = states.shape[:2]
        codes.append(
            encoder(states.flatten(0, 1), actions.flatten(0, 1)).reshape(g, v, -1).double().cpu()
        )
    return torch.cat(codes)


def _fold(payload: dict[str, Any], mean: Tensor, p: Tensor, r: Tensor) -> dict[str, Any]:
    state = dict(payload["model"])
    w, b = state["parameter_encoder.head.4.weight"], state["parameter_encoder.head.4.bias"]
    state["parameter_encoder.head.4.weight"] = (p @ w.double().cpu()).to(w)
    state["parameter_encoder.head.4.bias"] = (p @ (b.double().cpu() - mean)).to(b)
    w, b = state["predictor.first.weight"], state["predictor.first.bias"]
    code = w[:, -len(mean) :].double().cpu()
    state["predictor.first.weight"] = torch.cat((w[:, : -len(mean)], (code @ r).to(w)), 1)
    state["predictor.first.bias"] = (b.double().cpu() + code @ mean).to(b)
    return {"meta": dict(payload["meta"], code_dim=len(p), stage="project"), "model": state}


def _values(tensor: Tensor) -> list[Any]:
    return cast(list[Any], tensor.tolist())  # pyright: ignore[reportUnknownMemberType]


def _bootstrap(errors: dict[str, Tensor], seed: int) -> dict[str, Any]:
    errors = {k: v.double().cpu() for k, v in errors.items()}
    treatment = errors["predictive"]
    draws = torch.randint(
        len(treatment), (10000, len(treatment)), generator=torch.Generator().manual_seed(seed)
    )
    estimates = {}
    for name in sorted(errors.keys() - {"predictive"}):
        ref = errors[name]
        if ref.shape != treatment.shape or not (ref > 0).all():
            raise ValueError("paired native forecast losses must be finite, positive and aligned")
        deltas = treatment - ref
        quantiles = torch.quantile(
            deltas[draws].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        estimates[name] = {
            "difference": float(deltas.mean()),
            "difference_ci95": _values(quantiles),
            "ratio_of_means": float(treatment.mean() / ref.mean()),
        }
    return {
        "systems": len(treatment),
        "means": {k: float(v.mean()) for k, v in errors.items()},
        "predictive_minus": estimates,
        "group_mse": {k: _values(v) for k, v in errors.items()},
    }


@torch.no_grad()
def project(source: Path, data_dir: Path, output: Path, *, device: str) -> dict[str, Any]:
    """Fit one label-free map and measure its before-adaptation effect on the same systems."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = resolve_device(device)
    payload = train.load_checkpoint(source / "final.pt")
    upstream = train.model_from_checkpoint(payload).to(dev).eval()
    codes = _training_codes(upstream, data_dir, dev)
    rows = codes.flatten(0, 1)
    mean = rows.mean(0)
    centered = rows - mean
    shared, total = group_covariances(codes)
    w = payload["model"]["predictor.first.weight"][:, -rows.shape[1] :].double().cpu()
    dim = dgp.PARAM_DIM
    values, vectors = cast(tuple[Tensor, Tensor], torch.linalg.eigh(shared))  # pyright: ignore[reportUnknownMemberType]
    native_p = orient_columns(vectors[:, -dim:]).T
    native_r = total @ native_p.T @ cast(Tensor, torch.linalg.inv(native_p @ total @ native_p.T))  # pyright: ignore[reportUnknownMemberType]
    p, r = reduced_rank_map(total, shared, w, dim)
    diagnostics: dict[str, dict[str, Any]] = {}
    for k, pr, rec in (("native", native_p, native_r), ("predictive", p, r)):
        residual = centered - (centered @ pr.T) @ rec.T
        diagnostics[k] = {
            "code_relative_squared_error": float(residual.square().sum() / centered.square().sum()),
            "preactivation_relative_squared_error": float(
                (residual @ w.T).square().sum() / (centered @ w.T).square().sum()
            ),
            "selected_covariance_condition": float(
                cast(Tensor, torch.linalg.cond(pr @ total @ pr.T))  # pyright: ignore[reportUnknownMemberType]
            ),
            "projection": _values(pr),
            "reconstruction": _values(rec),
        }
    if (
        diagnostics["predictive"]["preactivation_relative_squared_error"]
        > diagnostics["native"]["preactivation_relative_squared_error"] + 1e-10
    ):
        raise RuntimeError("rank-constrained solution worsened its TRAIN objective")
    fitted = _fold(payload, mean, p, r)
    models = {
        "upstream": upstream,
        "native": train.model_from_checkpoint(_fold(payload, mean, native_p, native_r))
        .to(dev)
        .eval(),
        "predictive": train.model_from_checkpoint(fitted).to(dev).eval(),
    }
    controls = {}
    if source.resolve() == (ROOT / "seed0/acquire/a0").resolve():
        previous = OUTPUT / "vision-one-head-s0"
        previous_metrics = read(previous / "metrics.json")
        original = previous / "projected/final.pt"
        if (
            sha(original) != previous_metrics["checkpoint_sha256"]["projected"]
            or sha(source / "final.pt") != previous_metrics["checkpoint_sha256"]["a0"]
        ):
            raise ValueError("same-a0 archived projection lineage differs")
        controls = {
            str(original): sha(original),
            str(previous / "metrics.json"): sha(previous / "metrics.json"),
        }
        models["archived_native"] = (
            train.model_from_checkpoint(train.load_checkpoint(original)).to(dev).eval()
        )
    immediate = {}
    archived = read(source / "final.json")
    for split, key, seed in (
        ("calibration", "calibration_prediction_mse", 20261101),
        ("validation", "prediction_mse", 20261102),
    ):
        inputs = data.RenderedInputs(data_dir, split, dev)
        e = {k: group_errors(m, inputs) for k, m in models.items()}
        if not all(torch.isfinite(x).all() for x in e.values()) or not math.isclose(
            float(e["upstream"].mean()), float(archived[key]), rel_tol=1e-4, abs_tol=1e-8
        ):
            raise ValueError("upstream CAL/VAL first-window replay differs or non-finite forecast")
        immediate[split] = _bootstrap(e, seed)
    artifact: dict[str, Any] = {
        "input_sha256": controls,
        "source": str(source),
        "source_sha256": sha(source / "final.pt"),
        "data_manifest_sha256": sha(data_dir / "manifest.json"),
        "labels_used": False,
        "centered_train_mean": _values(mean),
        "shared_eigenvalues": _values(values),
        "first_layer_predictive_eigenvalues": _values(
            cast(Tensor, torch.linalg.eigvalsh(w @ total @ w.T))  # pyright: ignore[reportUnknownMemberType]
        ),
        "training": diagnostics,
        "immediate": immediate,
    }
    output.mkdir(parents=True, exist_ok=True)
    train._save(output / "final.pt", fitted)  # pyright: ignore[reportPrivateUsage]
    write_json(output / "final.json", artifact)
    return artifact


def _trigger(roots: Path) -> dict[str, Any]:
    previous = roots / "vision-one-head-s0"
    row = read(previous / "metrics.json")
    checks = adaptation_checks(row["adaptation"], row["reference_adaptation"])
    if (
        row["seed"] != 0
        or row["passed"]
        or row["reason"] != "adaptation"
        or row["adaptation_checks"] != checks
        or all(checks.values())
    ):
        raise ValueError("prior one-head adaptation receipt differs from its native gate")
    if row["candidate_sha256"] != sha(previous / "candidate.json"):
        raise ValueError("prior source-candidate receipt differs")
    hashes = row["checkpoint_sha256"]
    _stage_identity(previous / "adapt", row["commit"], hashes["adapt"], hashes["projected"])
    _stage_identity(previous / "projected", row["commit"], hashes["projected"], hashes["a0"])
    return scientific_trigger(roots) | {
        "one_head": {
            "path": str(previous / "metrics.json"),
            "sha256": sha(previous / "metrics.json"),
            "commit": row["commit"],
        }
    }


def main() -> None:
    """Launch one pinned phase or its recorded predictive-projection subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    if sys.argv[1:2] == ["project"]:
        parser.add_argument("mode", choices=("project",))
        parser.add_argument("--commit", required=True)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--data-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--device", required=True)
        args = parser.parse_args()
        _head(args.commit)
        project(args.source, args.data_dir, args.output, device=args.device)
        _head(args.commit)
        return
    parser.add_argument("commit")
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("phase", choices=PHASES)
    args = parser.parse_args()
    trial = OneHeadTrial(
        folder="rgb-predictive",
        module=MODULE,
        protocol=PROTOCOL,
        trigger=_trigger,
        project_command=(PYTHON, "-m", MODULE, "project", "--commit", args.commit),
    )
    run_phase(args.commit, args.seed, args.phase, trial=trial)


if __name__ == "__main__":
    main()
