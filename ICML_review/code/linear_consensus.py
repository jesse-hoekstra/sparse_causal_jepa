"""Fit a single-episode parameter readout to independent episodes of its training system."""

import argparse
import math
import shlex
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from diagnostics.masked_mlp_learned import acquisition
from diagnostics.masked_mlp_learned import run as syn
from diagnostics.masked_mlp_oracle import paper_analysis as paper
from diagnostics.shared.data import StateData, load_data
from diagnostics.shared.io import save_torch
from diagnostics.shared.oracle import objective_config
from diagnostics.vision_dimer import data, evaluate, train
from diagnostics.vision_dimer.model import LaggedCodeEncoder
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA
from scjepa.training.loop import TrainingObjective

from . import stable_readout
from .changed_queries import require
from .single_acquisition import COMPARISON, _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import DATA as SYN_DATA
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    ROOT,
    checkpoint_hashes,
    declared,
    read,
    sha,
)

SPLITS = ("calibration", "validation")
PROTOCOL = """Hypothesis: a fixed, already predictive code chart supplies a nonconstant target
for recovering the shared part of a system from one observed episode. Change only the target
of an affine least-squares refit of the last parameter-encoder layer. Control: reproduce the
teacher's code for the input episode. Treatment: predict the mean teacher code from the other
nine episodes in its training group. Both fits retain the teacher in unresolved feature
directions. The forecast net, coordinate system, hard graph, state encoders and original
optimizer budget are held fixed. Implementation cost is one TRAIN feature pass and a float64
least-squares solve. Use each group's first native context. No physical labels enter fitting.
Estimate all ranks, forecast changes and code dispersion on independent CAL and VAL systems.
Require at least ten percent lower maximum within/total generalized variance on BOTH splits
than the own-target control, with a negative upper paired 95% whole-system bootstrap bound
on VAL; full rank and at least 80% of the original minimum between-system eigenvalue in the
control chart on both splits. Require raw MSE within 10% of the original and control on both
splits, no additional aligned graph errors, MCC drop <=0.01, and VAL minimum physical
parameter R2 >= original minus 0.02 and >=0.80 synthetic or >=0.95 RGB. Reproduce the
original native forecast and validate the own-target control before interpretation. Physical
labels and simulator masks enter only the declared evaluation. A pilot passing all checks
licenses the remaining four seeds for that dataset. A failed scientific check closes this
branch; no tuning. No test scoring. A successful fit is a post-training loss on a single-episode
encoder and does not establish pointwise invariance, identifiability or the need for restarts.
"""


STABLE_PROTOCOL = """Hypothesis: frozen single-episode features include a more reproducible
predictive chart than the final readout. Change only the last affine layer. Choose its
TRAIN subspace by minimizing within-system code variance under unit total covariance, then
fit an affine map from TRAIN system means in that subspace to the original teacher means.
This objective explicitly protects rank during selection. Its within-system guarantee is
in sample and relative to a linear feature class. Fail the pilot if the boundary is
unresolved or alignment cannot retain all output directions. Use an exact copy of the
original last-layer coefficients as the paired control. Frozen trunk, state encoders, graph
gates, group membership, single-episode deployment and original upstream budget are shared.
Do one label-free feature pass and FP64 solve, with no SGD or added scalar penalty weight.
On each CAL and VAL cohort require 10% lower worst generalized within fraction, rank retained,
minimum between variance in original units at least 0.80, raw MSE within 10% of the original
control. The paired VAL 95% interval for the worst change must be entirely below zero.
Require VAL MCC loss at most 0.01, no extra aligned graph errors and minimum parameter R2
at least the larger of native minus 0.02 and the relevant floor (0.80 synthetic, 0.95 RGB).
Report the actual native tolerance separately. No physical labels are used to fit the map.
Advance a passing seed-zero fit separately within each benchmark. A failure retains its
receipt and any existing endpoint for independent validation queries; no test is scored."""


def eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Diagonalize one or a batch of symmetric empirical matrices."""
    return cast(tuple[Tensor, Tensor], torch.linalg.eigh((matrix + matrix.mT) / 2))  # pyright: ignore[reportUnknownMemberType]


@torch.no_grad()
def fit_readout(
    features: Tensor, layer: nn.Linear
) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, Any]]:
    """Solve both centered normal equations with a single feature covariance factorization."""
    x = features.detach().cpu().double()
    if x.ndim != 3 or min(x.shape[:2]) < 2 or x.shape[-1] != layer.in_features:
        raise ValueError("expected grouped features with distinct donor episodes")
    if not torch.isfinite(x).all():
        raise ValueError("nonfinite training features")
    w, b = layer.weight.detach().cpu().double(), layer.bias.detach().cpu().double()
    teacher = x @ w.T + b
    target = (teacher.sum(1, keepdim=True) - teacher) / (x.shape[1] - 1)
    mean = x.mean((0, 1))
    design = (x - mean).flatten(0, 1)
    gram = design.T @ design / len(design)
    values, vectors = eigh(gram)
    cutoff = max(design.shape) * torch.finfo(design.dtype).eps * float(values[-1])
    active = values > cutoff
    if not bool(active.any()):
        raise ValueError("no resolved training features")
    basis = vectors[:, active]
    coefficients: dict[str, tuple[Tensor, Tensor]] = {}
    fits: dict[str, float] = {}
    for name, y in (("own", teacher), ("sibling", target)):
        residual = (y - teacher).flatten(0, 1)
        offset = residual.mean(0)
        cross = design.T @ (residual - offset) / len(design)
        delta = basis @ ((basis.T @ cross) / values[active, None])
        new_w, new_b = w + delta.T, b + offset - mean @ delta
        coefficients[name] = (new_w, new_b)
        fits[name] = float((x @ new_w.T + new_b - y).square().mean())
    return coefficients, {
        "groups": x.shape[0],
        "episodes": x.shape[1],
        "features": x.shape[2],
        "design_rank": int(active.sum()),
        "eigenvalue_cutoff": cutoff,
        "feature_spectrum": values.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "target_mse": fits,
    }


def moments(codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Per-system sufficient statistics for population-form variance fractions."""
    x = codes.detach().cpu().double()
    x = x - x.mean((0, 1))
    means = x.mean(1)
    return means, x.mT @ x / x.shape[1], (x - means[:, None]).mT @ (x - means[:, None]) / x.shape[1]


def spectra(stats: tuple[Tensor, Tensor, Tensor], weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the total and within covariance and the largest resolved within fraction."""
    means, second, within = stats
    mu = weights @ means
    total = (weights @ second.flatten(1)).reshape(-1, means.shape[-1], means.shape[-1]) - mu[
        :, :, None
    ] * mu[:, None, :]
    w = (weights @ within.flatten(1)).reshape_as(total)
    eig, v = eigh(total)
    ok = eig > eig[:, -1:] * (len(means) * 10 * torch.finfo(eig.dtype).eps)
    whitener = (
        v * torch.where(ok, eig.clamp_min(torch.finfo(eig.dtype).tiny).rsqrt(), 0)[:, None, :]
    )
    normalized, _ = eigh(whitener.mT @ w @ whitener)
    return eig, total - w, normalized[:, -1]


def uncertainty(
    control: Tensor, trial: Tensor, first_mse: Tensor, second_mse: Tensor
) -> dict[str, Any]:
    """Resample matched systems, retaining their entire native collection of episodes."""
    n = len(control)
    draws = torch.randint(n, (1000, n), generator=torch.Generator().manual_seed(9047))
    weights = torch.zeros((len(draws), n), dtype=torch.float64)
    weights.scatter_add_(1, draws, torch.ones_like(weights))
    weights = torch.cat((torch.ones_like(weights[:1]), weights)) / n
    results = [spectra(moments(x), weights) for x in (control, trial)]
    views = control.shape[1]
    correction = (n * views - 1) / (n * (views - 1))
    old, new = (r[2] * correction for r in results)
    eb, vb = eigh(results[0][1][0])
    if float(eb.min()) <= 0:
        raise ValueError("unresolved between-system teacher covariance")
    v = vb / eb.sqrt()
    relative, _ = eigh(v.T @ results[1][1][0] @ v)
    delta = new - old
    mse = weights @ (second_mse - first_mse).double()
    probs = torch.tensor([0.025, 0.975], dtype=torch.float64)
    eg = results[1][0][0]
    return {
        "groups": n,
        "control_worst": float(old[0]),
        "trial_worst": float(new[0]),
        "worst_change": float(delta[0]),
        "worst_change_95": torch.quantile(delta[1:], probs).tolist(),  # pyright: ignore[reportUnknownMemberType]
        "forecast_change": float(mse[0]),
        "forecast_change_95": torch.quantile(mse[1:], probs).tolist(),  # pyright: ignore[reportUnknownMemberType]
        "between_minimum_control_units": float(relative[0]),
        "dimension": len(eg),
        "trial_rank": int((eg > eg[-1] * n * 10 * torch.finfo(eg.dtype).eps).sum()),
    }


def synthetic_model(seed: int, path: Path, device: torch.device) -> tuple[SCJEPA, dict[str, Any]]:
    """Restore the reported synthetic model weights without using optimizer state."""
    model = syn.build_control("full", seed, predictor_activation="gelu")
    payload = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), {"meta": {"seed": seed, "stage": "full", "activation": "gelu"}}


def synthetic_eval(
    model: SCJEPA, sets: dict[str, StateData], device: torch.device
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    """Reuse native calibration-fitted metrics and the paper's parameter-column assignment."""
    obj = TrainingObjective(objective_config("full", 0.0)).to(device)
    results, matching = syn._collect(model, sets, 16, device, obj, None)  # pyright: ignore[reportPrivateUsage]
    cal = results["calibration"].samples
    codes = cal["codes"].double()
    order = paper._code_order(  # pyright: ignore[reportPrivateUsage]
        codes.flatten(0, 1), cal["theta"].double()[:, None].expand_as(codes).flatten(0, 1)
    )  # pyright: ignore[reportPrivateUsage]
    graph = paper._aligned_graphs(dict(model.state_dict()), matching["online_columns"])  # pyright: ignore[reportPrivateUsage]
    aligned = paper._code_aligned_metrics(graph["parameter"][:, order])  # pyright: ignore[reportPrivateUsage]
    out: dict[str, Any] = {}
    for s, row in results.items():
        m = row.metrics
        out[s] = {
            "mse": m["prediction"],
            "mcc": m["state_mcc"],
            "r2": m["code_r2_min"],
            "shd": sum(
                (
                    m["masks/state_aligned_shd"],
                    m["masks/action_aligned_shd"],
                    aligned["masks/parameter_code_aligned_shd"],
                )
            ),
        }
    return out, {s: x.samples for s, x in results.items()}


@torch.no_grad()
def rgb_features(model: SCJEPA, inputs: data.RenderedInputs) -> Tensor:
    """Extract the trained penultimate features of every first-window training episode."""
    require(inputs.split == "train", "Training features must come from TRAIN")
    enc = cast(LaggedCodeEncoder, model.parameter_encoder)
    parts: list[Tensor] = []
    for batch in train.rendered_batches(inputs, size=8):
        obs = batch["observations"]
        states = model.online_encoder.encode_sequence(obs[:, :, : model.context_len].flatten(0, 1))
        actions = batch["actions"][:, :, : model.context_len - 1].flatten(0, 1)
        parts.append(enc.features(states, actions).reshape(*obs.shape[:2], -1).double().cpu())
    return torch.cat(parts)


def rgb_eval(
    model: SCJEPA, sets: dict[str, data.RenderedInputs], *, data_dir: Path = DATA
) -> tuple[dict[str, Any], dict[str, dict[str, Tensor]]]:
    """Score fixed CAL/VAL first windows using the released estimators."""
    samples = {s: evaluate.collect(model, train.rendered_batches(sets[s])) for s in SPLITS}
    labels: dict[str, dict[str, Tensor]] = {}
    for split in SPLITS:
        raw = data.load_labels(data_dir, split)
        # Mirror the released evaluator: rendered_batches selects start zero in every episode.
        labels[split] = {"states": raw["states"][:, :, : sets[split].frames], "theta": raw["theta"]}
    metrics: dict[str, Any] = {}
    for s in SPLITS:
        r = evaluate.evaluate(
            model, samples["calibration"], labels["calibration"], samples[s], labels[s]
        )
        metrics[s] = {
            "mse": r["prediction_mse"],
            "nmse": r["nmse"],
            "spearman_assignment": r["spearman_assignment"],
            "mcc": r["mcc"],
            "r2": r["parameter_r2_min"],
            "shd": sum(r[k + "_shd"] for k in ("state", "action", "parameter")),
        }
    return metrics, samples


def endpoint(dataset: str, seed: int) -> Path:
    """Resolve the reported full endpoint for the requested benchmark."""
    if dataset == "vision":
        return ROOT / f"seed{seed}/sparse-500k/full/final.pt"
    root = "masked-mlp-gelu-s0-350k-20261102" if seed == 0 else "masked-mlp-gelu-350k-20261102"
    return Path("outputs/diagnostics") / root / f"full-s{seed}/results/last.pt"


@torch.no_grad()
def run(commit: str, dataset: str, seed: int, *, stable: bool = False) -> None:
    """Fit, evaluate and archive one frozen-feature readout comparison."""
    if dataset not in {"synthetic", "vision", "vision-window"}:
        raise ValueError("unknown linear consensus benchmark")
    selector = "vision-window" if dataset in {"vision", "vision-window"} else dataset
    invoker = dataset
    dataset = "vision" if selector == "vision-window" else "synthetic"
    method = f"stable-readout-{dataset}" if stable else f"linear-consensus-{selector}"
    out = OUTPUT / f"{method}-s{seed}"
    _head(commit)
    path = endpoint(dataset, seed)
    expected = checkpoint_hashes()[str(path)]
    identity = DATA / "manifest.json" if dataset == "vision" else SYN_DATA / "completion.json"
    source_id = sha(identity)
    cmd = shlex.join(
        [PYTHON, "-m", "ICML_review.code.linear_consensus", commit, invoker, str(seed)]
        + (["--stable"] if stable else [])
    )
    note = (
        f"# {method}, seed {seed}\n\n{STABLE_PROTOCOL if stable else PROTOCOL}\n\n"
        f"Git {commit}\nData {identity} SHA-256 {source_id}\nReported {path} SHA-256 {expected}\n"
        f"\n    PYTHONPATH=$PWD:$PWD/src {cmd}\n"
    )
    if dataset == "vision" and not stable:
        # Both equivalent CLI spellings resolve to one revision-bound result.
        prefix = f"PYTHONPATH=$PWD:$PWD/src {PYTHON} -m ICML_review.code.linear_consensus {commit}"
        note = note.split("    PYTHONPATH=", maxsplit=1)[0] + (
            "Evaluate the physical state labels from the same first window as the RGB inputs, "
            "as in the release evaluator. The earlier run remains in its original directory. "
            "Equivalent commands follow.\n\n"
            f"    {prefix} vision-window {seed}\n    {prefix} vision {seed}\n"
        )
    with file_lock(out / ".lock"):
        declared(out / "NOTE.md", note)
        if seed:
            pilot = read(OUTPUT / f"{method}-s0/metrics.json")
            require(pilot["passed"] and pilot["commit"] == commit, "Pilot did not qualify")
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        require(sha(path) == expected, "Source checkpoint differs")
        if (out / "metrics.json").exists():
            r = read(out / "metrics.json")
            require(
                r["commit"] == commit
                and r["note_sha256"] == sha(out / "NOTE.md")
                and (
                    (r.get("reason") == "unresolved_training_chart" and not r["passed"])
                    or sha(out / "final.pt") == r["checkpoint_sha256"]
                ),
                "Stored run changed",
            )
            return
        _idle_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(1)
        device = torch.device("cuda")
        start = time.monotonic()
        if dataset == "vision":
            payload = train.load_checkpoint(path)
            model = train.model_from_checkpoint(payload, phase="full").to(device).eval()
            rgb = {s: data.RenderedInputs(DATA, s, device) for s in ("train", *SPLITS)}
            feats = rgb_features(model, rgb["train"])
            reference = read(path.with_name("final.json"))
            rmses = {
                "calibration": reference["calibration_prediction_mse"],
                "validation": reference["prediction_mse"],
            }

            evaluate_one = partial(rgb_eval, model, rgb)
            tau = float(payload["config"]["tau"])
            ref_shd = sum(reference[k + "_shd"] for k in ("state", "action", "parameter"))
        else:
            model, payload = synthetic_model(seed, path, device)
            sets, _ = load_data(SYN_DATA)
            feats = acquisition.train_contexts(model, sets["train"], device, codes=False)
            reference = read(COMPARISON)["per_run"]["full"][str(seed)]
            rmses = {s: read(path.with_name(s + ".json"))["metrics"]["prediction"] for s in SPLITS}

            evaluate_one = partial(synthetic_eval, model, sets, device)
            tau = float(read(path.with_name("config.json"))["tau_prediction"])
            ref_shd = sum(
                reference[k]
                for k in (
                    "masks/state_aligned_shd",
                    "masks/action_aligned_shd",
                    "masks/parameter_code_aligned_shd",
                )
            )

        layer = cast(
            nn.Linear, cast(nn.Sequential, model.parameter_encoder.get_submodule("head"))[-1]
        )
        original_readout = (layer.weight.detach().cpu().clone(), layer.bias.detach().cpu().clone())
        try:
            fits, training = (stable_readout.fit_readout if stable else fit_readout)(feats, layer)
        except stable_readout.StableReadoutError as failure:
            _head(commit)
            write_json(
                out / "metrics.json",
                {
                    "commit": commit,
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "command": cmd,
                    "source": str(path),
                    "source_sha256": expected,
                    "data_identity": source_id,
                    "note_sha256": sha(out / "NOTE.md"),
                    "passed": False,
                    "reason": "unresolved_training_chart",
                    "fit": {"reason": failure.reason, "diagnostics": failure.diagnostics},
                    "checks": {"training_subspace": False},
                },
            )
            return
        del feats
        elapsed = time.monotonic() - start
        checks: dict[str, bool] = {}
        arms: dict[str, Any] = {}
        kept: dict[str, dict[str, tuple[Tensor, Tensor]]] = {}
        for key, (weight, bias) in fits.items():
            _head(commit)
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
            arms[key], samples = evaluate_one()
            kept[key] = {}
            for s, x in samples.items():
                mse = (
                    (x["predictions"].double() - x["targets"].double()).square().flatten(1).mean(1)
                )
                kept[key][s] = (x["codes"], mse)
            if key == "own":
                if not all(
                    math.isclose(arms[key][s]["mse"], rmses[s], rel_tol=1e-4, abs_tol=1e-8)
                    for s in SPLITS
                ):
                    raise RuntimeError(f"Reference forecast replay failed: {arms[key]}, {rmses}")
                require(
                    torch.equal(layer.weight.cpu(), original_readout[0])
                    and torch.equal(layer.bias.cpu(), original_readout[1]),
                    "Reference target changed",
                )
        ref_mcc = reference["mcc"] if dataset == "vision" else reference["state_mcc"]
        ref_r2 = reference["parameter_r2_min"] if dataset == "vision" else reference["code_r2_min"]
        old = arms["own"]["validation"]
        require(
            math.isclose(old["mcc"], ref_mcc, abs_tol=1e-4)
            and math.isclose(old["r2"], ref_r2, abs_tol=1e-4)
            and old["shd"] == ref_shd,
            "Original labelled metrics did not reproduce",
        )
        record: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for s in SPLITS:
            c, ce = kept["own"][s]
            t, te = kept["sibling"][s]
            u = uncertainty(c, t, ce, te)
            record[s] = u
            checks[s + "_invariance"] = u["trial_worst"] <= 0.90 * u["control_worst"]
            checks[s + "_forecast"] = arms["sibling"][s]["mse"] <= 1.10 * min(
                rmses[s], arms["own"][s]["mse"]
            )
            checks[s + "_rank"] = (
                u["trial_rank"] == u["dimension"] and u["between_minimum_control_units"] >= 0.8
            )
            for name, v in (
                ("control_codes", c),
                ("trial_codes", t),
                ("control_mse", ce),
                ("trial_mse", te),
            ):
                arrays[s + "_" + name] = v.cpu().numpy()
        v = arms["sibling"]["validation"]
        checks.update(
            upper_bound=record["validation"]["worst_change_95"][1] < 0,
            parameter=v["r2"] >= max(old["r2"] - 0.02, 0.95 if dataset == "vision" else 0.8),
            graph=v["shd"] <= old["shd"],
            state=v["mcc"] >= old["mcc"] - 0.01,
        )
        _head(commit)
        payload = {k: v for k, v in payload.items() if k in ("meta", "config")}
        save_torch(out / "final.pt", payload | {"model": model.cpu().state_dict()})
        if dataset == "vision":
            write_json(out / "final.json", arms["sibling"]["validation"])
        tmp = out / "system-samples.tmp.npz"
        np.savez_compressed(tmp, **arrays)  # pyright: ignore[reportUnknownMemberType]
        tmp.replace(out / "system-samples.npz")
        write_json(
            out / "metrics.json",
            {
                "commit": commit,
                "seed": seed,
                "dataset": dataset,
                "method": method,
                "note_sha256": sha(out / "NOTE.md"),
                "source": str(path),
                "source_sha256": expected,
                "data_identity": source_id,
                "command": cmd,
                "feature_and_fit_seconds": elapsed,
                "fit": training,
                "arms": arms,
                "historical_raw_mse": rmses,
                "tau": tau,
                "trial_forecast_over_tau": {s: arms["sibling"][s]["mse"] / tau for s in SPLITS},
                "comparisons": record,
                "checks": checks,
                "passed": all(checks.values()),
                "samples_sha256": sha(out / "system-samples.npz"),
                "checkpoint_sha256": sha(out / "final.pt"),
            },
        )


def main() -> None:
    """Run the seed-zero pilot or a qualified replication on the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("dataset", choices=("synthetic", "vision", "vision-window"))
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument(
        "--stable", action="store_true", help="fit a minimum within-variance subspace"
    )
    args = parser.parse_args()
    run(args.commit, args.dataset, args.seed, stable=args.stable)


if __name__ == "__main__":
    main()
