"""Estimate a shared frozen code from two disjoint support episodes on development systems."""

import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from diagnostics.masked_mlp_learned import run as syn
from diagnostics.masked_mlp_oracle import paper_analysis as paper
from diagnostics.shared.data import load_data
from diagnostics.shared.oracle import EvaluationResult
from diagnostics.vision_dimer import data, evaluate, train
from scjepa.eval.recovery import fit_coordinate_match
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, cross_episode_parameters

from .linear_consensus import (
    SPLITS,
    SYN_DATA,
    eigh,
    endpoint,
    rgb_eval,
    synthetic_eval,
    synthetic_model,
)
from .single_acquisition import COMPARISON, _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    checkpoint_hashes,
    declared,
    read,
    sha,
)

DRAWS = 2000
BOOT_SEEDS = {"calibration": 20261126, "validation": 20261127}
PROTOCOL = """Hypothesis: the exact minimizer of sum_j ||u-c_j||^2 for two frozen support codes,
their arithmetic mean, reduces within-system dispersion at acceptable deployment cost. No SGD or
extra learned coefficients. Only the number of independent support episodes changes. For
conditionally independent identically distributed codes given the system, with finite
covariance, the conditional means stay fixed, expected within covariance W becomes W/2 and
between-system covariance B stays fixed. This is a change from one observed support episode to
two, not an invariance guarantee for a single episode.

Use the five disjoint support pairs (0,1),(2,3),(4,5),(6,7),(8,9) once each for dispersion,
rank, labelled parameter evaluation and calibration-only assignments; compare with all ten
distinct single-episode codes. For forecast recipient i, keep its original deterministic native
donor i-1 and add i-2 cyclically. Both are distinct from i. Use original two-step histories,
actions, target and hard gates. Reproduce stored original native forecast and original
state/parameter/graph scores before interpreting differences. Report original numeric
tolerance, forecast/tau, hard feasibility, paired raw errors and deltas. The gate requires raw
pair forecast MSE <=1.05 times native, and worst-direction within/total fraction <=0.80 times
native, on BOTH CAL and VAL. Require full code covariance rank for single and pair on both
splits and a strictly negative upper paired whole-system 95% validation bound for the change in
worst-direction fraction. Require VAL minimum physical parameter R2 within .02 of original and
>=.80 synthetic / .95 RGB, state MCC no lower (numerical tolerance 1e-8) and no additional
aligned SHD. Physical references and masks enter only labelled evaluation. Separately report
rank and mean dispersion in fixed CAL-single covariance units.

Use unbiased within-system sample covariance for each donor count and estimate between-system
covariance from the unbiased covariance of group means minus W/R. Use corrected total B+W.
Bootstrap 2000 aligned whole systems per split with fixed independent split seeds, keeping
every episode together; unresolved bootstrap ranks invalidate the interval gate. A passing seed
zero licenses the four remaining seeds only on its dataset. Otherwise stop and retain the
outcome. No test. Physical interventions are assessed separately and are not determined by this
forecast comparison."""


def donor_codes(codes: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return disjoint pair estimates, native donors and paired donors in recipient order."""
    if codes.ndim != 3 or codes.shape[1] != 10 or not torch.isfinite(codes).all():
        raise ValueError("expected ten finite single-episode codes per system")
    native = cross_episode_parameters(codes)
    return (codes[:, ::2] + codes[:, 1::2]) / 2, native, (native + codes.roll(2, 1)) / 2


def _moments(x: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    n, r, d = x.shape
    x = x.double().cpu()
    means = x.mean(1)
    means = means - means.mean(0)
    centered = x - x.mean(1, keepdim=True)
    within = (weights @ (centered.mT @ centered / (r - 1)).flatten(1)).reshape(-1, d, d)
    mu = weights @ means
    gm = (weights @ (means[:, :, None] * means[:, None, :]).flatten(1)).reshape(-1, d, d) - mu[
        :, :, None
    ] * mu[:, None, :]
    between = gm * (n / (n - 1)) - within / r
    return within, between, between + within


def _spectrum(w: Tensor, t: Tensor, observations: int) -> tuple[Tensor, Tensor, Tensor]:
    eig, vec = eigh(t)
    cutoff = (observations * torch.finfo(eig.dtype).eps * eig[:, -1:]).clamp_min(
        torch.finfo(eig.dtype).tiny
    )
    rank = (eig > cutoff).sum(1)
    white = vec / eig.clamp_min(cutoff).sqrt()[:, None, :]
    ratios, _ = eigh(white.mT @ w @ white)
    return ratios, rank, white


def _tolist(x: Tensor) -> list[Any]:
    return cast(list[Any], x.tolist())  # pyright: ignore[reportUnknownMemberType]


def dispersion(
    single: Tensor, pair: Tensor, split: str, common: Tensor | None, *, draws: int = DRAWS
) -> tuple[dict[str, Any], Tensor, Tensor]:
    """Bootstrap aligned complete systems, correcting covariance estimates for donor count."""
    n = len(single)
    if n < 2 or pair.shape != (n, 5, single.shape[2]) or split not in SPLITS or draws < 2:
        raise ValueError("unsupported cohort or pair geometry")
    ids = torch.randint(n, (draws, n), generator=torch.Generator().manual_seed(BOOT_SEEDS[split]))
    weights = torch.zeros((draws, n), dtype=torch.float64)
    weights.scatter_add_(1, ids, torch.ones_like(ids, dtype=weights.dtype))
    weights = torch.cat((torch.ones((1, n), dtype=weights.dtype), weights)) / n
    summary: dict[str, Any] = {}
    worst: list[Tensor] = []
    valid = torch.ones(draws + 1, dtype=torch.bool)
    for name, c in (("single", single), ("pair", pair)):
        w, b, t = _moments(c, weights)
        ratios, ranks, white = _spectrum(w, t, n * c.shape[1])
        full = ranks == c.shape[2]
        valid &= full
        if common is None:
            if split != "calibration" or name != "single":
                raise ValueError("missing CAL single covariance chart")
            # Keep only resolved CAL directions. A truncated chart fails the rank gate.
            common = white[0, :, -int(ranks[0]) :] if ranks[0] > 0 else white[0, :, :0]
        worst.append(ratios[:, -1])
        summary[name] = {
            "groups": n,
            "supports_per_group": c.shape[1],
            "dimension": c.shape[2],
            "rank": int(ranks[0]),
            "worst_within_total": float(ratios[0, -1]) if full[0] else None,
            "mean_within_total": float(ratios[0].mean()) if full[0] else None,
            "within": _tolist(w[0]),
            "between": _tolist(b[0]),
            "total": _tolist(t[0]),
            "within_cal_single_trace": float((common.T @ w[0] @ common).trace()),
            "between_cal_single_trace": float((common.T @ b[0] @ common).trace()),
            "total_cal_single_eigenvalues": _tolist(eigh(common.T @ t[0] @ common)[0])
            if common.shape[1]
            else [],
        }
    assert common is not None
    delta = worst[1] - worst[0]
    summary["worst_difference"] = float(delta[0]) if valid[0] else None
    summary["valid_bootstrap_draws"] = int(valid[1:].sum())
    summary["worst_difference_ci95"] = (
        _tolist(torch.quantile(delta[1:], delta.new_tensor([0.025, 0.975])))
        if valid.all()
        else None
    )
    return summary, common, weights


@torch.no_grad()
def forecast(
    model: SCJEPA, samples: dict[str, Tensor], actions: Tensor, device: torch.device
) -> tuple[Tensor, Tensor, float]:
    """Replay released rollouts and vary only the donor code passed to the predictor."""
    if not model.autoregressive:
        raise ValueError("this protocol assumes released autoregressive dynamics")
    c = samples["codes"].float()
    _, single, pair = donor_codes(c)
    states = samples["online_states"] if "online_states" in samples else samples["states"]
    h = states[:, :, model.context_len - model.predictor.history_len : model.context_len]
    a = actions[
        :,
        :,
        model.context_len - model.predictor.history_len : model.context_len
        + samples["targets"].shape[2]
        - 1,
    ]
    losses: list[list[Tensor]] = [[], []]
    maximum = 0.0
    for start in range(0, len(c), 16):
        sl = slice(start, start + 16)
        truth = samples["targets"][sl]
        for j, support in enumerate((single, pair)):
            pred = (
                model.rollout(
                    h[sl].flatten(0, 1).to(device=device, dtype=torch.float32),
                    support[sl].flatten(0, 1).to(device),
                    truth.shape[2],
                    a[sl].flatten(0, 1).to(device=device, dtype=torch.float32),
                )
                .reshape_as(truth)
                .cpu()
            )
            if j == 0:
                original = samples["predictions"][sl].cpu().to(pred)
                maximum = max(maximum, float((pred - original).abs().max()))
                torch.testing.assert_close(pred, original, atol=3e-5, rtol=1e-5)
            losses[j].append((pred.double() - truth.cpu().double()).square().mean((1, 2, 3)))
    return torch.cat(losses[0]), torch.cat(losses[1]), maximum


def physical(
    model: SCJEPA,
    samples: dict[str, dict[str, Tensor]],
    codes: dict[str, Tensor],
    dataset: str,
    native: dict[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    """Evaluate a code estimator with physical labels and assignments fitted only on CAL."""
    cal = samples["calibration"]
    codes = {split: x.double() for split, x in codes.items()}
    result: dict[str, Any] = {}
    if dataset == "vision":
        theta = {s: data.load_labels(data_dir, s)["theta"].double() for s in SPLITS}
        code_match = evaluate.code_parameter_assignment(codes["calibration"], theta["calibration"])
        labels = data.load_labels(data_dir, "calibration")
        labelled_window = labels["states"][:, :, : cal["states"].shape[2]]
        state_match = evaluate.state_metrics(
            cal["states"], labelled_window, cal["states"], labelled_window
        )["assignment"]
        graph = evaluate.graph_metrics(model, state_match, code_match)
        for split in SPLITS:
            retention = evaluate.parameter_retention(
                codes["calibration"], theta["calibration"], codes[split], theta[split]
            )
            result[split] = {
                "r2": retention["parameter_r2_min"],
                "per_parameter": retention["parameter_r2"],
                "mcc": native[split]["mcc"],
                "state_assignment": state_match,
                "parameter_assignment": code_match,
                "shd": sum(graph[k + "_shd"] for k in ("state", "action", "parameter")),
                "shd_by_type": {k: graph[k + "_shd"] for k in ("state", "action", "parameter")},
            }
    else:
        online = cal["online_states"].reshape(-1, 6).double()
        truth = cal["states"].reshape(-1, 6).double()
        state_match = list(fit_coordinate_match(online, truth, method="pearson").columns)
        z = codes["calibration"].double()
        code_match = paper._code_order(  # pyright: ignore[reportPrivateUsage]
            z.flatten(0, 1), cal["theta"].double()[:, None].expand_as(z).flatten(0, 1)
        )  # pyright: ignore[reportPrivateUsage]
        graphs = paper._aligned_graphs(dict(model.state_dict()), state_match)  # pyright: ignore[reportPrivateUsage]
        par = paper._code_aligned_metrics(graphs["parameter"][:, code_match])  # pyright: ignore[reportPrivateUsage]
        archived_graph = syn._graph_metrics(syn._predictor(model), state_match)  # pyright: ignore[reportPrivateUsage]
        ev = {
            s: EvaluationResult({}, {"codes": codes[s], "theta": samples[s]["theta"]})
            for s in SPLITS
        }
        for split in SPLITS:
            ret = syn.code_metrics(ev["calibration"], ev[split])
            result[split] = {
                "r2": ret["code_r2_min"],
                "per_parameter": [ret[f"code_r2/theta_{i}"] for i in range(1, 7)],
                "mcc": native[split]["mcc"],
                "state_assignment": state_match,
                "parameter_assignment": code_match,
                "shd_by_type": {
                    "state": archived_graph["masks/state_aligned_shd"],
                    "action": archived_graph["masks/action_aligned_shd"],
                    "parameter": par["masks/parameter_code_aligned_shd"],
                },
                "shd": archived_graph["masks/state_aligned_shd"]
                + archived_graph["masks/action_aligned_shd"]
                + par["masks/parameter_code_aligned_shd"],
            }
    return result


def scientific_checks(rows: dict[str, Any], dataset: str) -> dict[str, bool]:
    """Recompute each prespecified scientific decision from unrounded numbers."""
    checks: dict[str, bool] = {}
    for s in SPLITS:
        x = rows[s]
        ds = x["dispersion"]
        old, new = ds["single"], ds["pair"]
        checks[s + "_rank"] = all(r["rank"] == r["dimension"] for r in (old, new))
        checks[s + "_invariance"] = (
            checks[s + "_rank"] and new["worst_within_total"] <= 0.8 * old["worst_within_total"]
        )
        checks[s + "_forecast"] = x["forecast"]["pair_mse"] <= 1.05 * x["forecast"]["native_mse"]
    v = rows["validation"]
    interval = v["dispersion"]["worst_difference_ci95"]
    a, b = v["physical"]["single"], v["physical"]["pair"]
    checks.update(
        validation_upper_bound=interval is not None and interval[1] < 0,
        parameter_retention=b["r2"] >= max(a["r2"] - 0.02, 0.95 if dataset == "vision" else 0.80),
        state_mcc=b["mcc"] >= a["mcc"] - 1e-8,
        aligned_graph=b["shd"] <= a["shd"],
    )
    return checks


def _forecast_summary(
    native: Tensor, changed: Tensor, weights: Tensor, tau: float
) -> dict[str, Any]:
    old, new = weights @ native.double(), weights @ changed.double()
    if not (old > 0).all() or not (new >= 0).all() or not math.isfinite(tau) or tau <= 0:
        raise ValueError("invalid forecast loss or native tolerance")
    d = new - old
    ratio = new / old
    return {
        "native_mse": float(old[0]),
        "pair_mse": float(new[0]),
        "change": float(d[0]),
        "change_ci95": _tolist(torch.quantile(d[1:], d.new_tensor([0.025, 0.975]))),
        "pair_over_native": float(ratio[0]),
        "ratio_ci95": _tolist(torch.quantile(ratio[1:], d.new_tensor([0.025, 0.975]))),
        "tau": tau,
        "native_over_tau": float(old[0] / tau),
        "pair_over_tau": float(new[0] / tau),
        "native_feasible": bool(old[0] <= tau),
        "pair_feasible": bool(new[0] <= tau),
    }


def _same_numbers(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        left, right = cast(dict[str, object], a), cast(dict[str, object], b)
        return left.keys() == right.keys() and all(
            _same_numbers(v, right[k]) for k, v in left.items()
        )
    if isinstance(a, list) and isinstance(b, list):
        first, second = cast(list[object], a), cast(list[object], b)
        return len(first) == len(second) and all(
            _same_numbers(x, y) for x, y in zip(first, second, strict=True)
        )
    if isinstance(a, float) and isinstance(b, int | float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def validate(out: Path) -> dict[str, Any]:
    """Verify the receipt, sample digest, disjoint pairs, ranks, paired intervals and gates."""
    row = read(out / "metrics.json")
    if (
        sha(out / "NOTE.md") != row["note_sha256"]
        or sha(out / "system-samples.npz") != row["samples_sha256"]
    ):
        raise ValueError("support-pair declaration or samples changed")
    if row["command"] not in (out / "NOTE.md").read_text() or row["commit"] not in row["command"]:
        raise ValueError("declared command differs")
    common = None
    with np.load(out / "system-samples.npz", allow_pickle=False) as saved:
        for s in SPLITS:
            arr = {
                name: torch.from_numpy(saved[s + "_" + name])  # pyright: ignore[reportUnknownMemberType]
                for name in ("single_codes", "pair_codes", "native_mse", "pair_mse")
            }
            one, two = arr["single_codes"], arr["pair_codes"]
            if not torch.equal(donor_codes(one)[0], two):
                raise ValueError("stored donors are not the five adjacent disjoint pair means")
            info, common, w = dispersion(one, two, s, common)
            expected = row["splits"][s]
            if not _same_numbers(expected["dispersion"], info) or not _same_numbers(
                expected["forecast"],
                _forecast_summary(arr["native_mse"], arr["pair_mse"], w, row["tau"]),
            ):
                raise ValueError("stored support measurement differs from its system samples")
    checks = scientific_checks(row["splits"], row["dataset"])
    if checks != row["checks"] or all(checks.values()) != row["passed"]:
        raise ValueError("support-pair scientific gate differs")
    return row


@torch.no_grad()
def run(commit: str, dataset: str, seed: int) -> None:
    """Estimate and evaluate a two-support shared code at a fixed reported full endpoint."""
    if dataset not in ("synthetic", "vision") or seed not in range(5):
        raise ValueError("unsupported dataset or seed")
    out = OUTPUT / f"support-pair-{dataset}-s{seed}"
    src = endpoint(dataset, seed)
    expected = checkpoint_hashes()[str(src)]
    identity = (DATA / "manifest.json") if dataset == "vision" else (SYN_DATA / "completion.json")
    data_record = read(identity)
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.support_pair", commit, dataset, str(seed)]
    )
    files = {str(src): expected, str(identity): sha(identity), str(MANIFEST): sha(MANIFEST)}
    if dataset == "vision":
        summary = src.with_name("final.json")
        config = src.with_name("config.json")
        for s in SPLITS:
            files[str(DATA / (s + ".pt"))] = data_record["splits"][s]["sha256"]
        tau = float(read(config)["config"]["tau"])
    else:
        summary, config = COMPARISON, src.with_name("config.json")
        mse = {}
        for s in SPLITS:
            for name in (s + ".pt", s + "_states.pt"):
                files[str(SYN_DATA / name)] = data_record["hashes"][name]
            p = src.with_name(s + ".json")
            files[str(p)] = sha(p)
        tau = float(read(config)["tau_prediction"])
    for p in (summary, config):
        files[str(p)] = sha(p)
    note = (
        f"# Two support episodes, {dataset} seed {seed}\n\n{PROTOCOL}\n\nCheckout {commit}\n"
        f"Data {identity} SHA256 {files[str(identity)]}\nCheckpoint {src} SHA256 {expected}\n"
        f"Original numeric tolerance {tau!r}\n\n    {command}\n"
    )
    _head(commit)
    with file_lock(out / ".lock"):
        declared(out / "NOTE.md", note)
        if seed:
            pilot = validate(OUTPUT / f"support-pair-{dataset}-s0")
            if pilot["commit"] != commit or not pilot["passed"] or pilot["dataset"] != dataset:
                raise RuntimeError("dataset pilot did not qualify expansion")
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        if any(sha(Path(k)) != h for k, h in files.items()):
            raise ValueError("endpoint, split or reference identity differs")
        torch.set_num_threads(1)
        if (out / "metrics.json").is_file():
            previous = validate(out)
            if (
                previous["input_sha256"] != files
                or previous["commit"] != commit
                or previous["seed"] != seed
                or previous["dataset"] != dataset
            ):
                raise ValueError("run identity changed")
            return
        if dataset == "vision":
            ref = read(summary)
            mse = {
                "calibration": ref["calibration_prediction_mse"],
                "validation": ref["prediction_mse"],
            }
            ref_mcc, ref_r2 = ref["mcc"], ref["parameter_r2_min"]
            ref_shd = sum(ref[k + "_shd"] for k in ("state", "action", "parameter"))
        else:
            mse = {s: read(src.with_name(s + ".json"))["metrics"]["prediction"] for s in SPLITS}
            ref = read(summary)["per_run"]["full"][str(seed)]
            ref_mcc, ref_r2 = ref["state_mcc"], ref["code_r2_min"]
            ref_shd = sum(
                ref[k]
                for k in (
                    "masks/state_aligned_shd",
                    "masks/action_aligned_shd",
                    "masks/parameter_code_aligned_shd",
                )
            )
        _idle_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        if dataset == "vision":
            model = (
                train.model_from_checkpoint(train.load_checkpoint(src), phase="full")
                .to(device)
                .eval()
            )
            rgb = {s: data.RenderedInputs(DATA, s, device) for s in SPLITS}
            native, samples = rgb_eval(model, rgb)
            actions = {s: v.actions[:, :, : v.frames - 1].cpu() for s, v in rgb.items()}
        else:
            model, _ = synthetic_model(seed, src, device)
            sets, _ = load_data(SYN_DATA)
            native, samples = synthetic_eval(model, sets, device)
            actions = {s: sets[s].tensors["actions"].cpu() for s in SPLITS}
        codes = {s: samples[s]["codes"].float() for s in SPLITS}
        pairs = {s: donor_codes(codes[s])[0] for s in SPLITS}
        metrics = {
            k: physical(model, samples, c, dataset, native, DATA)
            for k, c in (("single", codes), ("pair", pairs))
        }
        original = metrics["single"]["validation"]
        for k, v in (("r2", ref_r2), ("mcc", ref_mcc), ("shd", ref_shd)):
            if not math.isclose(original[k], float(v), rel_tol=1e-4, abs_tol=1e-6):
                raise ValueError(f"original labelled validation {k} did not reproduce")
        common = None
        splits: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for s in SPLITS:
            nc, pc, maxerr = forecast(model, samples[s], actions[s], device)
            if not math.isclose(float(nc.mean()), mse[s], rel_tol=1e-4, abs_tol=1e-8):
                raise ValueError("released single-donor forecast MSE did not reproduce")
            stats, common, weights = dispersion(codes[s], pairs[s], s, common)
            splits[s] = {
                "dispersion": stats,
                "forecast": _forecast_summary(nc, pc, weights, tau),
                "native_replay_max_abs": maxerr,
                "physical": {k: a[s] for k, a in metrics.items()},
            }
            for name, x in (
                ("single_codes", codes[s]),
                ("pair_codes", pairs[s]),
                ("native_mse", nc),
                ("pair_mse", pc),
            ):
                arrays[s + "_" + name] = x.cpu().numpy()
        tmp = out / "system-samples.tmp.npz"
        np.savez_compressed(tmp, **arrays)  # pyright: ignore[reportUnknownMemberType]
        tmp.replace(out / "system-samples.npz")
        checks = scientific_checks(splits, dataset)
        _head(commit)
        write_json(
            out / "metrics.json",
            {
                "commit": commit,
                "seed": seed,
                "dataset": dataset,
                "command": command,
                "source": str(src),
                "source_sha256": expected,
                "data_identity": files[str(identity)],
                "input_sha256": files,
                "note_sha256": sha(out / "NOTE.md"),
                "samples_sha256": sha(out / "system-samples.npz"),
                "tau": tau,
                "splits": splits,
                "checks": checks,
                "passed": all(checks.values()),
                "verified": True,
            },
        )
        validate(out)


def main() -> None:
    """Run the declared development-only deployment comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit")
    parser.add_argument("dataset", choices=("synthetic", "vision"))
    parser.add_argument("seed", type=int, choices=range(5))
    args = parser.parse_args()
    run(args.commit, args.dataset, args.seed)


if __name__ == "__main__":
    main()
