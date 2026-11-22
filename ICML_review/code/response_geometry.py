"""Measure which reproducible RGB support-code directions affect deployed forecasts."""

import argparse
import shlex
from typing import Any, cast

import torch
from torch import Tensor

from diagnostics.vision_dimer.analysis import endpoint, load_model
from diagnostics.vision_dimer.data import RenderedInputs
from diagnostics.vision_dimer.evaluate import collect
from diagnostics.vision_dimer.train import rendered_batches
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA, cross_episode_parameters

from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA, OUTPUT, PYTHON, ROOT, checkpoint_hashes, declared, read, sha

MODULE = "ICML_review.code.response_geometry"
PROTOCOL = """Does every reproducible support-code direction locally affect the deployed RGB
forecast on held-out systems? Freeze both reported endpoints and use the first window of each
episode. Fit code units, target units and the weakest and strongest response directions on
calibration alone. Reuse the native deterministic donor routing. At fixed recipient states and
actions, differentiate the full two-step rollout with respect to the donor code. Report the
whole spectrum, response to a one-standard-deviation local perturbation divided by prediction
error, and the within-system fraction in those same directions. Check the actual forecast
effect of replacing the donor by another observed donor inside the same group or in another
group. The latter changes all code directions together. Bootstrap validation systems with fixed
calibration directions. This is an evaluation without physical labels.

An endpoint shows evidence that all measured directions locally affect the forecast above its
residual scale if the calibration code and target have full numerical rank and the lower
system-bootstrap bound for the minimum validation Gram eigenvalue divided by residual MSE
exceeds one. Also report how the preselected weakest calibration direction transfers. An
upper bound below one for that fixed direction puts its response below the residual scale.
The other-system exchange resamples disjoint pairs. Report intervals that cross one
even if the point estimate exceeds one. This is a local sensitivity and noise-scale comparison,
not a test for an exact null, a global inverse or physical parameter identity. Reproduce the native
forecast to absolute tolerance 0.00003. Expand a technically verified seed-zero measurement to
all seeds irrespective of effect direction. No training or test evaluation.
"""


def eig(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Diagonalize a symmetric estimate in ascending order."""
    return cast(tuple[Tensor, Tensor], torch.linalg.eigh((matrix + matrix.T) / 2))  # pyright: ignore[reportUnknownMemberType]


def covariance_axes(rows: Tensor) -> tuple[Tensor, Tensor, Tensor, float]:
    """Return the resolved basis and positive square roots in float64 precision."""
    flat = rows.detach().double().reshape(-1, rows.shape[-1])
    centered = flat - flat.mean(0)
    values, basis = eig(centered.T @ centered / len(flat))
    cutoff = float(values[-1]) * max(flat.shape) * torch.finfo(flat.dtype).eps
    keep = values > cutoff
    if not bool(keep.any()):
        raise ValueError("All calibration directions are constant")
    return basis[:, keep], values[keep].sqrt(), values, cutoff


def jacobian(forecast: Tensor, donors: Tensor) -> Tensor:
    """Differentiate independent per-example forecasts with respect to their own donor."""
    flat = forecast.flatten(1)
    columns = [
        torch.autograd.grad(flat[:, j].sum(), donors, retain_graph=j + 1 < flat.shape[1])[0]
        for j in range(flat.shape[1])
    ]
    return torch.stack(columns, 1).reshape(*forecast.shape, donors.shape[-1])


def bootstrap_ratio(numerator: Tensor, denominator: Tensor) -> dict[str, Any]:
    """Paired system resampling for a ratio of group means; calibration axes stay fixed."""
    gen = torch.Generator().manual_seed(7812)
    draws = torch.randint(len(numerator), (2000, len(numerator)), generator=gen)
    if bool((denominator <= 0).any()):
        raise ValueError("A system has zero prediction residual")
    ratios = numerator[draws].mean(1) / denominator[draws].mean(1)
    bounds = torch.quantile(ratios, torch.tensor([0.025, 0.975], dtype=ratios.dtype))
    return {"estimate": float(numerator.mean() / denominator.mean()), "interval": bounds.tolist()}  # pyright: ignore[reportUnknownMemberType]


def minimum_response(grams: Tensor, errors: Tensor) -> dict[str, Any]:
    """Include rotation of the weakest empirical direction when resampling systems."""
    generator = torch.Generator().manual_seed(7812)
    draws = torch.randint(len(grams), (2000, len(grams)), generator=generator)
    values: list[Tensor] = []
    for start in range(0, len(draws), 64):
        batch = draws[start : start + 64]
        mean = grams[batch].mean(1)
        eigvals = cast(Tensor, torch.linalg.eigvalsh(mean))  # pyright: ignore[reportUnknownMemberType]
        values.append(eigvals[:, 0] / errors[batch].mean(1))
    quantiles = torch.quantile(torch.cat(values), errors.new_tensor([0.025, 0.975]))
    estimate = eig(grams.mean(0))[0][0] / errors.mean()
    return {"estimate": float(estimate), "interval": [float(q) for q in quantiles]}


def geometry(
    model: SCJEPA,
    samples: dict[str, Tensor],
    actions: Tensor,
    device: torch.device,
    code_root: Tensor,
    target_inverse: Tensor,
    reference_directions: Tensor | None,
) -> tuple[dict[str, Any], Tensor]:
    """Evaluate the native rollout and keep calibration choices fixed on validation."""
    histories = samples["states"][
        :, :, model.context_len - model.predictor.history_len : model.context_len
    ]
    actions = actions[:, :, model.context_len - model.predictor.history_len :]
    codes = samples["codes"]
    donor = cross_episode_parameters(codes)
    # Complementary donors are always distinct from the recipient and original donor.
    other_episode = codes.roll(2, dims=1)
    targets = samples["targets"]
    size, views, steps, state_dim = targets.shape
    if size % 2 or views < 3:
        raise ValueError("Observed alternatives require paired systems and three distinct episodes")
    other_system = donor[torch.arange(size) ^ 1]
    observed_errors: list[Tensor] = []
    differences: list[Tensor] = []
    grams: list[Tensor] = []
    max_error = 0.0
    for start in range(0, size, 16):
        sl = slice(start, min(start + 16, size))
        batch = len(codes[sl])
        hist = histories[sl].flatten(0, 1).float().to(device)
        act = actions[sl].flatten(0, 1).float().to(device)
        c = donor[sl].flatten(0, 1).float().to(device).requires_grad_()
        p = model.rollout(hist, c, steps, act)
        jac = jacobian(p, c).double().cpu()
        # Map whitened code coordinates into native inputs and native outputs into whitened targets.
        white_jac = torch.einsum("so,nhsk,ki->nhoi", target_inverse, jac, code_root)
        flat = white_jac.reshape(batch, views * steps * target_inverse.shape[1], -1)
        grams.append(flat.transpose(1, 2) @ flat / flat.shape[1])
        native = p.detach().double().cpu().reshape(batch, views, steps, state_dim)
        max_error = max(max_error, float((native - samples["predictions"][sl]).abs().max()))
        errors = ((native - targets[sl]) @ target_inverse).square().mean((1, 2, 3))
        observed_errors.append(errors)
        with torch.no_grad():
            delta: list[Tensor] = []
            for alternative in (other_episode, other_system):
                alt = alternative[sl].flatten(0, 1).float().to(device)
                moved = model.rollout(hist, alt, steps, act).double().cpu().reshape_as(native)
                delta.append(((moved - native) @ target_inverse).square().mean((1, 2, 3)))
            differences.append(torch.stack(delta, 1))
    if max_error > 0.00003:
        raise RuntimeError(f"Diagnostic rollout differs from the native reference by {max_error}")
    withins = codes - codes.mean(1, keepdim=True)
    white_map = cast(Tensor, torch.linalg.pinv(code_root)).T  # pyright: ignore[reportUnknownMemberType]
    grams_all = torch.cat(grams)
    response, axes = eig(grams_all.mean(0))
    if reference_directions is None:
        reference_directions = axes
    strength = torch.diagonal(
        reference_directions.T @ grams_all @ reference_directions, dim1=-2, dim2=-1
    )
    residual = torch.cat(observed_errors)
    moved = torch.cat(differences)
    projected = (codes - codes.mean((0, 1))) @ white_map @ reference_directions
    within = (withins @ white_map @ reference_directions).square().mean((0, 1))
    varying = projected.square().mean((0, 1))
    fractions = [float(w / t) if t > 0 else None for w, t in zip(within, varying, strict=True)]
    return {
        "systems": size,
        "episodes": views,
        "horizons": steps,
        "maximum_reference_difference": max_error,
        "local_response_spectrum": response.tolist(),  # pyright: ignore[reportUnknownMemberType]
        "mean_whitened_forecast_mse": float(residual.mean()),
        "minimum_response_to_residual": minimum_response(grams_all, residual),
        "calibration_directions_response_to_residual": [
            bootstrap_ratio(column, residual) for column in strength.unbind(1)
        ],
        "within_total_in_calibration_directions": fractions,
        "observed_within_system_change_to_residual": bootstrap_ratio(moved[:, 0], residual),
        "observed_other_system_change_to_residual": bootstrap_ratio(
            moved[:, 1].reshape(-1, 2).mean(1), residual.reshape(-1, 2).mean(1)
        ),
    }, reference_directions


def measure(seed: int, arm: str, device: torch.device) -> dict[str, Any]:
    """Evaluate one frozen model on first-window calibration and validation systems."""
    path = endpoint(ROOT, seed, arm)
    digest = sha(path)
    if checkpoint_hashes()[str(path)] != digest:
        raise ValueError(f"Unexpected checkpoint identity for {path}")
    model, _, _ = load_model(path, device)
    model.eval().requires_grad_(False)
    out: dict[str, Any] = {"checkpoint": str(path), "sha256": digest}
    axes = None
    code_root = target_inverse = torch.empty(0)
    for split in ("calibration", "validation"):
        source = RenderedInputs(DATA, split, device)
        samples = collect(model, rendered_batches(source, size=8))
        actions = source.actions[:, :, : source.frames - 1].cpu()
        if split == "calibration":
            u, s, cv, ct = covariance_axes(samples["codes"])
            v, t, tv, tt = covariance_axes(samples["targets"])
            code_root, target_inverse = u * s, v / t
            out["calibration_basis"] = {
                "code_eigenvalues": [float(value) for value in cv],
                "target_eigenvalues": tv.tolist(),  # pyright: ignore[reportUnknownMemberType]
                "code_cutoff": ct,
                "target_cutoff": tt,
                "code_rank": len(s),
                "target_rank": len(t),
                "code_covariance_resolved_condition": float((s[-1] / s[0]).square()),
                "target_covariance_resolved_condition": float((t[-1] / t[0]).square()),
            }
        out[split], axes = geometry(
            model, samples, actions, device, code_root, target_inverse, axes
        )
        if split == "calibration":
            native_directions = code_root @ axes
            out["calibration_basis"]["unit_perturbation_columns_in_native_code"] = [
                [float(value) for value in row] for row in native_directions
            ]
    return out


def main() -> None:
    """Record the locked inputs, then assess the two reported RGB arms on a free GPU."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    args = cli.parse_args()
    rev, seed = args.commit, args.seed
    _head(rev)
    if seed:
        pilot = read(OUTPUT / "response-geometry-s0/metrics.json")
        if pilot["commit"] != rev or not pilot["verified"]:
            raise RuntimeError("A verified pilot at this revision is required")
    out = OUTPUT / f"response-geometry-s{seed}"
    manifest = sha(DATA / "manifest.json")
    command = shlex.join([PYTHON, "-m", MODULE, rev, str(seed)])
    note = (
        f"# Forecast sensitivity, seed {seed}\n\n{PROTOCOL}\nGit {rev}\n\n"
        f"Data manifest {manifest}\n\n    PYTHONPATH=$PWD:$PWD/src {command}\n"
    )
    with file_lock(out / ".lock"):
        declared(out / "NOTE.md", note)
        if (out / "metrics.json").exists():
            old = read(out / "metrics.json")
            if old["commit"] != rev or not old["verified"]:
                raise RuntimeError("Unverified existing result")
            return
        _idle_gpu()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        results: dict[str, Any] = {}
        for arm in ("full", "baseline"):
            _head(rev)
            results[arm] = measure(seed, arm, torch.device("cuda"))
        _head(rev)
        write_json(
            out / "metrics.json",
            {
                "commit": rev,
                "seed": seed,
                "verified": True,
                "data_manifest_sha256": manifest,
                "command": command,
                "arms": results,
            },
        )


if __name__ == "__main__":
    main()
