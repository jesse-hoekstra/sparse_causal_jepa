"""Evaluate the frozen hold and kick queries on individual validation systems."""

import argparse
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from diagnostics.vision_dimer import data, downstream
from scjepa.experiments.storage import file_lock, write_json

from . import changed_queries as previous
from . import chart, kicks
from .chart import _affine_query as affine_hold  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    ROOT,
    adaptation_checks,
    checkpoint_hashes,
    declared,
    read,
    sha,
    sparse_checks,
)

BOOT_SEED = 20261122
N_BOOT = 2000
BEADS = ("1a", "1b", "2a", "2b")
ERRORS = (
    "q1_system_mae",
    "q2_system_relative",
    "q1_population_abs_error",
    "q2_population_relative",
)
PROTOCOL = """Hypothesis: a low error of the pooled intervention response can coexist with failure
to predict differences among systems. Change only the unit at which the existing Q1 and Q2
responses are compared. Use all five original physical Q1 hold levels for u2a to u2b and four
Q2 held-bead partner matrices. Keep original controls and every authenticated changed terminal
endpoint, including negative scientific pilots. Preserve each chart's hold implementation and
current-position rule. The dimer-wide field acts on both beads.

Physical labels supply the original CAL chart fits and simulator responses for evaluation only.
For each system first average its three window starts and ten episodes, then compare systems
equally. The labelled constant predictor uses the CAL mean simulator response and no recipient
information. Its VAL error is the denominator of the reported squared-error skill. Require
VAL variance and fixed-constant MSE above max(1e-24, 1e-12 times mean squared CAL truth) for
skill in either query. For Q2 relative Frobenius error the per-bead denominator must exceed
max(1e-12, 1e-6 times the bead's CAL RMS matrix norm); if any required denominator is too small,
omit that quantity. Report unnormalized MSE and signed component biases regardless.

On VAL compare both the error of the pooled response (the archived estimand) and the mean
systemwise error. For Q1 use absolute slope error. For Q2 average four relative Frobenius
errors. Use 2000 common percentile bootstrap resamples of whole validation systems, conditioning
on fitted charts, the fixed CAL comparator and training seed. All paired differences are
model minus reference. A negative error difference favors the model. A positive skill indicates
smaller squared error than the CAL constant. Intervals wholly on one side of zero give
descriptive support on this cohort; intervals spanning zero leave direction unresolved.
Never count episodes, window starts, or model seeds as additional systems. This is descriptive.

Reproduce every corresponding prior pooled validation response before interpreting any
systemwise scores. Save raw partner endpoints in context-endpoints.npz, indexed by system,
start and episode. Q1 ends in the five physical levels. Q2 ends in held bead, partner u/v,
then zero/x/y added field. For every comparison the CAL comparator and charts stay fixed.
Simulator atol=1e-11, rtol=1e-10; model atol=5e-5, rtol=1e-4. Insufficient
variation is a scientific limitation, not a technical failure. Seed 0 is the technical
pilot; seeds 1-4 require its complete same-commit verified receipt. No training or test scoring."""


@dataclass
class Outcomes:
    """Raw partner position after each prescribed hold or field addition, grouped by system."""

    q1: Tensor
    q2: Tensor

    def effects(self, grid: Tensor) -> "Effects":
        """Reduce start and episode first, then fit the Q1 slope or subtract the unadded field."""
        centered = grid - grid.mean()
        slopes = self.q1.mean((1, 2)) @ centered / centered.square().sum()
        endpoints = self.q2.mean((1, 2))
        return Effects(slopes, endpoints[..., 1:] - endpoints[..., :1])


@dataclass
class Effects:
    """One partner slope and four held-bead response matrices per system."""

    q1: Tensor
    q2: Tensor

    def record(self) -> dict[str, Any]:
        """Serialize the equally weighted system responses in physical units."""
        return {"q1_slope": downstream.as_list(self.q1), "q2_held": downstream.as_list(self.q2)}

    def constant(self, size: int) -> "Effects":
        """Broadcast the calibration mean to each new system."""
        return Effects(self.q1.mean().expand(size), self.q2.mean(0).expand(size, -1, -1, -1))


def _group(rows: Tensor, groups: int, views: int) -> Tensor:
    # Both protected exporters iterate start, system batch, individual system, episode.
    return rows.reshape(len(downstream.STARTS), groups, views, *rows.shape[1:]).transpose(0, 1)


def outcomes(
    world: kicks.KickWorld, hold: Callable[[float], Tensor], grid: Tensor, groups: int, views: int
) -> Outcomes:
    """Collect natural-unit Q1 partner positions and Q2 partner positions with/without field."""
    q1 = torch.stack([hold(float(value))[:, 6] for value in grid], -1)
    blocks: list[Tensor] = []
    for bead, partner in enumerate(downstream.PARTNER):
        fixed = world.current(bead)
        positions = [
            world.decode(world.rollout(fixed, kick))[:, list(downstream.BEADS[partner])]
            for kick in (None, *downstream.kicks(bead // 2))
        ]
        blocks.append(torch.stack(positions, -1))
    q2 = torch.stack(blocks, 1)
    previous.require(
        bool(torch.isfinite(q1).all() and torch.isfinite(q2).all()), "Nonfinite physical endpoint"
    )
    return Outcomes(_group(q1, groups, views), _group(q2, groups, views))


def bootstrap_weights(groups: int, draws: int = N_BOOT) -> Tensor:
    """Point estimate first, then equal-sized paired multinomial system resamples."""
    generator = torch.Generator().manual_seed(BOOT_SEED)
    ids = torch.randint(groups, (draws, groups), generator=generator)
    counts = torch.zeros((draws, groups), dtype=torch.float64)
    counts.scatter_add_(1, ids, torch.ones_like(counts))
    return torch.cat((torch.ones_like(counts[:1]), counts)) / groups


def _means(weights: Tensor, values: Tensor) -> Tensor:
    return (weights @ values.reshape(len(values), -1)).reshape(len(weights), *values.shape[1:])


def _ratios(num: Tensor, denom: Tensor, floor: Tensor) -> Tensor:
    return torch.where(denom > floor, num / denom.clamp_min(1e-300), float("nan"))


def scores(model: Effects, truth: Effects, cal: Effects, weights: Tensor) -> dict[str, Tensor]:
    """Return point and bootstrap estimates from whole-system effects, preserving paired draws."""
    s, m = truth.q1, truth.q2
    ds, dm = model.q1 - s, model.q2 - m
    scale = cal.q2.square().sum((-2, -1)).mean(0).sqrt()
    norm_floor = (scale * 1e-6).clamp_min(1e-12)
    norms = m.square().sum((-2, -1)).sqrt()
    norm_error = dm.square().sum((-2, -1)).sqrt()
    relative = _ratios(norm_error, norms, norm_floor)
    bias = _means(weights, dm)
    pooled = _means(weights, m)
    result = {
        "q1_system_mae": weights @ ds.abs(),
        "q1_population_abs_error": (weights @ ds).abs(),
        "q1_bias": weights @ ds,
        "q2_bias": bias,
        "q2_population_relative": _ratios(
            bias.square().sum((-2, -1)).sqrt(), pooled.square().sum((-2, -1)).sqrt(), norm_floor
        ).mean(-1),
    }
    # An unselected low-norm system must not invalidate a bootstrap draw.
    invalid = (~torch.isfinite(relative)).any(1).double()
    result["q2_system_relative"] = torch.where(
        (weights @ invalid) == 0,
        weights @ relative.nan_to_num().mean(1),
        float("nan"),
    )
    for name, measured, delta, support in (("q1", s, ds, cal.q1), ("q2", m, dm, cal.q2)):
        target = measured.reshape(len(s), -1)
        support = support.reshape(len(cal.q1), -1)
        mse = weights @ delta.reshape(len(s), -1).square().mean(1)
        base = weights @ (target - support.mean(0)).square().mean(1)
        centered = target - target.mean(0)
        variability = (weights @ centered.square().mean(1)) - (weights @ centered).square().mean(1)
        floor = (support.square().mean() * 1e-12).clamp_min(1e-24)
        result[f"{name}_mse"] = mse
        result[f"{name}_skill"] = torch.where(
            variability > floor, 1 - _ratios(mse, base, floor), float("nan")
        )
    return result


def summaries(draws: dict[str, Tensor]) -> dict[str, Any]:
    """Use a fixed two-sided percentile interval only if all resamples have defined scores."""
    record: dict[str, Any] = {}
    for name, values in draws.items():
        point, replicates = values[0], values[1:]
        valid = torch.isfinite(replicates).reshape(len(replicates), -1).all(1)
        point_valid = bool(torch.isfinite(point).all())
        interval = None
        if point_valid and bool(valid.all()):
            interval = downstream.as_list(
                torch.quantile(replicates, values.new_tensor([0.025, 0.975]), dim=0)
            )
        record[name] = {
            "estimate": point.tolist() if point_valid else None,  # pyright: ignore[reportUnknownMemberType]
            "interval": interval,
            "valid_resamples": int(valid.sum()),
        }
    return record


def difference(new: dict[str, Tensor], old: dict[str, Tensor]) -> dict[str, Any]:
    """Signed paired error differences on identical systems, including the pooled estimands."""
    return summaries({name: new[name] - old[name] for name in ERRORS})


def reproduce(effect: Effects, q1: float, q2: Tensor, *, simulator: bool) -> dict[str, Any]:
    """Verify each archived pooled response, before reading a systemwise performance score."""
    atol, rtol = (1e-11, 1e-10) if simulator else (5e-5, 1e-4)
    slope, matrix = effect.q1.mean(), effect.q2.mean(0)
    return {
        "verified": math.isclose(float(slope), q1, abs_tol=atol, rel_tol=rtol)
        and torch.allclose(matrix, q2, atol=atol, rtol=rtol),
        "q1_abs_difference": abs(float(slope) - q1),
        "q2_max_abs_difference": float((matrix - q2).abs().max()),
        "q1_reference": q1,
        "q2_reference": downstream.as_list(q2),
        "atol": atol,
        "rtol": rtol,
    }


def _held(record: dict[str, Any]) -> Tensor:
    return torch.tensor([record[b]["held"] for b in BEADS], dtype=torch.float64)


def _command(commit: str, seed: int) -> str:
    module = "ICML_review.code.system_effects"
    return f"PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m {module} {commit} {seed}"


def _declared_inputs(seed: int) -> dict[str, str]:
    """Freeze the identities supplied by the immutable earlier evaluation before reading weights."""
    files = [OUTPUT / f"{suite}-paired-s{seed}/metrics.json" for suite in ("chart", "kicks")]
    files.append(OUTPUT / f"changed-queries-s{seed}/metrics.json")
    conditional = OUTPUT / f"one-head-queries-s{seed}/metrics.json"
    if conditional.exists():
        files.append(conditional)
    ids = {str(p): sha(p) for p in [DATA / "manifest.json", MANIFEST, *files]}
    split_manifest = read(DATA / "manifest.json")["splits"]
    for split in ("calibration", "validation"):
        ids[str(DATA / f"{split}.pt")] = split_manifest[split]["sha256"]
    original = read(files[0])
    for name, arm in original["arms"].items():
        path = ROOT / f"seed{seed}/sparse-500k/{name}/final.pt"
        previous.require(name in ("full", "baseline") and arm["checkpoint"] == str(path), str(path))
        ids[str(path)] = arm["checkpoint_sha256"]
    for folder in files[2:]:
        options = (
            {previous.ONE_HEAD: ("baseline", "full")} if folder == conditional else previous.ARMS
        )
        routes = read(folder)["methods"]
        previous.require(set(routes) == set(options), str(folder))
        for method, route in routes.items():
            if route["evaluated_arms"]:
                measured = route["queries"]["chart"]["measurement"]["arms"]
                for name, arm in measured.items():
                    base = OUTPUT / f"{method}-s{seed}"
                    stage = base if method == "vision-loss" else base / "sparse-500k" / name
                    path = stage / "final.pt"
                    previous.require(
                        name in options[method] and arm["checkpoint"] == str(path), str(path)
                    )
                    ids[str(path)] = arm["checkpoint_sha256"]
    return ids


def inventory(
    seed: int, splits: dict[str, str], ledger: previous.Inputs
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Authenticate every source and compare archived query status with current stage receipts."""
    reported = checkpoint_hashes()
    controls = previous.controls(seed, splits, reported, ledger)
    models: dict[str, dict[str, Any]] = {}
    skipped: dict[str, Any] = {}
    old_paths = {q: str(OUTPUT / f"{q}-paired-s{seed}/metrics.json") for q in previous.CONTROLS}

    def add(
        method: str, arm: str, path: Path, records: dict[str, Any], refs: dict[str, str]
    ) -> None:
        h = ledger.bind(path)
        for query in ("chart", "kicks"):
            head = records[query]["arms"][arm]
            previous.require(
                head["checkpoint"] == str(path) and head["checkpoint_sha256"] == h,
                f"Endpoint does not match the prior {query} measurement: {path}",
            )
            previous.require(
                all(
                    records[query][k] == controls[query][k]
                    for k in (
                        "data_sha256",
                        "calibration_groups",
                        "validation_groups",
                        "validation_contexts",
                    )
                )
                and records[query]["seed"] == seed,
                str(path),
            )
            key = "truth_slopes_held_by_outcome" if query == "chart" else "truth"
            torch.testing.assert_close(
                records[query][key], controls[query][key], atol=1e-11, rtol=1e-10
            )
        torch.testing.assert_close(records["chart"]["hold_grids"], controls["chart"]["hold_grids"])
        models[f"{method}/{arm}"] = {
            "method": method,
            "arm": arm,
            "checkpoint": str(path),
            "sha256": h,
            "aggregate_receipts": refs,
            "references": {
                interface: {
                    "q1": records["chart"]["arms"][arm][interface]["u2a_to_u2b"],
                    "q2": records["kicks"]["arms"][arm][interface]["matrices"],
                }
                for interface in ("scalar", "affine")
            },
        }

    for arm in ("full", "baseline"):
        add("original", arm, ROOT / f"seed{seed}/sparse-500k/{arm}/final.pt", controls, old_paths)
    for scope, methods in (
        ("changed-queries", tuple(previous.ARMS)),
        ("one-head-queries", (previous.ONE_HEAD,)),
    ):
        query_dir = OUTPUT / f"{scope}-s{seed}"
        if scope == "one-head-queries" and not (query_dir / "metrics.json").is_file():
            # A genuinely unactivated contingency has five favorable direct receipts. Never
            # silently omit a produced or partially produced model while awaiting query output.
            previous.require(
                not any(OUTPUT.glob("vision-one-head-s[0-4]")), "Missing one-head query receipt"
            )
            for s in range(5):
                p = OUTPUT / f"vision-direct-s{s}/metrics.json"
                d = ledger.record(p)
                ledger.bind(DATA / "manifest.json", d["data_manifest_sha256"])
                ledger.note(p.parent / "NOTE.md", previous.TRAINING, d["data_manifest_sha256"])
                init = adaptation_checks(d["adaptation"], d["reference_adaptation"])
                previous.require(init == d["adaptation_checks"] and all(init.values()), str(p))
                previous.require(
                    d["checks"] == sparse_checks(d["validation"], d["reference_validation"]), str(p)
                )
                previous.require(
                    d["passed"] is True
                    and d["seed"] == s
                    and d["commit"] == previous.TRAINING
                    and all(d["checks"].values()),
                    str(p),
                )
            for arm in ("full", "baseline"):
                skipped[f"{previous.ONE_HEAD}/{arm}"] = (
                    "direct projection passed all five seeds; contingency did not activate"
                )
            continue
        src = ledger.record(query_dir / "metrics.json")
        rev = src["commit"]
        previous.require(re.fullmatch("[0-9a-f]{40}", rev) is not None, str(query_dir))
        extra = " --one-head" if scope == "one-head-queries" else ""
        module = "ICML_review.code.changed_queries"
        cmd = f"PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m {module} {rev} {seed}{extra}"
        previous.require(
            src["verified"] is True
            and src["seed"] == seed
            and src["command"] == cmd
            and src["data_sha256"] == splits
            and set(src["methods"]) == set(methods),
            str(query_dir),
        )
        ledger.bind(query_dir / "NOTE.md", src["note_sha256"])
        ledger.note(query_dir / "NOTE.md", cmd)
        for method in methods:
            revision = rev if method == previous.ONE_HEAD else previous.TRAINING
            paths, state = previous.sources(
                method, seed, splits, reported, ledger, training_commit=revision
            )
            prior = src["methods"][method]
            previous.require(
                all(prior.get(k) == v for k, v in state.items())
                and set(paths) == set(prior["evaluated_arms"]),
                f"Prior query inventory is stale: {method}",
            )
            previous.require(
                state["training_gate_passed"] is not None or state.get("pilot_passed") is False,
                f"Await the terminal training or failed admission receipt: {method} seed {seed}",
            )
            for arm, reason in state["skipped"].items():
                previous.require(
                    reason != "no terminal checkpoint",
                    f"Unresolved missing checkpoint: {method}/{arm}",
                )
                skipped[f"{method}/{arm}"] = reason
            records = {q: value["measurement"] for q, value in prior["queries"].items()}
            previous.require(
                set(records) == (set(previous.CONTROLS) if paths else set())
                and all(set(r["arms"]) == set(paths) for r in records.values()),
                str(query_dir),
            )
            for arm, path in paths.items():
                refs: dict[str, str] = {
                    q: str(query_dir / "metrics.json") for q in previous.CONTROLS
                }
                ledger.bind(path, src["input_sha256"][str(path)])
                add(method, arm, path, records, refs)
        for name, digest in src["input_sha256"].items():
            path = Path(name)
            previous.require(
                not path.is_absolute()
                and ".." not in path.parts
                and (
                    path.is_relative_to(OUTPUT)
                    or path.is_relative_to(ROOT)
                    or path.is_relative_to(MANIFEST.parent)
                ),
                f"Unexpected source ledger path: {path}",
            )
            ledger.bind(path, digest)
    return models, skipped, controls


@torch.inference_mode()
def run(commit: str, seed: int, out: Path) -> None:
    """Run the hash-verified inventory and paired system-level comparisons on one unoccupied GPU."""
    _head(commit)
    expected = _declared_inputs(seed)
    command = _command(commit, seed)
    declared(
        out / "NOTE.md",
        f"# Frozen system-level effects, seed {seed}\n\n{PROTOCOL}\n\n"
        f"Git {commit}\n\n    {command}\n\n"
        + "\n".join(f"{h}  {name}" for name, h in expected.items())
        + "\n",
    )
    ledger = previous.Inputs()
    if seed:
        pilot = ledger.record(OUTPUT / "system-effects-s0/metrics.json")
        previous.require(
            pilot["commit"] == commit and pilot["seed"] == 0 and pilot["verified"] is True,
            "Verified same-revision technical pilot required",
        )
    subprocess.run(["sha256sum", "-c", "--quiet", MANIFEST.name], cwd="outputs", check=True)
    for name, digest in expected.items():
        ledger.bind(Path(name), digest)
    manifest = read(DATA / "manifest.json")
    splits = {s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")}
    reported = checkpoint_hashes()
    previous.require(
        all(reported[str(DATA / f"{s}.pt")] == digest for s, digest in splits.items()),
        "Split identities differ from the reported release",
    )
    models, skipped, controls = inventory(seed, splits, ledger)
    _head(commit)
    _idle_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cfg = data.load_config(DATA)
    grid = torch.tensor(controls["chart"]["hold_grids"][4], dtype=torch.float64)
    arrays: dict[str, Any] = {}
    truth: dict[str, Effects] = {}
    groups: dict[str, int] = {}
    for split in ("calibration", "validation"):
        payload = data._load(DATA, split)  # pyright: ignore[reportPrivateUsage]
        size = len(payload["states"])
        views = payload["states"].shape[1]
        previous.require(size == controls["chart"][f"{split}_groups"] and views == 10, split)
        groups[split] = size
        world = downstream.Truth(
            cfg.physics, downstream.truth_contexts(payload, downstream.STARTS, size)
        )

        def physical_hold(value: float, world: downstream.Truth = world) -> Tensor:
            return world.rollout(world.hold_value(4, value), None)

        measured = outcomes(world, physical_hold, grid, size, views)
        arrays[f"truth__{split}__q1"] = measured.q1.numpy()
        arrays[f"truth__{split}__q2"] = measured.q2.numpy()
        truth[split] = measured.effects(grid)
    previous.require(
        len(downstream.STARTS) * groups["validation"] * cfg.views
        == controls["chart"]["validation_contexts"],
        "Unexpected number of validation contexts",
    )
    val, cal = truth["validation"], truth["calibration"]
    check = reproduce(
        val,
        controls["chart"]["truth_slopes_held_by_outcome"][4][6],
        _held(controls["kicks"]["truth"]),
        simulator=True,
    )
    valid = bool(check["verified"])
    inputs = {s: data.RenderedInputs(DATA, s, device) for s in groups}
    phys = (
        data.load_labels(DATA, "calibration")["states"][:, :, : cfg.frames].reshape(-1, 8).double()
    )
    fixed_grid = torch.quantile(phys[:, 4], phys.new_tensor(downstream.QUANTILES))
    torch.testing.assert_close(fixed_grid, grid, atol=1e-12, rtol=1e-12)
    per_model: dict[tuple[str, str], Effects] = {}
    for key, row in models.items():
        _head(commit)
        model, frames, lat = chart.encode_endpoint(
            Path(row["checkpoint"]), row["arm"], device, inputs["calibration"], inputs["validation"]
        )
        scalar = downstream.Model(model, lat, frames, phys)
        affine = chart.AffineChart.fit(frames, phys)
        affine_world = kicks.AffineWorld(model, lat, affine)

        def scalar_query(value: float, scalar: downstream.Model = scalar) -> Tensor:
            return scalar.decode(scalar.rollout(scalar.hold_value(4, value), None))

        def affine_query(value: float, world: kicks.AffineWorld = affine_world) -> Tensor:
            return affine_hold(world.model, world.lat, world.chart, 4, value)

        refs = row.pop("references")
        row["interfaces"] = {}
        for interface, env, query in (
            ("scalar", scalar, scalar_query),
            ("affine", affine_world, affine_query),
        ):
            values = outcomes(env, query, grid, groups["validation"], inputs["validation"].views)
            prefix = key.replace("/", "__") + "__" + interface
            arrays[prefix + "__q1"], arrays[prefix + "__q2"] = values.q1.numpy(), values.q2.numpy()
            effect = values.effects(grid)
            per_model[key, interface] = effect
            comparison = reproduce(
                effect, refs[interface]["q1"], _held(refs[interface]["q2"]), simulator=False
            )
            row["interfaces"][interface] = {
                "per_system": effect.record(),
                "reproduction": comparison,
            }
            valid = valid and bool(comparison["verified"])
    weight = bootstrap_weights(len(val.q1))
    const = scores(cal.constant(len(val.q1)), val, cal, weight)
    if valid:
        by_model = {key: scores(e, val, cal, weight) for key, e in per_model.items()}
        for (key, iface), draw in by_model.items():
            row = models[key]
            row["interfaces"][iface].update(
                scores=summaries(draw),
                versus_constant=difference(draw, const),
                paired_change=None
                if row["method"] == "original"
                else difference(draw, by_model["original/" + row["arm"], iface]),
            )
    _head(commit)
    archive = out / "context-endpoints.npz"
    tmp = out / "context-endpoints.tmp.npz"
    np.savez_compressed(tmp, **arrays)  # pyright: ignore[reportUnknownMemberType]
    tmp.replace(archive)
    write_json(
        out / "metrics.json",
        {
            "verified": valid,
            "commit": commit,
            "seed": seed,
            "command": command,
            "note_sha256": sha(out / "NOTE.md"),
            "input_sha256": ledger,
            "data_sha256": {s: splits[s] for s in groups},
            "archive_sha256": sha(archive),
            "system_ids": {s: list(range(n)) for s, n in groups.items()},
            "hold_grid": downstream.as_list(grid),
            "starts": list(downstream.STARTS),
            "views": inputs["validation"].views,
            "beads": list(BEADS),
            "truth": {s: e.record() for s, e in truth.items()},
            "truth_reproduction": check,
            "bootstrap": {
                "draws": N_BOOT,
                "seed": BOOT_SEED,
                "unit": "validation system",
                "percentiles": [2.5, 97.5],
            },
            "models": models,
            "skipped": skipped,
            "constant_scores": summaries(const) if valid else None,
        },
    )


def main() -> None:
    """Evaluate one original seed and its eligible descendants in a separate receipt directory."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    args = cli.parse_args()
    out = OUTPUT / f"system-effects-s{args.seed}"
    with file_lock(out / ".lock"):
        run(args.commit, args.seed, out)


if __name__ == "__main__":
    main()
