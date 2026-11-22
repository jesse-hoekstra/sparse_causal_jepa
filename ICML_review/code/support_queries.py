"""Evaluate own-, one- and two-episode support on the frozen dimer interventions."""

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from diagnostics.vision_dimer import data, downstream
from scjepa.experiments.storage import file_lock, write_json
from scjepa.models.model import SCJEPA

from . import chart, kicks, support_pair
from . import system_effects as systems
from .changed_queries import Inputs, controls, require
from .chart import _affine_query as affine_hold  # pyright: ignore[reportPrivateUsage]
from .single_acquisition import _head, _idle_gpu  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import (
    DATA,
    MANIFEST,
    OUTPUT,
    PYTHON,
    ROOT,
    checkpoint_hashes,
    declared,
    sha,
)

MODES = ("own", "single", "pair")
INTERFACES = ("scalar", "affine")
PROTOCOL = (
    "Hypothesis: a second independent support episode can stabilize a system code without "
    "degrading held-out physical intervention accuracy. Use the released sparse full endpoint "
    "of this seed. Compare the original recipient's own code, the code of episode i-1 and the "
    "mean of codes i-1 and i-2 modulo the number of episodes. Preserve the recipient's state, "
    "recorded actions, decoded clamp, unperturbed target and physical truth. Donor indices "
    "stay within the same system and same window start. Both external-support comparisons "
    "exclude the recipient. Fit scalar and affine physical charts only on original CAL frames. "
    "Q1 is the two-step u2a-to-u2b slope on the five original CAL hold values. Q2 is the "
    "two-step held-bead partner response to each unit dimer-field kick on four beads. "
    "Reproduce the original own-context per-system and pooled Q1/Q2, the CAL assignment, "
    "simulator responses, and the one/two-donor first-window forecast before interpreting "
    "contrasts. Distinguish pooled-response error from mean error across systems. Compute "
    "paired percentile intervals with the original whole-system resampling plan. Evaluate "
    "even if the primary fixed-support gate failed. Scalar pooled-error increases above 0.01 "
    "Q1 or 0.05 Q2 are descriptive application flags, separate from the primary support gate. "
    "No fitting of model parameters, test scoring, or feedback to expansion."
)


def recipients(
    latent: dict[str, Tensor], mode: str, *, groups: int, views: int, starts: tuple[int, ...]
) -> dict[str, Tensor]:
    """Substitute only support codes in native start-major, group-major, episode-minor order."""
    if (
        mode not in MODES
        or views < 3
        or groups < 1
        or not starts
        or len(set(starts)) != len(starts)
    ):
        raise ValueError("invalid support policy or episode geometry")
    original = latent["code"]
    if original.ndim != 2 or any(
        v.shape[0] != len(starts) * groups * views for v in latent.values()
    ):
        raise ValueError("recipient rows disagree with the archived group/window layout")
    if mode == "own":
        return latent
    panel = original.reshape(len(starts), groups, views, -1)
    left = panel.roll(1, 2)
    donor = left if mode == "single" else (left + panel.roll(2, 2)) / 2
    return latent | {"code": donor.reshape_as(original)}


def _effects(raw: dict[str, Any]) -> systems.Effects:
    return systems.Effects(
        torch.tensor(raw["q1_slope"], dtype=torch.float64),
        torch.tensor(raw["q2_held"], dtype=torch.float64),
    )


def measured_truth(
    directory: Path, split: str, grid: Tensor, expected: dict[str, Any]
) -> systems.Effects:
    """Independently evaluate deterministic simulator interventions on the same contexts."""
    cfg = data.load_config(directory)
    raw = data._load(directory, split)  # pyright: ignore[reportPrivateUsage]
    groups, views = raw["states"].shape[:2]
    truth = downstream.Truth(cfg.physics, downstream.truth_contexts(raw, downstream.STARTS, groups))
    result = systems.outcomes(
        truth, lambda value: truth.rollout(truth.hold_value(4, value), None), grid, groups, views
    ).effects(grid)
    earlier = _effects(expected)
    torch.testing.assert_close(result.q1, earlier.q1, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(result.q2, earlier.q2, rtol=1e-10, atol=1e-11)
    return result


def responses(
    model: SCJEPA,
    cal: Tensor,
    physical: Tensor,
    lat: dict[str, Tensor],
    grid: Tensor,
    groups: int,
    views: int,
) -> tuple[dict[str, systems.Effects], list[int], Tensor]:
    """Measure both physical interfaces and the raw first-window forecast per system."""
    scalar = downstream.Model(model, lat, cal, physical)
    affine = chart.AffineChart.fit(cal, physical)
    world = kicks.AffineWorld(model, lat, affine)
    hold = systems.outcomes(
        scalar,
        lambda value: scalar.decode(scalar.rollout(scalar.hold_value(4, value), None)),
        grid,
        groups,
        views,
    )
    pooled = systems.outcomes(
        world, lambda value: affine_hold(model, lat, affine, 4, value), grid, groups, views
    )
    first = slice(0, groups * views)
    mse = (scalar.forecast()[first] - lat["target"][first]).square().reshape(groups, -1).mean(1)
    return {"scalar": hold.effects(grid), "affine": pooled.effects(grid)}, scalar.match, mse


def references(
    commit: str, seed: int, source: Path, ledger: Inputs
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Bind the support trial and original physical responses to identical weights and data."""
    manifest = ledger.record(DATA / "manifest.json")
    reported = checkpoint_hashes()
    ledger.bind(MANIFEST)
    ledger.bind(source, reported[str(source)])
    all_splits = {
        s: manifest["splits"][s]["sha256"] for s in ("train", "calibration", "validation")
    }
    splits = {s: all_splits[s] for s in ("calibration", "validation")}
    for split, h in splits.items():
        ledger.bind(DATA / f"{split}.pt", h)
        require(h == reported[str(DATA / f"{split}.pt")], split)
    support = OUTPUT / f"support-pair-vision-s{seed}"
    trial = ledger.record(support / "metrics.json")
    require(trial == support_pair.validate(support), "support admission receipt differs")
    require(
        trial["commit"] == commit
        and trial["seed"] == seed
        and trial["dataset"] == "vision"
        and type(trial["passed"]) is bool
        and all(trial["checks"].values()) is trial["passed"]
        and trial["source"] == str(source)
        and trial["source_sha256"] == reported[str(source)]
        and trial["data_identity"] == ledger[str(DATA / "manifest.json")],
        "completed support trial differs",
    )
    ledger.bind(support / "NOTE.md", trial["note_sha256"])
    ledger.note(support / "NOTE.md", commit, trial["command"])
    ledger.bind(support / "system-samples.npz", trial["samples_sha256"])
    for p in (DATA / "manifest.json", source, source.with_name("final.json")):
        require(ledger.bind(p) == trial["input_sha256"][str(p)], f"support source differs: {p}")
    agg = controls(seed, all_splits, reported, ledger)
    previous = OUTPUT / f"system-effects-s{seed}"
    prior = ledger.record(previous / "metrics.json")
    require(
        prior["verified"] is True
        and prior["seed"] == seed
        and prior["data_sha256"] == splits
        and prior["starts"] == list(downstream.STARTS)
        and prior["views"] == 10
        and prior["bootstrap"]
        == {
            "draws": systems.N_BOOT,
            "seed": systems.BOOT_SEED,
            "unit": "validation system",
            "percentiles": [2.5, 97.5],
        }
        and prior["truth_reproduction"]["verified"] is True,
        "unverified original system effects",
    )
    original = prior["models"]["original/full"]
    require(
        original["checkpoint"] == str(source)
        and original["sha256"] == reported[str(source)]
        and all(original["interfaces"][k]["reproduction"]["verified"] is True for k in INTERFACES),
        "original full physical response provenance differs",
    )
    for name in original["aggregate_receipts"].values():
        require(ledger[str(name)] == prior["input_sha256"][name], "prior pooled controls differ")
    ledger.bind(previous / "NOTE.md", prior["note_sha256"])
    ledger.bind(previous / "context-endpoints.npz", prior["archive_sha256"])
    return trial, prior, agg, splits


@torch.inference_mode()
def run(commit: str, seed: int) -> None:
    """Reproduce the recipient and simulator, then change only the recipient's donor code."""
    require(seed in range(5), "invalid reported seed")
    _head(commit)
    source = ROOT / f"seed{seed}/sparse-500k/full/final.pt"
    support = OUTPUT / f"support-pair-vision-s{seed}"
    out = OUTPUT / f"support-pair-queries-s{seed}"
    command = "PYTHONPATH=$PWD:$PWD/src " + shlex.join(
        [PYTHON, "-m", "ICML_review.code.support_queries", commit, str(seed)]
    )
    with file_lock(out / ".lock"):
        reported = checkpoint_hashes()
        upstream = support / "metrics.json"
        declared(
            out / "NOTE.md",
            f"# External episode support for physical queries, seed {seed}\n\n"
            f"{PROTOCOL}\n\nCommit {commit}\nData {DATA} manifest {sha(DATA / 'manifest.json')}\n"
            f"Full checkpoint {source} SHA-256 {reported[str(source)]}\n"
            f"Completed support assessment {upstream} SHA-256 {sha(upstream)}\n\n    {command}\n",
        )
        subprocess.run(
            ["sha256sum", "-c", "--quiet", MANIFEST.name], cwd=MANIFEST.parent, check=True
        )
        ledger = Inputs()
        trial, prior, archive, splits = references(commit, seed, source, ledger)
        result = out / "metrics.json"
        if result.exists():
            old = systems.read(result)
            require(
                old["commit"] == commit
                and old["seed"] == seed
                and old["command"] == command
                and old["note_sha256"] == sha(out / "NOTE.md")
                and old["input_sha256"] == ledger,
                "stored query provenance changed",
            )
            return
        _idle_gpu()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        cfg = data.load_config(DATA)
        dev = torch.device("cuda")
        cal, val = (data.RenderedInputs(DATA, s, dev) for s in ("calibration", "validation"))
        groups, views = len(val), val.views
        require(
            (len(cal), groups, views) == (512, 256, 10)
            and prior["system_ids"]
            == {"calibration": list(range(len(cal))), "validation": list(range(groups))}
            and len(downstream.STARTS) * groups * views == archive["chart"]["validation_contexts"],
            "changed scored context set",
        )
        truth_cal = (
            data.load_labels(DATA, "calibration")["states"][:, :, : cfg.frames]
            .reshape(-1, 8)
            .double()
        )
        grid = torch.quantile(truth_cal[:, 4], truth_cal.new_tensor(downstream.QUANTILES))
        torch.testing.assert_close(
            grid, grid.new_tensor(archive["chart"]["hold_grids"][4]), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            grid, grid.new_tensor(prior["hold_grid"]), rtol=1e-12, atol=1e-12
        )
        truth = {
            s: measured_truth(DATA, s, grid, prior["truth"][s])
            for s in ("calibration", "validation")
        }
        original = prior["models"]["original/full"]
        released = ledger.record(source.with_name("final.json"))
        model, cal_z, raw_lat = chart.encode_endpoint(source, "full", dev, cal, val)
        weights = systems.bootstrap_weights(groups)
        all_scores: dict[str, dict[str, dict[str, Tensor]]] = {}
        arms: dict[str, Any] = {}
        # The stored program uses native GPU first-window collection; the downstream forecasts
        # execute the historical physical-query predictor on CPU.
        shape = (groups, views, -1)
        first = raw_lat["code"][: groups * views].reshape(shape)
        with np.load(support / "system-samples.npz", allow_pickle=False) as samples:
            stored_codes = torch.as_tensor(samples["validation_single_codes"]).double()
            torch.testing.assert_close(first, stored_codes, rtol=1e-4, atol=5e-5)
            adjacent = torch.as_tensor(samples["validation_pair_codes"]).double()
            torch.testing.assert_close(
                (first[:, ::2] + first[:, 1::2]) / 2, adjacent, rtol=1e-4, atol=5e-5
            )
            for mode in MODES:
                _head(commit)
                lat = recipients(
                    raw_lat, mode, groups=groups, views=views, starts=downstream.STARTS
                )
                values, assignment, group_mse = responses(
                    model, cal_z, truth_cal, lat, grid, groups, views
                )
                require(
                    assignment == released["spearman_assignment"],
                    "original scalar assignment differs",
                )
                forecast_check: dict[str, Any] = {"first_window_mse": float(group_mse.mean())}
                if mode != "own":
                    name = "native" if mode == "single" else "pair"
                    expected = torch.as_tensor(samples[f"validation_{name}_mse"]).double()
                    torch.testing.assert_close(group_mse, expected, rtol=2e-4, atol=1e-8)
                    forecast_check["max_per_system_replay_difference"] = float(
                        (group_mse - expected).abs().max()
                    )
                else:
                    denom = raw_lat["target"][: groups * views].reshape(-1, 8).var(0).mean()
                    own_nmse = float(group_mse.mean() / denom)
                    for q in ("chart", "kicks"):
                        row = archive[q]["arms"]["full"]
                        torch.testing.assert_close(
                            own_nmse, row["recomputed_first_window_nmse"], rtol=1e-4, atol=5e-7
                        )
                    forecast_check["own_first_window_nmse"] = own_nmse
                arms[mode] = {}
                all_scores[mode] = {}
                for interface, measurement in values.items():
                    if mode == "own":
                        previous = _effects(original["interfaces"][interface]["per_system"])
                        torch.testing.assert_close(
                            measurement.q1, previous.q1, rtol=1e-4, atol=5e-5
                        )
                        torch.testing.assert_close(
                            measurement.q2, previous.q2, rtol=1e-4, atol=5e-5
                        )
                        q1 = archive["chart"]["arms"]["full"][interface]["u2a_to_u2b"]
                        k = archive["kicks"]["arms"]["full"][interface]["matrices"]
                        m = torch.tensor([k[b]["held"] for b in systems.BEADS], dtype=torch.float64)
                        require(
                            systems.reproduce(measurement, q1, m, simulator=False)["verified"]
                            is True,
                            "original own physical responses differ",
                        )
                    computed = systems.scores(
                        measurement, truth["validation"], truth["calibration"], weights
                    )
                    all_scores[mode][interface] = computed
                    arms[mode][interface] = {
                        "per_system": measurement.record(),
                        "scores": systems.summaries(computed),
                        "pooled_q1_slope": float(measurement.q1.mean()),
                        "pooled_q2_held": downstream.as_list(measurement.q2.mean(0)),
                    }
                arms[mode]["forecast_replay"] = forecast_check
        contrasts = {}
        flags = {}
        for new, old in (("single", "own"), ("pair", "own"), ("pair", "single")):
            name = f"{new}_minus_{old}"
            contrasts[name] = {
                i: systems.difference(all_scores[new][i], all_scores[old][i]) for i in INTERFACES
            }
            flags[name] = {
                k: contrasts[name]["scalar"][k]["estimate"] <= v
                for k, v in (("q1_population_abs_error", 0.01), ("q2_population_relative", 0.05))
            }
        _head(commit)
        write_json(
            result,
            {
                "verified": True,
                "commit": commit,
                "command": command,
                "seed": seed,
                "primary_passed": trial["passed"],
                "primary_checks": trial["checks"],
                "source": str(source),
                "source_sha256": reported[str(source)],
                "data_sha256": splits,
                "input_sha256": ledger,
                "note_sha256": sha(out / "NOTE.md"),
                "arms": arms,
                "contrasts": contrasts,
                "scalar_within_margin": flags,
                "hold_grid": downstream.as_list(grid),
                "bootstrap": prior["bootstrap"],
                "starts": list(downstream.STARTS),
                "views": views,
                "groups": groups,
                "calibration_groups": len(cal),
                "system_ids": list(range(groups)),
                "spearman_assignment": released["spearman_assignment"],
                "truth": truth["validation"].record(),
            },
        )


def main() -> None:
    """Read the expected revision and completed support seed."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("commit")
    cli.add_argument("seed", type=int, choices=range(5))
    args = cli.parse_args()
    run(args.commit, args.seed)


if __name__ == "__main__":
    main()
