"""Test disjoint support, covariance correction and deployment through both native models."""

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnostics.masked_mlp_learned import run as syn
from diagnostics.vision_dimer import data
from diagnostics.vision_dimer.model import build_model
from scjepa.models.model import SCJEPA

from .linear_consensus import rgb_eval, synthetic_eval
from .support_pair import SPLITS, dispersion, donor_codes, forecast, physical, scientific_checks


def test_pair_minimizer_and_recipient_exclusion() -> None:
    """Check actual support membership using independent one-hot identity codes."""
    x = torch.eye(10)[None].expand(3, -1, -1)
    pairs, single, two = donor_codes(x)
    for i in range(10):
        assert single[0, i].nonzero().flatten().item() == (i - 1) % 10
        assert {int(v) for v in two[0, i].nonzero().flatten()} == {(i - 1) % 10, (i - 2) % 10}  # pyright: ignore[reportUnknownMemberType]
        assert two[0, i, i] == 0
        assert torch.equal(pairs[0, i // 2] * 2, x[0, 2 * (i // 2)] + x[0, 2 * (i // 2) + 1])
    with pytest.raises(ValueError, match="ten"):
        donor_codes(x[:, :8])


def test_unbiased_covariances_use_disjoint_supports_and_common_groups() -> None:
    """Verify exact finite-sample identities, common units and singular chart handling."""
    g = torch.Generator().manual_seed(432)
    x = torch.randn(40, 1, 3, dtype=torch.float64, generator=g) + torch.randn(
        40, 10, 3, dtype=torch.float64, generator=g
    )
    pair = donor_codes(x)[0]
    s, common, weights = dispersion(x, pair, "calibration", None, draws=37)
    for name, z in (("single", x), ("pair", pair)):
        means = z.mean(1)
        raw = z - means[:, None]
        w = (raw.mT @ raw).mean(0) / (z.shape[1] - 1)
        b = means.T.cov() - w / z.shape[1]
        torch.testing.assert_close(torch.tensor(s[name]["within"], dtype=z.dtype), w)
        torch.testing.assert_close(torch.tensor(s[name]["between"], dtype=z.dtype), b)
    assert weights.shape == (38, 40)
    assert s["single"]["rank"] == common.shape[1] == 3
    assert s["worst_difference_ci95"][1] < 0
    constant = torch.ones(10, 10, 2)
    sc, _, _ = dispersion(constant, donor_codes(constant)[0], "calibration", None, draws=4)
    assert sc["worst_difference_ci95"] is None


def test_smoke_native_forecasts_and_physically_aligned_scores(tmp_path: Path) -> None:
    """Score the real GELU and RGB models on CPU with no test-split evaluation."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _native_smoke(tmp_path)
    finally:
        torch.set_num_threads(old)


def _native_smoke(tmp: Path) -> None:
    cpu = torch.device("cpu")
    cohort = tmp / "rgb"
    data.prepare(
        cohort,
        data.DataConfig(
            train_groups=6, calibration_groups=8, validation_groups=4, episode_frames=36
        ),
    )
    rgb = build_model(0, phase="full", code_dim=4, context=10).eval()
    inputs = {s: data.RenderedInputs(cohort, s, cpu) for s in SPLITS}
    r, sm = rgb_eval(rgb, inputs, data_dir=cohort)
    _check_model(
        rgb,
        r,
        sm,
        {s: v.actions[:, :, : v.frames - 1] for s, v in inputs.items()},
        "vision",
        cohort,
    )
    from diagnostics.shared import data as syn_data
    from diagnostics.shared import source_data

    physical_data = tmp / "physics"
    cache = tmp / "synthetic"
    source_data.prepare(
        source_data.PrepareConfig(
            str(physical_data),
            train_groups=6,
            calibration_groups=8,
            validation_groups=4,
            test_groups=2,
            profile="smoke",
        )
    )
    syn_data.prepare(syn_data.PrepareConfig(str(cache), str(physical_data), profile="smoke"))
    sets, _ = syn_data.load_data(cache)
    synthetic = syn.build_control("full", 0, predictor_activation="gelu").eval()
    r, sm = synthetic_eval(synthetic, sets, cpu)
    _check_model(
        synthetic, r, sm, {s: sets[s].tensors["actions"] for s in SPLITS}, "synthetic", cache
    )


def _check_model(
    model: SCJEPA,
    r: dict[str, Any],
    sm: dict[str, dict[str, torch.Tensor]],
    actions: dict[str, torch.Tensor],
    dataset: str,
    data_dir: Path,
) -> None:
    cpu = torch.device("cpu")
    codes = {s: sm[s]["codes"].float() for s in SPLITS}
    two = {s: donor_codes(codes[s])[0] for s in SPLITS}
    both = {
        k: physical(model, sm, c, dataset, r, data_dir)
        for k, c in (("single", codes), ("pair", two))
    }
    snapshots: dict[str, Any] = {}
    for s in SPLITS:
        native, pair, replay = forecast(model, sm[s], actions[s], cpu)
        assert replay < 1e-5
        assert math.isclose(float(native.mean()), r[s]["mse"], rel_tol=1e-5)
        assert math.isclose(both["single"][s]["r2"], r[s]["r2"], rel_tol=1e-5, abs_tol=1e-6)
        assert both["single"][s]["shd"] == r[s]["shd"]
        assert pair.shape == native.shape
        snapshots[s] = {
            "dispersion": {
                "single": {"rank": 3, "dimension": 3, "worst_within_total": 0.5},
                "pair": {"rank": 3, "dimension": 3, "worst_within_total": 0.3},
                "worst_difference_ci95": [-0.3, -0.1],
            },
            "forecast": {"native_mse": float(native.mean()), "pair_mse": float(pair.mean())},
            "physical": {k: v[s] for k, v in both.items()},
        }
    flags = scientific_checks(snapshots, dataset)
    assert flags["validation_rank"]
    snapshots["validation"]["dispersion"]["worst_difference_ci95"] = [-0.12, 0.01]
    assert not scientific_checks(snapshots, dataset)["validation_upper_bound"]


def test_validator_rebuilds_sample_statistics_and_rejects_gate_tampering(tmp_path: Path) -> None:
    """Reconstruct the complete numerical receipt from arrays and catch a changed verdict."""
    import numpy as np

    from scjepa.experiments.storage import write_json

    from .support_pair import _forecast_summary, validate  # pyright: ignore[reportPrivateUsage]
    from .vision_acquisition import sha

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        cmd = "python support_pair fixture"
        (tmp_path / "NOTE.md").write_text(cmd)
        arr: dict[str, Any] = {}
        splits: dict[str, Any] = {}
        common = None
        for idx, s in enumerate(SPLITS):
            gen = torch.Generator().manual_seed(123 + idx)
            z = torch.randn(18, 1, 3, generator=gen) + torch.randn(18, 10, 3, generator=gen)
            p = donor_codes(z)[0]
            native = torch.rand(18, generator=gen).double() + 0.2
            pair = native * 0.99
            m, common, w = dispersion(z, p, s, common)
            ph = {"r2": 0.99, "shd": 0, "mcc": 0.9}
            splits[s] = {
                "dispersion": m,
                "forecast": _forecast_summary(native, pair, w, 2.0),
                "physical": {"single": ph, "pair": ph},
            }
            for k, v in (
                ("single_codes", z),
                ("pair_codes", p),
                ("native_mse", native),
                ("pair_mse", pair),
            ):
                arr[s + "_" + k] = v.numpy()
        np.savez_compressed(tmp_path / "system-samples.npz", **arr)  # pyright: ignore[reportUnknownMemberType]
        checks = scientific_checks(splits, "vision")
        r = {
            "command": cmd,
            "commit": "fixture",
            "dataset": "vision",
            "tau": 2.0,
            "splits": splits,
            "checks": checks,
            "passed": all(checks.values()),
            "note_sha256": sha(tmp_path / "NOTE.md"),
            "samples_sha256": sha(tmp_path / "system-samples.npz"),
        }
        write_json(tmp_path / "metrics.json", r)
        assert validate(tmp_path)["passed"] == r["passed"]
        write_json(tmp_path / "metrics.json", r | {"passed": not r["passed"]})
        with pytest.raises(ValueError, match="gate differs"):
            validate(tmp_path)
    finally:
        torch.set_num_threads(threads)
