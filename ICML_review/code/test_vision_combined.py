"""Check joint constituent gates, fixed native controls and sibling continuation."""

import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from diagnostics.vision_dimer import data, train
from scjepa.experiments.storage import write_json

from . import vision_combined as combo
from . import vision_invariant, vision_loss
from .changed_queries import Inputs


def test_both_physical_margins_must_hold_on_the_same_four_seeds() -> None:
    """Separate four marginal successes from four jointly successful seeds."""
    rows: dict[str, list[dict[str, Any]]] = {
        f: [
            {
                "seed": s,
                "commit": f,
                "passed": s != 2,
                "query_checks": {"chart": s != 3, "kicks": s != 3},
            }
            for s in range(5)
        ]
        for f in combo.FAMILIES
    }
    assert combo.eligibility(rows)["passed"]
    family = rows["rgb-four"]
    family[3]["query_checks"]["kicks"] = True
    family[4]["query_checks"]["kicks"] = False
    result = combo.eligibility(rows)
    assert result["status"] == "ineligible"
    assert result["passing_counts"]["rgb-four"] == {
        "primary": 4,
        "chart": 4,
        "kicks": 4,
        "both_queries": 3,
    }
    family.pop()
    assert combo.eligibility(rows)["status"] == "pending"
    family.append(dict(family[0]))
    with pytest.raises(ValueError, match="rgb-four"):
        combo.eligibility(rows)


def test_absolute_gate_and_four_head_baseline_survive_the_combination() -> None:
    """Reject a changed baseline gain and an excessive calibration forecast."""
    old = {
        "mse_matched": True,
        "mse": 0.1,
        "chart": {"full_rank": True, "eigenvalues": [1.0] * 4},
        "reported_spread": {"worst_direction_share": 0.5},
    }
    new = old | {"reported_spread": {"worst_direction_share": 0.35}}
    semantic = {
        "mcc": 0.96,
        "parameter_r2_min": 0.97,
        "state_shd": 1,
        "action_shd": 0,
        "parameter_shd": 0,
        "nmse": 0.04,
    }
    controls = {
        "validation": {"baseline": {"mcc": 0.81}},
        "reference_validation": {"full": semantic, "baseline": {"mcc": 0.78}},
    }
    nat: dict[str, Any] = {"semantic": semantic, "validation": old, "calibration": old}
    child: dict[str, Any] = {"semantic": semantic, "validation": new, "calibration": dict(new)}
    assert all(combo.endpoint_checks(nat, child, controls).values())
    child["calibration"]["mse"] = 0.106
    controls["validation"]["baseline"]["mcc"] = 0.89
    assert {k for k, v in combo.endpoint_checks(nat, child, controls).items() if not v} == {
        "calibration_mse",
        "gain",
    }
    child["validation"]["mse_matched"] = False
    with pytest.raises(ValueError, match="Unreproduced"):
        combo.endpoint_checks(nat, child, controls)


def test_real_native_replay_and_combined_changed_parent_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the fresh-route trainer, teacher, numerical replay and first-window measurements."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _smoke(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(threads)


def _smoke(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = root / "data"
    data.prepare(
        cohort,
        data.DataConfig(
            episode_frames=16, train_groups=8, calibration_groups=8, validation_groups=4
        ),
    )
    cfg = train.StageConfig(
        "joint",
        str(cohort),
        str(root / "joint"),
        device="cpu",
        batch_groups=2,
        code_dim=4,
        steps=1,
        log_every=1,
        checkpoint_every=1,
        eval_every=1000,
    )
    train.train(cfg)
    cfg = replace(
        cfg,
        stage="full",
        parent=str(root / "joint/final.pt"),
        output_dir=str(root / "parent"),
        tau=0.05,
    )
    train.train(cfg)
    parent = root / "parent/last.pt"
    # The fixture retains the first real optimizer state and uses the release's continuation clock.
    toy = train.load_checkpoint(parent)
    toy["step"] = 300000
    torch.save(toy, parent)
    cfg = replace(cfg, output_dir=str(root / "native"), steps=300002)
    train.fork(parent, cfg)
    train.train(cfg, resume=True)
    metadata = root / "native/wandb/run-toy/files/wandb-metadata.json"
    write_json(metadata, {"gpu": "CPU fixture", "root": str(root / "native")})
    out = root / "combined"
    out.mkdir()
    note = "CPU training smoke"
    (out / "NOTE.md").write_text(note)
    monkeypatch.setattr(combo, "_idle_gpu", lambda: None)
    monkeypatch.setattr(combo, "_runtime", lambda: {"gpu_model": "CPU fixture"})
    monkeypatch.setattr(vision_loss, "_runtime", lambda: {"gpu_model": "CPU fixture"})

    def timing(path: Path, start: int) -> dict[str, Any]:
        # The shared cost reporter otherwise insists on the production endpoint at 500000.
        history = [json.loads(r) for r in path.read_text().splitlines()]
        return {"updates": history[-1]["step"] - start}

    monkeypatch.setattr(vision_invariant, "performance", timing)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fresh = combo._fresh  # pyright: ignore[reportPrivateUsage]
    result, ready = fresh(
        rev,
        0,
        out,
        parent,
        root / "native/final.pt",
        Inputs(),
        note,
        {},
        {"route": "previous_native_one_head"},
    )
    assert ready
    assert result["new_sibling_updates"] == 2
    prep = combo.read(out / "prepare.json")
    assert prep["ready"]
    assert prep["checks"]["native_replay"]
    saved = train.load_checkpoint(out / "sibling/last.pt")
    native = train.load_checkpoint(root / "native/last.pt")
    assert native["step"] == saved["step"] == 300002
    for name in ("sampler", "windows", "cpu_rng"):
        torch.testing.assert_close(saved[name], native[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, native["model"][key])
        for key, value in saved["model"].items()
        if key.startswith("parameter_encoder.")
    )
    stage = combo.read(out / "sibling/endpoint.json")
    assert all(stage[s]["mse_matched"] for s in combo.SPLITS)
    assert fresh(
        rev,
        0,
        out,
        parent,
        root / "native/final.pt",
        Inputs(),
        note,
        {},
        {"route": "previous_native_one_head"},
    ) == (result, ready)


def test_inherited_queries_compare_paper_and_four_head_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identical trajectory reuses both measured arms and retains both paired comparisons."""
    cp = tmp_path / "sibling/final.pt"
    cohort: dict[str, Any] = {
        "seed": 0,
        "data_sha256": {"calibration": "c", "validation": "v"},
        "validation_groups": 4,
        "validation_contexts": 120,
        "calibration_groups": 8,
    }
    diag: dict[str, Any] = {
        "checkpoint": str(cp),
        "checkpoint_sha256": "digest",
        "recomputed_first_window_nmse": 0.01,
        "spearman_assignment_matches_stored": True,
        "assignment_matches_stored": True,
    }

    def measurements(value: float) -> dict[str, Any]:
        grid = [[0.0] * 8 for _ in range(8)]
        interfaces = {"scalar": {"u2a_to_u2b": value}, "affine": {"u2a_to_u2b": value}}
        kick = {"mean_across_beads": {"held": {"relative_error": value}}}
        return {
            "chart": cohort
            | {
                "hold_grids": grid,
                "truth_slopes_held_by_outcome": grid,
                "arms": {"full": diag | interfaces},
            },
            "kicks": cohort
            | {"truth": [1.0], "arms": {"full": diag | {"scalar": kick, "affine": kick}}},
        }

    def unexpected(*_args: object) -> None:
        pytest.fail("An inherited model must not evaluate or train again")

    monkeypatch.setattr(combo.chart, "run", unexpected)
    monkeypatch.setattr(combo.kicks, "run", unexpected)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    original = measurements(0.03)
    four = {k: {"measurement": v} for k, v in measurements(0.02).items()}
    sibling = {k: {"measurement": v} for k, v in measurements(0.035).items()}
    evaluate = combo._evaluate  # pyright: ignore[reportPrivateUsage]
    metrics, gates = evaluate(0, cp, "digest", original, four, sibling, rev)
    assert gates == {"chart": True, "kicks": True}
    for q in combo.MARGINS:
        assert math.isclose(metrics[q]["paired"]["scalar"]["trial_minus_control"], 0.005)
        assert math.isclose(
            metrics[q]["versus_native_four"]["scalar"]["trial_minus_control"], 0.015
        )
    sibling["chart"]["measurement"]["arms"]["full"]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="query model"):
        evaluate(0, cp, "digest", original, four, sibling, rev)
