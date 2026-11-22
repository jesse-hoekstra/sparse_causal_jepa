"""Check system membership, paired uncertainty and the hash-verified query inventory."""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from diagnostics.vision_dimer import dgp, downstream
from diagnostics.vision_dimer.model import build_model
from scjepa.experiments.storage import write_json

from . import system_effects as audit


def test_system_membership_and_original_queries() -> None:
    """Each system matches an independent release evaluation on exactly its own episodes."""
    rng = torch.Generator().manual_seed(7)
    groups, views = downstream.BATCH_GROUPS + 1, 2
    trajectories = {
        "states": torch.randn(groups, views, 40, 8, generator=rng),
        "actions": torch.randn(groups, views, 39, 5, generator=rng),
        "theta": torch.rand(groups, 4, generator=rng) + 0.1,
    }
    grids = [torch.linspace(-0.7, 1.2, 5, dtype=torch.double) for _ in range(8)]

    def world(data: dict[str, torch.Tensor], size: int) -> downstream.Truth:
        return downstream.Truth(
            dgp.Physics(), downstream.truth_contexts(data, downstream.STARTS, size)
        )

    simulator = world(trajectories, groups)
    result = audit.outcomes(
        simulator,
        lambda a: simulator.rollout(simulator.hold_value(4, a), None),
        grids[4],
        groups,
        views,
    )
    assert result.q1.shape == (groups, 3, views, 5)
    assert result.q2.shape == (groups, 3, views, 4, 2, 3)
    effects = result.effects(grids[4])
    for i in (0, downstream.BATCH_GROUPS - 1, downstream.BATCH_GROUPS):
        subset = {k: v[i : i + 1] for k, v in trajectories.items()}
        ref = downstream.experiments(world(subset, 1), grids)
        assert math.isclose(effects.q1[i].item(), ref["q1"]["u2a"]["slopes"][6])
        for b, name in enumerate(audit.BEADS):
            torch.testing.assert_close(
                effects.q2[i, b], torch.tensor(ref["q2"][name]["held"], dtype=torch.double)
            )
    ref = downstream.experiments(simulator, grids)
    matrix = torch.tensor([ref["q2"][b]["held"] for b in audit.BEADS], dtype=torch.double)
    slope = ref["q1"]["u2a"]["slopes"][6]
    assert audit.reproduce(effects, slope, matrix, simulator=True)["verified"]
    assert not audit.reproduce(effects, slope + 0.01, matrix, simulator=False)["verified"]


def test_bias_cancellation_and_calibrated_constant() -> None:
    """Equal pooled responses can coexist with large system-level errors and negative skill."""
    s = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
    m = s[:, None, None, None].expand(-1, 4, 2, 2)
    truth = audit.Effects(s, m)
    calibration = audit.Effects(s + 1, m + 1)
    distortion = torch.tensor([-2.0, 2.0, -2.0, 2.0], dtype=torch.double)
    model = audit.Effects(s + distortion, m + distortion[:, None, None, None])
    weights = audit.bootstrap_weights(4, 64)
    result = audit.scores(model, truth, calibration, weights)
    assert result["q1_system_mae"][0] == 2
    assert result["q1_population_abs_error"][0] == 0
    assert math.isclose(result["q2_system_relative"][0].item(), float((2 / s).mean()))
    assert result["q2_population_relative"][0] == 0
    expected_skill = 1 - 4 / (s - (s + 1).mean()).square().mean().item()
    assert math.isclose(result["q1_skill"][0].item(), expected_skill)
    assert math.isclose(result["q2_skill"][0].item(), expected_skill)
    fixed = audit.scores(calibration.constant(4), truth, calibration, weights)
    assert fixed["q1_skill"][0] == 0
    assert fixed["q2_skill"][0] == 0
    change = audit.difference(result, result)
    assert change["q1_system_mae"]["interval"] == [0.0, 0.0]
    assert change["q2_population_relative"]["interval"] == [0.0, 0.0]
    torch.testing.assert_close(audit.bootstrap_weights(4, 64), weights)
    assembled = audit.summaries(result)
    assert assembled["q1_bias"]["estimate"] == 0.0
    assert len(assembled["q2_bias"]["estimate"]) == 4


def test_undefined_skill_and_relative_error() -> None:
    """Keep absolute errors while flagging near-constant truth and unresolvable matrix norms."""
    s = torch.ones(3, dtype=torch.double)
    m = s[:, None, None, None].expand(-1, 4, 2, 2).clone()
    m[0, 0] = 1e-16
    truth = audit.Effects(s, m)
    constant = audit.Effects(s + 1, m + 1)
    weights = torch.tensor(
        [[1 / 3, 1 / 3, 1 / 3], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]], dtype=torch.double
    )
    scores = audit.summaries(audit.scores(constant, truth, constant, weights))
    assert scores["q1_skill"]["estimate"] is None
    assert scores["q1_system_mae"]["estimate"] == 1.0
    assert scores["q2_system_relative"] == {
        "estimate": None,
        "interval": None,
        "valid_resamples": 1,
    }
    assert math.isclose(scores["q2_mse"]["estimate"], 1.0)


def test_receipts_select_every_existing_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare both original/direct arms, loss full and one-head arms; detect stale omissions."""
    monkeypatch.chdir(tmp_path)
    out, archive = Path("trials"), Path("released")
    monkeypatch.setattr(audit, "OUTPUT", out)
    monkeypatch.setattr(audit, "ROOT", archive)
    monkeypatch.setattr(audit, "checkpoint_hashes", Mock(return_value={}))
    splits = {s: s for s in ("train", "calibration", "validation")}
    cohort = {
        "seed": 0,
        "calibration_groups": 8,
        "validation_groups": 4,
        "validation_contexts": 120,
        "data_sha256": {s: s for s in ("calibration", "validation")},
    }
    all_arms = ("full", "baseline")

    def prior(root: Path, arms: tuple[str, ...]) -> tuple[dict[str, Path], dict[str, Any]]:
        paths = {a: root / f"sparse-500k/{a}/final.pt" for a in arms}
        rows: dict[str, Any] = {}
        for a, p in paths.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(p))
            rows[a] = {
                "checkpoint": str(p),
                "checkpoint_sha256": audit.sha(p),
                **{
                    interface: {"u2a_to_u2b": 1.0, "matrices": {}}
                    for interface in ("scalar", "affine")
                },
            }
        return paths, {
            "chart": cohort
            | {"arms": rows, "truth_slopes_held_by_outcome": [1.0], "hold_grids": [[1.0]]},
            "kicks": cohort | {"arms": rows, "truth": {"bead": [1.0]}},
        }

    _, old = prior(archive / "seed0", all_arms)
    monkeypatch.setattr(audit.previous, "controls", Mock(return_value=old))
    states: dict[str, Any] = {}
    for scope, methods in (
        ("changed-queries", audit.previous.ARMS),
        ("one-head-queries", {audit.previous.ONE_HEAD: all_arms}),
    ):
        rev = "a" * 40
        cmd = (
            f"PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.changed_queries {rev} 0"
        )
        if scope == "one-head-queries":
            cmd += " --one-head"
        directory = out / (scope + "-s0")
        directory.mkdir(parents=True, exist_ok=True)
        note = directory / "NOTE.md"
        note.write_text(cmd)
        steps: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for method, arms in methods.items():
            paths, records = prior(out / (method + "-s0"), arms)
            inputs.update({str(p): audit.sha(p) for p in paths.values()})
            status: dict[str, Any] = {"training_gate_passed": False, "skipped": {}}
            states[method] = (paths, status)
            steps[method] = status | {
                "evaluated_arms": list(paths),
                "queries": {k: {"measurement": v} for k, v in records.items()},
            }
        write_json(
            directory / "metrics.json",
            {
                "seed": 0,
                "command": cmd,
                "commit": rev,
                "verified": True,
                "methods": steps,
                "note_sha256": audit.sha(note),
                "data_sha256": splits,
                "input_sha256": inputs,
            },
        )

    def source(
        method: str, *_args: object, **_kwargs: object
    ) -> tuple[dict[str, Path], dict[str, Any]]:
        return states[method]

    monkeypatch.setattr(audit.previous, "sources", source)
    models, skipped, _ = audit.inventory(0, splits, audit.previous.Inputs())
    assert set(models) == {
        "original/full",
        "original/baseline",
        "vision-direct/full",
        "vision-direct/baseline",
        "vision-loss/full",
        "vision-one-head/full",
        "vision-one-head/baseline",
    }
    assert not skipped
    path = out / "changed-queries-s0/metrics.json"
    original = audit.read(path)
    changed = deepcopy(original)
    changed["methods"]["vision-loss"]["evaluated_arms"] = []
    write_json(path, changed)
    with pytest.raises(ValueError, match="stale"):
        audit.inventory(0, splits, audit.previous.Inputs())
    write_json(path, original)
    # Removing a query receipt while a conditional model exists is a technical failure.
    conditional = out / "one-head-queries-s0/metrics.json"
    prior_conditional = audit.read(conditional)
    conditional.unlink()
    with pytest.raises(ValueError, match="Missing one-head"):
        audit.inventory(0, splits, audit.previous.Inputs())
    # A genuinely unactivated conditional route requires all five matched direct decisions.
    trained = out / "vision-one-head-s0"
    moved = out / "fixture-held-out"
    trained.rename(moved)
    init = {"prediction_mse": 1.0, "calibration_prediction_mse": 1.0, "parameter_r2_min": 1.0}
    arm = {
        "mcc": 0.9,
        "state_shd": 0,
        "action_shd": 0,
        "parameter_shd": 0,
        "nmse": 1.0,
        "parameter_r2_min": 1.0,
    }
    arms = {"full": arm, "baseline": arm | {"mcc": 0.7}}
    monkeypatch.setattr(audit, "DATA", Path("fixture-data"))
    write_json(audit.DATA / "manifest.json", {})
    data_id = audit.sha(audit.DATA / "manifest.json")
    for n in range(5):
        direct = out / f"vision-direct-s{n}"
        write_json(
            direct / "metrics.json",
            {
                "commit": audit.previous.TRAINING,
                "seed": n,
                "passed": True,
                "data_manifest_sha256": data_id,
                "adaptation": init,
                "reference_adaptation": init,
                "adaptation_checks": audit.adaptation_checks(init, init),
                "validation": arms,
                "reference_validation": arms,
                "checks": audit.sparse_checks(arms, arms),
            },
        )
        (direct / "NOTE.md").write_text(audit.previous.TRAINING + " " + data_id)
    available, missing, _ = audit.inventory(0, splits, audit.previous.Inputs())
    assert len(available) == 5
    assert set(missing) == {"vision-one-head/full", "vision-one-head/baseline"}
    moved.rename(trained)
    write_json(conditional, prior_conditional)
    changed = deepcopy(original)
    changed["note_sha256"] = "f" * 64
    write_json(path, changed)
    with pytest.raises(ValueError, match="SHA-256"):
        audit.inventory(0, splits, audit.previous.Inputs())
    write_json(path, original)
    stage: Path = states["vision-loss"][0]["full"]
    stage.write_text("replaced")
    with pytest.raises(ValueError, match="match|SHA-256"):
        audit.inventory(0, splits, audit.previous.Inputs())


@torch.inference_mode()
def test_both_interfaces_reproduce_native_held_predictions() -> None:
    """Preserve the deployed affine versus coordinatewise hold for each context."""
    groups, views = 2, 2
    n = groups * views * len(downstream.STARTS)
    rng = torch.Generator().manual_seed(42)

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=rng, dtype=torch.double)

    lat = {"online": sample(n, 12, 8), "actions": sample(n, 11, 5), "code": sample(n, 4)}
    cal = sample(80, 8)
    physical = cal + 0.3 * cal.roll(1, 1) + 0.2
    model = build_model(4, phase="full", code_dim=4, context=downstream.CONTEXT).eval()
    scalar = downstream.Model(model, lat, cal, physical)
    affine = audit.chart.AffineChart.fit(cal, physical)
    joint = audit.kicks.AffineWorld(model, lat, affine)
    grid = torch.tensor([-0.8, -0.4, 0.0, 0.3, 0.7], dtype=torch.double)

    def scalar_hold(value: float) -> torch.Tensor:
        return scalar.decode(scalar.rollout(scalar.hold_value(4, value), None))

    def joint_hold(value: float) -> torch.Tensor:
        return audit.affine_hold(model, lat, affine, 4, value)

    for world, hold in ((scalar, scalar_hold), (joint, joint_hold)):
        result = audit.outcomes(world, hold, grid, groups, views)
        raw = torch.stack([hold(float(level))[:, 6] for level in grid])
        q1 = float(downstream.slope(grid, raw.mean(1, keepdim=True))[0])
        matrices = audit.kicks.responses(world)
        q2 = torch.stack([matrices[bead]["held"] for bead in audit.BEADS])
        assert audit.reproduce(result.effects(grid), q1, q2, simulator=False)["verified"]
