"""Check checkpoint and control routing for validation intervention comparisons."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from scjepa.experiments.storage import write_json

from . import changed_queries as query
from . import chart, kicks


def test_default_pair_and_explicit_subset(tmp_path: Path) -> None:
    """Default to both original endpoints; a supplied changed endpoint is the only model."""
    for endpoints in (chart.endpoints, kicks.endpoints):
        assert endpoints(2) == {
            arm: chart.RUNS / f"seed2/sparse-500k/{arm}/final.pt" for arm in ("baseline", "full")
        }
        explicit = {"full": tmp_path / "changed.pt"}
        assert endpoints(2, explicit) == explicit
        with pytest.raises(ValueError, match="sparse arm"):
            endpoints(2, {})
        with pytest.raises(ValueError, match="sparse arm"):
            endpoints(2, {"joint": tmp_path})


@pytest.mark.parametrize("method", ["vision-direct", "vision-loss"])
def test_terminal_routing_and_provenance(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain failed pilots' actual endpoints, reject wrong bytes and skip unqualified seeds."""

    def blob(path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(path))
        return query.sha(path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(query, "DATA", Path("dataset"))
    monkeypatch.setattr(query, "ROOT", Path("archive"))
    monkeypatch.setattr(query, "OUTPUT", Path("trials"))
    root, original = query.OUTPUT / f"{method}-s0", query.ROOT / "seed0"
    root.mkdir(parents=True)
    split_ids = {s: s for s in ("train", "calibration", "validation")}
    manifest = blob(query.DATA / "manifest.json")
    prefix = "PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code."
    cmd = prefix + (
        f"vision_acquisition {query.TRAINING} 0 direct"
        if method == "vision-direct"
        else f"vision_loss {query.TRAINING} 0"
    )
    (root / "NOTE.md").write_text("\n".join((manifest, cmd, *split_ids.values())))
    meta: dict[str, Any] = {
        "seed": 0,
        "commit": query.TRAINING,
        "passed": False,
        "checks": {"forecast": False},
    }
    reported: dict[str, str] = {}
    digests: dict[str, str] = {}
    result = {"step": 500_000}
    if method == "vision-direct":
        joint = original / "joint/final.pt"
        reported[str(joint)] = blob(joint)
        digests["joint"] = reported[str(joint)]
        meta.update(data_manifest_sha256=manifest, validation={}, checkpoint_sha256=digests)
    for arm in query.ARMS[method]:
        directory = root / f"sparse-500k/{arm}" if method == "vision-direct" else root
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / "final.pt"
        torch.save(
            {
                "meta": {"stage": arm, "seed": 0},
                "config": {"stage": arm, "seed": 0, "steps": 500_000, "data_dir": str(query.DATA)},
            },
            final,
        )
        write_json(directory / "final.json", result)
        last = (root if method == "vision-direct" else original) / f"sparse-300k/{arm}/last.pt"
        fork = {"source": str(last), "source_step": 300_000, "source_sha256": blob(last)}
        write_json(directory / "fork.json", fork)
        if method == "vision-loss":
            control = original / f"sparse-500k/{arm}/final.pt"
            reported[str(control)] = blob(control)
            write_json(control.parent / "fork.json", fork)
            meta.update(
                command=cmd,
                splits=split_ids,
                source=str(last),
                source_sha256=query.sha(last),
                reference=str(control),
                reference_sha256=query.sha(control),
                sha256=query.sha(final),
                validation=result,
                scientific_gate_passed=False,
                control_replay_matched=True,
            )
        else:
            meta["validation"][arm] = result
            parent = original / "joint/final.pt"
            for name, key in (
                ("projected", "projected"),
                ("adapt", "adapt"),
                (f"sparse-300k/{arm}", f"300k/{arm}"),
                (f"sparse-500k/{arm}", f"500k/{arm}"),
            ):
                stage = root / name
                if stage == directory:
                    parent = last
                else:
                    blob(stage / "final.pt")
                    write_json(stage / "final.json", {})
                (stage / "NOTE.md").write_text(
                    f"{query.TRAINING}\nParent: {parent}\n{query.sha(parent)}"
                )
                digests[key] = query.sha(stage / "final.pt")
                write_json(
                    stage / "review-completion.json",
                    {
                        "commit": query.TRAINING,
                        "checkpoint_sha256": digests[key],
                        "parent_sha256": query.sha(parent),
                        "summary_sha256": query.sha(stage / "final.json"),
                    },
                )
                parent = stage / "final.pt"
    write_json(root / "metrics.json", meta)
    paths, status = query.sources(method, 0, split_ids, reported, query.Inputs())
    assert status["scientific_gate_passed"] is False
    assert paths == (
        {a: root / f"sparse-500k/{a}/final.pt" for a in ("baseline", "full")}
        if method == "vision-direct"
        else {"full": root / "final.pt"}
    )
    assert status["skipped"] == {}
    paths["full"].write_text("wrong endpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        query.sources(method, 0, split_ids, reported, query.Inputs())
    paths, status = query.sources(method, 1, split_ids, reported, query.Inputs())
    assert paths == {}
    assert status["skipped"] == dict.fromkeys(query.ARMS[method], "pilot did not qualify expansion")


@pytest.mark.parametrize("name", ["chart", "kicks"])
def test_paired_signed_error_and_changed_reference(name: str) -> None:
    """Calibrated hold and held-kick errors use their own original arm and physical reference."""
    row: dict[str, Any] = {
        "u2a_to_u2b": -1.0,
        "mean_across_beads": {"held": {"relative_error": 3.0}},
    }
    ref: dict[str, Any] = {
        "seed": 0,
        "data_sha256": {},
        "validation_groups": 2,
        "validation_contexts": 5,
        "calibration_groups": 3,
        "arms": {"full": {chart: deepcopy(row) for chart in ("scalar", "affine")}},
        "hold_grids": [[1.0]],
        "truth_slopes_held_by_outcome": [[1.0] * 8 for _ in range(8)],
        "truth": {"2a": {"held": [[1.0]]}},
    }
    trial = deepcopy(ref)
    trial["arms"]["full"]["scalar"].update(
        u2a_to_u2b=0.0, mean_across_beads={"held": {"relative_error": 2.0}}
    )
    deltas = query.paired(name, trial, ref, "full")
    assert deltas["scalar"]["trial_minus_control"] == -1
    assert deltas["affine"]["trial_minus_control"] == 0
    trial["truth"]["2a"]["held"][0][0] += 0.1
    trial["hold_grids"][0][0] += 0.1
    with pytest.raises(AssertionError):
        query.paired(name, trial, ref, "full")


def test_absent_kick_retains_relative_error() -> None:
    """An identically zero partner response has unit relative error and undefined cosine."""
    reference = {"2a": {s: torch.eye(2) for s in ("held", "free", "held_minus_free")}}
    trial = {"2a": reference["2a"] | {"held": torch.zeros((2, 2))}}
    outcome = kicks.score(trial, reference, allow_zero_prediction=True)
    assert outcome["mean_across_beads"]["held"] == {"relative_error": 1.0, "cosine": None}
    with pytest.raises(ValueError, match="undefined relative matrix score"):
        kicks.score(trial, reference)
