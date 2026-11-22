"""Check checkpoint and control routing for validation intervention comparisons."""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

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


@pytest.mark.parametrize("method", ["vision-direct", "vision-loss", query.ONE_HEAD])
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
    training = "b" * 40 if method == query.ONE_HEAD else query.TRAINING
    cmd = (
        prefix
        + {
            "vision-direct": f"vision_acquisition {training} 0 direct",
            "vision-loss": f"vision_loss {training} 0",
            query.ONE_HEAD: f"vision_one_head {training} 0 prepare",
        }[method]
    )
    (root / "NOTE.md").write_text("\n".join((manifest, cmd, *split_ids.values())))
    meta: dict[str, Any] = {
        "seed": 0,
        "commit": training,
        "passed": False,
        "checks": {"forecast": False},
    }
    reported: dict[str, str] = {}
    digests: dict[str, str] = {}
    result = {"step": 500_000}
    staged = method != "vision-loss"
    arms = ("baseline", "full") if staged else ("full",)
    if staged:
        joint = original / "joint/final.pt"
        reported[str(joint)] = blob(joint)
        digests["joint"] = reported[str(joint)]
        meta.update(data_manifest_sha256=manifest, validation={}, checkpoint_sha256=digests)
    if method == query.ONE_HEAD:
        source = original / "acquire/a0/final.pt"
        digests["a0"] = blob(source)
        candidate = root / "candidate.json"
        write_json(
            candidate,
            {
                "final_sha256": digests["a0"],
                "last_sha256": blob(source.with_name("last.pt")),
                "joint_sha256": digests["joint"],
                "verified": True,
                "final_and_last_model_identical": True,
                "seed": 0,
                "commit": training,
                "manifest_sha256": manifest,
                "candidate_path": str(source),
                "archive_audit_sha256": blob(query.OUTPUT / "vision-head-audit-s0/metrics.json"),
            },
        )
        failed = query.OUTPUT / "vision-direct-s0/metrics.json"
        write_json(
            failed,
            {
                "seed": 0,
                "commit": query.TRAINING,
                "passed": False,
                "data_manifest_sha256": manifest,
            },
        )
        trigger = {
            "failed_direct": [
                {
                    "seed": 0,
                    "commit": query.TRAINING,
                    "sha256": query.sha(failed),
                    "path": str(failed),
                }
            ]
        }
        prepared = root / "prepare.json"
        write_json(
            prepared,
            {
                "commit": training,
                "seed": 0,
                "candidate_sha256": query.sha(candidate),
                "trigger": trigger,
            },
        )
        meta.update(prepare_sha256=query.sha(prepared), trigger=trigger)
    for arm in arms:
        directory = root / f"sparse-500k/{arm}" if staged else root
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
        last = (root if staged else original) / f"sparse-300k/{arm}/last.pt"
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
            parent = original / (
                "acquire/a0/final.pt" if method == query.ONE_HEAD else "joint/final.pt"
            )
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
                (stage / "NOTE.md").write_text(f"{training}\nParent: {parent}\n{query.sha(parent)}")
                digests[key] = query.sha(stage / "final.pt")
                write_json(
                    stage / "review-completion.json",
                    {
                        "commit": training,
                        "checkpoint_sha256": digests[key],
                        "parent_sha256": query.sha(parent),
                        "summary_sha256": query.sha(stage / "final.json"),
                    },
                )
                parent = stage / "final.pt"
    write_json(root / "metrics.json", meta)
    paths, status = query.sources(
        method, 0, split_ids, reported, query.Inputs(), training_commit=training
    )
    assert status["scientific_gate_passed"] is False
    assert paths == (
        {a: root / f"sparse-500k/{a}/final.pt" for a in ("baseline", "full")}
        if staged
        else {"full": root / "final.pt"}
    )
    assert status["skipped"] == {}
    if method == query.ONE_HEAD:
        with pytest.raises(ValueError, match="metrics.json"):
            query.sources(
                method, 0, split_ids, reported, query.Inputs(), training_commit=query.TRAINING
            )
        a0 = original / "acquire/a0/final.pt"
        original_bytes = a0.read_bytes()
        a0.write_bytes(b"changed candidate")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            query.sources(method, 0, split_ids, reported, query.Inputs(), training_commit=training)
        a0.write_bytes(original_bytes)
    paths["full"].write_text("wrong endpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        query.sources(method, 0, split_ids, reported, query.Inputs(), training_commit=training)
    paths, status = query.sources(
        method, 1, split_ids, reported, query.Inputs(), training_commit=training
    )
    assert paths == {}
    assert status["skipped"] == dict.fromkeys(arms, "pilot did not qualify expansion")


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


@pytest.mark.parametrize("one_head", [False, True])
def test_cli_scope_selects_independent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, one_head: bool
) -> None:
    """Default and conditional diagnostics write independent receipts with disjoint routes."""
    monkeypatch.chdir(tmp_path)
    output = Path("trials")
    data = Path("dataset")
    manifest = Path("original.sha256")
    manifest.touch()
    hashes = {s: {"sha256": s} for s in ("train", "calibration", "validation")}
    write_json(data / "manifest.json", {"splits": hashes})
    monkeypatch.setattr(query, "MANIFEST", manifest)
    monkeypatch.setattr(query, "DATA", data)
    monkeypatch.setattr(query, "OUTPUT", output)
    monkeypatch.setattr(query.subprocess, "run", Mock())
    monkeypatch.setattr(query, "_head", Mock())
    monkeypatch.setattr(
        query, "checkpoint_hashes", Mock(return_value={str(data / f"{s}.pt"): s for s in hashes})
    )
    monkeypatch.setattr(query, "controls", Mock(return_value={}))
    monkeypatch.setattr(query, "_idle_gpu", Mock(side_effect=AssertionError("Unexpected scoring")))
    revision = "f" * 40
    opts = ["--one-head"] if one_head else []
    monkeypatch.setattr(sys, "argv", ["changed_queries", revision, "3", *opts])
    query.main()
    prefix = "one-head-queries" if one_head else "changed-queries"
    receipt = query.read(output / f"{prefix}-s3/metrics.json")
    assert receipt["command"].split()[-(len(opts) + 2) :] == [revision, "3", *opts]
    assert set(receipt["methods"]) == ({query.ONE_HEAD} if one_head else set(query.ARMS))
    for row in receipt["methods"].values():
        assert row["training_commit"] == (revision if one_head else query.TRAINING)
        assert not row["evaluated_arms"]


def test_one_head_candidate_rejection_has_no_descendants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a signed early scientific stop and reject an unrelated trigger."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(query, "DATA", Path("dataset"))
    monkeypatch.setattr(query, "OUTPUT", Path("trials"))
    data = query.DATA / "manifest.json"
    write_json(data, {})
    root = query.OUTPUT / f"{query.ONE_HEAD}-s0"
    prior = query.OUTPUT / "vision-direct-s0/metrics.json"
    digest, commit = query.sha(data), "d" * 40
    write_json(
        prior,
        {"commit": query.TRAINING, "seed": 0, "passed": False, "data_manifest_sha256": digest},
    )
    failed = {"commit": query.TRAINING, "seed": 0, "path": str(prior), "sha256": query.sha(prior)}
    record: dict[str, Any] = {
        "seed": 0,
        "commit": commit,
        "passed": False,
        "data_manifest_sha256": digest,
        "reason": "candidate",
        "candidate_checks": {"forecast": False},
        "trigger": {"failed_direct": [failed]},
    }
    write_json(root / "metrics.json", record)
    (root / "NOTE.md").write_text(
        "PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.vision_one_head "
        f"{commit} 0 prepare\n{digest}"
    )
    paths, status = query.sources(query.ONE_HEAD, 0, {}, {}, query.Inputs(), training_commit=commit)
    assert paths == {}
    assert status["skipped"] == dict.fromkeys(
        ("baseline", "full"), "candidate gate stopped training"
    )
    assert status["training_checks"] == record["candidate_checks"]
    failed["sha256"] = "0" * 64
    write_json(root / "metrics.json", record)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        query.sources(query.ONE_HEAD, 0, {}, {}, query.Inputs(), training_commit=commit)
