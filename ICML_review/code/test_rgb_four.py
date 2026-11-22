"""Test four-head TRAIN selection, exact-stage inheritance, and native CPU continuation."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from diagnostics.vision_dimer import data, train
from scjepa.experiments.storage import write_json

from . import rgb_four as four
from . import vision_acquisition as stages
from . import vision_one_head as one


def test_selector_never_uses_later_candidates_or_development_metrics() -> None:
    """A later global winner, tied scores and a malformed full replay are distinguishable."""
    assert four.select_four([3.0, 2.0, 2.0, 7.0, 0.0, *([8.0] * 11)]) == 1
    assert four.select_four([4.0, 3.0, 2.0, 1.0, *([0.0] * 12)]) == 3
    with pytest.raises(ValueError, match="sixteen"):
        four.select_four([0.0] * 4)
    with pytest.raises(ValueError, match="finite"):
        four.select_four([0.0] * 15 + [float("nan")])


def test_cached_authentication_rebinds_only_the_selected_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional indexed route binds both the selected history and the native summaries."""
    base = tmp_path / "archive/seed0"
    d = base / "acquire/a3"
    output = tmp_path / "reviews"
    cohort = tmp_path / "data"
    monkeypatch.setattr(one, "ROOT", base.parent)
    monkeypatch.setattr(one, "OUTPUT", output)
    monkeypatch.setattr(one, "DATA", cohort)
    sources = {
        "final_sha256": d / "final.pt",
        "last_sha256": d / "last.pt",
        "joint_sha256": base / "joint/final.pt",
        "config_sha256": d / "config.json",
        "selection_sha256": base / "selection.json",
        "history_sha256": d / "history.jsonl",
        "archive_audit_sha256": output / "vision-head-audit-s0/metrics.json",
        "manifest_sha256": cohort / "manifest.json",
    }
    summaries = {"a3": d / "final.json", "joint": base / "joint/final.json"}
    for path in (*sources.values(), *summaries.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(path))
    row: dict[str, Any] = {
        "commit": "revision",
        "seed": 0,
        "candidate_index": 3,
        "first_window": {
            label: {"summary_sha256": four.sha(path)} for label, path in summaries.items()
        },
        **{key: four.sha(path) for key, path in sources.items()},
    }
    target = output / "rgb-four-s0/candidate.json"
    write_json(target, row)
    receive = one._candidate_receipt  # pyright: ignore[reportPrivateUsage]
    assert receive("revision", 0, target, 3) == row
    with pytest.raises(ValueError, match="candidate receipt"):
        receive("revision", 0, target, 1)
    sources["history_sha256"].write_text("changed")
    with pytest.raises(ValueError, match="candidate receipt"):
        receive("revision", 0, target, 3)


@pytest.mark.parametrize(
    "case", ["reported_pass", "reported_sparse_fail", "reported_adapt_fail", "old_adapt_fail"]
)
def test_inheritance_recomputes_gates_and_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """The exact old lineage carries its actual gate and candidate digest without any SGD."""
    output = tmp_path / "review"
    base = tmp_path / "archive"
    out = output / "rgb-four-s0"
    out.mkdir(parents=True)
    note = "test"
    (out / "NOTE.md").write_text(note)
    previous = case == "old_adapt_fail"
    index = 0 if previous else 2
    values = [float(16 - i) for i in range(16)]
    values[index] = 0.0
    choose: dict[str, Any] = {
        "input_sha256": {},
        "selected": index,
        "reported_selected": 6 if previous else index,
        "train_scores": values,
    }
    write_json(out / "selection.json", choose)
    source = base / f"acquire/a{index}/final.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"chosen")
    source.with_name("last.pt").write_bytes(b"resume")
    joint = base / "joint/final.pt"
    joint.parent.mkdir()
    joint.write_bytes(b"joint")
    c: dict[str, Any] = {
        "verified": True,
        "candidate_path": str(source),
        "historical_train_score": values[index],
        "final_sha256": four.sha(source),
        "last_sha256": four.sha(source.with_name("last.pt")),
        "joint_sha256": four.sha(joint),
    }
    write_json(out / "candidate.json", c)
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{}")
    monkeypatch.setattr(four, "_head", Mock(return_value=None))
    monkeypatch.setattr(four, "DATA", dataset)
    monkeypatch.setattr(four, "OUTPUT", output)
    monkeypatch.setattr(four, "_selection", Mock(return_value=choose))
    monkeypatch.setattr(four, "_trigger", Mock(return_value=None))
    monkeypatch.setattr(four, "_candidate_receipt", Mock(return_value=c))
    monkeypatch.setattr(four, "_native_chain", Mock(return_value={"projected": "p", "adapt": "d"}))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inherited stage launched training")

    monkeypatch.setattr(four, "project_adapt", forbidden)
    monkeypatch.setattr(four, "sparse_arm", forbidden)
    native = {
        "calibration_prediction_mse": 0.01,
        "prediction_mse": 0.02,
        "parameter_r2_min": 0.90 if case == "reported_adapt_fail" else 0.97,
    }
    write_json(base / "adapt/final.json", native)
    if previous:
        old = output / "vision-one-head-s0"
        write_json(old / "candidate.json", c)
        rejected = native | {"calibration_prediction_mse": 0.02}
        write_json(old / "adapt/final.json", rejected)
        write_json(
            old / "metrics.json",
            {
                "seed": 0,
                "commit": "earlier",
                "data_manifest_sha256": four.sha(dataset / "manifest.json"),
                "candidate_sha256": four.sha(old / "candidate.json"),
                "checkpoint_sha256": {"a0": c["final_sha256"]},
                "adaptation": rejected,
                "reference_adaptation": native,
                "adaptation_checks": four.adaptation_checks(rejected, native),
                "passed": False,
                "reason": "adaptation",
            },
        )
    else:
        ref: dict[str, Any] = {}
        for arm in four.ARMS:
            d = base / f"sparse-500k/{arm}"
            steps = {
                "step": 500000,
                "mcc": 0.9 if arm == "full" else 0.6,
                "state_shd": 0,
                "action_shd": 0,
                "parameter_shd": 0,
                "nmse": 0.01,
                "parameter_r2_min": 0.89 if case == "reported_sparse_fail" else 0.97,
            }
            write_json(d / "final.json", steps)
            (d / "final.pt").write_bytes(arm.encode())
            ref[arm] = steps
        monkeypatch.setattr(four, "_reference", Mock(return_value=ref))

        def lineage(
            stage: Path,
            base: Path,
            seed: int,
            arm: str,
            files: dict[str, str],
            *,
            review_rev: str | None,
        ) -> tuple[dict[str, str], Path]:
            assert base == stage
            assert seed == 0
            assert review_rev is None
            cp = stage / f"sparse-500k/{arm}/final.pt"
            files[str(cp)] = four.sha(cp)
            return {"500k/" + arm: four.sha(cp)}, cp

        monkeypatch.setattr(four, "_sparse_lineage", lineage)
    four._prepare("fixture", 0, out, base, note)  # pyright: ignore[reportPrivateUsage]
    done = four.read(out / "metrics.json")
    assert done["passed"] == (case == "reported_pass")
    if case.endswith("adapt_fail"):
        assert done["reason"] == "adaptation"
        assert done["terminal_paths"] == {}
    else:
        assert done["route"] == "reported"
        assert set(done["terminal_paths"]) == set(four.ARMS)
        if case == "reported_sparse_fail":
            assert not done["checks"]["retention"]


def test_real_four_head_native_downstream_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train four real tiny native heads, fit the selected projection, adapt and run both forks."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _cpu(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(threads)


def _cpu(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cohort = tmp / "data"
    base = tmp / "archive"
    out = tmp / "out"
    rev = "cpu-fixture"
    data.prepare(
        cohort,
        data.DataConfig(
            train_groups=8, calibration_groups=8, validation_groups=4, episode_frames=16
        ),
    )
    cfg = train.StageConfig(
        "joint",
        str(cohort),
        str(base / "joint"),
        seed=1,
        device="cpu",
        steps=2,
        batch_groups=2,
        eval_every=2,
        log_every=1,
        checkpoint_every=1,
    )
    train.train(cfg)
    heads: list[Path] = []
    for i in range(4):
        d = base / f"acquire/a{i}"
        heads.append(d)
        train.train(
            replace(
                cfg,
                stage="acquire",
                output_dir=str(d),
                parent=str(base / "joint/final.pt"),
                acquisition_seed=1001 + 100 * i,
            )
        )
    scored = train.select(heads, base / "selection.json", window=2)
    index = four.select_four([scored["scores"][str(p)] for p in heads] + [0.0] * 12)
    assert Path(scored["selected"]) == heads[index]
    source = heads[index] / "final.pt"
    adapt = replace(cfg, stage="adapt", code_dim=4, parent=str(source))
    write_json(base / "adapt/config.json", {"config": asdict(adapt)})
    write_json(base / "adapt/final.json", four.read(source.with_name("final.json")))
    for arm in four.ARMS:
        for label, n in (("300k", 2), ("500k", 4)):
            sp = replace(adapt, stage=arm, tau=0.05, steps=n)  # pyright: ignore[reportArgumentType]
            write_json(base / f"sparse-{label}/{arm}/config.json", {"config": asdict(sp)})

    # Only the dispatcher is replaced: execute the native operations in process on CPU.
    def native(commit: str, directory: Path, cmd: list[str], note: str, parent: Path) -> str:
        assert note == "fixture"
        directory.mkdir(parents=True, exist_ok=True)
        if "project" in cmd:
            train.project(parent.parent, cohort, directory, device="cpu")
        else:
            arm = directory.name
            if directory.name == "adapt":
                cfg2 = replace(adapt, parent=str(parent), output_dir=str(directory))
            else:
                budget = directory.parent.name
                steps = 2 if budget == "sparse-300k" else 4
                cfg2 = replace(
                    adapt,
                    stage=arm,
                    steps=steps,
                    tau=0.05,
                    parent=str(out / "adapt/final.pt"),
                    output_dir=str(directory),
                )  # pyright: ignore[reportArgumentType]
            if "fork" in cmd:
                train.fork(parent, cfg2)
            train.train(cfg2, resume="fork" in cmd)
        cp = directory / "final.pt"
        h = four.sha(cp)
        write_json(
            directory / "review-completion.json",
            {
                "commit": commit,
                "checkpoint_sha256": h,
                "parent_sha256": four.sha(parent),
                "summary_sha256": four.sha(directory / "final.json"),
            },
        )
        return h

    monkeypatch.setattr(stages, "stage_run", native)
    rec = four.project_adapt(
        1, rev, out, "fixture", source, base=base, data_dir=cohort, group="smoke"
    )
    files: dict[str, str] = {}
    hashes = four._native_chain(out, base, source, 1, files, review_rev=rev)  # pyright: ignore[reportPrivateUsage]
    assert hashes == rec["checkpoint_sha256"]
    for arm in four.ARMS:
        original = four.sparse_arm(
            1, rev, out, "fixture", arm, base=base, data_dir=cohort, group="smoke"
        )
        found, endpoint = four._sparse_lineage(  # pyright: ignore[reportPrivateUsage]
            out, base, 1, arm, files, review_rev=rev, budgets=(2, 4)
        )
        assert original == found
        assert four.read(endpoint.with_name("final.json"))["step"] == 4
    four._verify(files)  # pyright: ignore[reportPrivateUsage]
    corrupted = out / "sparse-300k/full/last.pt"
    corrupted.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bound input changed"):
        four._verify(files)  # pyright: ignore[reportPrivateUsage]
