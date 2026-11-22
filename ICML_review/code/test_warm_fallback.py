"""Check branch freezing, validation-failure inheritance and full attempt charging."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from diagnostics.shared.io import read_json
from scjepa.experiments.storage import write_json

from . import warm_fallback as policy


@pytest.mark.parametrize(
    ("warm", "cold", "selected", "count"),
    [
        (True, True, "warm", 1),
        (True, False, "warm", 1),
        (False, True, "cold_a0", 2),
        (False, False, None, 2),
    ],
)
def test_train_only_route(warm: bool, cold: bool, selected: str | None, count: int) -> None:
    """Warm admission commits before cold is called and only one retry is available."""
    seen: list[str] = []

    def attempt(name: str) -> dict[str, Any]:
        seen.append(name)
        return {"method": name, "admitted": warm if name == "warm" else cold}

    winner, attempts = policy.choose(attempt)
    assert winner == selected
    assert len(attempts) == count
    assert seen == ["warm", "cold_a0"][:count]


@pytest.mark.parametrize(
    ("n", "admitted", "deploy", "paired"),
    [(1, True, 600000, 950000), (2, True, 700000, 1050000), (2, False, 300000, 300000)],
)
def test_budgets(n: int, admitted: bool, deploy: int, paired: int) -> None:
    """Include both acquisitions on fallback and count the baseline only in the paired study."""
    b = policy.budget(n, admitted)
    assert b["deployment_updates"] == deploy
    assert b["paired_study_updates"] == paired
    assert b["acquisition"] == 100000 * n
    assert b["projection_fits"] == int(admitted)
    assert b["new_optimizer_updates"] == 0


@pytest.mark.parametrize(
    ("warm_admit", "warm_pass", "cold_admit", "cold_pass", "expected", "reason"),
    [
        (True, False, True, True, "warm", "endpoint"),
        (False, False, True, True, "cold_a0", None),
        (False, False, False, False, None, "train_admission"),
    ],
)
def test_replay_freezes_route_before_endpoint_inspection(
    tmp_path: Path,
    warm_admit: bool,
    warm_pass: bool,
    cold_admit: bool,
    cold_pass: bool,
    expected: str | None,
    reason: str | None,
) -> None:
    """Reproduce final policy receipts on toy records, including unfavorable warm endpoints."""
    d = tmp_path / "runs"
    parent = tmp_path / "historic"
    cohort = tmp_path / "data"
    folder = d / "warm-fallback-s0"
    for p in (parent / "calibration", cohort, d / "consensus-queue"):
        p.mkdir(parents=True)
    write_json(cohort / "completion.json", {"identity": "toy"})
    write_json(parent / "selection-s0.json", {"window": 2, "maximum_loss": 0.03})
    write_json(parent / "calibration/calibration.json", {"tau_prediction": 0.1})
    summary = tmp_path / "summary.json"
    write_json(summary, {"per_run": {"baseline": {"0": {}}, "full": {"0": {}}}})
    manifest = tmp_path / "manifest"
    manifest.write_text("test")
    (d / "consensus-queue/warm-fallback-declaration.md").write_text("frozen protocol")
    gates = {"warm": (warm_admit, warm_pass), "cold_a0": (cold_admit, cold_pass)}

    def candidate(
        _seed: int, name: str, _selector: dict[str, Any], _joint: dict[str, Any]
    ) -> dict[str, Any]:
        if name == "cold_a0" and warm_admit:
            assert (folder / "route.json").is_file()
        return {"method": name, "admitted": gates[name][0]}

    def endpoint(
        _seed: int, row: dict[str, Any], _tau: float, _identity: str, _ref: dict[str, Any]
    ) -> dict[str, Any]:
        route = read_json(folder / "route.json")
        assert route["selected"] == expected
        ok, passed = gates[row["method"]]
        if not ok:
            return {"passed": False, "reason": "train_admission"}
        return {
            "passed": passed,
            "reason": None if passed else "endpoint",
            "checks": {"full_mcc": passed},
            "validation": {"source": row["method"]},
            "checkpoint_sha256": {"full": "source"},
        }

    with ExitStack() as stack:
        for name, value in (
            ("PRIVATE", d),
            ("DATA", cohort),
            ("PARENT", parent),
            ("COMPARISON", summary),
        ):
            stack.enter_context(patch.object(policy.schedule, name, value))
        stack.enter_context(patch.object(policy, "MANIFEST", manifest))
        stack.enter_context(patch.object(policy, "check_head"))
        stack.enter_context(patch.object(policy.subprocess, "run"))
        stack.enter_context(
            patch.object(
                policy.trainer,
                "verify_completion",
                return_value={
                    "data_identity": "toy",
                    "seed": 0,
                    "steps": 100000,
                    "successful_updates": 100000,
                    "hashes": {"last.pt": "joint"},
                },
            )
        )
        stack.enter_context(patch.object(policy, "_candidate", side_effect=candidate))
        stack.enter_context(patch.object(policy, "_comparison", side_effect=endpoint))
        stack.enter_context(patch.object(policy, "_reported", return_value={"original": True}))
        result = policy.run("test", 0)
        assert result["selected"] == expected
        assert result["reason"] == reason
        assert result["passed"] is (reason is None)
        assert result["prior_declaration_sha256"]
        assert result["budget"]["acquisition"] == (1 if warm_admit else 2) * 100000
        assert result == policy.run("test", 0)
        if not result["passed"]:
            with pytest.raises(ValueError, match="passing same-revision"):
                policy.run("test", 1)
