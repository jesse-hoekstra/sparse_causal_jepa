"""Check that phase selection respects pilot results and warm and cold controls."""

import shlex
from pathlib import Path

import pytest

from scjepa.experiments.storage import write_json

from . import consensus_queue as q
from .consensus_queries import sources
from .restart_diagnosis import hit_probability


def test_declared_order_and_sibling_replication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preparation failure stops training; queries remain available and expansion is local."""
    monkeypatch.setattr(q, "OUTPUT", tmp_path)
    monkeypatch.setattr(q, "QUEUE", tmp_path / "queue")
    first = q.pending("id")
    assert all(t.seed == 0 for t in first)
    assert q.Task("vision-invariant", 0, "sibling") not in first
    inst = q.Task("vision-invariant", 0, "prepare")
    inst.directory.mkdir(parents=True)
    write_json(
        inst.directory / "prepare.ok.json",
        {
            "commit": "id",
            "seed": 0,
            "phase": "prepare",
            "exit_code": 0,
            "command": shlex.join(inst.command("id")),
        },
    )
    pending = q.pending("id")
    assert q.Task("vision-invariant", 0, "sibling") in pending
    assert q.Task("vision-invariant", 0, "assess") not in pending
    assert inst.matches_gpu("NVIDIA RTX 3090")
    assert not inst.matches_gpu("RTX 2080 Ti")
    for method, passed in (("vision-invariant", False), ("linear-consensus-synthetic", True)):
        p = tmp_path / f"{method}-s0"
        p.mkdir(exist_ok=True)
        write_json(p / "metrics.json", {"seed": 0, "commit": "id", "passed": passed})
    after = q.pending("id")
    assert q.Task("vision-invariant", 0, "queries") in after
    assert all(t.seed == 0 for t in after if t.method == "vision-invariant")
    assert {t.seed for t in after if t.method == "linear-consensus-synthetic"} == {1, 2, 3, 4}
    assert sources("vision-invariant", tmp_path, {"passed": False}) == {}
    assert hit_probability(16, 1, 4) == 0.25


def test_readout_query_source_keeps_full_native_arm(tmp_path: Path) -> None:
    """Changed single-episode readouts retain the original sparse-arm interpretation."""
    paths = sources("linear-consensus-vision", tmp_path, {"checkpoint_sha256": "digest"})
    assert paths["sibling"] == ("full", tmp_path / "final.pt", "digest")
