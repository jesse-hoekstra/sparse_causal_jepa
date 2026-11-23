"""Check the production fork command on a real tiny RGB cohort and exercise forensic controls."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from diagnostics.vision_dimer.train import load_checkpoint
from tests.test_vision_dimer import _sparse_pair  # pyright: ignore[reportPrivateUsage]

from . import vision_native_control as control
from .changed_queries import Inputs
from .vision_loss import compare_replay


def test_two_native_processes_from_one_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch the protected public CLI twice, resuming two actual CPU optimizer updates."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, source = _sparse_pair(tmp_path)
    finally:
        torch.set_num_threads(threads)
    monkeypatch.setattr(control, "END", 4)
    monkeypatch.setattr(control, "PARENT", source)
    evidence = Inputs()
    archived = control._reading(tmp_path / "whole/history.jsonl", evidence)  # pyright: ignore[reportPrivateUsage]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join((str(Path.cwd()), str(Path.cwd() / "src"))),
    }
    for name in ("first", "second"):
        stage = tmp_path / name
        argv = control._argv(stage)  # pyright: ignore[reportPrivateUsage]
        argv[argv.index("--wandb-mode") + 1] = "disabled"
        subprocess.run(argv, capture_output=True, text=True, check=True, env=env)
        assert json.loads((stage / "fork.json").read_text())["source_step"] == 2
        assert json.loads((stage / "final.json").read_text())["step"] == 4
        measured = control._reading(stage / "history.jsonl", evidence)  # pyright: ignore[reportPrivateUsage]
        native = compare_replay(measured, archived)
        assert set(native["fields"]) == control.FIELDS
        assert native["numerically_matched"]
    a, b = (load_checkpoint(tmp_path / name / "last.pt") for name in ("first", "second"))
    torch.testing.assert_close(a["model"], b["model"], rtol=0, atol=0)
    # The CPU trainer stores no CUDA generator. Use byte states to exercise the CUDA comparison.
    a["cuda_rng"], b["cuda_rng"] = (
        torch.tensor([3], dtype=torch.uint8),
        torch.tensor([4], dtype=torch.uint8),
    )
    report = control.snapshot_comparison(a, b)
    assert all(report["sampler"].values())
    assert report["windows"]
    assert report["cpu_rng"]
    assert not report["cuda_rng"]
    assert report["optimizer_steps"]["equal"]


def test_history_requires_all_fields_and_one_endpoint(tmp_path: Path) -> None:
    """Reject a duplicated, missing, or schema-incomplete scientific update."""
    row = dict.fromkeys(control.FIELDS, 0.5) | {"seconds_per_step": 0.1, "step": control.END}
    h = tmp_path / "history.jsonl"
    for lines in (
        [dict(row, step=1)],
        [row, row],
        [{k: v for k, v in row.items() if k != "sigreg"}],
    ):
        h.write_text("".join(json.dumps(v) + "\n" for v in lines))
        with pytest.raises(ValueError, match="exactly one complete"):
            control._reading(h, Inputs())  # pyright: ignore[reportPrivateUsage]
