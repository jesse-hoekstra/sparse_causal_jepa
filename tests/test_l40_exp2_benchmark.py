"""Exercise the short hardware comparison without allocating GPUs or training."""

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/l40_exp2_benchmark.sbatch"
PHYSICS_SHA256 = "2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85"


def _environment(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data/bounce_train_v2_100000.pt").touch()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    python = tmp_path / ".venv/bin/python"
    python.write_text(
        textwrap.dedent("""\
            #!/usr/bin/env python3
            import json, os, pathlib, sys, types
            args = sys.argv[1:]
            with open(os.environ['CALLS'], 'a') as out:
                out.write(json.dumps(args) + '\\n')
            if args[0] == '-c':
                sys.modules['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: int(os.environ.get('VISIBLE_GPUS', '2'))))
                sys.argv = ['-c', *args[2:]]
                exec(args[1])
            elif args[0] == '-':
                sys.argv = ['-', *args[1:]]
                exec(sys.stdin.read())
            else:
                steps = next(x.split('=', 1)[1] for x in args if x.startswith('train.steps='))
                seconds = 0.8 if '--nproc_per_node=1' in args else 0.5
                skips = os.environ.get('SKIPPED_STEPS', '0')
                print(f'done at step {steps}: health/skipped_steps={skips}, '
                      f'train/seconds_per_step={seconds}')
            """)
    )
    python.chmod(0o755)
    checksum = tmp_path / ".venv/bin/sha256sum"
    checksum.write_text('#!/bin/sh\nprintf "%s  %s\\n" "$MOCK_SHA256" "$1"\n')
    checksum.chmod(0o755)
    return {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "PATH": f"{tmp_path / '.venv/bin'}:{os.environ['PATH']}",
        "CALLS": str(tmp_path / "calls.jsonl"),
        "MOCK_SHA256": PHYSICS_SHA256,
    }


def test_benchmark_compares_same_protocol_and_reports_scaling(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    result = subprocess.run(
        ["bash", str(SCRIPT), "test"],
        env=_environment(tmp_path),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    training = [call for call in calls if "scripts/train.py" in call]
    assert len(training) == 2
    for count, call in enumerate(training, start=1):
        assert f"--nproc_per_node={count}" in call
        assert "--local-addr=127.0.0.1" in call
        assert "experiment=bounce_visual_to_visual" in call
        assert "train.batch_size=4" in call
        assert f"train.num_workers={8 // count}" in call
        assert "train.steps=400" in call
        assert "model.spartan_dense=true" in call
        assert "train.sparsity_enabled=false" in call
        assert "train.lambda_rollout_t2=1.0" in call
        assert "train.num_rollout_t2_anchors=8" in call
        assert "data.uniform_appearance=true" in call
        assert "data.render_radius_from_mass=false" in call
        assert "train.eval_every=null" in call
        assert "wandb.enabled=false" in call
        assert "train.checkpoint_every=401" in call
    summary = json.loads(
        (tmp_path / "outputs/bounce_exp2_benchmark_test/benchmark.json").read_text()
    )
    assert summary["two_gpu_speedup"] == 1.6
    assert summary["timed_final_steps"] == 100
    assert summary["global_batch_size"] == 4
    assert "does not establish convergence" in result.stdout


@pytest.mark.parametrize(
    ("override", "steps", "message"),
    [
        ({"MOCK_SHA256": "0" * 64}, "400", "wrong preload checksum"),
        ({"VISIBLE_GPUS": "1"}, "400", "allocate two GPUs"),
        ({}, "100", "at least 200 steps"),
        ({}, "201", "multiple of 100"),
    ],
)
def test_benchmark_rejects_invalid_setup_before_training(
    tmp_path: Path, override: dict[str, str], steps: str, message: str
) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", steps],
        env={**_environment(tmp_path), **override},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert message in result.stderr
    assert "scripts/train.py" not in (tmp_path / "calls.jsonl").read_text()


def test_benchmark_does_not_report_rejected_updates_as_speedup(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "test"],
        env={**_environment(tmp_path), "SKIPPED_STEPS": "1"},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid 1-GPU benchmark" in result.stderr
    assert not (tmp_path / "outputs/bounce_exp2_benchmark_test/benchmark.json").exists()
