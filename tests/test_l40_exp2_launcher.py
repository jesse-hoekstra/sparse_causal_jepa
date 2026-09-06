"""L40 visual pipeline protocol and preservation of the recorded physics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "l40_exp2_pipeline.sbatch"
PHYSICS_SHA256 = "2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85"


def _environment(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/eval_visual_to_visual.py").touch()
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    python = tmp_path / ".venv/bin/python"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,pathlib,sys,types\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['CALLS'], 'a') as out: out.write(json.dumps(args)+'\\n')\n"
        "if args[0]=='-c':\n"
        "    if 'constraint_loss' in args[1]: print(0.25)\n"
        "    else:\n"
        "        visible_count=int(os.environ.get('MOCK_VISIBLE_GPUS', '4'))\n"
        "        sys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(\n"
        "            is_available=lambda:visible_count>0, device_count=lambda:visible_count,\n"
        "            get_device_name=lambda index:'Mock L40'))\n"
        "        sys.argv=['-c', *args[2:]]\n"
        "        exec(args[1])\n"
        "elif 'scripts/train.py' in args:\n"
        "    directory=next(x.split('=',1)[1] for x in args if x.startswith('hydra.run.dir='))\n"
        "    pathlib.Path(directory).mkdir(parents=True)\n"
        "elif args[0]=='scripts/eval_visual_to_visual.py':\n"
        "    if args[1].endswith('/dense') and os.environ.get('DENSE_COLLAPSED')=='1':\n"
        "        print('dense target variance is below the screening threshold', file=sys.stderr)\n"
        "        sys.exit(23)\n"
    )
    python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(SLURM_SUBMIT_DIR=str(tmp_path), CALLS=str(tmp_path / "calls.jsonl"))
    return environment


@pytest.mark.parametrize("gpu_argument", [None, "1", "2", "4"])
def test_l40_exp2_runs_matched_dense_sparse_objectives_with_separate_eval_splits(
    tmp_path: Path,
    gpu_argument: str | None,
) -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    environment = _environment(tmp_path)
    (tmp_path / "data").mkdir()
    # Nonzero seed exercises preload selection without substituting the seed-0 checksum.
    (tmp_path / "data/bounce_train_v2_100000_seed3.pt").touch()
    subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500"]
        + ([] if gpu_argument is None else [gpu_argument]),
        env=environment,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    train = [call for call in calls if "scripts/train.py" in call]
    assert len(train) == 2
    for call in train:
        gpu_count = int(gpu_argument or "2")
        assert call[:4] == [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={gpu_count}",
        ]
        assert "train.batch_size=4" in call
        assert "--local-addr=127.0.0.1" in call
        assert f"train.num_workers={8 // gpu_count}" in call
        assert "hydra.output_subdir=null" in call
        assert "hydra/job_logging=disabled" in call
        assert "experiment=bounce_visual_to_visual" in call
        assert "train.steps=1500" in call
        assert "train.lambda_rollout_t2=1.0" in call
        assert "train.num_rollout_t2_anchors=8" in call
        assert "train.rollout_t2_horizon=2" in call
        assert "train.oe_eval_horizon=30" in call
        assert "data.preload=data/bounce_train_v2_100000_seed3.pt" in call
        assert "data.cache=false" in call
        assert "data.uniform_appearance=true" in call
        assert "data.render_radius_from_mass=false" in call
    assert "model.spartan_dense=true" in train[0]
    assert "train.sparsity_enabled=false" in train[0]
    assert "model.spartan_dense=false" in train[1]
    assert "train.sparsity_tau=0.25" in train[1]
    evaluations = [call for call in calls if call[0] == "scripts/eval_visual_to_visual.py"]
    assert [call[call.index("--seed-offset") + 1] for call in evaluations] == ["17", "29"]
    assert all("--require-complete-protocol" in call for call in evaluations)
    assert all("--require-noncollapsed" in call for call in evaluations)
    # Read the real argparse help so the shell cannot silently grow flags that
    # only this subprocess stub accepts.
    help_text = subprocess.run(
        [sys.executable, str(SCRIPT.parent / "eval_visual_to_visual.py"), "--help"],
        cwd=SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for call in evaluations:
        for option in (argument for argument in call if argument.startswith("--")):
            assert option in help_text
    assert "pregenerate_bounce.py" not in SCRIPT.read_text()


def test_l40_exp2_refuses_missing_physics_before_training(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500"],
        env=_environment(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "do NOT regenerate" in result.stderr
    assert "scripts/train.py" not in (tmp_path / "calls.jsonl").read_text()


@pytest.mark.parametrize("checksum", [PHYSICS_SHA256, "0" * 64])
def test_l40_exp2_requires_recorded_seed_zero_checksum(tmp_path: Path, checksum: str) -> None:
    environment = _environment(tmp_path)
    (tmp_path / "data").mkdir()
    preload = tmp_path / "data/bounce_train_v2_100000.pt"
    preload.write_bytes(b"recorded physics placeholder")
    checksum_bin = tmp_path / "checksum_bin"
    checksum_bin.mkdir()
    checksum_program = checksum_bin / "sha256sum"
    checksum_program.write_text('#!/bin/sh\nprintf "%s  %s\\n" "$MOCK_SHA256" "$1"\n')
    checksum_program.chmod(0o755)
    environment.update(PATH=f"{checksum_bin}:{environment['PATH']}", MOCK_SHA256=checksum)

    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "0", "1500"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    training = [call for call in calls if "scripts/train.py" in call]
    if checksum == PHYSICS_SHA256:
        assert result.returncode == 0, result.stderr
        assert len(training) == 2
        assert all("data.preload=data/bounce_train_v2_100000.pt" in call for call in training)
    else:
        assert result.returncode == 2
        assert "wrong preload checksum" in result.stderr
        assert training == []
    assert preload.read_bytes() == b"recorded physics placeholder"


def test_l40_exp2_dense_collapse_stops_before_tau_or_sparse_training(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["DENSE_COLLAPSED"] = "1"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/bounce_train_v2_100000_seed3.pt").touch()
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    training = [call for call in calls if "scripts/train.py" in call]
    evaluations = [call for call in calls if call[0] == "scripts/eval_visual_to_visual.py"]
    assert result.returncode == 23
    assert "dense target variance" in result.stderr
    assert len(training) == 1
    assert "model.spartan_dense=true" in training[0]
    assert len(evaluations) == 1
    assert "--require-noncollapsed" in evaluations[0]
    assert all("constraint_loss" not in call[1] for call in calls if call[0] == "-c")
    assert not (tmp_path / "outputs/bounce_exp2_test/main").exists()


def test_l40_exp2_preserves_existing_run_directory(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    previous_run = tmp_path / "outputs/bounce_exp2_test"
    previous_run.mkdir(parents=True)
    checkpoint = previous_run / "previous-checkpoint"
    checkpoint.write_bytes(b"keep this run")
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    assert checkpoint.read_bytes() == b"keep this run"
    assert not (tmp_path / "calls.jsonl").exists()


@pytest.mark.parametrize("gpu_argument", ["0", "-1", "3", "8", "two"])
def test_l40_exp2_rejects_invalid_gpu_counts(tmp_path: Path, gpu_argument: str) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500", gpu_argument],
        env=_environment(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "GPUS must be 1, 2, or 4" in result.stderr
    assert not (tmp_path / "calls.jsonl").exists()


def test_l40_exp2_rejects_insufficient_visible_gpus(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["MOCK_VISIBLE_GPUS"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT), "test", "1e-5", "3", "1500"],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requested 2 workers but only 1 CUDA GPUs visible" in result.stderr
    calls = [json.loads(row) for row in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert all("scripts/train.py" not in call for call in calls)
