"""Static contracts for the fixed-tau L40 eight-seed launcher."""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "l40_exp1_8seed_pipeline.sbatch"


def test_l40_multiseed_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_l40_multiseed_launcher_runs_exact_paired_protocol() -> None:
    source = SCRIPT.read_text()

    assert "#SBATCH --array=0-7%2" in source
    assert 'if [ "$#" -ne 1 ]' in source
    assert "CUDA_VISIBLE_DEVICES=GPU_A,GPU_B bash $0 TAU" in source
    assert 'run_lane "${GPU_A}" 0 2 4 6 &' in source
    assert 'run_lane "${GPU_B}" 1 3 5 7 &' in source
    assert 'CUDA_VISIBLE_DEVICES="${lane_gpu}"' in source
    assert source.count('"${PYTHON}" scripts/train.py') == 2
    assert source.count("--episodes 5000") == 2
    assert source.count("--seed-offset 29") == 2
    assert "model.spartan_dense=true" in source
    assert "model.spartan_dense=false" in source
    assert "model.spartan_identity=true" not in source
    assert '"train.sparsity_tau=${TAU}"' in source
    assert "scripts/aggregate_dense_sparse.py" in source


def test_direct_mode_assigns_even_and_odd_seeds_to_separate_gpus(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    call_dir = tmp_path / "calls"
    fake_bin.mkdir()
    call_dir.mkdir()
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        'printf "%s|%s|%s|%s|%s\\n" "$CUDA_VISIBLE_DEVICES" '
        '"$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_JOB_ID" '
        '"$SLURM_CPUS_PER_TASK" "$2" > "$CALL_DIR/$SLURM_ARRAY_TASK_ID"\n'
    )
    fake_bash.chmod(0o755)

    environment = os.environ.copy()
    for key in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
        environment.pop(key, None)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "CUDA_VISIBLE_DEVICES": "4,5",
            "SCJEPA_BATCH_ID": "test_batch",
            "CALL_DIR": str(call_dir),
        }
    )
    subprocess.run(
        ["/bin/bash", str(SCRIPT), "0.05"],
        cwd=SCRIPT.parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    for seed in range(8):
        gpu, recorded_seed, batch_id, cpus, tau = (call_dir / str(seed)).read_text().split("|")
        assert gpu == ("4" if seed % 2 == 0 else "5")
        assert recorded_seed == str(seed)
        assert batch_id == "test_batch"
        assert cpus == "16"
        assert tau.strip() == "0.05"
