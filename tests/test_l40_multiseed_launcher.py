"""Static contracts for the fixed-tau L40 eight-seed launcher."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "l40_exp1_8seed_pipeline.sbatch"


def test_l40_multiseed_launcher_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_l40_multiseed_launcher_runs_exact_paired_protocol() -> None:
    source = SCRIPT.read_text()

    assert "#SBATCH --array=0-7%2" in source
    assert 'if [ "$#" -ne 1 ]' in source
    assert 'echo "usage: sbatch $0 TAU"' in source
    assert source.count('"${PYTHON}" scripts/train.py') == 2
    assert source.count("--episodes 5000") == 2
    assert source.count("--seed-offset 29") == 2
    assert "model.spartan_dense=true" in source
    assert "model.spartan_dense=false" in source
    assert "model.spartan_identity=true" not in source
    assert '"train.sparsity_tau=${TAU}"' in source
    assert "scripts/aggregate_dense_sparse.py" in source
