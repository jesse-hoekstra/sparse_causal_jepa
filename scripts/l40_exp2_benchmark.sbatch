#!/bin/bash
# Compare dense visual training throughput on one and two L40s after warm-up.
# sbatch scripts/l40_exp2_benchmark.sbatch speed_check [STEPS=400]
# Same global batch four and recorded seed-zero physics as the full pipeline.
# This short run measures throughput, not convergence or sparse-stage runtime.
#SBATCH --job-name=scjepa-exp2-speed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: sbatch $0 RUN_TAG [STEPS=400; multiple of 100, at least 200]" >&2
  exit 2
fi
RUN_TAG=$1
STEPS=${2:-400}
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PYTHON="${PWD}/.venv/bin/python"
ROOT="outputs/bounce_exp2_benchmark_${RUN_TAG}"
PRELOAD="data/bounce_train_v2_100000.pt"
if [ ! -x "${PYTHON}" ] || [ ! -f scripts/train.py ]; then
  echo "submit from the repository root with .venv/bin/python available" >&2
  exit 2
fi
if [ -e "${ROOT}" ]; then
  echo "refusing to overwrite existing benchmark: ${ROOT}" >&2
  exit 2
fi
"${PYTHON}" -c \
  'import sys,torch; n=int(sys.argv[1]); assert n >= 200, "use at least 200 steps to exclude warmup"; assert n % 100 == 0, "steps must be a multiple of 100 for a complete timed interval"; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "allocate two GPUs for the comparison"' \
  "${STEPS}"
if [ ! -f "${PRELOAD}" ]; then
  echo "missing ${PRELOAD}; copy the recorded preload, do NOT regenerate it" >&2
  exit 2
fi
EXPECTED_SHA256=2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85
if [ "$(sha256sum "${PRELOAD}" | awk '{print $1}')" != "${EXPECTED_SHA256}" ]; then
  echo "wrong preload checksum for ${PRELOAD}" >&2
  exit 2
fi
export PYTHONUNBUFFERED=1
# Keep CPU threading identical across the one- and two-process launches.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "${ROOT}"
for NUM_GPUS in 1 2; do
  echo "Timing ${NUM_GPUS} GPU(s), global batch four, ${STEPS} updates"
  "${PYTHON}" -m torch.distributed.run --standalone "--nproc_per_node=${NUM_GPUS}" \
    --local-addr=127.0.0.1 scripts/train.py \
    experiment=bounce_visual_to_visual wandb.enabled=false \
    model.spartan_dense=true train.sparsity_enabled=false \
    train.device=cuda train.batch_size=4 train.seed=0 \
    "train.steps=${STEPS}" "train.num_workers=$((8 / NUM_GPUS))" \
    train.log_every=100 train.eval_every=null \
    "train.checkpoint_every=$((STEPS + 1))" train.checkpoint_keep_every=null \
    train.lambda_logit=1e-5 train.lambda_rollout_t2=1.0 \
    train.num_rollout_t2_anchors=8 train.rollout_t2_horizon=2 \
    data.seed=0 "data.preload=${PRELOAD}" data.cache=false \
    data.uniform_appearance=true data.render_radius_from_mass=false \
    hydra.output_subdir=null hydra/job_logging=disabled \
    "hydra.run.dir=${ROOT}/gpu${NUM_GPUS}" | tee "${ROOT}/gpu${NUM_GPUS}.log"
done

"${PYTHON}" - "${ROOT}" "${STEPS}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
seconds = {}
for count in (1, 2):
    text = (root / f"gpu{count}.log").read_text()
    rows = [line for line in text.splitlines() if line.startswith(f"done at step {steps}:")]
    if len(rows) != 1:
        raise SystemExit(f"missing completed {count}-GPU run")
    metrics = dict(re.findall(r"([\w/]+)=([^,\s]+)", rows[0]))
    skips = float(metrics["health/skipped_steps"])
    elapsed = float(metrics["train/seconds_per_step"])
    if skips != 0 or not math.isfinite(elapsed) or elapsed <= 0:
        raise SystemExit(f"invalid {count}-GPU benchmark: skips={skips}, seconds/step={elapsed}")
    seconds[str(count)] = elapsed
result = {
    "phase": "dense",
    "global_batch_size": 4,
    "steps": steps,
    "timed_final_steps": 100,
    "seconds_per_step": seconds,
    "two_gpu_speedup": seconds["1"] / seconds["2"],
}
(root / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"1 L40: {seconds['1']:.4g} s/step; 2 L40s: {seconds['2']:.4g} s/step")
print(f"Two-GPU speedup: {result['two_gpu_speedup']:.3g}x (final interval after warm-up)")
print(f"Report: {root / 'benchmark.json'}; short dense timing does not establish convergence.")
PY
