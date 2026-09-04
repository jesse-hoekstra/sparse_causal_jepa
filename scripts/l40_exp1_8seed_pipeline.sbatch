#!/usr/bin/env bash
# Eight-seed, fixed-tau Experiment-1 confirmation on the shared L40 server.
#
# Submit from the repository root and provide ONLY the already selected tau:
#
#   sbatch scripts/l40_exp1_8seed_pipeline.sbatch 0.05330275518342852
#
# Seeds 0..7 are dispatched as an array, with at most two concurrent one-GPU
# tasks to respect the server's two-GPU allowance.  Every task trains and
# terminally evaluates a matched dense/sparse pair.  The final successful task
# validates all 16 reports and creates the paired box plots automatically.
#
# Seed 0 uses the canonical preload and its recorded checksum.  Missing
# seed-specific preloads for seeds 1..7 are generated once on the L40 before
# their corresponding pair is trained.
#SBATCH --job-name=scjepa-exp1-8seed
#SBATCH --array=0-7%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: sbatch $0 TAU" >&2
  exit 2
fi

TAU=$1
SEED=${SLURM_ARRAY_TASK_ID:?this file must be submitted as its 0-7 Slurm array}
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?missing Slurm job id}}

REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "${REPO_ROOT}"
if [ ! -f pyproject.toml ] || [ ! -f scripts/eval_identifiability.py ]; then
  echo "submit this script from the sparse_causal_jepa repository root" >&2
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  echo "missing .venv/bin/python; create the Python 3.12 environment first" >&2
  exit 2
fi

export PYTHON="${REPO_ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

"${PYTHON}" -c \
  'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value > 0, "tau must be finite and positive"' \
  "${TAU}"
"${PYTHON}" -c \
  'import sys; value=int(sys.argv[1]); assert 0 <= value <= 7, "array seed must be in 0..7"' \
  "${SEED}"
"${PYTHON}" -c \
  'import torch; assert torch.cuda.is_available(), "no CUDA GPU allocated"; print(f"GPU: {torch.cuda.get_device_name(0)}; visible GPUs: {torch.cuda.device_count()}")'

BASE="outputs/l40_exp1_8seed_job${ARRAY_JOB_ID}"
SEED_ROOT="${BASE}/seed${SEED}"
DENSE_DIR="${SEED_ROOT}/dense"
SPARSE_DIR="${SEED_ROOT}/sparse"
if [ -e "${DENSE_DIR}" ] || [ -e "${SPARSE_DIR}" ]; then
  echo "refusing to overwrite an existing seed pair under ${SEED_ROOT}" >&2
  exit 2
fi
mkdir -p "${SEED_ROOT}" "${BASE}/.completed"

# Keep a copy beside the checkpoints as well as Slurm's submission-directory
# log, so copying the output root preserves the full per-seed transcript.
exec > >(tee -a "${SEED_ROOT}/pipeline.log") 2>&1

echo "host=$(hostname) array_job_id=${ARRAY_JOB_ID} task_job_id=${SLURM_JOB_ID} seed=${SEED}"
echo "tau=${TAU} output_root=${BASE}"

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: the worktree is dirty; resolved configs will record a -dirty git SHA" >&2
fi

NUM_CLIPS=100000
if [ "${SEED}" -eq 0 ]; then
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}.pt"
else
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}_seed${SEED}.pt"
fi

if [ ! -f "${PRELOAD}" ]; then
  if [ "${SEED}" -eq 0 ]; then
    echo "missing canonical seed-0 physics preload: ${PRELOAD}" >&2
    echo "copy the canonical preload to this server before submitting" >&2
    exit 2
  fi
  echo "${PRELOAD} is absent; generating seed ${SEED} on this L40 server"
  "${PYTHON}" scripts/pregenerate_bounce.py \
    experiment=bounce_baumgartner \
    "data.seed=${SEED}" \
    "train.seed=${SEED}" \
    "data.num_clips=${NUM_CLIPS}"
fi

if [ "${SEED}" -eq 0 ]; then
  EXPECTED_SHA256=2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85
  ACTUAL_SHA256=$(sha256sum "${PRELOAD}" | awk '{print $1}')
  if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "wrong preload checksum for ${PRELOAD}" >&2
    echo "expected ${EXPECTED_SHA256}" >&2
    echo "actual   ${ACTUAL_SHA256}" >&2
    exit 2
  fi
else
  ACTUAL_SHA256=$(sha256sum "${PRELOAD}" | awk '{print $1}')
fi
echo "preload=${PRELOAD} sha256=${ACTUAL_SHA256}"

RUN_TAG="l40_exp1_8seed_job${ARRAY_JOB_ID}_seed${SEED}"
COMMON_ARGS=(
  experiment=bounce_baumgartner
  wandb.enabled=true
  train.device=cuda
  train.steps=300000
  train.lambda_logit=1e-5
  "train.seed=${SEED}"
  "data.seed=${SEED}"
  "wandb.run_tag=${RUN_TAG}"
  "data.num_clips=${NUM_CLIPS}"
  "data.preload=${PRELOAD}"
)

echo "== seed ${SEED}, step 1/4: dense training (A=1) =="
"${PYTHON}" scripts/train.py \
  "${COMMON_ARGS[@]}" \
  "hydra.run.dir=${DENSE_DIR}" \
  model.spartan_dense=true \
  model.spartan_identity=false \
  train.sparsity_enabled=false

echo "== seed ${SEED}, step 2/4: dense final test =="
"${PYTHON}" scripts/eval_identifiability.py "${DENSE_DIR}" \
  --episodes 5000 \
  --seed-offset 29 \
  --device cuda \
  --require-complete-protocol

echo "== seed ${SEED}, step 3/4: sparse training (tau=${TAU}) =="
"${PYTHON}" scripts/train.py \
  "${COMMON_ARGS[@]}" \
  "hydra.run.dir=${SPARSE_DIR}" \
  model.spartan_dense=false \
  model.spartan_identity=false \
  train.sparsity_enabled=true \
  "train.sparsity_tau=${TAU}"

echo "== seed ${SEED}, step 4/4: sparse final test =="
"${PYTHON}" scripts/eval_identifiability.py "${SPARSE_DIR}" \
  --episodes 5000 \
  --seed-offset 29 \
  --device cuda \
  --require-complete-protocol

"${PYTHON}" - \
  "${SEED_ROOT}/completed.json" "${SEED}" "${TAU}" "${ARRAY_JOB_ID}" \
  "${PRELOAD}" "${ACTUAL_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

path, seed, tau, job_id, preload, preload_sha256 = sys.argv[1:]
record = {
    "seed": int(seed),
    "tau": float(tau),
    "slurm_array_job_id": job_id,
    "preload": preload,
    "preload_sha256": preload_sha256,
}
Path(path).write_text(json.dumps(record, indent=2) + "\n")
PY
touch "${BASE}/.completed/seed${SEED}"
echo "seed ${SEED} pair complete: ${SEED_ROOT}"

# Whichever successful task observes all eight completion markers becomes the
# sole aggregator.  mkdir is an atomic cross-process lock on the shared output
# filesystem, so simultaneous final tasks cannot create duplicate plots.
ALL_COMPLETE=true
for EXPECTED_SEED in {0..7}; do
  if [ ! -f "${BASE}/.completed/seed${EXPECTED_SEED}" ]; then
    ALL_COMPLETE=false
    break
  fi
done

if [ "${ALL_COMPLETE}" = true ]; then
  LOCK_DIR="${BASE}/.aggregation.lock"
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    remove_lock() {
      rmdir "${LOCK_DIR}" 2>/dev/null || true
    }
    trap remove_lock EXIT
    echo "all eight seed pairs complete; validating and plotting"
    "${PYTHON}" scripts/aggregate_dense_sparse.py "${BASE}" \
      --expected-tau "${TAU}" \
      --output-dir "${BASE}/aggregate"
    touch "${BASE}/AGGREGATION_COMPLETE"
    remove_lock
    trap - EXIT
    echo "final aggregate artifacts: ${BASE}/aggregate"
  else
    echo "another array task is creating the aggregate artifacts"
  fi
else
  echo "other array seeds are still running; aggregation will start after the last succeeds"
fi
