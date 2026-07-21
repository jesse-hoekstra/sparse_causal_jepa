#!/bin/bash
# Isambard-AI multi-GPU dense sweep for Baumgartner's unspecified lambda_logit.
# Submit from the repository root, optionally overriding the colon-separated
# grid or the usual 300k-step/full-eval defaults:
#
#   sbatch --account=<PROJECT> scripts/isambard_logit_sweep.sbatch logit_seed0 0
#   LAMBDA_LOGIT_GRID='0:3e-5:1e-4:3e-4:1e-3:3e-3:1e-2:3e-2' \
#     sbatch --account=<PROJECT> scripts/isambard_logit_sweep.sbatch logit_seed0 0
#
# Eight one-GPU srun steps execute concurrently across two complete Isambard-AI
# nodes; additional grid points wait for a GPU to become free. Override
# --nodes/--gres/--ntasks together if using a different allocation shape.
#SBATCH --job-name=scjepa-logit-sweep
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
RUN_TAG=${1:-seed0_${SLURM_JOB_ID:-local}}
SEED=${2:-0}
case "${SEED}" in
  ''|*[!0-9]*) echo "SEED must be a non-negative integer, got: ${SEED}" >&2; exit 2 ;;
esac
GRID=${LAMBDA_LOGIT_GRID:-0:3e-5:1e-4:3e-4:1e-3:3e-3:1e-2:3e-2}
STEPS=${SWEEP_STEPS:-300000}
EVAL_EPISODES=${EVAL_EPISODES:-5000}
VALIDATION_SEED_OFFSET=${VALIDATION_SEED_OFFSET:-23}
NUM_CLIPS=${NUM_CLIPS:-100000}
if [ "${SEED}" -eq 0 ]; then
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}.pt"
else
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}_seed${SEED}.pt"
fi
BASE="outputs/lambda_logit_sweep_${RUN_TAG}"
if [ -e "${BASE}" ]; then
  echo "refusing to mix a new sweep with existing directory: ${BASE}" >&2
  echo "choose a new RUN_TAG or move the old directory" >&2
  exit 2
fi
mkdir -p "${BASE}"

source ~/miniforge3/bin/activate scjepa
export PYTHON=python
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
python -c 'import math,sys; values=[float(x) for x in sys.argv[1].split(":")]; assert len(values) >= 2, "need at least two lambda values"; assert all(math.isfinite(x) and x >= 0 for x in values), "lambda values must be finite and non-negative"; assert len(values) == len(set(values)), "lambda values must be unique"; assert 0.0 in values, "grid must include the unregularized lambda=0 control"' "${GRID}"

# Generate/validate the shared dataset once before concurrent readers start.
# ProcessPoolExecutor runs on the batch node only, so expose the requested CPUs
# on ONE node rather than incorrectly summing CPUs across the multi-node job.
ALLOCATED_NODES=${SLURM_JOB_NUM_NODES:-2}
ALLOCATED_TASKS=${SLURM_NTASKS:-8}
TASKS_PER_NODE=$(( (ALLOCATED_TASKS + ALLOCATED_NODES - 1) / ALLOCATED_NODES ))
PREGEN_CPUS=$(( TASKS_PER_NODE * ${SLURM_CPUS_PER_TASK:-16} ))
env SLURM_CPUS_PER_TASK="${PREGEN_CPUS}" \
  python scripts/pregenerate_bounce.py \
  experiment=bounce_baumgartner \
  "data.num_clips=${NUM_CLIPS}" \
  "data.seed=${SEED}"

IFS=':' read -r -a LAMBDAS <<< "${GRID}"
if [ "${#LAMBDAS[@]}" -lt 2 ]; then
  echo "need at least two colon-separated values in LAMBDA_LOGIT_GRID, got: ${GRID}" >&2
  exit 2
fi

PIDS=()
RUN_DIRS=()
for VALUE in "${LAMBDAS[@]}"; do
  LABEL=${VALUE//./p}
  LABEL=${LABEL//-/m}
  LABEL=${LABEL//+/p}
  RUN_DIR="${BASE}/lambda_${LABEL}/dense"
  RUN_DIRS+=("${RUN_DIR}")
  mkdir -p "${RUN_DIR}"
  echo "launching lambda_logit=${VALUE} -> ${RUN_DIR}"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --gres=gpu:1 --output="${RUN_DIR}/slurm.out" --error="${RUN_DIR}/slurm.err" \
    bash scripts/run_dense_logit_value.sh \
      "${VALUE}" "${RUN_DIR}" "${STEPS}" "${NUM_CLIPS}" "${PRELOAD}" \
      "${SEED}" "${RUN_TAG}" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then STATUS=1; fi
done
if [ "${STATUS}" -ne 0 ]; then
  echo "one or more sweep runs failed; inspect ${BASE}/lambda_*/dense/slurm.err" >&2
  exit "${STATUS}"
fi

# Keep all GPU slots available for training first. Once every coefficient has
# a checkpoint, evaluate the held-out validation split in parallel.
PIDS=()
for RUN_DIR in "${RUN_DIRS[@]}"; do
  echo "evaluating validation split -> ${RUN_DIR}"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --gres=gpu:1 --output="${RUN_DIR}/eval-slurm.out" \
    --error="${RUN_DIR}/eval-slurm.err" \
    python scripts/eval_identifiability.py "${RUN_DIR}" \
      --episodes "${EVAL_EPISODES}" \
      --seed-offset "${VALIDATION_SEED_OFFSET}" \
      --device cuda &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then STATUS=1; fi
done
if [ "${STATUS}" -ne 0 ]; then
  echo "one or more sweep evaluations failed; inspect ${BASE}/lambda_*/dense/eval-slurm.err" >&2
  exit "${STATUS}"
fi

python scripts/summarize_logit_sweep.py "${BASE}"
SELECTED=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_lambda_logit"])' "${BASE}/sweep_summary.json")
echo "dense lambda_logit sweep complete: ${BASE}/sweep_summary.json"
echo "selected lambda_logit=${SELECTED}"
echo "next: sbatch --account=<PROJECT> scripts/isambard_pipeline.sbatch <run_tag> ${SELECTED} ${SEED}"
