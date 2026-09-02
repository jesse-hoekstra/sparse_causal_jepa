#!/bin/bash
# Portable Slurm launcher for the fixed T=2 Experiment-1 pipeline on the shared
# multi-GPU x86_64 server (L40/RTX Ada), requesting exactly ONE GPU.
#
# Submit FROM THE REPOSITORY ROOT. Account and partition names are deliberately
# not hard-coded because they are site-specific:
#
#   sbatch scripts/l40_exp1_pipeline.sbatch t2_seed0_l40 1e-5 0
#
# If the site requires them, add them on the command line:
#
#   sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> \
#     scripts/l40_exp1_pipeline.sbatch t2_seed0_l40 1e-5 0
#
# This is not the Isambard wrapper: it uses the repository's x86_64 `.venv`,
# never regenerates the cross-machine-sensitive physics preload, and allows a
# multi-day runtime. Command-line sbatch options can override the resources.
# The pipeline remains single-device; requesting a second GPU would not speed it
# up and would unnecessarily consume the shared server's two-GPU allowance.
#SBATCH --job-name=scjepa-t2-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: sbatch $0 RUN_TAG LAMBDA_LOGIT [SEED]" >&2
  exit 2
fi

RUN_TAG=$1
LAMBDA_LOGIT=$2
SEED=${3:-0}

# Slurm normally starts a job in its submission directory. Make that contract
# explicit so relative data/output paths cannot silently target another clone.
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "${REPO_ROOT}"
if [ ! -f pyproject.toml ] || [ ! -f scripts/run_bounce_example.sh ]; then
  echo "submit this script from the sparse_causal_jepa repository root" >&2
  exit 2
fi

if [ -e "outputs/bounce_example_${RUN_TAG}" ]; then
  echo "refusing to overwrite existing run: outputs/bounce_example_${RUN_TAG}" >&2
  exit 2
fi

if [ ! -x .venv/bin/python ]; then
  echo "missing .venv/bin/python; create the Python 3.12 environment first" >&2
  exit 2
fi
export PYTHON="${REPO_ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1

"${PYTHON}" -c \
  'import math,sys; value=float(sys.argv[1]); assert math.isfinite(value) and value >= 0, "lambda_logit must be finite and non-negative"' \
  "${LAMBDA_LOGIT}"
"${PYTHON}" -c \
  'import sys; value=int(sys.argv[1]); assert value >= 0, "seed must be non-negative"' \
  "${SEED}"
"${PYTHON}" -c \
  'import torch; assert torch.cuda.is_available(), "no CUDA GPU allocated"; print(f"GPU: {torch.cuda.get_device_name(0)}; visible GPUs: {torch.cuda.device_count()}")'

NUM_CLIPS=100000
if [ "${SEED}" -eq 0 ]; then
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}.pt"
else
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}_seed${SEED}.pt"
fi
if [ ! -f "${PRELOAD}" ]; then
  echo "missing required physics preload: ${PRELOAD}" >&2
  echo "copy the existing preload to this server; do NOT regenerate it" >&2
  exit 2
fi

# The seed-0 preload is the paper's physics of record. It is not reproducible
# by regenerating on another machine, so reject even a subtly different file.
if [ "${SEED}" -eq 0 ]; then
  EXPECTED_SHA256=2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85
  ACTUAL_SHA256=$(sha256sum "${PRELOAD}" | awk '{print $1}')
  if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "wrong preload checksum for ${PRELOAD}" >&2
    echo "expected ${EXPECTED_SHA256}" >&2
    echo "actual   ${ACTUAL_SHA256}" >&2
    exit 2
  fi
fi

echo "host=$(hostname) job_id=${SLURM_JOB_ID:-none} run_tag=${RUN_TAG} seed=${SEED}"
echo "preload=${PRELOAD} sha256=$(sha256sum "${PRELOAD}" | awk '{print $1}')"

bash scripts/run_bounce_example.sh \
  "--run-tag=${RUN_TAG}" \
  --main-steps=300000 \
  --eval-episodes=5000 \
  --final-seed-offset=29 \
  --eval-device=cuda \
  --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda \
  "train.lambda_logit=${LAMBDA_LOGIT}" \
  "train.seed=${SEED}" \
  "data.seed=${SEED}" \
  "wandb.run_tag=${RUN_TAG}" \
  "data.num_clips=${NUM_CLIPS}" \
  "data.preload=${PRELOAD}"
