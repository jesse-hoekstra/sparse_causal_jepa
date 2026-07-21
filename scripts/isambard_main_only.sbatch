#!/bin/bash
# Isambard-AI: MAIN sparse run only — for when tau was calibrated under the
# exact current architecture and verified below a matched identity reference.
# D24 changed the simulator, parameter bottleneck, prediction-loss
# units, and path objective, so pre-D24 values such as 0.0839 are invalid and
# deliberately have no default here.
#
# Submit FROM THE REPO ROOT:
#   sbatch scripts/isambard_main_only.sbatch RUN_TAG TAU LAMBDA_LOGIT [STEPS] [SEED]
# Tau and lambda_logit are mandatory. Prefer scripts/isambard_pipeline.sbatch.
#
# PROVENANCE NOTE: record the sibling calibration run and factor for every tau.
#SBATCH --job-name=scjepa-main
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "usage: sbatch $0 RUN_TAG CALIBRATED_TAU LAMBDA_LOGIT [STEPS] [SEED]" >&2
  echo "prefer scripts/isambard_pipeline.sbatch so tau is recalibrated in-pipeline" >&2
  exit 2
fi
RUN_TAG=$1
TAU=$2
LAMBDA_LOGIT=$3
STEPS=${4:-300000}
SEED=${5:-0}
OUT_DIR="outputs/bounce_example_${RUN_TAG}/main"
if [ -e "${OUT_DIR}" ]; then
  echo "refusing to overwrite existing run: ${OUT_DIR}" >&2
  exit 2
fi

source ~/miniforge3/bin/activate scjepa
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
python -c 'import math,sys; values=[float(x) for x in sys.argv[1:]]; assert all(math.isfinite(x) and x >= 0 for x in values), "tau and lambda_logit must be finite and non-negative"' "${TAU}" "${LAMBDA_LOGIT}"
python -c 'import sys; value=int(sys.argv[1]); assert value >= 0, "seed must be non-negative"' "${SEED}"
echo "main-only run: tag=${RUN_TAG} tau=${TAU} lambda_logit=${LAMBDA_LOGIT} steps=${STEPS} -> ${OUT_DIR}"

if [ "${SEED}" -eq 0 ]; then
  PRELOAD=data/bounce_train_v2_100000.pt
else
  PRELOAD="data/bounce_train_v2_100000_seed${SEED}.pt"
fi

python scripts/train.py \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda \
  "train.sparsity_tau=${TAU}" \
  train.sparsity_warmup_steps=0 \
  "train.lambda_logit=${LAMBDA_LOGIT}" \
  "train.steps=${STEPS}" \
  "train.seed=${SEED}" \
  "data.seed=${SEED}" \
  "data.preload=${PRELOAD}" \
  "wandb.run_tag=${RUN_TAG}" \
  "hydra.run.dir=${OUT_DIR}"

echo "== final identifiability evaluation =="
python scripts/eval_identifiability.py "${OUT_DIR}" \
  --episodes 5000 --seed-offset 29 --device cuda
echo "artifacts in ${OUT_DIR} (checkpoint, resolved config, recovery_grid.png)"
