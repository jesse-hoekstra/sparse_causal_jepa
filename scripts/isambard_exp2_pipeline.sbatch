#!/bin/bash
# Isambard: Experiment 2, visual states and EMA targets, with TF + sampled T=2 endpoints.
# Submit from the repository root:
#   sbatch --account=<PROJECT> scripts/isambard_exp2_pipeline.sbatch RUN_TAG LAMBDA_LOGIT [SEED] [STEPS]
# Reuse the recorded Experiment-1 physics preload. Render identical-looking balls on demand.
# Train a dense reference, reject gross collapse, calibrate the normalized visual constraint,
# then train/evaluate the sparse model. The inherited 1e-5 logit value is exploratory.
#SBATCH --job-name=scjepa-exp2
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "usage: sbatch $0 RUN_TAG LAMBDA_LOGIT [SEED] [STEPS]" >&2
  exit 2
fi
RUN_TAG=$1
LAMBDA_LOGIT=$2
SEED=${3:-0}
STEPS=${4:-300000}
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
   [[ ! "${SEED}" =~ ^[0-9]+$ ]] || [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "run tag must be a plain identifier, seed non-negative, and steps positive" >&2
  exit 2
fi
ROOT="outputs/bounce_exp2_${RUN_TAG}"
if [ -e "${ROOT}" ]; then
  echo "refusing to overwrite existing run: ${ROOT}" >&2
  exit 2
fi

source ~/miniforge3/bin/activate scjepa
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
python -c 'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and v >= 0, "lambda_logit must be finite and non-negative"' "${LAMBDA_LOGIT}"

NUM_CLIPS=100000
if [ "${SEED}" -eq 0 ]; then
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}.pt"
else
  PRELOAD="data/bounce_train_v2_${NUM_CLIPS}_seed${SEED}.pt"
fi
if [ ! -f "${PRELOAD}" ]; then
  echo "missing recorded Experiment-1 physics preload: ${PRELOAD}; copy it from the Experiment-1 server" >&2
  exit 2
fi

COMMON=(
  experiment=bounce_visual_to_visual
  wandb.enabled=true
  train.device=cuda
  train.lambda_rollout_t2=1.0
  train.num_rollout_t2_anchors=8
  train.rollout_t2_horizon=2
  train.oe_eval_horizon=30
  "train.lambda_logit=${LAMBDA_LOGIT}"
  "train.seed=${SEED}"
  "train.steps=${STEPS}"
  "data.seed=${SEED}"
  "data.num_clips=${NUM_CLIPS}"
  "data.preload=${PRELOAD}"
  data.cache=false
  data.render_radius_from_mass=false
  data.uniform_appearance=true
)

echo "== recorded physics: ${PRELOAD} =="
echo "== dense reference: TF + T=2 =="
DENSE_DIR="${ROOT}/dense"
python scripts/train.py "${COMMON[@]}" \
  model.spartan_dense=true \
  train.sparsity_enabled=false \
  "wandb.run_tag=${RUN_TAG}-dense" \
  "hydra.run.dir=${DENSE_DIR}"
python scripts/eval_visual_to_visual.py "${DENSE_DIR}" \
  --episodes 1000 --seed-offset 17 --device cuda \
  --require-complete-protocol --require-noncollapsed

TAU=$(python -c 'import json,math,sys; v=float(json.load(open(sys.argv[1]))["constraint_loss"]); assert math.isfinite(v) and v > 0, "invalid dense constraint"; print(v)' "${DENSE_DIR}/metrics.json")
echo "calibrated visual tau = ${TAU} (1.0x dense normalized held-out TF+T=2 constraint)"
echo "Tau is exploratory: different learned target spaces need not imply equal physical fidelity."

echo "== sparse run =="
MAIN_DIR="${ROOT}/main"
python scripts/train.py "${COMMON[@]}" \
  "train.sparsity_tau=${TAU}" \
  "wandb.run_tag=${RUN_TAG}-sparse" \
  "hydra.run.dir=${MAIN_DIR}"
python scripts/eval_visual_to_visual.py "${MAIN_DIR}" \
  --episodes 1000 --seed-offset 29 --device cuda \
  --require-complete-protocol --require-noncollapsed
echo "artifacts in ${ROOT}: dense/, main/, metrics.json in each"
