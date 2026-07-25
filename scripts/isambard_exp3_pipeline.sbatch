#!/bin/bash
# Isambard-AI: the complete Experiment-3 pipeline (experiments.pdf §6.6) in one
# job. Submit FROM THE REPO ROOT:
#
#   sbatch --account=<PROJECT> scripts/isambard_exp3_pipeline.sbatch RUN_TAG LAMBDA_LOGIT [SEED] [STEPS]
#
# e.g.  sbatch --account=<PROJECT> scripts/isambard_exp3_pipeline.sbatch exp3_seed0 1e-5 0
#
# Three stages, in order:
#   1. DATA   pregenerate the states preload (idempotent; skipped if present).
#             The visual condition renders frames on the fly FROM these states,
#             so this is the same physics Experiment 1 used and there is no
#             separate frames file — 100k x 60 frames would be ~294 GB.
#   2. DENSE  A = 1 everywhere, no path penalty. Its converged held-out
#             constraint IS tau_3 (Eq. 123 units, variance-normalized).
#   3. SPARSE the gated run at tau_3, then a final held-out evaluation.
#
# lambda_logit is chosen up front rather than swept: 1e-5 is the Experiment-1
# value (D30) and is a defensible starting point because the logit penalty acts
# on SPARTAN's own attention logits, which are unchanged by the switch to a
# visual state interface. It is NOT a calibrated choice for this experiment —
# §6.1.3 asks for a fresh label-free dense sweep, and the write-up requires
# reporting one before the confirmatory seeds.
#SBATCH --job-name=scjepa-exp3
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: sbatch $0 RUN_TAG LAMBDA_LOGIT [SEED] [STEPS]" >&2
  exit 2
fi
RUN_TAG=$1
LAMBDA_LOGIT=$2
SEED=${3:-0}
STEPS=${4:-300000}
ROOT="outputs/bounce_exp3_${RUN_TAG}"
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

COMMON=(
  experiment=bounce_visual_to_visual
  wandb.enabled=true
  train.device=cuda
  "train.lambda_logit=${LAMBDA_LOGIT}"
  "train.seed=${SEED}"
  "train.steps=${STEPS}"
  "data.seed=${SEED}"
  "data.num_clips=${NUM_CLIPS}"
  "data.preload=${PRELOAD}"
)

# ---------------------------------------------------------------- 1. DATA ----
# Idempotent. NOTE the physics is identical to Experiment 1's, so if the file
# already exists it is REUSED rather than regenerated — deliberately: the
# simulator is deterministic per machine but not across machines, and
# regenerating would give Experiment 3 a different ground-truth contact graph
# from the Experiment-1 result it is meant to build on.
echo "== stage 1/3: data =="
python scripts/pregenerate_bounce.py \
  experiment=bounce_visual_to_visual \
  "data.num_clips=${NUM_CLIPS}" \
  "data.seed=${SEED}"

# --------------------------------------------------------------- 2. DENSE ----
echo "== stage 2/3: dense reference (tau_3 calibration) =="
DENSE_DIR="${ROOT}/dense"
python scripts/train.py "${COMMON[@]}" \
  model.spartan_dense=true \
  train.sparsity_enabled=false \
  "wandb.run_tag=${RUN_TAG}-dense" \
  "hydra.run.dir=${DENSE_DIR}"
python scripts/eval_visual_to_visual.py "${DENSE_DIR}" \
  --episodes 1000 --seed-offset 17 --device cuda

TAU=$(python -c "import json,sys; print(json.load(open('${DENSE_DIR}/metrics.json'))['constraint_loss'])")
echo "calibrated tau_3 = ${TAU} (1.0x the dense held-out constraint, Eq. 123 units)"

# -------------------------------------------------------------- 3. SPARSE ----
echo "== stage 3/3: sparse run at tau_3 = ${TAU} =="
MAIN_DIR="${ROOT}/main"
python scripts/train.py "${COMMON[@]}" \
  "train.sparsity_tau=${TAU}" \
  "wandb.run_tag=${RUN_TAG}-sparse" \
  "hydra.run.dir=${MAIN_DIR}"

echo "== final held-out evaluation (disjoint split) =="
python scripts/eval_visual_to_visual.py "${MAIN_DIR}" \
  --episodes 1000 --seed-offset 29 --device cuda
echo "artifacts in ${ROOT} (dense/, main/, metrics.json in each)"
