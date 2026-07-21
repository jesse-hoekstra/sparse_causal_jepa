#!/usr/bin/env bash
# Train and evaluate one fully-connected bounce reference for a logit weight.
# This helper is launched once per GPU by isambard_logit_sweep.sbatch.
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 LAMBDA_LOGIT OUT_DIR STEPS NUM_CLIPS PRELOAD SEED RUN_TAG" >&2
  exit 2
fi

LAMBDA_LOGIT=$1
OUT_DIR=$2
STEPS=$3
NUM_CLIPS=$4
PRELOAD=$5
SEED=$6
RUN_TAG=$7
PY=${PYTHON:-python}

mkdir -p "${OUT_DIR}"
echo "dense logit run: lambda_logit=${LAMBDA_LOGIT} steps=${STEPS} -> ${OUT_DIR}"

"${PY}" scripts/train.py \
  experiment=bounce_baumgartner \
  "hydra.run.dir=${OUT_DIR}" \
  model.spartan_dense=true \
  model.spartan_identity=false \
  train.sparsity_enabled=false \
  train.sparsity_warmup_steps=0 \
  "train.lambda_logit=${LAMBDA_LOGIT}" \
  "train.steps=${STEPS}" \
  "train.seed=${SEED}" \
  "data.seed=${SEED}" \
  train.device=cuda \
  wandb.enabled=true \
  "wandb.run_tag=lambda_sweep_${RUN_TAG}_ll${LAMBDA_LOGIT}" \
  "data.num_clips=${NUM_CLIPS}" \
  "data.preload=${PRELOAD}"
