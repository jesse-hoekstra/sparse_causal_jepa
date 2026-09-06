#!/bin/bash
# Experiment 2: identical rendered balls -> learned slots -> EMA latent targets.
# Submit from the repository root:
#   sbatch scripts/l40_exp2_pipeline.sbatch exp2_seed0_l40 1e-5 0
# Optional fourth argument changes BOTH dense and sparse training lengths:
#   sbatch scripts/l40_exp2_pipeline.sbatch exp2_smoke 1e-5 0 1500
# Optional fifth argument selects 1, 2 (default), or 4 GPUs; match the allocation:
#   sbatch --gres=gpu:4 scripts/l40_exp2_pipeline.sbatch exp2_four 1e-5 0 300000 4
# Set site account/partition and time limits via sbatch command-line options.
# Training splits the fixed global batch of four across x86_64 L40 GPUs.
# Evaluation uses one GPU; all stages use the existing .venv/physics preload.
# Dense -> label-free collapse screen + fresh tau -> sparse -> disjoint test.
# A short run tests numerical health; it cannot establish experiment success.
#SBATCH --job-name=scjepa-exp2-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 5 ]; then
  echo "usage: sbatch $0 RUN_TAG LAMBDA_LOGIT [SEED] [STEPS] [GPUS=2]" >&2
  exit 2
fi
RUN_TAG=$1
LAMBDA_LOGIT=$2
SEED=${3:-0}
STEPS=${4:-300000}
NUM_GPUS=${5:-2}
case "${NUM_GPUS}" in
  1|2|4) ;;
  *) echo "GPUS must be 1, 2, or 4 (global batch stays four)" >&2; exit 2 ;;
esac
REPO_ROOT=${SLURM_SUBMIT_DIR:-$PWD}
cd "${REPO_ROOT}"
if [ ! -f pyproject.toml ] || [ ! -f scripts/eval_visual_to_visual.py ]; then
  echo "submit this script from the sparse_causal_jepa repository root" >&2
  exit 2
fi
ROOT="outputs/bounce_exp2_${RUN_TAG}"
if [ -e "${ROOT}" ]; then
  echo "refusing to overwrite existing run: ${ROOT}" >&2
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  echo "missing .venv/bin/python; create the Python 3.12 environment first" >&2
  exit 2
fi
export PYTHON="${REPO_ROOT}/.venv/bin/python"
export PYTHONUNBUFFERED=1
"${PYTHON}" -c \
  'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and v >= 0, "lambda_logit must be finite and non-negative"; assert int(sys.argv[2]) >= 0, "seed must be non-negative"; assert int(sys.argv[3]) > 0, "steps must be positive"' \
  "${LAMBDA_LOGIT}" "${SEED}" "${STEPS}"
"${PYTHON}" -c \
  'import sys,torch; n=int(sys.argv[1]); assert torch.cuda.is_available(), "no CUDA GPU allocated"; assert torch.cuda.device_count() >= n, f"requested {n} workers but only {torch.cuda.device_count()} CUDA GPUs visible; match sbatch --gres=gpu:N to the fifth argument"; print(f"GPU: {torch.cuda.get_device_name(0)}; training GPUs: {n}; global batch: 4; batch per GPU: {4//n}")' \
  "${NUM_GPUS}"

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
if [ "${SEED}" -eq 0 ]; then
  EXPECTED_SHA256=2d2dd9751a48bd0523dae92451091b706653bc44ec89cb1f1fb62b136935fe85
  ACTUAL_SHA256=$(sha256sum "${PRELOAD}" | awk '{print $1}')
  if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "wrong preload checksum for ${PRELOAD}" >&2
    echo "expected ${EXPECTED_SHA256}; actual ${ACTUAL_SHA256}" >&2
    exit 2
  fi
fi

echo "host=$(hostname) job_id=${SLURM_JOB_ID:-none} run_tag=${RUN_TAG} seed=${SEED} steps=${STEPS}"
echo "training GPUs=${NUM_GPUS}; global batch=4; evaluation uses one GPU"
echo "preload=${PRELOAD} sha256=$(sha256sum "${PRELOAD}" | awk '{print $1}')"
COMMON=(
  experiment=bounce_visual_to_visual
  wandb.enabled=true
  train.device=cuda
  train.batch_size=4
  # Divide the existing eight data workers across ranks rather than multiplying them.
  "train.num_workers=$((8 / NUM_GPUS))"
  hydra.output_subdir=null
  hydra/job_logging=disabled
  "train.lambda_logit=${LAMBDA_LOGIT}"
  "train.seed=${SEED}"
  "train.steps=${STEPS}"
  train.lambda_rollout_t2=1.0
  train.num_rollout_t2_anchors=8
  train.rollout_t2_horizon=2
  train.oe_eval_horizon=30
  "data.seed=${SEED}"
  "data.num_clips=${NUM_CLIPS}"
  "data.preload=${PRELOAD}"
  data.cache=false
  data.render_radius_from_mass=false
  data.uniform_appearance=true
)
DENSE_DIR="${ROOT}/dense"
MAIN_DIR="${ROOT}/main"
echo "stage 1/3: dense reference with teacher forcing and sampled T=2 endpoints"
"${PYTHON}" -m torch.distributed.run --standalone "--nproc_per_node=${NUM_GPUS}" \
  --local-addr=127.0.0.1 \
  scripts/train.py "${COMMON[@]}" \
  model.spartan_dense=true train.sparsity_enabled=false \
  "wandb.run_tag=${RUN_TAG}-dense" "hydra.run.dir=${DENSE_DIR}"
"${PYTHON}" scripts/eval_visual_to_visual.py "${DENSE_DIR}" \
  --episodes 1000 --probe-episodes 128 --seed-offset 17 --device cuda \
  --require-complete-protocol --require-noncollapsed
TAU=$("${PYTHON}" -c \
  'import json,math,sys; value=json.load(open(sys.argv[1]))["constraint_loss"]; assert math.isfinite(value) and value > 0; print(value)' \
  "${DENSE_DIR}/metrics.json")
echo "stage 2/3: sparse run at tau=${TAU} (1.0x dense raw TF+T=2+logit constraint)"
echo "Tau is exploratory: different learned target spaces need not imply equal physical fidelity."
"${PYTHON}" -m torch.distributed.run --standalone "--nproc_per_node=${NUM_GPUS}" \
  --local-addr=127.0.0.1 \
  scripts/train.py "${COMMON[@]}" \
  model.spartan_dense=false train.sparsity_enabled=true \
  "train.sparsity_tau=${TAU}" "wandb.run_tag=${RUN_TAG}-sparse" "hydra.run.dir=${MAIN_DIR}"
echo "stage 3/3: held-out test and concise W&B report"
"${PYTHON}" scripts/eval_visual_to_visual.py "${MAIN_DIR}" \
  --episodes 1000 --probe-episodes 128 --seed-offset 29 --device cuda \
  --require-complete-protocol --require-noncollapsed
echo "Artifacts: ${ROOT}/{dense,main}/{metrics.json,mcc_matrix.json,slot_tracks.png,recovery_grid.png}"
echo "Inspect MCC with SHD/density, tracking, held-out position/velocity probes, and collapse health."
echo "A low latent loss or low SHD alone does not establish success."
