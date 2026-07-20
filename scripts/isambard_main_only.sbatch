#!/bin/bash
# Isambard-AI: MAIN sparse run only — for when tau was calibrated under the
# exact current architecture and verified below a matched identity reference.
# D24 changed the simulator, parameter bottleneck, pairing, prediction-loss
# units, and path objective, so pre-D24 values such as 0.0839 are invalid and
# deliberately have no default here.
#
# Submit FROM THE REPO ROOT:
#   sbatch scripts/isambard_main_only.sbatch [run_tag] [tau] [steps]
# The tau argument is mandatory. Prefer scripts/isambard_pipeline.sbatch.
#
# PROVENANCE NOTE: record the sibling calibration run and factor for every tau.
#SBATCH --job-name=scjepa-main
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
RUN_TAG=${1:-d24_seed0}
if [ "$#" -lt 2 ]; then
  echo "usage: sbatch $0 RUN_TAG CALIBRATED_TAU [STEPS]" >&2
  echo "prefer scripts/isambard_pipeline.sbatch so tau is recalibrated in-pipeline" >&2
  exit 2
fi
TAU=$2
STEPS=${3:-300000}
OUT_DIR="outputs/bounce_example_${RUN_TAG}/main"
mkdir -p logs

source ~/miniforge3/bin/activate scjepa
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
echo "main-only run: tag=${RUN_TAG} tau=${TAU} steps=${STEPS} -> ${OUT_DIR}"

python scripts/train.py \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda \
  "train.sparsity_tau=${TAU}" \
  "train.steps=${STEPS}" \
  data.preload=data/bounce_train_v2_100000.pt \
  "hydra.run.dir=${OUT_DIR}"

echo "== final identifiability evaluation =="
python scripts/eval_identifiability.py "${OUT_DIR}" --episodes 5000
echo "artifacts in ${OUT_DIR} (checkpoint, resolved config, recovery_grid.png)"
