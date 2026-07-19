#!/bin/bash
# Isambard-AI: MAIN sparse run only — for when tau is already known and the
# dense calibration does not need re-running (e.g. D22: the dual-schedule
# changes are inert for a sparsity-disabled reference, so fybk7ukv's number
# is reused: tau = 1.2 x 0.0839 = 0.1007).
#
# Submit FROM THE REPO ROOT:
#   sbatch scripts/isambard_main_only.sbatch [run_tag] [tau] [steps]
# Defaults: d22_seed0 / 0.1007 / 300000.
#
# PROVENANCE NOTE: the run's tau traces to a PREVIOUS pipeline's calibration,
# not a sibling run. Fine for schedule iteration; for a paper-headline run,
# use scripts/isambard_pipeline.sbatch for a clean same-pipeline lineage (D12).
#SBATCH --job-name=scjepa-main
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
RUN_TAG=${1:-d22_seed0}
TAU=${2:-0.1007}
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
  data.preload=data/bounce_train_100000.pt \
  "hydra.run.dir=${OUT_DIR}"

echo "== final identifiability evaluation =="
python scripts/eval_identifiability.py "${OUT_DIR}" --episodes 512
echo "artifacts in ${OUT_DIR} (checkpoint, resolved config, recovery_grid.png)"
