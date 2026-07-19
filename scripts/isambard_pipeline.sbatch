#!/bin/bash
# Isambard-AI (GH200, aarch64, Slurm) launch for the D21 bounce_baumgartner
# pipeline. Submit FROM THE REPO ROOT:
#
#   sbatch --account=<PROJECT> scripts/isambard_pipeline.sbatch [run_tag]
#
# Fill in --account (from `sacctmgr show assoc user=$USER format=account%20`)
# and adjust --partition if `sinfo -s` names one explicitly for GPU work.
# NOTE: unlike the omi box there is NO nohup here — sbatch detaches the job;
# follow it with `tail -f logs/slurm-<jobid>.out` or `squeue --me`.
#
# Walltime: single job runs BOTH stages (calibration then main, D12). If the
# GPU smoke shows the two stages cannot fit the partition's MaxTime, split
# into two jobs at the tau boundary instead of raising --time blindly.
#SBATCH --job-name=scjepa-bounce
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
# workq reports MaxTime=UNLIMITED (checked 2026-07-19); 48h is a self-imposed
# cap so a hung job cannot squat on a GH200 forever. If sbatch rejects the
# submission with an account error, find yours via
#   sacctmgr show assoc user=$USER format=account%20
# and submit with --account=<it>.

set -euo pipefail
RUN_TAG=${1:-d21_isambard_seed0}
mkdir -p logs

# Isambard-AI login/compute nodes offer at most cray-python/3.11; the project
# pins >=3.12,<3.13, so the env lives in Miniforge (aarch64) instead of .venv.
source ~/miniforge3/bin/activate scjepa
export PYTHON=python
# Sanity: fail fast if the node/torch build has no CUDA (aarch64 wheel issue).
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

# Pre-generate the train split in parallel (idempotent: skips if the file
# exists). Single-threaded generation costs 7.6h per stage on a Grace core
# (measured 2026-07-19); on 64 workers this is ~10 min, paid once, shared by
# BOTH stages via data.preload (same override to both = D12 by construction;
# --cpus-per-task=64 above exists for this step — training itself needs few).
NUM_CLIPS=100000
PRELOAD="data/bounce_train_${NUM_CLIPS}.pt"
python scripts/pregenerate_bounce.py experiment=bounce_baumgartner "data.num_clips=${NUM_CLIPS}"

bash scripts/run_bounce_example.sh \
  "--run-tag=${RUN_TAG}" \
  --main-steps=300000 \
  --tau-max=0.19 \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda \
  "data.num_clips=${NUM_CLIPS}" \
  "data.preload=${PRELOAD}"
