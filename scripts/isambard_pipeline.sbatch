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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
RUN_TAG=${1:-d21_isambard_seed0}
mkdir -p logs

source .venv/bin/activate
export PYTHON=python
# Sanity: fail fast if the node/torch build has no CUDA (aarch64 wheel issue).
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"

bash scripts/run_bounce_example.sh \
  "--run-tag=${RUN_TAG}" \
  --main-steps=300000 \
  --tau-max=0.19 \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda
