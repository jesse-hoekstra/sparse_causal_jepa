#!/bin/bash
# Isambard-AI (GH200, aarch64, Slurm) launch for the D24 bounce_baumgartner
# pipeline. Submit FROM THE REPO ROOT:
#
#   sbatch --account=<PROJECT> scripts/isambard_pipeline.sbatch [run_tag]
#
# Fill in --account (from `sacctmgr show assoc user=$USER format=account%20`)
# and adjust --partition if `sinfo -s` names one explicitly for GPU work.
# NOTE: unlike the omi box there is NO nohup here — sbatch detaches the job;
# follow it with `tail -f logs/slurm-<jobid>.out` or `squeue --me`.
#
# Walltime: one job runs dense and identity references, then main (D12). If the
# GPU smoke shows the three stages cannot fit the partition's MaxTime, split
# them at the reference/main boundaries instead of raising --time blindly.
#SBATCH --job-name=scjepa-bounce
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
# workq_qos caps MaxWall at 1-00:00:00 (48h was rejected, 2026-07-19); ask
# for the full 24h. Expected runtime is ~8-12h (3x 1.9h training + ~60
# periodic evals + final evals + pregen), but train.py has NO resume: a job
# killed at the wall loses its in-flight stage entirely, while over-asking
# costs only backfill priority on a busy queue.
# If sbatch rejects with an ACCOUNT error, find yours via
#   sacctmgr show assoc user=$USER format=account%20
# and submit with --account=<it>.

set -euo pipefail
# Paper-faithful SPARTAN App. A.2: tau is exactly the converged fully connected
# reference loss. If the gated model cannot reach it, diagnose gate commitment;
# do not silently loosen the reported constraint.
RUN_TAG=${1:-d24_seed0}
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
# all stages via data.preload (same override throughout = D12 by construction;
# --cpus-per-task=64 above exists for this step — training itself needs few).
NUM_CLIPS=100000
PRELOAD="data/bounce_train_v2_${NUM_CLIPS}.pt"
python scripts/pregenerate_bounce.py experiment=bounce_baumgartner "data.num_clips=${NUM_CLIPS}"

bash scripts/run_bounce_example.sh \
  "--run-tag=${RUN_TAG}" \
  --main-steps=300000 \
  --eval-episodes=5000 \
  --identity-check \
  --tau-factor=1.0 \
  experiment=bounce_baumgartner \
  wandb.enabled=true \
  train.device=cuda \
  "data.num_clips=${NUM_CLIPS}" \
  "data.preload=${PRELOAD}"
