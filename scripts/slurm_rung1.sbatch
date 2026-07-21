#!/bin/bash
# Rung 1 (Baumgartner-aligned true-state JEPA, D24): 8 seeds in parallel.
# Submit from the repo root after the dense sweep:
#   sbatch --export=ALL,LAMBDA_LOGIT=YOUR_SELECTED_VALUE scripts/slurm_rung1.sbatch
#
# Prerequisites on the cluster (once):
#   python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
#   wandb login            # or export WANDB_API_KEY=... ; use offline mode if
#                          # compute nodes have no internet (sync afterwards).
#
# GPU note: the states-regime model is small; CPU tasks are perfectly viable.
# For GPUs, uncomment the two GPU lines and add train.device=cuda below —
# and do a short smoke run first (the code is CPU-tested; see decisions.md).

#SBATCH --job-name=rung1_bounce
#SBATCH --output=outputs/rung1_%A_task%t.out
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
##SBATCH --gpus=8
##SBATCH --ntasks-per-gpu=1
#SBATCH --time=48:00:00

# module load cray-python   # only if the cluster requires it for the venv

: "${LAMBDA_LOGIT:?set LAMBDA_LOGIT to the dense-sweep selection}"

srun --ntasks=8 bash -c '
  SEED=${SLURM_PROCID}
  export PYTHON=.venv/bin/python
  bash scripts/run_bounce_example.sh \
    --run-tag="rung1_seed${SEED}" --tau-factor=1.0 --main-steps=300000 \
    --eval-episodes=5000 --identity-check \
    experiment=bounce_baumgartner \
    train.lambda_logit=${LAMBDA_LOGIT} train.sparsity_warmup_steps=0 \
    train.seed=${SEED} data.seed=${SEED} \
    wandb.enabled=true train.eval_every=5000 train.eval_episodes=128 \
    > "outputs/rung1_seed${SEED}.log" 2>&1
'
