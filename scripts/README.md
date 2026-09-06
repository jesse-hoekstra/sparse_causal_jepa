# scripts/

Hydra-driven training, evaluation, data-preparation, aggregation, and cluster launch entry
points. Model, loss, evaluation, and optimization logic lives in `src/scjepa/`.

For Experiment 1, `train.py` runs the fixed objective

```text
L_pred = L_TF + lambda_rollout_t2 * L_AR2
```

at every optimizer step. It has no rollout stage transitions, horizon warmup, accepted-update
curriculum, gradient-cut schedule, or K=30 backward pass. The old D34–D36 schedule is historical
only and must not be reconstructed through launch overrides.

The full pipeline remains:

1. train a dense reference with the same teacher-forcing-plus-T=2 settings as the sparse run;
2. train an identity reference with exactly the same settings;
3. evaluate both references on the same held-out split and select the first factor in
   `[2.0, 1.8, 1.6, 1.4]` for which `C_dense < factor * C_dense < C_identity`;
4. abort if no candidate is feasible; otherwise train the sparse model with the selected tau;
5. run terminal identifiability and observational-equivalence evaluation.

Do not carry forward D30's `tau=0.02` or any threshold calibrated for the K=30 training
objective. The historical stable setup supplies the 300k-step budget, `lambda_logit=1e-5`, and
`sparsity_lambda_init=1e4`; tau is newly measured for the final predictive objective.

Once a fixed tau has been selected, the manually managed L40 server can run two worker lanes
on GPUs 4 and 5:

```bash
SCJEPA_BATCH_ID=d38_8seed_fixed_tau_l40 CUDA_VISIBLE_DEVICES=4,5 \
  bash scripts/l40_exp1_8seed_pipeline.sbatch TAU
```

GPU 4 processes seeds 0/2/4/6 and GPU 5 processes seeds 1/3/5/7. On a Slurm-managed server,
the same file can instead be submitted with `sbatch scripts/l40_exp1_8seed_pipeline.sbatch TAU`;
its `0-7%2` array provides the same two-way concurrency. Each task trains a dense and sparse
model with the same seed/data and evaluates both on 5,000 held-out episodes at offset 29.
Missing seed-specific preloads for seeds 1–7 are generated on the L40; the canonical seed-0
preload must already be present. Outputs live under
`outputs/l40_exp1_8seed_job<ARRAY_JOB_ID>/seed<SEED>/{dense,sparse}`. After all eight pairs
succeed, the final task writes validated summary JSON and paired box plots under the root's
`aggregate/` directory.

The Figure-3-style outputs distinguish two loss definitions. `constraint_loss` is the actual
tau quantity, `L_TF + lambda_T2 L_AR2 + lambda_logit L_logit`, and is never labelled MSE. The
separate MSE figure uses `trajectory_reconstruction_mse_k30`, the raw MSE from our held-out
30-step autoregressive rollout and the closest analogue to Baumgartner et al.'s pure
autoregressive trajectory-reconstruction MSE. The one-step teacher-forced `pred_loss` remains
available in the summary JSON but is not presented as the paper metric.

Experiment-1 training logs
`train/loss_teacher_forcing`, `train/loss_rollout_t2_raw`,
`train/loss_rollout_t2_weighted`, and `train/loss_total`. If branch-gradient logging is enabled,
use `train/grad_norm_teacher_forcing` and `train/grad_norm_rollout_t2_weighted`. Evaluation logs
`eval/trajectory_reconstruction_mse_k30`, `eval/oe_sample_satisfaction_k30`,
`eval/oe_k30_worst_step_nrmse_p50`, and
`eval/oe_k30_worst_step_nrmse_p95` from a fixed held-out, no-gradient K=30 rollout normalized by
fixed training-set coordinate standard deviations. These values estimate tolerance-based
agreement on sampled trajectories; they do not prove population observational equivalence.

Experiment 2 uses the same TF+T=2 sampling and endpoint loss in learned visual state space.
The L40 pipeline defaults to two GPUs on one machine:

```bash
bash scripts/l40_exp2_pipeline.sbatch visual_seed0 1e-5 0
# Slurm uses the same arguments:
sbatch scripts/l40_exp2_pipeline.sbatch visual_seed0 1e-5 0
# A short two-GPU check uses 1,500 updates in both training stages:
sbatch scripts/l40_exp2_pipeline.sbatch visual_smoke 1e-5 0 1500
# For one or four GPUs, match the Slurm allocation and fifth argument:
sbatch --gres=gpu:1 scripts/l40_exp2_pipeline.sbatch visual_one 1e-5 0 300000 1
sbatch --gres=gpu:4 scripts/l40_exp2_pipeline.sbatch visual_four 1e-5 0 300000 4
```

Arguments are `RUN_TAG LAMBDA_LOGIT [SEED] [STEPS] [GPUS]` (defaults: seed 0, 300,000 steps,
two GPUs; GPU count must be 1, 2, or 4). The launcher uses `torch.distributed.run` for training,
checks the visible GPU count, and keeps the total batch at four, dividing episodes and the
eight data workers across processes. Optimizer gradients, the global mean raw constraint,
rejected updates, and EMA updates are synchronized. Target variance is gathered only for logged
diagnostics; it no longer requires an every-update collective for the loss. Rank zero owns W&B, checkpoints, and evaluation.
Dense and sparse stages still run sequentially because sparse training requires the calibrated
dense threshold. Final evaluation remains single GPU, so total pipeline speedup will be lower
than training speedup. No L40 scaling measurement has been made yet.

For custom distributed launches, `train.batch_size` always means the global batch and must be
divisible by the process count. Use `hydra.output_subdir=null hydra/job_logging=disabled` when
launching `scripts/train.py` directly with torchrun so worker processes do not compete to write
Hydra logs. The resolved configuration and its `distributed` metadata are saved once by rank zero.
Distributed training retains the combined gradient norm; separate branch-gradient diagnostics
are disabled because they use `autograd.grad`, which PyTorch DDP does not support. W&B logs
`train/seconds_per_step` and `train/episodes_per_second` over each logging interval. They include
data loading and the training work, with CUDA synchronized at interval boundaries, and exclude
evaluation, checkpoint writes, and logger calls. The first interval includes worker startup.
Total pipeline runtime still includes the excluded stages.

To compare hardware scaling before a full run:

```bash
sbatch scripts/l40_exp2_benchmark.sbatch speed_check
```

`RUN_TAG [STEPS]` defaults to 400 updates per GPU count; use a multiple of 100, at least 200.
One job requests
two L40s and runs the production dense visual model on one GPU and then two, with global batch
four, eight total data workers, and identical CPU thread limits. It uses the recorded seed-zero
preload and reports the last logging interval after warm-up. W&B and evaluation are disabled;
run configs/checkpoints, stdout logs, and `benchmark.json` go under
`outputs/bounce_exp2_benchmark_<RUN_TAG>`. Any skipped update invalidates the speed report. This
is a short dense-stage scaling measurement; sparse training and convergence require full runs.
It includes the normal single-GPU branch-gradient logging, which DDP omits as described above,
so it compares practical training throughput. Repeat with unique tags if the timings are noisy.

CUDA training now pins batches in the data loader and submits input copies without a host wait.
The five-slot context assignment is solved exactly on-device, and scalar monitoring metrics are
only materialized on logged updates. Numerical rejection guards still run on every update.

The corresponding Isambard entry point is `isambard_exp2_pipeline.sbatch`. Both require the
matching Experiment-1 physics preload and render equal-radius white balls on demand. Do not
regenerate the canonical preload on another machine. Outputs are
`outputs/bounce_exp2_<RUN_TAG>/{dense,main}`; an existing output root is rejected.

The visual pipeline trains a dense model, evaluates 1,000 held-out episodes at seed offset 17,
rejects a grossly collapsed reference, and sets tau to its raw
`L_TF + lambda_rollout_t2*L_AR2 + lambda_logit*L_logit` constraint. It then trains a sparse model
with identical TF+T=2 settings and evaluates 1,000 disjoint episodes at offset 29. Both stages
must carry `visual_constraint_version=raw_tf_t2_v1`; earlier normalized thresholds/checkpoints
cannot supply this calibration. Experiment 1's identity-reference slack selection remains
specific to its true-state experiment.

`eval_visual_to_visual.py` additionally accepts `--probe-episodes` (default 128),
`--require-noncollapsed`, and `--require-complete-protocol`. Final metrics and figures append
to the original W&B training run automatically when W&B is enabled in its saved configuration. Probe coefficients are fitted on frozen features from
training episodes; the scores use held-out episodes. Final W&B evaluation includes the core
MCC/SHD/path-density results, concise tracking/state-recovery diagnostics, and a small slot panel.
Dense and sparse models learn separate target feature spaces, so equal raw latent constraints
do not establish equal physical fidelity. The visual representation's coordinates need not
literally equal true states; position/velocity decodability and consistent tracking are the
relevant checks. Variance, rank, and temporal diagnostics remain, with the evaluation-only
`min_target_variance=1e-4` screening threshold. This threshold does not divide the loss. EMA and
stop-gradient alone do not establish successful representation learning.

**Owner:** experiment-infra-engineer (`prepare_data.py` jointly with data-pipeline-engineer).
