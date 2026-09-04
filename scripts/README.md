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

Once a fixed tau has been selected, the L40 eight-seed confirmation is one submission:

```bash
sbatch scripts/l40_exp1_8seed_pipeline.sbatch TAU
```

The Slurm array covers seeds 0–7 and limits itself to two concurrent one-GPU tasks. Each task
trains a dense and sparse model with the same seed/data and evaluates both on 5,000 held-out
episodes at offset 29. Missing seed-specific preloads for seeds 1–7 are generated on the L40;
the canonical seed-0 preload must already be present. Outputs live under
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

Experiment 3's visual-to-visual fixed-K rollout is a separate latent-space protocol and remains
unchanged.

**Owner:** experiment-infra-engineer (`prepare_data.py` jointly with data-pipeline-engineer).
