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
2. evaluate its held-out final constraint and calibrate a fresh tau;
3. train the sparse model with that tau;
4. run terminal identifiability and observational-equivalence evaluation.

Do not carry forward D30's `tau=0.02` or any threshold calibrated for the K=30 training
objective. The historical stable setup supplies the 300k-step budget, `lambda_logit=1e-5`, and
`sparsity_lambda_init=1e4`; tau is newly measured for the final predictive objective.

Experiment-1 training logs
`train/loss_teacher_forcing`, `train/loss_rollout_t2_raw`,
`train/loss_rollout_t2_weighted`, and `train/loss_total`. If branch-gradient logging is enabled,
use `train/grad_norm_teacher_forcing` and `train/grad_norm_rollout_t2_weighted`. Evaluation logs
`eval/oe_sample_satisfaction_k30`, `eval/oe_k30_worst_step_nrmse_p50`, and
`eval/oe_k30_worst_step_nrmse_p95` from a fixed held-out, no-gradient K=30 rollout normalized by
fixed training-set coordinate standard deviations. These values estimate tolerance-based
agreement on sampled trajectories; they do not prove population observational equivalence.

Experiment 3's visual-to-visual fixed-K rollout is a separate latent-space protocol and remains
unchanged.

**Owner:** experiment-infra-engineer (`prepare_data.py` jointly with data-pipeline-engineer).
