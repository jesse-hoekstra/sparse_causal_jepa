# Configuration

`experiments/independent_gaussian.yaml` runs the nonlinear Gaussian-preserving
DGP with the complementary graph, K=3, T=5, paired episodes, SIGReg and 100,000
steps per stage at constant learning rate 0.001. Its companion
`experiments/independent_gaussian_smoke.yaml` runs a tiny CPU check. See
[the server guide](../docs/independent_gaussian_experiment.md). Both use the existing
experiment entrypoint with the explicit `data_family=independent_gaussian` selector.

`experiments/controlled_vector.yaml` defines dense calibration and two GECO models on
complementary data with seed 0 and two four-frame episodes per parameter vector
(K=2, L=1), with two-step autoregressive prediction across the episodes in both directions;
`experiments/controlled_vector_smoke.yaml` checks its infrastructure with two CPU
microbatches. Use them with `scripts/run_controlled_experiment.py`, not the generic
Hydra entrypoint. Both require SIGReg for state and parameter regularization.
See [the experiment guide](../docs/controlled_experiment.md) for configuration,
selection, W&B, selectors, and reproducibility rules.

The vector coefficients are an approved modification of the paper's DGP, described
in [data generation](../docs/data_generation.md). Train a fresh dense reference and
calibrate tau for this version; earlier one-step, teacher-forced, single-episode and weak-coupling runs are
incompatible. `episodes_per_parameter=2` preserves independent initial states and
actions within each group. `train_episodes`, evaluation episode counts and batch
sizes count parameter groups: 8,192 training groups contain 16,384 trajectories,
and batch size 64 contains 128 trajectories. SIGReg uses each view separately
across 64 independent groups. No additional consistency-loss weight is needed.
The experiment sets `model.autoregressive=true`: each rollout keeps the partner's
parameter code fixed and feeds the first prediction into the second step.
Prediction loss averages both steps and both directions. Generic model defaults
retain teacher forcing with `model.autoregressive=false`.

Set `dense_reference_dir` to a compatible experiment root with a new `output_dir`
to reuse its selected dense checkpoint and original tau. This runs only baseline
and full with fresh optimizer/controller state, leaving the reference untouched.
Omit `--resume` on the initial invocation; add it to the same command after an
interruption. Without a reference, the default still runs all three stages.

`config.yaml` supplies small generic defaults, not an experiment preset.
`ModelConfig` in `scjepa.training.factory` defines architecture fields;
`TrainConfig`, `ObjectiveConfig` and `GECOConfig` define optimization, explicit
loss weights and optional adaptive sparsity. The experiment derives tau, enables
GECO, and sets gate modes per condition automatically. Baseline sets
`model.spartan_parameter_gates_open=true` and parameter sparsity weight zero;
full leaves those gates learnable and uses the configured positive parameter weight.
Manual gate-mode overrides are rejected by the experiment configuration.
Unknown model/training keys are rejected. The CLI requires an explicit data
group or a saved tensor/bundle path; no scientific dataset is selected implicitly.

`context_len` counts observations (with `K-1` context actions), `history_len`
counts dynamics lags, and `train.steps` counts microbatches. Dimensions are
independent. In generic training, setting `train.objective.parameter_sparsity_weight`
to zero removes its loss term but leaves gates learnable. Set
`model.spartan_parameter_gates_open=true` to force parameter-source columns into
decoded latest-state receiver rows open in every layer during training and evaluation.
Other gates and all attention weights remain learned. Regularizer choices and
weights are separate fields; see the architecture document for sample axes.

Data groups `data=controlled_vector` and `data=coupled_springs` select the modified
vector DGP and paper spring DGP with compatible interface dimensions and independent
data/loss settings.
`data.parameter_graph` selects the vector graph; `data.split`, `data.root_seed`,
`data.observation_seed`, count and trajectory length control reproducible data.
Spring appearance and Gaussian initial-condition choices are explicit.
The spring preset uses eight context frames and two history states.
See [data generation](../docs/data_generation.md) for complete conventions.

The separate `wandb` section controls experiment tracking, enabled online by
default for CLI training. `mode` supports `online`, `offline`, and `disabled`.
Project/entity/mode defaults accept `WANDB_PROJECT`, `WANDB_ENTITY`, and
`WANDB_MODE`; explicit overrides take precedence. Set name/group/tags/job_type
to organize comparisons. Tracking settings do not change losses or sampling.
See the root README for authentication, local modes and checkpoint resumption.
