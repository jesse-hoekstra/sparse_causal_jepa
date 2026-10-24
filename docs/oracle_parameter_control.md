# Oracle-parameter control

This control supplies the six raw system parameters to the dynamics predictor.
It tests whether parameter inference limits prediction and the benefit of
parameter-graph sparsity. It does not demonstrate learned parameter recovery or
joint identifiability. A strong baseline may still leave little state-MCC headroom.

`configs/experiments/simplified_gaussian_r10_oracle.yaml` keeps the simplified
Gaussian generator, complementary graphs, observation map, seed 0, ten episode
views per parameter group, K=10/T=12, two autoregressive forecast steps, and the
state encoder/predictor architecture. Batch size is 16 independent parameter
groups, state SIGReg weight is 0.4, and each condition has 100,000 updates.

## Supplied information and objective

`model.parameter_source=oracle` explicitly enables the control. Each training
item additionally contains `oracle_parameters` of shape `(6,)`, taken directly
from the same addressed parameter draw as its ten trajectories. Batches contain
`(B,6)`; the model reuses that group's detached code across all episode views and
forecast steps. These are the original uniform parameters in `[-0.75,0.75]`,
without normalization, Gaussianization, or learned transformation. True states,
graph masks and inverse observations are never training inputs.

The parameter encoder is still constructed, preserving the seed-dependent
initialization of the other modules, but is frozen and never called. The default
`learned` mode rejects supplied parameter inputs. Ordinary generated datasets
continue to exclude all scientific labels; a separate oracle adapter exposes
only the authorized parameter vector and has a distinct training-data identity.

Parameter SIGReg and parameter consistency weights are both zero and nonzero
values are rejected in oracle mode. They would otherwise contribute only fixed,
data-dependent terms, rather than regularize a learned representation. The base
objective is therefore:

\[
L_{\mathrm{base}}=L_{\mathrm{prediction}}
  +0.4\,L_{\mathrm{SIGReg,state}}+10^{-4}L_{\mathrm{logit}}.
\]

Dense keeps every gate open. Baseline learns state/action gates while forcing
parameter-source gates into decoded state receivers open. Full additionally
learns and penalizes parameter gates. Both sparse conditions start from the same
selected oracle dense checkpoint, with fresh optimizer/controller state and the
same newly calibrated tau. Final held-out evaluation uses each condition's
`last.pt`; the dense `best.pt` is reserved for calibration and sparse warm starts.

## Run

Use a separate checkout/copy on the GPU server so an existing experiment can
continue with its original source files. Activate the usual Python 3.12 environment
with the project's dependencies. The explicit `PYTHONPATH=src` below selects this
copy without changing an existing environment's editable-install location. In a
new environment, install dependencies with `python -m pip install -e '.[dev,logging]'`
first. From the new repository directory, run:

```bash
PYTHONPATH=src python -u scripts/run_controlled_experiment.py \
  --config configs/experiments/simplified_gaussian_r10_oracle.yaml \
  train.device=cuda \
  wandb.mode=online \
  wandb.project=sparse_causal_jepa \
  wandb.entity=jesse-hoekstra-university-of-oxford
```

This automatically runs dense, baseline and full. The preset uses a fresh output
directory, `outputs/simplified_gaussian_m3_k10_r10_oracle_b16_sigreg04_100k`, and
W&B group `simplified-gaussian-m3-k10-r10-oracle-b16-sigreg04`; run names begin
with `oracle-parameter`. Do not reuse the learned-parameter experiment's output
directory, dense checkpoint or tau. The objectives and parameter inputs differ.

For a launch-plan check, add `--dry-run` before the configuration overrides.
After interruption, rerun the identical command with `--resume` before `--config`.
Keep the planned step count unchanged. Ctrl+C once requests a checkpointed stop
at an optimizer boundary; wait for exit before restarting. `--evaluate-only`
also supplies the oracle parameters and evaluates the final checkpoints.

## Interpretation

Compare baseline and full within this oracle experiment using state MCC, state
block linear R2, prediction error and aligned routing-support graph metrics.
Parameter MCC and column-pair linear R2 should be approximately one by
construction: these score supplied parameters, not learned recovery. The
Gaussianized-parameter MCC can be below one because oracle inputs are raw
uniform values. Parameter consistency variance should be zero. Evaluation
protocol metadata records these interpretations explicitly.

The shuffled-parameter diagnostic supplies another parameter group's true theta
to test whether the predictor uses the known system parameters. It does not
change the scientific trajectories or training. Improved state prediction with
oracle parameters supports an inference bottleneck; a remaining baseline/full
tie would leave architecture, optimization and generator cues as explanations.
Base losses and tau are not directly comparable to the learned-parameter runs,
whose objectives also include parameter regularization and consistency.
Raw oracle inputs also differ in scale and distribution from the learned
SIGReg-regularized codes. An oracle-versus-learned improvement can therefore
reflect both better parameter information and different optimization/conditioning.
