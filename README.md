# Sparse causal JEPA

An experiment-independent PyTorch implementation of framewise state encoding,
context-based parameter inference, and sparse finite-memory latent dynamics.
The motivating manuscript is `sources/Parameter_and_State_Identifiability.pdf`
(local source PDFs are ignored by Git).

The package supplies architecture, training infrastructure, the paper's data
generators, and a three-stage controlled-vector experiment runner. See
[the first experiment](docs/controlled_experiment.md) for the comparison,
tmux/W&B server commands, validation, recovery metrics, and resume. The scientific
protocol lives separately from the reusable model/trainer. See
[data generation](docs/data_generation.md) for equations and storage.

The [independent Gaussian experiment](docs/independent_gaussian_experiment.md)
uses the verified two-reflection and radial-rotation DGP, Gaussian actions,
three context frames and two forecast targets. Its separate preset preserves the
older controlled-vector experiment and requires fresh dense calibration.

The [simplified m=3 experiment](docs/simplified_gaussian_experiment.md) uses six
states, six uniform parameters and sixteen densely mixed observed actions. It
uses ten context frames and two forecast targets, comparing state/action sparsity
with state/action/parameter sparsity. Its preset requires fresh dense calibration.

## Model and batch interface

```python
import torch
from scjepa.training.factory import ModelConfig, build_model

model = build_model(ModelConfig(
    observation_dim=8, state_dim=5, param_dim=3, action_dim=2,
    context_len=4, history_len=2,
))
x = torch.randn(3, 8, 8)       # episodes, frames, observation coordinates
a = torch.randn(3, 7, 2)       # a[:, t] advances x[:, t] to x[:, t + 1]
out = model(x, a)
assert out.predictions.shape == (3, 4, 5)
assert out.parameters.shape == (3, 3)
loss = (out.predictions - out.targets).square().mean()
loss.backward()
```

- Vector observations: `(B,T,d_x)`; images: `(B,T,C,H,W)`, floating point.
- Actions: `(B,T-1,d_a)`; use `None` with `action_dim=0`.
- Each frame independently produces `(B,T,d_z)` through an MLP or CNN and its
  vector head. Normalization never spans frames or episodes.
- Parameter inference consumes the first `K=context_len` online states and the
  first `K-1` actions, producing **one attached `(B,d_theta)` tensor**.
- Prediction consumes `L=history_len` states and aligned actions. Source indices
  start at `s=max(K,L)-1` by default, with targets `s+1 ... T-1`. The result is
  `(B,T-1-s,d_z)`. `K`, `L`, `d_z`, `d_theta`, and `d_a` are independent.
- `out.targets` is a stopped-gradient representation from the separate complete
  EMA encoder, including its output head. There is no decoder or matching step.

See [architecture and choices](docs/architecture.md) for graph semantics, rollout,
EMA, and loss details.

## Training integration

`Trainer(model, TrainConfig(...), dataset=None)` exposes `train_step(batch)`.
A batch is a mapping with `observations`, optional `actions`, and `episode_ids`
(int64 `(B,)`) when parameter regularization is enabled. IDs identify original
episodes; separate clips from one episode must share an ID.

The basic loss is latent prediction MSE. `ObjectiveConfig` exposes independent
state/action/parameter sparsity weights, attention-logit weight, and independent
state/parameter `none` or `sigreg` selections and weights. Generic auxiliary
weights default to zero. Optional `GECOConfig` controls adaptive sparsity under
a calibrated bound on prediction plus representation/logit regularization.
The controlled experiment enables it for both sparse conditions. Its baseline
forces parameter-source gates into decoded latest-state rows open in every SPARTAN
layer, during training and evaluation, and sets `parameter_sparsity_weight=0`.
The full model learns those gates and adds their sparsity penalty. Both retain
learned attention weights, all other learned gates, and state/action regularization.
Both infer and reuse parameters through the same encoder and predictor interfaces.
The generic `spartan_parameter_gates_open` option defaults to false; setting a loss
weight to zero alone does not force any gates open.

`TensorEpisodes` is a tensor adapter, not a scientific data generator. To use the
saved-tensor CLI route, supply a `torch.save` mapping of episode-major `observations`,
optional `actions`, and optional `episode_ids`:

```bash
python scripts/train.py data_path=/absolute/path/episodes.pt train.steps=10
# Exact resume requires matching model config, batch size, world size, and data.
python scripts/train.py data_path=/absolute/path/episodes.pt \
  resume=/absolute/path/outputs/generic/last.pt train.steps=20
```

Alternatively select `data=controlled_vector` or `data=coupled_springs`, or pass
a versioned bundle directory as `data_path`. Model/data compatibility is checked
before training and checkpoints validate dataset identity.

These commands illustrate the interface; they prescribe no scientific schedule.
`steps` counts microbatches, `batch_size` is global across ranks, and
`accumulation_steps` controls microbatches per optimizer update. A final partial
accumulation group is explicitly discarded; checkpoints require a boundary.
EMA advances only after accepted optimizer updates. Both single-process and
`torchrun` execution are supported. CLI training logs to W&B and prints JSON;
direct `Trainer` use stays local unless a logger is supplied.

## Weights & Biases

Install the logging extra and authenticate once on the training machine:

```bash
python -m pip install -e '.[logging]'
wandb login
```

W&B is enabled by default in `scripts/train.py`. The project defaults to
`sparse_causal_jepa`; the entity defaults to your authenticated W&B settings.
Use `WANDB_PROJECT` / `WANDB_ENTITY`, or explicit Hydra settings:

```bash
python scripts/train.py data=controlled_vector \
  wandb.entity=YOUR_ENTITY wandb.project=sparse_causal_jepa \
  wandb.name=vector-check train.out_dir=outputs/vector-check
```

Set `wandb.mode=offline` to keep W&B records locally, or `wandb.mode=disabled`
for terminal logging only. `WANDB_MODE` supplies the default when set.
`wandb.group`, `wandb.tags`, and `wandb.job_type` are also configurable.
Missing dependencies or online initialization failures stop the run instead of
silently dropping tracking. Keep API keys in W&B's credential store or environment,
never in YAML or command-line configuration.

The controlled experiment logs only objective/GECO controls, MCC and SHD.
Generic training retains detailed diagnostics. Only rank zero creates the W&B
run and logs the reduced metrics. It records the
resolved model/data/training configuration, dataset identity, loss components,
gradient health, optimizer/skip counts and per-microbatch timing. Charts use
`train/microbatch` as their training axis. Logging preserves training RNG state;
datasets and checkpoint weights are not uploaded automatically.

`run_metadata.json` and checkpoints retain the run ID, entity and project.
Resuming an online checkpoint reconnects to that same W&B run, even with a new
output directory; conflicting entity/project overrides are rejected. A checkpoint
without W&B metadata starts a new tracking run. W&B's internal event index remains
monotonic when replaying an older checkpoint; the custom training axis can repeat.
Offline continuation writes separate local segments with the retained ID, without
claiming online resume or merging them automatically. Sync offline segments explicitly
using the [W&B CLI](https://docs.wandb.ai/ref/cli/wandb-sync).

## Development and verification

Python 3.12, with the project virtual environment activated:

```bash
python -m pip install -e '.[dev,logging]'
OMP_NUM_THREADS=1 pytest
ruff format --check .
ruff check .
pyright
python scripts/smoke.py
```

The smoke uses random CPU tensors only, exercises image/vector forward/backward,
regularizers, accumulation, and an EMA update, and writes no training artifacts.
Tests also exercise two CPU Gloo processes. No GPU run is required.

Reusable evaluation helpers cover ridge probes, nonlinear coordinate recovery,
structural Hamming distance, and trajectory-error summaries. Callers supply
reference coordinates, alignment, independent splits, and tolerances. Graph
reachability, low loss, EMA, and representation regularization **do not prove
injectivity or identifiability**.

SIGReg is the only enabled representation regularizer. Upstream attribution
and licensing are retained under
[src/scjepa/third_party](src/scjepa/third_party/README.md). Local data, checkpoints,
run outputs, logs, and unrelated notes are outside refactor cleanup.
