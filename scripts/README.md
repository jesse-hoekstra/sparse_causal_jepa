# Entrypoints

- `run_controlled_experiment.py`: dense calibration followed by two GECO-sparse comparisons, validation,
  best checkpoints, held-out evaluation, W&B, safe resume and aggregation. See
  [the experiment guide](../docs/controlled_experiment.md) for tmux/server commands.
  The separate [independent Gaussian preset](../docs/independent_gaussian_experiment.md)
  uses this same entrypoint with three context frames and two forecast targets.

- `smoke.py`: CPU forward/backward check using random image and vector tensors.
- `generate_data.py`: inspect small deterministic episodes or stream a versioned
  training/evaluation bundle. See the data-generation document for commands.
- `train.py`: Hydra-configured training through a selected DGP, bundle, or tensor adapter;
  accepts ordinary Python or `torchrun` execution. Prints scalar JSON metrics
  and logs to W&B by default; set `wandb.mode=offline` or `disabled` for local use.
- `check_nccl.py`: optional GPU communication diagnostic, independent of models
  and data. No GPU check is required for the CPU refactor verification.

The controlled-vector scientific protocol is isolated under `scjepa.experiments`;
the other entrypoints retain generic data/model interfaces. No sweep is launched
automatically. See the root README for the batch contract.
