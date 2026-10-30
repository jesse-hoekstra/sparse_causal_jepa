# Verification contracts

Run `.venv/bin/python -m pytest`. The masked-MLP tests cover paired
initialization, deterministic chart equivariance, label exclusion, attached
two-step gradients, fixed-budget guards, exact resume, source authentication,
TRAIN-only selection, code projection and paper metrics. DGP tests verify the
Paley orientation, smooth rotations, analytic Jacobians and separate cache
labels. Core tests cover the reusable model, loss and training components.

`python scripts/smoke.py` provides a short CPU check of both published model
paths. Passing software checks is distinct from scientific recovery evidence,
which is recorded in `results/`.
