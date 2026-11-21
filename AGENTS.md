# Experiment 1 working instructions

Read [Experiment 1](raw/experiment1.md) for the DGP and state-recovery goal.
The active task is parameter-information acquisition within that experiment.
Read [current experiment](docs/current_experiment.md) before
changing a model, loss, evaluation, or interpreting results. It records the
completed controls, artifact locations and the next proposed diagnostic.

## Scientific contracts

- Recover all six physical parameters from the complete code of one context;
  latent coordinates need not be disentangled. Keep retention and predictive use
  distinct, and distinguish empirical evidence from identifiability assumptions.
- Preserve the DGP and current protocol unless the user requests a change.
  Read [Experiment 1](raw/experiment1.md) for equations;
  read [architecture](docs/architecture.md) when changing model interfaces.
- Keep ordinary training label-free. True parameters and inverse/reference maps
  belong to evaluation or explicitly identified diagnostic controls.
- Keep donor codes attached through both autoregressive prediction steps, and
  feed the first prediction into the second without detaching. Frozen predictor
  weights must still transmit gradients to the trainable parameter encoder.
- Keep train, calibration, validation and test groups separate. Use validation
  for probe selection and the final fixed-budget checkpoint for test reporting.
- Before a new scientific run, state its hypothesis, controlled change and how
  each possible result changes the next decision. A software test passing does
  not establish parameter recovery; failed recovery alone does not prove a bug.

## Working and verification

- Short CPU/offline-W&B checks may run locally. Runs longer than a couple of
  minutes belong on the server; provide the exact command and artifact paths.
- Preserve source checkpoints, caches, outputs, logs, prior controls, third-party
  licenses and provenance. Keep checkpoint/source hashes meaningful on resume.
- For code changes, run focused existing tests, `scripts/smoke.py`, Ruff
  format/lint and strict Pyright, including new standalone diagnostics explicitly.
  Use `.venv/bin/python -m pytest`; write tests for meaningful behavior.
- For documentation changes, check links and scientific settings against the
  implementation and recorded protocol. Keep the raw experiment brief and
  current investigation authoritative; avoid duplicating them in new guides.

## Agent skills

Project-local skills live in `.agents/skills`. Read
[skill integration](.agents/README.md) when invoking or updating them. The current
user request and experiment record are the local specification; implementation
standards remain in this file, the architecture reference and existing checks.
Apply a skill within the authorized task and reuse already-settled decisions.

### Issue tracker

Track issues and specs as local Markdown in `.scratch/`. Before ticket or spec
operations, read [tracker conventions](docs/agents/issue-tracker.md).

### Triage labels

Use the five default triage roles as local status values. Before triaging, read
[the status vocabulary](docs/agents/triage-labels.md).

### Domain docs

Use a single context: `GLOSSARY.md` and `docs/adr/`. Before domain exploration,
read [the domain rules](docs/agents/domain.md).
