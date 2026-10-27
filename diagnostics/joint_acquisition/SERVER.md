# Bounded server handoff

This is an implementation handoff, not a recorded scientific execution. No new
session, log, output directory or W&B run has been created. Ticket 08 excludes
launching a scientific run while closing the implementation tickets. Use the
launch command only when executing the agreed diagnostic is authorized.

The [current experiment](../../docs/current_experiment.md) is authoritative for
sources and scientific interpretation. The runner keeps the DGP, architecture,
ordinary objective and attached two-step rollout unchanged. It diagnoses copies
of the original 30k joint endpoint; diagnostic supervision is not a production
training change or evidence of fresh acquisition.

## Hypothesis and fixed policy

Useful physical-parameter responses may be accessible through the code, shared
encoder weights, predictor weights, or coordinated corrections, while the
ordinary objective/Adam update may support or oppose those corrections.

Use one seed, `202610261301`; 64 calibration groups for fitting and 64 distinct
validation groups for assessment; separate donor/four fit/five query episodes;
all six physical parameters; both attached forecast horizons. Local code repair
is capped at 0.1 calibration code standard deviations. Local module corrections
use each module's next saved-Adam update norm and shrinking finite steps. The
three shared cases run sequentially on one GPU for 1,000 updates each, recording
0, 1, 10, 100 and 1,000. No early stopping for weak improvement and no automatic
extension beyond the budget. The launch fixes physical displacement `0.01`,
`rcond=1e-6`, diagnostic learning rate `0.001`, 1,000 group bootstrap samples and
one CPU thread; the complete policy is persisted in `protocol.json`.

- Better compatible source-code forecasts imply a code-dependent limitation;
  poor forecasts with both codes leave predictor adequacy open.
- Code repair that transfers to separate queries establishes local accessibility
  within its bound. Shared held-out improvement then distinguishes encoder,
  predictor and joint correction cases; a predictor-only update cannot improve
  retention in unchanged codes.
- Useful corrections opposed by objective/optimizer directions motivate a
  targeted incentive intervention. Delayed improvement motivates a timescale
  investigation. No improvement means failure within this tested budget.
- None of these outcomes establishes fresh joint acquisition. The follow-up order
  remains evidence-backed intervention, fresh joint training with fixed states,
  transfer to learned states, then the controlled state-MCC comparison.

## Required sources and synchronization

The script uses `omi-rapid-22`, repository
`/nfs/home/jesseh/sparse_causal_jepa`, Python `.venv/bin/python`, and SSH socket
`/Users/hoekstra/.ssh/codex-omi-rapid-22` with `BatchMode=yes`.

The recorded defaults are:

| Input | Path relative to the remote repository |
|---|---|
| Primary source directory | `outputs/diagnostics/predictive_supervision_1_4_16_frozen_mlp` |
| Primary checkpoint within that source | `seed-202610261301/prediction_1/last.pt` |
| Compatible portable bundle | `outputs/diagnostics/frozen_predictor_control_20261027/source_bundle` |
| Current frozen-predictor checkpoint | `outputs/diagnostics/fresh_encoder_frozen_all6_predictor_100k/seed-202610261301/last.pt` |

The primary directory must retain its protocol, training/evaluation caches and
completed arm artifacts. The portable bundle needs `manifest.json`,
`frozen_modules.pt`, `train_inputs.pt` and `evaluation_data.pt`. The current
checkpoint also needs its original `completed.json` and parent `protocol.json`.

The actual all-six encoder checkpoint is resolved from the portable manifest's
`provenance.original_paths.checkpoint`. A relocated file requires an explicit
`--all-six-checkpoint /absolute/server/path`; its bytes must match the manifest's
checkpoint SHA256. Bundle code statistics cannot replace the source encoder.

`--training-source-root` is mandatory and must identify an existing archive of
the original training source tree, including `src/scjepa`, `scripts` and its
historical diagnostic dependencies. Its hash must equal the primary protocol's
`code_identity`, and training dependency semantics must match the runtime. The
archive path and source availability have not been verified by this handoff.

Every script invocation—including preflight, status and reattachment—runs local
`git push` before SSH, then remote `git pull --ff-only` before remote work. It
checks the same branch and exact commit. Local edits to the diagnostic workflow
must be committed; unrelated local edits are preserved and are not deployed.
The remote checkout must have no tracked uncommitted changes. The script never
resets or overwrites either checkout.

## Commands and artifacts

Set `TRAINING_ARCHIVE` to the actual absolute remote archive path, then run the
read-only source/session/GPU preflight from the local repository:

```bash
: "${TRAINING_ARCHIVE:?Set the actual original training-source archive path}"
bash diagnostics/joint_acquisition/server_handoff.sh preflight \
  --training-source-root "$TRAINING_ARCHIVE"
```

When a scientific execution is authorized, the exact launch is:

```bash
bash diagnostics/joint_acquisition/server_handoff.sh launch \
  --training-source-root "$TRAINING_ARCHIVE"
```

Both commands accept explicit `--source-dir`, `--comparison-bundle`,
`--current-checkpoint` and `--all-six-checkpoint` overrides. Preflight checks
recorded primary/source-encoder hashes, archived training identity and required
files. The runner additionally checks complete caches, optimizer mapping and
compatible frozen modules before fitting.

Launch inspects existing tmux sessions and GPU compute processes, refuses a live
joint-acquisition session or occupied GPU, and uses launch/runtime locks to avoid
concurrent handoff jobs. It creates a unique `joint-acquisition-<UTC>-<commit>-<id>`
session and directory; it refuses overwrites and starts detached tmux. The exact
Python command is retained in `run.sh`, with W&B explicitly online.

For the reported run name `NAME`, paths are:

- Run directory: `/nfs/home/jesseh/sparse_causal_jepa/outputs/diagnostics/NAME`.
- Log: `NAME/stdout-stderr.log`, containing stdout and stderr.
- Diagnostic output: `NAME/results`, including protocol, raw measurements,
  trajectory checkpoints, report and `tracking.json`.
- Process result: `NAME/exit_code`; failures preserve the log and artifacts.

Launch prints the actual session and absolute paths. W&B initializes after source
validation; status derives its exact URL from `results/tracking.json` when ready:

```bash
bash diagnostics/joint_acquisition/server_handoff.sh status --run-name NAME
bash diagnostics/joint_acquisition/server_handoff.sh attach --run-name NAME
```

The attach command performs the required push/pull and then uses the supplied SSH
socket to execute `tmux attach-session -t NAME`. The handoff deliberately exposes
no automatic resume or longer-budget command: the runner supports identity-checked
resume, but a failed launch should first be inspected and its unchanged command
reviewed. Software verification is separate: `bash -n` and local `--help` were
checked without SSH, GPU work or W&B initialization.
