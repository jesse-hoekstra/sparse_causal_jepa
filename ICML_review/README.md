# ICML review experiments

Run from the repository root with its Python environment and the reported inputs listed in `outputs/REPORTED.md`. The Python package in `code/` reads the original models and training histories. Run notes, logs, exit status and metrics live in `outputs/`. The private manuscript and PDF live in `review/`.

On a clean server checkout create `ICML_review/outputs` and run the following commands for seed zero. Verify the reported input hashes before the first job, check GPU availability for model evaluation, and keep the checkout at the same commit until every job exits.

```bash
(cd outputs && sha256sum -c --quiet reported-checkpoints.sha256)
mkdir -p ICML_review/outputs
revision=$(git rev-parse HEAD)
bash ICML_review/code/launch.sh "$revision" donors 0 full
bash ICML_review/code/launch.sh "$revision" donors 0 baseline
bash ICML_review/code/launch.sh "$revision" chart 0 paired
bash ICML_review/code/launch.sh "$revision" prune 0 full
bash ICML_review/code/launch.sh "$revision" kicks 0 paired
bash ICML_review/code/launch.sh "$revision" labels 0 paired
bash ICML_review/code/launch.sh "$revision" halving 0 synthetic
bash ICML_review/code/launch.sh "$revision" halving 0 vision
```

The kicks control extends the two calibration interfaces to the held-bead kick experiment. The labels control requires the completed chart output for the same seed and fixes its physical query grid. For donor, chart, kick and label-budget diagnostics repeat for seeds 1 through 4. For acquisition replay extend each dataset separately only when its pilot retains the reported admitted winner. The pruning pilot tests one deterministic gate deletion at a time against the saved calibration tolerance. Extend it only when seed zero meets both declared validation criteria in its run note. Every launch takes a fresh output directory and writes its hypothesis, criterion, input identity and command before evaluation. The donor assay compares individual, pooled and other-system support on fixed queries. The chart control varies the calibration interface of a frozen dynamics model. The replay screens acquisitions on their archived midpoint training losses and audits the final choice. All comparisons use existing weights or logs.

Run `ruff format --check ICML_review/code`, `ruff check ICML_review/code`, `pyright ICML_review/code` and `pytest ICML_review/code` from the repository environment. Compile the local review from its own directory with `pdflatex --enable-installer icml_review.tex` on MiKTeX.

## Single-acquisition training comparison

From the same checkout on a gateway, the following command uses four named workers sharing the checkout. It trains the pilot on the first worker pair and admits any further seeds only after the recorded validation gate passes. The worker names are positional arguments. Each stage uses an idle GPU, a pinned commit, the reported numeric tolerance and the existing synthetic training modules.

```bash
bash ICML_review/code/single_queue.sh "$revision" "$worker_a" "$worker_b" "$worker_c" "$worker_d"
```

For individual phases use `PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.single_acquisition "$revision" 0 prepare` followed by `full` and `baseline` on distinct available GPUs, then `assess`. The phase names replace the final argument. The assessment reads both completed endpoints. The run folder contains the acquisition receipt, per-stage commands, checkpoint digests and paired decision.

To test a sequential retry, run the same queue with the selection flag before the revision.

```bash
bash ICML_review/code/single_queue.sh --first-admitted "$revision" "$worker_a" "$worker_b" "$worker_c" "$worker_d"
```

This policy checks the archived acquisition candidates in initialization order until one passes the original training-loss cutoff or all candidates have been tried. It writes separate `single-admitted-s<seed>/` folders. For individual phases append `--first-admitted` to the Python command. A matching completed first-candidate run can supply both endpoint assessments after verification of its input hashes, stage ancestry and validation results. The new receipt identifies the original run and its commit.

## Acquisition, excitation and loss comparisons

The head audit reads summary JSON and training histories. It compares the vision joint head, acquisition zero, the original winning head, and the first head matching the joint head's calibration forecast error. It also examines train and deployed-mask forecast losses against the reported tolerance.

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.vision_acquisition "$revision" 0 audit
```

Use `direct` in place of `audit` on a free server GPU to project the joint RGB model directly and run the adaptation screen. A qualifying initializer is trained through both full-length sparse arms. The historical per-seed tolerance is fixed. Seeds 1 through 4 require a qualifying pilot in the same mode. Training and audit records use distinct output directories.

The controlled excitation assay fits least-squares predictors with a fixed support encoder and varies which action directions the training systems experience. It runs on CPU using generated observations and independent training, calibration and validation systems.

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.excitation "$revision" ICML_review/outputs/excitation-s0 --seed 0
```

After a passing pilot, repeat with the remaining seeds, using a fresh output directory for each. Full experiment notes and interpretation are in the private review folder.

The vision loss pilot forks the reported full model at 300k and compares forecast losses for the sampled training mask and the deployed deterministic mask on the same recipient and donor. The synthetic loss pilot modifies only the first frozen-state acquisition, adding normalized within-system code disagreement to the original prediction objective. It uses a prespecified stress-case seed for the pilot. Scientific criteria, controls, exact invocations and outcomes are recorded per run.

From a gateway with SSH access to the workers, run each queue in its own detached session. Assign distinct idle workers to concurrent queues. Each queue consults its own completed pilot before expansion.

```bash
bash ICML_review/code/loss_queue.sh "$revision" vision-loss "$vision_worker"
bash ICML_review/code/loss_queue.sh "$revision" synthetic-loss "$synthetic_worker" "$second_synthetic_worker"
bash ICML_review/code/loss_queue.sh "$revision" vision-direct "$direct_worker"
```

The noisy-support control fits a support encoder and a predictor on generated systems with two levels of support measurement error. It compares forecast-only fitting, normalized within-system regularization, and minimum disagreement subject to a calibration forecast budget. Run the pilot first, then use a fresh matching output directory for each qualified seed.

```bash
PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.noisy_support "$revision" ICML_review/outputs/noisy-support-s0 --seed 0
```
