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
bash ICML_review/code/launch.sh "$revision" halving 0 synthetic
bash ICML_review/code/launch.sh "$revision" halving 0 vision
```

For donor and chart diagnostics repeat for seeds 1 through 4. For acquisition replay extend each dataset separately only when its pilot retains the reported admitted winner. The pruning pilot tests one deterministic gate deletion at a time against the saved calibration tolerance. Extend it only when seed zero meets both declared validation criteria in its run note. Every launch takes a fresh output directory and writes its hypothesis, criterion, input identity and command before evaluation. The donor assay compares individual, pooled and other-system support on fixed queries. The chart control varies the calibration interface of a frozen dynamics model. The replay screens acquisitions on their archived midpoint training losses and audits the final choice. All comparisons use existing weights or logs.

Run `ruff format --check ICML_review/code`, `ruff check ICML_review/code`, `pyright ICML_review/code` and `pytest ICML_review/code` from the repository environment. Compile the local review from its own directory with `pdflatex --enable-installer icml_review.tex` on MiKTeX.
