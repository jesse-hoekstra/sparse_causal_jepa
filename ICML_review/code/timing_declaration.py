"""Declare the fixed-head sibling-loss timing comparison before training."""

import argparse
from pathlib import Path

from .single_acquisition import DATA as SYNTHETIC_DATA
from .single_acquisition import PRIVATE
from .single_acquisition import _head as check_head  # pyright: ignore[reportPrivateUsage]
from .vision_acquisition import DATA as RGB_DATA
from .vision_acquisition import MANIFEST, read, sha


def declare(revision: str, dataset: str, schedule: str) -> Path:
    """Write one immutable declaration and verify it on a repeated invocation."""
    check_head(revision)
    if dataset not in {"synthetic", "rgb"} or schedule not in {"entire", "sparse"}:
        raise ValueError("Unknown dataset or loss schedule")
    synthetic = dataset == "synthetic"
    seed = 4 if synthetic else 3
    out = PRIVATE / f"timing-{dataset}-{schedule}-s{seed}"
    manifest = SYNTHETIC_DATA / "completion.json" if synthetic else RGB_DATA / "manifest.json"
    data = read(manifest)
    split_text: list[str] = []
    for split in ("train", "calibration", "validation"):
        digest = data["hashes"][f"{split}.pt"] if synthetic else data["splits"][split]["sha256"]
        split_text.append(f"- {split} SHA-256 `{digest}`")
    module = "timing_synthetic" if synthetic else "timing_vision"
    command = f"PYTHONPATH=$PWD:$PWD/src .venv/bin/python -m ICML_review.code.{module}"
    sparse_budget = 350_000 if synthetic else 500_000
    if schedule == "entire":
        treatment = (
            "Activate the auxiliary gradient on the first joint update. During joint, cold "
            "acquisition and adaptation the student targets the detached mean of the other "
            "nine episodes' codes from this forward pass. Measure code displacements in the "
            "current detached TRAIN minibatch covariance with a ridge of one millionth of the "
            "average coordinate variance and a positive numerical floor. Projection has no "
            "optimizer update. Freeze the adapted teacher immediately before the first sparse "
            "update and use it for the full sparse budget."
        )
    else:
        treatment = (
            "Use the native objective throughout joint training, cold acquisition and adaptation. "
            "Reuse authenticated matching prefixes where available. Activate the auxiliary "
            "gradient on the first sparse update, using the just-adapted frozen teacher for the "
            "full sparse budget."
        )
    common = (
        "Both schedules prescribe historical cold acquisition index zero irrespective of its "
        "TRAIN forecast error. Fit no other candidate. Preserve the original seed, data, native "
        "state freezes, deterministic projection, optimizer rules, GECO tolerance, SIGReg and "
        "update budgets. During sparse training the target is the mean of the other nine "
        "episodes of the TRAIN system, encoded by this arm's frozen adapted teacher. Whiten "
        "using its full TRAIN covariance. RGB uses the sampled windows. Cap the auxiliary "
        "gradient on the parameter encoder at its forecast gradient norm before the native "
        "global clipping. Deploy the usual single-episode student."
    )
    if synthetic:
        control = (
            "The first-cold seed-four full endpoint without sibling loss is the matched control "
            "for starting the loss early. Also compare the released best-of-sixteen endpoint "
            "and the historical late sibling intervention. Against each native control require a "
            "resolved six-dimensional chart and a sixth shared eigenvalue/total trace at least "
            "0.80 times its control on both splits. Require validation raw MSE at most 1.10 "
            "times its control and minimum physical parameter R² at least max(0.80, control "
            "minus 0.02)."
        )
    else:
        control = (
            "Compare the released seed-three full endpoint and the historical late sibling "
            "intervention. Neither archived selection supplies a matched untreated first-candidate "
            "final endpoint. The entire-versus-sparse contrast tests early guidance conditional on "
            "using sparse guidance. Against the release require a resolved four-dimensional "
            "chart and raw MSE at most 1.05 times the control on calibration and validation, and "
            "minimum validation drag R² at least max(0.95, control minus 0.01). Report Spearman "
            "MCC against 0.97. Require the validation scalar hold error to increase by at most "
            "0.01 and the kick error by at most 0.05 in the original query units."
        )
    note = f"""# Fixed first acquisition, {dataset}, seed {seed}, {schedule} schedule

## Scientific question

Can guidance from sibling episodes reduce the worst direction of within-context code variation
with one prescribed acquisition while preserving forecast, physical recovery and graph quality?
The stress seed is fixed from the preceding reduced-acquisition studies.

## Intervention

{treatment}

{common}

Joint and acquisition each receive 100,000 updates. Adaptation receives 50,000 and sparse
training receives {sparse_budget:,}. Reused native updates count towards the algorithm budget.
Record their reuse separately from newly executed updates. Evaluate the terminal checkpoint.

## Comparisons and interpretation

{control}

For each native reference report whether both calibration and validation worst within/total
variance fractions fall to at most 0.80 times the corresponding reference. Require validation
Pearson state MCC within 0.01 below its reference and total aligned graph SHD no larger.
Report absolute raw MSE/tau and exact graph recovery separately. Apply criteria to point
estimates. Use whole-system paired uncertainty for within-seed comparisons. Report parameter
retention and graph errors alongside invariance, including collapsed or unresolved directions.
A pass supports a five-seed replication of this schedule. A stability gain with recovery
failure supports only improved repeatability of the code. A failure of both leaves the restart
question open. A failed or undefined endpoint stays in the evidence. Do not stop because of an
admission miss. Neither a pass nor a miss on this seed measures across-seed robustness.
Use TRAIN observations, actions and episode grouping in the loss. Labels enter held-out
diagnostics only. Evaluation and decisions use calibration and validation.

## Reproduction

- Checkout `{revision}`
- Data manifest `{manifest}`, SHA-256 `{sha(manifest)}`
{chr(10).join(split_text)}
- Released checkpoint list `{MANIFEST}`, SHA-256 `{sha(MANIFEST)}`
- Server checkout-relative output `{out}`

```bash
{command} {revision} {schedule}
```
"""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "NOTE.md"
    try:
        with path.open("x") as stream:
            stream.write(note)
    except FileExistsError:
        if path.read_text() != note:
            raise ValueError(f"Conflicting prior declaration at {path}") from None
    return path


def main() -> None:
    """Declare one or all arms from a pinned checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    args = parser.parse_args()
    for dataset in ("synthetic", "rgb"):
        for schedule in ("entire", "sparse"):
            print(declare(args.revision, dataset, schedule))


if __name__ == "__main__":
    main()
